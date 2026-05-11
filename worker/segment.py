"""Stage 1 — Semantic segmentation via Grounded-SAM 2.

For each photo, produce binary masks for a fixed prompt set covering the labels the
constrained splat trainer needs:

    floor, wall, ceiling, window, mirror, tv_screen, glass_door, furniture

The mirror / tv_screen / glass_door masks gate reflective-surface exclusion in the
trainer (mask them out of the photometric loss to avoid floaters from view-dependent
artifacts). The floor / wall / ceiling masks gate planar regularization.

Output is one composite PNG per input photo where each pixel is a class index 0-7
(0 = none/background). Easier to load than 8 binary PNGs.

This module is a scaffold — Grounded-SAM-2 requires careful dependency setup
(GroundingDINO + SAM2 weights). Production deployment will pin a specific docker
image. For local development, the depth + organize stages are testable without this.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

LABELS = ["floor", "wall", "ceiling", "window", "mirror", "tv_screen", "glass_door", "furniture"]
PROMPT = ". ".join(LABELS)


def run_segment(photo_dir: Path, output_dir: Path) -> list[Path]:
    """Run Grounded-SAM 2 over photo_dir, write per-photo composite masks to output_dir.

    Returns:
        List of output mask PNG paths (uint8 single-channel, pixel = class index 0..len(LABELS)).
    """
    import torch  # noqa: PLC0415
    from sam2.build_sam import build_sam2  # type: ignore # noqa: PLC0415
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore # noqa: PLC0415
    from groundingdino.util.inference import load_model, predict  # type: ignore # noqa: PLC0415
    from groundingdino.util import box_ops  # type: ignore # noqa: PLC0415
    import torchvision.transforms.functional as TF  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gdino = load_model(
        "/app/weights/GroundingDINO_SwinT_OGC.cfg.py",
        "/app/weights/groundingdino_swint_ogc.pth",
        device=device,
    )
    sam2 = build_sam2("sam2_hiera_l.yaml", "/app/weights/sam2_hiera_large.pt", device=device)
    predictor = SAM2ImagePredictor(sam2)

    photos = sorted(
        p for p in photo_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    outputs: list[Path] = []
    for p in photos:
        out_path = output_dir / f"{p.stem}.png"
        if out_path.exists():
            outputs.append(out_path)
            continue

        img = Image.open(p).convert("RGB")
        img_np = np.array(img)
        H, W = img_np.shape[:2]
        img_tensor = TF.to_tensor(img).to(device)

        boxes, logits, phrases = predict(
            model=gdino,
            image=img_tensor,
            caption=PROMPT,
            box_threshold=0.30,
            text_threshold=0.25,
            device=device,
        )
        if len(boxes) == 0:
            Image.fromarray(np.zeros((H, W), dtype=np.uint8), mode="L").save(out_path)
            outputs.append(out_path)
            continue

        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([W, H, W, H], device=device)
        predictor.set_image(img_np)
        masks, _, _ = predictor.predict(box=boxes_xyxy.cpu().numpy(), multimask_output=False)

        composite = np.zeros((H, W), dtype=np.uint8)
        for mask, phrase in zip(masks, phrases):
            phrase = phrase.lower().strip()
            class_idx = next((i + 1 for i, lbl in enumerate(LABELS) if lbl.replace("_", " ") in phrase or lbl in phrase), 0)
            if class_idx == 0:
                continue
            mask_bin = (mask[0] > 0.5) if mask.ndim == 3 else (mask > 0.5)
            composite[mask_bin] = class_idx
        Image.fromarray(composite, mode="L").save(out_path)
        outputs.append(out_path)

    return outputs
