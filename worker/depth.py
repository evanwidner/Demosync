"""Stage 1 — Depth Anything V2 dense per-pixel depth.

Wraps the Depth-Anything-V2 model checkpoint (vitl variant) for batched inference
over a directory of photos. Outputs 16-bit single-channel PNGs aligned to source
resolution, normalized to [0, 65535] from the model's relative-depth output.

The depth-supervised splat trainer (stage 3, DNGaussian fork) handles the
relative-to-metric scale + shift alignment per scene during training, so this stage
is intentionally simple — no calibration here.

Runs on GPU (cuda). Falls back to CPU if no GPU; ~20× slower but useful for local
sanity checks on a couple of photos.

Requires:
    pip install transformers accelerate
    Model is auto-downloaded from HuggingFace on first run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def run_depth(photo_dir: Path, output_dir: Path, model_size: str = "Large") -> list[Path]:
    """Run Depth Anything V2 over photo_dir, write 16-bit depth PNGs to output_dir.

    Args:
        photo_dir: directory of input photos.
        output_dir: where to write per-photo depth maps (same basename, .png).
        model_size: "Small" | "Base" | "Large". "Large" is the recommended default.

    Returns:
        List of output depth PNG paths in input order.
    """
    import torch  # noqa: PLC0415 — lazy import; heavy
    from transformers import pipeline  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = pipeline(
        task="depth-estimation",
        model=f"depth-anything/Depth-Anything-V2-{model_size}-hf",
        device=device,
        torch_dtype=dtype,
    )

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
        result = pipe(img)
        depth = np.array(result["depth"], dtype=np.float32)
        # Normalize to 16-bit. depth from pipeline is already 0..255 uint8 by default;
        # use predicted_depth tensor for full-precision when available.
        if "predicted_depth" in result:
            t = result["predicted_depth"]
            d = t.squeeze().cpu().float().numpy()
            d_min, d_max = float(d.min()), float(d.max())
            if d_max - d_min < 1e-6:
                d_norm = np.zeros_like(d, dtype=np.uint16)
            else:
                d_norm = ((d - d_min) / (d_max - d_min) * 65535.0).astype(np.uint16)
            # Resize to source resolution
            d_img = Image.fromarray(d_norm, mode="I;16").resize(img.size, Image.BILINEAR)
            d_img.save(out_path)
        else:
            depth_u8 = depth.astype(np.uint8)
            Image.fromarray(depth_u8, mode="L").resize(img.size, Image.BILINEAR).save(out_path)
        outputs.append(out_path)

    return outputs
