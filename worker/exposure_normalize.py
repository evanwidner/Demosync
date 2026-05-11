"""A7 — HDR / exposure normalization preprocessing.

Listing photos arrive bracketed with inconsistent exposure across rooms (a dark
bathroom hero shot next to a blown-out south-facing window). This breaks two things:
  1. Claude's organize pass — room labeling degrades on extreme exposures.
  2. Splatfacto training — bakes exposure variance into the gaussians as color noise.

Strategy: per-photo gain that pulls each photo's mean luma toward the median across
the set, with a clamped multiplier so we don't blow out hero shots. Operates in
sRGB space (good enough for consistency; not photometrically rigorous). Pillow +
numpy only — no OpenCV dep.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".PNG", ".JPG", ".JPEG"}


@dataclass
class NormalizeReport:
    n_photos: int
    median_luma: float
    gains: dict[str, float]  # filename → applied gain
    clipped_count: int       # how many photos hit the gain clamp


def _luma_mean(img: Image.Image) -> float:
    """Rec.601 luma mean in sRGB. Cheap, good enough for relative exposure ranking."""
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        return float(arr.mean())
    luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return float(luma.mean())


def _apply_gain(img: Image.Image, gain: float) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    arr = np.clip(arr * gain, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode=img.mode)


def normalize_directory(
    src_dir: Path,
    dst_dir: Path,
    gain_min: float = 0.5,
    gain_max: float = 2.0,
    jpeg_quality: int = 95,
) -> NormalizeReport:
    """Normalize every photo in src_dir into dst_dir.

    Skips files that already exist in dst_dir (idempotent).
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    photos = sorted(p for p in src_dir.iterdir() if p.suffix in PHOTO_SUFFIXES)
    if not photos:
        return NormalizeReport(n_photos=0, median_luma=0.0, gains={}, clipped_count=0)

    # Pass 1: compute per-photo luma means (skip photos already done).
    luma_by_path: dict[Path, float] = {}
    for p in photos:
        try:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                luma_by_path[p] = _luma_mean(im)
        except Exception as e:
            print(f"[exposure_normalize] skipping unreadable {p.name}: {e}")
            continue

    if not luma_by_path:
        return NormalizeReport(n_photos=0, median_luma=0.0, gains={}, clipped_count=0)

    median_luma = float(np.median(list(luma_by_path.values())))
    if median_luma <= 1e-3:
        # Degenerate (all-black input). Just copy through.
        median_luma = max(median_luma, 1.0)

    gains: dict[str, float] = {}
    clipped = 0
    # Pass 2: apply clamped gain, write to dst.
    for p, luma in luma_by_path.items():
        out_path = dst_dir / p.name
        if out_path.exists():
            gains[p.name] = 1.0  # already done; don't re-record
            continue
        raw_gain = median_luma / max(luma, 1.0)
        gain = float(np.clip(raw_gain, gain_min, gain_max))
        if abs(gain - raw_gain) > 1e-6:
            clipped += 1
        gains[p.name] = gain
        try:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                normalized = _apply_gain(im, gain) if abs(gain - 1.0) > 1e-3 else im
                # Re-encode as JPEG (worker downstream is JPG-friendly). Preserve EXIF
                # so COLMAP can still consume EXIF-derived intrinsics if it wants to.
                save_path = out_path.with_suffix(".jpg")
                normalized.save(save_path, "JPEG", quality=jpeg_quality, exif=im.info.get("exif", b""))
        except Exception as e:
            print(f"[exposure_normalize] failed to write {p.name}: {e}")

    return NormalizeReport(
        n_photos=len(luma_by_path),
        median_luma=median_luma,
        gains=gains,
        clipped_count=clipped,
    )
