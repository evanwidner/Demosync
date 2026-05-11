"""Tier 0 acceptance check: assert that every sampled camera position in a rendered
camera_path.json stays within the envelope of the COLMAP training cameras.

Usage:
    python scripts/verify_envelope.py \\
        --transforms worker/runs/<job_id>/out/workdir/processed/transforms.json \\
        --path       worker/runs/<job_id>/out/workdir/camera_path.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from worker.camera_path import _envelope_radius


def _load_training_positions(transforms_json: Path) -> np.ndarray:
    data = json.loads(transforms_json.read_text())
    return np.array([np.array(f["transform_matrix"])[:3, 3] for f in data["frames"]])


def _load_sampled_positions(camera_path_json: Path) -> np.ndarray:
    data = json.loads(camera_path_json.read_text())
    out = []
    for entry in data["camera_path"]:
        m = np.array(entry["camera_to_world"]).reshape(4, 4)
        out.append(m[:3, 3])
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transforms", type=Path, required=True)
    ap.add_argument("--path", type=Path, required=True)
    ap.add_argument("--multiplier", type=float, default=1.5,
                    help="Envelope radius multiplier (matches camera_path.py default).")
    args = ap.parse_args()

    train = _load_training_positions(args.transforms)
    sampled = _load_sampled_positions(args.path)
    envelope = _envelope_radius(train, multiplier=args.multiplier)

    dists = np.array([
        np.min(np.linalg.norm(train - p, axis=1)) for p in sampled
    ])
    violations = int(np.sum(dists > envelope))

    print(f"training cameras: {len(train)}")
    print(f"sampled frames:   {len(sampled)}")
    print(f"envelope radius:  {envelope:.4f}  (={args.multiplier}× median NN distance)")
    print(f"max sample dist:  {float(dists.max()):.4f}")
    print(f"mean sample dist: {float(dists.mean()):.4f}")
    print(f"violations:       {violations}/{len(sampled)}")

    if violations > 0:
        print("FAIL — envelope clamp did not catch all out-of-volume samples.", file=sys.stderr)
        return 1
    print("PASS — all sampled positions inside envelope.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
