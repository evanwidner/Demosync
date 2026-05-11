"""Generate a Nerfstudio-compatible camera path JSON from COLMAP camera positions.

MVP heuristic: order COLMAP cameras by a greedy nearest-neighbor traversal starting
from the camera with the lowest centroid distance to the median, fit a Catmull-Rom
spline through the positions, and sample per-frame poses along the curve.

The look-at target for each frame is a sliding-window average of upcoming positions,
which produces a forward-facing dolly with smooth turns rather than rigid look-at-center.

Later stages will replace this with Claude property-aware path generation; the output
schema is the same so downstream code is unaffected.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class CameraPose:
    position: np.ndarray  # shape (3,)
    forward: np.ndarray   # shape (3,) unit vector
    up: np.ndarray        # shape (3,) unit vector


def _load_colmap_poses(transforms_json: Path) -> list[CameraPose]:
    data = json.loads(transforms_json.read_text())
    poses: list[CameraPose] = []
    for frame in data["frames"]:
        m = np.array(frame["transform_matrix"], dtype=np.float64)
        position = m[:3, 3]
        forward = -m[:3, 2]
        up = m[:3, 1]
        poses.append(CameraPose(position=position, forward=forward, up=up))
    return poses


def _greedy_tsp_order(positions: np.ndarray) -> list[int]:
    n = len(positions)
    if n == 0:
        return []
    medoid = int(np.argmin(np.linalg.norm(positions - np.median(positions, axis=0), axis=1)))
    visited = {medoid}
    order = [medoid]
    while len(order) < n:
        last = positions[order[-1]]
        best = -1
        best_d = math.inf
        for i in range(n):
            if i in visited:
                continue
            d = float(np.linalg.norm(positions[i] - last))
            if d < best_d:
                best_d = d
                best = i
        order.append(best)
        visited.add(best)
    return order


def _catmull_rom(points: np.ndarray, samples_per_segment: int) -> np.ndarray:
    if len(points) < 2:
        return points.copy()
    p = np.vstack([points[0], points, points[-1]])
    out: list[np.ndarray] = []
    for i in range(len(points) - 1):
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        for j in range(samples_per_segment):
            t = j / samples_per_segment
            t2, t3 = t * t, t * t * t
            point = 0.5 * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            out.append(point)
    out.append(points[-1])
    return np.array(out)


def _ease_in_out(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * t)


def _look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f /= np.linalg.norm(f) + 1e-9
    s = np.cross(f, up)
    s_norm = np.linalg.norm(s)
    if s_norm < 1e-6:
        up = np.array([0.0, 0.0, 1.0]) if abs(up[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        s = np.cross(f, up)
        s_norm = np.linalg.norm(s)
    s = s / max(s_norm, 1e-9)
    u = np.cross(s, f)
    m = np.eye(4)
    m[:3, 0] = s
    m[:3, 1] = u
    m[:3, 2] = -f
    m[:3, 3] = eye
    return m


def generate_camera_path(
    transforms_json: Path,
    output_json: Path,
    duration_seconds: float = 75.0,
    fps: int = 30,
    render_width: int = 1920,
    render_height: int = 1080,
) -> Path:
    """Generate a Nerfstudio render camera-path JSON and write it to output_json."""
    poses = _load_colmap_poses(transforms_json)
    if len(poses) < 2:
        raise ValueError(f"need at least 2 COLMAP poses, got {len(poses)}")

    positions = np.array([p.position for p in poses])
    order = _greedy_tsp_order(positions)
    ordered = positions[order]

    total_frames = int(duration_seconds * fps)
    samples_per_segment = max(1, total_frames // max(1, len(ordered) - 1))
    curve = _catmull_rom(ordered, samples_per_segment=samples_per_segment)

    # Resample the curve to exactly total_frames using arclength parameterization with easing.
    deltas = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(deltas)])
    total_len = cum[-1] if cum[-1] > 0 else 1.0
    eased_ts = np.array([_ease_in_out(i / (total_frames - 1)) for i in range(total_frames)])
    target_arc = eased_ts * total_len
    sampled = np.array([_interp_curve(curve, cum, s) for s in target_arc])

    # Look-at target = sliding window mean of upcoming positions (10% lookahead).
    lookahead = max(1, total_frames // 10)
    up = np.array([0.0, 0.0, 1.0])  # world Z-up; Nerfstudio convention
    camera_path = []
    for i in range(total_frames):
        eye = sampled[i]
        future = sampled[min(i + lookahead, total_frames - 1)]
        if np.linalg.norm(future - eye) < 1e-4:
            future = eye + np.array([0.1, 0.0, 0.0])
        m = _look_at_matrix(eye, future, up)
        camera_path.append(
            {
                "camera_to_world": m.flatten().tolist(),
                "fov": 70,
                "aspect": render_width / render_height,
            }
        )

    payload = {
        "render_height": render_height,
        "render_width": render_width,
        "fps": fps,
        "seconds": duration_seconds,
        "camera_type": "perspective",
        "camera_path": camera_path,
        "keyframes": [
            {
                "matrix": _look_at_matrix(positions[order[i]], positions[order[min(i + 1, len(order) - 1)]], up).flatten().tolist(),
                "fov": 70,
                "aspect": render_width / render_height,
                "properties": "",
            }
            for i in range(len(order))
        ],
        "smoothness_value": 0.5,
        "is_cycle": False,
    }
    output_json.write_text(json.dumps(payload, indent=2))
    return output_json


def _interp_curve(curve: np.ndarray, cum: np.ndarray, s: float) -> np.ndarray:
    idx = int(np.searchsorted(cum, s))
    if idx <= 0:
        return curve[0]
    if idx >= len(curve):
        return curve[-1]
    seg_len = cum[idx] - cum[idx - 1]
    if seg_len < 1e-9:
        return curve[idx]
    t = (s - cum[idx - 1]) / seg_len
    return curve[idx - 1] * (1 - t) + curve[idx] * t
