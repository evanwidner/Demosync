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


def _catmull_rom(points: np.ndarray, samples_per_segment: int, alpha: float = 0.5) -> np.ndarray:
    """Parametric Catmull-Rom spline.

    alpha=0   → uniform (cusps + overshoot on uneven spacing)
    alpha=0.5 → centripetal (no cusps, no self-intersections — what we want indoors)
    alpha=1   → chordal
    """
    if len(points) < 2:
        return points.copy()
    p = np.vstack([points[0], points, points[-1]])

    def _knot(ti: float, pa: np.ndarray, pb: np.ndarray) -> float:
        d = float(np.linalg.norm(pb - pa))
        return ti + max(d, 1e-9) ** alpha

    out: list[np.ndarray] = []
    for i in range(len(points) - 1):
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        t0 = 0.0
        t1 = _knot(t0, p0, p1)
        t2 = _knot(t1, p1, p2)
        t3 = _knot(t2, p2, p3)
        for j in range(samples_per_segment):
            t = t1 + (t2 - t1) * (j / samples_per_segment)
            # de Boor / Barry-Goldman form for non-uniform Catmull-Rom.
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            c = (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
            out.append(c)
    out.append(points[-1])
    return np.array(out)


def _envelope_radius(positions: np.ndarray, multiplier: float = 1.5) -> float:
    """1.5× median nearest-neighbor distance among training cameras.

    Used to define the envelope outside which sampled curve points are clamped back.
    """
    if len(positions) < 2:
        return float("inf")
    nn_dists: list[float] = []
    for i, p in enumerate(positions):
        others = np.delete(positions, i, axis=0)
        d = float(np.min(np.linalg.norm(others - p, axis=1)))
        nn_dists.append(d)
    return float(np.median(nn_dists)) * multiplier


def clamp_to_envelope(
    sampled: np.ndarray,
    training_positions: np.ndarray,
    max_dist: float,
) -> np.ndarray:
    """Project any sampled point further than max_dist from the nearest training camera
    back onto the segment between the previous valid sample and the nearest training camera.

    Operates left-to-right; first sample, if out-of-envelope, snaps to the nearest training
    camera. This eliminates the void-fly-through that produces fractal renders.
    """
    if len(sampled) == 0 or len(training_positions) == 0 or not np.isfinite(max_dist):
        return sampled
    out = sampled.copy()
    last_valid = None
    for i in range(len(out)):
        p = out[i]
        d_to_train = np.linalg.norm(training_positions - p, axis=1)
        nearest_idx = int(np.argmin(d_to_train))
        nearest_d = float(d_to_train[nearest_idx])
        if nearest_d <= max_dist:
            last_valid = out[i]
            continue
        nearest = training_positions[nearest_idx]
        if last_valid is None:
            out[i] = nearest
        else:
            # Walk from last_valid toward nearest until inside the envelope.
            direction = nearest - last_valid
            seg_len = float(np.linalg.norm(direction))
            if seg_len < 1e-9:
                out[i] = nearest
            else:
                # Step the maximum allowable distance along the segment.
                t = min(1.0, max_dist / seg_len)
                out[i] = last_valid + direction * t
        last_valid = out[i]
    return out


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

    # A2 — envelope clamp. Centripetal Catmull-Rom alone reduces overshoot but does not
    # prevent the spline from wandering outside the registered volume between distant
    # waypoints. Force every sample to stay within max_dist of the nearest training camera.
    sampled = clamp_to_envelope(
        sampled,
        training_positions=positions,
        max_dist=_envelope_radius(positions),
    )

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
