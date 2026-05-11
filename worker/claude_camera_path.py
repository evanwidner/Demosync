"""Stage 4 (upgrade) — Claude property-aware camera path.

Replaces the heuristic Catmull-Rom path with a Claude-generated waypoint sequence
informed by the listing description, the organizational pass (hero features, detected
rooms), and the COLMAP camera positions (so waypoints stay inside the registered
volume — no clipping into unreconstructed void).

The Catmull-Rom + ease-in-out smoothing from worker/camera_path.py is reused; Claude
only decides waypoint ORDER + per-waypoint hero-moment hold time + speed. Numeric
positions are taken from existing COLMAP camera positions (or interpolations between
adjacent ones) so we don't trust Claude to invent 3D coordinates.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
import numpy as np

from worker.camera_path import _catmull_rom, _ease_in_out, _interp_curve, _look_at_matrix

CLAUDE_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are designing a cinematic camera path through a real estate property.
You will receive (a) an organizational analysis of the listing photos with detected rooms
and hero features, (b) a list of COLMAP-registered camera positions tagged with their
detected room labels, and optionally (c) the listing description text.

Output a sequence of waypoints that produces a 60-90 second cinematic dolly through the
property. Start with an exterior or foyer approach, hit each major room in a sensible
order (typically: foyer → great room → kitchen → dining → primary suite → other bedrooms
→ outdoor spaces), hold for 2-3 seconds on hero features, and end on a memorable shot
(often a view, a kitchen island, or a primary suite).

Refer to waypoints by camera_index (the index into the provided camera list). Do not
invent positions — only choose from the provided indices.

Return strict JSON. No prose, no markdown fences."""

RESPONSE_SCHEMA_HINT = """{
  "duration_seconds": 75,
  "ordered_waypoints": [
    {"camera_index": 0, "label": "exterior approach", "hold_seconds": 1.5, "speed": "slow"},
    {"camera_index": 7, "label": "foyer", "hold_seconds": 1.0, "speed": "medium"},
    {"camera_index": 3, "label": "great room — Sandia view", "hold_seconds": 3.0, "speed": "slow", "hero": true}
  ],
  "music_style": "warm_acoustic | ambient_electronic | cinematic_strings | indie_folk | sparse_piano | upbeat_pop"
}"""


@dataclass
class ClaudeCameraPlan:
    duration_seconds: float
    ordered_waypoints: list[dict[str, Any]]
    music_style: str
    raw: dict[str, Any]


def generate_claude_camera_path(
    transforms_json: Path,
    organize_json: Path,
    output_json: Path,
    listing_description: str | None = None,
    fps: int = 30,
    render_width: int = 1920,
    render_height: int = 1080,
) -> ClaudeCameraPlan:
    plan = _ask_claude_for_plan(transforms_json, organize_json, listing_description)
    _write_render_path(plan, transforms_json, output_json, fps, render_width, render_height)
    return plan


def _ask_claude_for_plan(
    transforms_json: Path,
    organize_json: Path,
    listing_description: str | None,
) -> ClaudeCameraPlan:
    transforms = json.loads(transforms_json.read_text())
    organize = json.loads(organize_json.read_text())

    # Build cam → room label mapping by matching transforms frames' file_path back to
    # the per_photo organize entries (by filename basename).
    photo_to_room: dict[str, str] = {}
    for p in organize.get("per_photo", []):
        photo_to_room[Path(p.get("filename", "")).name] = p.get("room_label", "other")

    cam_list: list[dict[str, Any]] = []
    for i, frame in enumerate(transforms["frames"]):
        fname = Path(frame["file_path"]).name
        cam_list.append({
            "camera_index": i,
            "filename": fname,
            "room_label": photo_to_room.get(fname, "unknown"),
        })

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    user_text = (
        "Cameras (COLMAP-registered):\n"
        f"{json.dumps(cam_list, indent=2)}\n\n"
        "Listing analysis:\n"
        f"{json.dumps(organize.get('listing', {}), indent=2)}\n\n"
    )
    if listing_description:
        user_text += f"Listing description:\n{listing_description}\n\n"
    user_text += f"Return strict JSON:\n{RESPONSE_SCHEMA_HINT}"

    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": [{"type": "text", "text": user_text}]}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text").strip()
    text = _strip_markdown_fences(text)
    parsed = json.loads(text)
    return ClaudeCameraPlan(
        duration_seconds=float(parsed.get("duration_seconds", 75)),
        ordered_waypoints=parsed.get("ordered_waypoints", []),
        music_style=parsed.get("music_style", "warm_acoustic"),
        raw=parsed,
    )


def _write_render_path(
    plan: ClaudeCameraPlan,
    transforms_json: Path,
    output_json: Path,
    fps: int,
    render_width: int,
    render_height: int,
) -> None:
    transforms = json.loads(transforms_json.read_text())
    frames = transforms["frames"]
    positions = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames])
    n_cams = len(frames)

    waypoint_positions: list[np.ndarray] = []
    holds: list[float] = []
    speeds: list[str] = []
    for w in plan.ordered_waypoints:
        idx = int(w.get("camera_index", 0))
        if idx < 0 or idx >= n_cams:
            continue
        waypoint_positions.append(positions[idx])
        holds.append(float(w.get("hold_seconds", 1.0)))
        speeds.append(str(w.get("speed", "medium")))
    if len(waypoint_positions) < 2:
        raise ValueError("Claude returned fewer than 2 valid waypoints; cannot render path")

    waypoint_positions_arr = np.array(waypoint_positions)
    # Constraint: waypoints must lie within the convex hull of all COLMAP positions.
    # We already guaranteed that by picking indices into the COLMAP list — no fakes.

    total_frames = int(plan.duration_seconds * fps)
    samples_per_segment = max(1, total_frames // max(1, len(waypoint_positions_arr) - 1))
    curve = _catmull_rom(waypoint_positions_arr, samples_per_segment=samples_per_segment)

    deltas = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(deltas)])
    total_len = max(cum[-1], 1e-6)
    eased_ts = np.array([_ease_in_out(i / max(1, total_frames - 1)) for i in range(total_frames)])
    target_arc = eased_ts * total_len
    sampled = np.array([_interp_curve(curve, cum, s) for s in target_arc])

    lookahead = max(1, total_frames // 10)
    up = np.array([0.0, 0.0, 1.0])
    camera_path = []
    for i in range(total_frames):
        eye = sampled[i]
        future = sampled[min(i + lookahead, total_frames - 1)]
        if np.linalg.norm(future - eye) < 1e-4:
            future = eye + np.array([0.1, 0.0, 0.0])
        m = _look_at_matrix(eye, future, up)
        camera_path.append({
            "camera_to_world": m.flatten().tolist(),
            "fov": 70,
            "aspect": render_width / render_height,
        })

    payload = {
        "render_height": render_height,
        "render_width": render_width,
        "fps": fps,
        "seconds": plan.duration_seconds,
        "camera_type": "perspective",
        "camera_path": camera_path,
        "keyframes": [
            {
                "matrix": _look_at_matrix(
                    waypoint_positions_arr[i],
                    waypoint_positions_arr[min(i + 1, len(waypoint_positions_arr) - 1)],
                    up,
                ).flatten().tolist(),
                "fov": 70,
                "aspect": render_width / render_height,
                "properties": json.dumps({
                    "label": plan.ordered_waypoints[i].get("label", ""),
                    "hero": bool(plan.ordered_waypoints[i].get("hero", False)),
                }),
            }
            for i in range(len(waypoint_positions_arr))
        ],
        "smoothness_value": 0.5,
        "is_cycle": False,
        "demosync_meta": {
            "source": "claude",
            "music_style": plan.music_style,
            "ordered_waypoints": plan.ordered_waypoints,
        },
    }
    output_json.write_text(json.dumps(payload, indent=2))


def _strip_markdown_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()
