"""Stage 1 — Claude organizational pass over input photos.

One multimodal API call batches every input photo and returns:
    - per-photo: room label, exposure quality, mirror/glass/tv flags, framing notes,
      coverage score (1-5)
    - listing-level: detected rooms, missing rooms, coverage gaps, hero features

Output JSON is consumed by stage 4 (Claude camera path), stage 6 (music selection),
and stage 7 (QA reshoot recommendations).

Cost: ~$0.02/listing on Sonnet 4.6 (50 photos × ~1500 input tokens each).
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

CLAUDE_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You analyze real estate listing photos for a 3D reconstruction pipeline.
For each photo identify: which room it's in, exposure quality, presence of reflective surfaces
(mirrors, glass, TVs), framing notes affecting reconstruction (ultra-wide distortion, occluded
corners, hero composition), and a coverage score (1=single hero angle, 5=multiple overlapping
angles of same space). At listing level identify rooms present, rooms likely missing, coverage
gaps that will break reconstruction, and hero features the camera path should emphasize.

Respond with strict JSON matching the schema. No prose, no markdown fences."""

RESPONSE_SCHEMA_HINT = """{
  "per_photo": [
    {
      "photo_index": 0,
      "filename": "string",
      "room_label": "kitchen | living_room | primary_bedroom | bedroom_2 | bathroom_primary | bathroom_2 | dining | office | exterior_front | exterior_back | patio | garage | hallway | foyer | laundry | other",
      "exposure_quality": "good | overexposed | underexposed | mixed_hdr",
      "has_mirrors": false,
      "has_glass": false,
      "has_tv_screen": false,
      "framing_notes": "string",
      "coverage_score": 3
    }
  ],
  "listing": {
    "detected_rooms": ["kitchen", "living_room"],
    "missing_rooms": ["primary_bathroom"],
    "coverage_gaps": ["only one angle of kitchen", "no exterior rear shots"],
    "hero_features": ["mountain view from great room", "vaulted ceiling in primary"],
    "architectural_style_guess": "spanish_mediterranean | modern | craftsman | colonial | ranch | contemporary | farmhouse | other",
    "ceiling_height_estimate_ft": 9
  }
}"""


@dataclass
class OrganizeResult:
    per_photo: list[dict[str, Any]]
    listing: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)

    def write(self, path: Path) -> None:
        path.write_text(json.dumps({"per_photo": self.per_photo, "listing": self.listing}, indent=2))


def _encode_image(path: Path) -> dict[str, Any]:
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "image/jpeg")
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def organize_photos(photo_dir: Path, output_json: Path, listing_description: str | None = None) -> OrganizeResult:
    photos = sorted(
        p for p in photo_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not photos:
        raise ValueError(f"no photos found in {photo_dir}")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    content: list[dict[str, Any]] = []
    for i, p in enumerate(photos):
        content.append({"type": "text", "text": f"Photo {i}: {p.name}"})
        content.append(_encode_image(p))

    user_text = (
        "Analyze these listing photos. Return strict JSON matching this schema:\n"
        f"{RESPONSE_SCHEMA_HINT}"
    )
    if listing_description:
        user_text += f"\n\nListing description for context:\n{listing_description}"
    content.append({"type": "text", "text": user_text})

    resp = client.messages.create(
        model=CLAUDE_MODEL,
        # 8192 truncates around ~50 photos and produces unterminated JSON. Sonnet
        # 4.6 supports much higher; 32K gives ~250 photos of headroom and the
        # listing-level summary on top.
        max_tokens=32000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text").strip()
    text = _strip_markdown_fences(text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        # Surface the failure with enough context to debug without re-running. Include
        # the stop_reason (max_tokens vs end_turn) so we know if Claude was cut off.
        stop_reason = getattr(resp, "stop_reason", "unknown")
        tail = text[-400:] if len(text) > 400 else text
        raise RuntimeError(
            f"Claude returned invalid JSON (stop_reason={stop_reason}, "
            f"output_len={len(text)} chars, n_photos={len(photos)}). "
            f"Last 400 chars:\n{tail}\n\nParse error: {e}"
        ) from e

    result = OrganizeResult(
        per_photo=parsed.get("per_photo", []),
        listing=parsed.get("listing", {}),
        raw=parsed,
    )
    result.write(output_json)
    return result


def _strip_markdown_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()
