"""Stage 7 — Claude QA vision pass over rendered output.

Extracts N evenly-spaced frames from the final video and asks Claude to flag
floaters, geometric breakage, exposure jumps, reflective-surface artifacts, and
under-reconstructed rooms. Returns a structured severity + reshoot-recommendation
report. Advisory only — if severity == fail, the caller should hold delivery.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

CLAUDE_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are reviewing rendered frames from a Gaussian-Splatting reconstruction
of a real estate listing. For each frame identify visible failure modes typical to 3DGS
output: floaters (orphan splats in empty space), geometric breakage (walls clipping, floor
holes, ceiling tears), reflective-surface artifacts (warped mirrors/TVs/glass), motion
blur or judder between adjacent frames, exposure jumps between rooms, and rooms that look
under-reconstructed (blurry/blobby vs the rest).

For under-reconstructed rooms or missing coverage, recommend specific reshoot angles the
agent can capture with a phone — e.g. "wide shot from kitchen doorway looking toward the
island" — not "more photos of the kitchen."

Return strict JSON. No prose, no markdown fences."""

RESPONSE_SCHEMA_HINT = """{
  "severity": "ok | minor | major | fail",
  "summary": "one paragraph plain-English",
  "per_frame": [
    {"frame_index": 0, "issues": ["floater top-right", "wall clipping"], "rooms_visible": ["kitchen"]}
  ],
  "reshoot_requests": [
    {"room": "primary_bathroom", "angle": "wide shot from doorway", "reason": "no coverage of vanity wall"}
  ]
}"""


@dataclass
class QaReport:
    severity: str
    summary: str
    per_frame: list[dict[str, Any]]
    reshoot_requests: list[dict[str, Any]]
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def should_hold(self) -> bool:
        return self.severity == "fail"

    def write(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "severity": self.severity,
                    "summary": self.summary,
                    "per_frame": self.per_frame,
                    "reshoot_requests": self.reshoot_requests,
                },
                indent=2,
            )
        )


def extract_frames(video: Path, output_dir: Path, n: int = 8) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = _probe_duration(video)
    timestamps = [duration * (i + 0.5) / n for i in range(n)]
    paths: list[Path] = []
    for i, ts in enumerate(timestamps):
        out = output_dir / f"qa_frame_{i:02d}.jpg"
        if out.exists():
            paths.append(out)
            continue
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{ts:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-q:v",
                "3",
                str(out),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        paths.append(out)
    return paths


def _probe_duration(video: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(out.stdout.strip())


def qa_video(video: Path, output_json: Path, frames_dir: Path | None = None, n_frames: int = 8) -> QaReport:
    frames_dir = frames_dir or video.parent / "qa_frames"
    frames = extract_frames(video, frames_dir, n=n_frames)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    content: list[dict[str, Any]] = []
    for i, frame in enumerate(frames):
        content.append({"type": "text", "text": f"Frame {i}:"})
        data = base64.standard_b64encode(frame.read_bytes()).decode("ascii")
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
        })
    content.append({
        "type": "text",
        "text": f"Return strict JSON matching this schema:\n{RESPONSE_SCHEMA_HINT}",
    })

    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text").strip()
    text = _strip_markdown_fences(text)
    parsed = json.loads(text)

    report = QaReport(
        severity=parsed.get("severity", "minor"),
        summary=parsed.get("summary", ""),
        per_frame=parsed.get("per_frame", []),
        reshoot_requests=parsed.get("reshoot_requests", []),
        raw=parsed,
    )
    report.write(output_json)
    return report


def _strip_markdown_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()
