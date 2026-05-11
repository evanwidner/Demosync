"""Stage 6 add-on — optional ElevenLabs voiceover.

Claude writes a 60-90 word voiceover script from the listing description and the
organizational pass; ElevenLabs Flash v2.5 synthesizes it to mp3. The mp3 is mixed
into the final video at -10 dB under the music track.

Off by default — gated by job.voiceover_enabled in the database.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import anthropic

CLAUDE_MODEL = "claude-sonnet-4-6"
ELEVENLABS_MODEL = "eleven_flash_v2_5"
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # Sarah (warm female)


def write_voiceover_script(
    organize_json: Path,
    listing_description: str | None,
    target_duration_s: float,
    out_path: Path,
) -> str:
    target_words = int(target_duration_s * 2.2)  # ~2.2 words/sec for paced narration
    listing = json.loads(organize_json.read_text()).get("listing", {}) if organize_json.exists() else {}

    user_text = (
        f"Write a {target_words}-word voiceover script for a {target_duration_s:.0f}-second "
        f"cinematic real estate walkthrough video. Calm, warm, paced for narration. "
        f"Open with a hook about the property's distinguishing feature. Highlight 3-4 hero "
        f"features in order. Close with a memorable line. No 'welcome home' clichés. No price.\n\n"
        f"Detected rooms: {listing.get('detected_rooms', [])}\n"
        f"Hero features: {listing.get('hero_features', [])}\n"
        f"Style: {listing.get('architectural_style_guess', 'unknown')}\n"
    )
    if listing_description:
        user_text += f"\nListing description:\n{listing_description}\n"
    user_text += "\nReturn the script only, no preamble or markdown."

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": user_text}],
    )
    script = "".join(b.text for b in resp.content if b.type == "text").strip()
    out_path.write_text(script)
    return script


def synthesize(script: str, out_path: Path) -> Path:
    import urllib.request  # noqa: PLC0415

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not set")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    payload = json.dumps({
        "text": script,
        "model_id": ELEVENLABS_MODEL,
        "voice_settings": {"stability": 0.55, "similarity_boost": 0.75, "style": 0.15, "use_speaker_boost": True},
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        out_path.write_bytes(resp.read())
    return out_path
