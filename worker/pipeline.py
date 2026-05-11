"""DemoSync worker pipeline — MVP orchestrator.

Stage coverage for v1 step 1 (vanilla Splatfacto, no constraints):
    1. preprocess: ns-process-data images   (wraps COLMAP)
    2. train:      ns-train splatfacto      (Nerfstudio Gaussian Splatting)
    3. camera path: heuristic Catmull-Rom from COLMAP poses (worker/camera_path.py)
    4. render:     ns-render camera-path
    5. post:       ffmpeg color grade + music + title card

Later stages (depth, segment, planar regularization, Claude camera path, QA) bolt
onto this file; each gets its own function and is gated by a CLI flag.

Usage:
    python -m worker.pipeline run \\
        --photos test_data/corrales/photos \\
        --out  test_data/corrales/output \\
        --duration 75
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from worker.camera_path import generate_camera_path
from worker.claude_camera_path import generate_claude_camera_path
from worker.claude_organize import organize_photos
from worker.claude_qa import qa_video
from worker.exposure_normalize import normalize_directory
from worker.mesh_render import extract_mesh, render_mesh_along_path
from worker.segment import composite_to_train_masks, run_segment
from worker.voiceover import synthesize as synth_voiceover
from worker.voiceover import write_voiceover_script

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True)


MIN_PHOTOS_FOR_RECONSTRUCTION = 40
COLMAP_VOCAB_TREE_MIN_PHOTOS = 50


@dataclass
class PipelineConfig:
    photos: Path
    out: Path
    duration_s: float = 75.0
    fps: int = 30
    width: int = 1920
    height: int = 1080
    train_iterations: int = 30000
    # A6 — default to splatfacto-w; _detect_splatfacto_method falls back to vanilla
    # splatfacto if W isn't registered in the running nerfstudio install.
    splatfacto_method: str = "splatfacto-w"
    appearance_embedding: bool = True  # A6 — per-image embeddings (free with splatfacto-w)
    colmap_matching_method: str | None = None  # None → auto: vocab_tree if photos≥50 else exhaustive
    colmap_num_downscales: int = 1  # was 3 (8x); 1 = 2x. Listing photos are 4-6K, COLMAP needs the pixels.
    densify_grad_threshold: float = 0.0008  # indoor-tuned; default is too aggressive on edges
    densification_interval: int = 200
    supersample: bool = True  # render at 2x then ffmpeg-downscale → kills splat shimmer
    exposure_normalize_enabled: bool = True  # A7 — runs BEFORE stage_organize
    exposure_gain_min: float = 0.5
    exposure_gain_max: float = 2.0
    use_semantic_masking: bool = False  # L3/A10 — default off until SAM2 deps verified
    use_mesh_fallback: bool = False     # mesh re-render when QA severity ∈ {major, fail}
    music_track: Path | None = None
    lut: Path | None = None
    title_text: str | None = None
    listing_description: str | None = None
    use_claude_organize: bool = False
    use_claude_camera_path: bool = False
    use_claude_qa: bool = False
    voiceover_enabled: bool = False
    # set by stages as they run
    workdir: Path = field(init=False)
    photos_normalized: Path = field(init=False)
    masks_composite_dir: Path = field(init=False)
    masks_train_dir: Path = field(init=False)
    mesh_dir: Path = field(init=False)
    mesh_mp4: Path = field(init=False)
    processed_dir: Path = field(init=False)
    train_output_dir: Path = field(init=False)
    organize_json: Path = field(init=False)
    camera_path_json: Path = field(init=False)
    rendered_mp4: Path = field(init=False)
    final_mp4: Path = field(init=False)
    qa_json: Path = field(init=False)
    reshoot_request_json: Path = field(init=False)

    def __post_init__(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        self.workdir = self.out / "workdir"
        self.workdir.mkdir(exist_ok=True)
        # A7 — normalized photos sit alongside photos/ in worker/runs/<job_id>/ so the
        # rest of the pipeline (which reads cfg.photos) can be redirected by swapping
        # cfg.photos to this dir after stage_exposure_normalize runs.
        self.photos_normalized = self.out.parent / "photos_normalized"
        # L3/A10 — segmentation outputs.
        self.masks_composite_dir = self.workdir / "masks_composite"
        self.masks_train_dir = self.workdir / "masks_train"
        # Mesh-fallback outputs (only populated when QA triggers re-render).
        self.mesh_dir = self.workdir / "mesh"
        self.mesh_mp4 = self.out / "final_mesh_fallback.mp4"
        self.processed_dir = self.workdir / "processed"
        self.train_output_dir = self.workdir / "train"
        self.organize_json = self.workdir / "organize.json"
        self.camera_path_json = self.workdir / "camera_path.json"
        self.rendered_mp4 = self.workdir / "rendered.mp4"
        self.final_mp4 = self.out / "final.mp4"
        self.qa_json = self.out / "qa_report.json"
        self.reshoot_request_json = self.out / "reshoot_request.json"
        self.voiceover_script = self.workdir / "voiceover.txt"
        self.voiceover_mp3 = self.workdir / "voiceover.mp3"
        self.vertical_mp4 = self.out / "final_9x16.mp4"
        self.square_mp4 = self.out / "final_1x1.mp4"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    pretty = " ".join(str(c) for c in cmd)
    console.log(f"[dim]$[/] {pretty}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {pretty}")
    console.log(f"[dim]done in {time.time() - t0:.1f}s[/]")


def stage_exposure_normalize(cfg: PipelineConfig) -> None:
    """A7 — luma-based per-photo gain normalization. Runs BEFORE stage_organize so
    Claude sees a consistent set of photos and downstream room labeling is robust to
    bracketed/inconsistent listing-shoot exposures.

    Mutates cfg.photos to point at photos_normalized/ on success — every later stage
    transparently picks up the normalized inputs.
    """
    if not cfg.exposure_normalize_enabled:
        return
    console.rule("[bold cyan]Stage 0 — Exposure normalization (A7)")
    # Idempotent: if photos_normalized exists and contains the same number of files
    # as the source, just swap cfg.photos and skip the recompute.
    src_count = sum(1 for p in cfg.photos.iterdir() if p.is_file())
    if cfg.photos_normalized.exists():
        dst_count = sum(1 for p in cfg.photos_normalized.iterdir() if p.is_file())
        if dst_count >= src_count and dst_count > 0:
            console.log(
                f"photos_normalized has {dst_count} files (source {src_count}), skipping recompute"
            )
            cfg.photos = cfg.photos_normalized
            return
    report = normalize_directory(
        src_dir=cfg.photos,
        dst_dir=cfg.photos_normalized,
        gain_min=cfg.exposure_gain_min,
        gain_max=cfg.exposure_gain_max,
    )
    console.log(
        f"normalized {report.n_photos} photos → median_luma={report.median_luma:.1f} "
        f"clipped={report.clipped_count} (gain ∉ [{cfg.exposure_gain_min}, {cfg.exposure_gain_max}])"
    )
    cfg.photos = cfg.photos_normalized


def stage_segment(cfg: PipelineConfig) -> None:
    """L3/A10 — Grounded-SAM 2 semantic masking.

    Produces per-photo composite masks (8-class), then a binary train-mask in nerfstudio's
    format (255 = train on this pixel, 0 = ignore). Train masks pass through to
    stage_train via the `--pipeline.datamanager.dataparser.mask-path` flag.

    Default-off. Enable with cfg.use_semantic_masking=True (or jobs.use_semantic_masking).
    """
    if not cfg.use_semantic_masking:
        return
    console.rule("[bold cyan]Stage 1.5 — Semantic masking (Grounded-SAM 2)")
    if cfg.masks_train_dir.exists() and any(cfg.masks_train_dir.glob("*.png")):
        console.log("masks_train/ already populated, skipping segmentation")
        return
    try:
        run_segment(cfg.photos, cfg.masks_composite_dir)
        composite_to_train_masks(cfg.masks_composite_dir, cfg.masks_train_dir)
    except ImportError as e:
        # SAM2/GroundingDINO not installed — disable masking for this run, do NOT fail.
        console.log(
            f"[yellow]semantic masking deps missing ({e}); proceeding without masks. "
            f"Add SAM2 + GroundingDINO to the worker image to enable.[/yellow]"
        )
        cfg.use_semantic_masking = False
        return
    n_masks = len(list(cfg.masks_train_dir.glob("*.png")))
    console.log(f"wrote {n_masks} train masks → {cfg.masks_train_dir}")


def stage_organize(cfg: PipelineConfig) -> None:
    """Optional Stage 1 — Claude organizational pass over photos."""
    if not cfg.use_claude_organize:
        return
    console.rule("[bold cyan]Stage 1 — Claude organizational pass")
    if cfg.organize_json.exists():
        console.log("organize.json exists, skipping")
        return
    organize_photos(cfg.photos, cfg.organize_json, listing_description=cfg.listing_description)
    console.log(f"wrote organization → {cfg.organize_json}")


def _count_photos(photos_dir: Path) -> int:
    return (
        len(list(photos_dir.glob("*.[jJ][pP]*[gG]")))
        + len(list(photos_dir.glob("*.png")))
        + len(list(photos_dir.glob("*.PNG")))
    )


def _resolve_matching_method(cfg: PipelineConfig, n_photos: int) -> str:
    if cfg.colmap_matching_method:
        return cfg.colmap_matching_method
    return "vocab_tree" if n_photos >= COLMAP_VOCAB_TREE_MIN_PHOTOS else "exhaustive"


def stage_preprocess(cfg: PipelineConfig) -> None:
    """COLMAP via ns-process-data. Produces transforms.json + downsampled images."""
    console.rule("[bold cyan]Stage 2 — COLMAP via ns-process-data")
    if (cfg.processed_dir / "transforms.json").exists():
        console.log("transforms.json exists, skipping preprocess")
        return
    n_photos = _count_photos(cfg.photos)
    matcher = _resolve_matching_method(cfg, n_photos)
    console.log(
        f"matcher={matcher} num_downscales={cfg.colmap_num_downscales} on {n_photos} photos"
    )
    _run(
        [
            "ns-process-data",
            "images",
            "--data",
            str(cfg.photos),
            "--output-dir",
            str(cfg.processed_dir),
            "--matching-method",
            matcher,
            "--num-downscales",
            str(cfg.colmap_num_downscales),
            "--verbose",
        ]
    )
    _check_colmap_registration(cfg, matcher_used=matcher)


def _check_colmap_registration(cfg: PipelineConfig, matcher_used: str | None = None) -> None:
    transforms = cfg.processed_dir / "transforms.json"
    if not transforms.exists():
        _write_reshoot_request(
            cfg,
            reason="colmap_no_transforms",
            rate=0.0,
            matcher_used=matcher_used,
            unregistered=[],
        )
        raise RuntimeError(
            f"COLMAP produced no transforms.json — feature matching likely failed "
            f"(matcher={matcher_used}). Try with more overlapping photos."
        )
    data = json.loads(transforms.read_text())
    n_input = _count_photos(cfg.photos)
    n_registered = len(data.get("frames", []))
    rate = n_registered / max(1, n_input)
    console.log(f"COLMAP registered {n_registered}/{n_input} photos ({rate:.0%}) via {matcher_used}")
    if rate < 0.6:
        registered_names = {Path(f.get("file_path", "")).name for f in data.get("frames", [])}
        all_names = {p.name for p in cfg.photos.iterdir() if p.is_file()}
        unregistered = sorted(all_names - registered_names)
        _write_reshoot_request(
            cfg,
            reason="colmap_registration_below_threshold",
            rate=rate,
            matcher_used=matcher_used,
            unregistered=unregistered,
        )
        raise RuntimeError(
            f"COLMAP registered only {rate:.0%} ({n_registered}/{n_input}) of photos with "
            f"matcher={matcher_used} — insufficient coverage for reconstruction. "
            f"Reshoot guidance written to {cfg.reshoot_request_json}."
        )
    if rate < 0.8:
        console.log(f"[yellow]warning: only {rate:.0%} registration — output may have gaps[/yellow]")


def _write_reshoot_request(
    cfg: PipelineConfig,
    reason: str,
    rate: float,
    matcher_used: str | None,
    unregistered: list[str],
) -> None:
    """Emit per-room reshoot guidance JSON. Joined with organize.json room labels when present."""
    by_room: dict[str, list[str]] = {}
    organize_data: dict | None = None
    if cfg.organize_json.exists():
        try:
            organize_data = json.loads(cfg.organize_json.read_text())
        except Exception:
            organize_data = None

    if organize_data:
        unreg_set = set(unregistered)
        for photo in organize_data.get("photos", []):
            fname = photo.get("filename") or photo.get("file") or ""
            room = photo.get("room") or photo.get("room_label") or "unlabeled"
            if fname in unreg_set:
                by_room.setdefault(room, []).append(fname)
    else:
        if unregistered:
            by_room["unlabeled"] = list(unregistered)

    per_room_guidance = []
    for room, files in sorted(by_room.items()):
        per_room_guidance.append({
            "room": room,
            "unregistered_count": len(files),
            "unregistered_files": files[:20],  # cap for readability
            "guidance": (
                f"{room} lost {len(files)} photo(s). Reshoot with overlapping mid-height "
                "angles between adjacent shots; avoid single hero angles, pure overhead, "
                "and shots dominated by mirrors / windows."
            ),
        })

    payload = {
        "reason": reason,
        "registration_rate": round(rate, 4),
        "matcher_used": matcher_used,
        "total_unregistered": len(unregistered),
        "per_room": per_room_guidance,
        "general_guidance": [
            "Aim for ≥40 overlapping photos with 60%+ visual overlap between adjacent shots.",
            "Cover each room with a slow walk-through, not single hero angles.",
            "Avoid heavy reflections (mirrors, glossy floors) as the dominant subject.",
            "Keep exposure consistent — disable in-camera HDR bracket-merge if possible.",
        ],
    }
    cfg.reshoot_request_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.reshoot_request_json.write_text(json.dumps(payload, indent=2))
    console.log(f"[yellow]wrote reshoot guidance → {cfg.reshoot_request_json}[/yellow]")


_SPLATFACTO_DETECT_CACHE: dict[str, tuple[str, list[str]]] = {}


def _ns_train_help(method: str | None = None) -> str:
    """Return `ns-train [method] --help` stdout, or empty string on failure."""
    cmd = ["ns-train"] + ([method] if method else []) + ["--help"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        # Tyro writes help to stdout but errors / "method not found" go to stderr.
        return (proc.stdout or "") + (proc.stderr or "")
    except Exception as e:
        console.log(f"[yellow]{' '.join(cmd)} failed: {e}[/yellow]")
        return ""


def _detect_splatfacto_method(requested: str, appearance_embedding: bool) -> tuple[str, list[str]]:
    """Probe ns-train for which splatfacto variant is registered, fall back gracefully.

    Returns (method, extra_args). Caches per-process (cheap subprocess but called per job).

    Decision tree:
    - Requested method exists verbatim → use it.
    - Requested = splatfacto-w but absent → fall back to vanilla splatfacto + try to add
      appearance-embedding flags; if vanilla doesn't accept those either, log and proceed
      WITHOUT A6 (we'd rather train a working vanilla splat than crash for a missing flag).
    - Other absent method → return as-is and let ns-train surface the error to the user.
    """
    cache_key = f"{requested}|{appearance_embedding}"
    if cache_key in _SPLATFACTO_DETECT_CACHE:
        return _SPLATFACTO_DETECT_CACHE[cache_key]

    help_text = _ns_train_help()
    available: set[str] = set()
    for line in help_text.splitlines():
        for token in line.replace(",", " ").replace("{", " ").replace("}", " ").split():
            if token.startswith("splatfacto"):
                available.add(token.strip(".:"))

    method = requested
    if requested not in available:
        if requested == "splatfacto-w" and "splatfacto" in available:
            console.log(
                "[yellow]splatfacto-w not registered in this nerfstudio install — "
                "falling back to vanilla splatfacto. A6 (per-image appearance embeddings) "
                "may be unavailable; install splatfacto-w to enable.[/yellow]"
            )
            method = "splatfacto"
        elif requested == "splatfacto-2dgs":
            # A3 — prefer 2DGS for surface-aligned reconstruction; fall through to
            # splatfacto-big (more gaussians) and then vanilla splatfacto if absent.
            for candidate in ("splatfacto-big", "splatfacto"):
                if candidate in available:
                    console.log(
                        f"[yellow]splatfacto-2dgs not registered — falling back to "
                        f"{candidate}. Install gsplat with 2DGS to get surface priors.[/yellow]"
                    )
                    method = candidate
                    break
        # else: leave method as requested; ns-train will surface the error.

    extra: list[str] = []
    if appearance_embedding:
        if method == "splatfacto-w":
            # Splatfacto-w turns embeddings on by default; no extra flags.
            pass
        else:
            method_help = _ns_train_help(method)
            if "use-appearance-embedding" in method_help or "appearance-embed" in method_help:
                extra = [
                    "--pipeline.model.use-appearance-embedding", "True",
                    "--pipeline.model.appearance-embedding-dim", "32",
                ]
            else:
                console.log(
                    f"[yellow]A6 disabled: method={method!r} does not accept appearance-embedding "
                    f"flags. Continuing without per-image embeddings.[/yellow]"
                )

    console.log(f"resolved splatfacto: method={method} extra_args={extra}")
    result = (method, extra)
    _SPLATFACTO_DETECT_CACHE[cache_key] = result
    return result


def _inject_masks_into_transforms(cfg: PipelineConfig) -> int:
    """L3/A10 — point each transforms.json frame at its corresponding train mask.

    Nerfstudio's NerfstudioDataParser reads per-frame `mask_path` to skip masked-out
    pixels during photometric loss. Injecting here (rather than passing a CLI flag)
    is the path that works across nerfstudio versions.

    Returns: number of frames a mask was attached to.
    """
    transforms_path = cfg.processed_dir / "transforms.json"
    if not transforms_path.exists() or not cfg.masks_train_dir.exists():
        return 0
    data = json.loads(transforms_path.read_text())
    attached = 0
    for frame in data.get("frames", []):
        photo_name = Path(frame["file_path"]).stem
        # Look for a mask matching the photo stem; common case is .png with same stem.
        candidates = list(cfg.masks_train_dir.glob(f"{photo_name}.png"))
        if not candidates:
            continue
        # Use a relative path from the transforms.json location so nerfstudio resolves it.
        rel = Path("..") / cfg.masks_train_dir.name / candidates[0].name
        frame["mask_path"] = str(rel)
        attached += 1
    transforms_path.write_text(json.dumps(data, indent=2))
    return attached


def stage_train(cfg: PipelineConfig) -> None:
    console.rule("[bold cyan]Stage 3 — Splatfacto training")
    final_ckpt = cfg.train_output_dir / "nerfstudio_models"
    if final_ckpt.exists() and any(final_ckpt.rglob("step-*.ckpt")):
        console.log("training checkpoints exist, skipping train")
        return
    if cfg.use_semantic_masking:
        n_masked = _inject_masks_into_transforms(cfg)
        console.log(f"attached {n_masked} train masks to transforms.json")
    method, extra_args = _detect_splatfacto_method(cfg.splatfacto_method, cfg.appearance_embedding)
    cmd = [
        "ns-train",
        method,
        "--data",
        str(cfg.processed_dir),
        "--output-dir",
        str(cfg.train_output_dir),
        "--max-num-iterations",
        str(cfg.train_iterations),
    ]
    # A9 — densification flags. Only added if the chosen model exposes them; otherwise
    # ns-train would error on the unknown flag. Probe the method's --help once per process.
    method_help = _ns_train_help(method)
    if "densify-grad-thresh" in method_help:
        cmd += ["--pipeline.model.densify-grad-thresh", str(cfg.densify_grad_threshold)]
    if "refine-every" in method_help:
        cmd += ["--pipeline.model.refine-every", str(cfg.densification_interval)]
    cmd += extra_args
    cmd += [
        "--viewer.quit-on-train-completion",
        "True",
        "--vis",
        "tensorboard",
        "--experiment-name",
        "demosync",
    ]
    _run(cmd)


def _resolve_latest_train_config(cfg: PipelineConfig) -> Path:
    candidates = sorted(cfg.train_output_dir.rglob("config.yml"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError(f"no Nerfstudio config.yml under {cfg.train_output_dir}")
    return candidates[0]


def stage_camera_path(cfg: PipelineConfig) -> None:
    if cfg.camera_path_json.exists():
        console.log("camera_path.json exists, skipping")
        return
    transforms = cfg.processed_dir / "transforms.json"
    if cfg.use_claude_camera_path and cfg.organize_json.exists():
        console.rule("[bold cyan]Stage 4 — Camera path (Claude property-aware)")
        generate_claude_camera_path(
            transforms_json=transforms,
            organize_json=cfg.organize_json,
            output_json=cfg.camera_path_json,
            listing_description=cfg.listing_description,
            fps=cfg.fps,
            render_width=cfg.width,
            render_height=cfg.height,
        )
    else:
        console.rule("[bold cyan]Stage 4 — Camera path (heuristic)")
        generate_camera_path(
            transforms_json=transforms,
            output_json=cfg.camera_path_json,
            duration_seconds=cfg.duration_s,
            fps=cfg.fps,
            render_width=cfg.width,
            render_height=cfg.height,
        )
    console.log(f"wrote camera path → {cfg.camera_path_json}")


def stage_render(cfg: PipelineConfig) -> None:
    console.rule("[bold cyan]Stage 5 — Render")
    if cfg.rendered_mp4.exists():
        console.log("rendered.mp4 exists, skipping render")
        return
    config_yml = _resolve_latest_train_config(cfg)

    # A4 — supersample. Render at 2× target dims; _post_master downscales with lanczos.
    # ns-render reads dimensions from camera_path.json, so we patch the JSON in-place
    # for this run rather than threading another CLI flag.
    if cfg.supersample:
        path_payload = json.loads(cfg.camera_path_json.read_text())
        original_w = path_payload.get("render_width", cfg.width)
        original_h = path_payload.get("render_height", cfg.height)
        path_payload["render_width"] = original_w * 2
        path_payload["render_height"] = original_h * 2
        path_payload.setdefault("demosync_meta", {})["supersampled"] = True
        path_payload["demosync_meta"]["target_width"] = original_w
        path_payload["demosync_meta"]["target_height"] = original_h
        cfg.camera_path_json.write_text(json.dumps(path_payload, indent=2))
        console.log(f"supersample on: rendering at {original_w * 2}×{original_h * 2}")

    _run(
        [
            "ns-render",
            "camera-path",
            "--load-config",
            str(config_yml),
            "--camera-path-filename",
            str(cfg.camera_path_json),
            "--output-path",
            str(cfg.rendered_mp4),
            "--output-format",
            "video",
        ]
    )


def stage_post(cfg: PipelineConfig) -> None:
    console.rule("[bold cyan]Stage 6 — FFmpeg post")

    if cfg.voiceover_enabled and cfg.use_claude_organize:
        if not cfg.voiceover_mp3.exists():
            console.log("generating voiceover script + audio")
            script = write_voiceover_script(
                organize_json=cfg.organize_json,
                listing_description=cfg.listing_description,
                target_duration_s=cfg.duration_s,
                out_path=cfg.voiceover_script,
            )
            synth_voiceover(script, cfg.voiceover_mp3)

    vf_filters: list[str] = []
    # A4 — when stage_render supersampled, downscale here with lanczos. Must run BEFORE
    # any other vf filter (LUT, drawtext) so those operate at target resolution.
    if cfg.supersample:
        vf_filters.append(f"scale={cfg.width}:{cfg.height}:flags=lanczos")
    if cfg.lut and cfg.lut.exists():
        vf_filters.append(f"lut3d={shlex_quote(str(cfg.lut))}")
    if cfg.title_text:
        safe = cfg.title_text.replace(":", r"\:").replace("'", r"\'")
        vf_filters.append(
            f"drawtext=text='{safe}':fontcolor=white:fontsize=48:"
            "x=(w-text_w)/2:y=h-100:enable='between(t,0,3)':"
            "box=1:boxcolor=black@0.4:boxborderw=12"
        )
    vf_16x9 = ",".join(vf_filters) if vf_filters else None

    _post_master(cfg, vf_16x9)
    _post_crop(cfg, target=cfg.vertical_mp4, w=1080, h=1920, label="9:16")
    _post_crop(cfg, target=cfg.square_mp4,   w=1080, h=1080, label="1:1")

    console.log(Panel(
        f"final 16:9 → {cfg.final_mp4}\n"
        f"final 9:16 → {cfg.vertical_mp4}\n"
        f"final 1:1  → {cfg.square_mp4}",
        style="green",
    ))


def _post_master(cfg: PipelineConfig, vf: str | None) -> None:
    if cfg.final_mp4.exists():
        console.log("final 16:9 exists, skipping")
        return
    cmd: list[str] = ["ffmpeg", "-y", "-i", str(cfg.rendered_mp4)]
    audio_inputs: list[Path] = []
    if cfg.music_track and cfg.music_track.exists():
        cmd += ["-i", str(cfg.music_track)]
        audio_inputs.append(cfg.music_track)
    if cfg.voiceover_enabled and cfg.voiceover_mp3.exists():
        cmd += ["-i", str(cfg.voiceover_mp3)]
        audio_inputs.append(cfg.voiceover_mp3)

    if vf:
        cmd += ["-vf", vf]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p"]

    if audio_inputs:
        # input indices: 0=video, then music if present, then voiceover if present
        filter_parts: list[str] = []
        mix_inputs: list[str] = []
        idx = 1
        if cfg.music_track and cfg.music_track.exists():
            filter_parts.append(
                f"[{idx}:a]volume=0.5,afade=t=in:st=0:d=2,"
                f"afade=t=out:st={cfg.duration_s - 2}:d=2[music]"
            )
            mix_inputs.append("[music]")
            idx += 1
        if cfg.voiceover_enabled and cfg.voiceover_mp3.exists():
            filter_parts.append(f"[{idx}:a]volume=1.0[vo]")
            mix_inputs.append("[vo]")
            idx += 1
        if len(mix_inputs) > 1:
            filter_parts.append(f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=longest[a]")
            map_audio = "[a]"
        else:
            map_audio = mix_inputs[0]
        cmd += [
            "-filter_complex", ";".join(filter_parts),
            "-map", "0:v", "-map", map_audio,
            "-shortest",
            "-c:a", "aac", "-b:a", "192k",
        ]
    cmd += [str(cfg.final_mp4)]
    _run(cmd)


def _post_crop(cfg: PipelineConfig, target: Path, w: int, h: int, label: str) -> None:
    if target.exists():
        console.log(f"{label} exists, skipping")
        return
    src = cfg.final_mp4 if cfg.final_mp4.exists() else cfg.rendered_mp4
    # Smart center crop: scale to cover the target box, then crop to exact dims.
    vf = (
        f"scale=if(gt(a\\,{w}/{h})\\,-2\\,{w}):if(gt(a\\,{w}/{h})\\,{h}\\,-2),"
        f"crop={w}:{h}"
    )
    _run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p",
            "-c:a", "copy" if cfg.final_mp4.exists() else "aac", "-b:a", "192k",
            str(target),
        ]
    )


def stage_qa(cfg: PipelineConfig):
    """Optional Stage 7 — Claude QA over final video. Returns the report (or None)
    so downstream stages (mesh fallback) can react to severity."""
    if not cfg.use_claude_qa:
        return None
    console.rule("[bold cyan]Stage 7 — Claude QA")
    report = qa_video(cfg.final_mp4, cfg.qa_json, frames_dir=cfg.workdir / "qa_frames")
    style = "red" if report.should_hold else ("yellow" if report.severity == "major" else "green")
    console.log(Panel(
        f"severity: {report.severity}\n{report.summary}\n\n"
        f"reshoot requests: {len(report.reshoot_requests)}",
        title="QA",
        border_style=style,
    ))
    return report


def stage_mesh_render(cfg: PipelineConfig, qa_severity: str | None) -> Path | None:
    """Re-render the camera path against an extracted mesh when QA severity is bad.

    Trade: cinematically stable (no spike artifacts, no popping) at the cost of
    view-dependent specular. Default-off; gated by cfg.use_mesh_fallback.

    Returns the produced MP4 path, or None if disabled / extraction failed.
    """
    if not cfg.use_mesh_fallback:
        return None
    if qa_severity not in {"major", "fail"}:
        return None
    console.rule("[bold cyan]Stage 8 — Mesh fallback re-render")
    if cfg.mesh_mp4.exists():
        console.log(f"mesh fallback already produced: {cfg.mesh_mp4}")
        return cfg.mesh_mp4

    train_config = _resolve_latest_train_config(cfg)
    obj_path = extract_mesh(train_config, cfg.mesh_dir)
    if obj_path is None:
        console.log(
            "[yellow]mesh extraction failed (ns-export tsdf/poisson both failed). "
            "Skipping mesh fallback; splat MP4 remains primary output.[/yellow]"
        )
        return None
    ok = render_mesh_along_path(obj_path, cfg.camera_path_json, cfg.mesh_mp4)
    if not ok:
        console.log(
            "[yellow]Open3D OffscreenRenderer not available; cannot complete mesh "
            "fallback. Install open3d to enable.[/yellow]"
        )
        return None
    console.log(f"mesh fallback render → {cfg.mesh_mp4}")
    return cfg.mesh_mp4


def shlex_quote(s: str) -> str:
    # ffmpeg filter args are picky about commas + colons; for filenames we keep it simple.
    return s.replace("\\", "/").replace(":", "\\:")


@app.command()
def run(
    photos: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True, help="Directory of input photos."),
    out: Path = typer.Option(..., help="Output directory."),
    duration: float = typer.Option(75.0, help="Output video duration in seconds."),
    fps: int = typer.Option(30),
    width: int = typer.Option(1920),
    height: int = typer.Option(1080),
    iterations: int = typer.Option(30000, help="Splatfacto training iterations."),
    method: str = typer.Option("splatfacto"),
    music: Path = typer.Option(None, exists=False),
    lut: Path = typer.Option(None, exists=False),
    title: str = typer.Option(None, help="Title-card text shown at intro."),
    skip_post: bool = typer.Option(False),
    listing_description: str = typer.Option(None, help="Listing description text for Claude context."),
    organize: bool = typer.Option(False, help="Run Claude organizational pass (stage 1)."),
    claude_camera: bool = typer.Option(False, help="Use Claude property-aware camera path (requires --organize)."),
    qa: bool = typer.Option(False, help="Run Claude QA pass on final video (stage 7)."),
    voiceover: bool = typer.Option(False, help="Generate + mix in Claude+ElevenLabs voiceover."),
) -> None:
    """Run the full MVP pipeline end-to-end."""
    cfg = PipelineConfig(
        photos=photos,
        out=out,
        duration_s=duration,
        fps=fps,
        width=width,
        height=height,
        train_iterations=iterations,
        splatfacto_method=method,
        music_track=music,
        lut=lut,
        title_text=title,
        listing_description=listing_description,
        use_claude_organize=organize or claude_camera or voiceover,
        use_claude_camera_path=claude_camera,
        use_claude_qa=qa,
        voiceover_enabled=voiceover,
    )
    console.print(Panel.fit(
        f"photos: {cfg.photos}\nout:    {cfg.out}\nduration: {cfg.duration_s}s @ {cfg.fps}fps {cfg.width}x{cfg.height}\n"
        f"method: {cfg.splatfacto_method}\n"
        f"claude: organize={cfg.use_claude_organize} camera={cfg.use_claude_camera_path} qa={cfg.use_claude_qa}",
        title="DemoSync MVP",
        border_style="cyan",
    ))
    stage_exposure_normalize(cfg)
    stage_organize(cfg)
    stage_segment(cfg)
    stage_preprocess(cfg)
    stage_train(cfg)
    stage_camera_path(cfg)
    stage_render(cfg)
    if not skip_post:
        stage_post(cfg)
    else:
        shutil.copy2(cfg.rendered_mp4, cfg.final_mp4)
        console.log(f"--skip-post: copied rendered → {cfg.final_mp4}")
    qa_report = stage_qa(cfg)
    qa_sev = qa_report.severity if qa_report is not None else None
    stage_mesh_render(cfg, qa_sev)


@app.command()
def smoke() -> None:
    """Sanity-check that ns-process-data, ns-train, ns-render, ffmpeg, colmap are available."""
    tools = ["ns-process-data", "ns-train", "ns-render", "ffmpeg", "colmap"]
    missing: list[str] = []
    for t in tools:
        if shutil.which(t) is None:
            missing.append(t)
            console.log(f"[red]missing: {t}[/red]")
        else:
            console.log(f"[green]found:   {t}[/green]")
    if missing:
        console.print(f"[red]install missing tools before running pipeline: {missing}[/red]")
        sys.exit(1)
    console.print("[green]all required tools available[/green]")


if __name__ == "__main__":
    app()
