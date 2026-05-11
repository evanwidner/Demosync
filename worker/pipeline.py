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
from worker.voiceover import synthesize as synth_voiceover
from worker.voiceover import write_voiceover_script

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True)


@dataclass
class PipelineConfig:
    photos: Path
    out: Path
    duration_s: float = 75.0
    fps: int = 30
    width: int = 1920
    height: int = 1080
    train_iterations: int = 30000
    splatfacto_method: str = "splatfacto"  # later: "splatfacto-big" or our depth-supervised fork
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
    processed_dir: Path = field(init=False)
    train_output_dir: Path = field(init=False)
    organize_json: Path = field(init=False)
    camera_path_json: Path = field(init=False)
    rendered_mp4: Path = field(init=False)
    final_mp4: Path = field(init=False)
    qa_json: Path = field(init=False)

    def __post_init__(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        self.workdir = self.out / "workdir"
        self.workdir.mkdir(exist_ok=True)
        self.processed_dir = self.workdir / "processed"
        self.train_output_dir = self.workdir / "train"
        self.organize_json = self.workdir / "organize.json"
        self.camera_path_json = self.workdir / "camera_path.json"
        self.rendered_mp4 = self.workdir / "rendered.mp4"
        self.final_mp4 = self.out / "final.mp4"
        self.qa_json = self.out / "qa_report.json"
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


def stage_preprocess(cfg: PipelineConfig) -> None:
    """COLMAP via ns-process-data. Produces transforms.json + downsampled images."""
    console.rule("[bold cyan]Stage 2 — COLMAP via ns-process-data")
    if (cfg.processed_dir / "transforms.json").exists():
        console.log("transforms.json exists, skipping preprocess")
        return
    _run(
        [
            "ns-process-data",
            "images",
            "--data",
            str(cfg.photos),
            "--output-dir",
            str(cfg.processed_dir),
            "--matching-method",
            "exhaustive",
            "--num-downscales",
            "3",
            "--verbose",
        ]
    )
    _check_colmap_registration(cfg)


def _check_colmap_registration(cfg: PipelineConfig) -> None:
    transforms = cfg.processed_dir / "transforms.json"
    if not transforms.exists():
        raise RuntimeError(
            f"COLMAP produced no transforms.json — feature matching likely failed. "
            f"Try with more overlapping photos."
        )
    data = json.loads(transforms.read_text())
    n_input = len(list(cfg.photos.glob("*.[jJ][pP]*[gG]"))) + len(list(cfg.photos.glob("*.png")))
    n_registered = len(data.get("frames", []))
    rate = n_registered / max(1, n_input)
    console.log(f"COLMAP registered {n_registered}/{n_input} photos ({rate:.0%})")
    if rate < 0.6:
        raise RuntimeError(
            f"COLMAP registered only {rate:.0%} of photos — insufficient coverage for "
            f"reconstruction. Need overlapping shots, not single hero angles per room."
        )
    if rate < 0.8:
        console.log(f"[yellow]warning: only {rate:.0%} registration — output may have gaps[/yellow]")


def stage_train(cfg: PipelineConfig) -> None:
    console.rule("[bold cyan]Stage 3 — Splatfacto training")
    final_ckpt = cfg.train_output_dir / "nerfstudio_models"
    if final_ckpt.exists() and any(final_ckpt.rglob("step-*.ckpt")):
        console.log("training checkpoints exist, skipping train")
        return
    _run(
        [
            "ns-train",
            cfg.splatfacto_method,
            "--data",
            str(cfg.processed_dir),
            "--output-dir",
            str(cfg.train_output_dir),
            "--max-num-iterations",
            str(cfg.train_iterations),
            "--viewer.quit-on-train-completion",
            "True",
            "--vis",
            "tensorboard",
            "--experiment-name",
            "demosync",
        ]
    )


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


def stage_qa(cfg: PipelineConfig) -> None:
    """Optional Stage 7 — Claude QA over final video."""
    if not cfg.use_claude_qa:
        return
    console.rule("[bold cyan]Stage 7 — Claude QA")
    report = qa_video(cfg.final_mp4, cfg.qa_json, frames_dir=cfg.workdir / "qa_frames")
    style = "red" if report.should_hold else ("yellow" if report.severity == "major" else "green")
    console.log(Panel(
        f"severity: {report.severity}\n{report.summary}\n\n"
        f"reshoot requests: {len(report.reshoot_requests)}",
        title="QA",
        border_style=style,
    ))


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
    stage_organize(cfg)
    stage_preprocess(cfg)
    stage_train(cfg)
    stage_camera_path(cfg)
    stage_render(cfg)
    if not skip_post:
        stage_post(cfg)
    else:
        shutil.copy2(cfg.rendered_mp4, cfg.final_mp4)
        console.log(f"--skip-post: copied rendered → {cfg.final_mp4}")
    stage_qa(cfg)


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
