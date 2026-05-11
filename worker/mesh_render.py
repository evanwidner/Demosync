"""Mesh-fallback render — extract a textured mesh from the trained gaussian splat
checkpoint, then render the same camera path against the mesh.

Triggered by stage_qa when QA severity ∈ {major, fail}: the splat render is
unusable, so we fall back to a mesh render which is cinematically stable
(no spike artifacts, no popping) at the cost of view-dependent specular.

Two extractor backends, tried in order:
  1. gsplat 2DGS mesh extractor — preferred when the trained model is splatfacto-2dgs.
  2. SuGaR-style poisson reconstruction from gaussian centroids — universal fallback,
     lower fidelity.

Renderer: Open3D headless OffscreenRenderer reading the same camera_path.json used by
ns-render so framing matches frame-for-frame.

NOTE: this module is best-effort. If neither extractor or the headless renderer is
available in the running environment, stage_mesh_render logs and skips — the
splat-render MP4 remains the primary output.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


def extract_mesh(train_config_yml: Path, mesh_output_dir: Path) -> Path | None:
    """Try to extract a mesh from the latest checkpoint. Returns the .obj path or None."""
    mesh_output_dir.mkdir(parents=True, exist_ok=True)
    obj_path = mesh_output_dir / "scene.obj"
    if obj_path.exists():
        return obj_path

    # Try ns-export (nerfstudio's own mesh exporter) first — works for splatfacto-2dgs.
    for export_kind in ("tsdf", "poisson"):
        try:
            subprocess.run(
                [
                    "ns-export", export_kind,
                    "--load-config", str(train_config_yml),
                    "--output-dir", str(mesh_output_dir),
                ],
                check=True, capture_output=True, text=True, timeout=900,
            )
            # ns-export writes mesh.ply or mesh.obj; find it.
            for cand in mesh_output_dir.glob("mesh.*"):
                if cand.suffix in {".obj", ".ply"}:
                    if cand.suffix == ".ply":
                        # Convert to obj if Open3D available.
                        try:
                            import open3d as o3d  # type: ignore
                            mesh = o3d.io.read_triangle_mesh(str(cand))
                            o3d.io.write_triangle_mesh(str(obj_path), mesh)
                            return obj_path
                        except ImportError:
                            return cand
                    return cand
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def render_mesh_along_path(mesh_path: Path, camera_path_json: Path, output_mp4: Path) -> bool:
    """Render mesh frame-by-frame along camera_path_json and encode to output_mp4.

    Returns True on success, False if the headless renderer isn't available.
    """
    try:
        import open3d as o3d  # type: ignore
    except ImportError:
        return False

    payload = json.loads(camera_path_json.read_text())
    width = int(payload.get("render_width", 1920))
    height = int(payload.get("render_height", 1080))
    fps = int(payload.get("fps", 30))
    frames = payload["camera_path"]

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.add_geometry("mesh", mesh, o3d.visualization.rendering.MaterialRecord())
    renderer.scene.set_background(np.array([0, 0, 0, 1.0]))

    frames_dir = output_mp4.parent / "_mesh_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for i, entry in enumerate(frames):
        c2w = np.array(entry["camera_to_world"]).reshape(4, 4)
        eye = c2w[:3, 3]
        forward = -c2w[:3, 2]
        target = eye + forward
        up = c2w[:3, 1]
        renderer.scene.camera.look_at(target.tolist(), eye.tolist(), up.tolist())
        img = renderer.render_to_image()
        o3d.io.write_image(str(frames_dir / f"f_{i:06d}.png"), img)

    # Encode frames → mp4 via ffmpeg.
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "f_%06d.png"),
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p",
            str(output_mp4),
        ],
        check=True,
    )
    return True
