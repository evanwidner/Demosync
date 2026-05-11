"""Run a single dashboard-queued job end-to-end.

Loaded by the daemon as:

    python -m worker.job_handler <job_id>

Responsibilities:
    1. Fetch the job + property + inputs from Postgres.
    2. Stage input photos into a local working directory (read from the dashboard's
       uploads/ tree; in prod this would pull from S3).
    3. Call worker.pipeline.run with the right flags.
    4. Stream status updates back to Postgres at each stage transition.
    5. On success: create `outputs` rows for the rendered MP4(s), populate
       `reconstruction_features`, and persist a `qa_reports` row if QA ran.
    6. On failure: mark status=failed with error_message + fail_stage.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=True)

DB_URL = os.environ.get("DATABASE_URL", "postgresql://demosync:demosync@localhost:5432/demosync")
UPLOADS_ROOT = Path(os.environ.get("STORAGE_ROOT", ROOT / "dashboard" / "uploads"))


def fetch_job(conn: psycopg.Connection, job_id: str) -> dict[str, Any]:
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
        job = cur.fetchone()
        if not job:
            raise RuntimeError(f"no such job {job_id}")
        cur.execute("SELECT * FROM properties WHERE id = %s", (job["property_id"],))
        prop = cur.fetchone()
        cur.execute(
            "SELECT * FROM inputs WHERE property_id = %s ORDER BY uploaded_at",
            (job["property_id"],),
        )
        inputs = cur.fetchall()
    return {"job": job, "property": prop, "inputs": inputs}


def update_status(conn: psycopg.Connection, job_id: str, status: str, **extra: Any) -> None:
    sets = ["status = %s"]
    vals: list[Any] = [status]
    for k, v in extra.items():
        sets.append(f"{k} = %s")
        vals.append(v)
    vals.append(job_id)
    sql = f"UPDATE jobs SET {', '.join(sets)} WHERE id = %s"
    with conn.cursor() as cur:
        cur.execute(sql, vals)
    conn.commit()


def stage_inputs(inputs: list[dict[str, Any]], dest: Path) -> Path:
    import httpx
    dest.mkdir(parents=True, exist_ok=True)

    supabase_url = os.environ.get("SUPABASE_URL", "")
    service_role_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    headers = {"Authorization": f"Bearer {service_role_key}"}

    for inp in inputs:
        target = dest / inp["original_filename"]
        if target.exists():
            continue

        # Download from Supabase Storage
        download_url = f"{supabase_url}/storage/v1/object/inputs/{inp['storage_key']}"
        resp = httpx.get(download_url, headers=headers, timeout=60)

        if resp.status_code != 200:
            # Try public URL as fallback
            download_url = f"{supabase_url}/storage/v1/object/public/inputs/{inp['storage_key']}"
            resp = httpx.get(download_url, timeout=60)

        if resp.status_code != 200:
            raise RuntimeError(f"failed to download {inp['storage_key']}: {resp.status_code}")

        target.write_bytes(resp.content)

    return dest


def register_output(
    conn: psycopg.Connection, job_id: str, kind: str, source: Path, storage_subdir: str
) -> str:
    import httpx

    supabase_url = os.environ.get("SUPABASE_URL", "")
    service_role_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

    storage_key = f"{storage_subdir}/{source.name}"

    # Upload to Supabase Storage
    with open(source, "rb") as f:
        video_bytes = f.read()

    upload_url = f"{supabase_url}/storage/v1/object/outputs/{storage_key}"
    headers = {
        "Authorization": f"Bearer {service_role_key}",
        "Content-Type": "video/mp4",
        "x-upsert": "true",
    }

    resp = httpx.put(upload_url, content=video_bytes, headers=headers, timeout=300)
    resp.raise_for_status()

    public_url = f"{supabase_url}/storage/v1/object/public/outputs/{storage_key}"

    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO outputs (job_id, kind, storage_key, public_url)
               VALUES (%s, %s, %s, %s)""",
            (job_id, kind, storage_key, public_url),
        )
    conn.commit()
    return storage_key


def register_qa(conn: psycopg.Connection, job_id: str, qa_json_path: Path) -> str | None:
    if not qa_json_path.exists():
        return None
    payload = json.loads(qa_json_path.read_text())
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO qa_reports (job_id, severity, summary, per_frame, reshoot_requests)
               VALUES (%s, %s, %s, %s, %s)""",
            (
                job_id,
                payload.get("severity", "minor"),
                payload.get("summary"),
                json.dumps(payload.get("per_frame", [])),
                json.dumps(payload.get("reshoot_requests", [])),
            ),
        )
    conn.commit()
    return payload.get("severity")


def register_features(
    conn: psycopg.Connection, job_id: str, organize_json_path: Path
) -> None:
    if not organize_json_path.exists():
        return
    data = json.loads(organize_json_path.read_text())
    listing = data.get("listing", {})
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO reconstruction_features
               (job_id, ceiling_height_estimate_ft, room_count,
                architectural_style_guess)
               VALUES (%s, %s, %s, %s)""",
            (
                job_id,
                listing.get("ceiling_height_estimate_ft"),
                len(listing.get("detected_rooms", [])) or None,
                listing.get("architectural_style_guess"),
            ),
        )
    conn.commit()


def run(job_id: str) -> int:
    # prepare_threshold=None disables psycopg's automatic prepared-statement
    # caching. Required when DATABASE_URL points at Supabase's transaction
    # pooler (port 6543) — pgbouncer in transaction mode breaks on prepared
    # statements. Harmless when running against a direct connection.
    with psycopg.connect(DB_URL, prepare_threshold=None) as conn:
        try:
            ctx = fetch_job(conn, job_id)
        except Exception as e:
            print(f"FATAL: could not load job {job_id}: {e}", file=sys.stderr)
            return 2

        job = ctx["job"]
        prop = ctx["property"]
        inputs = ctx["inputs"]
        update_status(conn, job_id, "preprocessing", started_at="now()", worker_pod_id="local")

        workdir = ROOT / "worker" / "runs" / job_id
        photos_dir = workdir / "photos"
        out_dir = workdir / "out"
        photos_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            stage_inputs(inputs, photos_dir)
        except Exception as e:
            update_status(conn, job_id, "failed", error_message=str(e), fail_stage="ingest")
            return 3

        # Lazy import to avoid hard-deps if running on a CPU-only node for testing
        try:
            from worker.pipeline import PipelineConfig, stage_organize, stage_preprocess, stage_train, stage_camera_path, stage_render, stage_post, stage_qa
        except Exception as e:
            update_status(conn, job_id, "failed", error_message=f"pipeline import error: {e}", fail_stage="bootstrap")
            return 4

        cfg = PipelineConfig(
            photos=photos_dir,
            out=out_dir,
            duration_s=float(job["duration_seconds"]),
            listing_description=prop.get("listing_description"),
            use_claude_organize=bool(
                job["use_claude_organize"]
                or job["use_claude_camera_path"]
                or job["voiceover_enabled"]
            ),
            use_claude_camera_path=bool(job["use_claude_camera_path"]),
            use_claude_qa=bool(job["use_claude_qa"]),
            voiceover_enabled=bool(job["voiceover_enabled"]),
        )

        stage_sequence = [
            ("preprocessing", lambda: stage_organize(cfg)),
            ("colmap",        lambda: stage_preprocess(cfg)),
            ("training",      lambda: stage_train(cfg)),
            ("rendering",     lambda: (stage_camera_path(cfg), stage_render(cfg), stage_post(cfg))),
            ("qa",            lambda: stage_qa(cfg)),
        ]
        for status, fn in stage_sequence:
            update_status(conn, job_id, status)
            try:
                fn()
            except Exception as e:
                tb = traceback.format_exc()
                update_status(
                    conn, job_id, "failed",
                    error_message=f"{e}\n\n{tb}",
                    fail_stage=status,
                    finished_at="now()",
                )
                return 5

        out_subdir = f"properties/{prop['id']}/outputs/{job_id}"
        if cfg.final_mp4.exists():
            register_output(conn, job_id, "mp4_16x9", cfg.final_mp4, out_subdir)
        if cfg.vertical_mp4.exists():
            register_output(conn, job_id, "mp4_9x16", cfg.vertical_mp4, out_subdir)
        if cfg.square_mp4.exists():
            register_output(conn, job_id, "mp4_1x1", cfg.square_mp4, out_subdir)
        register_features(conn, job_id, cfg.organize_json)
        qa_severity = register_qa(conn, job_id, cfg.qa_json) if cfg.use_claude_qa else None

        final_status = "needs_review" if qa_severity == "fail" else "complete"
        update_status(conn, job_id, final_status, finished_at="now()")
        return 0


if __name__ == "__main__":
    job_id = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DEMOSYNC_JOB_ID")
    if not job_id:
        print(
            "usage: python -m worker.job_handler <job_id>  "
            "(or set DEMOSYNC_JOB_ID env var)",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.exit(run(job_id))
