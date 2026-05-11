"""DemoSync orchestration daemon.

Listens for new jobs on the Postgres `job_queued` NOTIFY channel and dispatches them
to a worker. Two execution modes:

    local:  subprocess spawn of `python -m worker.job_handler <job_id>` on the same
            host. Reads + writes through the shared dashboard/uploads directory.
            Suitable for laptop dev runs (CPU/MPS reconstruction or a local GPU).

    runpod: POSTs to RunPod's GraphQL API to create a pod from a pre-built image,
            passing the job_id via env var. Worker reports status back to Postgres
            directly.

Selects mode via DEMOSYNC_DISPATCH=local|runpod (default: local).

Single-process design — for v1 internal use we don't need horizontal scale. The
daemon does *not* run the heavy work itself; it just dispatches and tracks.
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=True)

console = Console()

DB_URL = os.environ.get("DATABASE_URL", "postgresql://demosync:demosync@localhost:5432/demosync")
DISPATCH = os.environ.get("DEMOSYNC_DISPATCH", "local")
WORKER_PYTHON = os.environ.get("WORKER_PYTHON", sys.executable)
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_TEMPLATE_ID = os.environ.get("RUNPOD_TEMPLATE_ID")

_running: dict[str, subprocess.Popen[bytes]] = {}
_shutdown = threading.Event()


def dispatch_local(job_id: str) -> None:
    if job_id in _running:
        console.log(f"[yellow]job {job_id[:8]} already running, skipping[/yellow]")
        return
    cmd = [WORKER_PYTHON, "-m", "worker.job_handler", job_id]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT))
    console.log(f"spawn local: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env)
    _running[job_id] = proc
    threading.Thread(target=_reap, args=(job_id, proc), daemon=True).start()


def _reap(job_id: str, proc: subprocess.Popen[bytes]) -> None:
    rc = proc.wait()
    _running.pop(job_id, None)
    if rc == 0:
        console.log(f"[green]job {job_id[:8]} exited 0[/green]")
    else:
        console.log(f"[red]job {job_id[:8]} exited {rc}[/red]")
        with psycopg.connect(DB_URL) as conn, conn.cursor() as cur:
            cur.execute(
                """UPDATE jobs SET status = 'failed', error_message = COALESCE(error_message, %s),
                   finished_at = COALESCE(finished_at, now())
                   WHERE id = %s AND status NOT IN ('complete','needs_review','failed')""",
                (f"worker exited {rc}", job_id),
            )
            conn.commit()


def dispatch_runpod(job_id: str) -> None:
    if not RUNPOD_API_KEY or not RUNPOD_TEMPLATE_ID:
        console.log("[red]RUNPOD_API_KEY or RUNPOD_TEMPLATE_ID not set; falling back to local[/red]")
        dispatch_local(job_id)
        return
    # Minimal RunPod GraphQL pod-spawn. Full impl tracked separately.
    import urllib.request, json  # noqa: PLC0415
    payload: dict[str, Any] = {
        "query": "mutation podRentInterruptable($input: PodRentInterruptableInput!) { podRentInterruptable(input: $input) { id machineId } }",
        "variables": {
            "input": {
                "templateId": RUNPOD_TEMPLATE_ID,
                "gpuTypeId": os.environ.get("RUNPOD_GPU_TYPE", "NVIDIA GeForce RTX 4090"),
                "cloudType": "SECURE",
                "containerDiskInGb": 50,
                "volumeInGb": 0,
                "minVcpuCount": 4,
                "minMemoryInGb": 16,
                "env": [
                    {"key": "DEMOSYNC_JOB_ID", "value": job_id},
                    {"key": "DATABASE_URL", "value": DB_URL},
                    {"key": "ANTHROPIC_API_KEY", "value": os.environ.get("ANTHROPIC_API_KEY", "")},
                    {"key": "SUPABASE_URL", "value": os.environ.get("SUPABASE_URL", "")},
                    {"key": "SUPABASE_SERVICE_ROLE_KEY", "value": os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")},
                ],
            },
        },
    }
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        pod_id = body.get("data", {}).get("podRentInterruptable", {}).get("id")
        console.log(f"runpod pod {pod_id} created for job {job_id[:8]}")
        with psycopg.connect(DB_URL) as conn, conn.cursor() as cur:
            cur.execute("UPDATE jobs SET worker_pod_id = %s WHERE id = %s", (pod_id, job_id))
            conn.commit()
    except Exception as e:
        console.log(f"[red]runpod dispatch failed: {e}[/red]")
        with psycopg.connect(DB_URL) as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET status = 'failed', error_message = %s, fail_stage = 'dispatch' WHERE id = %s",
                (f"runpod dispatch failed: {e}", job_id),
            )
            conn.commit()


def dispatch(job_id: str) -> None:
    if DISPATCH == "runpod":
        dispatch_runpod(job_id)
    else:
        dispatch_local(job_id)


def claim_queued_jobs(conn: psycopg.Connection) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT id::text FROM jobs WHERE status = 'queued' ORDER BY created_at ASC")
        return [r[0] for r in cur.fetchall()]


def main() -> None:
    console.print(f"[bold cyan]demosync daemon[/] dispatch={DISPATCH} db={DB_URL}")

    def _sigterm(*_: object) -> None:
        _shutdown.set()
        console.log("shutting down…")

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    with psycopg.connect(DB_URL, autocommit=True) as conn:
        conn.execute("LISTEN job_queued;")
        # On startup, claim anything already queued (daemon may have been offline).
        for jid in claim_queued_jobs(conn):
            dispatch(jid)

        while not _shutdown.is_set():
            # poll the socket
            r, _, _ = select.select([conn], [], [], 1.0)
            if not r:
                continue
            conn.execute("SELECT 1")  # flush
            for notify in conn.notifies(timeout=0):
                console.log(f"NOTIFY {notify.channel} → {notify.payload}")
                dispatch(notify.payload)

    # graceful: don't kill running workers; just exit
    for jid in list(_running):
        console.log(f"leaving job {jid[:8]} running")


if __name__ == "__main__":
    main()
