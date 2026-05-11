# dromni/nerfstudio ships with CUDA, PyTorch, tiny-cuda-nn, gsplat, and nerfstudio
# precompiled — we skip the painful tiny-cuda-nn isolated-build problem entirely.
# https://hub.docker.com/r/dromni/nerfstudio
#
# Image inspection (sha256:dca63d3...):
#   User: '1000'   (the home is /home/user but the user is referenced by UID)
#   PATH includes /home/user/.local/bin — that's where nerfstudio's ns-* scripts
#                                          and its python's site-packages live.
FROM dromni/nerfstudio:1.1.4

# ----- root layer: system packages + permissions -----
USER root
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Tools needed on top of the base image:
#   ffmpeg — invoked by worker.pipeline.stage_post and the QA frame extractor
#   colmap — invoked by ns-process-data for SfM
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    colmap \
    && rm -rf /var/lib/apt/lists/*

# Safety net: any code that shells out to `python` resolves to python3.
RUN ln -sf /usr/bin/python3 /usr/bin/python || true

# /app is created here so the unprivileged layer can write to it. Use numeric
# UID/GID — the base image references the user by UID 1000 (no `user` entry
# in /etc/passwd that we can rely on for USER/chown by name).
RUN mkdir -p /app && chown -R 1000:1000 /app

# ----- unprivileged layer: pip install + code copy -----
# Running as UID 1000 means our subprocesses inherit the same env nerfstudio
# was installed into. ns-process-data, ns-train, ns-render all use the python
# in /home/user/.local that has the nerfstudio module on its path.
USER 1000
WORKDIR /app

# pip --user lands our deps in /home/user/.local alongside nerfstudio's own
# site-packages — same import path for both.
COPY --chown=1000:1000 requirements.txt /app/requirements.txt
RUN python3 -m pip install --user --no-cache-dir -r /app/requirements.txt

COPY --chown=1000:1000 . /app/worker

# RunPod passes DEMOSYNC_JOB_ID via env var; the local daemon passes it as argv[1].
# job_handler.py accepts either.
ENTRYPOINT ["python3", "-m", "worker.job_handler"]
