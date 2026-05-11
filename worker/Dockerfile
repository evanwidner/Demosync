# dromni/nerfstudio ships with CUDA, PyTorch, tiny-cuda-nn, gsplat, and nerfstudio
# precompiled — we skip the painful tiny-cuda-nn isolated-build problem entirely.
# https://hub.docker.com/r/dromni/nerfstudio
FROM dromni/nerfstudio:1.1.4

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

WORKDIR /app

# Only the packages NOT already in the nerfstudio base image. torch / torchvision /
# numpy / scipy / opencv / Pillow / tqdm / rich / typer / tinycudann / gsplat /
# nerfstudio itself are all preinstalled — adding them here would risk version drift.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/worker

# RunPod passes DEMOSYNC_JOB_ID via env var; the local daemon passes it as argv[1].
# job_handler.py accepts either.
ENTRYPOINT ["python", "-m", "worker.job_handler"]
