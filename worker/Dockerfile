# dromni/nerfstudio ships with CUDA, PyTorch, tiny-cuda-nn, gsplat, and nerfstudio
# precompiled — we skip the painful tiny-cuda-nn isolated-build problem entirely.
# https://hub.docker.com/r/dromni/nerfstudio
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

# Safety net: any code in the image (ours or nerfstudio's) that shells out to
# `python` resolves to python3.
RUN ln -sf /usr/bin/python3 /usr/bin/python || true

# Create /app as user-owned. The base image puts nerfstudio under
# /home/user/.local; running the ENTRYPOINT as root would invoke a python
# that can't see that site-packages, and ns-process-data crashes with
# "No module named 'nerfstudio'". We run as `user` to inherit the same
# environment nerfstudio was installed into.
RUN mkdir -p /app && chown -R user:user /app

# ----- user layer: pip install + code copy -----
USER user
WORKDIR /app

# `pip install --user` lands our packages in /home/user/.local alongside
# nerfstudio's own deps, on the same site-packages the ns-* binaries use.
COPY --chown=user:user requirements.txt /app/requirements.txt
RUN python3 -m pip install --user --no-cache-dir -r /app/requirements.txt

COPY --chown=user:user . /app/worker

# RunPod passes DEMOSYNC_JOB_ID via env var; the local daemon passes it as argv[1].
# job_handler.py accepts either.
ENTRYPOINT ["python3", "-m", "worker.job_handler"]
