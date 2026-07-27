FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ffmpeg/ffprobe: used by helpers/video_helper.py and tasks/video_scan_task.py
# libgl1/libglib2.0-0: required by opencv-python at import time
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Model weights are baked into the AMI at /model-cache/{SERVICE_NAME}/ and bind-mounted
# into the container at runtime (not part of the image). /tmp/shared_jobs is likewise a
# host bind mount shared with the lip_sync and scene_detection containers.
RUN mkdir -p /model-cache /tmp/shared_jobs

CMD ["python", "worker.py"]
