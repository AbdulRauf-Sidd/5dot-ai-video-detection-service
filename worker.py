"""video_ai_service worker.

Standalone script (no FastAPI/Celery): long-polls its own SQS queue in an
infinite loop, does raw-SQL Postgres reads/writes, and reports completion to
the core service via webhook. See config/project_config.py for env vars.
"""

import json
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import boto3

import db
import shared_storage
import webhook
from config.project_config import (
    AWS_REGION,
    DEVICE,
    IDLE_TIMEOUT_SECONDS,
    SERVICE_NAME,
    SQS_QUEUE_URL,
    THRESHOLD,
)
from helpers.video_helper import chunk_bounds, infer_chunk, split_video_into_chunks
from ml_models.video import load_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(SERVICE_NAME)

CHUNK_LENGTH_SECONDS = 5
INFERENCE_MAX_ATTEMPTS = 3  # 1 initial attempt + 2 retries, per transient errors like CUDA OOM


def extract_job_id(message: dict) -> str:
    body = message["Body"]
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict) and "job_id" in parsed:
            return str(parsed["job_id"])
    except (json.JSONDecodeError, TypeError):
        pass
    return body.strip()


def _run_chunk_inference(chunks: list[str]) -> list[dict]:
    if DEVICE == "cuda":
        return [infer_chunk(c) for c in chunks]
    with ThreadPoolExecutor(max_workers=min(len(chunks), os.cpu_count() or 4)) as ex:
        return list(ex.map(infer_chunk, chunks))


def _process_chunks(job_id: str, source_path: str) -> list[dict]:
    chunks_dir = tempfile.mkdtemp(prefix=f"{job_id}_")
    try:
        chunks = split_video_into_chunks(source_path, chunks_dir, CHUNK_LENGTH_SECONDS)
        if not chunks:
            raise RuntimeError("No video chunks could be extracted.")
        return _run_chunk_inference(chunks)
    finally:
        for name in os.listdir(chunks_dir):
            try:
                os.remove(os.path.join(chunks_dir, name))
            except OSError:
                pass
        try:
            os.rmdir(chunks_dir)
        except OSError:
            pass


def _run_inference_with_retry(job_id: str, source_path: str) -> list[dict]:
    last_exc = None
    for attempt in range(1, INFERENCE_MAX_ATTEMPTS + 1):
        try:
            return _process_chunks(job_id, source_path)
        except Exception as exc:
            last_exc = exc
            logger.warning("Inference attempt %s/%s failed for job %s: %s",
                            attempt, INFERENCE_MAX_ATTEMPTS, job_id, exc)
            if DEVICE == "cuda":
                import torch
                torch.cuda.empty_cache()
    raise last_exc


def process_job(conn, job_id: str) -> None:
    job = db.fetch_job(conn, job_id)
    if not job:
        logger.error("Job %s not found in detection_requests", job_id)
        return
    if not job.get("detect_ai_video"):
        logger.info("Job %s did not request video detection, skipping", job_id)
        return

    db.mark_processing(conn, job_id)

    try:
        source_path = shared_storage.get_source_file(job)
        chunk_results = _run_inference_with_retry(job_id, source_path)

        for i, r in enumerate(chunk_results):
            start, end = chunk_bounds(r["chunk"])
            probability = r["result"].get("probability", 0.0)
            db.update_chunk(conn, job_id, i, probability, start, end)

        overall_score = sum(r["result"].get("probability", 0.0) for r in chunk_results) / len(chunk_results)
        db.save_result(conn, job_id, overall_score)

        webhook.notify(job_id, "complete", {"score": overall_score, "threshold": THRESHOLD})

    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        db.mark_failed(conn, job_id, str(exc))
        webhook.notify(job_id, "failed", {"error": str(exc)})

    finally:
        shared_storage.cleanup_if_last(conn, job_id)


def main():
    logger.info("Loading %s models from /model-cache/%s ...", SERVICE_NAME, SERVICE_NAME)
    load_models()
    logger.info("Models loaded, connecting to Postgres...")
    conn = db.connect()

    sqs = boto3.client("sqs", region_name=AWS_REGION)
    last_activity_at = time.time()

    logger.info("Polling %s", SQS_QUEUE_URL)
    while True:
        resp = sqs.receive_message(
            QueueUrl=SQS_QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
        )
        messages = resp.get("Messages", [])

        if not messages:
            if time.time() - last_activity_at >= IDLE_TIMEOUT_SECONDS:
                logger.info("Idle: no messages in the last %ss", IDLE_TIMEOUT_SECONDS)
                last_activity_at = time.time()
            continue

        last_activity_at = time.time()
        message = messages[0]
        job_id = extract_job_id(message)

        logger.info("Received job %s", job_id)
        try:
            process_job(conn, job_id)
        except Exception:
            logger.exception("Unhandled error processing job %s", job_id)
        finally:
            sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=message["ReceiptHandle"])
            logger.info("Deleted SQS message for job %s", job_id)


if __name__ == "__main__":
    main()
