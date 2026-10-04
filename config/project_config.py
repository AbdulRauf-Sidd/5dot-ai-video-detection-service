import os

from dotenv import load_dotenv

load_dotenv()

# --- Service identity ---
SERVICE_NAME = os.environ["SERVICE_NAME"]

# --- Paths ---
SHARED_JOBS_DIR = os.getenv("SHARED_JOBS_DIR", "/tmp/shared_jobs")
MODEL_CACHE_DIR = f"/model-cache/{SERVICE_NAME}"
CHECKPOINT = os.path.join(MODEL_CACHE_DIR, "fused_best.pt")
RAFT_CHECKPOINT = os.path.join(MODEL_CACHE_DIR, "raft-sintel.pth")
XCLIP_LOCAL_DIR = os.path.join(MODEL_CACHE_DIR, "xclip-base-patch16")

# --- ML ---
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
# --- RAFT optical-flow speed/accuracy knobs ---
# Original pipeline: RAFT_BATCH_SIZE=1, FLOW_PAIR_STRIDE=1, GPU_CHUNK_CONCURRENCY=1
# (all 95 consecutive frame pairs per chunk, one pair per RAFT call, chunks in
# sequence). Set all three to 1 to restore it exactly.
#
# RAFT dominates inference time and the GPU was already ~90% busy, so only less
# RAFT work gave a real speedup. Measured with benchmark.py on a 107s video:
#   original             135.7s
#   stride 2              74.6s  (1.82x)
#   batch 8 + stride 2    69.2s  (1.96x)  <- chosen default
# Per-chunk probabilities moved by mean 0.033 / max 0.12; video score
# 0.5368 -> 0.5377, no verdict flips. Validate on labelled videos before relying
# on it near the threshold.

# Frame pairs per RAFT call on GPU. Original: 1. Alone gives ~1.08x (GPU already
# busy); with stride 2 it removes per-call overhead. Shifts probabilities ~1e-3
# (RAFT runs partly in fp16, batched kernels round differently).
RAFT_BATCH_SIZE = int(os.getenv("RAFT_BATCH_SIZE", "8"))
# Use every Nth consecutive frame pair (i, i+1): 2 -> 48 flows instead of 95,
# each still frame-to-frame. Original: 1. Roughly halves RAFT time; changes results.
FLOW_PAIR_STRIDE = int(os.getenv("FLOW_PAIR_STRIDE", "2"))
# Chunks processed concurrently on GPU, sharing one copy of the models.
# Original: 1. Kept at 1: benchmarked 0.97x at 2 (GPU already saturated) while
# adding memory and OOM risk -- an OOM'd chunk is silently scored 0.0.
GPU_CHUNK_CONCURRENCY = int(os.getenv("GPU_CHUNK_CONCURRENCY", "1"))

# --- SQS ---
SQS_QUEUE_URL = os.environ["SQS_QUEUE_URL"]
IDLE_TIMEOUT_SECONDS = int(os.getenv("IDLE_TIMEOUT_SECONDS", "60"))

# --- Postgres ---
DB_HOST = os.environ["DB_HOST"]
DB_NAME = os.environ["DB_NAME"]
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_PORT = int(os.getenv("DB_PORT", "5432"))

# --- Webhook ---
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")
WEBHOOK_MAX_RETRIES = 3

# --- AWS S3 (source download / thumbnail upload) ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "5dot-storage")
