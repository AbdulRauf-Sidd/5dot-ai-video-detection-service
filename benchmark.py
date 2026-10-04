"""
Compare the original inference path against batched/concurrent configurations
on a real video from S3: wall-clock time, per-chunk probabilities, peak GPU memory.

Run inside the worker container on the GPU instance (needs the model weights
under /model-cache/{SERVICE_NAME}/ and AWS credentials/instance role):

    python benchmark.py <s3_url> [BATCH:CONCURRENCY ...]

    python benchmark.py s3://5dot-storage/uploads/video.mp4
    python benchmark.py s3://5dot-storage/uploads/video.mp4 8:1 1:2 8:2 16:4

<s3_url> may be s3://bucket/key, an https S3 object URL
(bucket.s3.region.amazonaws.com/key or s3.region.amazonaws.com/bucket/key),
or a bare key in AWS_BUCKET_NAME. With no configs, the RAFT_BATCH_SIZE /
GPU_CHUNK_CONCURRENCY values from the environment are tested. The baseline is
always 1:1, i.e. the exact pre-change behaviour.
"""

import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import unquote, urlparse

os.environ.setdefault("SERVICE_NAME", "video_ai_service")
os.environ.setdefault("SQS_QUEUE_URL", "unused")
os.environ.setdefault("DB_HOST", "unused")
os.environ.setdefault("DB_NAME", "unused")
os.environ.setdefault("DB_USER", "unused")
os.environ.setdefault("DB_PASSWORD", "unused")

import boto3
import torch

from config.project_config import (
    AWS_ACCESS_KEY_ID,
    AWS_BUCKET_NAME,
    AWS_REGION,
    AWS_SECRET_ACCESS_KEY,
    DEVICE,
    GPU_CHUNK_CONCURRENCY,
    RAFT_BATCH_SIZE,
    THRESHOLD,
)
from helpers.video_helper import split_video_into_chunks
from ml_models.video import load_models
from repositories.validation_tool.validate import validate_video


def parse_s3_url(url: str) -> tuple[str, str]:
    """Return (bucket, key) for s3://, virtual-hosted/path-style https URLs, or a bare key."""
    p = urlparse(url)
    if p.scheme == "s3":
        return p.netloc, unquote(p.path.lstrip("/"))
    if p.scheme in ("http", "https"):
        host, path = p.netloc, unquote(p.path.lstrip("/"))
        if ".s3." in host or host.endswith(".s3.amazonaws.com"):
            return host.split(".s3", 1)[0], path          # bucket.s3.<region>.amazonaws.com/key
        if host.startswith("s3.") or host.startswith("s3-"):
            bucket, _, key = path.partition("/")          # s3.<region>.amazonaws.com/bucket/key
            return bucket, key
        raise ValueError(f"Not an S3 URL: {url}")
    return AWS_BUCKET_NAME, url.lstrip("/")


def download(url: str, dest_dir: str) -> str:
    bucket, key = parse_s3_url(url)
    dest = os.path.join(dest_dir, "source" + (os.path.splitext(key)[1] or ".mp4"))
    print(f"[*] Downloading s3://{bucket}/{key}")
    boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    ).download_file(bucket, key, dest)
    return dest


_CLK_TCK = os.sysconf("SC_CLK_TCK")


def _proc_ticks(path):
    # utime + stime: fields 14/15 of /proc/.../stat, counted after the ")" of the comm field
    fields = open(path).read().rsplit(")", 1)[1].split()
    return int(fields[11]) + int(fields[12])


def _thread_ticks():
    out = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            out[tid] = _proc_ticks(f"/proc/self/task/{tid}/stat")
        except (OSError, IndexError):
            pass
    return out


def _host_cpu():
    vals = list(map(int, open("/proc/stat").readline().split()[1:]))
    idle = vals[3] + vals[4]  # idle + iowait
    return sum(vals) - idle, sum(vals)


def _gpu_sample():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        ).stdout.splitlines()[0]
        util, mem = (float(x) for x in out.split(","))
        return util, mem
    except Exception:
        return None


def available_cores():
    cores = len(os.sched_getaffinity(0))
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return min(cores, float(quota) / float(period))
    except (OSError, ValueError):
        pass
    return cores


class UtilSampler:
    """Samples GPU util (nvidia-smi, whole GPU -- includes other containers),
    this process's CPU use, its busiest thread, and host CPU, every `interval` s."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self._stop = threading.Event()
        self.gpu, self.gpu_mem, self.proc, self.thread_max, self.host = [], [], [], [], []

    def _loop(self):
        prev_t, prev_p = time.perf_counter(), _proc_ticks("/proc/self/stat")
        prev_th, prev_h = _thread_ticks(), _host_cpu()
        while not self._stop.wait(self.interval):
            now, p, th, h = time.perf_counter(), _proc_ticks("/proc/self/stat"), _thread_ticks(), _host_cpu()
            dt_ticks = (now - prev_t) * _CLK_TCK
            self.proc.append(100.0 * (p - prev_p) / dt_ticks)
            deltas = [th[t] - prev_th[t] for t in th if t in prev_th]
            self.thread_max.append(100.0 * max(deltas, default=0) / dt_ticks)
            if h[1] > prev_h[1]:
                self.host.append(100.0 * (h[0] - prev_h[0]) / (h[1] - prev_h[1]))
            g = _gpu_sample()
            if g:
                self.gpu.append(g[0])
                self.gpu_mem.append(g[1])
            prev_t, prev_p, prev_th, prev_h = now, p, th, h

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def summary(self):
        avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        return {
            "gpu": avg(self.gpu),
            "gpu_mem": max(self.gpu_mem, default=float("nan")),
            "proc": avg(self.proc),
            "thread_max": avg(self.thread_max),
            "host": avg(self.host),
        }


def diagnose(u):
    if u["gpu"] >= 85:
        return "GPU-bound"
    if u["thread_max"] >= 85:
        return "CPU: one thread pinned (Python/GIL-bound)"
    if u["host"] >= 85:
        return "CPU: all cores busy"
    return "neither saturated (waiting on sync/IO)"


def run(chunks, raft, fused, batch_size, concurrency):
    def one(c):
        try:
            return validate_video(c, raft, fused, DEVICE, THRESHOLD, raft_batch_size=batch_size)
        except Exception as e:  # e.g. CUDA OOM -- in the worker this would silently score 0.0
            return {"probability": float("nan"), "prediction": "ERROR", "error": str(e)}

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    with UtilSampler() as sampler:
        t0 = time.perf_counter()
        if concurrency <= 1:
            results = [one(c) for c in chunks]
        else:
            with ThreadPoolExecutor(max_workers=min(len(chunks), concurrency)) as ex:
                results = list(ex.map(one, chunks))
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if DEVICE == "cuda" else 0.0
    return elapsed, results, peak_mb, sampler.summary()


def parse_configs(args):
    if not args:
        return [(RAFT_BATCH_SIZE, GPU_CHUNK_CONCURRENCY)]
    configs = []
    for a in args:
        b, _, c = a.partition(":")
        configs.append((int(b), int(c or 1)))
    return configs


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    url = sys.argv[1]
    configs = parse_configs(sys.argv[2:])
    if DEVICE != "cuda":
        print("[WARN] No CUDA device: batching/concurrency settings only affect the GPU path.")

    raft, fused, _ = load_models()

    with tempfile.TemporaryDirectory() as work_dir:
        source = download(url, work_dir)
        chunks = split_video_into_chunks(source, os.path.join(work_dir, "chunks"), 5)
        print(f"[*] {len(chunks)} chunks")

        # Warm-up so CUDA init / cuDNN autotuning doesn't skew the first timed run
        for b, _ in [(1, 1)] + configs:
            validate_video(chunks[0], raft, fused, DEVICE, THRESHOLD, raft_batch_size=b)

        t_base, base, mem_base, u_base = run(chunks, raft, fused, 1, 1)
        rows = [("1:1 (baseline)", t_base, mem_base, 0.0, 0, 0, u_base)]
        for b, c in configs:
            t, res, mem, u = run(chunks, raft, fused, b, c)
            errors = sum(r["prediction"] == "ERROR" for r in res)
            ok = [(x, y) for x, y in zip(base, res) if y["prediction"] != "ERROR"]
            max_diff = max((abs(x["probability"] - y["probability"]) for x, y in ok), default=0.0)
            flips = sum(x["prediction"] != y["prediction"] for x, y in ok)
            rows.append((f"{b}:{c}", t, mem, max_diff, flips, errors, u))
            for ch, x, y in zip(chunks, base, res):
                if y["prediction"] == "ERROR":
                    print(f"[ERROR] {b}:{c} {os.path.basename(ch)}: {y['error']}")

    print()
    print(f"{'batch:conc':<16}{'time (s)':>10}{'speedup':>9}{'peak MB':>9}"
          f"{'max prob diff':>15}{'flips':>7}{'errors':>8}")
    for name, t, mem, d, f, e, _ in rows:
        print(f"{name:<16}{t:>10.2f}{t_base / t:>8.2f}x{mem:>9.0f}{d:>15.2e}{f:>7}{e:>8}")

    print()
    print(f"Utilization (averages over each run; {available_cores():g} CPU cores available to this container)")
    print(f"{'batch:conc':<16}{'GPU %':>7}{'GPU mem MB':>12}{'proc CPU %':>12}"
          f"{'busiest thr %':>15}{'host CPU %':>12}  bottleneck")
    for name, *_, u in rows:
        print(f"{name:<16}{u['gpu']:>7.0f}{u['gpu_mem']:>12.0f}{u['proc']:>12.0f}"
              f"{u['thread_max']:>15.0f}{u['host']:>12.0f}  {diagnose(u)}")
    print()
    print("GPU % / GPU mem: whole GPU via nvidia-smi (includes other containers on it).")
    print("proc CPU %: this process, 100 = one full core. busiest thr %: its single busiest thread;")
    print("~100 there with low GPU % means Python/GIL is the limit, so more threads won't help.")


if __name__ == "__main__":
    main()
