import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from ml_models import video
from repositories.validation_tool import validate_video
from config.project_config import DEVICE, THRESHOLD

def create_chunk(input_path, start, end, chunk_path):
    print(f"[INFO] Creating chunk: {chunk_path} (from {start}s to {end}s)")
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", input_path,
        "-c", "copy",
        "-loglevel", "error",
        chunk_path
    ]
    subprocess.run(cmd, check=True)
    print(f"[INFO] Chunk created: {chunk_path}")
    return chunk_path


def split_video_into_chunks(input_path, output_dir, chunk_length=5):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[INFO] Getting video duration for: {input_path}")
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", input_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    duration = float(result.stdout)
    print(f"[INFO] Video duration: {duration:.2f} seconds")

    chunk_infos = []
    for start in range(0, int(duration), chunk_length):
        end = min(start + chunk_length, duration)
        chunk_path = os.path.join(output_dir, f"chunk_{start}-{end}.mp4")
        chunk_infos.append((input_path, start, end, chunk_path))

    chunks = []

    max_workers = min(len(chunk_infos), os.cpu_count())
    print(f"[INFO] Starting parallel chunk creation with {max_workers} workers")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(create_chunk, *info) for info in chunk_infos]
        for future in as_completed(futures):
            chunks.append(future.result())

    chunks.sort(key=lambda p: float(os.path.splitext(os.path.basename(p))[0].split("chunk_")[1].split("-")[0]))
    print(f"[INFO] Total chunks created: {len(chunks)}")
    return chunks

def infer_chunk(chunk_path):
    try:
        print(f"[INFO] Starting inference for chunk: {chunk_path}")
        result = validate_video(chunk_path, video.raft_model, video.fused_model, DEVICE, THRESHOLD)
        print(f"[INFO] Completed inference for chunk: {chunk_path}, probability: {result['probability']:.4f}")
        return {
            "chunk": os.path.basename(chunk_path),
            "path": chunk_path,
            "result": result
        }
    except Exception as e:
        print(f"[ERROR] Chunk inference failed: {chunk_path}, error: {e}")
        return {
            "chunk": os.path.basename(chunk_path),
            "path": chunk_path,
            "result": {"probability": 0.0, "error": str(e)}
        }
