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


def _probe_duration(input_path):
    """Video-stream duration, not the container-level `format=duration` tag:
    a stream-copy trim (or a corrupt secondary stream) can leave the
    container tag reporting a duration far longer than the video actually
    plays, which silently over-chunks past the real content."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", input_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    try:
        return float(result.stdout)
    except ValueError:
        pass

    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", input_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    return float(result.stdout)


def split_video_into_chunks(input_path, output_dir, chunk_length=5):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Getting video duration for: {input_path}")
    duration = _probe_duration(input_path)
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

    chunks.sort(key=lambda p: chunk_bounds(p)[0])
    print(f"[INFO] Total chunks created: {len(chunks)}")
    return chunks


def chunk_bounds(chunk_path):
    """Real (start, end) seconds encoded in a `chunk_{start}-{end}.mp4` name,
    rounded to the nearest second -- the last chunk is usually shorter than
    `chunk_length`, so this must be read back rather than assumed uniform."""
    name = os.path.splitext(os.path.basename(chunk_path))[0]
    start_str, end_str = name.split("chunk_")[1].split("-")
    return round(float(start_str)), round(float(end_str))


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
