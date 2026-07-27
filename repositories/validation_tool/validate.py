#!/usr/bin/env python3
"""
validate.py - Single video deepfake detection

Usage:
    Configure the global variables below and run:
    python validate.py

Label Convention:
    0 = Real
    1 = Fake
"""
import os
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch

from repositories.validation_tool.raft.raft import RAFT
from repositories.validation_tool.raft.utils.utils import InputPadder
from repositories.validation_tool.utils.augmentations import ValidationTransform


# ============================================================
# RAFT Utilities
# ============================================================

class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, value):
        self[key] = value


def decode_video_cv2(video_path: str, max_frames: int = 96) -> List[np.ndarray]:
    """Decode video to list of RGB frames"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    while len(frames) < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    
    # Loop if too short
    if len(frames) < max_frames and len(frames) > 0:
        while len(frames) < max_frames:
            frames.append(frames[-1].copy())
    
    return frames[:max_frames]


def resize_min_side(frame: np.ndarray, min_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if min(h, w) >= min_side:
        return frame
    scale = float(min_side) / float(min(h, w))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)


def crop_center(frame: np.ndarray, crop_size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h < crop_size or w < crop_size:
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        frame = cv2.copyMakeBorder(
            frame, pad_h // 2, pad_h - pad_h // 2,
            pad_w // 2, pad_w - pad_w // 2,
            borderType=cv2.BORDER_REFLECT_101
        )
        h, w = frame.shape[:2]
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    return frame[y0:y0 + crop_size, x0:x0 + crop_size].copy()


@torch.no_grad()
def compute_flow_pair(model: RAFT, device: str, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """Compute optical flow between two frames. Returns (H, W, 2) float32."""
    im1 = torch.from_numpy(frame1).permute(2, 0, 1).unsqueeze(0).float().to(device)
    im2 = torch.from_numpy(frame2).permute(2, 0, 1).unsqueeze(0).float().to(device)

    padder = InputPadder(im1.shape)
    im1, im2 = padder.pad(im1, im2)

    _, flow_up = model(im1, im2, iters=12, test_mode=True)
    flow_up = padder.unpad(flow_up)
    return flow_up[0].permute(1, 2, 0).cpu().numpy()


def compute_all_flows(raft_model: RAFT, device: str, frames: List[np.ndarray]) -> np.ndarray:
    """Compute optical flows for all frame pairs. Returns (T-1, 2, H, W)"""
    # Preprocess frames for RAFT (resize + crop to 448x448)
    proc = [crop_center(resize_min_side(f, 448), 448) for f in frames]

    def compute_single_flow(i):
        flow = compute_flow_pair(raft_model, device, proc[i], proc[i + 1])
        # Resize to 224x224 for model input
        flow_resized = np.stack([
            cv2.resize(flow[..., 0], (224, 224), interpolation=cv2.INTER_LINEAR),
            cv2.resize(flow[..., 1], (224, 224), interpolation=cv2.INTER_LINEAR),
        ], axis=-1)
        # Normalize: clip to ±20 and scale to [-1, 1]
        flow_norm = np.clip(flow_resized, -20.0, 20.0) / 20.0
        return flow_norm.astype(np.float32)

    # GPU: sequential is faster — CUDA serializes ops and threads only add overhead.
    # CPU: thread pool helps since PyTorch releases the GIL during computation.
    if device == "cuda":
        flows = [compute_single_flow(i) for i in range(len(proc) - 1)]
    else:
        max_workers = min(len(proc) - 1, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            flows = list(executor.map(compute_single_flow, range(len(proc) - 1)))

    flows = np.stack(flows, axis=0)  # (T-1, H, W, 2)
    return flows.transpose(0, 3, 1, 2)  # (T-1, 2, H, W)


def load_raft_model(ckpt_path: str, device: str) -> RAFT:
    """Load RAFT model"""
    args = AttrDict({"small": False, "mixed_precision": False, "dropout": 0.0, 
                     "alternate_corr": False, "corr_levels": 4, "corr_radius": 4})
    model = RAFT(args).to(device)
    raw = torch.load(ckpt_path, map_location="cpu")
    state = raw.get("state_dict", raw)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ============================================================
# Main
# ============================================================

def validate_video(video_path, raft_model, fused_model, device, threshold):
    """Run deepfake detection on a single video"""
    
    print(f"[*] Video: {video_path}")
    
    # Decode frames once
    frames = decode_video_cv2(video_path, max_frames=96)
    frames_flow = frames
    frames_video = frames[:8] if len(frames) >= 8 else frames + [frames[-1]] * (8 - len(frames))
    print(f"[*] Decoded {len(frames_flow)} frames for RAFT, {len(frames_video)} for DeMamba")
    
    # Compute optical flows
    print(f"[*] Computing optical flows...")
    flows = compute_all_flows(raft_model, device, frames_flow)
    print(f"[*] Flows shape: {flows.shape}")
    
    # Prepare video frames for DeMamba
    cfg = {
        "augmentation": {
            "resize": {"height": 224, "width": 224},
            "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "max_pixel_value": 255.0},
            "horizontal_flip": {"prob": 0.0},
            "jpeg_compression": {"prob": 0.0, "quality_range": [50, 100]},
            "gaussian_noise": {"prob": 0.0, "var_limit": [10.0, 50.0]},
            "gaussian_blur": {"prob": 0.0, "kernel_range": [3, 5]},
            "grayscale": {"prob": 0.0},
        }
    }
    video_tensor = ValidationTransform(cfg)(frames_video)
    
    flows_t = torch.from_numpy(flows).unsqueeze(0).to(device)
    video_t = torch.from_numpy(video_tensor).unsqueeze(0).to(device)
    
    print(f"[*] Running inference...")
    with torch.no_grad():
        prob = fused_model(flows_t, video_t)
    
    prob_val = float(prob.cpu().item())
    prediction = "Fake" if prob_val >= threshold else "Real"
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    return {"probability": prob_val, "prediction": prediction, "threshold": threshold}