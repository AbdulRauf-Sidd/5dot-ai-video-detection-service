"""
FusedHeadModel for validation - combines OpticalFlowBranch + DeMamba
"""
import torch
import torch.nn as nn

from .optical_flow_model import OpticalFlowBranch
from .demamba.DeMamba import XCLIP_DeMamba

class FusedHeadModel(nn.Module):
    def __init__(self, pretrained_xclip_encoder: XCLIP_DeMamba):
        super().__init__()

        self.optical = OpticalFlowBranch(pretrained=False, backbone="resnet50")
        self.demamba = pretrained_xclip_encoder

        # Freeze base models for inference
        for p in self.optical.parameters():
            p.requires_grad = False
        for p in self.demamba.parameters():
            p.requires_grad = False

        # Fusion head: 2 logits -> 1 probability
        self.head = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )
        self.out_act = nn.Sigmoid()

    def forward(self, flows: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flows: (B, T, 2, H, W) optical flow tensor
            frames: (B, T, 3, H, W) video frames tensor
        Returns:
            prob: (B,) probability of being fake (0=real, 1=fake)
        """
        with torch.no_grad():
            flow_logit = self.optical(flows).view(-1, 1)
            demamba_logit = self.demamba(frames).view(-1, 1)

        fused = torch.cat([flow_logit, demamba_logit], dim=1)
        head_logit = self.head(fused)
        prob = self.out_act(head_logit).squeeze(-1)
        return prob

    def load_checkpoint(self, path: str):
        """Load fused model checkpoint"""
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt["model"]  # Training script saves in "model" key
        self.load_state_dict(state)
        print(f"[+] Loaded checkpoint from {path}")
