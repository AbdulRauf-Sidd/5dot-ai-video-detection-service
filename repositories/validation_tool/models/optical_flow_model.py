import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet50


class OpticalFlowBranch(nn.Module):
    """
    Input: flows (B, T, 2, H, W)  raw (dx,dy)
    Strategy: run backbone per-frame, aggregate video-level logit by mean.
    Output: video logits (B,)
    """

    def __init__(self, pretrained=True, backbone="resnet50"):
        super().__init__()

        if backbone != "resnet50":
            raise ValueError("Only resnet50 supported for now")

        # resnet50 returns logits of num_classes, we want features -> replace fc
        self.backbone = resnet50(pretrained=pretrained, num_classes=1000, in_channels=2)

        # replace classifier to output 1 logit
        in_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_dim, 1)

    def forward(self, flows):
        """
        flows: (B, T, 2, H, W)
        return: (B,) logits
        """
        B, T, C, H, W = flows.shape
        x = flows.view(B * T, C, H, W)
        logits = self.backbone(x)            # (B*T, 1)
        logits = logits.view(B, T)           # (B, T)
        video_logit = logits.mean(dim=1)     # (B,)
        return video_logit
