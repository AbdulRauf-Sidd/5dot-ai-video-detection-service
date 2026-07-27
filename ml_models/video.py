import threading
from transformers import XCLIPVisionModel
from repositories.validation_tool.models.demamba.DeMamba import XCLIP_DeMamba
from repositories.validation_tool.models.fused_model import FusedHeadModel
from repositories.validation_tool.validate import load_raft_model
from config.project_config import CHECKPOINT, DEVICE, RAFT_CHECKPOINT, XCLIP_LOCAL_DIR

raft_model = None
fused_model = None
xclip_demamba = None

_load_lock = threading.Lock()

def load_models():
    global raft_model, fused_model, xclip_demamba

    with _load_lock:
        # --------------------------
        # RAFT
        # --------------------------
        if raft_model is None:
            print("[*] Loading RAFT model...")
            raft_model = load_raft_model(RAFT_CHECKPOINT, DEVICE)
            print("[+] RAFT model loaded")

        # --------------------------
        # XCLIP Vision
        # --------------------------
        if xclip_demamba is None:
            print("[*] Loading XCLIPVisionModel + DeMamba...")
            # Baked into the AMI at XCLIP_LOCAL_DIR -- loaded offline, no Hugging Face Hub call.
            xclip_encoder = XCLIPVisionModel.from_pretrained(
                XCLIP_LOCAL_DIR, local_files_only=True
            ).to(DEVICE)
            xclip_encoder.eval()
            xclip_demamba = XCLIP_DeMamba(pretrained_encoder=xclip_encoder).to(DEVICE)
            xclip_demamba.eval()
            print("[+] XCLIPVisionModel + DeMamba loaded")

        # --------------------------
        # FusedHeadModel
        # --------------------------
        if fused_model is None:
            print("[*] Loading FusedHeadModel...")
            fused_model = FusedHeadModel(pretrained_xclip_encoder=xclip_demamba).to(DEVICE)
            fused_model.load_checkpoint(CHECKPOINT)
            fused_model.eval()
            print("[+] FusedHeadModel loaded")

    return raft_model, fused_model, xclip_demamba