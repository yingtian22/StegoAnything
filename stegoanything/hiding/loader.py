"""Load the formal Full-Canvas HiNet-style checkpoint."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from stegoanything.models.hinet_style_hiding_net import HiNetStyleSecretHidingModel

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "hinet_full_canvas_seed0.pt"


def load_hinet(
    checkpoint: str | Path | None = None,
    device: str = "cuda",
) -> tuple[HiNetStyleSecretHidingModel, dict[str, Any]]:
    path = Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT
    if not path.is_file():
        raise FileNotFoundError(
            f"HiNet checkpoint not found: {path}\n"
            "Place the formal weight at checkpoints/hinet_full_canvas_seed0.pt"
        )
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        cfg = dict(ckpt.get("config") or {})
    else:
        state = ckpt.get("model_state", ckpt)
        if isinstance(state, dict) and any(k.startswith("hinet.") for k in state):
            state = {k[len("hinet.") :]: v for k, v in state.items() if k.startswith("hinet.")}
        cfg = dict(ckpt.get("config", {}) or {})
    num_blocks = int(cfg.get("num_blocks", 8))
    hidden = int(cfg.get("hidden_channels", 64))
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = HiNetStyleSecretHidingModel(num_blocks=num_blocks, hidden_channels=hidden).to(dev)
    incompatible = model.load_state_dict(state, strict=True)
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch missing={missing} unexpected={unexpected}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    meta = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "num_blocks": num_blocks,
        "hidden_channels": hidden,
        "device": str(dev),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "recovery": "zero_z",
    }
    return model, meta
