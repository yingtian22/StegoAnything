"""Full-canvas hide + zero-z recover (256). No Grounded-SAM / BrushNet required."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from stegoanything.hiding.loader import load_hinet
from stegoanything.metrics.foreground_metrics import psnr_masked, psnr_u8, ssim_u8
from stegoanything.object_hiding.hinet_normalize import hinet_tensor_to_uint8, uint8_to_hinet_tensor


class HideRecoverPipeline:
    def __init__(self, checkpoint: str | Path | None = None, canvas_size: int = 256, device: str = "cuda"):
        self.model, self.ckpt_meta = load_hinet(checkpoint, device=device)
        self.canvas_size = int(canvas_size)
        self.device = next(self.model.parameters()).device

    @staticmethod
    def secret_from_mask(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
        m = mask[..., 0] if mask.ndim == 3 else mask
        binary = (m > 127).astype(np.uint8)
        secret = original.copy()
        secret[binary == 0] = 0
        return secret

    @torch.inference_mode()
    def hide(self, cover: np.ndarray, secret: np.ndarray) -> dict[str, Any]:
        cover_t = uint8_to_hinet_tensor(cover).unsqueeze(0).to(self.device)
        secret_t = uint8_to_hinet_tensor(secret).unsqueeze(0).to(self.device)
        out = self.model(cover_t, secret_t)
        stego = hinet_tensor_to_uint8(out["stego"][0])
        return {
            "stego": stego,
            "latent_z_shape": list(out["latent_z"].shape),
            "true_latent_used": False,
        }

    @torch.inference_mode()
    def recover(self, stego: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        stego_t = uint8_to_hinet_tensor(stego).unsqueeze(0).to(self.device)
        recovered_t = self.model.recover_from_stego(stego_t, None)
        recovered = hinet_tensor_to_uint8(recovered_t[0])
        m = mask[..., 0] if mask.ndim == 3 else mask
        bin_m = (m > 127).astype(np.float32)[..., None]
        restored = (1.0 - bin_m) * stego.astype(np.float32) + bin_m * recovered.astype(np.float32)
        restored_u8 = np.clip(np.rint(restored), 0, 255).astype(np.uint8)
        return {
            "recovered_secret": recovered,
            "restored": restored_u8,
            "recovery": "zero_z",
        }

    def run(self, cover: np.ndarray, original: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        secret = self.secret_from_mask(original, mask)
        hid = self.hide(cover, secret)
        rec = self.recover(hid["stego"], mask)
        metrics = {
            "stego_psnr": psnr_u8(hid["stego"], cover),
            "stego_ssim": ssim_u8(hid["stego"], cover),
            "object_fg_psnr": psnr_masked(rec["recovered_secret"], secret, mask),
            "restored_psnr": psnr_u8(rec["restored"], original),
            "restored_ssim": ssim_u8(rec["restored"], original),
        }
        return {
            "cover": cover,
            "original": original,
            "mask": mask,
            "secret": secret,
            "stego": hid["stego"],
            "recovered_secret": rec["recovered_secret"],
            "restored": rec["restored"],
            "metrics": metrics,
            "checkpoint": self.ckpt_meta,
        }
