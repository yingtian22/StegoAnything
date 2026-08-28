"""Explicit HiNet uint8 <-> [-1, 1] conversion (no silent Normalize)."""
from __future__ import annotations

import numpy as np
import torch


def uint8_to_hinet_tensor(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert uint8 image [0,255] to float32 CHW tensor in [-1, 1].

    Formula: x_model = x_uint8 / 127.5 - 1
    Accepts HWC or CHW uint8 (or float arrays still in [0,255] scale).
    """
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"expected 3D image, got shape {arr.shape}")
    # Detect CHW vs HWC
    if arr.shape[0] in (1, 3, 4) and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
        chw = arr
    else:
        chw = np.transpose(arr, (2, 0, 1))
    if float(np.max(chw)) <= 1.0 + 1e-6 and float(np.min(chw)) >= -1.0 - 1e-6 and arr.dtype != np.uint8:
        raise ValueError(
            "uint8_to_hinet_tensor expects uint8-scale [0,255]; refusing already-normalized input"
        )
    t = torch.from_numpy(chw.astype(np.float32) / 127.5 - 1.0)
    return t.float()


def hinet_tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """x_uint8 = clip((x_model+1)/2 * 255, 0, 255). CHW/BCHW -> HWC / stacked HWC."""
    t = tensor.detach().cpu().float()
    if t.ndim == 4:
        return np.stack([hinet_tensor_to_uint8(t[i]) for i in range(t.shape[0])], axis=0)
    if t.ndim != 3:
        raise ValueError(f"expected CHW or BCHW, got {tuple(t.shape)}")
    hwc = ((t + 1.0) * 0.5 * 255.0).clamp(0, 255).permute(1, 2, 0).numpy()
    return np.rint(hwc).astype(np.uint8)


def mask_uint8_to_binary_tensor(mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert mask image to float32 [1,H,W] with values in {0.0, 1.0}. No RGB norm."""
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] in (1, 3, 4) else arr[0]
    if arr.ndim != 2:
        raise ValueError(f"expected 2D mask, got {arr.shape}")
    if arr.dtype == np.uint8 or float(arr.max()) > 1.0:
        binary = (arr > 127).astype(np.float32)
    else:
        binary = (arr > 0.5).astype(np.float32)
    uniq = set(np.unique(binary).tolist())
    if not uniq.issubset({0.0, 1.0}):
        raise ValueError(f"mask not binary after threshold, unique={sorted(uniq)}")
    return torch.from_numpy(binary).unsqueeze(0).float()
