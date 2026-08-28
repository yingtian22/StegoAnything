"""Local compositing of BrushNet outputs with original images."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _as_rgb(x: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(x, Image.Image):
        return np.array(x.convert("RGB"))
    arr = np.asarray(x)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr[..., :3].astype(np.uint8)


def _as_bool_mask(mask: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(mask, Image.Image):
        m = np.array(mask.convert("L"))
    else:
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m.sum(axis=-1)
    return m > 127


def hard_composite(original: Image.Image | np.ndarray, generated: Image.Image | np.ndarray, mask: Image.Image | np.ndarray) -> Image.Image:
    """I_h = (1-M)*I + M*I_raw."""
    I = _as_rgb(original).astype(np.float32)
    G = _as_rgb(generated).astype(np.float32)
    M = _as_bool_mask(mask).astype(np.float32)[..., None]
    if I.shape[:2] != G.shape[:2] or I.shape[:2] != M.shape[:2]:
        raise ValueError(f"shape mismatch I{I.shape} G{G.shape} M{M.shape}")
    out = (1.0 - M) * I + M * G
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def feather_composite(
    original: Image.Image | np.ndarray,
    generated: Image.Image | np.ndarray,
    mask: Image.Image | np.ndarray,
    feather_width: int = 5,
) -> Image.Image:
    """Soft-boundary composite with fixed feather width (pixels)."""
    I = _as_rgb(original).astype(np.float32)
    G = _as_rgb(generated).astype(np.float32)
    M = _as_bool_mask(mask).astype(np.uint8)
    if feather_width <= 0:
        return hard_composite(original, generated, mask)
    k = int(feather_width) * 2 + 1
    soft = cv2.GaussianBlur(M.astype(np.float32), (k, k), 0)
    soft = soft[..., None]
    out = (1.0 - soft) * I + soft * G
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
