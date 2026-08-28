"""Foreground-region image metrics with adaptive SSIM window for tiny masks.

Extracted from Phase 7B-1A (scripts/phase7b1a_run_lvis_train_pilot.py) for reuse
in cover generation and downstream HiNet compatibility audits.
"""
from __future__ import annotations

import math

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr_u8(a: np.ndarray, b: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(a, b, data_range=255))


def ssim_u8(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM with adaptive win_size for tiny crops (OUT_OF_PROTOCOL masks)."""
    aa = np.asarray(a)
    bb = np.asarray(b)
    if aa.ndim == 1 or bb.ndim == 1 or min(aa.shape[:2]) < 3:
        return float("nan")
    win = min(7, int(aa.shape[0]), int(aa.shape[1]))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return float("nan")
    return float(structural_similarity(aa, bb, channel_axis=2, data_range=255, win_size=win))


def psnr_masked(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if m.ndim == 3:
        m = m[..., 0]
    if not m.any():
        return float("nan")
    diff = a[m].astype(np.float64) - b[m].astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * math.log10((255.0**2) / mse))


def ssim_masked(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if m.ndim == 3:
        m = m[..., 0]
    if not m.any():
        return float("nan")
    ys, xs = np.where(m)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h, w = a.shape[:2]
    ch, cw = y1 - y0, x1 - x0
    if ch < 7:
        pad = 7 - ch
        y0 = max(0, y0 - pad // 2)
        y1 = min(h, y0 + max(7, ch))
        y0 = max(0, y1 - max(7, ch))
    if cw < 7:
        pad = 7 - cw
        x0 = max(0, x0 - pad // 2)
        x1 = min(w, x0 + max(7, cw))
        x0 = max(0, x1 - max(7, cw))
    return ssim_u8(a[y0:y1, x0:x1], b[y0:y1, x0:x1])


def mae_masked(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if m.ndim == 3:
        m = m[..., 0]
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(a[m].astype(np.float32) - b[m].astype(np.float32))))
