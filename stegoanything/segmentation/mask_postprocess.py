"""Mask post-processing recipes and WAM area gating (Phase 2A)."""

from __future__ import annotations

from typing import Any, Literal

import cv2
import numpy as np
from scipy import ndimage as ndi

PostprocessRecipe = Literal["R0", "R1", "R2", "R3", "R4", "R5"]
AreaGatePolicy = Literal["reject", "dilate_to_threshold", "bbox_expand", "union_same_class"]


def _as_bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=bool)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    binm = _as_bool(mask)
    if not binm.any():
        return binm
    labeled, n = ndi.label(binm)
    if n <= 1:
        return binm
    sizes = ndi.sum(binm, labeled, index=range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    return labeled == keep


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    binm = _as_bool(mask)
    if not binm.any():
        return binm
    filled = ndi.binary_fill_holes(binm)
    return _as_bool(filled)


def _morph_close_5x5(mask: np.ndarray) -> np.ndarray:
    binm = _as_bool(mask).astype(np.uint8)
    kernel = np.ones((5, 5), dtype=np.uint8)
    closed = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, kernel)
    return closed.astype(bool)


def _erode_pixels(mask: np.ndarray, pixels: int) -> np.ndarray:
    binm = _as_bool(mask)
    if pixels <= 0 or not binm.any():
        return binm
    structure = np.ones((3, 3), dtype=bool)
    eroded = binm.copy()
    for _ in range(pixels):
        eroded = ndi.binary_erosion(eroded, structure=structure)
        if not eroded.any():
            break
    return eroded


def _erode_radius(mask: np.ndarray) -> int:
    h, w = mask.shape[:2]
    return max(1, min(h, w) // 200)


def apply_recipe(mask: np.ndarray, recipe: PostprocessRecipe) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a named post-processing recipe to a boolean or {0,1}/uint8 mask."""
    original_area = int(_as_bool(mask).sum())
    out = _as_bool(mask)

    if recipe == "R0":
        pass
    elif recipe == "R1":
        out = _largest_component(out)
    elif recipe == "R2":
        out = _fill_holes(out)
    elif recipe == "R3":
        # Spec R3: largest connected component first, then fill holes.
        out = _fill_holes(_largest_component(out))
    elif recipe == "R4":
        out = _morph_close_5x5(out)
    elif recipe == "R5":
        radius = _erode_radius(out)
        out = _erode_pixels(out, radius)
    else:
        raise ValueError(f"Unknown postprocess recipe: {recipe!r}")

    meta = {
        "recipe": recipe,
        "original_area": original_area,
        "processed_area": int(out.sum()),
        "erode_pixels": _erode_radius(mask) if recipe == "R5" else None,
    }
    return out, meta


def apply_wam_area_gate(
    mask: np.ndarray,
    min_area_ratio: float = 0.20,
    policy: AreaGatePolicy = "reject",
    other_masks: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """
    Enforce a minimum mask area ratio for WAM embedding compatibility.

    Return dict keys:
    - ``passed`` (bool): whether the result meets ``min_area_ratio``
    - ``policy`` (str): gate policy applied
    - ``min_area_ratio`` (float): threshold
    - ``original_area_ratio`` (float): area ratio before gating
    - ``area_ratio`` (float): area ratio after gating
    - ``mask`` (np.ndarray bool HxW): possibly modified mask
    - ``action`` (str): human-readable action taken
    - ``rejected`` (bool): True when policy rejects an undersized mask
    - ``rejection_reason`` (str | None): set when rejected
    """
    binm = _as_bool(mask)
    h, w = binm.shape[:2]
    image_area = float(h * w)
    original_ratio = float(binm.sum()) / image_area if image_area > 0 else 0.0

    result_mask = binm.copy()
    action = "unchanged"
    rejected = False
    rejection_reason: str | None = None

    if original_ratio >= min_area_ratio:
        return {
            "passed": True,
            "policy": policy,
            "min_area_ratio": float(min_area_ratio),
            "original_area_ratio": original_ratio,
            "area_ratio": original_ratio,
            "mask": result_mask,
            "action": "already_above_threshold",
            "rejected": False,
            "rejection_reason": None,
        }

    if policy == "reject":
        rejected = True
        rejection_reason = (
            f"mask area ratio {original_ratio:.4f} < min_area_ratio {min_area_ratio:.4f}"
        )
        action = "reject_undersized"
    elif policy == "dilate_to_threshold":
        target_pixels = int(np.ceil(min_area_ratio * image_area))
        structure = np.ones((3, 3), dtype=bool)
        dilated = result_mask.copy()
        max_iters = max(h, w)
        for _ in range(max_iters):
            if int(dilated.sum()) >= target_pixels:
                break
            dilated = ndi.binary_dilation(dilated, structure=structure)
        result_mask = _largest_component(dilated)
        action = "dilate_to_threshold"
    elif policy == "bbox_expand":
        ys, xs = np.where(result_mask)
        if ys.size == 0:
            rejected = True
            rejection_reason = "empty mask cannot bbox_expand"
            action = "reject_empty_mask"
        else:
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            bh, bw = y1 - y0, x1 - x0
            cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
            scale = float(np.sqrt(min_area_ratio / max(original_ratio, 1e-12)))
            scale = max(1.0, scale)
            nh = int(np.clip(round(bh * scale), 1, h))
            nw = int(np.clip(round(bw * scale), 1, w))
            ny0 = int(np.clip(round(cy - nh / 2), 0, h - nh))
            nx0 = int(np.clip(round(cx - nw / 2), 0, w - nw))
            expanded = np.zeros_like(result_mask)
            expanded[ny0 : ny0 + nh, nx0 : nx0 + nw] = True
            result_mask = expanded
            action = "bbox_expand"
    elif policy == "union_same_class":
        if not other_masks:
            rejected = True
            rejection_reason = "union_same_class requires other_masks"
            action = "reject_no_union_candidates"
        else:
            union = result_mask.copy()
            for other in other_masks:
                union = np.logical_or(union, _as_bool(other))
            result_mask = union
            action = "union_same_class"
    else:
        raise ValueError(f"Unknown area gate policy: {policy!r}")

    area_ratio = float(result_mask.sum()) / image_area if image_area > 0 else 0.0
    passed = area_ratio >= min_area_ratio and not rejected

    if not passed and not rejected:
        rejection_reason = (
            f"policy {policy!r} could not raise area ratio to {min_area_ratio:.4f} "
            f"(got {area_ratio:.4f})"
        )
        rejected = policy != "reject"

    return {
        "passed": passed,
        "policy": policy,
        "min_area_ratio": float(min_area_ratio),
        "original_area_ratio": original_ratio,
        "area_ratio": area_ratio,
        "mask": result_mask,
        "action": action,
        "rejected": rejected or (policy == "reject" and not passed),
        "rejection_reason": rejection_reason,
    }
