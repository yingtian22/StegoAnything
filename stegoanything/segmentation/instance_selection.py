"""Instance selection strategies for multi-detection segmentation outputs."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

SelectionStrategy = Literal[
    "top_score",
    "largest_mask",
    "largest_box",
    "oracle_matched",
    "union_all",
]


def _mask_area(mask: np.ndarray) -> int:
    return int(np.asarray(mask, dtype=bool).sum())


def _box_area(box_xyxy: np.ndarray) -> float:
    x1, y1, x2, y2 = box_xyxy
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bin = np.asarray(a, dtype=bool)
    b_bin = np.asarray(b, dtype=bool)
    inter = int(np.logical_and(a_bin, b_bin).sum())
    union = int(np.logical_or(a_bin, b_bin).sum())
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def select_instance(
    strategy: SelectionStrategy,
    boxes_xyxy: list[np.ndarray] | np.ndarray,
    scores: list[float] | np.ndarray,
    phrases: list[str],
    masks: list[np.ndarray] | np.ndarray,
    oracle_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Select one detection or union all masks.

    Returns a metadata dict. Index strategies include ``selected_index``;
    ``union_all`` includes a boolean ``mask`` (H, W).
    """
    n = len(phrases)
    if n == 0:
        return {
            "strategy": strategy,
            "num_candidates": 0,
            "selected_index": None,
            "mask": None,
            "reason": "no_detections",
        }

    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    score_arr = np.asarray(scores, dtype=np.float32)
    mask_arr = np.stack([np.asarray(m, dtype=bool) for m in masks], axis=0)

    if strategy == "top_score":
        idx = int(np.argmax(score_arr))
        return {
            "strategy": strategy,
            "num_candidates": n,
            "selected_index": idx,
            "score": float(score_arr[idx]),
            "phrase": phrases[idx],
            "mask_area": _mask_area(mask_arr[idx]),
            "box_area": _box_area(boxes[idx]),
        }

    if strategy == "largest_mask":
        areas = np.array([_mask_area(m) for m in mask_arr], dtype=np.int64)
        idx = int(np.argmax(areas))
        return {
            "strategy": strategy,
            "num_candidates": n,
            "selected_index": idx,
            "score": float(score_arr[idx]),
            "phrase": phrases[idx],
            "mask_area": int(areas[idx]),
            "box_area": _box_area(boxes[idx]),
        }

    if strategy == "largest_box":
        box_areas = np.array([_box_area(b) for b in boxes], dtype=np.float64)
        idx = int(np.argmax(box_areas))
        return {
            "strategy": strategy,
            "num_candidates": n,
            "selected_index": idx,
            "score": float(score_arr[idx]),
            "phrase": phrases[idx],
            "mask_area": _mask_area(mask_arr[idx]),
            "box_area": float(box_areas[idx]),
        }

    if strategy == "oracle_matched":
        if oracle_mask is None:
            raise ValueError("oracle_matched requires oracle_mask.")
        ious = np.array([_mask_iou(m, oracle_mask) for m in mask_arr], dtype=np.float64)
        idx = int(np.argmax(ious))
        return {
            "strategy": strategy,
            "num_candidates": n,
            "selected_index": idx,
            "score": float(score_arr[idx]),
            "phrase": phrases[idx],
            "mask_area": _mask_area(mask_arr[idx]),
            "box_area": _box_area(boxes[idx]),
            "oracle_iou": float(ious[idx]),
            "oracle_ious": ious.tolist(),
        }

    if strategy == "union_all":
        union_mask = np.any(mask_arr, axis=0)
        return {
            "strategy": strategy,
            "num_candidates": n,
            "selected_index": None,
            "mask": union_mask,
            "mask_area": _mask_area(union_mask),
            "union_count": n,
            "phrases": list(phrases),
            "mean_score": float(score_arr.mean()) if n else 0.0,
        }

    raise ValueError(f"Unknown selection strategy: {strategy!r}")
