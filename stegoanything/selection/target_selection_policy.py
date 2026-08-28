"""Unified deployment-facing text-and-click target selection policy (Phase 6A-1H)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


class NoCandidateError(ValueError):
    """Raised when the candidate set is empty."""


class InvalidClickError(ValueError):
    """Raised when a click is outside the candidate mask / image bounds."""


class InvalidCandidateInputError(ValueError):
    """Raised for inconsistent or invalid candidate arrays."""


@dataclass(frozen=True)
class TargetSelectionResult:
    selected_index: int
    mode: str
    reason: str
    click_xy: Optional[Tuple[int, int]]
    containing_mask_indices: tuple[int, ...]
    containing_box_indices: tuple[int, ...]
    fallback_used: bool


def _as_boxes(candidate_boxes: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    boxes = np.asarray(candidate_boxes, dtype=np.float64)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise InvalidCandidateInputError(f"candidate_boxes must be (N,4), got {boxes.shape}")
    if not np.isfinite(boxes).all():
        raise InvalidCandidateInputError("candidate_boxes contain NaN/Inf")
    # Reject clearly inverted / non-finite geometry after finite check
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        if x2 < x1 or y2 < y1:
            raise InvalidCandidateInputError(f"illegal box at index {i}: {(x1, y1, x2, y2)}")
    return boxes


def _as_scores(candidate_scores: np.ndarray | Sequence[float], n: int) -> np.ndarray:
    scores = np.asarray(candidate_scores, dtype=np.float64).reshape(-1)
    if scores.shape[0] != n:
        raise InvalidCandidateInputError(
            f"score/box count mismatch: scores={scores.shape[0]} boxes={n}"
        )
    if not np.isfinite(scores).all():
        raise InvalidCandidateInputError("candidate_scores contain NaN/Inf")
    return scores


def _as_masks(
    candidate_masks: Optional[np.ndarray],
    n: int,
) -> Optional[np.ndarray]:
    if candidate_masks is None:
        return None
    masks = np.asarray(candidate_masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        raise InvalidCandidateInputError(f"candidate_masks must be (N,H,W), got {masks.shape}")
    if masks.shape[0] != n:
        raise InvalidCandidateInputError(
            f"mask/box count mismatch: masks={masks.shape[0]} boxes={n}"
        )
    if masks.shape[0] > 0 and (masks.shape[1] == 0 or masks.shape[2] == 0):
        raise InvalidCandidateInputError("empty mask spatial dimensions")
    return masks.astype(bool, copy=False)


def _argmax_score_min_index(scores: np.ndarray) -> int:
    """Highest score; ties broken by smallest candidate index."""
    best = float(scores[0])
    best_i = 0
    for i in range(1, len(scores)):
        s = float(scores[i])
        if s > best:
            best = s
            best_i = i
    return best_i


def _argmax_among(indices: Sequence[int], scores: np.ndarray) -> int:
    best_i = int(indices[0])
    best = float(scores[best_i])
    for i in indices[1:]:
        s = float(scores[i])
        if s > best or (s == best and int(i) < best_i):
            best = s
            best_i = int(i)
    return best_i


class TargetSelectionPolicy:
    """
    Unified deployment-facing target selection policy.

    Default:
        highest detector score

    Optional click:
        mask containment
        -> box containment
        -> highest-score fallback
    """

    def select(
        self,
        candidate_boxes: np.ndarray,
        candidate_scores: np.ndarray,
        candidate_masks: Optional[np.ndarray] = None,
        click_xy: Optional[Tuple[int, int]] = None,
        image_hw: Optional[Tuple[int, int]] = None,
    ) -> TargetSelectionResult:
        boxes = _as_boxes(candidate_boxes)
        n = int(boxes.shape[0])
        if n == 0:
            raise NoCandidateError("zero candidates")
        scores = _as_scores(candidate_scores, n)
        masks = _as_masks(candidate_masks, n)

        if click_xy is None:
            idx = _argmax_score_min_index(scores)
            return TargetSelectionResult(
                selected_index=idx,
                mode="text_only",
                reason="highest_detector_score",
                click_xy=None,
                containing_mask_indices=(),
                containing_box_indices=(),
                fallback_used=False,
            )

        click_x, click_y = int(click_xy[0]), int(click_xy[1])

        # Bounds: prefer explicit image_hw; else mask spatial size when available
        if image_hw is not None:
            h, w = int(image_hw[0]), int(image_hw[1])
        elif masks is not None:
            h, w = int(masks.shape[1]), int(masks.shape[2])
        else:
            # Without masks, validate click against union of box extents only if
            # image_hw omitted — still require non-negative and within max box.
            # Spec: click outside image range raises. Without image size, use
            # max box corner as a conservative image bound proxy only when all
            # boxes are valid; prefer callers pass image_hw.
            h = int(np.ceil(float(boxes[:, 3].max()))) + 1
            w = int(np.ceil(float(boxes[:, 2].max()))) + 1

        if click_x < 0 or click_y < 0 or click_x >= w or click_y >= h:
            raise InvalidClickError(
                f"click ({click_x},{click_y}) outside image bounds ({w}x{h})"
            )

        mask_hits: list[int] = []
        mask_unavailable = masks is None
        if masks is not None:
            for i in range(n):
                if bool(masks[i, click_y, click_x]):
                    mask_hits.append(i)

        if len(mask_hits) == 1:
            return TargetSelectionResult(
                selected_index=int(mask_hits[0]),
                mode="click_assisted",
                reason="single_mask_containment",
                click_xy=(click_x, click_y),
                containing_mask_indices=tuple(mask_hits),
                containing_box_indices=(),
                fallback_used=False,
            )
        if len(mask_hits) > 1:
            return TargetSelectionResult(
                selected_index=_argmax_among(mask_hits, scores),
                mode="click_assisted",
                reason="multiple_mask_containment_highest_score",
                click_xy=(click_x, click_y),
                containing_mask_indices=tuple(mask_hits),
                containing_box_indices=(),
                fallback_used=False,
            )

        box_hits: list[int] = []
        for i in range(n):
            x1, y1, x2, y2 = [float(v) for v in boxes[i]]
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                box_hits.append(i)

        if len(box_hits) == 1:
            reason = "single_box_containment"
            if mask_unavailable:
                reason = "mask_unavailable_single_box_containment"
            return TargetSelectionResult(
                selected_index=int(box_hits[0]),
                mode="click_assisted",
                reason=reason,
                click_xy=(click_x, click_y),
                containing_mask_indices=(),
                containing_box_indices=tuple(box_hits),
                fallback_used=False,
            )
        if len(box_hits) > 1:
            reason = "multiple_box_containment_highest_score"
            if mask_unavailable:
                reason = "mask_unavailable_multiple_box_containment_highest_score"
            return TargetSelectionResult(
                selected_index=_argmax_among(box_hits, scores),
                mode="click_assisted",
                reason=reason,
                click_xy=(click_x, click_y),
                containing_mask_indices=(),
                containing_box_indices=tuple(box_hits),
                fallback_used=False,
            )

        reason = "highest_score_fallback"
        if mask_unavailable:
            reason = "mask_unavailable_highest_score_fallback"
        return TargetSelectionResult(
            selected_index=_argmax_score_min_index(scores),
            mode="click_assisted",
            reason=reason,
            click_xy=(click_x, click_y),
            containing_mask_indices=(),
            containing_box_indices=(),
            fallback_used=True,
        )
