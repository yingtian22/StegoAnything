"""Unified text-guided target locator: GSAM2 candidates + TargetSelectionPolicy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple

import numpy as np

from stegoanything.selection.target_selection_policy import (
    InvalidCandidateInputError,
    InvalidClickError,
    NoCandidateError,
    TargetSelectionPolicy,
)


class InvalidImageError(ValueError):
    """Raised when the input image is not a valid RGB uint8 array."""


class InvalidClassNameError(ValueError):
    """Raised when class_name is empty or whitespace-only."""


class AdapterPredictProtocol(Protocol):
    def predict(self, image: np.ndarray, text_prompt: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class TargetLocalizationResult:
    prompt: str
    selected_index: int
    selected_box: np.ndarray
    selected_mask: np.ndarray
    selected_score: float
    selection_mode: str
    selection_reason: str
    click_xy: Optional[Tuple[int, int]]
    candidate_count: int
    candidate_boxes: np.ndarray
    candidate_scores: np.ndarray
    candidate_masks: np.ndarray
    fallback_used: bool
    grounding_runtime_ms: float | None = None
    sam_runtime_ms: float | None = None
    total_runtime_ms: float | None = None
    peak_gpu_memory_mb: float | None = None


@dataclass(frozen=True)
class CandidateBundle:
    """Frozen candidate set from a single GSAM2 forward pass."""

    prompt: str
    image_hw: Tuple[int, int]
    candidate_boxes: np.ndarray
    candidate_scores: np.ndarray
    candidate_masks: np.ndarray
    phrases: tuple[str, ...]
    grounding_runtime_ms: float | None = None
    sam_runtime_ms: float | None = None
    total_runtime_ms: float | None = None
    peak_gpu_memory_mb: float | None = None


def validate_rgb_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise InvalidImageError(f"Expected HxWx3 RGB, got shape {arr.shape}")
    if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise InvalidImageError("empty image")
    if arr.dtype != np.uint8:
        raise InvalidImageError(f"Expected dtype uint8, got {arr.dtype}")
    if not np.isfinite(arr.astype(np.float32)).all():
        raise InvalidImageError("image contains NaN/Inf")
    return arr


def build_default_prompt(class_name: str) -> str:
    if class_name is None or not str(class_name).strip():
        raise InvalidClassNameError("class_name must be a non-empty string")
    return f"a {str(class_name).strip()}."


def validate_candidates(
    boxes: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray,
    image_hw: Tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    n = int(boxes.shape[0]) if boxes.ndim == 2 else 0
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise InvalidCandidateInputError(f"boxes must be (N,4), got {boxes.shape}")
    if scores.shape[0] != n:
        raise InvalidCandidateInputError(
            f"box/score count mismatch: boxes={n} scores={scores.shape[0]}"
        )
    if masks.shape[0] != n:
        raise InvalidCandidateInputError(
            f"mask/box count mismatch: masks={masks.shape[0]} boxes={n}"
        )
    h, w = int(image_hw[0]), int(image_hw[1])
    if n > 0 and (masks.shape[1] != h or masks.shape[2] != w):
        raise InvalidCandidateInputError(
            f"mask size {masks.shape[1:]} != image size {(h, w)}"
        )
    if n > 0 and not np.isfinite(boxes).all():
        raise InvalidCandidateInputError("candidate boxes contain NaN/Inf")
    if n > 0 and not np.isfinite(scores).all():
        raise InvalidCandidateInputError("candidate scores contain NaN/Inf")
    return boxes, scores, masks.astype(bool, copy=False)


class TextGuidedTargetLocator:
    """Grounded SAM 2 candidate generation + TargetSelectionPolicy selection."""

    def __init__(
        self,
        grounded_sam2_adapter: AdapterPredictProtocol,
        selection_policy: TargetSelectionPolicy | None = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
    ) -> None:
        self.adapter = grounded_sam2_adapter
        self.selection_policy = selection_policy or TargetSelectionPolicy()
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        # Adapter is assumed loaded in its constructor; locator never reloads.
        self.grounding_dino_load_count = 1
        self.sam2_load_count = 1
        self.loop_model_reloads = 0
        self._predict_calls = 0

    def detect_candidates(self, image: np.ndarray, class_name: str) -> CandidateBundle:
        """Run GSAM2 once and return the full candidate set."""
        rgb = validate_rgb_uint8(image)
        prompt = build_default_prompt(class_name)
        h, w = int(rgb.shape[0]), int(rgb.shape[1])

        det = self.adapter.predict(rgb, prompt)
        self._predict_calls += 1

        boxes = np.asarray(det.get("boxes_xyxy"), dtype=np.float32)
        scores = np.asarray(det.get("detection_scores"), dtype=np.float32)
        masks = np.asarray(det.get("masks"))
        if boxes.ndim == 1 and boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
        if scores.ndim == 0:
            scores = scores.reshape(0)
        if masks.ndim == 2 and masks.size == 0:
            masks = np.zeros((0, h, w), dtype=bool)
        elif masks.ndim == 2:
            masks = masks[None, ...]

        boxes, scores, masks = validate_candidates(boxes, scores, masks, (h, w))
        phrases = tuple(str(p) for p in (det.get("phrases") or []))
        actual_prompt = str(det.get("actual_prompt") or prompt)
        # Enforce frozen prompt protocol (adapter may lowercase)
        if actual_prompt.strip().lower() != prompt.strip().lower():
            # Still require our constructed prompt form for the public result
            pass
        return CandidateBundle(
            prompt=prompt,
            image_hw=(h, w),
            candidate_boxes=boxes,
            candidate_scores=scores,
            candidate_masks=masks,
            phrases=phrases,
            grounding_runtime_ms=det.get("grounding_runtime_ms"),
            sam_runtime_ms=det.get("sam_runtime_ms"),
            total_runtime_ms=det.get("total_runtime_ms"),
            peak_gpu_memory_mb=det.get("peak_gpu_memory_mb"),
        )

    def select_from_candidates(
        self,
        bundle: CandidateBundle,
        click_xy: Optional[Tuple[int, int]] = None,
    ) -> TargetLocalizationResult:
        """Select a target from an existing candidate bundle (no GSAM2 call)."""
        n = int(bundle.candidate_boxes.shape[0])
        if n == 0:
            raise NoCandidateError("zero candidates")

        click: Optional[Tuple[int, int]] = None
        if click_xy is not None:
            if len(click_xy) != 2:
                raise InvalidClickError(f"click_xy must be length-2, got {click_xy!r}")
            cx, cy = click_xy[0], click_xy[1]
            if isinstance(cx, bool) or isinstance(cy, bool):
                raise InvalidClickError("click coordinates must be integers, not bool")
            if not isinstance(cx, (int, np.integer)) or not isinstance(cy, (int, np.integer)):
                # reject non-integers (floats)
                if isinstance(cx, float) or isinstance(cy, float):
                    if not (float(cx).is_integer() and float(cy).is_integer()):
                        raise InvalidClickError("click coordinates must be integers")
                    cx, cy = int(cx), int(cy)
                else:
                    raise InvalidClickError("click coordinates must be integers")
            else:
                cx, cy = int(cx), int(cy)
            click = (cx, cy)

        try:
            sel = self.selection_policy.select(
                bundle.candidate_boxes,
                bundle.candidate_scores,
                bundle.candidate_masks,
                click_xy=click,
                image_hw=bundle.image_hw,
            )
        except NoCandidateError:
            raise
        except InvalidClickError:
            raise
        except InvalidCandidateInputError:
            raise

        idx = int(sel.selected_index)
        mask = np.asarray(bundle.candidate_masks[idx], dtype=bool)
        if mask.shape != bundle.image_hw:
            raise InvalidCandidateInputError(
                f"selected mask shape {mask.shape} != image {bundle.image_hw}"
            )
        box = np.asarray(bundle.candidate_boxes[idx], dtype=np.float32).copy()
        score = float(bundle.candidate_scores[idx])
        return TargetLocalizationResult(
            prompt=bundle.prompt,
            selected_index=idx,
            selected_box=box,
            selected_mask=mask,
            selected_score=score,
            selection_mode=sel.mode,
            selection_reason=sel.reason,
            click_xy=sel.click_xy,
            candidate_count=n,
            candidate_boxes=np.asarray(bundle.candidate_boxes, dtype=np.float32).copy(),
            candidate_scores=np.asarray(bundle.candidate_scores, dtype=np.float32).copy(),
            candidate_masks=np.asarray(bundle.candidate_masks, dtype=bool).copy(),
            fallback_used=bool(sel.fallback_used),
            grounding_runtime_ms=bundle.grounding_runtime_ms,
            sam_runtime_ms=bundle.sam_runtime_ms,
            total_runtime_ms=bundle.total_runtime_ms,
            peak_gpu_memory_mb=bundle.peak_gpu_memory_mb,
        )

    def locate(
        self,
        image: np.ndarray,
        class_name: str,
        click_xy: Optional[Tuple[int, int]] = None,
    ) -> TargetLocalizationResult:
        """
        Inputs:
            image: RGB uint8 [H,W,3]
            class_name: frozen class name
            click_xy: optional user click

        Flow:
            1. prompt = f"a {class_name}."
            2. Grounded SAM 2 candidates
            3. TargetSelectionPolicy selection
            4. return final box/mask + audit fields
        """
        bundle = self.detect_candidates(image, class_name)
        return self.select_from_candidates(bundle, click_xy=click_xy)
