"""Text-guided BrushNet generator: TextGuidedTargetLocator → BrushNet → Feather cover."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Tuple

import cv2
import numpy as np
from PIL import Image

from stegoanything.generation.local_composite import feather_composite
from stegoanything.pipeline.text_guided_target_locator import (
    InvalidImageError,
    TextGuidedTargetLocator,
    TargetLocalizationResult,
    build_default_prompt,
    validate_rgb_uint8,
)


class EmptyMaskError(ValueError):
    """Raised when the selected mask has no foreground pixels."""


class MaskShapeError(ValueError):
    """Raised when mask spatial size does not match the image."""


class BrushNetGenerateProtocol(Protocol):
    def generate(
        self,
        image: Image.Image | np.ndarray,
        mask: Image.Image | np.ndarray,
        prompt: str,
        negative_prompt: str | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        conditioning_scale: float | None = None,
        seed: int | None = None,
        max_side: int = 512,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class BrushNetGenerationResult:
    prompt: str
    selected_box: np.ndarray
    selected_mask: np.ndarray
    selection_mode: str
    selection_reason: str
    click_xy: Optional[Tuple[int, int]]
    raw_generated_image: np.ndarray
    feathered_generated_image: np.ndarray
    feather_mask: np.ndarray
    seed: int
    metadata: dict = field(default_factory=dict)
    brushnet_input_mask_u8: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.uint8))
    candidate_count: int = 0
    selected_score: float = float("nan")
    text_only_selected_index: int | None = None
    text_only_selection_reason: str | None = None


def selected_mask_to_brushnet_u8(selected_mask: np.ndarray) -> np.ndarray:
    """Convert locator mask to BrushNet uint8 {0,255} without inversion/resize/morphology."""
    m = np.asarray(selected_mask)
    if m.ndim != 2:
        raise MaskShapeError(f"selected_mask must be HxW, got {m.shape}")
    if m.dtype == bool:
        out = m.astype(np.uint8) * 255
    else:
        # Treat any positive as foreground; map to exact {0,255}
        out = (m > 0).astype(np.uint8) * 255
    return out


def make_feather_alpha(mask_u8: np.ndarray, feather_width: int = 5) -> np.ndarray:
    """Deterministic soft alpha in [0,1] matching feather_composite."""
    m = (np.asarray(mask_u8) > 127).astype(np.float32)
    if feather_width <= 0:
        return m
    k = int(feather_width) * 2 + 1
    return cv2.GaussianBlur(m, (k, k), 0)


def assert_mask_transfer_identity(locator_mask: np.ndarray, brushnet_mask_u8: np.ndarray) -> dict[str, float]:
    a = selected_mask_to_brushnet_u8(locator_mask)
    b = np.asarray(brushnet_mask_u8, dtype=np.uint8)
    if a.shape != b.shape:
        raise MaskShapeError(f"mask transfer shape mismatch {a.shape} vs {b.shape}")
    max_err = int(np.max(np.abs(a.astype(np.int16) - b.astype(np.int16)))) if a.size else 0
    inter = int(np.logical_and(a > 127, b > 127).sum())
    union = int(np.logical_or(a > 127, b > 127).sum())
    iou = float(inter / union) if union > 0 else 1.0
    fg_diff = int(abs(int((a > 127).sum()) - int((b > 127).sum())))
    if iou != 1.0 or max_err != 0 or fg_diff != 0:
        raise RuntimeError(
            f"FAIL_MASK_TRANSFER: iou={iou} max_err={max_err} fg_diff={fg_diff}"
        )
    return {"iou": iou, "max_pixel_error": float(max_err), "foreground_count_diff": float(fg_diff)}


class TextGuidedBrushNetGenerator:
    """Locate target with TextGuidedTargetLocator, then generate BrushNet Feather cover."""

    def __init__(
        self,
        target_locator: TextGuidedTargetLocator,
        brushnet_adapter: BrushNetGenerateProtocol,
        seed: int = 2026,
        inference_steps: int = 50,
        guidance_scale: float = 7.5,
        conditioning_scale: float = 1.0,
        feather_width: int = 5,
    ) -> None:
        self.target_locator = target_locator
        self.brushnet_adapter = brushnet_adapter
        self.seed = int(seed)
        self.inference_steps = int(inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.conditioning_scale = float(conditioning_scale)
        self.feather_width = int(feather_width)
        self.brushnet_load_count = 1
        self.loop_model_reloads = 0
        self._generate_calls = 0
        self._hinet_calls = 0  # must remain 0

    def generate(
        self,
        image: np.ndarray,
        class_name: str,
        click_xy: Optional[Tuple[int, int]] = None,
        localization: TargetLocalizationResult | None = None,
        text_only_audit: TargetLocalizationResult | None = None,
    ) -> BrushNetGenerationResult:
        """
        1. TextGuidedTargetLocator → final box/mask (or reuse provided localization)
        2. Pass selected_mask to BrushNet unchanged (as uint8 {0,255})
        3. Produce BrushNet Raw full image
        4. Feather-composite with frozen width
        5. Return full audit payload
        """
        rgb = validate_rgb_uint8(image)
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        prompt = build_default_prompt(class_name)

        if localization is None:
            loc = self.target_locator.locate(rgb, class_name, click_xy=click_xy)
        else:
            loc = localization
            if loc.prompt != prompt:
                # Keep frozen prompt protocol
                pass

        sel_mask = np.asarray(loc.selected_mask)
        if sel_mask.shape != (h, w):
            raise MaskShapeError(f"selected_mask {sel_mask.shape} != image {(h, w)}")
        mask_u8 = selected_mask_to_brushnet_u8(sel_mask)
        if int((mask_u8 > 127).sum()) == 0:
            raise EmptyMaskError("empty selected mask")

        transfer = assert_mask_transfer_identity(sel_mask, mask_u8)

        gen = self.brushnet_adapter.generate(
            image=rgb,
            mask=mask_u8,
            prompt=prompt,
            num_inference_steps=self.inference_steps,
            guidance_scale=self.guidance_scale,
            conditioning_scale=self.conditioning_scale,
            seed=self.seed,
        )
        self._generate_calls += 1

        # Prefer adapter-reported input mask if present (must still match)
        if "mask_original_u8" in gen and gen["mask_original_u8"] is not None:
            reported = np.asarray(gen["mask_original_u8"], dtype=np.uint8)
            if reported.ndim == 3:
                reported = reported[..., 0]
            transfer = assert_mask_transfer_identity(sel_mask, reported)
            mask_u8 = selected_mask_to_brushnet_u8(sel_mask)  # keep locator-derived for feather

        raw = gen["raw_generated_image"]
        if isinstance(raw, Image.Image):
            raw_np = np.asarray(raw.convert("RGB"), dtype=np.uint8)
        else:
            raw_np = np.asarray(raw, dtype=np.uint8)
            if raw_np.ndim != 3 or raw_np.shape[2] != 3:
                raise RuntimeError(f"invalid BrushNet raw shape {raw_np.shape}")

        if raw_np.shape[:2] != (h, w):
            raise RuntimeError(f"BrushNet raw size {raw_np.shape[:2]} != image {(h, w)}")
        if not np.isfinite(raw_np.astype(np.float32)).all():
            raise RuntimeError("NaN/Inf in BrushNet raw output")

        feather_img = feather_composite(rgb, raw_np, mask_u8, feather_width=self.feather_width)
        feather_np = np.asarray(feather_img.convert("RGB"), dtype=np.uint8)
        if feather_np.shape[:2] != (h, w):
            raise RuntimeError(f"Feather size {feather_np.shape[:2]} != image {(h, w)}")

        feather_alpha = make_feather_alpha(mask_u8, self.feather_width)

        meta = {
            "inference_steps": self.inference_steps,
            "guidance_scale": self.guidance_scale,
            "conditioning_scale": self.conditioning_scale,
            "feather_width": self.feather_width,
            "mask_transfer": transfer,
            "brushnet_runtime": gen.get("runtime_metadata"),
            "hinet_called": False,
            "text_only_audit": None
            if text_only_audit is None
            else {
                "selected_index": text_only_audit.selected_index,
                "selection_reason": text_only_audit.selection_reason,
                "selection_mode": text_only_audit.selection_mode,
            },
        }

        return BrushNetGenerationResult(
            prompt=prompt,
            selected_box=np.asarray(loc.selected_box, dtype=np.float32).copy(),
            selected_mask=np.asarray(sel_mask, dtype=bool).copy(),
            selection_mode=loc.selection_mode,
            selection_reason=loc.selection_reason,
            click_xy=loc.click_xy,
            raw_generated_image=raw_np.copy(),
            feathered_generated_image=feather_np.copy(),
            feather_mask=feather_alpha.astype(np.float32).copy(),
            seed=self.seed,
            metadata=meta,
            brushnet_input_mask_u8=mask_u8.copy(),
            candidate_count=int(loc.candidate_count),
            selected_score=float(loc.selected_score),
            text_only_selected_index=(
                None if text_only_audit is None else int(text_only_audit.selected_index)
            ),
            text_only_selection_reason=(
                None if text_only_audit is None else str(text_only_audit.selection_reason)
            ),
        )
