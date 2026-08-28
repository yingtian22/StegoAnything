"""Locator → BrushNet Feather → 256 full-canvas HiNet end-to-end pipeline.

Formal recovery always uses zero-z. No RevBridge cover, no hard-mask projection,
no local-full-local round-trip.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Tuple

import cv2
import numpy as np
import torch

from stegoanything.object_hiding.hinet_normalize import hinet_tensor_to_uint8, uint8_to_hinet_tensor
from stegoanything.pipeline.full_canvas_geometry import CanvasMetadata, FullCanvasGeometry
from stegoanything.pipeline.text_guided_brushnet_generator import (
    BrushNetGenerationResult,
    TextGuidedBrushNetGenerator,
    make_feather_alpha,
    selected_mask_to_brushnet_u8,
)
from stegoanything.pipeline.text_guided_target_locator import (
    TargetLocalizationResult,
    build_default_prompt,
    validate_rgb_uint8,
)


class TrainingForbiddenError(RuntimeError):
    """Raised if training APIs are invoked during end-to-end inference."""


class RevBridgeForbiddenError(RuntimeError):
    """Raised if RevBridge cover generation is requested."""


class LocalFullLocalForbiddenError(RuntimeError):
    """Raised if local↔full round-trip paths are requested."""


class HiNetModelProtocol(Protocol):
    def __call__(self, cover: torch.Tensor, secret: torch.Tensor) -> dict[str, torch.Tensor]: ...

    def recover_from_stego(
        self, stego: torch.Tensor, latent_z: torch.Tensor | None = None
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class StegoAnythingResult:
    class_name: str
    prompt: str
    click_xy: Optional[Tuple[int, int]]

    selected_box: np.ndarray
    selected_mask_full: np.ndarray
    selection_mode: str
    selection_reason: str

    brushnet_raw_full: np.ndarray
    brushnet_feather_full: np.ndarray

    original_canvas: np.ndarray
    cover_canvas: np.ndarray
    mask_canvas: np.ndarray
    secret_canvas: np.ndarray

    stego_canvas: np.ndarray
    recovered_secret_zero_z: np.ndarray
    restored_canvas: np.ndarray

    stego_full_resolution: np.ndarray
    restored_full_resolution: np.ndarray

    metadata: dict = field(default_factory=dict)


def apply_fixed_geometry_rgb(image: np.ndarray, meta: CanvasMetadata) -> np.ndarray:
    """Map an RGB image with the exact same letterbox geometry as `meta`."""
    cs = meta.canvas_size
    resized = cv2.resize(
        image,
        (meta.resized_width, meta.resized_height),
        interpolation=cv2.INTER_CUBIC,
    )
    canvas = np.full((cs, cs, 3), meta.letterbox_rgb_uint8, dtype=np.uint8)
    canvas[
        meta.padding_top : meta.padding_top + meta.resized_height,
        meta.padding_left : meta.padding_left + meta.resized_width,
    ] = resized
    return canvas


def construct_object_secret(original_canvas: np.ndarray, mask_canvas_u8: np.ndarray) -> np.ndarray:
    """S = M ⊙ I on canvas; outside mask (incl. letterbox) is black."""
    if original_canvas.shape[:2] != mask_canvas_u8.shape[:2]:
        raise ValueError("original/mask canvas shape mismatch")
    m = (mask_canvas_u8 > 127).astype(np.uint8)
    secret = original_canvas.copy()
    secret[m == 0] = 0
    outside_max = int(secret[m == 0].max()) if (m == 0).any() else 0
    if outside_max != 0:
        raise RuntimeError(f"secret outside mask max={outside_max}, expected 0")
    # Construction identity: inside mask equals original
    if m.any():
        max_err = int(np.max(np.abs(secret[m == 1].astype(np.int16) - original_canvas[m == 1].astype(np.int16))))
        if max_err != 0:
            raise RuntimeError(f"Secret construction max error={max_err}")
    return secret


def restore_with_known_mask(
    stego_canvas: np.ndarray,
    recovered_secret: np.ndarray,
    mask_canvas_u8: np.ndarray,
) -> np.ndarray:
    """Î = (1-M)⊙C_stego + M⊙Ŝ"""
    m = (mask_canvas_u8 > 127).astype(np.float32)[..., None]
    restored = (1.0 - m) * stego_canvas.astype(np.float32) + m * recovered_secret.astype(np.float32)
    return np.clip(np.rint(restored), 0, 255).astype(np.uint8)


class StegoAnythingEndToEndPipeline:
    """Online: text(+optional click) → locator → BrushNet Feather → HiNet zero-z."""

    def __init__(
        self,
        target_locator,
        brushnet_generator: TextGuidedBrushNetGenerator,
        hinet_model: HiNetModelProtocol,
        full_canvas_geometry: FullCanvasGeometry,
        canvas_size: int = 256,
        device: str = "cuda",
    ) -> None:
        self.target_locator = target_locator
        self.brushnet_generator = brushnet_generator
        self.hinet_model = hinet_model
        self.geometry = full_canvas_geometry
        self.canvas_size = int(canvas_size)
        if self.geometry.canvas_size != self.canvas_size:
            raise ValueError(
                f"geometry.canvas_size={self.geometry.canvas_size} != canvas_size={self.canvas_size}"
            )
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self._latent_shape: tuple[int, ...] | None = None
        self._train_calls = 0
        self._revbridge_calls = 0
        self._local_full_local_calls = 0
        self._hinet_hide_calls = 0
        self._hinet_reveal_calls = 0
        self._true_latent_used_for_formal_recovery = False

    def train(self, *args, **kwargs):  # pragma: no cover - must never be used
        self._train_calls += 1
        raise TrainingForbiddenError("training forbidden in Phase 6A-2C end-to-end")

    def _forbid_revbridge(self) -> None:
        self._revbridge_calls += 1
        raise RevBridgeForbiddenError("RevBridge cover generation is forbidden")

    def _forbid_local_full_local(self) -> None:
        self._local_full_local_calls += 1
        raise LocalFullLocalForbiddenError("local-full-local is forbidden")

    @torch.inference_mode()
    def hide(self, cover_canvas: np.ndarray, secret_canvas: np.ndarray) -> dict[str, Any]:
        if cover_canvas.shape != (self.canvas_size, self.canvas_size, 3):
            raise ValueError(f"cover_canvas shape {cover_canvas.shape}")
        if secret_canvas.shape != cover_canvas.shape:
            raise ValueError("secret/cover shape mismatch")
        cover_t = uint8_to_hinet_tensor(cover_canvas).unsqueeze(0).to(self.device)
        secret_t = uint8_to_hinet_tensor(secret_canvas).unsqueeze(0).to(self.device)
        out = self.hinet_model(cover_t, secret_t)
        stego_t = out["stego"]
        latent_z = out["latent_z"]
        self._latent_shape = tuple(latent_z.shape)
        self._hinet_hide_calls += 1
        return {
            "stego_canvas": hinet_tensor_to_uint8(stego_t[0]),
            "latent_z": latent_z.detach(),
            "cover_tensor_source": "brushnet_feather_canvas",
            "secret_tensor_source": "original_object_secret_canvas",
        }

    @torch.inference_mode()
    def recover_zero_z(self, stego_canvas: np.ndarray) -> dict[str, Any]:
        """Formal recovery: z_deploy = zeros_like(latent_z). True latent never used."""
        stego_t = uint8_to_hinet_tensor(stego_canvas).unsqueeze(0).to(self.device)
        if self._latent_shape is None:
            dummy = torch.zeros_like(stego_t)
            probe = self.hinet_model(dummy, dummy)
            self._latent_shape = tuple(probe["latent_z"].shape)
        # Explicit zero-z creation (formal path)
        z_deploy = torch.zeros(self._latent_shape, device=self.device, dtype=stego_t.dtype)
        assert float(z_deploy.abs().sum().item()) == 0.0
        recovered_t = self.hinet_model.recover_from_stego(stego_t, z_deploy)
        self._hinet_reveal_calls += 1
        self._true_latent_used_for_formal_recovery = False
        return {
            "recovered_secret_zero_z": hinet_tensor_to_uint8(recovered_t[0]),
            "z_deploy_is_zero": True,
            "true_latent_used": False,
            "recovery": "zero_z",
        }

    @torch.inference_mode()
    def recover_with_true_latent_diagnostic(
        self, stego_canvas: np.ndarray, latent_z: torch.Tensor
    ) -> dict[str, Any]:
        """DIAGNOSTIC_ONLY_WITH_TRUE_LATENT — must not feed formal metrics."""
        stego_t = uint8_to_hinet_tensor(stego_canvas).unsqueeze(0).to(self.device)
        recovered_t = self.hinet_model.recover_from_stego(stego_t, latent_z.to(self.device))
        return {
            "recovered_secret_with_z": hinet_tensor_to_uint8(recovered_t[0]),
            "DIAGNOSTIC_ONLY_WITH_TRUE_LATENT": True,
            "true_latent_used_for_formal_recovery": False,
        }

    def prepare_canvases(
        self,
        original_full: np.ndarray,
        brushnet_feather_full: np.ndarray,
        selected_mask_full: np.ndarray,
    ) -> dict[str, Any]:
        if original_full.shape[:2] != brushnet_feather_full.shape[:2]:
            raise ValueError("original/cover full-resolution size mismatch")
        mask_u8 = selected_mask_to_brushnet_u8(selected_mask_full)
        if mask_u8.shape[:2] != original_full.shape[:2]:
            raise ValueError("mask/original size mismatch")

        orig_pack = self.geometry.to_canvas(original_full, mask_u8, canvas_size=self.canvas_size)
        meta: CanvasMetadata = orig_pack["metadata"]
        cover_canvas = apply_fixed_geometry_rgb(brushnet_feather_full, meta)
        original_canvas = orig_pack["image_canvas"]
        mask_canvas = orig_pack["mask_canvas"]
        assert mask_canvas is not None
        mask_bin = (mask_canvas > 127).astype(np.uint8) * 255
        secret = construct_object_secret(original_canvas, mask_bin)
        return {
            "original_canvas": original_canvas,
            "cover_canvas": cover_canvas,
            "mask_canvas": mask_bin,
            "secret_canvas": secret,
            "metadata": meta,
            "geometry_shared": True,
            "local_full_local_used": False,
        }

    def run(
        self,
        image: np.ndarray,
        class_name: str,
        click_xy: Optional[Tuple[int, int]] = None,
        localization: TargetLocalizationResult | None = None,
        text_only_audit: TargetLocalizationResult | None = None,
        compute_with_z_diagnostic: bool = False,
    ) -> StegoAnythingResult:
        """
        1. TextGuidedTargetLocator → final box/mask
        2. BrushNet Feather cover
        3. Map to 256×256 canvas (shared geometry)
        4. Construct object secret
        5. HiNet hide
        6. zero-z reveal
        7. Known-mask restore → full-res viz
        """
        rgb = validate_rgb_uint8(image)
        prompt = build_default_prompt(class_name)

        gen: BrushNetGenerationResult = self.brushnet_generator.generate(
            rgb,
            class_name,
            click_xy=click_xy,
            localization=localization,
            text_only_audit=text_only_audit,
        )

        packs = self.prepare_canvases(
            rgb,
            gen.feathered_generated_image,
            gen.selected_mask,
        )
        hide = self.hide(packs["cover_canvas"], packs["secret_canvas"])
        # Formal recovery: zero-z only (do not pass hide["latent_z"])
        rec = self.recover_zero_z(hide["stego_canvas"])
        restored = restore_with_known_mask(
            hide["stego_canvas"],
            rec["recovered_secret_zero_z"],
            packs["mask_canvas"],
        )
        meta: CanvasMetadata = packs["metadata"]
        stego_full = self.geometry.from_canvas(hide["stego_canvas"], meta)
        restored_full = self.geometry.from_canvas(restored, meta)

        feather_alpha = make_feather_alpha(
            selected_mask_to_brushnet_u8(gen.selected_mask),
            self.brushnet_generator.feather_width,
        )

        md: dict[str, Any] = {
            "selection_mode": gen.selection_mode,
            "selection_reason": gen.selection_reason,
            "candidate_count": gen.candidate_count,
            "selected_score": gen.selected_score,
            "mask_transfer": gen.metadata.get("mask_transfer"),
            "canvas_metadata": meta.to_dict(),
            "geometry_shared": True,
            "secret_outside_max": 0,
            "zero_z_formal": True,
            "true_latent_used_for_formal_recovery": False,
            "true_latent_transmitted": False,
            "local_full_local_used": False,
            "revbridge_used": False,
            "hard_mask_projection": False,
            "hinet_cover_source": "brushnet_feather_canvas",
            "hinet_secret_source": "original_object_secret_canvas",
            "feather_width": self.brushnet_generator.feather_width,
            "brushnet_seed": gen.seed,
            "text_only_audit": gen.metadata.get("text_only_audit"),
            "feather_alpha_stats": {
                "min": float(feather_alpha.min()),
                "max": float(feather_alpha.max()),
            },
        }
        if compute_with_z_diagnostic:
            diag = self.recover_with_true_latent_diagnostic(hide["stego_canvas"], hide["latent_z"])
            md["with_z_diagnostic"] = {
                "DIAGNOSTIC_ONLY_WITH_TRUE_LATENT": True,
                "note": "not used for formal metrics",
            }
            md["recovered_secret_with_z_diagnostic"] = diag["recovered_secret_with_z"]

        # Drop heavy tensor from public metadata
        return StegoAnythingResult(
            class_name=class_name,
            prompt=prompt,
            click_xy=gen.click_xy,
            selected_box=np.asarray(gen.selected_box, dtype=np.float32).copy(),
            selected_mask_full=np.asarray(gen.selected_mask, dtype=bool).copy(),
            selection_mode=gen.selection_mode,
            selection_reason=gen.selection_reason,
            brushnet_raw_full=gen.raw_generated_image.copy(),
            brushnet_feather_full=gen.feathered_generated_image.copy(),
            original_canvas=packs["original_canvas"].copy(),
            cover_canvas=packs["cover_canvas"].copy(),
            mask_canvas=packs["mask_canvas"].copy(),
            secret_canvas=packs["secret_canvas"].copy(),
            stego_canvas=hide["stego_canvas"].copy(),
            recovered_secret_zero_z=rec["recovered_secret_zero_z"].copy(),
            restored_canvas=restored.copy(),
            stego_full_resolution=stego_full.copy(),
            restored_full_resolution=restored_full.copy(),
            metadata=md,
        )
