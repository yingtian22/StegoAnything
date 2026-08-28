"""Canonical full-image canvas: aspect-preserving resize + constant-gray letterbox.

No reflect padding. RGB uses bicubic; mask uses nearest.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CanvasMetadata:
    original_width: int
    original_height: int
    resize_scale: float
    resized_width: int
    resized_height: int
    padding_left: int
    padding_right: int
    padding_top: int
    padding_bottom: int
    canvas_size: int
    letterbox_rgb_uint8: int = 128
    letterbox_mask_uint8: int = 0
    rgb_interpolation: str = "bicubic"
    mask_interpolation: str = "nearest"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "CanvasMetadata":
        return CanvasMetadata(
            original_width=int(d["original_width"]),
            original_height=int(d["original_height"]),
            resize_scale=float(d["resize_scale"]),
            resized_width=int(d["resized_width"]),
            resized_height=int(d["resized_height"]),
            padding_left=int(d["padding_left"]),
            padding_right=int(d["padding_right"]),
            padding_top=int(d["padding_top"]),
            padding_bottom=int(d["padding_bottom"]),
            canvas_size=int(d["canvas_size"]),
            letterbox_rgb_uint8=int(d.get("letterbox_rgb_uint8", 128)),
            letterbox_mask_uint8=int(d.get("letterbox_mask_uint8", 0)),
            rgb_interpolation=str(d.get("rgb_interpolation", "bicubic")),
            mask_interpolation=str(d.get("mask_interpolation", "nearest")),
        )


class FullCanvasGeometry:
    """Map arbitrary full-resolution images/masks onto a fixed square canvas."""

    def __init__(self, canvas_size: int = 256, letterbox_rgb: int = 128, letterbox_mask: int = 0):
        self.canvas_size = int(canvas_size)
        self.letterbox_rgb = int(letterbox_rgb)
        self.letterbox_mask = int(letterbox_mask)
        if self.canvas_size % 2 != 0:
            raise ValueError("canvas_size must be even for HiNet/Haar")

    def to_canvas(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        canvas_size: int | None = None,
    ) -> dict[str, Any]:
        """
        Returns:
            image_canvas (HxWx3 uint8),
            mask_canvas (HxW uint8) if mask provided else None,
            metadata (CanvasMetadata / dict)
        """
        cs = int(canvas_size or self.canvas_size)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must be HxWx3, got {image.shape}")
        h0, w0 = int(image.shape[0]), int(image.shape[1])
        scale = min(cs / float(w0), cs / float(h0))
        rw = max(1, int(round(w0 * scale)))
        rh = max(1, int(round(h0 * scale)))
        # Ensure fit
        if rw > cs:
            rw = cs
        if rh > cs:
            rh = cs
        img_r = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_CUBIC)

        pad_l = (cs - rw) // 2
        pad_r = cs - rw - pad_l
        pad_t = (cs - rh) // 2
        pad_b = cs - rh - pad_t

        canvas = np.full((cs, cs, 3), self.letterbox_rgb, dtype=np.uint8)
        canvas[pad_t : pad_t + rh, pad_l : pad_l + rw] = img_r

        mask_canvas = None
        if mask is not None:
            m = mask
            if m.ndim == 3:
                m = m[..., 0]
            if m.shape[0] != h0 or m.shape[1] != w0:
                raise ValueError(f"mask shape {m.shape} != image {(h0, w0)}")
            m_r = cv2.resize(m.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST)
            mask_canvas = np.full((cs, cs), self.letterbox_mask, dtype=np.uint8)
            mask_canvas[pad_t : pad_t + rh, pad_l : pad_l + rw] = m_r

        meta = CanvasMetadata(
            original_width=w0,
            original_height=h0,
            resize_scale=float(scale),
            resized_width=rw,
            resized_height=rh,
            padding_left=pad_l,
            padding_right=pad_r,
            padding_top=pad_t,
            padding_bottom=pad_b,
            canvas_size=cs,
            letterbox_rgb_uint8=self.letterbox_rgb,
            letterbox_mask_uint8=self.letterbox_mask,
        )
        return {
            "image_canvas": canvas,
            "mask_canvas": mask_canvas,
            "metadata": meta,
            "metadata_dict": meta.to_dict(),
        }

    def from_canvas(
        self,
        canvas: np.ndarray,
        metadata: CanvasMetadata | dict[str, Any],
    ) -> np.ndarray:
        """Map canvas back to original resolution (visualization only; not for HiNet recovery)."""
        meta = metadata if isinstance(metadata, CanvasMetadata) else CanvasMetadata.from_dict(metadata)
        if canvas.shape[0] != meta.canvas_size or canvas.shape[1] != meta.canvas_size:
            raise ValueError(f"canvas size {canvas.shape[:2]} != {meta.canvas_size}")
        crop = canvas[
            meta.padding_top : meta.padding_top + meta.resized_height,
            meta.padding_left : meta.padding_left + meta.resized_width,
        ]
        if canvas.ndim == 3:
            out = cv2.resize(
                crop,
                (meta.original_width, meta.original_height),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            out = cv2.resize(
                crop,
                (meta.original_width, meta.original_height),
                interpolation=cv2.INTER_NEAREST,
            )
        return out.astype(np.uint8)
