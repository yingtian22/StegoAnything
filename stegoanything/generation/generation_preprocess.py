"""Image/mask geometry for BrushNet (letterbox to max-side 512, pad to multiple of 8)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass
class GeometryTransform:
    original_size: tuple[int, int]  # (W, H)
    scaled_size: tuple[int, int]  # (W, H) after aspect-preserving resize
    canvas_size: tuple[int, int]  # (W, H) after padding
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    max_side: int = 512
    pad_multiple: int = 8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ceil_multiple(x: int, m: int) -> int:
    return int(np.ceil(x / m) * m)


def preprocess_image_mask(
    image: Image.Image | np.ndarray,
    mask: Image.Image | np.ndarray,
    max_side: int = 512,
    pad_multiple: int = 8,
) -> tuple[Image.Image, Image.Image, GeometryTransform, np.ndarray]:
    """Keep aspect ratio, longest side -> max_side, symmetric pad to multiple of pad_multiple.

    Returns RGB PIL image, RGB mask (white=edit), transform, and bool mask on canvas.
    """
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
    else:
        rgb = np.asarray(image)
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        rgb = rgb[..., :3].astype(np.uint8)

    if isinstance(mask, Image.Image):
        m = np.array(mask.convert("L"))
    else:
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m.sum(axis=-1)
        m = m.astype(np.uint8)

    h0, w0 = rgb.shape[:2]
    if m.shape[:2] != (h0, w0):
        raise ValueError(f"mask size {m.shape[:2]} != image {(h0, w0)}")

    binary = (m > 127).astype(np.uint8)
    if binary.sum() == 0:
        raise ValueError("empty mask")

    scale = float(max_side) / float(max(h0, w0))
    if scale > 1.0:
        scale = 1.0  # do not upscale small images beyond native; still pad
        # Actually for BrushNet SD1.5, operating near 512 is expected. Allow upscale of short side via max_side.
        scale = float(max_side) / float(max(h0, w0))

    new_w = max(1, int(round(w0 * scale)))
    new_h = max(1, int(round(h0 * scale)))
    rgb_rs = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    mask_rs = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas_w = _ceil_multiple(new_w, pad_multiple)
    canvas_h = _ceil_multiple(new_h, pad_multiple)
    pad_left = (canvas_w - new_w) // 2
    pad_right = canvas_w - new_w - pad_left
    pad_top = (canvas_h - new_h) // 2
    pad_bottom = canvas_h - new_h - pad_top

    rgb_pad = cv2.copyMakeBorder(rgb_rs, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    mask_pad = cv2.copyMakeBorder(mask_rs, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

    geom = GeometryTransform(
        original_size=(w0, h0),
        scaled_size=(new_w, new_h),
        canvas_size=(canvas_w, canvas_h),
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        max_side=max_side,
        pad_multiple=pad_multiple,
    )
    mask_rgb = np.stack([mask_pad * 255] * 3, axis=-1).astype(np.uint8)
    return (
        Image.fromarray(rgb_pad).convert("RGB"),
        Image.fromarray(mask_rgb).convert("RGB"),
        geom,
        mask_pad.astype(bool),
    )


def invert_geometry(
    canvas_image: Image.Image | np.ndarray,
    geom: GeometryTransform,
    is_mask: bool = False,
) -> np.ndarray:
    """Remove padding and resize back to original resolution."""
    if isinstance(canvas_image, Image.Image):
        arr = np.array(canvas_image.convert("L" if is_mask else "RGB"))
    else:
        arr = np.asarray(canvas_image)

    w_c, h_c = geom.canvas_size
    if arr.ndim == 2:
        if arr.shape != (h_c, w_c):
            raise ValueError(f"canvas shape {arr.shape} != {(h_c, w_c)}")
    else:
        if arr.shape[0] != h_c or arr.shape[1] != w_c:
            raise ValueError(f"canvas shape {arr.shape} != {(h_c, w_c, '?')}")

    y0, y1 = geom.pad_top, geom.pad_top + geom.scaled_size[1]
    x0, x1 = geom.pad_left, geom.pad_left + geom.scaled_size[0]
    cropped = arr[y0:y1, x0:x1]
    ow, oh = geom.original_size
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
    if cropped.ndim == 2:
        out = cv2.resize(cropped, (ow, oh), interpolation=interp)
    else:
        out = cv2.resize(cropped, (ow, oh), interpolation=interp)
    return out


def geometry_regression_check(image: Image.Image, mask: Image.Image, atol_area: float = 0.01) -> dict[str, Any]:
    """Verify area ratio preserved within atol and round-trip size."""
    m0 = np.array(mask.convert("L")) > 127
    area0 = float(m0.mean())
    img_p, mask_p, geom, m_bool = preprocess_image_mask(image, mask)
    area_c = float(m_bool.mean())
    # area on canvas vs original differs due to padding zeros; compare on scaled content
    y0, y1 = geom.pad_top, geom.pad_top + geom.scaled_size[1]
    x0, x1 = geom.pad_left, geom.pad_left + geom.scaled_size[0]
    area_s = float(m_bool[y0:y1, x0:x1].mean())
    back = invert_geometry(mask_p, geom, is_mask=True)
    back_b = back > 127
    area_b = float(back_b.mean())
    img_back = invert_geometry(img_p, geom, is_mask=False)
    ok = abs(area_b - area0) < atol_area and img_back.shape[:2] == (geom.original_size[1], geom.original_size[0])
    return {
        "ok": ok,
        "area_original": area0,
        "area_scaled_content": area_s,
        "area_canvas": area_c,
        "area_roundtrip": area_b,
        "area_abs_err": abs(area_b - area0),
        "geom": geom.to_dict(),
    }
