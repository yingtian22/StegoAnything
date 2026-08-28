"""Project-level BrushNet adapter wrapping the official StableDiffusionBrushNetPipeline."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from stegoanything.generation.generation_preprocess import GeometryTransform, invert_geometry, preprocess_image_mask


class BrushNetAdapter:
    def __init__(
        self,
        repo_dir: str | Path,
        base_model_path: str | Path,
        brushnet_checkpoint: str | Path,
        device: str = "cuda",
        dtype: str = "float16",
        seed: int = 2026,
        enable_cpu_offload: bool = False,
    ) -> None:
        if device != "cuda":
            raise RuntimeError("BrushNetAdapter refuses non-CUDA device (no silent CPU fallback).")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

        self.repo_dir = Path(repo_dir)
        self.base_model_path = Path(base_model_path)
        self.brushnet_checkpoint = Path(brushnet_checkpoint)
        self.device = device
        self.dtype = torch.float16 if dtype in ("float16", "fp16") else torch.float32
        self.default_seed = int(seed)
        self.enable_cpu_offload = bool(enable_cpu_offload)

        # Prefer editable install; also ensure repo src is importable as fallback without modifying files.
        src = self.repo_dir / "src"
        if src.is_dir() and str(src) not in sys.path:
            sys.path.insert(0, str(src))

        from diffusers import BrushNetModel, StableDiffusionBrushNetPipeline, UniPCMultistepScheduler

        t0 = time.perf_counter()
        brushnet = BrushNetModel.from_pretrained(str(self.brushnet_checkpoint), torch_dtype=self.dtype)
        # Prefer local fp16 weights when full fp32 unet safetensors is incomplete.
        load_kwargs = dict(
            brushnet=brushnet,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=False,
            safety_checker=None,
            requires_safety_checker=False,
        )
        unet_fp16 = self.base_model_path / "unet" / "diffusion_pytorch_model.fp16.safetensors"
        unet_full = self.base_model_path / "unet" / "diffusion_pytorch_model.safetensors"
        if unet_fp16.is_file() and (not unet_full.is_file() or unet_full.stat().st_size < 1_000_000_000):
            load_kwargs["variant"] = "fp16"
        self.pipe = StableDiffusionBrushNetPipeline.from_pretrained(
            str(self.base_model_path),
            **load_kwargs,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        if self.enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)
        self.load_time_s = time.perf_counter() - t0
        self._loaded = True

        # Official defaults
        self.default_num_inference_steps = 50
        self.default_guidance_scale = 7.5
        self.default_conditioning_scale = 1.0

    @staticmethod
    def prepare_masked_conditioning(image_rgb: Image.Image, mask_rgb: Image.Image) -> tuple[Image.Image, Image.Image, np.ndarray]:
        """Official mask direction: white = edit; zero-out edit region on conditioning image."""
        init = np.array(image_rgb.convert("RGB")).astype(np.float32)
        mask = 1.0 * (np.array(mask_rgb.convert("RGB")).sum(-1) > 255)
        if mask.sum() == 0:
            # also accept single-channel style
            mask = 1.0 * (np.array(mask_rgb.convert("L")) > 127)
        if mask.sum() == 0:
            raise ValueError("empty mask after binarization")
        mask = mask[:, :, np.newaxis]
        cond = init * (1.0 - mask)
        cond_img = Image.fromarray(cond.astype(np.uint8)).convert("RGB")
        mask_img = Image.fromarray((mask[:, :, 0] * 255).astype(np.uint8)).convert("RGB")
        # Expand to 3ch white as official
        mask_np = np.array(mask_img.convert("L"))
        mask_img = Image.fromarray(np.stack([mask_np] * 3, axis=-1)).convert("RGB")
        return cond_img, mask_img, mask[:, :, 0].astype(bool)

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
    ) -> dict[str, Any]:
        if not self._loaded:
            raise RuntimeError("adapter not loaded")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA disappeared")

        steps = int(self.default_num_inference_steps if num_inference_steps is None else num_inference_steps)
        gs = float(self.default_guidance_scale if guidance_scale is None else guidance_scale)
        cs = float(self.default_conditioning_scale if conditioning_scale is None else conditioning_scale)
        seed_i = int(self.default_seed if seed is None else seed)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        else:
            image = image.convert("RGB")
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask.astype(np.uint8))
        # ensure non-empty
        m0 = np.array(mask.convert("L")) > 127
        if m0.sum() == 0:
            raise ValueError("empty mask")

        pre_img, pre_mask, geom, canvas_bool = preprocess_image_mask(image, mask, max_side=max_side)
        cond_img, mask_img, _ = self.prepare_masked_conditioning(pre_img, pre_mask)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        generator = torch.Generator(device="cuda").manual_seed(seed_i)
        with torch.inference_mode():
            out = self.pipe(
                prompt,
                cond_img,
                mask_img,
                num_inference_steps=steps,
                guidance_scale=gs,
                brushnet_conditioning_scale=cs,
                generator=generator,
                negative_prompt=negative_prompt,
            ).images[0]
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        peak_mb = float(torch.cuda.max_memory_allocated() / (1024**2))

        raw_canvas = out.convert("RGB")
        raw_np = np.array(raw_canvas)
        if not np.isfinite(raw_np).all():
            raise RuntimeError("NaN/Inf in BrushNet output")
        if raw_np.min() == raw_np.max():
            raise RuntimeError("degenerate constant output")

        # Map back to original resolution
        raw_orig = invert_geometry(raw_canvas, geom, is_mask=False)
        raw_orig_img = Image.fromarray(raw_orig.astype(np.uint8)).convert("RGB")
        mask_orig = (np.array(mask.convert("L")) > 127).astype(np.uint8) * 255

        return {
            "raw_generated_image": raw_orig_img,
            "raw_generated_canvas": raw_canvas,
            "preprocessed_image": pre_img,
            "preprocessed_mask": pre_mask,
            "conditioning_image": cond_img,
            "transform_metadata": geom.to_dict(),
            "runtime_metadata": {
                "runtime_ms": runtime_ms,
                "peak_gpu_memory_mb": peak_mb,
                "seed": seed_i,
                "num_inference_steps": steps,
                "guidance_scale": gs,
                "conditioning_scale": cs,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "dtype": str(self.dtype),
                "device": self.device,
                "load_time_s": self.load_time_s,
            },
            "mask_original_u8": mask_orig,
            "geometry": geom,
        }
