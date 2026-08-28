"""Grounded-SAM-2 local API adapter (no HuggingFace primary path)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert

ImageInput = Union[str, Path, np.ndarray]


class GroundedSAM2Adapter:
    def __init__(
        self,
        repo_dir: str | Path,
        grounding_config: str | Path,
        grounding_checkpoint: str | Path,
        sam2_config: str | Path,
        sam2_checkpoint: str | Path,
        device: str = "cuda",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
    ) -> None:
        if device != "cuda":
            raise RuntimeError("GroundedSAM2Adapter requires device='cuda' (no CPU fallback).")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but device='cuda' was requested.")

        self.repo_dir = Path(repo_dir).resolve()
        self.grounding_config = self._resolve_repo_path(grounding_config)
        self.grounding_checkpoint = self._resolve_repo_path(grounding_checkpoint)
        # SAM2 config is a Hydra config name (e.g. configs/sam2.1/sam2.1_hiera_l.yaml),
        # resolved inside the sam2 package — not a project filesystem path.
        self.sam2_config = str(sam2_config).replace("\\", "/")
        self.sam2_checkpoint = self._resolve_repo_path(sam2_checkpoint)
        self.device = "cuda"
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)

        if not self.repo_dir.is_dir():
            raise FileNotFoundError(f"repo_dir not found: {self.repo_dir}")
        for label, path in [
            ("grounding_config", self.grounding_config),
            ("grounding_checkpoint", self.grounding_checkpoint),
            ("sam2_checkpoint", self.sam2_checkpoint),
        ]:
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")

        repo_str = str(self.repo_dir)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        prev_cwd = os.getcwd()
        os.chdir(self.repo_dir)
        try:
            from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            import grounding_dino.groundingdino.datasets.transforms as T

            self._load_image = load_image
            self._predict = predict
            self._image_transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            t0 = time.perf_counter()
            self.grounding_model = load_model(
                model_config_path=str(self.grounding_config),
                model_checkpoint_path=str(self.grounding_checkpoint),
                device=self.device,
            )
            sam2_model = build_sam2(
                self.sam2_config,
                str(self.sam2_checkpoint),
                device=self.device,
            )
            self.sam2_predictor = SAM2ImagePredictor(sam2_model)
            self.load_time_s = time.perf_counter() - t0
        finally:
            os.chdir(prev_cwd)

        self._loaded = True

    def _resolve_repo_path(self, path: str | Path) -> Path:
        p = Path(path)
        if p.is_file():
            return p.resolve()
        candidate = (self.repo_dir / p).resolve()
        return candidate

    @staticmethod
    def _normalize_prompt(text_prompt: str) -> tuple[str, str]:
        original = text_prompt
        actual = text_prompt.strip().lower()
        if not actual.endswith("."):
            actual += "."
        return original, actual

    def _prepare_image(self, image: ImageInput) -> tuple[np.ndarray, torch.Tensor]:
        if isinstance(image, (str, Path)):
            image_path = str(Path(image).resolve())
            return self._load_image(image_path)

        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB uint8 array, got shape {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        image_source = arr
        pil_image = Image.fromarray(image_source, mode="RGB")
        image_transformed, _ = self._image_transform(pil_image, None)
        return image_source, image_transformed

    @staticmethod
    def _peak_gpu_memory_mb() -> float:
        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))

    @torch.inference_mode()
    def predict(self, image: ImageInput, text_prompt: str) -> dict[str, Any]:
        original_prompt, actual_prompt = self._normalize_prompt(text_prompt)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        image_source, image_tensor = self._prepare_image(image)
        h, w = int(image_source.shape[0]), int(image_source.shape[1])

        prev_cwd = os.getcwd()
        os.chdir(self.repo_dir)
        try:
            t_ground0 = time.perf_counter()
            boxes, confidences, labels = self._predict(
                model=self.grounding_model,
                image=image_tensor,
                caption=actual_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )
            grounding_runtime_ms = (time.perf_counter() - t_ground0) * 1000.0

            n = int(boxes.shape[0]) if boxes is not None else 0
            if n == 0:
                return {
                    "boxes_xyxy": np.zeros((0, 4), dtype=np.float32),
                    "detection_scores": np.zeros((0,), dtype=np.float32),
                    "phrases": [],
                    "masks": np.zeros((0, h, w), dtype=bool),
                    "mask_scores": np.zeros((0,), dtype=np.float32),
                    "original_size": (h, w),
                    "original_prompt": original_prompt,
                    "actual_prompt": actual_prompt,
                    "grounding_runtime_ms": float(grounding_runtime_ms),
                    "sam_runtime_ms": 0.0,
                    "total_runtime_ms": float(grounding_runtime_ms),
                    "peak_gpu_memory_mb": self._peak_gpu_memory_mb(),
                    "num_detections": 0,
                }

            self.sam2_predictor.set_image(image_source)

            boxes_scaled = boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype)
            input_boxes = box_convert(boxes=boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy").numpy()

            if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            t_sam0 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                masks, mask_scores, _logits = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )
            sam_runtime_ms = (time.perf_counter() - t_sam0) * 1000.0
        finally:
            os.chdir(prev_cwd)

        if masks.ndim == 4:
            masks = masks.squeeze(1)
        masks = np.asarray(masks, dtype=bool)
        mask_scores_arr = np.asarray(mask_scores, dtype=np.float32).reshape(-1)
        detection_scores = np.asarray(confidences.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        phrases = list(labels)

        total_runtime_ms = grounding_runtime_ms + sam_runtime_ms
        return {
            "boxes_xyxy": np.asarray(input_boxes, dtype=np.float32),
            "detection_scores": detection_scores,
            "phrases": phrases,
            "masks": masks,
            "mask_scores": mask_scores_arr,
            "original_size": (h, w),
            "original_prompt": original_prompt,
            "actual_prompt": actual_prompt,
            "grounding_runtime_ms": float(grounding_runtime_ms),
            "sam_runtime_ms": float(sam_runtime_ms),
            "total_runtime_ms": float(total_runtime_ms),
            "peak_gpu_memory_mb": self._peak_gpu_memory_mb(),
            "num_detections": int(masks.shape[0]),
        }
