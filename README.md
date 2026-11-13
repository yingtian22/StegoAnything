# StegoAnything

Official implementation of **StegoAnything: Text-Guided Reversible Object Hiding with Zero-Latent Decoding**, published in *Expert Systems with Applications*.

**Paper:** [StegoAnything: Text-Guided Reversible Object Hiding with Zero-Latent Decoding](https://doi.org/10.1016/j.eswa.2026.135076)  
**DOI:** [10.1016/j.eswa.2026.135076](https://doi.org/10.1016/j.eswa.2026.135076)  
**Repository update:** November 13, 2026

---

## Overview

StegoAnything combines text-guided object localization, same-class semantic cover generation, and reversible object hiding with **zero-latent decoding**.

Given an image and a class prompt (`a {class}.`), the system localizes the target object, generates a same-class replacement, and hides the original object secret `S = M ⊙ I` in the resulting image. During recovery, the reverse network uses a fixed `z = 0`; the true forward latent is never transmitted.

```text
image + class text (+ optional click)
    → Grounded-SAM-2 localization
    → BrushNet same-class cover
    → 256×256 full-canvas HiNet hiding
    → secret recovery with z = 0
    → restored = (1 − M) ⊙ stego + M ⊙ recovered_secret
```

Recovery provides a high-fidelity approximation of the original object rather than guaranteed pixel-exact reconstruction. Object restoration requires the corresponding model, target mask, and geometric information for canvas mapping.

This repository provides a cleaned public release of the StegoAnything method. Official comparison baseline implementations—HiNet, UDH, DDH, StegFormer, DeepMIH, IICNet, and StegTransX—are not included. The HiNet-style invertible hiding network used within StegoAnything is included.

---

## What you can run

| Demo | Needs extra downloads? | Command |
|---|---|---|
| **Hide + zero-latent recovery** (recommended first) | No additional model or dataset downloads | `python scripts/demo_hide_recover.py` |
| Full text → localization → cover generation → hiding pipeline | Yes; see Section 3 | `python scripts/demo_e2e.py` |

The first demo uses three bundled example tuples containing a prepared cover, mask, and original image. After installing the environment, you can run it using the bundled checkpoint.

---

## 1. Environment

Use **Python 3.10**. An NVIDIA GPU with a compatible CUDA environment is recommended.

```bash
conda create -n stegoanything python=3.10 -y
conda activate stegoanything

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

CPU execution is supported for the hide/recover demo but is slower. The full end-to-end demo requires CUDA.

---

## 2. Run the hide / recover demo

No additional model or dataset downloads are required for this demo.

From the repository root, run:

```bash
python scripts/demo_hide_recover.py
```

Outputs are saved to `outputs/demo_hide_recover/{cat,dog,backpack}/`:

| File | Description |
|---|---|
| `stego.png` | Container image carrying the hidden object information |
| `recovered_secret.png` | Object secret recovered with `z = 0` |
| `restored.png` | Image restored using the known target mask |
| `panel.png` | Side-by-side visualization |
| `metrics.json` | Evaluation metrics |

To run a single example:

```bash
python scripts/demo_hide_recover.py --example cat
```

**Bundled checkpoint:** `checkpoints/hinet_full_canvas_seed0.pt`  
**Configuration:** formal Full-Canvas model, seed 0  
**Size:** approximately 9.4 MB

---

## 3. Optional: full pipeline setup

Skip this section if you only want to run the hide/recover demo.

The full pipeline additionally requires Grounded-SAM-2, BrushNet, and their pretrained weights.

### 3.1 Download the external code

Clone the repositories into the following folders:

```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git third_party/grounded-sam-2
git clone https://github.com/TencentARC/BrushNet.git third_party/BrushNet
```

Install the Grounded-SAM-2 packages from these local folders:

```bash
pip install -e "third_party/grounded-sam-2[notebooks]"
pip install --no-build-isolation -e third_party/grounded-sam-2/grounding_dino
```

For BrushNet, add `third_party/BrushNet/src` to `PYTHONPATH`, or install its package:

```bash
pip install -e third_party/BrushNet
```

The segmentation and generation stages may require different dependency versions. The paper experiments used separate environments where needed:

- **Segmentation:** Grounded-SAM-2 may require `transformers==4.33.2`.
- **Generation:** BrushNet may require `transformers==4.38.2` and `accelerate==0.20.3`.

If these dependencies conflict in a single environment, use separate Conda environments, such as `stego-seg` and `stego-gen`, for the corresponding stages.

### 3.2 Download the pretrained weights

Place the weights in the following locations:

| Model | Required location | Download source |
|---|---|---|
| GroundingDINO Swin-T | `checkpoints/grounded_sam2/groundingdino_swint_ogc.pth` | [GroundingDINO releases](https://github.com/IDEA-Research/GroundingDINO/releases); approximately 662 MB |
| SAM 2.1 Hiera-L | `checkpoints/grounded_sam2/sam2.1_hiera_large.pt` | [SAM 2 checkpoints](https://github.com/facebookresearch/sam2); approximately 857 MB |
| Stable Diffusion 1.5 | `checkpoints/brushnet/stable-diffusion-v1-5/` | Hugging Face: [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5); FP16 weights are sufficient |
| BrushNet segmentation-mask model | `checkpoints/brushnet/segmentation_mask_brushnet_ckpt/` | [BrushNet repository](https://github.com/TencentARC/BrushNet) / Hugging Face: [`TencentARC/BrushNet`](https://huggingface.co/TencentARC/BrushNet) |

Create the parent directories before placing the downloaded files:

```bash
mkdir -p checkpoints/grounded_sam2 checkpoints/brushnet
```

Model paths are also listed in `configs/paths.yaml`.

### 3.3 Run the full pipeline

From the repository root:

```bash
python scripts/demo_e2e.py \
    --image examples/cat/original_full.png \
    --class-name cat
```

To distinguish between multiple objects of the same class, optionally provide a click using pixel coordinates `x,y` in the original image:

```bash
python scripts/demo_e2e.py \
    --image examples/dog/original_full.png \
    --class-name dog \
    --click 320,200
```

---

## 4. Optional: LVIS / COCO for paper-scale evaluation

The full datasets are not required for either demo.

| Dataset | Suggested location | Source |
|---|---|---|
| COCO 2017 validation images | `data/coco2017/val2017/` | [COCO downloads](https://cocodataset.org/#download) |
| LVIS v1 validation annotations | `data/lvis/annotations/lvis_v1_val.json` | [LVIS dataset](https://www.lvisdataset.org/dataset) |

The three examples under `examples/` are derived from COCO/LVIS samples and are included so that the hide/recover demo can run without downloading the full datasets. Source images remain subject to their individual original licenses.

---

## Repository layout

```text
stegoanything/          # StegoAnything method package
  models/              # HiNet-style invertible hiding and recovery
  hiding/              # Checkpoint loading
  pipeline/            # 256×256 canvas processing, localization, cover generation, e2e
  generation/          # BrushNet adapter
  segmentation/        # Grounded-SAM-2 adapter
  selection/           # Text-guided and optional-click selection policy
  metrics/             # Evaluation metrics
scripts/
  demo_hide_recover.py # Hide and zero-latent recovery demo
  demo_e2e.py          # Full pipeline demo
examples/              # Cat, dog, and backpack example tuples
checkpoints/           # Bundled hiding checkpoint and external model locations
third_party/           # External Grounded-SAM-2 and BrushNet repositories
configs/paths.yaml     # Model and resource paths
```

---

## Method notes

- **Canvas:** images are letterboxed onto a `256×256` canvas.
- **Normalization:** RGB values are mapped to `[-1, 1]` using `uint8 / 127.5 - 1`.
- **Secret construction:** the object secret is `S = M ⊙ I`, where `M` is the target mask and `I` is the image in the corresponding canvas coordinates.
- **Zero-latent decoding:** deployment recovery always uses `recover_from_stego(stego, z=0)`. The true forward latent is not transmitted.
- **Mask usage:** the mask is used for secret construction, masked restoration, and region-specific evaluation. It is not an input to the hiding/recovery network.
- **Restoration:** the recovered object is composited into the stego image using the target mask. Original-resolution restoration additionally uses the corresponding geometric mapping.
- **Evaluation scope:** canvas-space secret reconstruction and original-resolution object restoration are distinct evaluations and should be reported separately.
- **Paper protocol:** the formal three-seed LVIS results use the same hiding and zero-latent recovery protocol. The bundled demo checkpoint corresponds to seed 0.

---

## Citation

If you use this code or build upon this work, please cite:

> **StegoAnything: Text-Guided Reversible Object Hiding with Zero-Latent Decoding.**  
> *Expert Systems with Applications*, 2026.  
> DOI: [10.1016/j.eswa.2026.135076](https://doi.org/10.1016/j.eswa.2026.135076)

Please use the publisher's citation export for the complete author list and final bibliographic metadata.

---

## Acknowledgments

StegoAnything builds upon the following open-source components:

- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) for text-guided object localization and segmentation.
- [BrushNet](https://github.com/TencentARC/BrushNet) for same-class semantic cover generation.
- [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) as the generative backbone.
- [HiNet](https://github.com/TomTomTommi/HiNet) for the invertible image-hiding formulation.

Please also cite the relevant original works when using these components. External code, pretrained weights, and datasets remain subject to their respective licenses and terms of use.
