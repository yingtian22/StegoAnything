# StegoAnything

Semantic object-level reversible hiding with **zero-latent recovery**.

Given an image and a class prompt (`a {class}.`), the system localizes the object, builds a same-class semantic cover, hides the object secret `S = M ⊙ I`, and recovers it with a fixed `z = 0`. The true forward latent is never transmitted.

```text
image + class text (+ optional click)
    → Grounded-SAM-2 localization
    → BrushNet same-class cover
    → 256×256 full-canvas HiNet hide
    → recover with z = 0
    → restored = (1-M)⊙stego + M⊙secret̂
```

This folder is a cleaned public snapshot of the StegoAnything method only. Official comparison baselines (HiNet, UDH, DDH, StegFormer, DeepMIH, IICNet, StegTransX) are **not** included.

---

## What you can run

| Demo | Needs extra downloads? | Command |
|---|---|---|
| **Hide + zero-z recover** (recommended first) | No | `python scripts/demo_hide_recover.py` |
| Full text → localize → cover → hide | Yes (see below) | `python scripts/demo_e2e.py` |

The first demo uses three bundled example tuples (already-made cover / mask / original). After the environment step it should run as-is.

---

## 1. Environment

Python **3.10**, NVIDIA GPU + CUDA recommended.

```bash
conda create -n stegoanything python=3.10 -y
conda activate stegoanything
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

CPU can run hide/recover but is slow. The full e2e demo requires CUDA.

---

## 2. Run the hide / recover demo (no extra downloads)

From this repository root:

```bash
python scripts/demo_hide_recover.py
```

Outputs go to `outputs/demo_hide_recover/{cat,dog,backpack}/`:

- `stego.png` — container
- `recovered_secret.png` — secret recovered with `z = 0`
- `restored.png` — known-mask restoration
- `panel.png` — side-by-side
- `metrics.json`

Single example:

```bash
python scripts/demo_hide_recover.py --example cat
```

Bundled weight: `checkpoints/hinet_full_canvas_seed0.pt` (formal Full-Canvas seed 0, ~9.4 MB).

---

## 3. Optional: full pipeline downloads

Skip this section if you only want hide/recover.

### 3.1 Open-source code (clone into these exact folders)

```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git third_party/grounded-sam-2
git clone https://github.com/TencentARC/BrushNet.git third_party/BrushNet
```

Install their Python packages **from those folders**:

```bash
pip install -e "third_party/grounded-sam-2[notebooks]"
pip install --no-build-isolation -e third_party/grounded-sam-2/grounding_dino
# BrushNet: put its src on PYTHONPATH or `pip install -e third_party/BrushNet`
```

Grounded-SAM-2 may need `transformers==4.33.2`. BrushNet typically needs `transformers==4.38.2` and `accelerate==0.20.3`. If both fail in one env, use two conda envs (`stego-seg` / `stego-gen`) as in the paper experiments.

### 3.2 Weights (download into these exact folders)

| File | Put it here | Where to get it |
|---|---|---|
| GroundingDINO Swin-T | `checkpoints/grounded_sam2/groundingdino_swint_ogc.pth` | [GroundingDINO release](https://github.com/IDEA-Research/GroundingDINO/releases) `groundingdino_swint_ogc.pth` (~662 MB) |
| SAM 2.1 Hiera-L | `checkpoints/grounded_sam2/sam2.1_hiera_large.pt` | [SAM 2 checkpoints](https://github.com/facebookresearch/sam2) `sam2.1_hiera_large.pt` (~857 MB) |
| Stable Diffusion 1.5 | `checkpoints/brushnet/stable-diffusion-v1-5/` | Hugging Face [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) (fp16 is enough) |
| BrushNet (segmentation) | `checkpoints/brushnet/segmentation_mask_brushnet_ckpt/` | [BrushNet](https://github.com/TencentARC/BrushNet) / Hugging Face `TencentARC/BrushNet` segmentation-mask checkpoint |

Example:

```bash
mkdir -p checkpoints/grounded_sam2 checkpoints/brushnet
# then move the downloaded files to the paths in the table
```

Paths are also listed in `configs/paths.yaml`.

### 3.3 Run e2e

```bash
python scripts/demo_e2e.py --image examples/cat/original_full.png --class-name cat
```

Optional click (pixel `x,y` on the original image):

```bash
python scripts/demo_e2e.py --image examples/dog/original_full.png --class-name dog --click 320,200
```

---

## 4. Optional: LVIS / COCO for paper-scale evaluation

Not required for either demo.

| Dataset | Suggested location | Source |
|---|---|---|
| COCO 2017 val images | `data/coco2017/val2017/` | [COCO download](https://cocodataset.org/#download) |
| LVIS v1 val annotations | `data/lvis/annotations/lvis_v1_val.json` | [LVIS](https://www.lvisdataset.org/dataset) |

The three images under `examples/` are small COCO/LVIS-derived samples (CC BY 4.0) so you can run the demos without downloading the full datasets.

---

## Repository layout

```text
stegoanything/          # method package
  models/               # HiNet-style invertible hide/reveal
  hiding/               # checkpoint loader
  pipeline/             # 256 canvas, locator, cover, e2e
  generation/           # BrushNet adapter (imports third_party/BrushNet)
  segmentation/         # Grounded-SAM-2 adapter
  selection/            # text / optional-click policy
  metrics/
scripts/
  demo_hide_recover.py  # guaranteed demo
  demo_e2e.py           # full pipeline
examples/               # cat, dog, backpack
checkpoints/            # bundled HiNet weight only
third_party/            # clone GSAM2 + BrushNet here
configs/paths.yaml
```

---

## Method notes

- Canvas is **256×256**, letterboxed, RGB in `[-1, 1]` as `uint8 / 127.5 - 1`.
- Deployment reverse is **always** `recover_from_stego(stego, z=0)`.
- Mask is used for secret construction and restoration metrics, not as a network input.
- Formal LVIS three-seed numbers in the paper use this same hide/recover protocol.

## Citation

If you use this code, please cite the StegoAnything paper (title / venue TBD).

Related components: Grounded-SAM-2, BrushNet, Stable Diffusion 1.5, and the HiNet invertible-hiding formulation.
