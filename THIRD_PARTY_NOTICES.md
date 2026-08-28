# Third-party notices

This repository contains **StegoAnything** code only.

The optional full pipeline uses the following official projects. Download them yourself; they are not copied here.

| Project | Role | License (upstream) |
|---|---|---|
| [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) | Open-vocabulary localization | See upstream `LICENSE` / `LICENSE_sam2` / `LICENSE_groundingdino` |
| [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) (inside Grounded-SAM-2) | Text-box detector | Apache-2.0 |
| [SAM 2](https://github.com/facebookresearch/sam2) | Mask decoder | Apache-2.0 / BSD-3 (see upstream) |
| [BrushNet](https://github.com/TencentARC/BrushNet) | Same-class semantic cover | Apache-2.0 |
| [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) | BrushNet base | CreativeML Open RAIL-M |
| [COCO](https://cocodataset.org/) / [LVIS](https://www.lvisdataset.org/) | Demo example photos | CC BY 4.0 |

The hiding network is a HiNet-style invertible reimplementation used by StegoAnything (wavelet + coupling blocks). It is **not** a copy of the official ICCV 2021 HiNet training code or checkpoint.

Paper citations for methods we compare against (not included as code): HiNet, IICNet, DeepMIH, UDH, DDH, StegFormer, StegTransX.
