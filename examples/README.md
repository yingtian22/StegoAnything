# Example images

Three 256×256 canvas tuples for the hide/recover demo, plus the original full-resolution image for the optional end-to-end demo.

| Folder | Class prompt | Files |
|---|---|---|
| `cat/` | `a cat.` | `original_canvas.png`, `brushnet_cover_canvas.png`, `mask_canvas.png`, `secret_canvas.png`, `original_full.png` |
| `dog/` | `a dog.` | same |
| `backpack/` | `a backpack.` | same |

These images are derived from [COCO](https://cocodataset.org/) / [LVIS](https://www.lvisdataset.org/) validation photos (CC BY 4.0). They are included only as a small runnable demo, not as a dataset redistribution.

`brushnet_cover_canvas.png` is a same-class semantic replacement already generated offline, so `scripts/demo_hide_recover.py` can run without BrushNet.
