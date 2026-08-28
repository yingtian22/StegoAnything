"""Pure PyTorch Haar discrete wavelet transform (DWT / IWT)."""

from __future__ import annotations

import torch
import torch.nn as nn


def _haar_dwt(x: torch.Tensor) -> torch.Tensor:
    """Haar DWT: (B, C, H, W) -> (B, 4*C, H/2, W/2)."""
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_ll = x1 + x2 + x3 + x4
    x_hl = -x1 - x2 + x3 + x4
    x_lh = -x1 + x2 - x3 + x4
    x_hh = x1 - x2 - x3 + x4
    return torch.cat((x_ll, x_hl, x_lh, x_hh), dim=1)


def _haar_iwt(x: torch.Tensor) -> torch.Tensor:
    """Haar IWT: (B, 4*C, H/2, W/2) -> (B, C, H, W)."""
    r = 2
    in_batch, in_channel, in_height, in_width = x.shape
    out_channel = in_channel // (r * r)
    out_height, out_width = r * in_height, r * in_width

    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel : out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2 : out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3 : out_channel * 4, :, :] / 2

    h = torch.zeros(
        (in_batch, out_channel, out_height, out_width),
        device=x.device,
        dtype=x.dtype,
    )
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h


class HaarDWT(nn.Module):
    """Haar DWT for RGB images: (B, 3, H, W) -> (B, 12, H/2, W/2)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 2 != 0 or x.shape[-2] % 2 != 0:
            raise ValueError(f"DWT requires even H,W, got {tuple(x.shape)}")
        return _haar_dwt(x)


class HaarIWT(nn.Module):
    """Haar IWT: (B, 12, H/2, W/2) -> (B, 3, H, W)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _haar_iwt(x)


def _self_test() -> None:
    print("HaarDWT / HaarIWT self-test")
    print("-" * 40)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dwt = HaarDWT().to(device)
    iwt = HaarIWT().to(device)

    x = torch.randn(2, 3, 128, 128, device=device)
    y = dwt(x)
    assert y.shape == (2, 12, 64, 64), f"unexpected DWT shape {tuple(y.shape)}"
    x_rec = iwt(y)
    assert x_rec.shape == (2, 3, 128, 128), f"unexpected IWT shape {tuple(x_rec.shape)}"

    err = (x - x_rec).abs().max().item()
    assert err < 1e-5, f"reconstruction error too large: {err}"
    print(f"  [OK] DWT shape {tuple(y.shape)}")
    print(f"  [OK] IWT shape {tuple(x_rec.shape)}")
    print(f"  [OK] max reconstruction error {err:.2e}")
    print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()
