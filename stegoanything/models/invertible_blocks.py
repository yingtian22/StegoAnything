"""Lightweight additive invertible coupling blocks."""

from __future__ import annotations

import torch
import torch.nn as nn


class DenseConvNet(nn.Module):
    """3-layer conv subnet; spatial size unchanged."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 64) -> None:
        super().__init__()
        h = hidden_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, h, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(h, h, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(h, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdditiveCouplingBlock(nn.Module):
    """Split-channel additive coupling: y1=x1+F(x2), y2=x2+G(y1)."""

    def __init__(self, channels: int, split_len: int, hidden_channels: int = 64) -> None:
        super().__init__()
        if split_len <= 0 or split_len >= channels:
            raise ValueError(f"split_len must be in (0, {channels}), got {split_len}")
        self.split_len = split_len
        c1, c2 = split_len, channels - split_len
        self.f = DenseConvNet(c2, c1, hidden_channels=hidden_channels)
        self.g = DenseConvNet(c1, c2, hidden_channels=hidden_channels)

    def forward(self, x: torch.Tensor, rev: bool = False) -> torch.Tensor:
        x1 = x[:, : self.split_len]
        x2 = x[:, self.split_len :]
        if not rev:
            y1 = x1 + self.f(x2)
            y2 = x2 + self.g(y1)
        else:
            y1, y2 = x1, x2
            x2 = y2 - self.g(y1)
            x1 = y1 - self.f(x2)
            return torch.cat([x1, x2], dim=1)
        return torch.cat([y1, y2], dim=1)


class InvSequential(nn.Module):
    """Stack of invertible coupling blocks."""

    def __init__(
        self,
        channels: int = 24,
        split_len: int = 12,
        num_blocks: int = 8,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            AdditiveCouplingBlock(channels, split_len, hidden_channels=hidden_channels)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor, rev: bool = False) -> torch.Tensor:
        if not rev:
            for block in self.blocks:
                x = block(x, rev=False)
            return x
        for block in reversed(self.blocks):
            x = block(x, rev=True)
        return x


def _self_test() -> None:
    print("Invertible blocks self-test")
    print("-" * 40)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = InvSequential(channels=24, split_len=12, num_blocks=4, hidden_channels=32).to(device)

    x = torch.randn(2, 24, 64, 64, device=device)
    y = net(x, rev=False)
    assert y.shape == x.shape
    x_rec = net(y, rev=True)
    err = (x - x_rec).abs().max().item()
    assert err < 1e-4, f"invertibility error too large: {err}"

    x.requires_grad_(True)
    y = net(x, rev=False)
    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None

    print(f"  [OK] shape preserved {tuple(y.shape)}")
    print(f"  [OK] max invertibility error {err:.2e}")
    print(f"  [OK] backward pass")
    print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()
