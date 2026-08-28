"""HiNet-style wavelet-domain invertible secret-in-cover hiding."""

from __future__ import annotations

import torch
import torch.nn as nn

from stegoanything.models.invertible_blocks import InvSequential
from stegoanything.models.wavelet import HaarDWT, HaarIWT


class HiNetStyleSecretHidingModel(nn.Module):
    """
    Hide RGB secret in RGB cover via Haar DWT + invertible coupling.

    Wavelet channels: 3 RGB -> 12 coeffs per image; concat -> 24 channels.
    """

    def __init__(
        self,
        num_blocks: int = 8,
        channels: int = 24,
        split_len: int = 12,
        hidden_channels: int = 64,
        wavelet_channels: int = 12,
    ) -> None:
        super().__init__()
        self.wavelet_channels = wavelet_channels
        self.dwt = HaarDWT()
        self.iwt = HaarIWT()
        self.inn = InvSequential(
            channels=channels,
            split_len=split_len,
            num_blocks=num_blocks,
            hidden_channels=hidden_channels,
        )

    def recover_from_stego(self, stego: torch.Tensor, latent_z: torch.Tensor | None = None) -> torch.Tensor:
        """Reveal secret from (possibly attacked) stego without re-encoding."""
        return self._reveal(stego, latent_z)

    def _reveal(self, stego: torch.Tensor, latent_z: torch.Tensor | None = None) -> torch.Tensor:
        stego_w = self.dwt(stego)
        if latent_z is None:
            latent_z = torch.zeros(
                stego_w.shape[0],
                self.wavelet_channels,
                stego_w.shape[2],
                stego_w.shape[3],
                device=stego_w.device,
                dtype=stego_w.dtype,
            )
        y_rev = torch.cat([stego_w, latent_z], dim=1)
        x_rec = self.inn(y_rev, rev=True)
        secret_rec_w = x_rec[:, self.wavelet_channels :]
        return torch.clamp(self.iwt(secret_rec_w), -1.0, 1.0)

    def forward(self, cover: torch.Tensor, secret: torch.Tensor) -> dict[str, torch.Tensor]:
        cover_w = self.dwt(cover)
        secret_w = self.dwt(secret)
        x = torch.cat([cover_w, secret_w], dim=1)
        y = self.inn(x, rev=False)

        stego_w = y[:, : self.wavelet_channels]
        latent_z = y[:, self.wavelet_channels :]
        stego = torch.clamp(self.iwt(stego_w), -1.0, 1.0)
        recovered = self._reveal(stego, torch.zeros_like(latent_z))

        return {
            "stego": stego,
            "recovered": recovered,
            "latent_z": latent_z,
            "stego_w": stego_w,
        }


def _self_test() -> None:
    print("HiNetStyleSecretHidingModel self-test")
    print("-" * 40)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HiNetStyleSecretHidingModel(num_blocks=8, hidden_channels=64).to(device)
    b, h, w = 2, 128, 128
    cover = torch.randn(b, 3, h, w, device=device)
    secret = torch.randn(b, 3, h, w, device=device)

    cover.requires_grad_(True)
    secret.requires_grad_(True)
    out = model(cover, secret)

    assert out["stego"].shape == (b, 3, h, w)
    assert out["recovered"].shape == (b, 3, h, w)
    assert out["latent_z"].shape == (b, 12, h // 2, w // 2)
    assert out["stego_w"].shape == (b, 12, h // 2, w // 2)
    assert out["stego"].min() >= -1.0 and out["stego"].max() <= 1.0
    assert out["recovered"].min() >= -1.0 and out["recovered"].max() <= 1.0

    loss = out["stego"].mean() + out["recovered"].mean()
    loss.backward()
    assert cover.grad is not None and secret.grad is not None

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [OK] stego {tuple(out['stego'].shape)}, recovered {tuple(out['recovered'].shape)}")
    print(f"  [OK] latent_z {tuple(out['latent_z'].shape)}")
    print(f"  [OK] range stego [{out['stego'].min():.3f}, {out['stego'].max():.3f}]")
    print(f"  [OK] params={n_params:,}")
    print(f"  [OK] backward pass")
    print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()
