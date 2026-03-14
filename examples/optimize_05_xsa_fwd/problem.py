import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Fused XSA rejection + output projection (forward only).

    Given Y from SDPA and V (both B, H, T, Dh):
      1. Vn = normalize(V, dim=-1)
      2. Z = Y - dot(Y, Vn) * Vn          (vector rejection, per-head)
      3. out = Z.transpose(1,2).reshape(B, T, D) @ Wo   (output projection)

    Returns out: (B, T, D)
    """

    def __init__(self, D: int):
        super().__init__()
        self.Wo = nn.Parameter(torch.randn(D, D) * (D ** -0.5))

    def forward(self, Y: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, H, T, Dh = Y.shape
        D = H * Dh

        # Per-head vector rejection
        Vn = F.normalize(V, dim=-1)
        Z = Y - (Y * Vn).sum(dim=-1, keepdim=True) * Vn

        # Output projection
        Z_flat = Z.transpose(1, 2).reshape(B, T, D)
        out = Z_flat @ self.Wo
        return out


# Typical transformer config
B = 4
T = 2048
H = 32
Dh = 128
D = H * Dh  # 4096


def get_inputs():
    Y = torch.randn(B, H, T, Dh)
    V = torch.randn(B, H, T, Dh)
    return [Y, V]


def get_init_inputs():
    return [D]
