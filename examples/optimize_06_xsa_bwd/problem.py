import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Backward pass of fused XSA rejection + output projection.

    Forward:
      Vn = normalize(V, dim=-1)
      Z  = Y - dot(Y, Vn) * Vn
      out = Z.transpose(1,2).reshape(B,T,D) @ Wo

    This model computes the gradients dY, dV, dWo given grad_out.
    We use autograd to get reference gradients.
    """

    def __init__(self, D: int):
        super().__init__()
        self.D = D

    def forward(
        self,
        grad_out: torch.Tensor,
        Y: torch.Tensor,
        V: torch.Tensor,
        Wo: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute backward pass, returning (dY, dV, dWo)."""
        Y = Y.detach().requires_grad_(True)
        V = V.detach().requires_grad_(True)
        Wo = Wo.detach().requires_grad_(True)

        B, H, T, Dh = Y.shape
        D = H * Dh

        Vn = F.normalize(V, dim=-1)
        Z = Y - (Y * Vn).sum(dim=-1, keepdim=True) * Vn
        Z_flat = Z.transpose(1, 2).reshape(B, T, D)
        out = Z_flat @ Wo

        out.backward(grad_out)

        return Y.grad, V.grad, Wo.grad


B = 4
T = 2048
H = 32
Dh = 128
D = H * Dh  # 4096


def get_inputs():
    grad_out = torch.randn(B, T, D)
    Y = torch.randn(B, H, T, Dh)
    V = torch.randn(B, H, T, Dh)
    Wo = torch.randn(D, D) * (D ** -0.5)
    return [grad_out, Y, V, Wo]


def get_init_inputs():
    return [D]
