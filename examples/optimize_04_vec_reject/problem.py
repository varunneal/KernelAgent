import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Vector rejection: project Y onto the plane orthogonal to V.
    Vn = normalize(V, dim=-1)
    Z = Y - (Y . Vn) * Vn
    """

    def __init__(self):
        super().__init__()

    def forward(self, Y: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        Vn = F.normalize(V, dim=-1)
        Z = Y - (Y * Vn).sum(dim=-1, keepdim=True) * Vn
        return Z


M = 4096
D = 1024


def get_inputs():
    return [torch.randn(M, D), torch.randn(M, D)]


def get_init_inputs():
    return []
