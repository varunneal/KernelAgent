"""Correctness test for XSA rejection + output projection backward kernel."""

import sys
import torch
import torch.nn.functional as F
from kernel import kernel_function
from problem import B, T, H, Dh, D


def test_kernel():
    device = "cuda"
    dtype = torch.bfloat16

    # Create inputs
    Y = torch.randn(B, H, T, Dh, device=device, dtype=dtype)
    V = torch.randn(B, H, T, Dh, device=device, dtype=dtype)
    Wo = torch.randn(D, D, device=device, dtype=dtype) * (D ** -0.5)
    grad_out = torch.randn(B, T, D, device=device, dtype=dtype)

    # Reference backward using autograd
    Y_ref = Y.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    Wo_ref = Wo.detach().clone().requires_grad_(True)

    Vn = F.normalize(V_ref, dim=-1)
    Z = Y_ref - (Y_ref * Vn).sum(dim=-1, keepdim=True) * Vn
    Z_flat = Z.transpose(1, 2).reshape(B, T, D)
    out = Z_flat @ Wo_ref
    out.backward(grad_out)

    ref_dY = Y_ref.grad
    ref_dV = V_ref.grad
    ref_dWo = Wo_ref.grad

    # Kernel backward
    dY, dV, dWo = kernel_function(grad_out, Y, V, Wo)

    # Compare each gradient
    ok = True
    for name, ref, got in [("dY", ref_dY, dY), ("dV", ref_dV, dV), ("dWo", ref_dWo, dWo)]:
        if ref.dtype != got.dtype:
            got = got.to(ref.dtype)
        if torch.allclose(ref, got, rtol=1e-2, atol=1e-2):
            continue
        max_diff = (ref - got).abs().max().item()
        print(f"FAIL: {name} max difference = {max_diff}")
        ok = False

    if ok:
        print("PASS")
    return ok


if __name__ == "__main__":
    success = test_kernel()
    sys.exit(0 if success else 1)
