"""Correctness test for fused XSA rejection + output projection (forward)."""

import sys
import torch
from kernel import kernel_function
from problem import Model, get_inputs, get_init_inputs


def test_kernel():
    device = "cuda"
    dtype = torch.bfloat16

    model = Model(*get_init_inputs()).to(device).to(dtype)
    inputs = [
        x.to(device).to(dtype) if isinstance(x, torch.Tensor) and x.is_floating_point()
        else (x.to(device) if isinstance(x, torch.Tensor) else x)
        for x in get_inputs()
    ]

    with torch.no_grad():
        ref_output = model(*inputs)

    # kernel_function(Y, V, Wo) — needs the weight from the model
    Wo = model.Wo.data
    kernel_output = kernel_function(inputs[0], inputs[1], Wo)

    if ref_output.dtype != kernel_output.dtype:
        kernel_output = kernel_output.to(ref_output.dtype)

    if torch.allclose(ref_output, kernel_output, rtol=1e-2, atol=1e-2):
        print("PASS")
        return True
    else:
        max_diff = (ref_output - kernel_output).abs().max().item()
        print(f"FAIL: max difference = {max_diff}")
        return False


if __name__ == "__main__":
    success = test_kernel()
    sys.exit(0 if success else 1)
