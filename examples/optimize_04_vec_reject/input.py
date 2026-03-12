import torch
import triton
import triton.language as tl


@triton.jit
def _vec_reject_kernel(
    Y_ptr,
    V_ptr,
    Z_ptr,
    M,
    D,
    stride_ym: tl.constexpr,
    stride_vm: tl.constexpr,
    stride_zm: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused vector rejection: Z = Y - dot(Y, Vn) * Vn where Vn = V / ||V||.

    Each program handles one row. We iterate over D in BLOCK_D tiles to:
      1) accumulate ||V||^2 and dot(Y, V)
      2) normalize and write Z = Y - (dot / norm) * (V / norm)
    """
    row = tl.program_id(0)

    # --- Pass 1: compute ||V||^2 and dot(Y, V) ---
    v_sq_acc = tl.zeros([], dtype=tl.float32)
    yv_dot_acc = tl.zeros([], dtype=tl.float32)

    for d_start in tl.range(0, D, BLOCK_D):
        offs = d_start + tl.arange(0, BLOCK_D)
        mask = offs < D

        y_vals = tl.load(Y_ptr + row * stride_ym + offs, mask=mask, other=0.0).to(tl.float32)
        v_vals = tl.load(V_ptr + row * stride_vm + offs, mask=mask, other=0.0).to(tl.float32)

        v_sq_acc += tl.sum(v_vals * v_vals)
        yv_dot_acc += tl.sum(y_vals * v_vals)

    # inv_norm = 1 / ||V||
    inv_norm = tl.math.rsqrt(v_sq_acc + 1e-12)
    # dot(Y, Vn) = dot(Y, V) / ||V||
    dot_y_vn = yv_dot_acc * inv_norm

    # --- Pass 2: Z = Y - dot(Y, Vn) * Vn ---
    for d_start in tl.range(0, D, BLOCK_D):
        offs = d_start + tl.arange(0, BLOCK_D)
        mask = offs < D

        y_vals = tl.load(Y_ptr + row * stride_ym + offs, mask=mask, other=0.0).to(tl.float32)
        v_vals = tl.load(V_ptr + row * stride_vm + offs, mask=mask, other=0.0).to(tl.float32)

        vn_vals = v_vals * inv_norm
        z_vals = y_vals - dot_y_vn * vn_vals

        tl.store(Z_ptr + row * stride_zm + offs, z_vals.to(y_vals.dtype), mask=mask)


def kernel_function(Y: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Fused vector rejection: Z = Y - dot(Y, normalize(V)) * normalize(V).

    Args:
        Y: [M, D] tensor
        V: [M, D] tensor

    Returns:
        Z: [M, D] tensor
    """
    assert Y.shape == V.shape and Y.ndim == 2
    assert Y.is_cuda and V.is_cuda
    M, D = Y.shape

    Z = torch.empty_like(Y)

    BLOCK_D = triton.next_power_of_2(D) if D <= 4096 else 1024

    grid = (M,)
    _vec_reject_kernel[grid](
        Y, V, Z,
        M, D,
        stride_ym=Y.stride(0),
        stride_vm=V.stride(0),
        stride_zm=Z.stride(0),
        BLOCK_D=BLOCK_D,
        num_warps=4 if BLOCK_D <= 1024 else 8,
        num_stages=1,
    )

    return Z
