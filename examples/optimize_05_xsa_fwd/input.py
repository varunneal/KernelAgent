import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: per-head vector rejection
#   Z[b, h, t, :] = Y[b,h,t,:] - dot(Y, Vn) * Vn   where Vn = V / ||V||
#   Input/output shape: (B, H, T, Dh)  laid out contiguously on last dim
# ---------------------------------------------------------------------------

@triton.jit
def _rejection_kernel(
    Y_ptr, V_ptr, Z_ptr,
    stride_y_bht,  # stride to advance one (b,h,t) row = Dh for contiguous
    stride_v_bht,
    stride_z_bht,
    Dh,
    num_rows,       # B * H * T
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= num_rows:
        return

    offs = tl.arange(0, BLOCK_D)
    mask = offs < Dh

    y = tl.load(Y_ptr + row * stride_y_bht + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(V_ptr + row * stride_v_bht + offs, mask=mask, other=0.0).to(tl.float32)

    vsq = tl.sum(v * v)
    dot = tl.sum(y * v)
    scale = dot / tl.maximum(vsq, 1e-24)
    z = y - scale * v

    tl.store(Z_ptr + row * stride_z_bht + offs, z.to(y.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Kernel 2: batched matmul  (B*T, D) @ (D, D) -> (B*T, D)
#   Standard tiled GEMM.  A = Z_flat, B_mat = Wo, C = out
# ---------------------------------------------------------------------------

@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in tl.range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=a_mask, other=0.0,
        ).to(tl.float32)
        b = tl.load(
            B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=b_mask, other=0.0,
        ).to(tl.float32)

        acc += tl.dot(a, b)

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(C_ptr.dtype.element_ty),
        mask=c_mask,
    )


def kernel_function(Y: torch.Tensor, V: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Fused XSA rejection + output projection.

    Args:
        Y: (B, H, T, Dh) - SDPA output
        V: (B, H, T, Dh) - value vectors
        weight: (D, D)    - output projection Wo

    Returns:
        out: (B, T, D)
    """
    assert Y.shape == V.shape and Y.ndim == 4
    assert Y.is_cuda and V.is_cuda and weight.is_cuda
    B, H, T, Dh = Y.shape
    D = H * Dh
    assert weight.shape == (D, D)

    # --- Step 1: rejection ---
    # Flatten to (B*H*T, Dh) for the rejection kernel
    Y_flat = Y.contiguous().view(-1, Dh)
    V_flat = V.contiguous().view(-1, Dh)
    Z_flat_heads = torch.empty_like(Y_flat)

    num_rows = B * H * T
    BLOCK_D = triton.next_power_of_2(Dh)

    _rejection_kernel[(num_rows,)](
        Y_flat, V_flat, Z_flat_heads,
        stride_y_bht=Y_flat.stride(0),
        stride_v_bht=V_flat.stride(0),
        stride_z_bht=Z_flat_heads.stride(0),
        Dh=Dh,
        num_rows=num_rows,
        BLOCK_D=BLOCK_D,
        num_warps=4 if BLOCK_D <= 256 else 8,
        num_stages=1,
    )

    # Reshape: (B, H, T, Dh) -> (B, T, H, Dh) -> (B*T, D)
    Z_bhTd = Z_flat_heads.view(B, H, T, Dh)
    Z_flat = Z_bhTd.transpose(1, 2).reshape(B * T, D).contiguous()

    # --- Step 2: matmul  (B*T, D) @ (D, D) -> (B*T, D) ---
    M_dim = B * T
    N_dim = D
    K_dim = D
    out_flat = torch.empty(M_dim, N_dim, device=Y.device, dtype=Y.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64

    grid = (triton.cdiv(M_dim, BLOCK_M), triton.cdiv(N_dim, BLOCK_N))
    _matmul_kernel[grid](
        Z_flat, weight, out_flat,
        M_dim, N_dim, K_dim,
        Z_flat.stride(0), Z_flat.stride(1),
        weight.stride(0), weight.stride(1),
        out_flat.stride(0), out_flat.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    return out_flat.view(B, T, D)
