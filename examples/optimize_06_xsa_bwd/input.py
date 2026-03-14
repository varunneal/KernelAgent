import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Backward of: out = reject(Y, V).transpose(1,2).reshape(B*T, D) @ Wo
#
# Given grad_out (B, T, D), Y (B, H, T, Dh), V (B, H, T, Dh), Wo (D, D):
#
#   dWo      = Z_flat^T @ grad_out              (D, D)
#   dZ_flat  = grad_out @ Wo^T                  (B*T, D)
#   dZ       = dZ_flat.reshape(B,T,H,Dh).transpose(1,2)   (B, H, T, Dh)
#   dY, dV   = rejection_backward(dZ, Y, V)     (both B, H, T, Dh)
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


# ---------------------------------------------------------------------------
# Rejection forward recompute (needed for dWo = Z^T @ grad_out)
# ---------------------------------------------------------------------------

@triton.jit
def _rejection_fwd_kernel(
    Y_ptr, V_ptr, Z_ptr,
    stride_row, Dh, num_rows,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= num_rows:
        return
    offs = tl.arange(0, BLOCK_D)
    mask = offs < Dh
    base = row * stride_row

    y = tl.load(Y_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(V_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    vsq = tl.sum(v * v)
    dot = tl.sum(y * v)
    scale = dot / tl.maximum(vsq, 1e-24)
    z = y - scale * v

    tl.store(Z_ptr + base + offs, z.to(y.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Rejection backward kernel (per-row)
#
# Forward: Vn = V / ||V||,  s = dot(Y, Vn),  Z = Y - s * Vn
#
# Derivation:
#   dY = dZ - dot(dZ, Vn) * Vn
#   dVn = -s * dZ - dot(dZ, Vn) * Y
#   dV = (dVn - Vn * dot(dVn, Vn)) / ||V||    (through normalize)
# ---------------------------------------------------------------------------

@triton.jit
def _rejection_bwd_kernel(
    dZ_ptr, Y_ptr, V_ptr,
    dY_ptr, dV_ptr,
    stride_row, Dh, num_rows,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= num_rows:
        return

    offs = tl.arange(0, BLOCK_D)
    mask = offs < Dh
    base = row * stride_row

    dz = tl.load(dZ_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(V_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    # Recompute forward quantities
    v_norm_sq = tl.sum(v * v)
    inv_norm = tl.math.rsqrt(v_norm_sq + 1e-12)
    vn = v * inv_norm
    s = tl.sum(y * vn)

    # dY = dZ - dot(dZ, Vn) * Vn
    dz_dot_vn = tl.sum(dz * vn)
    dy = dz - dz_dot_vn * vn

    # dVn = -s * dZ - dot(dZ, Vn) * Y
    dvn = -s * dz - dz_dot_vn * y

    # dV = (dVn - Vn * dot(dVn, Vn)) / ||V||
    dvn_dot_vn = tl.sum(dvn * vn)
    dv = (dvn - vn * dvn_dot_vn) * inv_norm

    tl.store(dY_ptr + base + offs, dy.to(dz.dtype), mask=mask)
    tl.store(dV_ptr + base + offs, dv.to(dz.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Python helpers
# ---------------------------------------------------------------------------

def _triton_matmul(A, B, out, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64):
    M, K = A.shape
    _, N = B.shape
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_kernel[grid](
        A, B, out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )


def _run_rejection_fwd(Y_flat, V_flat, Z_out, num_rows, Dh, BLOCK_D):
    _rejection_fwd_kernel[(num_rows,)](
        Y_flat, V_flat, Z_out,
        stride_row=Dh, Dh=Dh, num_rows=num_rows,
        BLOCK_D=BLOCK_D,
        num_warps=4 if BLOCK_D <= 256 else 8,
        num_stages=1,
    )


def kernel_function(
    grad_out: torch.Tensor,
    Y: torch.Tensor,
    V: torch.Tensor,
    Wo: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward of fused XSA rejection + output projection.

    Args:
        grad_out: (B, T, D)
        Y: (B, H, T, Dh)
        V: (B, H, T, Dh)
        Wo: (D, D)

    Returns:
        dY: (B, H, T, Dh)
        dV: (B, H, T, Dh)
        dWo: (D, D)
    """
    B, H, T, Dh = Y.shape
    D = H * Dh

    grad_out_2d = grad_out.reshape(B * T, D).contiguous()
    Y_flat = Y.contiguous().view(-1, Dh)
    V_flat = V.contiguous().view(-1, Dh)
    num_rows = B * H * T
    BLOCK_D = triton.next_power_of_2(Dh)

    # --- Recompute Z for dWo (avoids saving from forward) ---
    Z_heads = torch.empty_like(Y_flat)
    _run_rejection_fwd(Y_flat, V_flat, Z_heads, num_rows, Dh, BLOCK_D)
    Z_flat = Z_heads.view(B, H, T, Dh).transpose(1, 2).reshape(B * T, D).contiguous()

    # --- dWo = Z_flat^T @ grad_out  =>  (D, B*T) @ (B*T, D) ---
    dWo = torch.empty(D, D, device=Y.device, dtype=Y.dtype)
    _triton_matmul(Z_flat.T.contiguous(), grad_out_2d, dWo)

    # --- dZ_flat = grad_out @ Wo^T  =>  (B*T, D) @ (D, D) ---
    dZ_flat = torch.empty_like(grad_out_2d)
    _triton_matmul(grad_out_2d, Wo.T.contiguous(), dZ_flat)

    # Reshape: (B*T, D) -> (B, T, H, Dh) -> (B, H, T, Dh) -> flat
    dZ = dZ_flat.view(B, T, H, Dh).transpose(1, 2).contiguous().view(-1, Dh)

    # --- Rejection backward: dY, dV ---
    dY_flat = torch.empty_like(Y_flat)
    dV_flat = torch.empty_like(V_flat)

    _rejection_bwd_kernel[(num_rows,)](
        dZ, Y_flat, V_flat,
        dY_flat, dV_flat,
        stride_row=Dh, Dh=Dh, num_rows=num_rows,
        BLOCK_D=BLOCK_D,
        num_warps=4 if BLOCK_D <= 256 else 8,
        num_stages=1,
    )

    return (
        dY_flat.view(B, H, T, Dh),
        dV_flat.view(B, H, T, Dh),
        dWo,
    )
