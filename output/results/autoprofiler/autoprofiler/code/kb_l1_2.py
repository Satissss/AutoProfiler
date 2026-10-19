import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _matmul_relu_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Basic 2D block ID
    pid = tl.program_id(axis=0)
    pid_m = pid // tl.cdiv(N, BLOCK_N)
    pid_n = pid % tl.cdiv(N, BLOCK_N)

    # Block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointer arithmetic for A and B (both fp16)
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load A and B blocks as fp16
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Apply ReLU and store
    c = tl.maximum(acc, 0.0)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c.to(tl.float32), mask=mask)

def _matmul_relu_launcher(A, B):
    assert A.is_contiguous() and B.is_contiguous()
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    # Cast to float16 *before* kernel launch — safe and required for HMMA
    A_fp16 = A.to(torch.float16)
    B_fp16 = B.to(torch.float16)
    C_fp16 = torch.empty((M, N), device=A.device, dtype=torch.float16)
    
    # Simple, robust tile sizes known to work on most GPUs
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    grid = lambda META: (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    _matmul_relu_kernel[grid](
        A_fp16, B_fp16, C_fp16,
        M, N, K,
        A_fp16.stride(0), A_fp16.stride(1),
        B_fp16.stride(0), B_fp16.stride(1),
        C_fp16.stride(0), C_fp16.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return C_fp16.to(torch.float32)

class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    and applies ReLU via Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication and applies ReLU via Triton.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N) after ReLU.
        """
        return _matmul_relu_launcher(A, B)