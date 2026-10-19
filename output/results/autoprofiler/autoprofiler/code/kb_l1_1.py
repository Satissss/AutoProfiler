import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _matmul_add_epsilon_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    FP16: tl.constexpr, TF32: tl.constexpr
):
    # 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets (scalar, constexpr-safe)
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    offs_k = 0
    
    # Block pointers — offsets must be scalars, so use offs_m, offs_n directly
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(offs_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0)
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, offs_n),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1)
    )
    
    # Allocate accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load tiles
        a_tile = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b_tile = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        # Cast if needed
        if FP16:
            a_tile = a_tile.to(tl.float16)
            b_tile = b_tile.to(tl.float16)
        
        # Compute dot
        if TF32:
            acc += tl.dot(a_tile, b_tile, input_precision="tf32", out_dtype=tl.float32)
        else:
            acc += tl.dot(a_tile, b_tile)
        
        # Advance block pointers
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
    
    # Convert and add epsilon
    c = acc.to(c_ptr.dtype.element_ty) + 1e-10
    
    # Store output
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def _matmul_add_epsilon_launcher(A, B):
    A = A.contiguous()
    B = B.contiguous()
    assert A.shape[0] == A.shape[1] and B.shape[0] == B.shape[1] and A.shape[0] == B.shape[0], "Square matrices only"
    N = A.shape[0]
    C = torch.empty((N, N), device=A.device, dtype=A.dtype)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    FP16 = A.dtype == torch.float16
    TF32 = A.dtype == torch.float32
    
    grid = lambda META: (triton.cdiv(N, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    _matmul_add_epsilon_kernel[grid](
        A, B, C,
        N, N, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        FP16, TF32
    )
    return C

class ModelNew(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    and then adds a tiny epsilon elementwise via Triton.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication and adds epsilon via Triton.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N) with epsilon added.
        """
        return _matmul_add_epsilon_launcher(A, B)