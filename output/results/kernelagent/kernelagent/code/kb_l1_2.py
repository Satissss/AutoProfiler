import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _abs_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.abs(x)
    tl.store(out_ptr + offs, y, mask=mask)

def _abs_launcher(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n_elements, BLOCK),)
    _abs_kernel[grid](x, out, n_elements, BLOCK=BLOCK)
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        C = torch.matmul(A, B)
        return _abs_launcher(C)