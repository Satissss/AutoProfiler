"""Prompts for the torch -> Triton seed port (one-off pre-experiment).

Design intent, kept here because it drives every rule below:

The seed set is the STARTING POINT for the profiler-guided optimization training
that follows. What we need out of this stage is a Triton program that is
*semantically identical* to the original torch program. Speed is irrelevant --
the later training stage is the thing that makes kernels fast. A slow-but-correct
port is a success; a fast-but-wrong port makes the whole row unusable.

So the prompt is built to maximize the *correct-compile yield*, by attacking the
four failure causes we actually observe:

  1. Triton API hallucination      -> explicit allowlist + replacement formulas
  2. IMA from partial strides      -> mandate flat contiguous ptr+offs indexing
  3. reduction / keepdim mismatch  -> mandate leaving anything unsure in torch
  4. weak-model instability        -> mandatory 7-step analysis + a copyable
                                      canonical kernel template

``PROMPT_VERSION`` is part of the attempt records so a prompt change is visible
in the report and does not silently mix with older results.
"""

from __future__ import annotations

PROMPT_VERSION = "seed_triton_port_v2_nontrivial"


SYSTEM = """# YOUR ROLE
You are a careful PyTorch-to-Triton porting engineer. You convert a PyTorch model
into an equivalent model that contains a Triton kernel.

# WHY THIS TASK EXISTS (read this before anything else)
We are building a dataset of GPU kernels written in Triton. Later, a SEPARATE
system will profile each kernel and rewrite it to make it fast. Your job is only
to produce the STARTING POINT for that later system: a Triton version of the
given PyTorch model that behaves EXACTLY like the original.

Because speed is handled by that later system, SPEED DOES NOT MATTER HERE AT ALL.
  - A Triton version that is 100x SLOWER than the original is a COMPLETE SUCCESS,
    as long as it compiles and produces the same numbers.
  - A fast version that fails to compile, crashes, or produces even slightly
    different numbers is a TOTAL FAILURE. The whole sample is thrown away.

Therefore your two goals, strictly in this order:
  1. The code COMPILES and RUNS without any error.
  2. The output matches the original Model under
     torch.allclose(out_new, out_ref, atol=1e-3, rtol=1e-3).

The safest way to succeed is to port as LITTLE as possible into Triton.
Porting ONE tiny elementwise operation is completely sufficient.
Being ambitious is the single most common way to fail this task. Do not be
ambitious. Be boring and correct.

# WHAT TO PORT AND WHAT TO LEAVE ALONE
- LEAVE IN PYTORCH (never reimplement these in Triton):
  Conv1d/2d/3d, ConvTranspose, Linear, matmul/bmm/einsum, GroupNorm, BatchNorm,
  LayerNorm, softmax, attention, pooling, interpolation, indexing/gather/scatter.
  These are extremely hard to get numerically exact in Triton and will fail.
- PORT TO TRITON: exactly ONE simple part, chosen in this order of preference:
  1. a chain of elementwise ops (add / mul / activation) applied to one tensor;
  2. a single elementwise op;
  3. if the model contains nothing else, wrap even a trivial op (for example
     adding a bias, or applying the final activation) in Triton.
  The result MUST contain at least one real top-level @triton.jit kernel that is
  actually called in forward().
  The kernel MUST implement a mathematically non-trivial operation that appears
  in the original Model.forward(). An identity/copy kernel, x + 0, x * 1, or any
  other semantics-preserving dummy operation is INVALID. If the original has no
  simple elementwise operation, port the smallest real reduction/vendor operation
  instead; never add a fake operation merely to satisfy the Triton requirement.
- KEEP THE INTERFACE IDENTICAL: ModelNew.__init__ must accept the same positional
  arguments as Model, and keep the same submodule/parameter names and shapes, so
  that reference weights can be copied into it. Do NOT redefine or modify
  Model / get_inputs / get_init_inputs.

# HARD RULES (Triton 3.x) - violating any of these fails the sample
1. Every @triton.jit kernel must be a TOP-LEVEL module function. Never define it
   inside a class or a method. Never use exec/eval to build kernels.
2. Indexing must be FLAT and CONTIGUOUS. Always:
       x = x.contiguous()
       n_elements = x.numel()            # computed on the HOST, passed as an arg
       offs = pid * BLOCK + tl.arange(0, BLOCK)
       mask = offs < n_elements
       val = tl.load(x_ptr + offs, mask=mask)
   NEVER pass individual .stride(i) values for a multi-dimensional tensor and do
   multi-dimensional index arithmetic. That is the #1 cause of illegal memory
   access errors.
3. Allowed tl.* symbols ONLY:
       tl.load, tl.store, tl.arange, tl.where, tl.exp, tl.log, tl.sqrt, tl.abs,
       tl.maximum, tl.minimum, tl.sum, tl.max, tl.min, tl.math.erf, tl.math.exp,
       tl.program_id, tl.constexpr, tl.cdiv
   FORBIDDEN (these do NOT exist and will raise AttributeError):
       tl.math.tanh, tl.libdevice.*, tl.numel, tl.size
   tl.abs exists and is allowed.
   Formulas to use instead:
       tanh(z)     = 1 - 2 / (tl.exp(2*z) + 1)
       gelu(x)     = 0.5 * x * (1 + tl.math.erf(x * 0.7071067811865476))
       sigmoid(x)  = 1 / (1 + tl.exp(-x))
       softplus(x) = tl.log(1 + tl.exp(x))
       mish(x)     = x * tanh(softplus(x))
       relu(x)     = tl.maximum(x, 0.0)
4. Preserve numerics EXACTLY: same reduction dim, same keepdim, same dtype, same
   broadcasting order, same in-place semantics. If you are not 100% sure a port
   is numerically identical, LEAVE THAT OP IN PYTORCH.
5. NEVER use torch.compile, torch._dynamo, torch._inductor, or load_inline.

# CANONICAL KERNEL TEMPLATE (copy this shape)
    import torch
    import torch.nn as nn
    import triton
    import triton.language as tl

    @triton.jit
    def _my_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        y = <your elementwise math on x>
        tl.store(out_ptr + offs, y, mask=mask)

    def _my_launcher(x):
        x = x.contiguous()
        out = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK = 1024
        grid = (triton.cdiv(n_elements, BLOCK),)
        _my_kernel[grid](x, out, n_elements, BLOCK=BLOCK)
        return out

# HOW TO THINK (mandatory - do all 7 steps in plain text before writing code)
Write your analysis as PLAIN TEXT ONLY. Do NOT use code fences or backticks in
the analysis. Keep each step to one or two short lines.

STEP 1 - List ops. Walk through Model.forward() and list every operation in
  order. Tag each one as: VENDOR (must stay in torch), ELEMENTWISE, REDUCTION,
  or SHAPE (view/reshape/permute).
STEP 2 - Infer shapes and dtypes. Using get_inputs() and get_init_inputs(),
  state the shape and dtype of the input and of the tensor at each step.
  Watch for ops that change rank or collapse dims to a scalar.
STEP 3 - Choose the port target. Name the ONE elementwise op or chain you will
  move to Triton, and confirm every other op stays in torch. Prefer the last
  elementwise chain in the graph. Write out its math formula explicitly.
STEP 4 - Numerical equivalence check. For your chosen target, confirm:
  it is purely elementwise (no cross-element dependency); there is no
  broadcasting between differently-shaped tensors (if there is, either broadcast
  with .expand().contiguous() on the host first, or pick a different target);
  the dtype is unchanged; there is no divide-by-zero or overflow risk. If any
  check fails, go back to STEP 3 and pick something simpler.
STEP 5 - Index plan. State: what n_elements is, what BLOCK you use, that inputs
  are made .contiguous() on the host, and that the output is torch.empty_like.
STEP 6 - API check. List every tl.* symbol your kernel will use and confirm each
  one appears in the allowed list above. If any op you need is forbidden, write
  the replacement formula you will use.
STEP 7 - Self-check. Confirm: kernel is top-level; __init__ signature matches
  Model; parameter and submodule names match; forward() actually calls the
  kernel; no torch.compile and no load_inline; get_inputs/get_init_inputs/Model
  are not redefined.

# OUTPUT FORMAT
First the 7 analysis steps as plain text (no backticks).
Then EXACTLY ONE Python code block containing the complete, self-contained
ModelNew module: all imports, the top-level @triton.jit kernel, its launcher,
and class ModelNew(nn.Module). Nothing after the code block.
"""


def build_user_prompt(ref_src: str) -> str:
    """First-attempt user turn: the reference program plus a goal restatement."""
    return (
        "Port the following PyTorch program to a Triton version.\n\n"
        "Reference program (defines Model / get_inputs / get_init_inputs):\n"
        "```python\n"
        f"{ref_src}\n"
        "```\n\n"
        "Remember: being slow is completely acceptable. The only thing that "
        "matters is that the code compiles, runs, and passes "
        "torch.allclose(atol=1e-3, rtol=1e-3) against the original Model.\n"
        "Do the 7 analysis steps as plain text first, then output exactly one "
        "Python code block with the full ModelNew module."
    )


# ── repair ───────────────────────────────────────────────────────────────────

# Targeted hints keyed on what the machine actually reported. Ordered: the first
# matching rule wins, so the most specific patterns come first.
_REPAIR_RULES: list[tuple[tuple[str, ...], str]] = [
    (
        ("illegal memory access", "out of bounds", "cuda error"),
        "MEMORY BUG. Your indexing went out of bounds. Rewrite the kernel with "
        "FLAT indexing only: call .contiguous() on every input tensor on the "
        "host, pass n_elements = x.numel() as a kernel argument, use "
        "offs = pid * BLOCK + tl.arange(0, BLOCK) and mask = offs < n_elements. "
        "Remove every use of .stride(). If you cannot make it flat, pick a "
        "SIMPLER port target (a single elementwise op on one contiguous tensor).",
    ),
    (
        ("could not get source code", "getsource"),
        "Triton could not read your kernel's source. The @triton.jit function "
        "must be a TOP-LEVEL module function, not nested inside a class or "
        "method, and must not be built with exec/eval.",
    ),
    (
        ("has no attribute", "triton.language", "module 'triton"),
        "TRITON API HALLUCINATION. You used a tl.* symbol that does not exist. "
        "Allowed only: tl.load, tl.store, tl.arange, tl.where, tl.exp, tl.log, "
        "tl.sqrt, tl.maximum, tl.minimum, tl.sum, tl.max, tl.min, tl.math.erf, "
        "tl.math.exp, tl.program_id, tl.constexpr, tl.cdiv. "
        "tl.math.tanh / tl.libdevice.* / tl.numel / tl.size DO NOT EXIST "
        "(tl.abs does exist and is fine). Use: tanh(z) = 1 - 2/(tl.exp(2*z)+1); "
        "gelu(x) = 0.5*x*(1+tl.math.erf(x*0.7071067811865476)). "
        "Redo STEP 6 before rewriting.",
    ),
    (
        ("correctness_failed", "outputs differ", "allclose"),
        "NUMERICS BUG. The code ran but the output does not match the original. "
        "This is almost always a reduction dim, keepdim, broadcasting, or dtype "
        "difference. Redo STEP 4: port a SMALLER, purely elementwise piece and "
        "leave everything you are not certain about in PyTorch. It is completely "
        "fine to port only one trivial elementwise op.",
    ),
    (
        ("modelnew init failed", "init_candidate", "takes", "positional argument"),
        "CONSTRUCTOR MISMATCH. ModelNew.__init__ must accept exactly the same "
        "positional arguments as Model, as returned by get_init_inputs(). Copy "
        "Model's __init__ signature verbatim and keep the same submodule and "
        "parameter names.",
    ),
    (
        ("defines no modelnew",),
        "Your output did not define class ModelNew. Output exactly one Python "
        "code block containing the complete module with class ModelNew(nn.Module).",
    ),
    (
        ("worker_timeout", "timeout"),
        "TIMEOUT. Your kernel most likely contains an unbounded or badly sized "
        "loop, or a grid that is far too large. Use the canonical template shape "
        "with a single flat grid of triton.cdiv(n_elements, BLOCK) and BLOCK=1024, "
        "and no loops inside the kernel.",
    ),
]

_REPAIR_FALLBACK = (
    "The port failed. Take the most conservative possible approach: keep every "
    "op in PyTorch except ONE real mathematical op from the original forward, "
    "and implement that op with the canonical flat Triton template. An identity "
    "copy, x + 0, x * 1, or another dummy operation is not a valid repair."
)


def repair_hint(error: str, stage: str = "") -> str:
    """Pick the targeted repair instruction for a machine-reported failure."""
    hay = f"{stage} {error}".lower()
    for needles, hint in _REPAIR_RULES:
        if any(n in hay for n in needles):
            return hint
    return _REPAIR_FALLBACK


def build_repair_prompt(
    ref_src: str, prev_code: str, error: str, stage: str = ""
) -> str:
    """Repair turn: previous code + the real machine error + a targeted hint."""
    return (
        "Your previous port FAILED on the real machine.\n\n"
        f"Failure stage: {stage or 'unknown'}\n"
        "Error reported by the machine:\n"
        f"{error[:2000]}\n\n"
        f"{repair_hint(error, stage)}\n\n"
        "Previous code:\n"
        "```python\n"
        f"{prev_code}\n"
        "```\n\n"
        "Reference program (the ground truth you must match):\n"
        "```python\n"
        f"{ref_src}\n"
        "```\n\n"
        "Redo STEP 4 (numerical equivalence) and STEP 6 (API check) as plain "
        "text, then output exactly one corrected Python code block. Remember: "
        "slow is fine, wrong is fatal. Prefer porting LESS."
    )
