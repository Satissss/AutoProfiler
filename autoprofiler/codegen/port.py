"""LLM side of the seed port: ask for ModelNew, then check it locally.

Local checks are deliberately cheap and strict. Every candidate rejected here is
a remote GPU round-trip we did not have to spend, so anything decidable from the
source alone (missing ModelNew, syntax error, no Triton kernel, forbidden APIs)
is decided here.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

from autoprofiler.llm.client import LLMClient
from autoprofiler.codegen.port_prompts import (
    PROMPT_VERSION,
    SYSTEM,
    build_repair_prompt,
    build_user_prompt,
)

_FENCE_RE = re.compile(r"```(?:python|py)?\s*([\s\S]*?)```", re.IGNORECASE)

# Symbols that do not exist in Triton 3.x. Catching these locally saves a remote
# round-trip that would only rediscover the same AttributeError.
#
# Only list a symbol after checking it against the Triton the GPU actually runs:
#   python -c "import triton.language as tl; print(hasattr(tl, 'abs'))"
# Listing a real one is worse than listing none, because the model writes correct
# code, we reject it unread, and the repair budget goes on re-deriving what it had.
FORBIDDEN_TL = (
    "tl.math.tanh",   # tl.math exists, tanh does not
    "tl.libdevice",
    "tl.numel",
    "tl.size",
)

FORBIDDEN_TORCH = (
    "torch.compile",
    "torch._dynamo",
    "torch._inductor",
    "load_inline",
    "cpp_extension",
)


class PortRejected(Exception):
    """The candidate is locally invalid; carries a machine-style reason."""

    def __init__(self, reason: str, *, stage: str = "local_check") -> None:
        super().__init__(reason)
        self.reason = reason
        self.stage = stage


@dataclass
class PortCandidate:
    code: str
    analysis: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def extract_code(text: str) -> tuple[str, str]:
    """Split the response into (analysis_text, code).

    The prompt asks for plain-text analysis followed by exactly one code block,
    so we take the LAST fence: that is robust even if the model slips a fenced
    snippet into its reasoning, which the first-fence rule would mistake for the
    answer.
    """
    fences = _FENCE_RE.findall(text)
    if fences:
        code = fences[-1].strip()
        analysis = text.split("```")[0].strip()
        return analysis, code
    if "class ModelNew" in text:
        return "", text.strip()
    raise PortRejected("response contained no Python code block")


def check_candidate(code: str) -> dict[str, Any]:
    """Static gate. Raises PortRejected, or returns descriptive metadata."""
    if "class ModelNew" not in code:
        raise PortRejected("candidate defines no ModelNew")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise PortRejected(f"candidate has a syntax error: {e}") from e

    for marker in FORBIDDEN_TORCH:
        if marker in code:
            raise PortRejected(f"candidate uses forbidden {marker}")
    hits = [s for s in FORBIDDEN_TL if s in code]
    if hits:
        raise PortRejected(
            f"candidate uses non-existent Triton symbols {hits}; "
            "see the allowed tl.* list"
        )

    jit_kernels = _toplevel_jit_kernels(tree)
    if not jit_kernels:
        nested = _has_nested_jit(tree)
        if nested:
            raise PortRejected(
                "the @triton.jit kernel is nested inside a class or function; "
                "Triton needs it as a top-level module function"
            )
        raise PortRejected("candidate contains no top-level @triton.jit kernel")

    if not _calls_any(tree, jit_kernels):
        raise PortRejected(
            f"kernel(s) {jit_kernels} are defined but never launched in the module"
        )
    kernel_defs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in set(jit_kernels)
    ]
    if kernel_defs and all(_is_identity_kernel(node) for node in kernel_defs):
        raise PortRejected(
            "candidate uses only identity/no-op Triton kernels; the kernel must "
            "implement a real mathematical operation from the original forward"
        )

    return {
        "jit_kernels": jit_kernels,
        "n_lines": len(code.splitlines()),
    }


def _is_jit_decorator(dec: ast.expr) -> bool:
    node = dec.func if isinstance(dec, ast.Call) else dec
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return "jit" in parts and ("triton" in parts or len(parts) == 1)


def _toplevel_jit_kernels(tree: ast.Module) -> list[str]:
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(_is_jit_decorator(d) for d in node.decorator_list)
    ]


def _has_nested_jit(tree: ast.Module) -> bool:
    for parent in ast.walk(tree):
        if parent is tree or not hasattr(parent, "body"):
            continue
        body = getattr(parent, "body", None)
        # Statement containers expose ``body`` as a list, while expression
        # nodes such as Lambda expose a single AST expression.  Treat only the
        # former as nested-definition containers; iterating the latter crashes
        # when, for example, a lambda returns a tuple.
        children = body if isinstance(body, list) else []
        for child in children:
            if isinstance(child, ast.FunctionDef) and any(
                _is_jit_decorator(d) for d in child.decorator_list
            ):
                if child not in tree.body:
                    return True
    return False


def _calls_any(tree: ast.Module, names: list[str]) -> bool:
    """True when any kernel name is referenced outside its own definition.

    Triton launches look like ``kernel[grid](...)`` (a Subscript then Call), so a
    plain Call-node scan is not enough; a Name reference is the reliable signal.
    """
    wanted = set(names)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            continue
        if isinstance(node, ast.Name) and node.id in wanted:
            return True
    return False


def _is_tl_call(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tl"
        and node.func.attr == name
    )


def _is_neutral_expr(node: ast.AST, loaded: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in loaded
    if not isinstance(node, ast.BinOp):
        return False
    if isinstance(node.op, (ast.Add, ast.Sub)):
        return (
            _is_neutral_expr(node.left, loaded)
            and isinstance(node.right, ast.Constant)
            and node.right.value == 0
        )
    if isinstance(node.op, (ast.Mult, ast.Div)):
        return (
            _is_neutral_expr(node.left, loaded)
            and isinstance(node.right, ast.Constant)
            and node.right.value == 1
        )
    return False


def _is_identity_kernel(fn: ast.FunctionDef) -> bool:
    """Catch copy/+0/*1 kernels used only to game the Triton requirement."""
    loaded: set[str] = set()
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and _is_tl_call(node.value, "load")
        ):
            loaded.add(node.targets[0].id)
    stored_values = [
        node.args[1]
        for node in ast.walk(fn)
        if _is_tl_call(node, "store") and len(node.args) >= 2
    ]
    return bool(stored_values) and all(
        _is_neutral_expr(value, loaded) for value in stored_values
    )


def propose_port(
    ref_src: str,
    *,
    client: LLMClient,
    temperature: float,
    prev_code: str | None = None,
    error: str | None = None,
    stage: str = "",
) -> PortCandidate:
    """One LLM turn: first attempt, or a repair when ``error`` is given."""
    if error and prev_code:
        user = build_repair_prompt(ref_src, prev_code, error, stage)
    else:
        user = build_user_prompt(ref_src)

    raw = client.chat(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    analysis, code = extract_code(raw)
    meta = check_candidate(code)
    return PortCandidate(
        code=code,
        analysis=analysis,
        meta={
            **meta,
            "prompt_version": PROMPT_VERSION,
            "temperature": temperature,
            "is_repair": bool(error),
            "raw_chars": len(raw),
            "analysis_chars": len(analysis),
        },
    )
