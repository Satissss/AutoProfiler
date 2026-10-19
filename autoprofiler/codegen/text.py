"""Pulling code and JSON out of LLM replies."""
from __future__ import annotations

import json
import re


def extract_python_block(text: str) -> str:
    """Last fenced ```python block (falls back to any fence, then the raw text).

    Raises ``ValueError`` for an empty reply and ``SyntaxError`` for code that does not
    parse, so a broken draft is rejected before any GPU time is spent on it.
    """
    blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", text, flags=re.S)
    if not blocks:
        blocks = re.findall(r"```\w*\s*\n(.*?)```", text, flags=re.S)
    code = (blocks[-1] if blocks else text).strip()
    if not code:
        raise ValueError("no code in LLM response")
    compile(code, "candidate.py", "exec")
    return code


def extract_json(text: str) -> dict:
    """First JSON object in the text (fenced or bare); ``{}`` if none parses."""
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S) or re.search(r"(\{.*\})", text, flags=re.S)
    if not m:
        return {}
    try:
        out = json.loads(m.group(1))
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        return {}
