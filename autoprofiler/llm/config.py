from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from autoprofiler.config import REPO_ROOT


class LLMConfig(BaseModel):
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    api_style: str = "openai"
    temperature: float = 0.2
    # Generous on purpose: a reasoning model can spend minutes thinking before it writes
    # a single token of its answer, so a short timeout does not fail fast, it fails
    # slow -- the request was always going to succeed, but 180s plus up to 8 retries
    # with backoff turned one 4-minute call into a 30-minute one. Every shipped
    # profile (gpt.yaml, claude.yaml) already overrides this to 600; make that the
    # default instead of a trap for anyone who writes an inline llm: block without it.
    timeout_s: float = 600.0
    request_retries: int = Field(default=8, ge=0)
    max_tokens: int = Field(default=4096, ge=256)


def _parse_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if " #" in val:
            val = val.split(" #", 1)[0].rstrip()
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            val = val[1:-1]
        out[key] = val
    return out


def resolve_api_key(provider: str | None = None, env_path: Path | None = None) -> str:
    """Look up the secret key for a provider id from env / .env.

    Tries "<PROVIDER>_API_KEY" first (e.g. provider=qwen -> QWEN_API_KEY), then
    the generic AUTOPROFILER_LLM_API_KEY / OPENAI_API_KEY fallbacks.
    """
    file_env = _parse_dotenv(env_path or (REPO_ROOT / ".env"))
    merged = {**file_env, **{k: v for k, v in os.environ.items() if v}}
    names: list[str] = []
    if provider:
        names.append(f"{provider.strip().upper()}_API_KEY")
    names += ["AUTOPROFILER_LLM_API_KEY", "OPENAI_API_KEY"]
    for n in names:
        val = (merged.get(n) or "").strip()
        if val:
            return val
    raise RuntimeError(
        f"Missing API key. Set {names[0]} in .env (or one of {names[1:]})."
    )


def load_llm_config(env_path: Path | None = None) -> LLMConfig:
    """Load LLM settings from process env, overlaying repo `.env` if present."""
    file_env = _parse_dotenv(env_path or (REPO_ROOT / ".env"))
    merged = {**file_env, **{k: v for k, v in os.environ.items() if v}}

    api_key = (
        merged.get("AUTOPROFILER_LLM_API_KEY")
        or merged.get("OPENAI_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError(
            "Missing AUTOPROFILER_LLM_API_KEY (or OPENAI_API_KEY). "
            "Set it in .env or the environment."
        )

    base_url = (
        merged.get("AUTOPROFILER_LLM_BASE_URL")
        or merged.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    ).strip().rstrip("/")
    model = (
        merged.get("AUTOPROFILER_LLM_MODEL")
        or merged.get("OPENAI_MODEL")
        or "gpt-4o-mini"
    ).strip()

    return LLMConfig(api_key=api_key, base_url=base_url, model=model)


def load_llm_profile(name_or_path: str, env_path: Path | None = None) -> LLMConfig:
    """Build an :class:`LLMConfig` from a named profile, e.g. ``gpt`` -> ``configs/llm/gpt.yaml``."""
    return resolve_llm({"profile": name_or_path}, env_path=env_path)


def _read_profile(name_or_path: str) -> dict[str, Any]:
    """The raw yaml of a profile in ``configs/llm/`` (or a path to one). No key lookup."""
    import yaml

    p = Path(name_or_path).expanduser()
    if not (p.suffix in (".yaml", ".yml") and p.exists()):
        p = REPO_ROOT / "configs" / "llm" / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"LLM profile not found: {name_or_path} (looked for {p})")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


_SPEC_FIELDS = ("model", "base_url", "api_style", "temperature", "max_tokens",
                "timeout_s", "api_key_env")


def resolve_llm(spec: Any, env_path: Path | None = None) -> LLMConfig:
    """Resolve one caller's LLM settings into a concrete config.

    ``spec`` is a mapping, or any object carrying these attributes. It may

    * name a ``profile`` in ``configs/llm/`` -- the short form;
    * give ``model`` / ``base_url`` / ``api_style`` / ``api_key_env`` directly; or
    * do both, where the explicit fields win, so a profile serves as a base with
      one field changed.

    The secret never appears here: ``api_key_env`` names the environment variable
    (or ``.env`` entry) that holds it.
    """
    get = spec.get if hasattr(spec, "get") else lambda k, d=None: getattr(spec, k, d)

    data: dict[str, Any] = {}
    profile = str(get("profile", "") or "").strip()
    if profile:
        data.update(_read_profile(profile))
    for field in _SPEC_FIELDS:
        value = get(field, None)
        if value not in (None, ""):
            data[field] = value

    key_env = str(data.pop("api_key_env", "") or "").strip()
    where = f"llm profile {profile!r}" if profile else "this llm section"
    if not data.get("model"):
        raise RuntimeError(f"{where}: no model set; give `model` and `base_url`, or a `profile`")

    merged = {**_parse_dotenv(env_path or (REPO_ROOT / ".env")),
              **{k: v for k, v in os.environ.items() if v}}
    names = [key_env] if key_env else ["AUTOPROFILER_LLM_API_KEY", "OPENAI_API_KEY"]
    api_key = next((merged[n].strip() for n in names if (merged.get(n) or "").strip()), "")
    if not api_key:
        raise RuntimeError(f"{where}: set {' or '.join(names)} in the environment or .env")
    return LLMConfig(api_key=api_key, **data)


class LLMSpec(BaseModel):
    """Which LLM one caller uses: a profile name, an explicit endpoint, or both.

    Subclass it to add that caller's own sampling knobs. :func:`resolve_llm` turns an
    instance into a concrete :class:`LLMConfig`.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    profile: str = ""      # a profile in configs/llm/, used as the base
    model: str = ""        # model name at the endpoint; overrides the profile
    base_url: str = ""     # OpenAI-compatible endpoint; overrides the profile
    api_style: str = ""    # openai | anthropic; overrides the profile
    api_key_env: str = ""  # env var (or .env entry) holding the key
