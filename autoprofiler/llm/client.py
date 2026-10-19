from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

from autoprofiler.llm.config import LLMConfig
from autoprofiler import metering
from autoprofiler.metering import Meter


class LLMClient:
    """Minimal OpenAI-compatible chat client (DashScope / OpenAI / etc.)."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        if config is None:
            from autoprofiler.llm.config import load_llm_config

            config = load_llm_config()
        self.config = config
        self.usage = Meter()  # cumulative over this client's lifetime

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retries: int | None = None,
        log=None,
        request_label: str = "",
    ) -> str:
        retries = self.config.request_retries if retries is None else retries
        request_id = uuid.uuid4().hex[:8]
        request_started = time.monotonic()

        def emit(event: str, *, attempt: int, error: Exception | None = None) -> None:
            if log is None:
                return
            elapsed = time.monotonic() - request_started
            suffix = ""
            if error is not None:
                detail = f"{type(error).__name__}: {error}".replace("\n", " ")[:500]
                suffix = f" error={detail}"
            log(
                f"[{event}] id={request_id} model={self.config.model} "
                f"label={request_label or '-'} attempt={attempt}/{retries + 1} "
                f"timeout_s={self.config.timeout_s:g} elapsed_s={elapsed:.1f}{suffix}"
            )

        api_style = self.config.api_style.strip().lower()
        token_limit = self.config.max_tokens if max_tokens is None else max_tokens
        temp = self.config.temperature if temperature is None else temperature
        if api_style == "anthropic":
            url = f"{self.config.base_url}/messages"
            system = "\n\n".join(
                str(m["content"]) for m in messages if m.get("role") == "system"
            )
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": m["role"], "content": m["content"]}
                    for m in messages
                    if m.get("role") != "system"
                ],
                "max_tokens": token_limit,
            }
            if system:
                payload["system"] = system
        elif api_style == "openai":
            url = f"{self.config.base_url}/chat/completions"
            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": token_limit,
            }
        else:
            raise ValueError(f"unsupported LLM api_style: {self.config.api_style!r}")
        data = json.dumps(payload).encode("utf-8")
        last_err: Exception | None = None
        dropped_unsupported = False
        for attempt in range(retries + 1):
            attempt_number = attempt + 1
            emit("llm-attempt", attempt=attempt_number)
            req = urllib.request.Request(
                url,
                data=data,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "x-api-key": self.config.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                try:
                    if api_style == "anthropic":
                        text = "".join(
                            str(block.get("text") or "")
                            for block in body["content"]
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                    else:
                        message = body["choices"][0]["message"]
                        # Some reasoning gateways leave content empty until the
                        # answer phase. Keeping reasoning_content as a fallback
                        # makes short diagnostic calls observable; production
                        # seed runs still use a large token budget so the final
                        # fenced program should be present in content.
                        text = str(
                            message.get("content")
                            or message.get("reasoning_content")
                            or ""
                        )
                    if not text:
                        raise ValueError("LLM returned an empty text response")
                    u = body.get("usage") or {}
                    n_in = u.get("prompt_tokens") or u.get("input_tokens") or 0
                    n_out = u.get("completion_tokens") or u.get("output_tokens") or 0
                    self.usage.add_llm(n_in, n_out)
                    metering.charge_llm(n_in, n_out)
                    if log is not None:
                        log(f"[llm-usage] id={request_id} model={self.config.model} tokens={n_in + n_out}")
                    emit("llm-done", attempt=attempt_number)
                    return text
                except (KeyError, IndexError, TypeError, ValueError) as e:
                    raise RuntimeError(f"Unexpected LLM response shape: {body!r}") from e
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")[:800]
                last_err = RuntimeError(f"LLM HTTP {e.code}: {detail}")
                # Some models (reasoning gateways especially) reject a sampling
                # parameter outright, e.g. "Parameter 'temperature'=0.2 is not
                # supported for kimi-k3 model" -- a fixed config value that works for
                # every other model turns into a permanent crash for this one. Drop
                # the named parameter and retry once rather than failing every call
                # a user makes with such a model; "429/5xx -> backoff" below still
                # handles genuine transient failures.
                if e.code == 400 and not dropped_unsupported:
                    bad_param = next(
                        (m.group(1) for m in re.finditer(r"[Pp]arameter '(\w+)'", detail)
                         if m.group(1) in payload),
                        None,
                    )
                    if bad_param:
                        payload.pop(bad_param)
                        data = json.dumps(payload).encode("utf-8")
                        dropped_unsupported = True
                        emit("llm-retry", attempt=attempt_number, error=last_err)
                        continue
                # retry 429 / 5xx
                if e.code not in (429, 500, 502, 503, 504) or attempt >= retries:
                    emit("llm-failed", attempt=attempt_number, error=last_err)
                    raise last_err from e
                emit("llm-retry", attempt=attempt_number, error=last_err)
            except urllib.error.URLError as e:
                # Transport / SSL EOF from DashScope is common; retry with backoff.
                last_err = RuntimeError(f"LLM request failed: {e}")
                if attempt >= retries:
                    emit("llm-failed", attempt=attempt_number, error=last_err)
                    raise last_err from e
                emit("llm-retry", attempt=attempt_number, error=last_err)
            except TimeoutError as e:
                last_err = RuntimeError(f"LLM timeout: {e}")
                if attempt >= retries:
                    emit("llm-failed", attempt=attempt_number, error=last_err)
                    raise last_err from e
                emit("llm-retry", attempt=attempt_number, error=last_err)
            except OSError as e:
                last_err = RuntimeError(f"LLM OSError: {e}")
                if attempt >= retries:
                    emit("llm-failed", attempt=attempt_number, error=last_err)
                    raise last_err from e
                emit("llm-retry", attempt=attempt_number, error=last_err)
            # Exponential-ish backoff; SSL flakes often need > few seconds.
            time.sleep(min(30.0, 2.0 * (2**attempt)))
        raise RuntimeError(f"LLM failed after retries: {last_err}")


__all__ = ["LLMClient"]
