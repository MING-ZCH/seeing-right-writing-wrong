"""Minimal client for an OpenAI-compatible chat-completions endpoint.

The endpoint is read from OPENAI_COMPAT_BASE_URL (e.g. ``https://<host>/v1``), the key from
OPENAI_COMPAT_API_KEY (never hard-coded or logged)."""
from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request

URL = os.environ.get("OPENAI_COMPAT_BASE_URL", "").rstrip("/") + "/chat/completions"


def _auth() -> str:
    key = os.environ["OPENAI_COMPAT_API_KEY"]
    return f"Bearer {key}"


def img_part(pil) -> dict:
    """PNG data-URL image content part."""
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


def chat(model: str, messages: list, max_tokens: int = 4096, effort: str | None = "high", temperature: float | None = None,
         retries: int = 4, timeout: int = 600) -> dict:
    """POST one chat completion; returns {"text", "reasoning", "usage", "raw_model"} or {"error"}."""
    body = {"model": model, "messages": messages, "max_completion_tokens": max_tokens, "stream": False}
    if effort and model.startswith("gemini") and not os.environ.get("OPENAI_COMPAT_GEMINI_REASONING_EFFORT"):
        # Gemini models take thinking_config.thinking_level (low / high) on the endpoint used for the paper;
        # set OPENAI_COMPAT_GEMINI_REASONING_EFFORT=1 to send reasoning_effort instead.
        body["thinking_config"] = {"thinking_level": effort}
    elif effort:
        body["reasoning_effort"] = effort
    if temperature is not None:
        body["temperature"] = temperature
    data = json.dumps(body).encode()
    last = ""
    for k in range(retries):
        req = urllib.request.Request(URL, data=data, headers={"Authorization": _auth(), "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            msg = d["choices"][0]["message"]
            return {"text": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
                    "usage": d.get("usage", {}), "raw_model": d.get("model"),
                    "finish_reason": d["choices"][0].get("finish_reason")}
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
            if e.code in (400, 403, 404) or "UserModelNotFound" in last:
                break
            if e.code == 429:  # per-account rpm limit: back off longer
                time.sleep(20 * (k + 1))
                continue
        except Exception as e:  # network / timeout
            last = repr(e)[:300]
        time.sleep(3 * (k + 1))
    return {"error": last}
