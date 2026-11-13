"""Build a ChatClient by provider name — for the subject model and the LLM judge.

Providers
---------
together   : Together AI (TOGETHER_API_KEY).
openrouter : OpenRouter, OpenAI-compatible (OPENROUTER_API_KEY).
vllm       : a vLLM server you started, OpenAI-compatible (VLLM_BASE_URL,
             default http://localhost:8000/v1; VLLM_API_KEY optional).
hf         : load the model in-process with HuggingFace (needed for activations).

The model name is passed per call (``config.base_model`` / ``config.monitor_model``),
so a bigger model on an existing provider needs no code change. Adding a provider =
one class implementing ``ChatClient`` + one branch below (or, if it speaks the
OpenAI API, one row in ``OPENAI_COMPAT``).
"""

from __future__ import annotations

import os

from src.clients import ChatClient

# provider -> (default base URL, env var overriding it, env var holding the key)
OPENAI_COMPAT: dict[str, tuple[str, str | None, str | None]] = {
    "openrouter": ("https://openrouter.ai/api/v1", None, "OPENROUTER_API_KEY"),
    "vllm": ("http://localhost:8000/v1", "VLLM_BASE_URL", "VLLM_API_KEY"),
}
PROVIDERS = ("together", "hf", *OPENAI_COMPAT)


def make_client(
    provider: str,
    model: str | None = None,
    enable_thinking: bool | None = None,
    **kwargs,
) -> ChatClient:
    """Return a ChatClient for ``provider``.

    enable_thinking: for reasoning models (Qwen3/3.6) — True/False sets the chat
        template's thinking switch; None leaves the template default. Ignored by
        Together.
    kwargs: passed to the client (e.g. ``dtype=``/``capture_mode=`` for hf,
        ``base_url=`` to override an OpenAI-compatible endpoint).
    """
    if provider == "together":
        from src.clients.together_client import TogetherClient

        api_key = kwargs.pop("api_key", None) or os.getenv("TOGETHER_API_KEY", "")
        if not api_key:
            raise ValueError(
                "provider 'together' needs TOGETHER_API_KEY (see .env.example)"
            )
        return TogetherClient(api_key=api_key, **kwargs)

    if provider == "hf":
        from src.clients.hf_client import HFClient

        if not model:
            raise ValueError("provider 'hf' needs a model name")
        return HFClient(model_name=model, enable_thinking=enable_thinking, **kwargs)

    if provider in OPENAI_COMPAT:
        from src.clients.openai_compat_client import OpenAICompatClient

        default_url, url_env, key_env = OPENAI_COMPAT[provider]
        base_url = (
            kwargs.pop("base_url", None)
            or (url_env and os.getenv(url_env))
            or default_url
        )
        api_key = kwargs.pop("api_key", None) or (key_env and os.getenv(key_env)) or ""
        if provider == "openrouter" and not api_key:
            raise ValueError(
                "provider 'openrouter' needs OPENROUTER_API_KEY (see .env.example)"
            )
        return OpenAICompatClient(
            base_url=base_url,
            api_key=api_key or "EMPTY",
            enable_thinking=enable_thinking,
            **kwargs,
        )

    raise ValueError(f"Unknown provider {provider!r}; expected one of {PROVIDERS}")


THINKING_CHOICES = ("default", "on", "off")


def parse_thinking(value: str) -> bool | None:
    """CLI --thinking {default,on,off} -> enable_thinking (None keeps the template default)."""
    return {"default": None, "on": True, "off": False}[value]
