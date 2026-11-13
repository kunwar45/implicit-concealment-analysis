"""ChatClient for any OpenAI-compatible endpoint: a vLLM server or OpenRouter.

Mirrors how lessons_from_constitutional_aft reaches its models
(src/infra/endpoints/vllm.py there): HF models are served by vLLM as an
OpenAI-compatible endpoint, API models go through OpenRouter.

Reasoning models (Qwen3 / Qwen3.6) can return their trace three ways — inline in
``content`` as ``<think>…</think>``, out of band in ``reasoning_content`` (vLLM
0.8.x) or ``reasoning`` (vLLM 0.26+ / OpenRouter), or not at all when thinking is
off. ``chat()`` normalises all of them to one string, ``<think>TRACE</think>ANSWER``,
so ``BaseModelStep`` splits every provider the same way.
"""

from __future__ import annotations


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        enable_thinking: bool | None = None,
        max_retries: int = 3,
        timeout: float = 600.0,
    ):
        from openai import OpenAI  # lazy: only needed for these providers

        self.base_url = base_url
        self.enable_thinking = enable_thinking
        # The SDK retries 429/5xx/timeouts itself; no extra retry loop on top.
        self.client = OpenAI(
            base_url=base_url, api_key=api_key, max_retries=max_retries, timeout=timeout
        )

    def chat(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        extra_body = {}
        if self.enable_thinking is not None:
            # vLLM forwards this into the chat template (Qwen3: enable_thinking).
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": self.enable_thinking
            }
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body or None,
        )
        msg = response.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or getattr(
            msg, "reasoning", None
        )
        if reasoning and "</think>" not in content:
            return f"<think>{reasoning}</think>{content}"
        return content
