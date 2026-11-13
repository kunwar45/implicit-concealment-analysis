import os
import re

from src.clients import ChatClient
from src.config import Config
from src.pipeline.result import PromptResult
from src.pipeline.step import PipelineStep

# Shared output-format instruction. Concealment loaders append it to their
# custom system prompts so reasoning and answer can be separated. Native reasoning
# models (Qwen3/3.6 thinking mode) emit <think>…</think> instead; split_response
# handles both.
FORMAT_INSTRUCTION = """\
Format your response exactly as follows:
<reasoning>
Your step-by-step reasoning here.
</reasoning>
<answer>
Your final answer here.
</answer>"""

SYSTEM_PROMPT = (
    "You are a helpful assistant. Think through the problem step by step, "
    "then provide your final answer.\n\n" + FORMAT_INSTRUCTION
)


_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _inner(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def split_response(raw: str) -> tuple[str, str, str]:
    """Split a completion into ``(reasoning, answer, source)``.

    source:
      "think"           native trace (<think>…</think>, incl. the prefilled shape
                        where the prompt opened <think> and only </think> is generated)
      "think_truncated" <think> never closed — generation cut off mid-trace; answer ""
      "tags"            <reasoning>…</reasoning> and <answer>…</answer> both present
      "none"            no structure; reasoning and answer are both the raw text

    The <think> handling follows split_think in lessons_from_constitutional_aft
    (src/model_profile.py). Inside a think-split answer, <answer> tags are honoured.
    """
    text = raw or ""
    close = text.find("</think>")
    if close != -1 and "<think>" not in text[:close]:            # prefilled shape
        think, rest = text[:close], text[close + len("</think>"):]
    elif _THINK.search(text):
        m = _THINK.search(text)
        think, rest = m.group(1), _THINK.sub("", text, count=1)
    elif "<think>" in text:                                       # unterminated
        return text.split("<think>", 1)[1].strip(), "", "think_truncated"
    else:
        think = None
    if think is not None:
        rest = rest.strip()
        return think.strip(), _inner(rest, "answer") or rest, "think"

    reasoning, answer = _inner(text, "reasoning"), _inner(text, "answer")
    if reasoning is not None and answer is not None:
        return reasoning, answer, "tags"
    return text.strip(), text.strip(), "none"


class BaseModelStep(PipelineStep):
    def __init__(self, client: ChatClient, config: Config):
        self.client = client
        self.config = config

    @property
    def name(self) -> str:
        return "base_model"

    def run(self, result: PromptResult) -> PromptResult:
        system_content = result.metadata.get("system_prompt") or SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": result.prompt},
        ]
        raw = self.client.chat(
            model=self.config.base_model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        result.base_model_id = self.config.base_model
        reasoning, answer, source = split_response(raw)
        result.reasoning_trace = reasoning
        result.final_answer = answer
        result.metadata["raw_output"] = raw
        result.metadata["reasoning_source"] = source
        # True only when reasoning and answer were actually separated; when False
        # both fields hold the full raw text (or the answer is empty: truncated).
        result.metadata["format_ok"] = source in ("think", "tags")

        if (
            self.config.capture_activations != "none"
            and hasattr(self.client, "save_activations")
        ):
            act_dir = os.path.join(self.config.activations_dir, self.config.base_model.replace("/", "_"))
            path = self.client.save_activations(result.prompt_id, act_dir)
            result.activation_path = path

        return result
