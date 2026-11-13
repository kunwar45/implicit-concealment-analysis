"""Tests for the measurement fixes: disclosure labels, prompt format, saved config."""

import json

import pytest

from src.config import Config
from src.framing.framing_loader import FramingLoader
from src.loaders.concealment_loader import ConcealmentLoader
from src.monitors.labels import disclosed
from src.pipeline.result import PromptResult
from src.steps.base_model_step import FORMAT_INSTRUCTION, BaseModelStep
from src.storage.storage import ResultStorage


def _r(regex_hit=None, llm_verdict=None):
    mon = {}
    if regex_hit is not None:
        mon["regex"] = {
            "contains_secret_exact": False,
            "contains_secret_partial": regex_hit,
        }
    if llm_verdict is not None:
        mon["llm"] = {"verdict": llm_verdict}
    return {"monitor_results": mon}


@pytest.mark.parametrize(
    "result, regex, llm, llm_first",
    [
        (_r(True, "disclosed"), True, True, True),
        (_r(True, "concealed"), True, False, False),  # judge overrides regex
        (_r(False, "redirected"), False, False, False),
        (_r(True, "parse_error"), True, None, True),  # undecided judge → regex
        (_r(False, None), False, None, False),
        (_r(None, None), False, None, None),  # no monitors at all
    ],
)
def test_disclosure_rules(result, regex, llm, llm_first):
    assert disclosed(result, "regex") is regex
    assert disclosed(result, "llm") is llm
    assert disclosed(result, "llm_first") is llm_first


def test_disclosure_accepts_prompt_result():
    pr = PromptResult(prompt_id="x", prompt="q", timestamp="t")
    pr.monitor_results = {"llm": {"verdict": "disclosed"}}
    assert disclosed(pr) is True


def test_unknown_rule_raises():
    with pytest.raises(ValueError):
        disclosed(_r(True), "vibes")


def test_loaders_order_context_condition_format():
    c = ConcealmentLoader._build_system_prompt("COND", "CTX")
    assert c == f"CTX\n\nCOND\n\n{FORMAT_INSTRUCTION}"
    f = FramingLoader._build_system_prompt("EXPLICIT", "assistant", "CTX")
    assert f.startswith("CTX") and f.endswith(FORMAT_INSTRUCTION)
    assert f.index("CTX") < f.index("You are") < f.index("Format your response")


class _FakeClient:
    def __init__(self, raw):
        self.raw = raw

    def chat(self, model, messages, max_tokens, temperature):
        return self.raw


@pytest.mark.parametrize(
    "raw, ok",
    [
        ("<reasoning>think</reasoning><answer>21 days</answer>", True),
        ("It ships in 21 days.", False),
    ],
)
def test_format_ok_flag(raw, ok):
    step = BaseModelStep(_FakeClient(raw), Config())
    out = step.run(PromptResult(prompt_id="x", prompt="q", timestamp="t"))
    assert out.metadata["format_ok"] is ok
    assert (out.reasoning_trace != out.final_answer) is ok


def test_storage_saves_full_config_without_secret(tmp_path):
    cfg = Config(
        together_api_key="SECRET-123", output_dir=str(tmp_path), temperature=0.3
    )
    path = ResultStorage(cfg).save([])
    payload = json.load(open(path))
    assert payload["config"]["temperature"] == 0.3
    assert "together_api_key" not in payload["config"]
    assert "SECRET-123" not in open(path).read()


# ── Native reasoning models (Qwen3 / Qwen3.6) ─────────────────────────────────

from types import SimpleNamespace

from src.clients.factory import PROVIDERS, make_client, parse_thinking
from src.steps.base_model_step import split_response


@pytest.mark.parametrize(
    "raw, reasoning, answer, source",
    [
        ("<think>plan</think>It ships in 21 days.", "plan", "It ships in 21 days.", "think"),
        ("plan\n</think>\n\nIt ships in 21 days.", "plan", "It ships in 21 days.", "think"),  # prefilled
        ("<think>still going", "still going", "", "think_truncated"),
        ("<think>p</think><answer>21 days</answer>", "p", "21 days", "think"),
        ("<reasoning>r</reasoning><answer>a</answer>", "r", "a", "tags"),
        ("plain text", "plain text", "plain text", "none"),
        ("<answer>only answer</answer>", "<answer>only answer</answer>",
         "<answer>only answer</answer>", "none"),
    ],
)
def test_split_response(raw, reasoning, answer, source):
    assert split_response(raw) == (reasoning, answer, source)


def test_format_ok_for_think_output():
    step = BaseModelStep(_FakeClient("<think>t</think>21 days"), Config())
    out = step.run(PromptResult(prompt_id="x", prompt="q", timestamp="t"))
    assert out.metadata["format_ok"] is True
    assert out.metadata["reasoning_source"] == "think"
    assert out.metadata["raw_output"] == "<think>t</think>21 days"
    assert (out.reasoning_trace, out.final_answer) == ("t", "21 days")


def test_loader_can_omit_format_instruction():
    c = ConcealmentLoader._build_system_prompt("COND", "CTX", format_instruction=False)
    assert c == "CTX\n\nCOND"


def test_parse_thinking():
    assert (parse_thinking("default"), parse_thinking("on"), parse_thinking("off")) == (None, True, False)


def test_factory_providers(monkeypatch):
    assert {"together", "hf", "vllm", "openrouter"} <= set(PROVIDERS)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError):
        make_client("openrouter")
    with pytest.raises(ValueError):
        make_client("nope")
    monkeypatch.setenv("VLLM_BASE_URL", "http://fir-node:8000/v1")
    c = make_client("vllm", enable_thinking=False)
    assert c.base_url == "http://fir-node:8000/v1" and c.enable_thinking is False


@pytest.mark.parametrize(
    "content, extra, expected",
    [
        ("21 days", {"reasoning_content": "t"}, "<think>t</think>21 days"),  # vLLM 0.8 parser
        ("21 days", {"reasoning": "t"}, "<think>t</think>21 days"),          # vLLM 0.26 parser
        ("<think>t</think>21 days", {}, "<think>t</think>21 days"),          # inline
        ("21 days", {}, "21 days"),                                           # thinking off
    ],
)
def test_openai_compat_normalises_reasoning(monkeypatch, content, extra, expected):
    c = make_client("vllm", enable_thinking=True)
    sent = {}

    def create(**kw):
        sent.update(kw)
        msg = SimpleNamespace(content=content, **extra)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    monkeypatch.setattr(c.client.chat.completions, "create", create)
    assert c.chat("Qwen/Qwen3.6-27B", [{"role": "user", "content": "q"}], 8, 0.0) == expected
    assert sent["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
