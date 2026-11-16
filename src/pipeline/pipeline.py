from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from tqdm import tqdm

from src.display import box, rule

from .result import PromptResult
from .step import PipelineStep

PREVIEW_PROMPT = 220
PREVIEW_ANSWER = 120
PREVIEW_REASONING = 180
PREVIEW_MONITOR = 200


def _preview(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "…"


def _format_monitor(data: dict) -> str:
    """Turn a monitor result dict into a human-readable string for display."""
    lines = []
    for key, value in data.items():
        if isinstance(value, list):
            lines.append(f"{key}: {', '.join(str(v) for v in value) or '(none)'}")
        else:
            lines.append(f"{key}: {value}")
    return _preview("\n".join(lines), PREVIEW_MONITOR)


class Pipeline:
    def __init__(self, steps: list[PipelineStep]):
        self.steps = steps

    def _run_one(self, prompt: dict) -> PromptResult:
        result = PromptResult(
            prompt_id=prompt["id"],
            prompt=prompt["text"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=dict(prompt.get("metadata", {})),
        )
        for step in self.steps:
            result = step.run(result)
        return result

    def run(self, prompts: list[dict], workers: int = 1) -> list[PromptResult]:
        """Run every prompt through the steps; results come back in input order.

        ``workers > 1`` runs prompts concurrently in threads — only for clients that
        are safe to call from several threads (API / vLLM server), never HFClient.
        """
        total = len(prompts)
        bar = tqdm(total=total, desc="Questions", unit="q")
        if workers <= 1:
            results = []
            for i, prompt in enumerate(prompts):
                results.append(self._run_one(prompt))
                bar.update(1)
                self._show(prompt, results[-1], i, total)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(self._run_one, p) for p in prompts]
                results = []
                for i, (prompt, fut) in enumerate(zip(prompts, futures)):
                    results.append(fut.result())
                    bar.update(1)
                    self._show(prompt, results[-1], i, total)
        bar.close()
        return results

    @staticmethod
    def _show(prompt: dict, result: PromptResult, i: int, total: int) -> None:
        """Rule, question header, then boxes (tqdm.write keeps the progress bar intact)."""
        prompt_preview = _preview(prompt["text"].replace("\n", " ").strip(), PREVIEW_PROMPT)
        base_content = (
            f"Answer: {_preview(result.final_answer, PREVIEW_ANSWER)}\n\n"
            f"Reasoning: {_preview(result.reasoning_trace, PREVIEW_REASONING)}"
        )
        tqdm.write("")
        tqdm.write(rule())
        tqdm.write(f"Question {i + 1} of {total}")
        tqdm.write(rule())
        tqdm.write(box("Prompt", prompt_preview))
        tqdm.write("")
        tqdm.write(box("Base model", base_content))
        for monitor_name, monitor_data in result.monitor_results.items():
            tqdm.write("")
            tqdm.write(box(f"Monitor: {monitor_name}", _format_monitor(monitor_data)))
