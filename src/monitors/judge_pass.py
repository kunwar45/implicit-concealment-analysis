"""Post-hoc LLM judging of a saved results JSON.

GPU runs (run_interp, run_framing --local, fir jobs) generate and capture
activations only; this pass adds the judge's verdict afterwards, from any
provider (src/clients/factory.py). The source file is never modified: output is
``<stem>_judged.json`` beside it.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from src.clients import ChatClient
from src.config import Config
from src.monitors.llm_monitor import JUDGE_PROMPT_VERSION, LLMMonitor
from src.pipeline.result import PromptResult


class AlreadyJudgedError(RuntimeError):
    pass


def judged_path(path: str | Path) -> Path:
    p = Path(path)
    return p.with_name(f"{p.stem}_judged{p.suffix}")


def judge_file(
    path: str | Path,
    client: ChatClient,
    monitor_model: str,
    provider: str,
    workers: int = 8,
    force: bool = False,
    out_path: str | Path | None = None,
    judge_max_tokens: int = 300,
) -> Path:
    """Judge every row of a results JSON; return the path of the judged copy."""
    payload = json.load(open(path))
    rows = payload["results"] if isinstance(payload, dict) else payload

    already = [r for r in rows if (r.get("monitor_results") or {}).get("llm", {}).get("prompt_version")]
    if already and not force:
        raise AlreadyJudgedError(
            f"{len(already)}/{len(rows)} rows already carry an answer-only judge verdict; "
            "pass --force to re-judge")

    cfg = Config(monitor_model=monitor_model, judge_max_tokens=judge_max_tokens)
    monitor = LLMMonitor(client, cfg)

    def one(row: dict) -> dict:
        return monitor.run(PromptResult.from_dict(row))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        verdicts = list(pool.map(one, rows))
    for row, v in zip(rows, verdicts):
        row.setdefault("monitor_results", {})["llm"] = v

    if isinstance(payload, dict):
        meta = payload.setdefault("run_metadata", {}) or {}
        payload["run_metadata"] = meta
        meta["judge"] = {
            "provider": provider,
            "model": monitor_model,
            "prompt_version": JUDGE_PROMPT_VERSION,
            "judged_at": datetime.now(timezone.utc).isoformat(),
            "source_file": str(path),
        }

    out = Path(out_path) if out_path else judged_path(path)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out


def verdict_counts(rows: list[dict]) -> Counter:
    return Counter((r.get("monitor_results") or {}).get("llm", {}).get("verdict", "unjudged")
                   for r in rows)
