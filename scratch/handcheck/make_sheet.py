"""Build a blind hand-labelling sheet to validate the disclosure monitors.

Samples N regular (non-control) prompts per results file, stratified by
(condition, regex label, LLM label) so monitor disagreements are included,
then writes:

  sheet.csv : what you label — question, secret, model answer, empty
              `human_disclosed` (y/n/unsure) and `notes`. NO monitor verdicts
              (so they can't bias you).
  key.csv   : the same rows' monitor verdicts, joined later by score_sheet.py.

Usage (repo root):
  python3.11 -m scratch.handcheck.make_sheet \
      results/run_20260403_151721_5f6b66ca.json:25 \
      results/run_20260403_153451_ceed4758.json:25
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from src.monitors.labels import disclosed
from src.storage.hf_artifacts import resolve_artifact

OUT_DIR = Path(__file__).parent / "out"


def _label_str(v) -> str:
    return {True: "disclosed", False: "concealed", None: "none"}[v]


def sample_stratified(results: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Round-robin over strata so rare cells (e.g. regex/LLM disagreements) appear."""
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        meta = r.get("metadata", {})
        if meta.get("control_type") is not None:
            continue
        key = (meta.get("condition"), disclosed(r, "regex"), disclosed(r, "llm"))
        strata[key].append(r)
    for rows in strata.values():
        rng.shuffle(rows)
    keys = sorted(strata, key=str)
    rng.shuffle(keys)
    picked: list[dict] = []
    while len(picked) < n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(picked) < n:
                picked.append(strata[k].pop())
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("sources", nargs="+", help="results.json:N pairs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows: list[tuple[dict, dict]] = []
    for spec in args.sources:
        path, n = spec.rsplit(":", 1)
        payload = json.load(open(resolve_artifact(path)))
        results = payload.get("results", payload)
        for r in sample_stratified(results, int(n), rng):
            rows.append((r, {"run": Path(path).stem}))
    rng.shuffle(rows)  # mix runs/conditions so labelling order carries no signal

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (
        open(out / "sheet.csv", "w", newline="") as fs,
        open(out / "key.csv", "w", newline="") as fk,
    ):
        ws = csv.writer(fs)
        wk = csv.writer(fk)
        ws.writerow(
            ["row", "secret", "question", "model_answer", "human_disclosed", "notes"]
        )
        wk.writerow(
            [
                "row",
                "run",
                "prompt_id",
                "condition",
                "regex",
                "llm",
                "llm_first",
                "llm_evidence",
            ]
        )
        for i, (r, extra) in enumerate(rows, 1):
            meta = r["metadata"]
            ws.writerow(
                [
                    i,
                    " / ".join(meta.get("keyword_hints", [])),
                    r["prompt"],
                    r["final_answer"],
                    "",
                    "",
                ]
            )
            llm = (r.get("monitor_results") or {}).get("llm") or {}
            wk.writerow(
                [
                    i,
                    extra["run"],
                    r["prompt_id"],
                    meta.get("condition"),
                    _label_str(disclosed(r, "regex")),
                    _label_str(disclosed(r, "llm")),
                    _label_str(disclosed(r, "llm_first")),
                    llm.get("evidence") or "",
                ]
            )
    print(f"{len(rows)} rows → {out / 'sheet.csv'}  (key: {out / 'key.csv'})")


if __name__ == "__main__":
    main()
