"""Judge a saved results JSON with the LLM judge (post-hoc; the only disclosure label).

  python -m scripts.judge_results results/run_X.json \
      --monitor-provider openrouter --monitor-model <model> [--workers 8] [--force]

Writes results/run_X_judged.json (the source file is never modified). Use for any
run made without an inline judge: run_interp, run_framing --local, fir/GPU jobs —
and to re-label old regex-era runs.
"""

from __future__ import annotations

import argparse
import json
import sys

from src.clients.factory import PROVIDERS, make_client
from src.monitors.judge_pass import AlreadyJudgedError, judge_file, verdict_counts
from src.storage.hf_artifacts import resolve_artifact


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="results JSON (fetched from the HF dataset repo if missing)")
    ap.add_argument("--monitor-provider", choices=PROVIDERS, required=True)
    ap.add_argument("--monitor-model", required=True)
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel judge calls (default 8 for API providers, 1 for hf)")
    ap.add_argument("--judge-max-tokens", type=int, default=300,
                    help="per call; raise for reasoning-model judges")
    ap.add_argument("--force", action="store_true", help="re-judge rows that already have a verdict")
    ap.add_argument("--out", default=None, help="output path (default <stem>_judged.json)")
    args = ap.parse_args()

    workers = args.workers or (1 if args.monitor_provider == "hf" else 8)
    try:
        client = make_client(args.monitor_provider, args.monitor_model)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    src = resolve_artifact(args.results)
    try:
        out = judge_file(src, client, args.monitor_model, args.monitor_provider,
                         workers=workers, force=args.force, out_path=args.out,
                         judge_max_tokens=args.judge_max_tokens)
    except AlreadyJudgedError as e:
        sys.exit(f"Error: {e}")

    rows = json.load(open(out))["results"]
    print(f"Judged {len(rows)} rows → {out}")
    for verdict, n in sorted(verdict_counts(rows).items()):
        print(f"  {verdict:<12} {n}")


if __name__ == "__main__":
    main()
