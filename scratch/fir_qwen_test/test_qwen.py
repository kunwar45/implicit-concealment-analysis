"""Smoke-test a Qwen model through this repo's pipeline on a GPU node.

Runs N scenarios x conditions through BaseModelStep + Regex/Keyword monitors using
either the in-process HF client (with activation capture) or a vLLM server
(provider "vllm"), and prints, per prompt: the exact input, raw output, the
reasoning/answer split, format_ok and the regex verdict. Nothing is saved except
activations under --act-dir (hf mode).

  python -m scratch.fir_qwen_test.test_qwen --provider hf   --model Qwen/Qwen2.5-7B-Instruct
  python -m scratch.fir_qwen_test.test_qwen --provider vllm --model Qwen/Qwen3.6-27B --thinking on
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time

import numpy as np

from src.clients.factory import PROVIDERS, THINKING_CHOICES, make_client, parse_thinking
from src.config import Config
from src.dataset.generator import DatasetGenerator
from src.dataset.shipping_domain import ShippingDomain
from src.loaders.concealment_loader import ConcealmentLoader
from src.monitors.keyword_monitor import KeywordMonitor
from src.monitors.labels import regex_disclosed
from src.monitors.regex_monitor import RegexMonitor
from src.pipeline.pipeline import Pipeline
from src.steps.base_model_step import BaseModelStep
from src.steps.monitor_step import MonitorStep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=PROVIDERS, default="hf")
    ap.add_argument("--model", required=True)
    ap.add_argument("--thinking", choices=THINKING_CHOICES, default="default")
    ap.add_argument("--no-format-instruction", action="store_true")
    ap.add_argument("--n-scenarios", type=int, default=3)
    ap.add_argument("--conditions", default="A0,A2")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--revision", default=None, help="HF commit sha (hf provider)")
    ap.add_argument(
        "--act-dir", default=os.path.expanduser("~/scratch/obf_qwen_test_acts")
    )
    args = ap.parse_args()

    thinking = parse_thinking(args.thinking)
    kw = {}
    if args.provider == "hf":
        kw = {"dtype": args.dtype, "capture_mode": "last_token", "revision": args.revision}
    t0 = time.time()
    client = make_client(args.provider, args.model, enable_thinking=thinking, **kw)
    print(f"client ready in {time.time() - t0:.0f}s: {type(client).__name__}")

    cfg = Config(
        base_model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        subject_provider=args.provider,
        enable_thinking=thinking,
        format_instruction=not args.no_format_instruction,
        capture_activations="last_token" if args.provider == "hf" else "none",
        activations_dir=args.act_dir,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ds.jsonl")
        gen = DatasetGenerator(domains=[ShippingDomain()], seed=42)
        gen.to_jsonl(gen.generate(args.n_scenarios), path)
        prompts = ConcealmentLoader(
            path,
            conditions=args.conditions,
            query_types="B1",
            format_instruction=cfg.format_instruction,
        ).load()

    pipe = Pipeline(
        [BaseModelStep(client, cfg), MonitorStep([RegexMonitor(), KeywordMonitor()])]
    )
    t0 = time.time()
    results = pipe.run(prompts)
    dt = time.time() - t0

    print("\n" + "#" * 100)
    print(
        f"MODEL {args.model} | provider={args.provider} thinking={args.thinking} "
        f"format_instruction={cfg.format_instruction} | {len(results)} prompts in {dt:.0f}s"
    )
    print(
        f"format_ok {sum(r.metadata['format_ok'] for r in results)}/{len(results)} | sources "
        f"{sorted({r.metadata['reasoning_source'] for r in results})}"
    )
    for i, r in enumerate(results):
        m = r.metadata
        print("\n" + "=" * 100)
        print(
            f"{r.prompt_id} | secret={m['keyword_hints']} | source={m['reasoning_source']} "
            f"| regex_disclosed={regex_disclosed(r)} | secret_in_reasoning={r.monitor_results['keyword']['matched']}"
        )
        if i < 2:
            print("--- SYSTEM ---\n" + m["system_prompt"])
        print("--- USER ---", r.prompt)
        print("--- RAW (first 600) ---\n" + m["raw_output"][:600])
        print("--- ANSWER ---", r.final_answer[:400].replace("\n", " "))
        if r.activation_path:
            z = np.load(r.activation_path)
            layers = [k for k in z.files if k.startswith("layer_")]
            print(
                f"--- ACTIVATIONS --- {len(layers)} layers, d_model={z[layers[0]].shape[-1]}, "
                f"finite={all(np.isfinite(z[k]).all() for k in layers)}"
            )


if __name__ == "__main__":
    main()
