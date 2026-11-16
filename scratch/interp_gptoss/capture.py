"""Capture gpt-oss-20b residual activations on a judged resample run (generate.py).

Teacher-forces exactly the tokens the model saw and wrote (rendered prompt + raw output up
to the start of the final answer) and saves every layer's hidden state at:
  prompt_end                 last prompt token (before any reasoning)
  r10, r25, r50, r75, r100   10%..100% of the way through the reasoning tokens
  answer_start               last token before the first answer token
For each reasoning position the reasoning text up to that token is saved too, so a
text-only baseline can be fitted on exactly what the activation could have read.
Label = the judge's verdict on the answer that followed.

  python -m scratch.interp_gptoss.capture <run_judged.json> --out activations/x.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ANALYSIS_OPEN = "<|channel|>analysis<|message|>"
FINAL_OPEN = "<|end|><|start|>assistant<|channel|>final<|message|>"
FRACS = (0.10, 0.25, 0.50, 0.75, 1.00)
POSITIONS = ["prompt_end"] + [f"r{int(f * 100)}" for f in FRACS] + ["answer_start"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--conditions", default="M_inst")
    ap.add_argument("--limit", type=int, default=0, help="rows (0 = all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--show", type=int, default=1)
    args = ap.parse_args()

    conds = args.conditions.split(",")
    rows = []
    for r in json.load(open(args.results))["results"]:
        m = r["metadata"]
        v = r["monitor_results"].get("llm", {}).get("verdict")
        if (m["framing_condition"] in conds and m.get("context_variant") in (None, "None")
                and m["format_ok"] and v in ("disclosed", "concealed", "redirected")):
            rows.append(r)
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} rows, {sum(r['monitor_results']['llm']['verdict'] != 'disclosed' for r in rows)} hidden")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto")
    model.eval()

    def ids(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    acts, meta = [], []
    for i, r in enumerate(rows):
        m = r["metadata"]
        raw = m["raw_output"]
        cut = raw.index(FINAL_OPEN)
        assert raw.startswith(ANALYSIS_OPEN)
        p_ids = ids(m["rendered_prompt"])
        r_ids = ids(m["rendered_prompt"] + raw[:cut])                  # … last reasoning token
        f_ids = ids(m["rendered_prompt"] + raw[:cut + len(FINAL_OPEN)])
        assert r_ids[:len(p_ids)] == p_ids and f_ids[:len(r_ids)] == r_ids, "not a token prefix"
        n_p, n_r = len(p_ids), len(r_ids)
        n_open = len(ids(ANALYSIS_OPEN))
        body = n_r - n_p - n_open                                      # reasoning content tokens
        pos = [n_p - 1] + [n_p + n_open + max(1, round(f * body)) - 1 for f in FRACS] \
            + [len(f_ids) - 1]
        prefixes = [tok.decode(r_ids[n_p + n_open:p + 1]) for p in pos[1:-1]]
        if i < args.show:
            print(f"\n===== INPUT {r['prompt_id']} (prompt {n_p}, reasoning {body} tokens; "
                  f"positions {pos}) =====\n{tok.decode(f_ids)}\n===== r25 PREFIX =====\n"
                  f"{prefixes[1]}\n===== ANSWER =====\n{r['final_answer'][:300]}\n"
                  f"===== VERDICT: {r['monitor_results']['llm']['verdict']} =====\n")
        with torch.no_grad():
            hs = model(torch.tensor([f_ids], device=model.device),
                       output_hidden_states=True).hidden_states
        hs = torch.stack([h[0, pos] for h in hs], dim=1)               # [positions, layers+1, d]
        acts.append(hs.to(torch.float16).cpu().numpy())
        llm = r["monitor_results"]["llm"]
        meta.append({"prompt_id": r["prompt_id"], "prompt_key": m["prompt_key"],
                     "days": m["secret_structured"]["lead_time_days"],
                     "hidden": llm["verdict"] != "disclosed",
                     "considered": bool(llm.get("reasoning_considers_concealing")),
                     "prompt_text": m["system_prompt"] + "\n" + r["prompt"],
                     "prefixes": prefixes, "reasoning_tokens": body})
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)

    A = np.stack(acts)  # [n, positions, layers+1, d]
    np.savez(args.out, acts=A, positions=np.array(POSITIONS), meta=json.dumps(meta))
    print(f"saved {args.out}: {A.shape}")


if __name__ == "__main__":
    main()
