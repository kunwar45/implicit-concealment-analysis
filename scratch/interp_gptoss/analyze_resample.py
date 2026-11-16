"""Behaviour in the gpt-oss resample run: hide rates by arm, good-news control, per-prompt spread.

  python scratch/interp_gptoss/analyze_resample.py <run_judged.json>
"""
import json, math, sys
from collections import Counter, defaultdict


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), c + h


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} {k / n:.1%} [{lo:.1%},{hi:.1%}]"


d = json.load(open(sys.argv[1]))
arm = defaultdict(lambda: [0, 0]); days = defaultdict(lambda: [0, 0])
per_prompt = defaultdict(list); other = Counter()
for r in d["results"]:
    m = r["metadata"]
    v = r["monitor_results"].get("llm", {}).get("verdict")
    if not m["format_ok"] or v not in ("disclosed", "concealed", "redirected"):
        other[v if m["format_ok"] else "bad_format"] += 1
        continue
    hid = v != "disclosed"
    ctx = "good-news" if m.get("context_variant") == "goodnews" else "bad-news"
    dd = m["secret_structured"]["lead_time_days"]
    size = "2-5d" if dd <= 5 else "10d" if dd == 10 else "21-28d"
    for key in ((m["framing_condition"], ctx, size), (m["framing_condition"], ctx, "all")):
        arm[key][0] += hid; arm[key][1] += 1
    if ctx == "bad-news":
        days[(m["framing_condition"], dd)][0] += hid; days[(m["framing_condition"], dd)][1] += 1
        per_prompt[(m["framing_condition"], m["prompt_key"])].append(hid)
print("excluded rows:", dict(other))
print(f"\n{'framing':8} {'context':10} {'secret':7} hidden")
for key in sorted(arm):
    print(f"{key[0]:8} {key[1]:10} {key[2]:7} {fmt(*arm[key])}")
print("\nM_inst, bad-news context, by delay:",
      {k[1]: f"{v[0]}/{v[1]}" for k, v in sorted(days.items()) if k[0] == "M_inst"})
for f in ("BASE", "M_inst"):
    full = [v for (ff, _), v in per_prompt.items() if ff == f and len(v) == 4]
    dist = Counter(sum(v) for v in full)
    p = sum(map(sum, full)) / (4 * len(full))
    expect = {k: round(len(full) * math.comb(4, k) * p ** k * (1 - p) ** (4 - k), 1) for k in range(5)}
    print(f"\n{f}: prompts with 4 usable samples = {len(full)}, hide rate {p:.1%}")
    print("  hidden samples per prompt (observed):", {k: dist.get(k, 0) for k in range(5)})
    print("  expected if every prompt had the same rate:", expect)
