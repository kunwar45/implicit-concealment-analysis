"""Linear probes on gpt-oss activations vs a text baseline, through the reasoning.

Question: will the answer hide the delivery time? (within M_inst, bad-news contexts)
Controls built in:
  * grouped CV: all samples of one prompt stay in the same fold (no prompt leakage)
  * shuffled-label baseline, shuffled by whole prompt groups
  * text baseline at every position: TF-IDF of exactly the text the model has read/written
    up to that token (prompt at prompt_end, reasoning prefix at r10..r100)
  * headline layer fixed in advance (13, the best layer of the first, independent run)

  python -m scratch.interp_gptoss.probe activations/x.npz --out results/interp_gptoss/v2
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LAYER = 13


def oof(clf, X, y, groups, seed=0):
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    return cross_val_predict(clf, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]


def act_clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced"))


def text_clf():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2),
                         LogisticRegression(C=10, max_iter=2000, class_weight="balanced"))


def boot_ci(y, p, groups, rng, n=1000):
    """95% CI of AUROC, resampling whole prompts."""
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    out = []
    for _ in range(n):
        take = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if 0 < y[take].sum() < len(take):
            out.append(roc_auc_score(y[take], p[take]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shuffles", type=int, default=5)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res_path = os.path.join(args.out, "probe_results.json")
    if args.plot_only:
        plot(json.load(open(res_path)), args.out)
        return

    d = np.load(args.npz)
    A = d["acts"]                                   # [n, positions, layers+1, d]
    positions = [str(p) for p in d["positions"]]
    meta = json.loads(str(d["meta"]))
    y = np.array([m["hidden"] for m in meta]).astype(int)
    groups = np.array([m["prompt_key"] for m in meta])
    ug = np.unique(groups)
    rng = np.random.default_rng(0)
    print(f"rows {len(y)}, hidden {y.sum()}, prompts {len(ug)}")

    def shuffled():
        """Permute labels by whole prompt groups (keeps within-prompt structure)."""
        perm = dict(zip(ug, rng.permutation(ug)))
        by = {g: y[groups == g] for g in ug}
        ys = np.empty_like(y)
        for g in ug:
            src = by[perm[g]]
            k = (groups == g).sum()
            ys[groups == g] = np.resize(src, k)
        return ys

    out = {"n": int(len(y)), "hidden": int(y.sum()), "prompts": int(len(ug)),
           "positions": positions, "headline_layer": LAYER, "by_position": {}}
    for pi, pos in enumerate(positions):
        layers = []
        for layer in range(A.shape[2]):
            p = oof(act_clf(), A[:, pi, layer].astype(np.float32), y, groups)
            layers.append(float(roc_auc_score(y, p)))
            if layer == LAYER:
                p_head = p
        X = A[:, pi, LAYER].astype(np.float32)
        shuf = []
        for s in range(args.shuffles):
            ys = shuffled()
            if 0 < ys.sum() < len(ys):
                shuf.append(roc_auc_score(ys, oof(act_clf(), X, ys, groups, s)))
        if pos == "prompt_end":
            texts = [m["prompt_text"] for m in meta]
        elif pos == "answer_start":
            texts = [m["prefixes"][-1] for m in meta]
        else:
            texts = [m["prefixes"][positions.index(pos) - 1] for m in meta]
        p_text = oof(text_clf(), np.array(texts, dtype=object), y, groups)
        both = np.column_stack([p_head, p_text])
        p_both = oof(LogisticRegression(class_weight="balanced"), both, y, groups, seed=1)
        r = {"act_by_layer": layers,
             "act": float(roc_auc_score(y, p_head)), "act_ci": boot_ci(y, p_head, groups, rng),
             "text": float(roc_auc_score(y, p_text)), "text_ci": boot_ci(y, p_text, groups, rng),
             "act_plus_text": float(roc_auc_score(y, p_both)),
             "shuffled": float(np.mean(shuf))}
        out["by_position"][pos] = r
        print(f"{pos:13} layer {LAYER}: activation {r['act']:.3f} {r['act_ci']}  "
              f"text {r['text']:.3f} {r['text_ci']}  both {r['act_plus_text']:.3f}  "
              f"shuffled {r['shuffled']:.3f}  best layer {int(np.argmax(layers))} "
              f"({max(layers):.3f})", flush=True)
    json.dump(out, open(res_path, "w"), indent=1)
    plot(out, args.out)


def plot(out: dict, out_dir: str) -> None:
    try:
        import matplotlib
    except ImportError:
        print("matplotlib missing: plot skipped (rerun locally with --plot-only)")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pos = out["positions"]
    bp = out["by_position"]
    x = np.arange(len(pos))
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    for key, c, lab in (("act", "C3", f"activation probe (layer {out['headline_layer']})"),
                        ("text", "C0", "text baseline (TF-IDF of text so far)")):
        v = np.array([bp[p][key] for p in pos])
        ci = np.array([bp[p][key + "_ci"] for p in pos])
        a.errorbar(x, v, yerr=[v - ci[:, 0], ci[:, 1] - v], color=c, marker="o", capsize=3,
                   label=lab)
    a.plot(x, [bp[p]["shuffled"] for p in pos], color="gray", ls=":", label="shuffled labels")
    a.axhline(0.5, color="gray", lw=0.5)
    a.set_xticks(x, ["prompt\nend", "10%", "25%", "50%", "75%", "100%", "answer\nstart"])
    a.set(xlabel="position (% of reasoning tokens)", ylabel="grouped 5-fold CV AUROC",
          ylim=(0.4, 1.0), title="When does the decision to hide become readable?")
    a.legend(fontsize=8, loc="upper left")
    for p in pos:
        b.plot(bp[p]["act_by_layer"], marker="o", ms=2, label=p)
    b.axhline(0.5, color="gray", lw=0.5)
    b.set(xlabel="layer (0 = embeddings)", ylabel="AUROC", ylim=(0.4, 1.0),
          title="Activation probe by layer")
    b.legend(fontsize=7, ncol=2)
    fig.suptitle(f"gpt-oss-20b, M_inst, 'confirm my order': n={out['n']} samples of "
                 f"{out['prompts']} prompts, {out['hidden']} hidden", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe_through_reasoning.png"), dpi=150)
    print("saved", out_dir)


if __name__ == "__main__":
    main()
