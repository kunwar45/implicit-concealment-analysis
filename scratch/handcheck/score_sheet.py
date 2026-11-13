"""Score a filled hand-labelling sheet against the monitor verdicts.

Fill `human_disclosed` in sheet.csv with y / n / unsure, then:
  python3.11 -m scratch.handcheck.score_sheet [--out-dir scratch/handcheck/out]

Prints, for each rule (regex, llm, llm_first): agreement with you on rows you
marked y/n, a 2x2 confusion table, and every disagreeing row.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

OUT_DIR = Path(__file__).parent / "out"
HUMAN = {"y": "disclosed", "yes": "disclosed", "n": "concealed", "no": "concealed"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out_dir)

    sheet = {r["row"]: r for r in csv.DictReader(open(out / "sheet.csv"))}
    key = {r["row"]: r for r in csv.DictReader(open(out / "key.csv"))}

    labelled = {
        i: HUMAN[s["human_disclosed"].strip().lower()]
        for i, s in sheet.items()
        if s["human_disclosed"].strip().lower() in HUMAN
    }
    skipped = len(sheet) - len(labelled)
    print(f"{len(labelled)} rows labelled y/n ({skipped} blank/unsure skipped)\n")
    if not labelled:
        return

    for rule in ("regex", "llm", "llm_first"):
        pairs = [
            (h, key[i][rule]) for i, h in labelled.items() if key[i][rule] != "none"
        ]
        agree = sum(h == m for h, m in pairs)
        print(
            f"== {rule}: {agree}/{len(pairs)} agree with you"
            + (f" ({agree / len(pairs):.0%})" if pairs else "")
        )
        print(f"   {'':>18} {'monitor disc':>13} {'monitor conc':>13}")
        for h in ("disclosed", "concealed"):
            print(
                f"   you: {h:<12} {sum(1 for a, m in pairs if a == h and m == 'disclosed'):>13}"
                f" {sum(1 for a, m in pairs if a == h and m == 'concealed'):>13}"
            )
        for i, h in labelled.items():
            m = key[i][rule]
            if m != "none" and m != h:
                ans = sheet[i]["model_answer"].replace("\n", " ")[:140]
                print(f"   row {i} [{key[i]['prompt_id']}] you={h} monitor={m}: {ans}")
        print()


if __name__ == "__main__":
    main()
