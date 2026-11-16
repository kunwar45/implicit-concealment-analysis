"""Escalation chains: disclosure by customer turn (t0..t3) per framing."""
import json, sys
from collections import Counter, defaultdict
d = json.load(open(sys.argv[1]))
cell = defaultdict(Counter); cons = defaultdict(Counter)
for r in d['results']:
    m = r['metadata']; llm = r['monitor_results'].get('llm', {})
    v = llm.get('verdict', 'missing')
    if not r['final_answer'].strip(): v = 'empty'
    if not m['saw_estimate']: v = 'no_estimate'
    k = (m['framing_condition'], int(m['turn']))
    cell[k][v] += 1
    if v in ('concealed', 'redirected'):
        cons[k][bool(llm.get('reasoning_considers_concealing'))] += 1
fr = sorted({k[0] for k in cell}, key=lambda f: (f == 'EXPLICIT', f))
print(f"{'framing':14}" + "".join(f"{'cust'+str(t+1):>12}" for t in range(4)) + "   (hidden/n; n excludes no-estimate/empty)")
for f in fr:
    row = ''
    for t in range(4):
        c = cell[(f, t)]; n = c['disclosed'] + c['concealed'] + c['redirected'] + c['unclear']
        hid = c['concealed'] + c['redirected']
        row += f"{f'{hid}/{n}':>12}"
    extra = sum((cell[(f, t)]['no_estimate'] + cell[(f, t)]['empty']) for t in range(4))
    print(f"{f:14}{row}   skipped={extra}")
print("\nhidden cases where reasoning considered hiding:",
      {f: f"{sum(cons[(f,t)][True] for t in range(4))}/{sum(sum(cons[(f,t)].values()) for t in range(4))}" for f in fr})
