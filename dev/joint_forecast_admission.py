"""Date-mask-only two-year prequential workload audit; no model/backtest run.

Reconstructs the Gate-0 first resolve and the non-statistical upper bound on
joint residual rank from the input bar calendar. Actual rank checks and the
Kelly/scenario evidence are intentionally NOT inferred by this script.
"""
from collections import Counter
import csv

with open("data/adj.csv", newline="") as file:
    table = csv.reader(file)
    symbols = next(table)[1:]
    rows = list(table)

dates = [row[0] for row in rows]
begin = dates.index("2024-10-01")
end = dates.index("2026-10-02")
N = len(symbols)
valid = lambda s: s not in ("", "NaN", "nan", "NA")
bar = [sum(1 << j for j, value in enumerate(row[1:]) if valid(value))
       for row in rows]
pairs = [0] + [bar[t-1] & bar[t] for t in range(1,len(rows))]
admitted = []
ever = 0
for mask in pairs:
    ever |= mask
    admitted.append(ever)

selected = set()
floor_sizes = Counter()
peeled_by_name = Counter()
ambiguous = []
first_origin = None


def joints(risk, row_ids, require_origin_admitted=False):
    return [u for u in row_ids
            if pairs[u] & risk == risk and
            (not require_origin_admitted or admitted[u-1] & risk == risk)]


def weakest(mask, row_ids):
    choices = [j for j in range(N) if mask & (1 << j)]
    return min(choices, key=lambda j:
               (sum(bool(pairs[u] & (1 << j)) for u in row_ids),j))


for today in range(begin,end):
    row_ids = [u for u in range(257,today+1) if pairs[u] & admitted[today]]
    risk = bar[today] & admitted[today]
    # The first driver resolve (D-056) uses raw paired observation availability.
    while risk and len(joints(risk,row_ids)) < 2:
        risk &= ~(1 << weakest(risk,row_ids))
    risk0 = risk
    discarded = []
    # Any rank-N_R residual covariance needs N_R jointly predicted rows.
    while risk and len(joints(risk,row_ids,True)) < risk.bit_count():
        j = weakest(risk,row_ids)
        discarded.append(j)
        risk &= ~(1 << j)
    if not risk:
        ambiguous.append((dates[today],"empty risk"))
        continue
    if discarded:
        counts = {j:sum(bool(pairs[u] & (1 << j)) for u in row_ids)
                  for j in range(N) if risk0 & (1 << j)}
        gap = min(counts[j] for j in range(N) if risk & (1 << j)) - max(
            counts[j] for j in discarded)
        if gap < 256:
            ambiguous.append((dates[today],f"weak peel gap={gap}"))
        peeled_by_name.update(symbols[j] for j in discarded)
    floor_sizes[risk.bit_count()] += 1
    for u in joints(risk,row_ids,True):
        selected.add(u-1)
    if selected and first_origin is None:
        first_origin = min(selected)

print(f"WINDOW decisions={end-begin} assets={N}")
print(f"JOINT_FORECAST_UNION origins={len(selected)} earliest={dates[min(selected)]}")
print(f"STRUCTURAL_FLOOR sizes={dict(sorted(floor_sizes.items()))}")
print(f"STRUCTURAL_PEELS symbols={dict(peeled_by_name)}")
print(f"AMBIGUOUS_WORKLOAD_DAYS n={len(ambiguous)} sample={ambiguous[:8]}")
print("NOT_CERTIFIED: actual residual ranks, posterior convergence, and "
      "daily decision runtime are not measured by this mask-only audit")
