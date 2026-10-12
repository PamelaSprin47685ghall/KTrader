"""Conservative planning aid for KTrader's *unlaunched* two-year run.

Reads a bounded real historical-forecast timing probe and the unchanged input
market CSV. Counts every required prequential forecast at its own historical
active-set dimension. Interpolates measured steady-state times by dimension.

This is an *estimate*, never a certificate of runtime, scenario convergence,
or numerical precision; no capital/trading/backtest calculation occurs here.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import re
from collections import Counter
from pathlib import Path


ROW = re.compile(
    r"SERIES_OK lap=(\d+) date=(\S+) N=(\d+) n=(\d+) "
    r"prep=([\d.]+) posterior=([\d.]+) total=([\d.]+)"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, type=Path)
    p.add_argument("--prices", default=Path("data/adj.csv"), type=Path)
    p.add_argument("--start", default="2024-10-01")
    p.add_argument("--end", default="2026-10-02")
    p.add_argument("--budget-seconds", type=float, default=3600)
    args = p.parse_args()

    samples = {}
    for text in args.log.read_text().splitlines():
        match = ROW.search(text)
        if match and int(match.group(1)) == 2:
            samples[int(match.group(3))] = float(match.group(7))
    if len(samples) < 6:
        raise RuntimeError("At least 6 steady-state dimensions are required")
    dimensions = sorted(samples)
    with args.prices.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = list(reader)
    dates = [r[0] for r in rows]
    start = dates.index(args.start)
    end = dates.index(args.end)
    if end - start < 400:
        raise ValueError("This admission calculation is only for a true two-year window")
    N = len(header) - 1
    first_return = [len(rows)] * N

    def good(x):
        return x not in ("", "NaN", "nan", "NA")

    for j in range(N):
        for t in range(1, len(rows)):
            if good(rows[t-1][j+1]) and good(rows[t][j+1]):
                first_return[j] = t
                break
    by_dim = Counter()
    # Mirroring _prime_historical_forecasts!'s origin interval exactly:
    # target row u=WARMUP+1..first_t, origin=u-1. WARMUP=256.
    for u in range(256, start + 1):
        # u is zero-indexed target row; origin=u-1.
        origin = u - 1
        if origin < 1:
            continue
        if not any(first_return[j] <= origin for j in range(N)):
            continue
        if not any(good(rows[u][j+1]) and good(rows[u-1][j+1])
                   for j in range(N)):
            continue
        by_dim[sum(s <= origin for s in first_return)] += 1
    # Later decision dates require only one new historical forecast each,
    # while the last valuation session has no decision or new forecast.
    for origin in range(start, end-1):
        u = origin + 1
        if any(good(rows[u][j+1]) and good(rows[u-1][j+1]) for j in range(N)):
            by_dim[sum(s <= origin for s in first_return)] += 1

    def sec_for(n):
        pos = bisect.bisect_left(dimensions, n)
        if pos == 0:
            return samples[dimensions[0]]
        if pos == len(dimensions):
            return samples[dimensions[-1]]
        n0, n1 = dimensions[pos-1:pos+1]
        ratio = (n - n0) / (n1 - n0)
        return samples[n0]*(1-ratio) + samples[n1]*ratio

    estimate = sum(sec_for(n) * count for n, count in by_dim.items())
    print(f"INPUT {args.start}..{args.end} symbols={N} decisions={end-start}")
    print(f"PREQUENTIAL_ORIGINS counted={sum(by_dim.values())} unique_dimensions={len(by_dim)}")
    print("MEASURED_STEADY_SECONDS_BY_DIMENSION", sorted(samples.items()))
    print(f"HISTORICAL_FORECAST_ONLY interpolated_serial_seconds={estimate:.1f} "
          f"({estimate/60:.1f}min)")
    print("OMITTED_FROM_THIS_NUMBER: 502 daily posterior fits, innovation d "
          "integration, adaptive RQMC/Kelly/certificates, launch/checkpoints")
    if estimate >= args.budget_seconds:
        print("ADMISSION_BLOCKED: historical forecasts alone exceed one-hour budget "
              "at interpolated sampled serial throughput")
    else:
        print("ADMISSION_BLOCKED: this partial cost estimate cannot certify "
              "the complete <=3600s run")


if __name__ == "__main__":
    main()
