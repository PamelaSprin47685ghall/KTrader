# OOF memory work, 2026-10-09

Final interpretation, measurements and failure records are in `summary.md`.
Only `qualification/final/` qualifies the final source/benchmark protocol;
the earlier parent qualification logs remain historical observations.

Existing master checkout, HEAD 602b897; no commit/reset/clean/tag. Initial
response SHA256 aa0d5bad090ba08de62c1456d0f920cef8b724f7a4a573565abae23495241f9c,
backtest 087b1d0885d10918de7265bb14c6ffb23b5fdf7b4674f6d3ed8d71d95aae984d.
No Julia processes in the initial read-only ps observation (RC1/no matches).

Target: fit-local storage for repeatedly produced weighted blocks, direct
constraint product and Jacobian cores. No repeated alpha/Sigma state reused
as a numerical answer; buffers are rewritten before use. Direct/public
cache constructors remain independently owned. No new public setting/type,
cross-fold workspace, rank cutoff, iteration limit or numerical tolerance.

Before/after production window records are written with complete source and
local CSV hashes, refuse overwrite, and compare the same complete histories.
This directory's new logs, not old source receipts, establish current status.
Every numerical command uses the unchanged 45s inner / 55s outer deadline
and sampled 2048MiB process-group RSS guard. No result is prefilled.

Before-change serial four-day replay completed with native compile=0:
8.232587813s / 8,112,685,976 cumulative bytes / 1.815692219s GC, peak1880MiB.
The p2 baseline used an explicit process-only 1024M heap-size hint to contain
peak memory. It timed out in real warmup (RC124, elapsed35s, peak1789MiB,
176 GC cycles), so it provides NO measured rate. Do not compare hint-on
and hint-off timings as an algorithm speedup. Original failure log retained.

The first new storage test had a harness error in its isolation assertion:
it compared the original cold/direct M result against a second call after
cores were built. All cold before/after and certificate comparisons passed,
but these 32 cold/warm self-comparisons did not. The assertion now snapshots
the independent warm-core result BEFORE poisoning scratch and compares the
same route afterwards; isequal and all existing numeric tests are unchanged.
This is not a production numerical repair. The failure log is retained.

Parallel four-day resource exploration: 1024M heap hint (old runtime) timed
out; 1280M (new runtime) completed warmup but timed out during measurement;
default GC (new runtime) completed warmup in 5.337s but the measured replay
hit RSS guard at2062MiB. These are nonpassing command observations, not
accepted throughput. Same 45s/55s/2048MiB guards in every case.

At1408M the new four-day p2 workload completed and replayed deterministically
(10.286s, GC5.940s, peak1895MiB), but the new comparison harness used the
same-topology return gate on a different topology: one assertion failed at
max return delta ~6.34e-11. The original cross-topology gate is 1e-9 in
dev/batch_window.jl; weights already passed the original per-day L1<=1e-7.
The harness now distinguishes the two comparison scopes, preserving each
original gate. It also saves a validated observation BEFORE comparison,
so future comparison failures do not destroy outputs. Runtime unchanged.
The failed log is retained; a fresh run is needed for a success claim.

Julia's memory-management manual documents heap hints as more aggressive GC
triggers, not process RSS limits (docs.julialang.org/en/v1/manual/memory-management/).
Only process-local hints were varied; production defaults and the sampled
RSS kill limit were not changed. No speedup attribution across different hints.

The direction producer also fuses only elementwise additions/subtractions,
leaving S*J*S, S*S and the left-associated scalar formula unchanged. Frozen
eager-state tests must verify that this removes temporaries, not precision.
