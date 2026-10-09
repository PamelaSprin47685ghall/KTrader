# Release follow-through, 2026-10-08

Scope: validate the latest certificate optimization on the authenticated
complete local N65 input before extending performance work. Then inspect
and measure exact incremental preparation routes. No release/version/tag
change, no relaxed tolerances, no unbounded or whole-backtest command.

Start-of-pass observed SHA256:
```
96faa12404dcb22094b21eb77b14907f20acb237833048b3e0e755c5903488d2  src/response.jl
bab652adfa74847a42033255a8559f1e46d0337313c19c130b5cd57a4a4b246d  src/incremental.jl
263bb4cda147a06514741faa92f8b841a7fe3232e24c710c9d40772206a2fc52  test/contract_registry.jl
1001e09893fdc6bfbe60f730cfff9bb71b2052385222680a25fa725d42b389e9  bin/scoped_run.sh
```
Preflight found no Julia processes and no existing model_certificate output.
Every numerical command uses the unchanged scoped wrapper (45s inner,
55s outer, sampled process-group RSS limit 2048 MiB). Logs below will record
actual outcomes, not prefilled success. Frozen artifacts remain unchanged.

## N65 response/certificate closure

`n65_verify.log`: latest response 96faa124, complete authenticated
T14309/N65/P910/F3 prefix. All 13 pipeline assertions passed. Reported alpha,
G, Sigma, mu, fractional posterior/forecasts and S300 scenario differences
were zero; weight L1=0; original Kelly certificate passed (feasibility
4.44e-16, KKT about 1e-10, objective gap 6.2868e-9). The output
`local_panel_20261008/model_certificate.jls` has SHA256
12ac65841a319379feb678fe3500915f8cb32dfaac4feed7d86ca63eab441589.
The unchanged authenticated baseline/model/decision files were not overwritten.

One explicit same-specialization warm-up preceded one measured solve; BOTH
started without alpha warm starts. Measurement: 1.897362224s, native
compile_time=0, recompile_time=0, cumulative allocation 1,876,100,056 bytes,
GC 0.259854356s. OOF bucket 1.600911348s. This is single-point prepared-solve
latency, not new-day prepare/scenario/Kelly time or multi-day throughput.
Do not subtract compilation or compare this with the previous profiled run
as a controlled whole-solver speedup. This replay preceded the subsequent
pending-filter edit; response/predict/data/prepare remained unchanged after it.

`certificate_micro.log`: actual N65 full statistics, no posterior refit,
all returned certificate fields exactly equal to the frozen old formula.
ABBA median milliseconds: old 13.845 / 15.460, reused 16.412 / 15.849.
Allocations old 41,543,880 -> reused 20,883,592 bytes, excluding already-owned
fit geometry on the reused side. Direct fresh API 34,566,016 bytes. Memory
allocation reduction is established; this sample does NOT establish a latency
improvement. No timing claim is rescued by selecting minima.

## Incremental route observation and the localized change

Both routes consume the SAME prepared artifact's authenticated original price
prefix, never carried marking prices inferred from a NaN count. Its full 65
columns equal the active set. No new data, truncation, posterior fitting or
checkpoint copy is hidden in these commands.

Before: default row limit 4096 routes 7803 materialized rows to exact batch.
Diagnostic limit 8192 permits the incremental contractor. Both have 60 runs,
60 unique masks and actual held Gram payload 146,016,000 bytes. Neither hits
the 512MiB payload budget. The 8192 choice is CLI-local diagnostic configuration,
not a changed production default. Both use one initialization and one prepare.

| route | init before / after, seconds | init allocation before / after, bytes | prepare before / after, seconds |
|---|---|---|---|
| default 4096 | 9.339 / 8.147 | 10,890,480,896 / 7,633,238,864 | 0.083 / 0.276 |
| diagnostic 8192 | 9.113 / 8.111 | 10,890,481,056 / 7,633,239,024 | 0.473 / 0.469 |

Files: route_default.log / route_default_after.log and
route_8192_before.log / route_8192_after.log. Init includes cold compilation
(about 1.2s, separately reported) and GC. The slower default prepare after
the edit includes 0.192s GC; no "every stage faster" claim. Cumulative
initialization allocation fell by about 3.26GB/30%, not peak RSS by 3.26GB.
The fast contractor was not faster in this slice; keep the original default.
No ratio here is extrapolated to all dates or concurrency topologies.

Only `compute_pending` changed in production: compound indexed broadcast
read/copied an N-element column for each lag; explicit qcol/pcol views remove
that temporary, and column packing uses copyto! between views. Exact same
triangular weights, loop/reduction order, band activation, mask keys and
ownership; no state schema, solver, rank, folds or resource-rule changes.
Asymptotic arithmetic remains O(N * sum(BANDS)) per pending row. The saved
temporary-column traffic also scaled with sum(BANDS); remaining allocations
belong to the returned mask-specific prototypes and unchanged state machinery.

`pending_micro.log`: synthetic N65/T420, recurrent masks, exact old/new values.
ABBA medians old 59.63/62.10us; new 30.47/31.75us. Allocation 531,968 ->
60,800 bytes (about 89% reduction) per pending call. Seven samples per entry,
explicit excluded specialization warm-up. This does not measure whole advance.

## Validation and limits

`pending_contracts.log`: new 216/216 pending arithmetic/ownership assertions,
existing primal prepare 46/46 and IPO budget 1041/1041 passed; the final-tree
architecture checks were 25/25. The new file is REQUIRED (28 total registrations,
not 28 suites all run). The reference fixture calculates from raw prefix sums
and mask runs, not from production's pending values or contracted Grams.

Each of the FOUR route observations passed its 28 original-tolerance field
comparisons plus three standalone route/no-solve checks. Default is exact to
the frozen batch for reported fields. Incremental before/after both report
s_perp max diff 3.6415e-14, full_xx relative Frobenius 7.7593e-13, full_xy
4.0778e-12, full_yy 4.8485e-16, at unchanged atol/rtol. Fold statistics,
raw returns/masks, current fields and metadata were checked too; no posterior
or all-cell residual claim is implied by a preparation comparison.

`incremental_regression.log`: attempted the entire existing incremental test
file ONCE. RC124 / killed=1 / elapsed40s / deadline45 / peak1155MiB, timeout
not RSS. Stack points to incremental_tests.jl:439 (final cross-Gram testset)
with LLVM frames; no native compile-time measurement for that command, so
its cause/share is not declared to be compilation. No usable summaries were
flushed; do not infer earlier test counts from the stop line. No assertion
green, whole-file green or release green is claimed; log retained, no deadline
increase or blind repeat. Further validation must use smaller explicit units.

Successful commands: elapsed at most 30s, sampled RSS at most 1965MiB; the
failed whole-file attempt was 40s. The full suite, multiday steady throughput,
and worker/BLAS topology remain open. Nothing here declares 2.0 or GPU
impossibility. No commit/reset/clean and no production default change.
