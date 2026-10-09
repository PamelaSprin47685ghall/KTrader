# Scalar tensor packing and release validation — 2026-10-08

Layer: CPU engineering, exact same posterior/rows/folds/certificates.
Pre-change response SHA256:
`69733860bb42bd18482619b81262976b95bcbcd625ee3b34f482caac21265ac5`.
Pre-change registry SHA256:
`ed8f409750a6ff63b93a2dfb0951ea64317ca83d19deacfb5c7a9469593830f3`.
Preflight found no Julia processes; checkout HEAD remains 602b897.

Static target: each fixed-Sigma scalar cache allocates nc transformed
d-by-rank matrices and stores both copies of each symmetric tensor entry.
Only r<=s is consumed. Reuse ONE scratch transform during construction and
store rank-by-nc(nc+1)/2 tensor columns. Keep every GEMM, spectral dot,
alpha-node, bracket, covariance update and acceptance formula unchanged.
No cross-call scratch, global cache, new tolerance or solver option.

Expected storage (Float64, d64/rank910/nc14): transforms 6,522,880 ->
465,920 payload bytes; tensor 1,426,880 -> 764,400 payload bytes. These are
shape-derived allocation costs, not measured RSS or throughput claims.
The scratch matrix is not retained by either evidence/derivative closure.

All numerical commands use a 45s inner / 55s outer deadline and a sampled
2048 MiB process-group RSS guard. No full backtest or full-suite command.
Per-file conditioned-EB tests include their existing frozen N65 fold
regression (not a new full-day fit); results are recorded after execution.

## Measured primitives

Julia 1.12.7, znver4/7500F, OpenBLAS. Synthetic N65/P910/rank910,
explicit warm-up, nine samples per entry. ABBA entries below are the two
separately recorded medians, not confidence bounds. Fixed geometry is
excluded on BOTH sides; all per-Sigma work and allocations are included.
Old formulas are retained only in test/fixtures/scalar_tensor_reference.jl.

| BLAS | cache before A/A2 (ms) | cache after B/B2 (ms) | build + 33 derivatives before/after (ms) |
|---|---|---|---|
| 1 | 5.9824 / 6.0801 | 3.6042 / 3.5661 | 4.8498 / 4.3789 |
| 6 | 3.7286 / 3.7428 | 4.3420 / 2.4827 | 3.6502 / 3.6106 |

Cache allocation 8,556,392 -> 1,835,824 bytes (about 78.5% less).
Cache+33 derivatives 9,531,136 -> 2,810,568 bytes. Derivatives-only
allocation remains 974,744 bytes on both paths; this change does not speed
up their algebra. The six-thread cache samples include a slower result,
and the combined six-thread medians are essentially unchanged. Do not
claim a robust six-thread latency win or extrapolate to backtest throughput.
The change is retained for its deterministic allocation reduction, simpler
storage and unchanged numeric outputs, not a cherry-picked best timing.

micro_b1.log and micro_b6.log both assert exact evidence/derivative values
on all 33 alpha nodes, and bind source SHA256 before/after to
`c0d005c5c4d9abde3fb93397af6549f111268e73da0e5b034e85fb538b49ecc2`.
Both RC0/elapsed8s; RSS peaks 906/926MiB.

## Validation actually run

architecture_and_tensor.log: 25 architecture + 2892 numerical assertions,
RC0/elapsed9s/peak945MiB. Packed tensors and all scalar values/derivatives
must agree with the old formulas exactly (isequal), including primal/dual,
with/without gauge, condition numbers 1 and 1e8, empty spectral space,
separate calls and immutable input geometry. Search result equality is
also checked on the well-conditioned cases. No old tolerance was relaxed.
The new test is registered in the 26-file REQUIRED list; registration
does not mean all 26 files were run in this pass.

conditioned_eb_b6.log: 145/145, RC0/elapsed25s/peak1283MiB. This closes the
previously unexecuted conditioned-EB file on the NEW source, including the
existing frozen t14294 N65 fold's BLAS1 regression and actual BLAS6
companion, original covariance/alpha certificates and original tolerances.
It is not the whole test suite or a throughput measurement.

pipeline_verify.log: 12/12, RC0/elapsed23s/peak1224MiB. The SHA-checked
frozen T340/N3/F3/S64 baseline remains unchanged:
`23a95350e58cacfa8cbb87231ad0b22512d5c432440e748a8428aa086dba174e`.
Default EB full+OOF solve, residual/innovation/scenario/Kelly pipeline;
reported alpha/G/Sigma/mu/covariance/d/v/X max differences all zero,
weights satisfy the existing comparison and Kelly certificates. One solve
observation 0.006201259s / 10,907,168 bytes is not a controlled full-fit
performance comparison or a substitute for representative data.

neighbors.log: 1350/1350 (Jacobian 118, lazy gradient 960, fit-local reuse
257, relative support 15), RC0/elapsed18s/peak936MiB. Total this pass:
4399 numerical assertions + 25 architecture assertions, all passed.
Six guarded numerical commands completed: architecture/tensor, two
microbenchmarks, conditioned EB, pipeline replay, neighbors. Longest
logged command elapsed25s, largest sampled process-group RSS1283MiB.

Final read-only diff check passed for response.jl/README.md. Post-validation
source/test/script/log SHA256 values are in final_snapshot.txt (not an
invented per-test pre/post binding; the two micros have their own source
pre/post checks). HEAD=602b897; final process check found no Julia process.
No production source changed after that observation.

## Release boundary

No change to release/version, economic objective, information set, folds,
rank, covariance floor, alpha search, solver budgets or fallback policy.
No commit/reset/clean. Real-fold regression is stronger than synthetic
kernel checks, but neither provides a current full-day N65 pipeline
throughput baseline. Complete current-source release regression and
representative steady-state/incremental routing measurements still gate
the 2.0 declaration. The prior 12307 full-suite result describes old source.
