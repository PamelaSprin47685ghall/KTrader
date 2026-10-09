# CPU kernel work, 2026-10-08

Scope: same posterior, same OOF isolation, cold start and certificates.
User authorized bounded microbenchmarks; no full suite/backtest or full
real-panel N65 inference. The existing conditioned-EB regression file also
checks its frozen t14294 fold (BLAS1 and BLAS6); this is a bounded component
solve, not new price preparation or a full/OOF/day pipeline benchmark.
Initial response.jl SHA256:
`4e20646150641455752d515f460dd46f7773512a773c1791a42733a33a462437`.
Initial registry SHA256:
`543f33820f64dbaf4780a1abfe658d7e1da5f4984a2c240d9d5e3911f015aa23`.
Preflight: AMD Ryzen 5 7500F, 6 cores / 12 logical CPUs; no Julia process.

Static target: alpha-cache repeats fixed gauge/block projection and bcat
packing; the same fit already owns these blocks in scalar geometry.
Share only fit-local read-only geometry. Alpha-dependent weights, R, h,
mutable cores and certificate computation remain independent.
Second target: symmetric-core assembly allocates intermediate matrices
after every GEMM although the result matrix can be symmetrized in place.

First architecture command failed with RC1 because its log directory did
not exist (stderr: scoped_run.sh lines 406/426/302). No successful test
claim from that attempt. This file creates the missing directory; the
command is then repeated with the same 45s inner / 55s outer bounds and
2048 MiB sampled process-group RSS guard. No production guard edit.

Production changes: one shared block projection per fit, one packed copy
for alpha consumers only, independently allocated alpha-weighted arrays and
cores, in-place core symmetrization with the original (a+b)/2 arithmetic.
Direct scalar/certificate calls do not allocate unused packed blocks.
No covariance/rank/iteration-budget/line-search/certificate changes.

Focused verification (`focused_final.log`): 2817 numerical assertions and
25 architecture assertions passed, RC0, elapsed17s, peak960MiB. New file
contributes 973 assertions (the earlier first revision was 965); REQUIRED
registry is now 24 files, which is NOT a claim all 24 ran in this pass.

N65/P910/rank910 synthetic primitive microbenchmarks, explicit warm-up,
five samples per entry, ABBA ordering for alpha construction:

| BLAS threads | alpha before medians (ms) | alpha after medians (ms) | cores before/after (ms) |
|---|---|---|---|
| 1 | 5.956 / 3.442 | 0.890 / 0.889 | 15.221 / 13.944 |
| 6 | 2.305 / 5.208 | 1.582 / 1.631 | 10.960 / 6.022 |

Allocation bytes per alpha: 20,267,632 -> 7,219,480. The extra once-per-fit
packed matrix costs 6,522,968 bytes and measured 0.251ms (BLAS1) / 0.910ms
(BLAS6); it is excluded from per-alpha timings, not hidden as zero cost.
Scalar geometry already existed before this change and remains separately
reported. Core allocation: 13,801,168 -> 3,450,808 bytes (old-formula helper
has a negligible vector-growth difference from the former production loop).
The N65 checks assert exact h/R/bwcat/core equality in both thread settings.
Timing variation is visible; these are primitive observations, not a
whole-fit speedup, full-profile percentage or cross-machine guarantee.

The 14-GEMM blocked candidate remains in the dev probe only. Including
packing it measured 15.289ms (BLAS1) / 7.951ms (BLAS6), slower in median
than the retained in-place version, and allocates 13,416,464 bytes. It is
not promoted despite exact values on this fixture. No GPU or global thread
configuration changes are justified by these primitive-only measurements.

The attempted additional conditioned_eb_tests + conditioned_jcore_tests
command (explicit BLAS6, same 45s/2048MiB guard) was BLOCKED before
execution by the tool safety layer, with no approval token. It was not
retried through a different wrapper. Thus the frozen-fold regression
mentioned in the scope was intended but NOT run in this pass. No new full
solver-certificate or full-suite green is claimed beyond the focused tests.

Final read-only SHA256 observation (after tests; not a fabricated pre/post
binding). No production changes followed this observation:

```text
adde30b78d89e7298f3e33cbf2955eaa80afc4861bac968c0b770d45f2af0546  src/response.jl
0633a183ead4fa60568b824ed04c466c72702c47a26b4027610a23bad2a184f4  test/conditioned_geometry_reuse_tests.jl
dade45ca2d485bcbcfcd89fa02f04c6c983c94716b55a726bb9cce41c77612e8  test/fixtures/conditioned_cpu_reference.jl
a063ab2181dc78ad874b11cbc8f363ad772b00a9ff7cd4039e654d97c4e44578  test/contract_registry.jl
72263b1d1b21222e98ed9ec98c1656c3a1bfdd52d785a60ef6e394cd7f38ce51  dev/conditioned_cpu_micro.jl
```
