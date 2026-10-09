# Continue toward 2.0 — 2026-10-08

Starting runtime: response.jl SHA256
adde30b78d89e7298f3e33cbf2955eaa80afc4861bac968c0b770d45f2af0546.
Registry SHA256 a063ab2181dc78ad874b11cbc8f363ad772b00a9ff7cd4039e654d97c4e44578.
No Julia process at preflight. Existing user changes remain uncommitted.

Scope: bounded CPU primitive experiments and exact implementation work.
No release/version bump, long backtest, GPU installation, tolerance change
or workaround for the previously blocked broad EB regression command.
All commands use <=45s inner / <=55s outer and sampled RSS <=2048MiB.
The old full-suite green predates the starting runtime and is not a new
runtime release certificate. Completion requires evidence, not a tag.

Static experiment: all 105 symmetric Jacobian cores are blocks of
U * Diagonal(w) * U' with U the vertically stacked constraint blocks.
Compare larger GEMM, sign-definite weighted SYRK and the current small
GEMMs, including packing and extraction. SYRK uses sqrt(abs(w)) and so
has different floating-point summation; it is not assumed bit-identical.
Candidates remain dev-only unless both benefit and correctness are shown.

## Decisions and production change

The larger-matrix candidate did NOT earn production admission. With BLAS6,
the existing core loop measured 6.924ms / 3.451MB. Reused-layout GEMM
measured 6.693ms / 16.396MB; including packing, 12.328ms / 22.919MB.
Reused-layout SYRK measured 7.343ms / 16.404MB. The extra memory and packing
cost outweigh that single noisy GEMM difference. See `core_layout_b6.log`;
this is a local rejection, not a proof that no batching method can improve.

Instead, response.jl now separates value/J/natural-direction evaluation
from the spectral full-gradient reconstruction. The optimizer needs only
value/J/direction for its current point and fixed-point candidate. It now
forms grad only in the line-search branch that actually reads it, reusing
that branch's already-computed eigendecomposition of the SAME S. The public
`conditioned_cached_state(...;gradient=true)` still returns the original
fields and formulas, and final `conditioned_eb_certificate` still uses it.
Cold start, iterations, floor, OOF, evidence acceptance, backtracking and
certificates are unchanged. No cached geometry is mutated or shared across
fits. This removes up to two unused full gradients per outer iteration;
it does NOT remove the other eigen/Cholesky operations or kernel assembly.

## Measured scope

`lazy_gradient_b1.log` and `lazy_gradient_b6.log`: synthetic N65/rank910,
randomly rotated SPD covariances, both free and 19 floor-active directions.
Cores are PREBUILT; their assembly cost is not included in these state
measurements. Seven samples after explicit warm-up; ABBA state ordering.

| Normal SPD candidate | Before | After |
|---|---:|---:|
| State only, BLAS1 | 0.8006–0.8023ms | 0.3868–0.3891ms |
| State only, BLAS6 | 0.8812–0.9029ms | 0.4267–0.4450ms |
| State plus UNCHANGED certificate, BLAS1 | 1.4303ms | 1.0865ms |
| State plus UNCHANGED certificate, BLAS6 | 1.8324ms | 1.1425ms |
| State allocation | 822720 bytes | 437432 bytes |

Ranges above are the two ABBA medians, not confidence intervals. With
floor-active directions, state-plus-certificate medians were 1.8247 to
1.3573ms (BLAS1), and 1.8931 to 1.4411ms (BLAS6). Timing variability is
visible in the logs. These are component measurements, NOT a fit/backtest
speedup or a measured global hotspot percentage. The full public gradient
API allocates 96 additional bytes in this microbenchmark; it is preserved
for callers that request it, not advertised as an optimized endpoint.

## Verification performed on the new source

* `lazy_gradient_tests.log`: 960 new eager/lazy assertions plus 973 existing
  geometry/core assertions, and 25 architecture assertions; RC0 / 12s.
* `algebra_neighbors.log`: 228 deadwork + 1359 contraction + 257 fit-local
  reuse assertions, RC0 / 15s.
* `predictive_neighbors.log`: 40 moving-fold warm/cold + 25 lazy/dense
  quadrature + 45 residual-flow assertions, RC0 / 30s.
* `pipeline_before.log` / `pipeline_after.log`: a BEFORE-change output was
  saved for a T340/N3/F3/S64 synthetic ragged panel, default EB. New-source
  verification checked the saved bytes' SHA256 BEFORE deserialization:
  `23a95350e58cacfa8cbb87231ad0b22512d5c432440e748a8428aa086dba174e`.
  Full/OOF solve -> predictive scenarios -> Kelly passed 12 comparisons.
  Reported alpha/G/Sigma/mu/covariance/d/v/scenario max differences were
  zero; weights passed the existing-style 1e-9 tolerance, Kelly feasibility
  2.22e-16, KKT 1.00e-9, objective gap 2.00e-9. Both phases RC0 / 23s.
  The two warmed solve observations (6.47 and 6.43ms) are a correctness
  companion, not evidence of meaningful whole-pipeline acceleration.

Total new-source coverage in this pass: 3899 numerical assertions and 25
architecture assertions. REQUIRED now has 25 files; registration is not
equivalent to running all 25. Successful commands had 45s inner / 55s
outer limits, RSS guard2048; observed maximum command elapsed30s and
sampled peak1224MiB. No full suite, backtest or frozen real N65 EB regression
was run; the prior blocked broad command was not worked around.

## Release boundary

This is continued CPU engineering, NOT a 2.0 sign-off. Remaining release
evidence includes current-source whole mathematical/integration regression,
representative provenance-known full/OOF daily inputs, measured steady-state
CPU/task topology and incremental routing costs, and a cost/benefit ledger
for the remaining major buckets. Only then can a device decision be attached
to an actual workload. A successful component optimization does not certify
these independent requirements; neither does renaming the version.

Final observed source/test identities (SHA256; no source changes after this
observation, not a claim of per-command pre/post hash capture):

```text
69733860bb42bd18482619b81262976b95bcbcd625ee3b34f482caac21265ac5  src/response.jl
ed8f409750a6ff63b93a2dfb0951ea64317ca83d19deacfb5c7a9469593830f3  test/contract_registry.jl
974e869ea63b05ae1d719155b0ec2871fedcbcd837ca0421f1f0d1a014f63309  test/conditioned_lazy_gradient_tests.jl
6e40e495f1873aa399027b00a0ad11609b6c3688d7b490a4a0a96c6953bfe207  test/fixtures/conditioned_eager_state_reference.jl
3929df17ff96a6a8a60e6dd1a1216f4f51d4399b3b9775ac22464d2b53c3d3d9  dev/release_small_pipeline.jl
5cb4c6d5482a5229246bbac11dd57844c25c227652c939b3042bce0303a6f9af  dev/lazy_gradient_micro.jl
```
