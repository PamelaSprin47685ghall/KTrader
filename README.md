# KTrader — Path Kelly 1.0 (Julia)

Price history → center-of-mass macro field and relative field → causal multiscale
path coordinates → trace-neutral response posterior → fractional innovation
mixture → posterior-predictive gross returns → exact sample log-Kelly → execution.

The account problem is long-only, fully invested, leverage one, with **no cash
asset**. Fees and execution friction do not enter the theoretical objective.

## Statistical definitions

- Signal prices and raw returns retain `NaN` for unobserved bars. Marking prices
  are separate. Prefix rulers count only observed endpoint pairs, use each
  asset's entire available prefix, and map active columns to original-universe
  columns explicitly.
- An asset enters the model only after its first observed consecutive-price
  return. Activity comes from `model.active_indices`, never from scenario values.
  Inactive scenario columns are `NaN`, not risk-free assets. A genuinely held
  inactive asset with no predictive law raises an error rather than receiving an
  invented constant return.
- For observed normalized returns, the macro field is their sum divided by the
  square root of the observed count. The relative field subtracts their observed
  mean. **The current model explicitly defines absent relative-field components
  as zero in the common coordinate embedding.** Raw returns remain missing, and
  `model.relative_observed` retains their mask. The cumulative embedded field is
  not an inferred latent price path. This chooses the explicit-zero-embedding
  branch; it is **not** a mask-marginalized asset-return likelihood. Such a
  likelihood would not retain a shared Matrix-Normal `V ⊗ Sigma` posterior.
- Relative output covariance lives in the exact `N-1` dimensional zero-sum
  subspace. A Helmert basis is only a numerical coordinate system; its choice
  does not define sectors or affect asset-coordinate predictions. The existing
  `1e-8` covariance floor applies inside that subspace, not to an artificial
  noisy macro direction. With one asset the relative space is exactly empty.
- Each multiscale relative operator has zero trace in both channels, imposed on
  the Gaussian parameter support. Empirical Bayes optimizes the corresponding
  conditional-support evidence:

  $$\log p(Y\mid Cg=0)=\log p(Y)+\log p(Cg=0\mid Y)-\log p(Cg=0).$$

  Alpha searches use the existing `[1e-4, 1e6]` bounds and bracketed stationary
  points. Covariance updates and scalar roots have explicit convergence checks;
  failures are errors, not silently accepted fits. As with empirical Bayes
  generally, this is not a proof of a global optimum of a nonconvex evidence.
- OOF folds have independent macro and relative EB fits. `F_folds` supports any
  value from two through the number of training rows. Residuals use the same
  **next-return index** as the regression target. Missing realized returns never
  become bootstrap observations.
- The fractional parameter has the existing trapezoidal-quadrature prior on
  `DGRID_V1`. Gaussian predictive uncertainty, discrete fractional mixture, and
  joint residual-row bootstrap define the predictive law. Missing members of a
  sampled row are drawn from that asset's observed residual rows.

## Exact numerical implementation

These choices concern calculation, not historical-performance tuning:

- Compute each fold's sufficient statistics once; full statistics are their sum.
  Do not repeat a full-panel `SYRK`. When `n < P`, use the dual `XX'` spectrum;
  otherwise use the primal `X'X` spectrum.
- Store ridge covariance spectrally, including dual prior null-space variance.
  No dense `P × P` posterior inverse is formed.
- Feature columns are grouped by **band, channel, asset**, making constraint
  blocks contiguous. This is a coordinate permutation of the same regression.
- Reuse task-owned design, prefix-sum, and sufficient-statistic buffers. Tasks
  can migrate between Julia threads, so mutable workspaces are not indexed by
  `threadid()`.
- Fractional convolution uses `nextpow(2, 2T)` FFT padding, leased buffers and
  cached plans/kernels. There is no fixed 16,384-observation history limit.
  Native FFT plans are cleared on module initialization after precompilation.
- Gaussian scenarios use a single batched matrix multiplication. Relative
  projection uses `v - mean(v)`, not a dense projection matrix. **No scenario
  mean shifting** is applied; the IID mode samples the unmodified predictive law.
- Backtests compute, consume in date order, and discard each bounded chunk.
  They precompute log prices, raw returns, and first-observation metadata without
  using future values in any decision prefix. Full/fold scalar warm starts use
  identical brackets and tolerances; covariance initialization is schedule
  independent.

### Kelly certificate

For free weights, locked scenario wealth `base`, and budget `b`, maximize

$$F(w)=\frac1S\sum_s\log(\mathrm{base}_s+X_sw),\quad w\ge0,\quad\sum_jw_j=b.$$

The custom solver must satisfy original-objective simplex feasibility and KKT
complementarity, and the concavity bound

$$F(w^*)-F(w)\le b\max_j g_j-g^Tw,\qquad g=\nabla F(w).$$

The default tolerance is `1e-8`. An uncertified custom result invokes Clarabel
for **the same log-Kelly objective**, and its result is certified as well. Gross
returns are never raised to a positivity floor; invalid inputs are rejected.
Only nonzero genuinely held columns enter locked wealth, avoiding `0 * NaN`.

### Optional adaptive quadrature

`path_kelly_v1(...; adaptive=true)` and `backtest_v1(...; adaptive=true)` use
nested randomly shifted Halton points and refinements `64 → 128 → 256 → 512`.
The same point coordinates map to Gaussian, fractional-mixture, residual-row,
and asset-specific residual draws. Halton quadrature is not IID sampling.

Refinement requires both an L1 weight change below `1e-3` and the previous
solution's original-objective gap on the refined quadrature below `1e-5`.
These are numerical convergence checks, not held-out Sharpe selection or a
statistical confidence interval. Reaching the configured maximum without
convergence raises an error. High-dimensional quadrature is not guaranteed to
converge at 512 points. Fixed `S=300` IID sampling remains the default/reference;
no silent reduction in sample count is made.

## Run and verify

```bash
julia --project=. bin/fetch.jl
julia -t 4 --project=. test/runtests.jl
julia -t 6 --project=. bin/backtest.jl [YYYY-MM-DD]
TRADIER_ACCOUNT_ID=.. TRADIER_TOKEN=.. julia --project=. bin/live.jl [--live]
```

Tests cover distribution invariance under price-unit and asset-coordinate
changes; interleaved inactive assets; missing-pair prefix rulers; arbitrary
folds; primal/dual null-space uncertainty; conditional evidence against dense
Gaussian oracles; causal FFT convolution beyond the old history limit; original
Kelly certificates and exact-objective fallback; nested quadrature; and chunk
schedule independence. They do not equate identical seeded Monte Carlo draws
under a basis change with invariance of the predictive distribution.

## Performance measurement

```bash
BENCH_DAYS=500 DATE_TASKS=6 BLAS_THREADS=1 julia -t 6 --project=. bin/bench.jl
BENCH_DAYS=500 DATE_TASKS=3 BLAS_THREADS=2 julia -t 6 --project=. bin/bench.jl
BENCH_DAYS=500 DATE_TASKS=2 BLAS_THREADS=3 julia -t 6 --project=. bin/bench.jl
ADAPTIVE_SCENARIOS=true julia -t 6 --project=. bin/backtest.jl
```

Both scripts report wall time and days/second. Nine per-decision timing buckets
are accumulated: `prep`, `basis`, `gram`, `eigen`, `EB`, `condition`, `fracFFT`,
`scenario`, and `Kelly`; `condition` also includes OOF and decision projections.
They report totals and the last 500 decisions. Summed parallel CPU seconds are
not wall time. Adaptive generation/optimization is timed together in `scenario`.

A ten-year **decision range** is not a ten-year **input history**: all earlier
available observations remain in every prefix. FFT size and Gram work must be
estimated from the actual history, not just the number of backtested decisions.
No 60-second runtime or 5.7× speedup is assumed.

## Changes

- 2026-10-06: correct prefix missingness and active-column mapping; explicit
  activity and ragged field semantics; general fold-only statistics; conditional
  EB evidence on relative support; certified log-Kelly; spectral primal/dual
  covariance; planned dynamic FFT; batched predictive draws without shifting;
  bounded streamed backtesting; reusable workspaces and timing/topology controls;
  opt-in nested randomized quadrature.
