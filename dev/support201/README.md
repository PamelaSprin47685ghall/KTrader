# 2.0.1 candidate — not released

This directory implements the structural-support covariance/log-alpha
corrector developed from `dev/theory_incremental_20261009/THEORY.md`.
It is NOT included by KTrader's production module or the2.0.0 release.

`solver.jl` implements mask-graph support compression, the retained null
prior, complete analytic log-alpha first/second/mixed derivatives,
106/107-dimensional Schur directions and line search. A separate
coordinate-schedule experiment retains the reference's global alpha search
and outer certificate threshold, with a sufficient fixed-alpha sublevel
uniqueness enclosure. The enclosure concerns the retained covariance
space, not the full joint-alpha problem or arbitrary off-face null modes.

`checks.jl` tests formula derivatives, original objective agreement,
ownership and rejection paths, and the fixed-alpha enclosure.
`real_probe.jl` compares the candidate with an authenticated real third-fold
posterior at the predeclared tolerances. It saves failing outputs rather
than replacing baselines or printing huge matrices.
`continuation_probe.jl` compares the original optimizer, a cold candidate,
and a previous-day candidate seed on one actual transition. Source hashes
are checked. Spectrum/preparation inputs are already resident in its local
timings; it is NOT an end-to-end backtest benchmark.

Current decision: **original-output gates fail** even where original KKT
certificates pass. In particular, the more-refined joint stopping point is
not interchangeable with the2.0 finite stopping point at the existing
alpha/covariance/mean thresholds. The coordinate-schedule experiment also
stops at a different accepted point. Do not use `accepted=true` from a local
candidate tuple as a release decision: it means only its nonlinear
certificate passed, not backward-output/OOF/performance acceptance.

No production wiring, version bump, tag, release archive or deployment is
authorized by these results. Full OOF-isolation and multi-day acceptance
are not bypassed by the three checks that pass on one cold/warm transition.

Typical bounded checks from repository root:

```sh
bash bin/scoped_run.sh 45 /tmp/support201-checks.log --rss-guard=2048 \
  env JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia --startup-file=no --compiled-modules=existing --project=. \
  dev/support201/checks.jl
```

This command is expected to remain nonzero while the retained original-law
evidence comparison fails. A failing acceptance test is not an installation
error to work around. The frozen2.0.0 CPU remains independently verifiable
with `bin/verify_release.jl`.
