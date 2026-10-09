# Changelog

## 2.0.0 Final — 2026-10-09

CPU release. GPU development belongs to 2.1; no GPU dependency is loaded by
the production package. The original 1.0 mathematical definitions and
public `*_v1` interfaces remain compatible.

### Delivered

- One typed prepare/solve boundary, independent full and OOF fits, exact
  incremental preparation with reference fallback and explicit ownership.
- Fit-local geometry, alpha-cache and work-array reuse; unused gradient
  elimination; compact scalar tensors and lower-allocation residual paths.
- Dense free-scenario packing for certified Kelly, and removal of redundant
  full-prefix caches from the backtest without weakening public cache guards.
- Producer exceptions reach the backtest caller; an existing certified
  terminal candidate covers a previously omitted failed-ray branch.

### Finalization

Package version is 2.0.0. Accepted CPU implementation, mathematical tests
and dependency lock remain byte-identical to the final engineering audit.
The release has a standalone integrity verifier and an explicit CPU source
archive allowlist. Private market data, model snapshots, generated binaries,
GPU prototypes and historical development reports are not shipped in that archive.

### Validation scope

41 standard files / 22 groups, plus independent model and process replay of
the earlier full-history eight-day window. This is grouped coverage, not a
single-process full-suite run. The numerical/data/execution group used O1;
other groups and production measurements used the default optimization level.
The scope is the documented local CPU profile, not convergence on all data,
proof of maximum speed, or safe memory headroom for arbitrary parallelism.

Existing log failures, timeouts and source snapshots remain historical
records. A later passing check does not rewrite an earlier command outcome.
