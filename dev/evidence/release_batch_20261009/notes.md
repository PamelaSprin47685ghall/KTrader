# Release batch, 2026-10-09

Final scope, measured results, preserved failures and reproduction settings
are in `summary.md`. The authoritative qualification snapshot and receipts
are under `qualification/final/`; earlier snapshots/logs remain historical.

Work in the existing checkout, master/602b897; no commit/reset/clean/tag.
Goal: bounded release coverage plus full-history short-window execution,
not another isolated arithmetic-only optimization. No economic changes.

Pre-change SHA256 observations:

```
31cd670411fd0f00a5b4e5105ff18b42e1d9ce7f85e98859965e705a668d41aa src/backtest.jl
f24d1a50f201e57a262ac7c03900b252aa33d87dc70f49b00014c94116706642 src/kelly.jl
96faa12404dcb22094b21eb77b14907f20acb237833048b3e0e755c5903488d2 src/response.jl
5ec389eee53d48ba3c732f98297bc00b89cbc9904319952b04d114aeea037e3e src/incremental.jl
186361d8c19949e42fb6e06539f9b811171b3693d524bc24f989b6590f1721a7 test/runtests.jl
cb3b1c39e10f155631d83cbab88525931c758593f2d7efa3f2ba62b36cc15459 test/contract_registry.jl
```

The deserialized/channel decision reaches the scheduler as Any. A small
numerical function barrier now receives its actual type and current holdings.
It owns no state and preserves both fixed and adaptive timing/seed/certificate
semantics. No new runner, model, fallback, resource threshold or GPU backend.
This is a compilation/specialization opportunity, not a measured speed claim.

All numeric commands use the existing sampled 2048MiB guard and 45s inner /
55s outer deadlines. Long file coverage is split at existing test boundaries;
the production/default test entry still runs everything. Source identity is
frozen before the qualification sweep and checked around each phase.
