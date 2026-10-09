# Local CSV pipeline, 2026-10-08

This run closes a different evidence gap from the synthetic microbenchmarks:
the full current local CSV prefix is explicitly loaded through production
`load_bars -> signal_prices -> prepare_reference -> solve -> scenarios/Kelly`.
Each phase is a separate <=45s scope (55s outer bound, 2048 MiB sampled RSS
guard). One fit only, no warm-up fit, full backtest or data download.

CSV SHA256 fixed before work:
`data/adj.csv`: e0c25e04a079947b1c5fbc3879703b76d9d35c20300dccd969fedcb268a29322
`data/close.csv`: 4b9b17c8d9dbf81f4274187e82c13f43ab094cbddc9d1b3be804b03a4b55aced
Both files have 14311 lines including headers. All 65 columns are retained;
the final bar is excluded from the decision prefix. No historical artifact
or old manifest is modified. New artifacts bind current source hashes.
These hashes establish a reproducible local input, NOT external vendor
authenticity or the historical origin of old /tmp artifacts.

No claim of steady-state throughput: stage calls can include specialization;
native @timed compile/recompile fields are printed when available. Bucket
logs expose the existing production boundaries without a second solver.

Before this turn: response.jl SHA256
c0d005c5c4d9abde3fb93397af6549f111268e73da0e5b034e85fb538b49ecc2;
registry 5fa7c42ab32ff06df9b4114dc99a7eb744868045d34b1ff2e95bf9c2c67c74ca.
Preflight found no Julia process. Mathematical input and tolerances unchanged.

Admission extension after measured solve cost: the first complete N65 solve
finished in 12.877s (@timed compile_time=10.399s, total scope26s/peak1673MiB).
A separate prepared-stage profile is now justified: exactly one explicitly
reported JIT warm-up plus one measured solve, both numerical-cold (no alpha
warm state). Same 45s/55s/2048MiB limits. This is NOT the original single-call
phase and is not hidden behind it. Profile overhead/native compile fields
are reported; no end-to-end/multi-day throughput inference from subtraction.

First profile attempt (`profile.log`) exited RC1 at script parsing before
any artifact load or fit: bare `@__FILE__ &&` was greedily parsed as a macro
argument. The main guard was corrected to an explicit if with the macro
inside `abspath(...)`; production and existing artifact bytes were unchanged.
The corrected attempt has a different log, never overwrites the failed log.

## Completed baseline, before the new certificate change

All source files were hashed around each successful pipeline stage. These
artifacts bind response SHA256 c0d005c5c4d9abde3fb93397af6549f111268e73da0e5b034e85fb538b49ecc2,
not the later certificate change. Complete prefix T14309/N65/P910/F3;
decision date 2026-10-01; 400030 finite signal cells. `load_bars` and
`signal_prices` were actually called, no inference from NaN counts alone.
Upstream CSV/vendor authenticity remains outside this scope, and this does
not retroactively establish the provenance of older /tmp artifacts.

| Stage | observed seconds | native compile_time | stage scope elapsed / peak RSS |
|---|---:|---:|---|
| prepare_reference | 1.427153064 | 1.218045727 | 16s / 1375MiB |
| solve_once | 12.877293254 | 10.398786088 | 26s / 1673MiB |
| scenarios S300 | 0.543962738 | 0.500351458 | decision scope 15s / 1298MiB |
| Kelly | 2.16279782 | 2.153628172 | same decision scope |

Load and persistence have separate log entries and are NOT included in
the stage times above. Kelly feasibility 4.44e-16, KKT 1.00e-10, objective
gap 6.2867999695e-9 pass the existing 1e-8 certificate. Independent output
records prepared.jls/model.jls/decision.jls and their .toml hash manifests
are retained locally, not overwritten. The large .jls payloads are ignored
by Git; the script, metadata and logs remain reviewable.

Profile: explicit warm-up12.2408s, profiled solve3.16256s, allocation
2359882472 bytes, GC0.785326s; native compile_time0.395036s remains in the
measured call (the warm-up used no timing keyword, measured call used one).
Thus this is NOT a clean steady-state or controlled before/after baseline;
no compile share is inferred by subtraction. OOF_fit2.367706s,
full EB0.301941s, full eigen0.058745s. Seven same-input model-field checks
against the earlier cold solve passed. `profile.txt` includes other runtime
threads (notably the Reseau I/O poller), inclusive stack counts and compiler
frames: never divide its counts into an exclusive "98% wall-time" claim.

The sampled stacks repeatedly visit conditioned_eb_certificate -> fresh
scalar/alpha geometry, gradient and core assembly. Static inspection
confirms geometry is already owned by this fit and the gradient is unused
by the certificate. This motivates the following bounded source change.

## Certificate reuse change and its exact validation boundary

Public `conditioned_eb_certificate` constructs fresh geometry from its own
arguments; only the optimizer calls the private helper with its existing
fit geometry. All Sigma-dependent scalar terms, the alpha cache, freshly
reprojected Sigma, covariance direction, alpha KKT, covariance KKT and finite
evidence checks are still evaluated. The reprojected Sigma is deliberately
not replaced by the optimizer's compact S (different floating-point order).
Only duplicate invariant preparation and the unused full gradient disappear.
No independent cross-fold state, tolerance, iteration budget or fallback
is altered. Both failed-line-search and exhausted-budget certificates keep
their original criteria and complete return fields.

Current-source tests: certificate_tests.log 424/424 numeric +25 architecture,
RC0/12s/peak989MiB; eb_regression.log 145/145, RC0/25s/peak1255MiB including
the frozen N65 fold's BLAS1 and actual BLAS6 companions. Original thresholds
unchanged. The registry now lists27 REQUIRED files, NOT27 suites all run.

The requested `dev/local_panel_certificate.jl micro` command was rejected
by the tool security layer BEFORE execution, without an approval token.
There is no micro log or benchmark result, and no alternate command was
used to work around the block. Its separate verify phase was NOT attempted.
Consequently no speedup or post-change full-N65 output-equivalence claim is
made. The test-only old certificate formula and bounded before/after script
are available for a future authorized execution; no new artifact replaced
the baseline. Existing evidence and current-source regression are distinct.

Release state: local-data pipeline provenance/replay baseline now exists;
current-source full release regression, full N65 after-change replay,
multi-date throughput and incremental-route economics remain open. No2.0
tag, version bump or CPU/GPU exhaustion claim. No commit/reset/clean.
