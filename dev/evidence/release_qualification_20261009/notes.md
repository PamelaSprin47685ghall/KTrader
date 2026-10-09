# Release qualification, 2026-10-09

Scope: complete the previously timed-out incremental regression using its
unchanged assertions/fixtures in bounded direct-file phases, then measure a
small consecutive-date workload on the existing N65 local CSV input.
No full backtest, reduced model, altered certificates or release tag.

`incremental_tests.jl` now accepts a single direct-entry `--phase=...`:
prefix (append/ruler/EB), ragged (gaps/IPO/masks), ownership
(resource/failure/checkpoints), kernels (filters/rwrap/cross-Gram).
No argument runs all. Included use with any phase argument fails before
numeric tests. All pre-existing test bodies, seeds and tolerances are kept.
The standard `runtests.jl` remains unchanged and still rejects phase flags.
Each phase reports completion explicitly; kernel phase tests the argument
boundary. Per-phase passes do not constitute whole-release suite passage.

Preflight workspace remained master/602b897 with existing user changes.
The combined read-only shell preflight (hashes/process list/output listing)
was tool-blocked without an approval token; it was not retried via another
wrapper. It provides no current process or hash evidence. Subsequent
allowed validation operations are recorded separately; no process killing
outside the owned scoped command or Git write is authorized here.

## Incremental regression coverage

All four direct phases completed under the same 45s inner / 55s outer
deadline and sampled RSS=2048MiB guard, Julia threads=1 / BLAS1.

| Phase | Original assertions | Additional phase checks | Elapsed / peak RSS |
|---|---:|---:|---|
| prefix | 6112 | 1 | 32s / 1152MiB |
| ragged | 780 | 1 | 25s / 1225MiB |
| ownership | 146 | 1 | 26s / 1224MiB |
| kernels | 85 (82+3) | 17 (16+1) | 12s / 916MiB |

Original coverage: 7123/7123. With entry checks: 7143, not four separate
copies of the full file. No old assertion or numeric tolerance was removed.
This closes the prior uncompleted *file coverage*, not a one-process full
release run. The phases above preceded the Kelly packing change below;
the caller phases touching scenario weights are explicitly rerun after it.
The post-packing prefix rerun passed 6112+1 (31s/1143MiB). Repeats are not
added to the unique assertion count.
The post-packing ragged rerun also passed 780+1 (25s/1219MiB).
`standard_entry_rejects_partial.log` records the actual standard entry with
`--phase=kernels`: expected RC1, elapsed1s, killed0, peak315MiB, with its
unknown-test-entry error before any numerical Test Summary. This is an
executed refusal, not merely a source-text expectation. No full-suite
configuration was weakened to obtain the phased coverage.

## Two-day attempt: no throughput result

`short_blas1.log`: actual `backtest_v1`, full local histories 1:14308 and
1:14309, dates 2026-09-30/2026-10-01, N65/F3/S300/default EB, one worker.
The explicitly requested first same-API warm-up timed out before returning.
RC124, elapsed40s, deadline45, killed1, peak1933MiB; termination stack was
in LLVM with the consumer `scenario_weights` caller at backtest.jl:270.
This proves a deadline hit and where it was stopped, NOT that all elapsed
time was compilation. No measured replay, output artifact or BLAS6 attempt
was produced; the planned two-configuration comparison remains incomplete.
The failed log is preserved. No longer deadline, same-workload retry or
compile-time subtraction was used to manufacture throughput.

## Isolated, measured Kelly optimization

The saved decision matrix was authenticated by SHA256
`b5a1374722e4ca0a494aa859c9583eaee97639d9146de2e4bc58f0cdca3c65a5`.
The probe is a controlled all-tradable decision on these same S300/N65
scenarios, not a re-fit and not an actual two-day holding trajectory.
Runtime confirmed `view(X,:,Vector{Int}) isa StridedMatrix == false` and
the dense copy strided, with all cells exactly equal. Packing makes repeated
Newton and line-search matvecs eligible for dense kernels. It changes
representation only: locked wealth is still formed from the original X,
all free columns are retained, and both solvers/certificates are unchanged.

Before source SHA256 (kelly.jl):
`52241a58499a061a35b7c2c72cd52f5e8f97955013a6398faa442c319014a575`.
After source SHA256, observed unchanged during the production probe:
`f24d1a50f201e57a262ac7c03900b252aa33d87dc70f49b00014c94116706642`.

Production-owner ABBA measurements (`kelly_packing_production.log`), five
samples per entry, all timed samples compile_time=0:

| Entry | Median time | Allocated bytes |
|---|---:|---:|
| indexed reference A | 8.991679ms | 373616 |
| production dense B | 3.806481ms | 529736 |
| production dense B2 | 4.032780ms | 529736 |
| indexed reference A2 | 9.025578ms | 373616 |

Extra allocation=156120 bytes (one S×N matrix plus overhead); asymptotic
solve complexity is unchanged, with an O(S·N_free) copy per decision.
Both original-objective certificates pass, weight L1=1.912083588715858e-13,
objective difference=2.5326962749261384e-16. The initial dev-candidate run
had the same outcome; its candidate was deleted after promotion, and the
final probe calls the real `KTrader.scenario_weights`. First-call timings
were printed separately, but their order shares compilation, so they are
NOT a controlled cold-start improvement claim.

New Kelly tests: 210/210, including noncontiguous active sets, NaN inactive
columns, positive locked risk, zero free budget, no tradability, one free
asset, caller ownership and original-law certificate comparison. Adjacent
principal-root tests: 15/15; architecture: 25/25. Combined command RC0,
26s/1103MiB. No economic parameter, seed, old tolerance or fallback changed.
The new test is REQUIRED; registration is not an all-suite pass claim.

Remaining release scope: current-source complete release regression,
multi-day measured throughput and broader execution topology comparison.
No 2.0 tag, commit, reset or clean in this work.

## Review and resource boundary

Final change review used the workspace's dedicated changes tool; its totals
include earlier user changes and are NOT the LOC of this work. Production
changes in this pass are confined to `scenario_weights` in src/kelly.jl.
All started command sessions returned exit codes; the two-day timeout and
expected CLI refusal remain nonzero evidence, not passing numeric tests.
Successful validation commands were at most 32s by scoped_run logs; the
failed short-window command recorded 40s and killed1. These are command
resource times, not API response/transport times. No all-machine process
count or final complete source-hash inventory is claimed: that preflight
was blocked. The production microbenchmark recorded the Kelly source hash
above and checked all runtime hashes unchanged during that measurement.
