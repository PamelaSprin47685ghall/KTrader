# KTrader — Path Kelly (Julia)

> Archived development README before the 2.0.0 Final release. Historical
> statuses and relative links below refer to the original repository root.
> The current release entry is the root README.md; old logs are not rewritten.

Price history → center-of-mass macro field and relative field → causal multiscale
path coordinates → trace-neutral response posterior → fractional innovation
mixture → posterior-predictive gross returns → exact sample log-Kelly → execution.

The account problem is long-only, fully invested, leverage one, with **no cash
asset**. Fees and execution friction do not enter the theoretical objective.

## CPU 2.0 engineering acceptance (2026-10-09)

CPU 2.0 has passed local engineering acceptance; GPU belongs to 2.1 and is
not a CPU release blocker. See [CPU 2.0 delivery](RELEASE_CPU_2_0.md) and
[GPU 2.1 scope](ROADMAP_2_1.md). The original 1.0 mathematical definitions,
independent OOF and all numerical certificates are unchanged by this scope
decision. No production code, tests or dependencies changed in this closure.

The same qualified source has 41 standard files in 22 passed groups. Both
saved earlier eight-day outputs now pass the independent chronological-model
comparisons (24 checks each); their mutual process replay passes four more
checks, with daily portfolio L1 and return differences both zero. This closes
the previously unrun CPU output verification. The original first producer's
postprocessing timeout remains recorded, rather than being relabelled green.

The new stdlib-only audit rechecks source/input/artifact hashes, actual pass
receipts and saved portfolios, and rejects stale/missing evidence. Its 21
checker assertions passed. The machine-readable result is
[`acceptance.toml`](dev/evidence/cpu20_acceptance_20261009/acceptance.toml);
`julia --startup-file=no --project=. dev/cpu20_acceptance.jl status` rechecks
it without fitting or running GPU work.

This is an engineering milestone in the existing checkout, not a registry
publication or Git tag. Project.toml still records package version 0.1.0;
no commit, tag or deployment was performed. Supported evidence is the local
single-task/BLAS6/batch profile, not arbitrary-history convergence, proven
CPU optimality or a safe two-task memory guarantee. The 42 days/s mission
has not been reached. Standard-file coverage is grouped, not one process;
the numerical/data/execution test group used -O1, while other groups and
production timing used the default optimization level.

## Earlier delivery history

The following checkpoints retain their original dates and then-current
failures/pending items. They are superseded by the acceptance above, not
rewritten as though the earlier commands succeeded.

Previous release checkpoint:
[`dev/evidence/earlier_closure_20261009/summary.md`](dev/evidence/earlier_closure_20261009/summary.md).
The repaired runtime now has current-source grouped coverage of all41
standard files in22 groups, including the22 exception-delivery assertions.
No production/test/dependency source was changed in this iteration.
All eight earlier dates completed individually with the exact chronological
full/fold alpha chain. Two COMPLETE eight-day numerical calls then measured
17.517s and17.330s with native compile/recompile0. The first command timed
out during subsequent baseline authentication, not the completed numerical
window; the second measurement/accounting command exited0. Independent
artifact comparisons remain unrun after a tool-blocked verification command.
The RX6800 now has an actual FP64 core-assembly prototype, not just device
inventory: all430080 entries pass the original elementwise core tolerance.
Observed CPU BLAS6 core construction5.474ms vs GPU1.475ms including both
input uploads and output download; resident compute/sync0.815ms. These are
primitive timings, excluding initial setup, not whole-EB or backtest speedup.
No GPU production backend, model tolerance change, dependency installation,
version bump or2.0 release. Prior failure and freeze history remains in
[`dev/evidence/release_freeze_20261009/summary.md`](dev/evidence/release_freeze_20261009/summary.md).

The following paragraphs preserve earlier work and its then-current status.

Latest work is recorded in
[`dev/evidence/batch_memory_20261009/summary.md`](dev/evidence/batch_memory_20261009/summary.md).
The backtest no longer builds a full PrefixRulerStats table: its consuming
guard already rescans the entire prefix, so the existing no-cache reference
avoids a 100456200-byte resident payload on the N65 panel. Public cache
arguments and their validation remain unchanged; incremental backtests also
avoid an optional initialization-only PriceHistoryCache.
Current-source 39 existing standard files passed in 20 groups. The NEW
backtest_cache_tests file is registered but its numeric command was
tool-blocked, so the aggregate correctly reports 20/21 groups and remains
INCOMPLETE. Syntax checks are not numeric tests. No full-current-suite claim.
Complete production windows at default GC and zero native compile time:
four days 8.072s/1611MiB (one task, BLAS6), 5.023s/2042MiB (two tasks,
BLAS1); eight days 15.767s/1684MiB (one task, BLAS6). Four-day old/new
single-task portfolios agree exactly; cross-topology comparisons pass the
existing tolerance. Two-task memory margin is only 6MiB, not a stability
guarantee. No eight-day parallel attempt, larger topology or 2.0 release.

The preceding OOF-storage qualification and measurements are preserved in
[`dev/evidence/oof_memory_20261009/summary.md`](dev/evidence/oof_memory_20261009/summary.md).
That preceding frozen source had complete standard-file coverage: 39 files in
20 bounded groups, including every incremental and process-guard phase.
This is not a single-process full-suite run. The numeric/data/execution group
uses the standard import preamble and `-O1` for regression; other groups and
all performance measurements use the default optimization level.
The optimizer now reuses weighted/product/core STORAGE within one fit. Every
alpha rewrites its data; old leases are rejected, public caches remain owned,
and returned model/state arrays do not alias the scratch buffers. Elementwise
direction operations are fused without changing the matrix products.
The complete four-day N65/F3/S300 production comparison (one task/BLAS6)
has zero native compile time and exact portfolio/return agreement:
8.113GB -> 6.210GB cumulative allocation; 8.233s -> 7.887s wall time.
These are paired single-window observations, not a steady-state guarantee.
Two tasks required process-local GC tuning to complete under 2048MiB; the
successful four-day sample was 6.812s/1987MiB, but another same-setting
observation was 10.286s. Default GC hit the RSS guard, so no stable parallel
speedup or larger topology is claimed. Production defaults stay unchanged.
Previous two-day evidence remains in
[`dev/evidence/release_batch_20261009/summary.md`](dev/evidence/release_batch_20261009/summary.md).
No 2.0 tag or claim that CPU/GPU opportunities are exhausted.

The following paragraphs retain the earlier delivery history; the latest
summary above supersedes their then-unrun/timed-out status, not their logs.

Current CPU changes and bounded validation are recorded in
[`dev/evidence/release_work_20261008/notes.md`](dev/evidence/release_work_20261008/notes.md).
The optimizer now defers the full covariance gradient until line search;
public gradient evaluation and all final certificates retain their formulas.
Fixed-Sigma scalar tensors now keep only the consumed upper pairs and reuse
one construction-local transform buffer. Current-source conditioned-EB
regressions (including the frozen N65 fold under BLAS1 and BLAS6), exact
scalar comparisons and the small full/OOF-to-Kelly replay are recorded in
[`dev/evidence/scalar_tensor_20261008/notes.md`](dev/evidence/scalar_tensor_20261008/notes.md).
The measured benefit is allocation reduction; six-thread timing is mixed,
not evidence of a stable whole-fit speedup.
The full local CSV prefix (T14309/N65/F3, decision date 2026-10-01) now has
a source/input-hashed prepare -> solve -> S300/Kelly baseline and an explicit
prepared-stage profile, recorded in
[`dev/evidence/local_panel_20261008/notes.md`](dev/evidence/local_panel_20261008/notes.md).
Those pipeline artifacts bind the PRE-certificate-reuse source c0d005c5;
they are not retroactively current-source end-to-end validation. The new
certificate path reuses fit-owned geometry and omits an unused full gradient,
with all returned certificate fields and gates preserved. Its 424 contract
assertions and 145 conditioned-EB regressions passed on the newer source;
the earlier tool-blocked microbenchmark and unrun replay were delivery-time
status. The pre-Kelly-packing follow-through records the N65 full/OOF/S300/Kelly replay
with zero reported model/scenario/weight differences and original certificates.
One explicitly warmed prepared solve measured 1.897s with native compile_time=0;
it is not multi-day throughput. Certificate allocation halved, but its measured
median latency did not improve; no stable certificate speedup is claimed.
The exact incremental pending filter now uses column views instead of repeated
slice copies. On the authenticated T14309/N65 prefix, initialization allocation
fell from 10.89GB to 7.63GB (cumulative allocation, NOT resident memory). Default
4096-row routing still falls back for 7803 materialized rows; the 8192 diagnostic
route remained slower in these single-point preparation observations, so no
resource default or production engine was changed. Both routes passed their
original field comparisons. Evidence and the preserved full incremental-file
timeout are in
[`dev/evidence/release_followthrough_20261008/notes.md`](dev/evidence/release_followthrough_20261008/notes.md).
The incremental file's original 7,123 assertions have now all passed in four
bounded direct-entry phases; no-argument/include execution still runs all,
and the standard full-suite entry does not accept partial phase flags.
The Kelly boundary now packs the identical free-asset scenario columns once,
instead of sending a non-strided indexed view through repeated matvecs.
The saved S300/N65 all-tradable fixture measured about 9ms -> 4ms at BLAS1,
at the cost of 156KB additional allocation; original certificates passed.
This is a local kernel result, not backtest speed. A two-day backtest warm-up
hit its deadline before measurement, so no new steady-state throughput or
BLAS1/BLAS6 winner is claimed. Current scope and failures are recorded in
[`dev/evidence/release_qualification_20261009/notes.md`](dev/evidence/release_qualification_20261009/notes.md).
The earlier full-suite green is not a 2.0 performance sign-off for this newer
source. Representative steady-state throughput and whole-release regression
remain separate acceptance requirements; no version or release tag is changed.

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
  the Gaussian parameter support. The conditional-support evidence is:

  $$\log p(Y\mid Cg=0)=\log p(Y)+\log p(Cg=0\mid Y)-\log p(Cg=0).$$

  The current relative fit invokes `optimize_conditioned_eb` on this support,
  followed by exact Gaussian conditioning. The macro fit separately uses
  `optimize_matrix_normal_eb`, a profile-evidence optimization whose only
  return path passes a joint bounded-alpha/covariance certificate (a failed
  certificate raises); the returned `converged`/`rel_res` fields are
  informational outputs of that gated solve, not the gate itself. Do not
  conflate these gates with a guarantee of a global evidence optimum. The
  response owner controls EB acceptance/fallback policy; the scheduler does
  not replace it. This describes the source as statically inspected, not a
  runtime certificate.
- Production OOF residuals are a lazy `ResidualOracle`: inputs frozen at
  construction (copied returns, rulers, macro basis, prefix-sum field), rows
  evaluated on demand with no shared mutable cache, so same-seed replays are
  bit-identical. `dense_oof_residuals` is a test-only independent reference;
  production never calls it. Lazy-versus-dense agreement is asserted at
  roundoff-scale tolerances (different summation orders), not claimed
  bit-exact across views.
- Anti-overfit and numerics are separate layers. Posterior uncertainty over
  the path-response law and trace neutrality are theoretical constraints that
  would remain under unlimited compute. The covariance floor, KKT certificate
  gates, and solver tolerances are numerical-implementation devices and are
  documented as such; neither layer may be presented as the other. Not every
  pipeline stage carries a certificate: both the conditioned and the
  matrix-normal (macro) EB solves return only through their
  alpha/covariance certificates (failure raises); the fractional FFT and the
  scalar macro series are verified by tolerance and fixed-snapshot
  comparisons instead.
- OOF folds have independent macro and relative EB fits: each fold's EB alphas
  are solved from that fold's train Grams only (full statistics minus the
  fold), and `ridge_alpha` stays exactly what the caller fixed globally —
  never the full fit's data-dependent alpha. OOF isolation is at the
  supervision layer only: prefix preprocessing (rulers, band scales, the
  embedded field) is one fixed full-prefix computation and is not re-estimated
  per fold. `F_folds` supports any value from two through the number of
  training rows. Residuals use the same **next-return index** as the
  regression target. Missing realized returns never become bootstrap
  observations.
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
- Backtests partition the whole decision range **once** into at most
  `date_tasks` contiguous timeblocks (one continuous schedule; no per-chunk
  windows). The default (non-adaptive) path spills each date's result to a
  single-writer per-date spool file and posts only a token, so block tasks
  never block on consumption progress and every block solves in parallel
  across the run; resident memory is the per-block transient results of the
  active solvers (O(date_tasks)) plus O(K) token metadata and the one result
  being consumed — no claim of "at most one live object process-wide".
  `chunk_size` is retained for compatibility and no longer bounds this path.
  The adaptive path keeps bounded in-memory channels (`chunk_size` caps
  outstanding results), and its throughput is limited by the strictly
  date-ordered consumer: later blocks run at most their buffer ahead, so the
  steady state is close to a single active block. Batch workspaces and warm
  starts are task-private. Precomputed history/ruler caches are shared only as
  read-only prefix sources; no future price may enter a current decision.
  Block-local warm starts are numerical hints, not permission to alter the
  fitted mathematical model; the EB warm start is likewise a numerical
  starting point only — the alpha candidate set is warm-independent and
  flat-evidence ties anchor to the log-grid geometric centre — and the
  returned solution must agree with a cold start within the KKT stopping
  tolerances.

### Backtest engines and checkpoint ownership

`backtest_v1(...; engine=:batch)` is the default prefix-fit reference.
`engine=:incremental` uses the production API in `src/incremental.jl`:
`initialize_inference(prefix; ridge_alpha, F_folds, history_cache, ruler_stats)`,
`advance_exact!(state, row)`, `solve_current!(state; timing)` and
`inference_checkpoint(state)`. Missing APIs are an explicit error, never an
unreported switch to batch. Module wiring remains with the engine owner.

M1 PreparedProblem boundary (source-written; run status is PARTIAL and
range-precise — NOT a claim that M1 or M0 is achieved, and no ALL-level
command is green): the Prepared contract suite (56 items, B4b comparing
values with `isequal`, not address binding), timed 7, admission 17 and
model-API small are all green; the architecture standard's real-tree
checks and fixture red/green are exercised; the 604-item timeblock suite
passes (root cause: a real `Base.close` call in the root path plus the
L91 inc expectation corrected to first/last−1 with batch counting 0 —
the original 7 reds were all inc-expectation failures, NOT close-order
failures); the standalone reference runs R1+R2 (52 items, single phase)
and R3 (33 items, single phase; N2 default EB, S24, oneDay, Channel
accounting) both exit RC0 — R3 ran under `--compile=min`, a
semantic-verification configuration only, NOT a throughput figure (the
default-mode timeout history stands; no "precise compile share" is
derived from it — compilation mode also changes runtime constants and no
native compile_time evidence exists). The required registry currently
holds 21 REQUIRED files plus the one architecture gate file. The
historical 18 REQUIRED count (2026-10-07 — confirmed by an actual
stdlib read AND a Gate re-read, including the lifecycle/CLI files and
the four promoted legacy items; the earlier "12" in prior reports was a
stale memory, now withdrawn) is the verified scope of that date and
stands as history, not rewritten to 21; the three admission-integration
deliveries (incremental budget / history cache / artifact replay
contract tests) brought the registry to 21. The standard full run
rejects partial-stage env to prevent silent skips. The true-N65 typed artifact comparison is MEASURED, RC0, no
repeated fit (timeline note, 2026-10-07 documentation audit: when this
sentence was first written the residual finite-check was pending; it has
SINCE BEEN MEASURED — see the residual paragraph below; the wording is
kept as history, not as a current pending item): the old incremental
Prep NamedTuple was mapped explicitly in dev to the typed
PreparedProblem (full T14309/N65/P910/F3, raw/stat recomputed without
truncation, owner None); a single cold default-EB BLAS6 solve took
7.449 s (saved to /tmp/model_m1_t14309.jls); against the old I0_inc
report the alpha/Gmean/Sigma/mu/covLprod/d/vforecasts fields are all
0-diff, and against batch_final alpha_rel relative 8.53e-14, Gc 2.47e-13,
Sigma 2.82e-13, mu 1.13e-14, cov 6.13e-15 — original tolerances pass;
the standalone same-seed canonical S300 gives X 9.77e-14, w_L1 2.31e-12,
both KKT ≈1e-10, feasibility 2.2e-16, objective gap 6.29e-9, min wealth
≈0.804, swap loss floating-point zero. Qualification: this is
single-instance REFACTOR FIDELITY, not a performance claim, not a
steady-state speedup of any kind, and not whole
M0/M1 — 7.449 s has no controlled before/after and must not be read as
"M1 made it faster"; the 0-diff covers exactly the listed fields, same
input, same machine — not a cross-cluster byte-level claim. The residual
finite-check is MEASURED RC0 (loading artifacts only, no repeated
solve/prepare): F3 first/middle/last rows per fold, deterministically
[1,2342,4684,4685,7026,9368,9369,11710,14052], a 9×65 = 585-cell block
with 260 finite / 325 NaN; shapes identical, isfinite and NaN positions
exactly equal on those 9 rows, finite maxdiff 0.0 vs I0_inc and 3.0e-14
vs batch (original tol 1e-9 passes). Qualification: SAMPLE-ROW fidelity
only — not a full T×N per-cell comparison and not a machine-cluster byte
guarantee; the sampled mask must not be stated as a full-history cell
pass. The minimal replay note lives in dev/m1_typed_replay.md, and the
explicit-phase replay script is dev/m1_artifact_replay.jl. HONESTY
CORRECTION (2026-10-07 documentation audit): an earlier version of this
paragraph claimed the note carried source/input hashes with algorithm
names and the preserved eight legacy artifacts — the note as actually
written contained NO hash values, NO complete commands and NO seeds, and
listed only four artifact paths; the "eight preserved artifacts"
inventory was never listed there and is unconfirmed. Those values are
NOT reconstructed here (they cannot be invented retroactively). DevOps has since delivered dev/m1_replay_manifest.toml — the ONE
manifest (13 artifacts with full 64-hex sha256 + byte sizes,
checker-script identities, historical log paths, current-tree source
hashes under an explicit sampling-time caveat, and the guard final
run). The replay script consumes exactly this manifest via the Julia
SHA stdlib (SHA.sha256 + bytes2hex) and creates no second inventory;
old_panel_hash is recorded as "unknown", so artifact-only local
comparisons stand on verified artifact hashes while panel provenance
stays UNcertified. Replay comparison gates (2026-10-08 REVISE; the gated behavior is
now run-verified narrow green — see the gated run status below): the
replay script's model-only / residual-sample / solve-once
phases previously printed differences and completed RC0 even on
over-tolerance / mask-mismatch / NaN-poisoned results (solve-once also
materialized the full T×N residual matrix). They now run under ONE
fail-loud comparison owner (error, never print-and-pass); solve-once
compares the deterministic 9-row block via residual_rows instead of the
full matrix, and refuses its recorded output path before any expensive
action. Tolerances of record (historical): mu 1e-10, Sigma 1e-9,
macro/relative alpha 1e-8 (absolute), w_L1 1e-7, residual finite-cell
max 1e-9, kelly certificate at the production default tol 1e-8. Items
with NO historical threshold are gated by explicit Manager
numerical-verification decisions of 2026-10-08 (recorded as new
decisions, not historical assertions): Sigma_rel / G_c_mean 1e-9
elementwise max, d_posterior / v_forecasts 1e-10, same-seed scenario X
1e-10 max, cross-model objective diff 1e-8 (numerical-reproduction
criterion, not a general two-law objective-equality theory), swap-loss
roundoff 1e-12 with the production certificates passing first, relative
predictive covariance L*L' Frobenius 1e-9, residual Inf rejected
outright — see dev/m1_typed_replay.md ("Comparison gates"). Contract
tests live at test/artifact_replay_contract_tests.jl (small-array gate
logic only; now registered among the 21 REQUIRED files and included
once by the standard runtests loop — the earlier "not wired" wording
was the pre-registration delivery-time state and stands as timeline).
Measured 54/54 RC0 in 1.9 s (2026-10-08, scoped 45 s / RSS 2048; first
run 52 Pass / 2 Fail — both hand-calculated fold-row index
expectations, corrected with no gate or tolerance touched).
solve-once needs no new fit: the recorded output already exists, and
the phase's verification is that the command refuses the overwrite
BEFORE any expensive action (no new output path, no new manifest
record).

Third-revision deliveries (2026-10-08, source-static readiness only;
DevOps runs pending — never pre-filled green): (1) IPO gram-budget
boundary — `embed_active!` budgets the HELD FoldGram payload (V+Xc+R2
matrix bytes) only, checks target bytes BEFORE any enlarged allocation,
and on over-budget releases the Grams and demotes their rows to the
SAME mathematics via exact per-row materialization (route-only: raw
history, coordinates and fold membership untouched; rows re-aggregate
as budget allows; the ledger is not an OS RSS cap and no measured
milliseconds are claimed), with the ledger kept equal to the held
payload and carried by checkpoint deepcopy
(test/incremental_budget_contract_tests.jl verifies via independent
byte accounting). (2) O(TN) cache-prefix guard —
`verify_history_cache_prefix` rejects foreign panels, payload and
metadata mutations on exactly the consumed facts (logs compared, not
just returns — a pure price-level rescale is rejected; NaN positions
exact, Inf rejected), reads only rows 1:T so a lawful future suffix
stays unconstrained, and is wired before the first cache read in
`_prepare_v1` and before any state construction in
`initialize_inference` (test/history_cache_contract_tests.jl). (3)
replay hard gates — the one comparison owner errors on failure; the
9-row residual block is finite-only (no T×N materialization);
solve-once refuses its recorded output path before any
deserialization/fit; tolerance provenance keeps historical thresholds
separate from the 2026-10-08 Manager decisions
(test/artifact_replay_contract_tests.jl, now in the 21 REQUIRED loop).
Shared constraints and chosen-path rationale (linked to the SPEC, not
copied; no new ADR, no per-helper decision log): all three fixes hold
the same three lines — original mathematics unchanged, mutable
input/resource paths must be able to go red, and <=60 s narrow
verification (SPEC §56 fail-loudly, §51 per-field incremental
acceptance; the resource-route semantics live in §50: budgets route,
never change outputs). The budget fix chose checking the HELD payload
and releasing/demoting on over-budget, rejecting the cheaper fakes —
"allocate first, fix the printed ledger after" (the failed-guard path
stays) and "counting a fixed mask-times-fold theoretical maximum as
actual usage" (decoupled from real held bytes, it mis-routes). The
cache fix chose exact consumed-prefix element-wise O(TN) verification
with no whole-panel allocation, rejecting shape/identity/sampled
comparisons and caller-discipline read-only conventions (each leaves a
silent-leak surface); the O(TN) cost has no throughput measurement yet.
The replay fix chose the shared hard comparator, the real production
certificates and the 9-row sample, rejecting println-only reporting
and full-history NaN-poisoned materialization. Revisit only on new
real evidence (e.g. a proven cheaper mutation-safe cache contract, or
an approved profile showing validation as a hotspot) — re-adjudicated
openly, never a quiet tolerance change.
The original mathematics, historical numbers, the fullsuite/multi-day/
GPU/throughput suspensions and the panel-provenance-unknown boundary
all stand unchanged. Replay run status
(2026-10-07, DevOps evidence, NOT a closure; these RC0 records are the
OLD println-only behavior's runs — the gated behavior has since been
run, see the gated run status below): check / rewrap / model-only /
residual-sample are RC0; the non-check toplevel load is now the standard `using KTrader` (the
Serialization PkgId failure was the include's anonymous module lacking
package-identity registration, not a UUID drift). Why `using` and not a
bare include (short rationale): deserialize resolves serialized module
identities through the package PkgId map — only the package registration
carries the UUID the old artifacts were serialized under; a
Main-anonymous include module has a different identity and fails with
KeyError PkgId. solve-once ran twice
on the current tree: 7.202 s with the object lost to stdout only
(mistake evidence retained), then 8.336 s including ~2 s persisting the
new model to /tmp/model_solveonce_t14309.jl2s (refuses to overwrite
baselines), with the dual-baseline field comparison and the 9-row
residual block done; the NEW artifact's sha256
ae54c564f3005cff022703a15637f21be4c0a44f6261c92791c3948db0a2016c is
registered in the one manifest and its same-seed numbers are measured:
new vs inc/m1 X / w / objective / swap all exactly 0; new vs batch
X 9.77e-14, w_L1 2.31e-12, both KKT ~1e-10, gap 6.29e-9, swap ±2.5e-17;
the 9-row 585-cell residual block: finite 260 / NaN 325, masks
identical, finite maxdiff 0.0 vs inc / 3.0e-14 vs batch (sample scope). Final DevOps facts:
Julia exit 0, HEAD 602b897 untouched, 38 uncommitted worktree items with
user edits uncleared; all narrow commands under the 60 s/RSS guards;
full-suite/multi-day/GPU remain suspended. (2026-10-08 closure update:
the worktree now carries BOTH the pre-existing user edits AND this
revision's engineer changes, all preserved — no commit, no reset, no
clean; the named final snapshot records the 16 key runtime objects.)

Gated replay runs (2026-10-08, DevOps evidence; logs preserved under
dev/evidence/manager3/): the REVISE hard-gate behavior is now
run-verified narrow green on this machine — distinct from the OLD
println-only RC0 records above. check RC0 (the four core artifacts of
record verified against the ONE manifest; never stated as a full-13
inventory verification). artifact_replay_contract_tests.jl 54/54 RC0
in 1.9 s (scoped 45 s / RSS 2048, peak 355 MiB; timeline: first run
52 Pass / 2 Fail, both hand-calculated fold-row index expectations in
the determinism testset, corrected to the function's actual
first/mid/last semantics — no gate or tolerance touched). model-only
RC0 (20 s): the load-only path now ALSO runs the strong model-field
gates against BOTH preserved baselines (Manager decision 2026-10-08;
no new fit/prepare/model, tol/seed unchanged; the paths bind exactly
the manifest's model artifacts — sha256-verified before any
comparison, no baseline switch): 15 GATED fields, vs inc all exactly
0; vs batch alpha_rel abs 1.27e-9, G_c 2.47e-13, Sigma 3.97e-14, mu
1.13e-14, L*L' Frobenius 6.13e-15; scenario/Kelly gates pass at the
original tolerances. residual-sample RC0 (16 s): 9×65 block, 260
finite / 325 NaN, 0.0 vs inc and 3.0035e-14 vs batch, masks exactly
equal, Inf rejected — still SAMPLE-ROW scope. solve-once RC1 EXPECTED
RED (17 s): the no-overwrite refusal fires before any
deserialization/rewrap/solve; no new fit, no new output path, no
manifest change; the recorded output's disk sha256 is unchanged. A
real bad-field red (producer→parser→consumer):
dev/evidence/manager3/model_only_gate_probe.jl deserializes the real
model artifacts, passes the positive compare, mutates ONE finite
mu_pred cell in memory (+1.0) — the same consumer throws — and
restores the value in memory; the disk artifact is untouched. The
directory also holds arch_f.log (ArchitectureGate
--architecture-only real tree, 25/25, RC0, 7 s). Qualifications
unchanged: single machine, artifact-only (panel provenance unknown),
no cross-cluster byte guarantee, no whole M0/M1 release, no
throughput.

Budget/cache contract runs of the same revision are now also MEASURED
(2026-10-08, logs in the same directory): budget 1041/1041 RC0 in
9.4 s (scoped 16 s / RSS 2048, peak 946 MiB); history-cache 4 testsets
(24+9+15+45) all green RC0, scoped 20 s (peak 998 MiB; the
initialize_inference testset itself 45/45 in 5.5 s, re-verified after
the ExactInferenceState constructor gained the two budget fields). Red
history retained: the FIRST budget run was 1025 Pass / 10 Fail (total
1035, bud1_red_history.log) — the 10 reds were NOT one defect: they
mix the real source defect (advance_exact! dropping the
initialize-validated gram_budget / materialize_row_limit at first
RawInferenceCore creation — dead kwargs, the core silently
re-defaulting to 512 MiB / 4096) with two independent TEST errors
(activation-day timing: first finite PRICE bar 360 vs first valid
adjacent RETURN bar 361, one day apart — the embed and any budget
crossing can only be validated on 361; and a false Cartesian
expectation of 6 Grams where the actual fold-driven aggregation
produces 4). The fix saves both values in ExactInferenceState BEFORE
the lazily-created core exists, passes them at core birth and through
inference_checkpoint — the two State fields hold validated resource
options for lazy core birth, NOT new theory knobs (defaults and
route-only semantics unchanged, the ledger is not an OS RSS hard cap,
no tolerance or guard relaxed). Adjacent narrow greens re-verified in
the same batch: incremental_primal_prep 46/46 RC0 10.2 s,
prepared_problem contract 61/61 RC0 22.6 s (contains small synthetic
solves — "no fit" here means only: no N65 real-artifact re-fit, no
large benchmark), prepared_lifecycle 24/24 RC0 11.8 s, and the
real-tree --architecture-only run with 5 summaries all pass, RC0 (the
21+1 admission scope).

Admission-negative and snapshot closure (2026-10-08, same directory):
required_negative.log exercises the REAL production primitive —
harness_required_negative.jl includes the production
test/contract_registry.jl and calls verify_required_contract_files:
temporarily removing one REQUIRED file deterministically throws ([NEG]
threw=true, message carries "fail-closed"), all-present passes ([POS]
true), ZERO numeric testsets in that run, and a non-throw fails the
harness. fixture_root_4way.log comes from
harness_fixture_root_4way.jl, which CONSTRUCTS the four fixture roots
and self-checks their CONTENT (filesize / include-count / occursin of
dev/probes / isfile asserts) — 5/5, RC0 — it does NOT itself invoke
`test/runtests.jl --architecture-only --architecture-root`. The
standard entry's fixture-root red/green behavior (legal minimal/nested
RC0, bad/empty RC1) was MEASURED in the earlier DevOps runs recorded
in the workspace-generation narrow-green paragraph above; this batch's
harness is a content self-check plus the real admission-negative
primitive, NOT a re-run of the standard entry. The as-run harness
sources and their sha256 are bound in harness_snapshot.txt. Timeline note: a harness-side sed typo
(occursin==9) was fixed during the runs — a HARNESS transcription
error, never a production Gate red; reported, not conflated. Scope
statement: "21 REQUIRED registered and the file-presence admission
verified" does NOT mean "21 numeric suites all ran" — the numeric
suites actually run in this revision are exactly the six files above
(budget/cache/primal/prepared/lifecycle/replay), each in its own
scoped run. The named final snapshot is
dev/evidence/manager3/final_snapshot.txt: full SHA256 for the 16 key
runtime objects (src/test/dev-script/manifest), taken 2026-10-08
01:54 +08:00 under "no commit/reset/clean"; README/AGENTS/dev notes
are documentation and are NOT part of the runtime snapshot; no
existing snapshot or manifest entry was changed.

The real standard-entry 4-way for THIS revision has since been run
(2026-10-08 02:03 +08:00; standard_entry_4way_summary.txt plus
std_entry_minimal/nested/bad/empty.log, each log's sha256 bound in the
summary; actual command `julia --startup-file=no --project=.
test/runtests.jl --architecture-only --architecture-root <ROOT>`):
minimal/nested RC0 (fixture-mode info declares KTrader NOT loaded, all
summaries Pass), bad RC1 natural exit killed=0 (violations: unsupported
include form, dev/probes.jl, dev/more_probes.jl — the
`violations == String[]` assertion fails at architecture_contract_
tests.jl:363), empty RC1 natural exit killed=0 (fail-closed L356/357
src-tree/module-present red plus "include target not present:
KTrader.jl") — every rejection cause is checkable from the ACTUAL
consumer output, none is a path error. The earlier
harness_fixture_root_4way.jl content self-check remains as auxiliary
evidence; the previous wording that elevated it to a standard-entry run
has been corrected. Timeout timeline kept verbatim: the first
12 s-deadline attempt returned RC124 (a timeout/stop with an LLVM
stack) — its cause has NO native compile-time measurement and the
later clean 3 s RC1 does NOT retroactively prove it; only the
25 s-deadline rerun's natural 3 s RC1 exit is direct evidence. The
original timeout record is not erased.

Guard-configuration transcription correction (2026-10-08, fifth
documentation pass; named record:
dev/evidence/manager5/evidence_corrections.md — summary and logs
preserved unchanged): the summary header's "(scoped_run 25s, RSS2048)"
does NOT match the four logs' scoped_run lines — minimal/nested
actually ran deadline=12 with elapsed=1s (the summary's uniform "3s"
for them disagrees), bad/empty ran deadline=25 with elapsed=3s, and
ALL FOUR logs carry rss_guard=0, rss_peak=0MiB (guard not in effect /
not measured). RSS-2048 compliance and the guard configuration
therefore CANNOT be derived from this summary or these logs; the
red/green verdicts themselves (RC0/RC1, killed=0 natural exits,
violation contents) stand — resting on the four sha256-bound logs,
not on the header line.

Manager-6 guard-scope note (2026-10-08 documentation passes; named
record: dev/evidence/manager6/guard_scope_notes.md): the
SixResourceGuardAudit source fix is DELIVERED and statically
cross-checked by this pass against the full 596-line
bin/scoped_run.sh — catchable own TERM/HUP/INT traps (cancel handler
installed before spawn; pending-cancel registered across the spawn
window; INT effective only when its inherited disposition is
default), an EXIT trap, the /proc/uptime "%.0f" monotonic clock with
explicit bad-clock fail-closed 125 (no silent arithmetic
degradation), bounded leader-exit confirmation before wait, and ONE
shared absolute cleanup deadline for the terminate/confirm/reap
phases (never stacked independent budgets). First-draft defects
(SIG_IGN inheritance, unbounded wait / implicit bad-clock
arithmetic, 32-bit uptime %d saturation, stacked cleanup budgets)
are statically verified fixed — a timeline, not per-item run claims.
Manager-7 calibration (2026-10-08 documentation pass; named record:
dev/evidence/manager7/evidence_corrections.md): a further STATIC
read of the same source found three boundary defects the wording
above over-covers — late-stage bad clocks in begin_cleanup_window /
mono_elapsed_ms only degrade to a 0 placeholder (an rc=0 run can
pass through), the start-path bad clock terminates and reports
Cleaned=1 without any leader/owned-group reap confirmation, and an
empty command after an optional --rss-guard is admitted (a
false-green admission path); the existing clock-fail phase covers
ONLY the monitor loop. The "no silent arithmetic degradation" /
four-site fail-closed wording above is therefore WITHDRAWN as
over-broad; the source owner's bounded fix is separately entrusted
and in progress, and this pass pre-fills no green. The final3
nine-phase 53-item RC0 and final std arch RC0 records stand within
their ALREADY-TESTED scope (monitor bad-clock, clock-injection
negatives, phase runs — never covering the three boundaries above);
the three defects are static findings, NOT dynamically reproduced
here, with dynamic verification left to a later DevOps phase.
Uncatchable limits stay honest: SIGKILL/SIGSTOP/host crash cannot
trap (orphaned group; only the caller-discipline outer hard timeout,
both deadlines ≤60 s, remains); no "atomic self-heal of any external
termination" claim. Run facts so far (this scope only, logs in
dev/evidence/manager6/): basic true entry 0/7 exact pass-through,
>60 refused RC2 before start, G1 real TERM marker with child_rc=0
while the wrapper reports 124 (timeout is never success); the early
TERM test-harness's own faults (127 subshell artifact, script
self-reported 143, and an OBSERVED 30 s outer timeout coexisting
with the wrapper's own 2 s self-report — the harness source's
cleanup defect is statically identifiable, but the causal chain to
the timeout is INFERRED, with no FD/native-wait trace measured at
the time; kept as unknown-cause, consistent with the LLVM-timeout
wording, failure not erased) are preserved in
harness_fault_timeline.txt — clean parent RC and post-cancel
bystander aliveness still await a durable C1 phase, so "all minimal
sequences complete / zero defects" is NOT claimed; the real
arch_admission.log is RC0 7 s / deadline 45 / RSS 2048 / peak
841 MiB / killed 0 / cancelled none with the four runtime/Gate
object hashes fixed before and after — admission scope only, not
generalized. FINAL RUN ESTABLISHED (2026-10-08 05:09 final3; this pass read
final3_closure_summary.txt, negative_control_corrected_record.txt,
negative_control_hashes.txt, final_runtime_snapshot.txt,
arch_after_final2.log and negative_control_clock_mono.log): final
objects bin c7048b3be064… and test 4b13b81fe3b5… (full SHAs on
record; the 05:05 snapshot's 3a754edf is the test's predecessor);
final std arch RC0 7 s / RSS 2048 / peak 841 MiB / killed 0 /
cancelled none; NINE phases, 53 items, ALL RC0 (baseline 6 +
owner TERM/INT/HUP 6 each + clock mutant/mono/fail/saturate
5/5/7/6 + selftest 6; unknown/duplicate are expected RC1 semantic
refusals by the phase selector, not test reds); same-consumer
true-injection negatives — date→clock-mono natural RC1 / killed=0
with THREE reds on the re-read log (negative_control_date_real:
2 Pass / 3 Fail — the two clock targets exitcode 0 != 124 and no
TIMEOUT-EXCEEDED, PLUS the third no_live_in_group assertion; the
corrected record's "two target assertions" named only the clock
targets, total reds are 3), %d→clock-saturate natural RC1 /
killed=0 with three red — against production-bin green controls:
bidirectional discrimination on the final bytes. The former "phase not
delivered / parent-RC & post-cancel sentinels await testing"
current is superseded by the nine narrow phases and stands as a
timestamped timeline; guard_closure_summary.txt's old NOT-done
list is superseded by final3 (old file preserved unchanged). Real
failures are NOT washed green: soft-scope / private-process-API /
.status early reds, the pre-fix normalleader cleanup-window red,
the harness faults (30 s outer-timeout cause unknown), the
negative-injection wrong-line error (sed hit comment L47 instead
of mono_ms L100) plus one negative-control outer RC124; the
first date RC0 was an UNINJECTED pseudo-green (injection-tooling
error, not a fixture blind spot), and the withdrawn "FAKE-HOLD
cover" explanation is corrected by the real-injection reds.
Wording guard: the consumer's 16 s natural RC1 with exitcode
0 != 124 proves the assertion red and the absence of TIMEOUT
only — the original "sleep-30 ran to completion" account is
UNPROVEN, and exitcode 0 is not proof of a natural wrapper
success; the cause of the third (no_live_in_group) red is not
natively measured — no SIGKILL-mechanism claim is substituted,
only the static note that the harness cleans up AFTER the
assertions (not a source or fixture blind spot); unmeasured
internal paths are not inferred. Scope
honesty: the runtime proven range is ONE required file's phase
runs, not the 21 REQUIRED all-run; the ALL whole-file run is
explicitly NOT authorized (an authorization boundary, not new
debt); all model / full-suite / multiday / GPU / throughput
pauses remain; no atomic-cancel self-heal, no cross-cluster byte
guarantee, no performance claim. Post-closure binding
(final3_runtime_snapshot.txt + final3_resource_state.txt, read
by this pass): the CURRENT repo's directly measured SHAs (bin
c704…, test 4b13…) match the final3 as-run objects, with each
phase's original log and both true-red records bound by full
SHA; the negroot badbin now holds the date-injected state (its
%d-era hash remains in negative_control_hashes.txt), so the
earlier "negroot-copy-only" caveat is downgraded to a pre-closure
timeline note; the current resource state is quoted from
final3_resource_state.txt as-is (julia_procs=0, sleep_orphans=0,
terminals_opened_this_route=0, HEAD=602b897, dirty=45,
git_writes=none). All prior per-mission history and un-run
declarations above are unchanged.

Status wording corrected (2026-10-07 documentation audit): the
earlier sentence "this closes the M1 minimal source scope" is WITHDRAWN
as a closure claim — what is verified is exactly the listed
single-instance comparisons and nothing more. The M1 boundary still
carries in-progress or freshly delivered, unverified items: the
workspace-generation boundary now carries NARROW GREEN of this
assignment (do not conflate with the earlier 56-item historical
number): prepared_problem_contract_tests 61/61 RC0 (23.3 s) and
prepared_lifecycle_tests 24/24 RC0 (12.4 s) under the scoped guard
(45 s / RSS 2048, clean load, single thread, no fixes, no relaxed
gates), plus the standard ArchitectureGate at the same entry (legal
minimal/nested RC0, bad/empty RC1, real-tree --architecture-only RC0
with object hashes unchanged before/after); within that fixture scope
these 85 items prove the caller finite/NaN old-Prepared freeze and
fresh response, the explicit mask copy, and the workspace-reuse model
snapshot. shared workspaces remain
generation-borrowed and the default no-ws path stays independent. (The
earlier in-progress list — B/CLI unrun, dev probe re-wiring unrun,
dynamic test-include in progress, CLI on the OLD admission protocol —
is superseded by the run-observed facts below and stands only as
timeline, not as current state.) Owner
work recorded as source facts and direction only — NOT as run-verified
results (final outcomes will be reported back and re-audited here):
ReviewMath has DELIVERED (source-written; the new
`prepared_lifecycle_tests.jl` has since been MEASURED 24/24 RC0 (12.4 s
under the scoped guard: 45 s / RSS 2048, clean load, single thread, no
fixes, no relaxed gates — narrow green of this assignment, never read
as whole M0/M1))
the `alive_now::Vector{Bool}` boundary — an owned constructor copy
defaulting to `isfinite.(adj_act last row)` computed only at constructor
time, with `solve` deriving e0 via `findall(prep.alive_now)` and no
longer reading `adj_act`, plus the same-day-observation owned mask
added without copying the whole panel; ReviewEngine is fixing the
daily-mask path (not activation rho); ReviewGate is making the dynamic
include fail closed; DevOps is fixing guards only (guard wrapper final run: 8/8 RC0 in 15 s,
deadline 50 s, RSS peak 321 MiB — see the measured-baseline section).
DevOps evidence update: 13 /tmp artifacts are preserved and the old
text scripts (m1_convert, m1_canon, m1_rescheck, i0_fit) exist and are
readable, exact paths to be supplied via the Manager; the canonical
checker seed is 1 with S=300; the prep NamedTuple's 22 keys and the
5-key incremental wrapper are evidenced in dev/m1_typed_replay.md §4.
The test-registry lifecycle is in place; the historical 2026-10-07
count was 18 REQUIRED files (including four legacy important test
items promoted), and the current registry requires 21 (the three
admission-integration deliveries joined after). The core
src new fields are covered by the 85-item narrow green above and the
CLI/Gate facts are run-observed below; the earlier "written but unrun"
wording stands only as the delivery-time record. Not whole M0/M1 release, not throughput. Whole M0/M1 release, throughput and
full-suite multi-day runs remain forbidden: `src/prepare.jl` now
defines the typed boundary `PreparedProblem` (with `FoldStatistics`,
`MacroStatistics`, `MacroFoldBlock`) and the two preparation backends
`prepare_reference` / `prepare_incremental`; `predict.jl`'s
`solve(prepared::PreparedProblem)` is the single assembly of the full fit,
the independent OOF stage and the fractional posterior, and `response.jl`
remains the single mathematical owner of the fit itself — the older
`fit_v1` / `_fit_prepared_v1` entries are forwarders onto this path, not
second implementations. Short rationale (links: src/prepare.jl ownership docstring; AGENTS.md
§44–47): `alive_now` is an OWNED O(N) Boolean copy frozen at
construction — chosen over copying the whole O(T·N) panel because solve
never reads the panel (e0 comes from `findall(alive_now)`) and the
owned copy makes post-prepare panel mutation unreachable from any solve
of that object; chosen over a read-only convention alone because
conventions are unenforced; the generation binding (deliberately not a
lease framework) then covers the workspace-borrowed matrices — relying
on generation ALONE would not stop caller-side panel mutation, and
copying ALL Grams was rejected for the real per-day cost on the
incremental path. Cost: one O(N) copy per prepare. Assumption: the
decision-day mask is fully determined by the panel's last row at
prepare time. Revisit triggers: a solve that needs panel data beyond
alive_now, a field-level regression in the comparison gates, or a real
ownership failure attributable to this boundary. Snapshot semantics: a
prepared snapshot is single-consumption; `X_rel`/design may borrow
workspace buffers, no per-day deepcopy is added, and `advance` never writes the workspace — but
a re-`prepare` may reuse those buffers, so no promise is made that an
arbitrary number of prepares keeps earlier snapshots frozen. The
production diagnostic counters of the pre-M1 era (`oof_shadow`,
`OOFCounters`, the config-identity exception) are removed (see the
historical protocol note in the baseline section). The removal is now
PHYSICAL: `src/ceiling_probes.jl` is deleted, KTrader's default
include/export of the probes is deleted, and backtest's `probe` parameter
and collector plumbing are deleted. The real dev entry point is now RUN-OBSERVED (2026-10-07, DevOps):
the CLI exercised a legal 400×2 CSV roundtrip, legal B and small-phase
solve commands, and the illegal unknown-flag / probe-matrix commands —
all observed RC; the library-side cli_boundary suite is 50/50 RC0,
admission 17/17, model-API 8/8. The B/collector tests are now 274/274 RC0 in 13.6 s under the
existing configuration (the earlier 104 pass / 6 fail / 1 error state
and its three fixture causes — T400 two-run 100/298 not 299,
first-price 200 → first-return 201 at position 152 not 151,
scalar-broadcast fixture — are kept as the fix timeline; a compile=min
failure is retained as a non-throughput data point, never a throughput
figure). The final dev oof_timing_split wrapper passes through
rep.observer_error and the timing, the tests' observer_error assertions
are restored and the warmup_error substitution withdrawn; admission 17
and model-API 8 re-verified green.
`dev/probes.jl` defines `module DevProbes`, which
imports Core and internally consumes `KTrader.prepare_reference`,
`KTrader.solve` and `KTrader.solve_current!`; `bin/ceiling_probes.jl` is
wired as `include("src/KTrader.jl")` then `include("dev/probes.jl")` then
`using .DevProbes`. The remaining ordinary probe contracts are being
re-wired to the dev consumer and remain standard REQUIRED files at
unchanged strength. AGENTS.md remains the single normative
SPEC for this boundary (its mathematics and targets unchanged — the
predictive law is still response mean plus innovation); this paragraph is
the implementation-side status, not a second spec.

M1 architecture decision (rationale, bounded to this one choice — the
governing contract is AGENTS.md §44–47 and §73–74; no second SPEC is
created here). Context: the mathematics is frozen; the pre-M1 shape
carried loosely-typed NamedTuple plumbing, production counters, and probe
concerns inside the solver's owners. Chosen: a typed `PreparedProblem`
boundary with a single `solve(prepared::PreparedProblem)` owner, probes
relegated to dev-only namespaces, and ordinary `DecisionTiming` for stage
accounting. Credible alternatives, rejected: continuing with loose
dictionaries/NamedTuples (no field-level contract, drift invisible until
runtime); replicating the solver or building an AOP layer for exact
internal counting (a second mathematical owner — exactly what the SPEC
forbids — bought only for diagnostics); a per-day deepcopy of the whole
Gram state (resource cost paid on every decision to purchase a permanence
the snapshot contract does not need); and substituting a large backtest
for boundary proofs (end-to-end numbers cannot localize a boundary
regression — the SPEC's field-level comparison and the test gates are the
proof instrument). Workspace-validity decision (supersedes the earlier
"await Ragged's certification" note; the mechanism was delivered
SOURCE-WRITTEN with tests updating and all unrun — that wording is the
delivery-time record, superseded by the narrow green recorded below;
the run-status paragraph at the end of this section carries the current
calibrated scope): Ragged confirmed a shared workspace
ACROSS consecutive prepares does pollute the earlier snapshot, and his
first fix — copying all Grams on every prepare — was REJECTED because the
incremental production path deliberately shares one workspace per day and
a daily copy would add a real per-day cost (the ~63 MB figure is an
N=100 full+fold EXAMPLE scale, not a measured N=65 baseline — do not cite
it as one). The landed mechanism, with its actual field names:
`FitWorkspace.generation::Base.RefValue{Int}` plus the `Prepared.ws_owner`
/ `Prepared.ws_generation` binding; `_prepare_v1` increments the
generation before ANY buffer use, so a partial prepare failure also
invalidates old borrowings; solve raises an explicit stale-generation
error on mismatch. The full Gram/design copies are WITHDRAWN — only the
narrow residual-source materialization (r / macro) is kept; the default
`ws_owner === nothing` is the independent reference path; repeated solves
of the same prepared input are legal until a re-`prepare`, and `advance`
does not affect validity. The borrowed generation marks CACHE validity,
not a mathematical winner version — it answers "is this buffer still the
one this snapshot was built from", never "which solve is correct". The two
credible alternatives for the workspace question, rejected: owned full
Grams per snapshot (pays the rejected per-prepare copy), and an open-ended
lease with no generation check (the original pollution bug, silently).
Consequences, stated honestly: the internal fp/m/core counters are no
longer a stable production API (they remain historical testimony in the
baseline section above); dev tooling requires explicit loading; the
generation mechanism was RUN PENDING at delivery time — the
delivery-time "tests updating, nothing executed" wording stands as the
timeline record — and has since been run-verified as NARROW green
(2026-10-07/08 DevOps, scoped runs under the existing configuration,
no fixes and no relaxed gates: prepared_problem_contract_tests 61/61
RC0 23.3 s — the stale-lease red, the failed-prepare generation bump,
the caller-mutation freeze and the workspace-borrowed snapshot
isolation are exercised on their fixtures; prepared_lifecycle_tests
24/24 RC0 12.4 s; narrow green is fixture-scoped and is never whole
M0/M1). Revisit triggers:
field-level or mathematics regressions in the comparison gates, a real
resource/ownership failure attributable to the boundary, or an explicit
change of the SPEC. Entry gates and tests: the admission gates and
contract tests referenced in the baseline and probes sections above are
the enforcement points.

Run-scope calibrations (2026-10-08, documentation-only assignment; no
command was executed for this paragraph — the named evidence-weight
correction of the cold-LLVM note line lives in
dev/evidence/manager4/evidence_corrections.md): "21 REQUIRED registered
+ file-presence admission verified" is NOT "21 numeric suites all ran" —
the numeric suites actually run before this assignment are exactly the
six files budget/cache/primal/prepared/lifecycle/replay, each its own
scoped run; the historical "18/12" registry figures remain the true
timeline record of their dates and are NOT rewritten as "21 all ran";
the 9×65 residual check is SAMPLE-ROW fidelity, never a full-history
T×N claim; floating-point tolerance agreement is not byte-level or
cross-cluster identity. Both fixes are now RUN-VERIFIED as narrow green
(2026-10-08 final DevOps evidence, logs in dev/evidence/manager4/;
the earlier 'entrusted, source not reported' and 'source-written, run
pending' wordings stand as the delivery-time timeline, superseded
here; narrow green is never whole M0/M1): (a) the ruler_stats same-source guard —
verify_ruler_stats_prefix (src/numerics.jl:301) streaming-recomputes
the consumed read points (row T, ACTIVE columns only, every tau) from
THIS prefix's log prices: counts exact (hard equality), acc at the
existing rtol=64eps/atol=0 roundoff tolerance, NaN/Inf rejected. The
guard's rejection surface is "caches whose CONSUMED statistics disagree
with THIS input prefix" (this round's same-shape different-amplitude
counterexample is one such rejection): it checks consumed-statistic
agreement, NOT provenance identity or a price-byte fingerprint — a
different history that happens to produce identical consumed statistics
would pass and the ruler would be unchanged, by design; observable
future suffix rows are never read. It is wired
in _prepare_v1 BEFORE ruler_from_stats (src/predict.jl:230) and unified
with initialize_inference's own accumulated statistics on the
incremental entry (src/incremental.jl:191, exact counts, same
tolerance); kwargs/schema unchanged; O(N_active·|TAUS|·T) time, O(1)
scalar working accumulators, no full 3D copy — a correctness guard,
explicitly NOT claimed as a performance optimization (no
zero-allocation or throughput figure pre-filled); red/green cases in
test/history_cache_contract_tests.jl. And (b) the replay's verification
and consumption sharing ONE manifest-certified .jls path —
path_of_record is the single path owner, every consumer deserializes
ONLY through load_verified (ONE read whose exact bytes are re-verified
sha256+size against the process-registered record before
deserialize(IOBuffer(data)) — a file replaced after verification can
never reach the consumer, and an unregistered lookalike is refused
before any read/parse), and persisted provenance must pass
assert_source_of_record, which the historical unbound lookalike fails
by construction — no 'suffix-only' fix, no 'once-verified path may be
blindly re-read'; no-overwrite, the manifest, old artifacts, phases,
tolerances and seeds unchanged. Run evidence (dev/evidence/manager4/):
artifact_replay_contract_guarded.log 82/82 RC0 (4 s, peak 367 MiB);
after the final comment-only temporalization of the contract test's
header, DevOps re-verified the final test bytes —
artifact_replay_contract_final.log 82/82 RC0 (3.1 s, peak 369 MiB,
killed=0; the guarded and final logs are two different time points
around the comment-only change, not competing numbers; the final
snapshot final_snapshot.txt, 16 objects with full per-path sha256, is
filed alongside — final_summary.txt is a 10-line run list and does NOT
record the process-0 / HEAD 602b897 / dirty-45 / no-commit-reset-clean
figures, and neither does final_snapshot.txt: both files were read in
full by the 2026-10-08 fifth pass, and those figures were a
PRIOR-REPORT carry, not file content — named correction:
dev/evidence/manager5/evidence_corrections.md; a later DevOps manager5
preflight observation recorded such fields for its OWN time point — it
cannot retroactively certify the manager4 files)
replay_check_guarded.log check RC0 (3 s, peak 501 MiB) — still four of
the thirteen manifest artifacts, never a full-13 claim;
rewrap_guarded.log RC0 (16 s, peak 1265 MiB) — the REAL .jls through
load_verified rewrapped into the typed PreparedProblem (T=14309, N=65,
P=910, F=3, owned alive_now 65, ws owner nothing, generation 0), no
prepare/fit/solve/scenario, no new artifact; solve_once_refusal_45s.log
natural RC1, killed=0 (deadline 45 s, elapsed 15 s, peak 1213 MiB),
refusal before any load/fit, recorded output sha256 unchanged
(solve_once_refusal_45s_shaset.txt). Elapsed times are observations
only, never performance comparisons — the 15 s refusal is NOT
attributed to warm IO/cache (untested) and the native compile share is
unknown. The real-tree architecture gate and the Registry 21+1
file-presence gate were exercised through the real entry
(arch_existing_final.log RC0, 7 s, peak 840 MiB; admission fail-closed
before and after) — still NOT "21 numeric suites all ran". The
failure timeline is preserved in AGENTS.md and
dev/evidence/manager4/evidence_corrections.md (RC124 attempts, the
cache-before-admission ordering violation, the guard-0 early runs as
behavior evidence only, the RC127 arg misuse with "zero side effect"
withdrawn, the 25 s solve-once interruption, the SHA-set construction
error corrected). The historical
rewrap RC0 record plus the current script's path literals must NOT be
spliced into any inference about the other input path hardcoded in the
replay script's history — /tmp/prep_inc_t14309.jl (no 's', unregistered
in the manifest; its existence and content format are UNKNOWN, and the
extension alone does not prove it is text) — which is distinct from the
manifest-certified /tmp/prep_inc_t14309.jls; and in particular NOT into
the claim that the RC0's consumed object was that unregistered .jl (no
time-bound record exists to support such a retroactive binding). All suspensions and protections stand: full
suite/multiday/macro-benchmarks/GPU/throughput paused, every command
≤60 s, the old HEAD and the user's uncommitted changes protected, panel
provenance unknown.

Fifth documentation pass (2026-10-08; documentation-only, no command
executed by THIS pass; named record:
dev/evidence/manager5/evidence_corrections.md): (a) the standard-entry
4-way summary's "(scoped_run 25s, RSS2048)" header is a MIS-transcription
of the four logs: minimal/nested ran deadline=12/elapsed=1s, bad/empty
deadline=25/elapsed=3s, and ALL FOUR logs carry rss_guard=0 /
rss_peak=0MiB — RSS-2048 compliance and the guard configuration are NOT
derivable from that summary; the red/green verdicts themselves stand on
the four sha256-bound logs. (b) the process-0 / HEAD 602b897 / dirty-45 /
no-commit-reset-clean figures attributed to final_summary.txt above are
a PRIOR-REPORT carry, not manager4 file content (both manager4 files
were read in full; neither records them; a later DevOps manager5
preflight observation recorded such fields for its OWN time point and
cannot retroactively certify the manager4 files). (c)
ReferenceIngressRepair: DELIVERED and RUN-VERIFIED narrow green
(source-owner static delivery; DevOps controlled runs; THIS fifth
documentation pass read the manager5 logs directly — arch_gate.log,
history_cache_ingress.log, primal_prep_ingress.log and
ingress_repair_summary.txt — and found them consistent with the
reported results, no discrepancy): prepare_reference (src/prepare.jl)
now has a closed explicit seven-keyword whitelist (ridge_alpha, F_folds,
ruler_stats, history_cache, alpha_initial, timing, workspace; defaults
unchanged), and Julia keyword dispatch rejects every other keyword —
including the three internal overrides (ruler_override /
scale_override / statistics_builder) even when passed as nothing —
BEFORE the function body runs, hence before any workspace generation
bump, cache read or builder execution; _prepare_v1's private path and
the incremental state-owned overrides are untouched (predict.jl
interface comment only); the new "closed public keyword surface"
testset lives in the existing REQUIRED
test/history_cache_contract_tests.jl. Run evidence
(dev/evidence/manager5/, each its own scoped run): arch gate RC0
(7 s, killed=0, guard 2048, peak 840 MiB, real standard entry first);
history_cache RC0 (28 s, killed=0, guard 2048, peak 1039 MiB) with six
testsets 210/210 = 24+9+15+45+32+85 — the new testset 85/85 with REAL
negative controls (P1: the three internal keywords even as nothing are
rejected; P2: refusal leaves the workspace generation unchanged and
the probe builder unexecuted); primal_prep RC0 (16 s, killed=0, guard
2048, peak 1029 MiB) 46/46 with the private incremental route
preserved. Four objects' SHA256 are fixed before/after in
ingress_repair_summary.txt; no self-fixes, no tolerance or guard
relaxation; existing single-threaded semantic configuration, not a
throughput figure. Wording calibration: the negative-injection
observation means invalid calls under the NEW interface throw with
zero workspace/builder side effects; "these assertions would be red
under the old wide interface" is a static counterfactual — no
old-source or mutant run was performed, never to be written as an
old-interface red run. Boundary note: this is NOT a
claim that all external callers must go through prepare_reference only
— fit_v1/solve remain the legal public chain; what is closed is the
internal-override surface at the public reference entry. The
design-level rationale (why an explicit whitelist over bare passthrough
/ a three-name blacklist / caller discipline; compatibility
consequences; revisit triggers; SPEC §46/§55/§56 links) is recorded in
dev/evidence/manager5/evidence_corrections.md §3. The 21-REQUIRED
registration is not 21 numeric suites all run — this verification
batch's numeric runs are exactly two files (test/history_cache_contract_tests.jl
and test/incremental_primal_prep_tests.jl) plus the one architecture-gate
command; no N65 refit, no artifact replay, no multiday/fullsuite/GPU/
bench. Every command stays ≤60 s / RSS 2048; full-suite / multiday /
GPU / throughput remain paused.

The coordinator initializes on the **full first decision prefix**, advances
statistics only, and clones at block starts. Each clone is exclusively owned by
one block; checkpoint copies mutable statistics/history/warm starts and creates
a fresh FitWorkspace. Read-only history/ruler caches alone may be borrowed.
Coordinator advancement must not change its snapshots. Each solve must return
a V1Model independent of subsequent advance/solve operations; production-boundary
regressions exercise both incremental clones and reused batch workspaces.
The incremental engine (RaggedExact, source frozen, verification in
progress) factorizes the ragged field by **mask runs**. For any row mask the
embedded field is linear in today's metric — e_t(d) = P_t·diag(z_t)·d with
P_t = M_t − m_t·m_t'/c_t (missing components exactly zero; fully observed
rows share the fixed P = I − (1/N)·11′): the return-row axis
is partitioned into maximal intervals of constant observation mask. Within a
run the projector is constant, so interior feature rows are single-mask and
aggregate into per-(mask, fold) Grams through a registry that merges repeated
masks into one entry. Rows whose 2·tau window (tau ≤ 128, deepest touched
index 256) or whose target row crosses a run boundary carry multi-mask
prototypes and are solved by exact per-row materialization — never by a
permanent batch fallback; `async_seen` is diagnostic only and a partially
observed row never disables the fast path. The remaining fallbacks are
numerical (`InferenceContractionError`) and `:resource_budget` — materialized
rows beyond `materialize_row_limit` (default 4096) or a new (mask, fold) Gram
beyond the byte budget (default 512 MiB per state, both route-only and
configurable) route the solve to the unchanged batch oracle on the **same
full prefix**, which is exact: the budget routes, it never changes outputs.
IPO/activation grows the active set by sorted insertion; history is
preserved by an explicit permutation embed of all statistics and stored rows
(new asset columns are exact zeros, matching the zero-filled history), with
`:embeds` counting structural embeds and `:rebuilds` remaining zero. The
ragged `s_perp` keeps the batch denominator (full `N·(L−tau)` count under
zero embedding, `N/(N−1)` normalization). Macro scaling is delegated to the
batch call (`scale_override.s_m === nothing` recomputes from the public
`X_m` path). With a statistics builder the n_res×P design matrix is not
built at all: primal fits consume Grams alone and dual fits (n < P) rebuild
exactly the rows they need from the cumulative field. Each solve still
recomputes the OOF residual machinery and the fractional likelihood — the
macro scalar series is O(T) per solve and the fractional FFT convolution is
O(T log T) — so no constant-time solve or whole-pipeline O(1) is promised.
Detailed raw-statistics architecture remains the engine owner's contract.
This describes the source as statically inspected; it is not a claim of
verified runs.

The schedule is one continuous window: `execution_timing.windows` is 1, blocks
are the contiguous partition of the whole decision range, and the consumer is
strictly date-ordered. Block tasks prepare date-fixed models/scenarios
(adaptive: models only); the ordered consumer computes all final weights using
actual drifted holdings, including locked and adaptive paths, then marks the
original adjusted-price returns and deletes the consumed spool file. The
per-date RNG remains MersenneTwister(seed + price_row_index). Inference
checkpoints are not whole-backtest resume checkpoints: they do not contain the
portfolio ledger. BLAS thread configuration is process-global, so independent
backtest calls must not concurrently change it in one process.

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

The standalone integration regression is `test/timeblock_tests.jl` (it also
runs as part of the suite):

```bash
julia -t 4 --project=. test/timeblock_tests.jl
```

It requires the production incremental API, and fails rather than skipping when
that API is absent. `test/runtests.jl` now includes it, together with
`conditioned_eb_tests.jl`, `ceiling_probes.jl` and `relative_support_tests.jl`.
The registry loop unconditionally covers all REQUIRED files (18 when
this sentence was written; currently 21 — the three admission-
integration deliveries joined; see the M1 boundary section),
including `residual_oracle_flow_tests.jl` (view-interchange and
fallback-entry flow contracts) and `posterior_contract_tests.jl`
(moving-fold warm/cold and quadrature lazy-vs-dense contracts) — both
formerly listed as "not yet wired"; that earlier state is superseded
and stands only as timeline.
Added assertions cover partitions/task bounds, batch versus incremental,
checkpoint and old-model independence, locked/adaptive consumption, late
activity/missing bars, tails, parameters, future-input isolation and BLAS
restoration after failures. Writing these assertions is not evidence they pass.

Tests cover distribution invariance under price-unit and asset-coordinate
changes; interleaved inactive assets; missing-pair prefix rulers; arbitrary
folds; primal/dual null-space uncertainty; conditional evidence against dense
Gaussian oracles; causal FFT convolution beyond the old history limit; original
Kelly certificates and exact-objective fallback; nested quadrature; and chunk
schedule independence. They do not equate identical seeded Monte Carlo draws
under a basis change with invariance of the predictive distribution.

## Performance measurement

**Single-point primitive measurement scope (2026-10-07, dev harness audit).**
The t14309 single-point primitive timings (M 0.06–0.17 ms, eigS 0.26–0.34 ms,
cached_state 0.66 ms min with GC spikes to ~320 ms where @timed attributes
304 ms to GC, certificate 71–126 ms / 59.6 MB per call) are valid **only for
that one well-conditioned point (cond(S)=66)**. They must NOT be multiplied
by another point's iteration counts (e.g. the pathological t14294 fold3
2232-step count) to claim fold-level or pipeline-level wall shares — the
pathological point's own primitive costs were never coherently measured
(eigSigma vs cached_state range inversion there remains unattributed), so
"share of fold wall" is **not determinable from current evidence**. The
certificate call structure (per-iteration + final, response.jl L882/L966)
is a source-code fact and may be cited; per-call costs are point-specific.
Note `cache.rank` is defined as `size(gauge,2)` (= d = 64), NOT the spectral
rank (910) — do not misread it as a truncated spectrum.

```bash
ENGINE=compare BENCH_DAYS=500 BENCH_REPEATS=3 CHUNK_SIZE=24 DATE_TASKS=6 BLAS_THREADS=1 julia -t 6 --project=. bin/bench.jl
BENCH_DAYS=500 DATE_TASKS=3 BLAS_THREADS=2 julia -t 6 --project=. bin/bench.jl
BENCH_DAYS=500 DATE_TASKS=2 BLAS_THREADS=3 julia -t 6 --project=. bin/bench.jl
ADAPTIVE_SCENARIOS=true julia -t 6 --project=. bin/backtest.jl
```

Both scripts accept `ENGINE=batch|incremental` and `CHUNK_SIZE`; the benchmark
additionally accepts `ENGINE=compare` and `BENCH_REPEATS`. They print Julia/thread
configuration, panel/input dates, decision start, seed, ridge/fold settings,
scenario/adaptive parameters and execution topology. Benchmark trials alternate
engine order, use the same loaded full-history panel and check economic output
equivalence; failed checks abort rather than count as speedups. Synthetic warmup
exercises the selected timeblock engine and is reported separately. No measured
history is shortened by warmup or by `BENCH_DAYS`.

`@timed` reports actual call wall time, allocations and GC time. Optional
`execution_timing=KTrader.BacktestExecutionTiming()` records preparation, full
prefix initialization, checkpoint cloning, coordinator-only advances, block
advances, model solves and date consumption, plus block/advance counts (one
continuous window: `windows` is 1). Block timing sums overlap across tasks:
they are elapsed-time sums, **not process CPU time**. Initialization and clone
costs remain included in overall wall time. Twelve decision buckets (the
original nine `prep`, `basis`, `gram`, `eigen`, `EB`, `condition`, `fracFFT`,
`scenario`, `Kelly` plus `:OOF_fit`, `:OOF_predict`, `:OOF_residual` for the
out-of-fold stage) are nested measurements relative to the execution totals,
not extra time to add to them. The per-fold refit is wrapped whole in
`:OOF_fit`, and the fold's `fit_response_operator` call does not receive the
timing object, so `:eigen`/`:EB`/`:condition` time the full-data fit only and
do not overlap the `:OOF_*` buckets; if that call ever forwards the timing
object, these buckets become nested and must not be summed — the
documentation must be updated together with such a change.
Adaptive work remains together in `scenario`.

The equivalence gate preserves dates/symbols, locked counts and scenario counts
exactly; return/wealth comparisons use `atol=rtol=1e-8`, weights `1e-6`. Model
regressions also compare predictive means at `atol=1e-10, rtol=1e-8` and OOF
residuals at `1e-8`. These are acceptance thresholds, not claims of achieved
results or permission to change the law. Original log-Kelly certificate gates
remain unchanged. Repeated same-machine measurements are needed before any
performance conclusion.

A ten-year **decision range** is not a ten-year **input history**: all earlier
available observations remain in every prefix. FFT size and Gram work must be
estimated from the actual history, not just the number of backtested decisions.
No 60-second runtime or 5.7× speedup is assumed.

### Ceiling probes CLI (`bin/ceiling_probes.jl`)

Diagnostic-only instrumentation; it never changes posterior mathematics,
fallback semantics or decision output. Source protocol as actually
implemented (read from the current script; older activation tables are
HISTORICAL records, not new data): the engine's daily mask is defined on
RETURN rows — finite(log s[k+1] − log s[k]), two real price endpoints —
and runs/transitions are computed on the FULL prefix, with `--from`
cropping only the in-range statistics over [from, T−1] (bar T is never
read; the final-active statistic explicitly excludes it). rho_256 uses
the forward price-bar window [t, t+255]. `boundary_cost` is a SEPARATE
statistic — the transition-driven exact-materialization boundary —
never to be conflated with rho_256 or with the exact materialized row
count (resource-budget demotion adds rows on top). Activation (first
observation of an asset) is a different, monotone statistic reported
under its own name, never as a mask run. Why the names stay separate
(short rationale): rho_256 answers a user question about forward-window
observation coverage, boundary_cost answers an engine question about the
exact-materialization boundary — conflating them mis-attributes cost,
and resource-budget demotion adds further rows on top, so neither equals
the materialized row count.

Saved-artifact B topology (finite-endpoint scope, T14309/N65 prefix,
last bar an ignored sentinel): whole prefix (from=2) K=14308, H=59, unique masks 60, median 121.5,
p90 617.3, max 1966, forward = boundary = 7803/14308 = 0.5453592395862454
(rounded report 0.5454); trailing 2000 decisions (from=12310): H=8,
unique 9, median 239, p90 441.6, max 460, forward 1508/2000 = 0.754,
boundary 1507/2000 = 0.7535. Provenance boundary, kept open: the original panel
source is unknown and no surviving original-call-chain `signal_prices`
evidence exists — the 530055 NaN count does NOT prove non-carried data,
and the historical rho 0.563 difference cannot be attributed to the
one-bar shortfall while provenance is unknown (different metric or
initial-mask definitions are possible but unproven). The manifest correction has landed (verified by direct read of the
final manifest; its external identifier is sha256
d290718e56fe1907196ffa62e3ef861dc11968d25005d315d6b441f6bdaa942d —
the manifest does not self-reference): the wrong 191/350 and 263.9/350
fractions are deleted, and an authorized single pure-mask precision
re-observation (zero prepare/solve/scenario/fit, scoped_run 50 s /
RSS 2048, RC0) recorded the exact integer coverage — whole prefix
from=2: forward = boundary = 7803/14308 = 0.5453592395862454; last-2000
from 12310: forward 1508/2000 = 0.754, boundary 1507/2000 = 0.7535. The
earlier wrong fractions remain only as a one-line timeline note, never
as metrics. Admission is shaped by an incident
(2026-10-07): a run with the old defaults (all five probes, D at
repeats=100 with cold **and** warm modes, each with an automatic warmup
solve) hid roughly 202 solves behind one flag, exceeded the per-command
60 s budget, and printed bucket summaries only after stages finished — an
aborted run left nothing to locate. Causal notes recorded so the detour is
not re-explored:

- `--from` only crops the **backtest evaluation range**. D/E never read it:
  their input is the full `1:t_star` signal prefix (t_star defaults to the
  last decision bar). Re-running the same stage with different `--from`
  values while waiting on a timeout produces no new information — the same
  work runs on the same full prefix. No N shrink, no history truncation, and
  no training-prefix cropping may ever be used to make a diagnostic green.
- A tool-level abort (timeout) is not process death: the flushed stderr
  stream survives. Every stage prints an immediate `[stage:prep|warmup|
  solve][start]` line **before** doing any work (carrying t_star/
  history_rows/active/features/mode/run) and an immediate `[...][end]` line
  after (with elapsed and per-bucket seconds), so an aborted run always
  shows which stage it entered. Inside the stages, every `DecisionTiming`
  bucket that flows through `timed()` (numerics.jl — the single owner of
  bucket timing) also prints an immediate `[stage:…][bucket:start|stop]`
  line, so a timeout inside a solve shows WHICH bucket was entered, not
  just the stage title. The hook is an optional `sink` on the `DecisionTiming`
  instance: default `nothing` keeps the numeric and additive timing semantics
  byte-identical; a live sink is owned by the single diagnostic phase that
  created it, is never global, and never escapes into fitted models.
  `:prep`/`:basis` accumulate by direct addition in predict.jl and expose
  no bucket boundary — the prepare stage's own start/end lines are their
  coverage; the ceiling side does not replicate the algorithm to fake one.
  Observer failure policy (same owner): a PRIMARY failure propagates
  unchanged with the observer's secondary error recorded separately as
  `timing.sink_error`; a bucket whose work succeeded but whose observer
  failed RAISES the observer's error — never a silent success; notification
  overhead stays out of the measured bucket time (stamps are notification
  wall clocks, `seconds` measures only the work).
- Admission: one command = at most one prepare, and at most one solve only
  under the explicit `--phase solve` selection (TWO phases: prepare, then
  one solve on that same prepared input — never described as a single
  phase). The default `--phase prepare` performs NO solve. D (elapsed
  baseline) and E (OOF bucket split) are readings of that single authorized
  solve. Multi-date A/C runs (K>1 decision days), repeats>1, dual-mode
  loops and `warmup=true` — including explicit opt-in — are **rejected**
  before any data is touched: a warmup is a second full solve and the
  single-solve cost is not yet verified. This is a 2026-10-07 gate, not a
  permanent ban: re-admission requires a measured single-day solve cost
  plus an explicit decision (the checks in solve_stage_probe and the
  compatibility wrappers are the single places to lift). There is no
  `--force` and none will be added; re-admission of multi-date A/C likewise
  requires verified single-day correctness and cost. All admitted commands
  plus teardown must fit in 60 s. No cross-process prepared-input cache
  exists: the CLI's solve phase consumes the prep built by its own prepare
  phase in the same command; the library's `solve_stage_probe(prep)` is the
  public single-solve API for any in-process consumer.
- The solve is a prepared-solve timing baseline — not a mathematical floor
  and not a CPU ceiling (single-worker latency does not bound worker-parallel
  throughput) — and excludes scenario generation and the Kelly solve. Stage
  buckets are the existing `DecisionTiming` wiring; see the OOF bucket
  scope note (fold refits currently receive `timing=nothing`, so full-fit
  buckets and the three `:OOF_*` buckets do not overlap) above.
- Compatibility: `solve_floor_probe`/`oof_timing_split_probe` keep their
  return shapes but `repeats` now must be 1 (was 100/20; >1 raises) and
  `warmup` defaults to false (was true, and never runs automatically).
  New single-stage entry points: `prepare_stage_probe` (prepare only) and
  `solve_stage_probe` (one solve on the shared prepared input; t_star and
  fold mismatches are rejected).

### Measured baseline (single sample, 2026-10-07)

One measured run of the ceiling probes' single-phase entries, recorded as
facts only: full prefix t_star=14309, N=65 active assets, P=910 features,
F_folds=3; Julia single-threaded, BLAS 6 threads; numerical cold init
(JIT specialization excluded by an uncounted warm-up call — this is cold
NUMBERS, not cold CODE). prepare 0.90 s; solve 17.844 s of which OOF
15.128 s (85%); scenario (S=300) 0.267 s; Kelly 0.036 s with the KKT
certificate at 1e-10. This single-phase measurement is ONE sample on which prepare did
not dominate — it does NOT prove that every historical timeout had this
cause, it does NOT exclude prepare as a general ceiling or bottleneck
(single point, single machine, one configuration; the 0.90 s figure must
not be used to declare prepare "solved" or "not the bottleneck"), and
it is NOT a throughput ceiling claim, a multi-sample measurement, or a
statement that the suite is green.
Additional measured facts from the same investigation, same qualifiers:
the default `materialize_row_limit` (4096, `src/incremental.jl`) triggered
`:resource_budget` on the real t14309 panel with 7803 materialized rows
(the route to the batch oracle — outputs unchanged by contract); an
explicit diagnostic run at 8192 ran the fast path with fast/rebuilds=0 and
the metric/relative and lazy macro Gram comparisons passed at their
original tolerances; the batch side's `macro_stats=nothing` is the legal
lazy-consumer semantics of predict.jl's statistics-builder branch, not a
version difference. The incremental-side timings from that run —
initialization 12.8 s, advance 8.8 ms, fast prepare 3.19 s — are
single-slice measurements, NOT cross-day throughput evidence, and the
1861 MiB figure is a stage-level RSS sample only. Process guarding:
`bin/scoped_run.sh` provides a sampled process-group kill monitor
(`--rss-guard` capped at 2048 MiB, ≤100 ms polling; nine self-test
scenarios green; deadline 50 s plus teardown keeps the total ≤60 s;
DevOps final wrapper run 2026-10-07: 8/8 RC0 in 15 s with RSS peak 321
MiB — the ALL-PASS summary line itself counts as one run, the real
independent script scenarios remain NINE, not ten; named full hashes
of record: bin/scoped_run.sh sha256
c3d776e40f39a32f0a144e3749a7ba2c3a9ccc6c7ddb3d4610d28277d04a26d6,
test/scoped_run_tests.jl sha256
eb9dec1b71982ab5d639c42c912a3020526c1d383c9201bb7dcd23006a8eac5f) — a
SAMPLED guard, not a cgroup instantaneous hard limit, and it must not be
documented as one. Source baseline at measurement time: `src/response.jl` sha1
`4d23e45ab61f` — a TRUNCATED historical citation: the full digest was
never recorded in the documentation, so this value CANNOT serve as a
final manifest field. Manifest policy (2026-10-07 audit): final manifest
fields require NAMED FULL hashes (algorithm plus complete digest, never
truncated and never reconstructed values); every truncated hash in this
file — this one and the sha1/sha256 correspondence pairs further down
(`f797050940f9`/`3747314d7c11`, `e05b40c6d89c`/`a8440fe1a3ea`) — is a
historical citation only, with full digests pending from DevOps. A
snapshot hash taken when the numbers were produced, not an a-priori
immutable reference; if the sources change, this paragraph is historical
comparison only.

Decision rationale recorded with the baseline (decisions, not chat
transcripts): no implicit warm-up, no multi-solve, default phase is
prepare-only; scaling up any workload requires a measured cost first plus
an explicit approval; the 60 s budget is a RESOURCE deadline (the command
is killed) — flushed log lines already written before the kill survive,
which is exactly why every stage prints its start line before doing work.
Status correction for the fixed-alpha contraction-kernel work (final,
2026-10-07 — this supersedes the earlier "must re-verify" note; no stale
pending remains): the diagnostic exception protocol reached its final
source form — the dedicated `OOFShadowConfigError`, a REAL `showerror`
method, and numerical-NaN PARTIAL semantics — and under those final
sources the OOF diagnostic, the forward-15 comparison, the API-8 checks
and the admission gates were ALL RE-VERIFIED GREEN. Two small fixtures are
independently green as well: `test/posterior_contract_tests.jl` and
`test/residual_oracle_flow_tests.jl` (the latter's NaN-matrix snapshot uses
`isequal`, not `==` — zero tolerance, no relaxation). The standard test
contract registry carried 18 REQUIRED files at that historical moment
(2026-10-07; the current count is 21 — see the M1 boundary section;
history not rewritten); `test/runtests.jl`
verifies each file exists BEFORE the mandatory include, and a real
known-bad entry (missing file) exits RC1 with zero testsets run. The FULL
suite has not been run — "re-verified green" names the listed items only.
The chosen path and its reasons stand
as recorded: keep per-fold evidence (folds must not see each other's
held-out rows through shared state), keep the Sigma cold reference/basin
(the posterior basin must be entered from the same cold reference the
certificates were defined on), keep certificates independent (each fold's
acceptance is its own gate), and avoid blind state caching (no reuse of
fit state across folds without an equality witness). Alternatives rejected:
sharing the full-fit alpha across folds (leaks held-out information into
fold refits), freezing the metric (changes the fitted law, not just its
numerics), rank/history reduction (a different mathematical input,
forbidden by the no-shrink/no-truncation admission contract), and Sigma
warm-starting (entered without basin verification — the same opt-in
failure mode the warmup gate closed). A local fixed-M witness exists:
n=120/N=3 direct oracle versus kernel agrees numerically at 2.78e-17;
a single small-panel timing is NOT extrapolated to N=65. Diagnostic exception protocol (HISTORICAL decision, 2026-10-07 — the
production diagnostic counters it governed have since been REMOVED from
the production sources as part of the M1 architecture work: `oof_shadow`,
`OOFCounters` and the config-identity exception are no longer production
APIs. The record is kept as the decision rationale, not as a description
of a current production facility; the historical counts and performance
baselines above are unaffected and are not re-presented as new APIs): the
former shadow/OOF diagnostic factory signaled CONFIG-IDENTITY misuse with
a dedicated `OOFShadowConfigError <: Exception`; the probe rethrew ONLY
that type, while numerical failures inside the measured work still yielded
the PARTIAL report semantics (recorded, never silent). Rationale (why a
dedicated type, kept for any future diagnostic that needs the same
distinction): the underlying layers also raise `ArgumentError` for
genuinely bad numeric input (e.g. a NaN matrix), so `ArgumentError` itself
cannot distinguish "the caller misused the factory" from "the measurement
found bad numbers". Alternatives rejected: a catch-all that stringifies
and swallows the type (a broken config would be recorded as a valid
measurement); rethrowing every `ArgumentError` (collapses the
numerical-PARTIAL semantics into a crash); subtype inheritance of
`ArgumentError` (not possible — it is a concrete type in Julia); a broad
`AppException` hierarchy (an unearned framework); message parsing
(stringly-typed identity). The contract test that pinned the protocol,
`test/solve_stage_shadow_contract_tests.jl`, has been DELETED together
with the feature it covered (the counter test file was removed first);
the remaining ordinary probe contracts are re-wired to the dev consumer
and remain standard REQUIRED files at unchanged strength.

Performance accounting (facts, no speedup claims): the timing artifacts use
DIFFERENT accounting conventions and must not be conflated — the
ports-carrying artifact reads fit_seconds 27.0206 s; a separate no-ports
artifact records the stage-internal total 19.293 s with the 7-bucket sum
17.212 s; the 24.825 s figure is the observation of the corresponding
OUTER call/startup wrapping, not a third comparable range, and must not be
mixed into either. There is NO warm steady-state throughput measurement.
The first 12.154 s figure included a synthetic specialization warm-up and
is not directly comparable. `/tmp/solve_final.log` (main EB 9.695 s, OOF
6.784 s) has been RE-CHECKED by DevOps against the original log — the
stage-internal 19.293 s, the 7-bucket sum 17.212 s and the EB/OOF figures
are confirmed; the not-fully-isolated portion of the differences must not
be attributed wholesale to JIT. Source-identity convention: hashes must be
cited WITH their algorithm name — the earlier response/numerics
"mismatch" was a hash-ALGORITHM mix-up, not a source change, resolved by
the real correspondence (response sha1 f797050940f9 = same file sha256
3747314d7c11; numerics sha1 e05b40c6d89c = sha256 a8440fe1a3ea); one
evidence list suffices when every hash carries its algorithm. The
`/tmp/ports_sidecar_primitives.jls` artifact carries the no-sink-callback
primitive table: fold-relative (fp, m_calls) = (16, 37), (23, 52),
(201, 2232); the macro side performs ONE alpha search of 78 evaluations
and does NO Sigma alternation — which is not the same claim as "the whole
macro EB is closed-form". The M path is kernelized; the basin-difference
question is unproven and the old 1e8-flops GEMM is not a demonstrated next
saving (neither direction is established). The current hotspot-reuse
investigation is READ-ONLY so far: no new optimization has passed, and no
new performance conclusion is recorded.

Accepted single-day evidence (t14309, I0 budget — Manager-adjudicated
acceptance of THIS one day, not a widened same-root 1e-12 claim): the
incremental prepare feeds the official COLD fit at 22.805 s (OOF 18.562 s),
compared against the batch oracle with alpha_m = 0, alpha_rel relative
8.5e-14 (abs ≈ 1.27e-9; the original 1e-8 also passes), mu_pred 1.13e-14,
Sigma Frobenius 6.1e-15, S=300 same-seed scenarios X 9.77e-14, w_L1
2.31e-12; both certificates KKT ≤ 1e-10 and feasibility ≤ 2.22e-16;
minimum wealth ≈ 0.8044. Cross-batch-solver differences (X 1.913e-9, w
3.078e-8) pass under the EXISTING combined tolerances and the I0 weight
tolerance 1e-7; the genuine cross-swap of weights between the two solves
changes utility by about ±1.2e-16. The earlier 6.29e-9 was a DUALITY-GAP
budget (inside the certified 1e-8 tolerance), NOT a swapped-weight loss —
the two must not be conflated.

Standing conclusion: performance admission is NOT met; macro/full-suite
runs remain forbidden and the production default has not been switched.
What is released is the exact incremental implementation plus the
restricted single-day evidence above — not a 20–30 days/s throughput claim
and not a one-minute acceptance.

What would trigger a revisit of any of this: new
real single-day per-stage timings, the principal-root contract
certificates (`test/principal_root_contract_tests.jl`), boundary
regressions in the resource-budget route / Gram tolerances, the full-suite
run (currently not executed), or evidence contradicting the accepted
single-day numbers recorded above.

## Changes

- 2026-10-07: ceiling probes single-stage admission (incident: the old CLI
  defaults hid ~202 solves behind one flag and exceeded the 60 s
  per-command budget with no stage boundary in the logs). D/E are now at
  most ONE prepare + ONE solve on the full `1:t_star` prefix (two readings
  of the same solve); `repeats` must be 1 (raises otherwise, no override),
  mode is a single cold-or-warm per run, and `warmup=true` is rejected
  before any data is touched — including explicit opt-in (2026-10-07 gate;
  re-admission requires a measured single-day solve cost plus an explicit
  decision). The CLI default is `--phase prepare` (no solve at all);
  `--phase solve` is two explicit phases (prepare, then one solve on that
  same prepared input), never labeled a single phase. Multi-date A/C runs
  (K>1) are rejected at the CLI and in `sequential_ceiling_probe`. Every
  stage (prepare/solve) prints an immediate flushed start/end line
  (t_star/history_rows/active/features/mode/run/elapsed) to stderr, so tool
  aborts leave locatable logs.
  New entry points `prepare_stage_probe`/`solve_stage_probe` (t_star and
  fold mismatches rejected; no re-preparation, no N shrink, no history
  truncation). `--from` documented as evaluation-only and unread by D.
  Bucket-level immediate start/stop lines via an optional observer `sink`
  on `DecisionTiming`, hooked in numerics.jl's `timed` (the bucket-timing
  owner): default `sink === nothing` preserves the numeric and additive
  semantics exactly; a live sink belongs to the single probe phase, is
  never global, and never enters fitted models. `:prep`/`:basis` (direct
  addition in predict.jl) have no bucket boundary — documented as such.
  Probe B unchanged (pure statistics). Compatibility: old return shapes
  kept; `repeats` defaults 100/20→1, `warmup` true→false.

- 2026-10-07: one continuous timeblock schedule for the whole decision range
  (no per-chunk windows; `execution_timing.windows` is 1); the default
  non-adaptive path spills per-date results to a single-writer disk spool so
  all blocks solve in parallel (resident memory: O(date_tasks) transient
  solver results plus O(K) token metadata — no "one live object" claim);
  adaptive keeps bounded in-memory channels with consumer-order-limited
  throughput; lazy `ResidualOracle` is the production OOF residual path with
  `dense_oof_residuals` as the test-only reference; documentation synchronized
  with these sources (lazy wiring, OOF supervision-layer isolation, warm-start
  semantics, fixed-metric contraction and the asynchronous-observation
  fallback as a representation boundary, theory/numerics layering, O(T)/
  O(T log T) per-solve recomputations). No measured throughput is claimed;
  RaggedExact's mask-run factorization is now implemented (source frozen,
  verification in progress) and documented above by its source semantics
  only.

- 2026-10-07: bounded-window contiguous timeblocks with task-owned state/workspaces;
  explicit batch/incremental selection; ordered locked/adaptive consumption;
  initialization/checkpoint/advance elapsed measurements and same-input engine
  comparisons; standalone production-boundary regressions (not executed here).

- 2026-10-06: correct prefix missingness and active-column mapping; explicit
  activity and ragged field semantics; general fold-only statistics; conditional
  EB evidence on relative support; certified log-Kelly; spectral primal/dual
  covariance; planned dynamic FFT; batched predictive draws without shifting;
  bounded streamed backtesting; reusable workspaces and timing/topology controls;
  opt-in nested randomized quadrature; fast closed-form EB fixed point with stationarity
  certificate; in-place Cholesky OOF fold solver; LAPACK syevr! for RRR eigensolve;
  block-direct contraction for trace-neutral condition moments; memory-stride aligned buffers.

## Scenario 数值耦合契约（2026-10-07，Manager 裁决）

`generate_scenarios_v1` 的 relative draw 使用 moments covariance 的**唯一
对称 PSD 主平方根**（`principal_sqrt_root`，SVD 路径：`U·Diag(s)·U'`，
与右正交自由度解耦）作为同 seed 有限场景的规范耦合。

- **性质**：这是数值耦合修复，不是 Gaussian 模型变更——概率律、秩、
  资产次序、Kelly 目标、负 λ 裁剪策略、S、随机变量数量均不变。
  契约：同 Σ 下 L 的任意右正交变换（列符号、一般正交、简并、秩亏）
  同 seed 场景逐点到 roundoff 一致（`test/principal_root_contract_tests.jl`）。
  canonical `principal_sqrt_root` 的契约定位：同一协方差、同一 Gaussian
  基底，消除的只是**根坐标自由度**（L 的右正交不唯一性）；现有
  roundoff 级对照（1e-12 容差，`src/numerics.jl` 的 SVD 路径）是同机
  同栈下的数值一致性，**不是跨机器字节级保证**——跨机器仍受
  LAPACK/SVD 数值差异影响，跨机器复现以容差与 seed 契约为准。
- **历史影响**：旧 raw `L·z` 路径下，非规范的有限 MC 节点权重可能
  改变（实测 t14309 切片 w_L1 漂移 9.6%）；**不声称旧 w 保持**。canonical
  化后 old/new 两侧 artifact 对齐（X 8.9e-16、w 1.2e-12、四路 KKT 证书
  同值）。
- **性能双历史观测**（非吞吐、非提速证明）：同一 frozen prep 单 cold
  solve 的两个记录观测——旧源 17.844s（特化已预热）与新源 12.154s（含
  首次 JIT、未预热）。两次观测口径不对称、无 controlled before/after，
  不构成任何优化幅度或其下界的证据（2026-10-08 文档校准：撤回此前
  "真实优化幅度 ≥32% 方向"的措辞——那是从两个不对称观测推出的未证
  下界；此处只陈述两个观测值本身，不添加新的性能归因）；OOF_fit
  15.1s→8.5s 为另一对历史观测，同样只作记录、不作提速归因。
- **未跑**：full suite、多日 backtest（未授权）。旧 /tmp artifact 仅
  调查对照用途。

### Julia 干净加载协议（诊断/验证用，非生产吞吐标准）

受控验证/诊断命令的推荐加载方式（避免自动磁盘 precompile 挤占 RSS
预算）：

```bash
JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
julia --project=. --startup-file=no --compiled-modules=existing \
  --pkgimages=existing -t 1 <test-file>
```

`--pkgimages=existing` 使 Julia 1.12 使用已有 pkgimage 缓存（同源码
内存加载），不触发自动磁盘编译；`JULIA_PKG_PRECOMPILE_AUTO=0` 阻止
包级自动 precompile。配合 scoped_run 的 `--rss-guard=2048`（采样级
kill 护栏）与 deadline ≤48s，可将 Julia 测试进程组 RSS 控制在 ~1GiB
（实测 kernel 899MiB、OOF 1008MiB、jcore 934MiB）。此协议用于诊断
与验证配置，不冒充生产吞吐标准。
