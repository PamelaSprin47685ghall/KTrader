# M1 typed PreparedProblem replay — dev-only, artifacts-only, explicit-phase checker

Status: DOCUMENTATION-AUDITED (2026-10-07). This note was rewritten to
separate three things that its previous version conflated: (a) values
that were actually recorded when the runs happened, kept verbatim as
history; (b) metadata that was CLAIMED to exist here but never did; and
(c) an explicit-phase replay plan that CANNOT run until DevOps supplies
the missing metadata. Nothing below is a new measurement; no command was
executed for this rewrite.

## 1. What this checker is allowed to do

- Consume ONLY preserved /tmp artifacts. No re-prepare of the full
  history, no baseline modification, no solver copy, no production
  compatibility layer, no diagnostic factory.
- - The only solve entry is `KTrader.solve` on a typed `PreparedProblem`
  built by the PRODUCTION typed constructors, never a second
  implementation. Source facts delivered by the prepare owner
  (ReviewMath, 2026-10-07; source-written at delivery time, since MEASURED): the typed
  `PreparedProblem` now carries `alive_now::Vector{Bool}` as an owned
  constructor copy, with default `isfinite.(adj_act last row)` computed
  ONLY at constructor time; `solve` derives e0 via
  `findall(prep.alive_now)` and no longer reads `adj_act`. Shared
  workspaces remain generation-borrowed; the default (no ws) path stays
  independent. A `prepared_lifecycle_tests.jl` is now a REQUIRED file and MEASURED
  24/24 RC0 (12.4 s, scoped 45 s / RSS 2048, clean load, single thread,
  no fixes, no relaxed gates — narrow green, never read as whole
  M0/M1); the earlier "about to be registered, not run" wording stands
  as the delivery-time record.
- Phases are individually and explicitly selected. There is NO default
  run-all. See `dev/m1_artifact_replay.jl`.

## 2. Recorded single-instance values (history, kept as recorded)

These were reported by the 2026-10-07 runs and are the ONLY numbers this
note may rest on. Source of record: the DevOps run reports transcribed
into README.md (M1 status paragraph and the "Measured baseline"
section) and AGENTS.md §53.

- Inputs of record: full prefix t_star=14309, N=65 active, P=910
  features, F_folds=3; one cold default-EB BLAS6 solve 7.449 s (artifact
  /tmp/model_m1_t14309.jls).
- Old prep artifact of record: /tmp/prep_inc_t14309.jls (old 22-key
  NamedTuple prep, incremental path).
- vs old I0_inc report (/tmp/model_t14309_inc.jls):
  alpha/Gmean/Sigma/mu/covLprod/d/vforecasts fields all exact 0-diff.
- vs batch baseline (/tmp/model_t14309_final.jls): alpha_rel relative
  8.53e-14, Gc 2.47e-13, Sigma 2.82e-13, mu 1.13e-14, cov 6.13e-15
  (original tolerances pass).
- Same-seed canonical S300 scenario/Kelly: X 9.77e-14, w_L1 2.31e-12,
  both KKT ≈1e-10, feasibility 2.2e-16, objective gap 6.29e-9, min
  wealth ≈0.804, swap loss floating-point zero.
- Residual sample check (artifacts only, no solve/prepare): per-fold
  first/mid/last rows [1,2342,4684,4685,7026,9368,9369,11710,14052],
  9×65 = 585 cells, 260 finite / 325 NaN; shapes identical; isfinite and
  NaN positions exactly equal on those rows; finite maxdiff 0.0 vs
  I0_inc and 3.0e-14 vs batch (original tol 1e-9 passes).

Qualifications that BIND every number above (do not drop them when
citing): single-case refactor fidelity; same input, same machine; the
0-diff covers exactly the listed fields; the residual check is
SAMPLE-ROW fidelity (9 rows), not a full T×N per-cell comparison; none
of this is a cross-machine or cross-cluster byte-level guarantee;
7.449 s has no controlled before/after and is NOT a performance claim,
NOT a steady-state speedup, and NOT evidence that M1 made anything
faster; the separately recorded 0.90 s prepare figure is one sample on
one machine and does NOT exclude prepare as a general ceiling or
bottleneck; none of this is whole M0/M1 release or throughput
qualification.

## 3. Honesty gap found in this note's previous version

The previous version of this note stated "sha1-frozen source", and the
README paragraph pointing here claimed this note carried "source/input
hashes with algorithm names, artifact/model paths …, the original eight
legacy artifacts preserved". Audit finding: this note contained NO hash
values, NO complete commands, NO seeds; it listed FOUR artifact paths,
not eight; the replay scripts themselves lived only in the /tmp of the
session that produced them and no copy exists in this repository. These
values are NOT reconstructed here — they cannot be invented after the
fact. DevOps update (2026-10-07, reported evidence): 13 /tmp artifacts
are now preserved, and the old text scripts (m1_convert, m1_canon,
m1_rescheck, i0_fit) exist and are readable — their exact paths are to
be supplied by DevOps via the Manager, not guessed here. Full artifact digests have ARRIVED: DevOps delivered
`dev/m1_replay_manifest.toml` — the ONE manifest (this note and the
replay script reference it and never create a second inventory): 13
artifacts with full 64-hex sha256 + byte sizes, checker-script
identities, historical log paths, current-tree source hashes (with the
sampling-time caveat: NOT the old-measurement sources) and the guard
final run. The numbers in §2 are now verifiable against the preserved
artifacts by sha256; panel provenance stays UNVERIFIED
(old_panel_hash = "unknown") — artifact-only local comparisons stand on
their own, but no claim of certified panel provenance may be made.

## 4. Metadata DevOps must supply before any replay phase runs

Each item is a precise fact; a phase whose items are unfilled refuses to
run (see dev/m1_artifact_replay.jl). Manifest fields take named FULL
hashes only — algorithm name plus complete digest; truncated values are
not acceptable and must not be reconstructed.

1. PARTIALLY RESOLVED, with a caveat that stands: the manifest's
   [source_tree_at_manifest] carries full sha256 for the CURRENT tree
   at manifest time — the tree was still receiving engineer edits while
   the artifacts were produced, so these hashes identify the tree as of
   the manifest, NOT the old-measurement sources. Old-measurement source
   identity is carried inside each artifact's own metadata and the
   historical logs; where no such evidence exists the manifest writes
   "unknown" rather than guessing.
2. RESOLVED AS UNKNOWN-BY-RECORD: the manifest records
   old_panel_hash = "unknown" — no hash was taken at prepare/solve time
   and the artifacts embed their own input snapshots instead. This does
   NOT block artifact-only local comparisons (each artifact's sha256 is
   verified), but NOTHING may claim the original panel provenance is
   certified.
3. RESOLVED: dev/m1_replay_manifest.toml [artifacts] carries ALL 13
   preserved /tmp artifacts with full 64-hex sha256 and byte sizes
   (superseding the old "eight" README claim and this note's four known
   paths). The replay script verifies the four paths of record against
   it via the Julia SHA stdlib (SHA.sha256 + bytes2hex). The manifest is
   DevOps-owned and is the single inventory — no second list is created
   anywhere.
4. RESOLVED by DevOps evidence (2026-10-07). The prep NamedTuple has
   exactly 22 keys: active_idx, N_universe, adj_act, T, N, r, s1,
   field, m, relative_embedding, X_rel, s_m, s_perp, B_m, ts_total,
   n_res, n_bands, P_features, X_rel_stacked, Y_target_rel, stats,
   macro_stats (the observation mask lives inside `field`). The old
   incremental wrapper around it has 5 keys: prep, fast,
   prepare_seconds, limit, peak_rss_kb. Source fact (ReviewMath): the
   old 22-key NamedTuple re-wraps through the keyword constructor, which
   auto-fills the `alive_now` mask; an old TYPED JLS artifact does NOT
   guarantee direct deserialization against current sources — the
   rewrap must be explicit and validated, never assumed green. Value
   TYPES per key are still to be confirmed by the check phase against
   the preserved artifact.
5. RESOLVED (static read, 2026-10-07): the old text scripts are
   /tmp/m1_convert.jl, /tmp/m1_canon.jl, /tmp/m1_rescheck.jl and
   /tmp/i0_fit.jl — glob-confirmed present and readable; their logic
   (rewrap field mapping, baseline diff field set, canonical seed flow,
   residual block selection) is now implemented in
   dev/m1_artifact_replay.jl. Still pending from logs: the exact
   command lines (julia flags, -t, BLAS threads, environment) as
   actually invoked.
6. RESOLVED (DevOps evidence + static read of /tmp/m1_canon.jl):
   seed=1, S=300, each model drawn from its own MersenneTwister(1)
   instance via generate_scenarios_v1(m; S=300, rng=MersenneTwister(1)),
   then scenario_weights on trues(N_universe) and kelly_certificate; the
   principal-square-root convention lives inside generate_scenarios_v1
   (the production canonical root). Implemented in the replay script's
   model-only phase.
7. RESOLVED: the manifest's [checker_commands] records the exact calls
   as run (m1_rescheck: finite-intersection absdiff tol 1e-9; m1_canon:
   seed 1, S=300, MersenneTwister per model). The replay script gates
   against these recorded tolerances only.

## 5. Explicit-phase replay plan (dev/m1_artifact_replay.jl)

One phase per invocation; no phase chains another; nothing re-prepares
the full history; nothing modifies a baseline; no solver is copied.

All phases except `check` verify artifact hashes against
`dev/m1_replay_manifest.toml` (the ONE manifest) before touching data.
Phase responsibilities: `check` loads NO KTrader and deserializes
NOTHING (model artifacts contain KTrader types and need those bindings —
that is the other phases' responsibility); `rewrap`/`solve-once`/
`model-only`/`residual-sample` load KTrader and deserialize.

- `check` — verify the listed artifacts exist; load each and report
  schema, dimensions and finite/NaN census. Hash verification runs only
  when §4 inventory metadata is filled; otherwise it reports
  HASH-UNVERIFIED rather than guessing. No solve, no prepare.
- `rewrap` — map the old 22-key NamedTuple prep to the typed
  `PreparedProblem` through the production keyword constructor (which
  auto-fills `alive_now` from the old `adj_act` at constructor time;
  `solve` later uses `findall(prep.alive_now)`, not `adj_act`). No
  solve. Old TYPED JLS artifacts are NOT assumed deserializable against
  current sources — explicit rewrap plus field-by-field validation,
  never an assumed green. Requires §4 items 1, 3, 4.
- `solve-once` — at most ONE cold `KTrader.solve` (default EB) of the
  validated rewrap, then GATED field-diff vs the two baseline artifacts
  plus the deterministic 9-row residual block (fail-loud gates; never
  materializes the full T×N residual matrix). Refuses its recorded
  output path BEFORE any deserialization / rewrap / solve, and refuses
  to repeat within one process. Requires §4 items 1, 5, 7.
  Recorded invocation of record (DevOps evidence): `KTrader.solve(prep)`
  with default cold EB, F_folds=3, ws owner nothing, generation 0.
- `model-only` — load an existing model artifact only; regenerate the
  same-seed canonical S=300 scenarios and Kelly; compare X, w_L1, KKT,
  feasibility, objective gap against §2. No solve, no prepare. Requires
  §4 item 6.
- `residual-sample` — load artifacts only; recompute the 9-row
  per-fold first/mid/last residual block comparison. No solve, no
  prepare. Can never certify full T×N cell equality. Requires §4
  items 4, 7.

Every phase verifies its metadata first and fails loudly (error, never
a warning, never a silent skip) when an item is unfilled. Success is
never assumed from a partial phase set; green requires an authorized
run.

## Comparison gates — REVISE (2026-10-08; the gated behavior is now RUN-VERIFIED narrow green — see "Replay run status — gated behavior (2026-10-08)" below)

The replay script's model-only / residual-sample / solve-once phases
previously PRINTED their differences and completed RC0 regardless: an
over-tolerance diff, a finite/NaN mask mismatch, or a NaN-poisoned
maximum(abs.(MA .- MB)) (solve-once materialized the full T×N residual
matrix, where any NaN cell made the "maxdiff" NaN and the phase stayed
green) all counted as pass. That false-green boundary is closed in
source (dev/m1_artifact_replay.jl):

- THE ONE comparison owner: pure gate functions (assert_masked_maxdiff,
  assert_matrix_close, assert_scalar_absdiff / assert_scalar_reldiff,
  assert_certified, assert_wealth_positive, assert_swap_nonneg) that
  ERROR on failure. The same numeric check is NOT re-implemented per
  phase; printed baseline numbers are context, never a pass.
- solve-once no longer materializes the full T×N residual matrix: it
  compares the deterministic 9-row block via residual_rows (shape,
  finite/NaN mask identity, finite-only maxdiff) — the same gate as
  residual-sample.
- solve-once refuses its recorded output path BEFORE any
  deserialization / rewrap / solve (previously the refusal came only
  after the expensive fit). The recorded path
  /tmp/model_solveonce_t14309.jl2s is registered in the ONE manifest
  and is never overwritten; a NEW output path is a Manager/DevOps
  decision plus a new manifest entry — the script never silently
  switches candidates or edits the inventory.
- The script is library-safe: include'd from the contract tests it
  defines the gates WITHOUT running main and WITHOUT loading KTrader;
  run as a script, the toplevel 'using KTrader' and the check-phase
  zero-KTrader contract are unchanged.

Tolerances of record — HISTORICAL sources (user-stated early /
original checker records; nothing reverse-engineered from observed
numbers): mu 1e-10; Sigma 1e-9; macro/relative alpha 1e-8 (absolute
difference); w_L1 1e-7; residual finite-cell maxdiff 1e-9
(/tmp/m1_rescheck.jl + manifest [checker_commands]); kelly certificate
feasibility / KKT / gap at the production default certificate tol 1e-8
(src/kelly.jl solver default).

Manager numerical-verification decisions (2026-10-08, THIS assignment)
— items with NO historical threshold, recorded as NEW gate decisions of
this assignment, NOT masquerading as historical assertions:

1. Sigma_rel and G_c_mean matrix differences: 1e-9, elementwise max
   |a-b| (the metric is stated in the gate output).
2. d_posterior / v_forecasts: absolute 1e-10.
3. Same-seed canonical scenario X: max 1e-10.
4. Cross-model objective difference: 1e-8 — a numerical-reproduction
criterion for this recorded same-input comparison, NOT a general
theory claim that two different probabilistic laws share objectives.
5. Swap-loss difference: 1e-12 pure-roundoff allowance (NOT relaxed to
observed values), applying only together with the production
certificates passing (the certificate gates run first).
6. Relative predictive covariance compares L*L' (never a bare factor
body), Frobenius 1e-9.
7. Residual sample: shape / finite / NaN positions exactly equal,
finite max 1e-9, Inf rejected outright — even at matching positions.

Contract tests (MEASURED 54/54 RC0, 1.9 s, 2026-10-08 DevOps under
scoped_run 45 s / RSS 2048, RSS peak 355 MiB; timeline: the first run
was 52 Pass / 2 Fail — both failures were hand-calculated index
expectations in the fold_sample_rows determinism testset, corrected to
the function's actual first/mid/last semantics; no gate logic or
tolerance was touched): test/artifact_replay_contract_tests.jl
exercises the gates on small arrays only — positive cases plus
adversarial reds (shape mismatch, finite↔NaN mask mismatch, Inf,
over-tolerance, bad certificates, negative swap, non-positive wealth,
fold-row determinism). No KTrader load, no artifacts, no
fit/prepare/scenario loops. It is registered among the 21 REQUIRED
files and included once by test/runtests.jl's standard loop (the
pre-registration "deliberately not wired" wording stands as timeline).
Green here certifies only the gate logic on these fixtures — never an
artifact comparison, never a replay phase.

solve-once verification scope (Manager, 2026-10-08): the recorded
output artifact already exists, so NO new fit is required or authorized
— the verification of this phase is exactly that the command FAILS
with the no-overwrite refusal BEFORE any deserialization / rewrap /
solve; no new output path and no new manifest record are added.
model-only / residual-sample / check remain load-only verifications
over the old objects.

The Replay run status section (2026-10-07) below records the OLD
println-only behavior's runs as history; the gated behavior has since
been run — see "Replay run status — gated behavior (2026-10-08)".

## Replay run status (2026-10-07, DevOps evidence — NOT a closure)

- check / rewrap / model-only / residual-sample: RC0. The old-model
canonical X diff 9.77e-14 and the sample finite maxdiffs 0.0 / 3.0e-14
are the EXISTING artifacts' comparisons (§2), now re-observed by the
replay phases.
- solve-once ran twice on the CURRENT source tree: 7.202 s with the
  object lost to stdout only (cost/mistake evidence retained, not
erased); 8.336 s including ~2 s persisting the new output to
/tmp/model_solveonce_t14309.jl2s (safe path — the script refuses to
overwrite any existing baseline), with the model-field dual-baseline
comparison and the 9-row residual block done on that run. The NEW artifact's sha256
  ae54c564f3005cff022703a15637f21be4c0a44f6261c92791c3948db0a2016c is
  registered in the ONE manifest and its same-seed numbers are MEASURED
  (final DevOps transmission): new vs inc/m1 X / w / objective / swap
  all exactly 0; new vs batch X 9.77e-14, w_L1 2.31e-12, both KKT ~1e-10,
  gap 6.29e-9, swap ±2.5e-17; 9-row 585-cell block: finite 260 / NaN 325,
  masks identical, finite maxdiff 0.0 vs inc and 3.0e-14 vs batch (still
  SAMPLE-ROW scope). 7.202 s (object not saved, rework cost retained)
  and 8.336 s (includes ~2 s persistence, NOT a speedup claim) stand as
  the timeline.
- DevOps changed the non-check toplevel load from include() to the
standard `using KTrader`: the Serialization PkgId error was caused by
the include's anonymous module lacking package-identity registration,
NOT by a Project UUID drift — the old artifacts were serialized under
the package UUID.
- Resolved: the script header's stale "NOT executed yet" line has been
  mechanically corrected by DevOps with zero control-flow change; no
  awaiting-correction note remains.

## Replay run status — gated behavior (2026-10-08, DevOps evidence; logs in archive/evidence/manager3/)

The hard-gate (REVISE) behavior is now RUN-VERIFIED narrow green on
this machine — distinct from the OLD println-only RC0 records above
(those runs belong to the pre-gate script and stand as history):

- check: RC0 (3 s) — the four core artifacts of record verified
  against the ONE manifest by sha256 + byte size; static schema
  cross-check PASS. Scope stays exactly FOUR of the 13 manifest
  artifacts (checker-script hashes displayed only) — never stated as
  a full-13 inventory verification.
- artifact_replay_contract_tests.jl: 54/54 RC0 in 1.9 s (scoped_run
  45 s / RSS 2048, RSS peak 355 MiB). Timeline: the first run was
  52 Pass / 2 Fail — both failures were hand-calculated index
  expectations in the fold_sample_rows determinism testset, corrected
  to the function's actual first/mid/last semantics; no gate logic or
  tolerance was touched.
- model-only: RC0 (20 s) — the load-only path now ALSO runs the
  strong model-field gates (compare_model_fields vs BOTH preserved
  baselines, Manager decision 2026-10-08; no new fit/prepare/model,
  tol/seed unchanged; the paths bind exactly the manifest's model
  artifacts, sha256-verified before any comparison — no baseline
  switch). 15 GATED field comparisons: vs inc ALL exactly 0; vs batch
  alpha_rel abs 1.27e-9 (tol 1e-8), G_c 2.47e-13, Sigma 3.97e-14, mu
  1.13e-14, L*L' Frobenius 6.13e-15. Scenario/Kelly gates pass at the
  original tolerances (X 9.77e-14, w_L1 2.31e-12, both certificates at
  the production 1e-8, min wealth ~0.804, swap within the 1e-12
  roundoff boundary, objective diff 5.72e-15).
- residual-sample: RC0 (16 s) — 9×65 block, 260 finite / 325 NaN,
  finite maxdiff 0.0 vs inc and 3.0035e-14 vs batch, mask positions
  exactly equal, Inf rejected. Still SAMPLE-ROW scope only.
- solve-once: RC1 EXPECTED RED (17 s) — the no-overwrite refusal
  fires BEFORE any deserialization/rewrap/solve (the stack trace
  confirms the error at the refusal; no [solve-once] elapsed output);
  no new fit, no new output path, no manifest change; the recorded
  output artifact's disk sha256 is unchanged.
- Real bad-field red (producer→parser→consumer):
  archive/evidence/manager3/model_only_gate_probe.jl deserializes the
  REAL model artifacts, runs compare_model_fields as the positive
  (PASS), then mutates ONE finite mu_pred cell in memory (+1.0) — the
  SAME consumer throws — and restores the in-memory value; the disk
  artifact is never touched (sha256 unchanged). Log:
  model_only_gate.log ([G] POSITIVE PASS / [N] NEGATIVE threw=true,
  scoped_run RC0).
- The directory also contains arch_f.log: the standard
  ArchitectureGate (--architecture-only, real tree) 25/25 pass, RC0
  (7 s) — recorded as directory content of the same run family.

Qualifications unchanged: single machine, artifact-only (panel
provenance still unknown), sample-row residual scope, no
cross-cluster byte guarantee, no whole M0/M1 release, no throughput.
The budget/cache contract runs of the same revision are now also
MEASURED (2026-10-08, logs in the same directory): budget 1041/1041
RC0 9.4 s (scoped 16 s / RSS 2048, peak 946 MiB); history-cache 4
testsets (24+9+15+45) all green RC0, scoped 20 s (the
initialize_inference testset 45/45 in 5.5 s, re-verified after the
ExactInferenceState constructor gained the two budget fields). Red
history retained: the first budget run was 1025 Pass / 10 Fail (total
1035, bud1_red_history.log) — the reds mix the real source defect
(advance_exact! dropping the validated gram_budget /
materialize_row_limit at first core creation — dead kwargs, silently
re-defaulting to 512 MiB / 4096) with two independent test errors
(activation-day timing: first finite price bar 360 vs first valid
adjacent return bar 361; a false Cartesian 6-Gram expectation vs the
actual 4). The fix saves both values in ExactInferenceState before the
lazy core exists, passes them at core birth and through checkpoint —
resource options for lazy core birth, not new theory knobs; no
tolerance or guard relaxed. Adjacent greens same batch:
incremental_primal_prep 46/46 10.2 s, prepared_problem 61/61 22.6 s
(contains small synthetic solves; "no fit" = no N65 real-artifact
re-fit only), prepared_lifecycle 24/24 11.8 s, architecture-only real
tree 5 summaries all pass (21+1 admission).

Admission-negative and snapshot closure (2026-10-08, same directory):
required_negative.log — the real production primitive
(harness_required_negative.jl includes test/contract_registry.jl and
calls verify_required_contract_files; a non-throw fails the harness):
one temporarily-missing REQUIRED file deterministically throws,
all-present passes, ZERO numeric testsets in that run;
fixture_root_4way.log — harness_fixture_root_4way.jl constructs the
four fixture roots and self-checks their CONTENT (occursin/isfile
asserts), 5/5 RC0 — it does NOT itself invoke the standard entry; the
standard entry's fixture-root red/green (minimal/nested RC0,
bad/empty RC1) was measured in the earlier DevOps runs recorded
above; as-run harness sources bound by sha256 in
harness_snapshot.txt;
a harness-side sed typo (occursin==9) was fixed during the runs — a
harness transcription error, never a production Gate red. "21
REQUIRED registered + file-presence admission verified" is NOT "21
numeric suites all ran": this revision's actually-run numeric suites
are exactly the six files above. The named final snapshot
archive/evidence/manager3/final_snapshot.txt carries full SHA256 for the
16 key runtime objects (no commit/reset/clean); this note and the
other documentation files are not part of the runtime snapshot; no
existing snapshot or manifest entry changed.

The real standard-entry 4-way for this revision has since been run
(2026-10-08 02:03 +08:00; standard_entry_4way_summary.txt +
std_entry_minimal/nested/bad/empty.log with sha256 binding):
minimal/nested RC0 (fixture mode, KTrader NOT loaded), bad RC1 natural
exit killed=0 (violations: unsupported include form, dev/probes.jl,
dev/more_probes.jl; violations==String[] fails at
architecture_contract_tests.jl:363), empty RC1 natural exit killed=0
(fail-closed L356/357 plus "include target not present: KTrader.jl")
— rejection causes checkable from the actual consumer output, no path
errors. The earlier content-self-check harness remains as auxiliary
evidence and the elevation wording has been corrected. Timeout
timeline kept: the first 12s-deadline attempt returned RC124
(timeout/stop, LLVM stack) — no native compile-time measurement exists
and the later 3 s clean RC1 does not retroactively prove its cause;
only the 25 s rerun's natural 3 s RC1 is direct evidence; the original
timeout record is not erased.

Guard-configuration transcription correction (2026-10-08 fifth
documentation pass; named record:
archive/evidence/manager5/evidence_corrections.md; summary and logs
preserved unchanged): the summary header's "(scoped_run 25s, RSS2048)"
does NOT match the four logs' scoped_run lines — minimal/nested
actually ran deadline=12 with elapsed=1s (the summary's "3s" for them
disagrees), bad/empty deadline=25 with elapsed=3s, and ALL FOUR logs
carry rss_guard=0 / rss_peak=0MiB (guard not in effect / not
measured). RSS-2048 compliance and the guard configuration are
therefore NOT derivable from that summary or these logs; the red/green
verdicts themselves (RC0/RC1, killed=0 natural exits, violation
contents) stand on the four sha256-bound logs, not on the header line.

Evidence-weight correction (2026-10-08, documentation assignment): the
summary's own note line attributes the first timeout to "cold LLVM
specialization latency" — that causal attribution is NOT measurement
evidence. The observed facts are RC124 (timeout/stop) with an LLVM
stack and the 25 s rerun's natural 3 s RC1; the cause is unknown and no
compile-share figure may be derived from it. Named correction of
record: archive/evidence/manager4/evidence_corrections.md; the original
summary/logs/snapshot are preserved unchanged.

Entrusted fix — FINAL RUN SYNC (2026-10-08: owner source formally
reported, statically cross-checked by this documentation pass, and now
RUN-VERIFIED narrow green by this assignment's DevOps — logs in
archive/evidence/manager4/; the earlier 'source NOT yet reported' and
'source-written, run pending' wordings stand as the delivery-time
timeline): the replay's verification and consumption share ONE
manifest-certified .jls path. Landed mechanism, as
delivered: path_of_record is THE one path owner; every consumer
deserializes ONLY through load_verified — ONE read of the path, whose
exact bytes are re-verified (sha256 + size against the
process-registered record) and then handed to
deserialize(IOBuffer(data)) with no second read, so a file replaced
after verification can never reach the consumer and an unregistered
lookalike is refused before any read/parse; persisted provenance passes
assert_source_of_record, which the historical unbound
'/tmp/prep_inc_t14309.jl' fails by construction — this closes both the
'suffix-only' pseudo-fix and the 'once-verified path may be blindly
re-read' pseudo-fix. No-overwrite refusal, the ONE manifest, old
artifacts, phases, tolerances and seeds are unchanged. The
historical rewrap
RC0 record (2026-10-07) must NOT be spliced into any inference about
the other input path hardcoded in the replay script's history —
/tmp/prep_inc_t14309.jl (no 's', unregistered in the manifest; its
existence and content format are UNKNOWN, and the extension alone does
not prove it is text) — which is distinct from the manifest-certified
/tmp/prep_inc_t14309.jls; and in particular NOT into the claim that the
RC0's consumed object was that unregistered .jl (no time-bound record
exists to support such a retroactive binding). The certified inventory
is the ONE manifest and path claims route through it. RUN EVIDENCE
(2026-10-08 final DevOps sync, archive/evidence/manager4/, each its own
scoped run; elapsed times are observations only, never performance
comparisons): artifact_replay_contract_guarded.log 82/82 RC0 (4 s,
peak 367 MiB); after the final comment-only temporalization of the
contract test's header (lines 55-57, history note only — dev script,
running source and assertions unchanged), DevOps re-verified the FINAL
test bytes: artifact_replay_contract_final.log 82/82 RC0 (3.1 s, peak
369 MiB, killed=0) — guarded and final are two different time points
around the comment-only change, not competing numbers; the final
snapshot final_snapshot.txt (16 objects, per-path full sha256) is
filed in the same directory; final_summary.txt is a 10-line run list
and does NOT record the process-0 / HEAD 602b897 / dirty-45 /
no-commit-reset-clean figures — neither does final_snapshot.txt (both
files were read in full by the 2026-10-08 fifth pass; those figures
were carried in the PRIOR REPORT about the file, not in the file
itself — named correction:
archive/evidence/manager5/evidence_corrections.md);
replay_check_guarded.log check RC0 (3 s, peak 501 MiB) —
still exactly FOUR of the 13 manifest artifacts; rewrap_guarded.log
RC0 (16 s, peak 1265 MiB) — the REAL /tmp/prep_inc_t14309.jls through
load_verified, rewrapped into the typed PreparedProblem (T=14309, N=65,
P=910, F=3, owned alive_now 65, ws owner nothing, generation 0), no
prepare/fit/solve/scenario, no new artifact; solve_once_refusal_45s.log
natural RC1, killed=0 (deadline 45 s, elapsed 15 s, peak 1213 MiB),
refusal before any load/fit, recorded output sha256 unchanged
(solve_once_refusal_45s_shaset.txt) — the 15 s elapsed is NOT
attributed to warm IO/cache (untested), native compile share unknown.
The failure timeline (RC124 attempts, the cache-before-admission
ordering violation, the guard-0 early runs as behavior evidence only,
the RC127 arg misuse with 'zero side effect' withdrawn, the 25 s
solve-once interruption, the SHA-set construction error corrected) is
preserved in AGENTS.md and archive/evidence/manager4/evidence_corrections.md.
The ruler_stats same-source public-entry guard is likewise RUN-VERIFIED
narrow green (history_cache_existing_final.log: five testsets
24+9+15+45+32 = 125/125 RC0, existing configuration, single thread,
deadline 45 s, elapsed 25 s, peak 1034 MiB, no self-fixes, no tolerance
relaxation, source/test hashes fixed before and after; recorded in
AGENTS.md):
verify_ruler_stats_prefix (src/numerics.jl) streaming-recomputes the
consumed (row-T, active-column, every-tau) acc/cnt points from THIS
prefix — counts exact, acc at the existing 64eps/atol=0 roundoff
tolerance — and REJECTS caches whose consumed statistics disagree with
this input prefix (the same-shape different-amplitude counterexample is
one such rejection); the guard checks consumed-statistic agreement,
NOT provenance identity or a price-byte fingerprint — a different
history producing identical consumed statistics would pass with the
ruler unchanged, by design; observable future suffix rows are never
read; wired in _prepare_v1 before ruler_from_stats and unified with
initialize_inference's own accumulated statistics; O(N_active·|TAUS|·T)
time, O(1) scalar working accumulators, no full 3D copy — a correctness
guard not claimed as a performance optimization (no zero-allocation or
throughput figure pre-filled); red/green cases in
test/history_cache_contract_tests.jl.

## B mask-topology facts (saved-artifact, finite-endpoint scope)

From the preserved B artifact on the T14309/N65 prefix (last bar an
ignored sentinel): whole prefix (from=2) K=14308, H=59 transitions, unique masks 60,
stable-run median 121.5, p90 617.3, max 1966, forward = boundary =
7803/14308 = 0.5453592395862454 (rounded report 0.5454); trailing 2000
decisions (from=12310): H=8, unique 9, median 239, p90 441.6, max 460,
forward 1508/2000 = 0.754, boundary 1507/2000 = 0.7535. These are "saved-artifact finite-
endpoint mask topology" facts ONLY — not a statement about any other
dataset.

Provenance boundary (kept open): the original panel source remains
unknown; no surviving original-call-chain `signal_prices` evidence
exists, so the 530055 NaN count does NOT prove non-carried data. The
historical rho 0.563 difference cannot be attributed to the one-bar
shortfall while provenance is unknown; different metric or initial-mask
definitions are possible but unproven. The manifest correction has LANDED (verified by direct read of the
final manifest; its external identifier is sha256
d290718e56fe1907196ffa62e3ef861dc11968d25005d315d6b441f6bdaa942d, the
manifest does not self-reference): the wrong 191/350 and 263.9/350
fractions are DELETED, and an authorized single pure-mask precision
re-observation (zero prepare/solve/scenario/fit, scoped_run 50 s /
RSS 2048, RC0) recorded the exact integer coverage cited above. The
rounded ragged6 log and the exact_cnt precision log are both preserved.
The earlier wrong fractions remain only as a one-line timeline note,
never as metrics.

## 6. What this replay can and cannot establish

Can (when green with complete metadata): that the typed rewrap and one
recorded solve reproduce the recorded single-instance values on this
machine — refactor fidelity of exactly the listed fields.

Cannot: whole M0/M1 release; throughput or steady-state speedup;
cross-machine byte identity; full T×N residual cell equality; and ANY
statement of run-verification beyond the fixture scopes actually
exercised. Narrow-verified WITHIN their fixture scopes: ReviewMath's
`alive_now` boundary (85-item narrow green; `prepared_lifecycle_tests.jl`
is a REQUIRED file, MEASURED 24/24 RC0 at 12.4 s under scoped 45 s /
RSS 2048, clean load, single thread, no fixes, no relaxed gates — the
earlier "about to register, not run" wording was the delivery-time
record); ReviewGate's fail-closed dynamic include (observed through the
standard Gate's positive AND negative cases: legal minimal/nested RC0,
bad/empty RC1); and ReviewEngine's daily-mask fix (not activation rho) —
the final ceiling_probes 274/274 RC0 run covers suspension/resumption,
from-range run sums, terminal-bar/IPO immunity and forward-vs-boundary
on its fixtures, and the real-prefix mask measurement is hash-bound in
the one manifest with exact integer counts. What remains unknown is
the original panel provenance, whole-pipeline behaviour and throughput —
not whether the daily-mask revision ran on those inputs. None of this
is a claim that the whole suite has run.

## 第6任评审注记引用（2026-10-08，文档 pass）

本 note 的 replay/guard 相关源码事实经第6任只读评审复核，具名记录于
archive/evidence/manager6/guard_scope_notes.md（本段仅引用，不重复数值、
不改上方任何历史记录）：guard 拒绝时序的精确口径——只有未知 keyword
的 Julia dispatch 拒绝先于 _prepare_v1 的 generation bump；合法
keyword 携带异源 cache/ruler_stats 的守卫拒绝在 bump 之后、首次数学
消费之前；failed/partial prepare 使同 owner 旧 lease stale 为既有
lease 契约、非缺陷。workspace slot 疑问已由 Manager 亲读 response.jl
关闭（fit_response_operator 函数体不用 workspace、build_X helper 仅
:path_sums/:design、solve 重建调用不传 workspace），不作为已证漏洞
记录。SixResourceGuardAudit 源码修复已交付并经本任文档 pass 静态核对
（bin/scoped_run.sh：TERM/HUP/INT 取消 trap + EXIT 兜底 +
pending-acquire、/proc/uptime %.0f 单调钟与坏钟 125、有界 leader
确认后 wait、三清理阶段共享唯一绝对截止；首版 SIG_IGN 继承/无界
wait/32 位 %d 饱和/预算叠加缺陷静态核对已修——时间线、不声称逐项
已运行）；不可捕捉界限（SIGKILL/SIGSTOP/宿主崩溃、继承忽略的
INT）保留，不写「任何外部终止原子自愈」；运行动态事实与 harness
错误时间线见 archive/evidence/manager6/；最终运行已成立（final3，
2026-10-08 05:09：九相 53 项全 RC0、unknown/duplicate 为预期 RC1
语义拒绝、same-consumer 真注入负控红 + 生产对照绿、最终 std
arch RC0 7s/RSS2048/peak841——runtime 证明范围是一个 required 文
件分相、非 21 全跑，ALL 整跑未授权；保留失败与措辞约束详见
guard_scope_notes.md §2.6；补齐：final3_runtime_snapshot.txt 实
测当前仓库 SHA 与 final3 as-run 绑定一致（各 phase 原 log 与
两真红记录 full SHA 绑定）、final3_resource_state.txt 记录当
前资源状态，均以文件实际内容为准）。

## 第7任文档校准引用（2026-10-08，文档 pass）

本段上方第6任引用中的「坏钟 125」口径经第7任静态核对收窄：
begin_cleanup_window / mono_elapsed_ms 晚期坏钟仅降级/0 占位、
可透传 rc0，start 坏钟 terminate 后 Cleaned=1 而无组净确认，
optional --rss-guard 后空 command 可被接纳（假绿准入）；现有
clock-fail 相只测 monitor。三条为本任静态发现、未动态复现、
源码 owner 修复进行中（不预填绿）；旧九相 53 项 RC0 与最终
std arch RC0 保留其已测范围有效（从未覆盖上述边界）。另：
guard_scope_notes.md:164 的 test/scoped_run_tests.jl full SHA
一字符转写（87339，source-of-record 为 87239）由
archive/evidence/manager7/evidence_corrections.md 具名纠正——本
note 只用短前缀、不含 full hash、无转写错误，上方历史记录一
字不改。
