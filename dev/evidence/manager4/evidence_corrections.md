# Evidence corrections — manager4 (2026-10-08)

Documentation-only assignment (Engineer, 4th tenure): no command, git,
test, or build was executed for this file; no production src, test,
manifest, summary, log, or snapshot was modified. This file corrects the
EVIDENCE WEIGHT of one recorded claim; it does not alter any record.

## 1. standard_entry_4way_summary.txt, note line (cold LLVM latency)

Record under correction (preserved verbatim, one line, not edited —
dev/evidence/manager3/standard_entry_4way_summary.txt:7):

    note: first 12s-deadline attempt hit cold LLVM specialization latency (124 timeout), not gate failure; 25s rerun exits naturally in 3s

Correction of evidence weight:

- FACTS (observed): the first 12 s-deadline attempt returned RC124
  (timeout/stop by the deadline guard) with an LLVM stack in the trace;
  the 25 s-deadline rerun exited naturally in 3 s with RC1 (the expected
  red for the `bad` fixture root).
- NOT A FACT (attribution, unproven): "hit cold LLVM specialization
  latency" as the CAUSE of the first timeout. No native compile-time
  measurement exists for that run; an LLVM stack during a Julia
  first-load under a tight deadline is consistent with several causes
  (specialization, pkgimage load, GC, machine noise). The attribution
  is a plausible reading, not a measurement, and must not be cited as
  one — in particular it is NOT a quantified "compile share" figure and
  must not be used to derive one.
- NO RETROACTIVE INFERENCE: the clean 3 s natural RC1 under the 25 s
  deadline is direct evidence for the rerun's behavior only; it does
  not retroactively prove the first attempt's cause. The original
  timeout record stands as history and is not erased.
- The four-way verdicts themselves (minimal/nested RC0; bad/empty RC1
  natural exit, killed=0) are unaffected by this correction; they rest
  on the four sha256-bound logs, not on the note line.

Original artifacts preserved unchanged:
dev/evidence/manager3/standard_entry_4way_summary.txt,
std_entry_{minimal,nested,bad,empty}.log, final_snapshot.txt.

Referenced from: README.md (run-scope calibrations paragraph), AGENTS.md
(第4任文档校准段), dev/m1_typed_replay.md (standard-entry 4-way section).

## 2. Final run synchronization (2026-10-08, DevOps evidence — logs in this directory)

Narrow greens of record (each its own scoped run; elapsed times are
observations only, never performance comparisons):

- history_cache_existing_final.log: five testsets 24+9+15+45+32 =
  125/125 RC0 (existing configuration, single thread; deadline 45 s,
  elapsed 25 s, RSS guard 2048, peak 1034 MiB); no self-fixes, no
  tolerance relaxation; source/test sha256 fixed before and after
  (seven files hashed in the log).
- arch_existing_final.log: real-tree architecture gate RC0 (7 s, guard
  2048, peak 840 MiB); the Registry 21+1 file-presence gate was
  exercised through the real entry, fail-closed, before and after —
  which is NOT "21 numeric suites all ran".
- artifact_replay_contract_guarded.log: replay contract suite 82/82
  RC0 (elapsed 4 s, deadline 20 s, peak 367 MiB).
- artifact_replay_contract_final.log: after the final COMMENT-ONLY
  temporalization of test/artifact_replay_contract_tests.jl (header
  lines 55-57 turned into a history note; dev script, running source
  and assertions unchanged), DevOps re-verified the FINAL test bytes:
  82/82 RC0 (3.1 s, deadline 20 s, peak 369 MiB, killed=0). The
  guarded log above and this final log are TWO DIFFERENT TIME POINTS
  (before / after the comment-only change), not competing numbers.
  Filed alongside: final_snapshot.txt (16 objects, per-path full
  sha256) and final_summary.txt (process 0, HEAD 602b897, dirty 45,
  no commit/reset/clean).
- replay_check_guarded.log: check RC0 (3 s, peak 501 MiB) — still
  exactly FOUR of the 13 manifest artifacts, never a full-13 claim.
- rewrap_guarded.log: rewrap RC0 (16 s, peak 1265 MiB) — the REAL
  /tmp/prep_inc_t14309.jls through load_verified, rewrapped into the
  typed PreparedProblem (T=14309, N=65, P=910, F=3, owned alive_now 65,
  ws owner nothing, generation 0); no prepare/fit/solve/scenario, no
  new artifact.
- solve_once_refusal_45s.log: natural RC1, killed=0 (deadline 45 s,
  elapsed 15 s, peak 1213 MiB); the refusal fires before any load/fit
  (stack at dev/m1_artifact_replay.jl:553, after the four .jls
  manifest verifications); the recorded output artifact's sha256 is
  unchanged (solve_once_refusal_45s_shaset.txt). The faster 15 s
  elapsed is NOT attributed to warm IO/cache (untested); the native
  compile share is unknown.

Failure timeline, preserved (not erased, not counted as greens):
- arch 15 s and cache 35 s attempts returned RC124 (deadline stop).
- Running cache before arch admission was an ORDERING VIOLATION
  (acknowledged; the final runs respect the admission order).
- The early 82/check runs without an RSS guard (guard 0:
  artifact_replay_contract.log, replay_check.log) stand as behavior
  evidence only, NOT as RSS-compliant runs; the guarded logs above are
  the compliance records.
- RC127: a scoped_run argument misuse created a 132-byte file "./2048"
  in the working tree; it was moved to scoped_run_arg_misuse_2048.log.
  The earlier "zero side effect" wording is WITHDRAWN.
- solve-once 25 s RC124: its 15 s work window was interrupted during
  verify/SHA — NOT a natural refusal; only the later 45 s-deadline run
  (RC1 above) is the direct refusal evidence.
- A SHA-set diff between different artifact sets was a construction
  error, corrected by re-running the comparison over the same
  path-identical set (solve_once_refusal_45s_shaset.txt).

Scope unchanged: model-only / residual-sample / the N65 real fit were
NOT run in this assignment; macro-benchmarks / full suite / multiday /
GPU remain paused; panel provenance unknown; tolerance agreement is
not a cross-machine byte guarantee. The manager3 records and every
pre-existing log are preserved unchanged; this section calibrates by
name only.
