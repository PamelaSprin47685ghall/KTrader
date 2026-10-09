# Frozen-candidate closure — 2026-10-09

Scope: freeze production, tests and dependencies; finish the previously
unrun cache-elision test; repeat complete default-EB/F3/S300 N65 windows.
No new mathematical optimization, thread defaults, memory limits, version
number or release tag. No promise of a release after an interaction count.

The last missing standard group has now run through the unchanged
`dev/release_batch.jl cache_elision` entry,110/110, RC0. Its existing
qualification snapshot is unchanged; `status_complete.log` reports21/21
groups and40 standard files. This is grouped coverage, not a single-process
full-suite run. Preserve the earlier failed/blocked records as history.

The only executable addition is `dev/release_freeze.jl`, an explicitly
separate evidence consumer. Every file in the existing qualification
snapshot must remain byte-identical. The new consumer is hashed in its
own snapshot; it is NOT retroactively asserted to belong to old test runs.

Measurements: one task, BLAS6, default GC, full-history batch inference.
Two independent process repeats of the latest eight-day window compare
against the authenticated existing eight-day result. A disjoint earlier
eight-day window is measured once and independently replayed once. Each
command has its own synthetic API startup and two-day numerical-cold warmup;
the measured window always starts with fresh holdings and alpha chains.
No rank/asset/history/fold reduction, fixed-alpha substitution or best-of
selection. Results count only if the original tolerances and source/input
hash checks pass. Formal repeat timings must have native compile_time=0.

Resource scope remains45s inner/55s outer and2048MiB sampled RSS guard.
Failure ends that command and is reported, not repaired with a larger limit.
Sequential commands, no concurrent numeric jobs. Four windows do not prove
long-horizon steady-state performance or CPU/GPU exhaustion.

The original AGENTS.md sections80–82 remain authoritative:2.0 includes
reference equivalence, replaceable acceleration, CPU/incremental decisions,
and conditional GPU work. The ~42days/s mission is not permission to change
mathematics, nor does short-window testing by itself satisfy these clauses.

Freeze STOPPED on real regression: `earlier1.log` failed in the real two-day
warmup, before any measured eight-day call, with InvalidStateException from
the consumer channel. The scheduler caught only EOFError and therefore
masked its producer's actual exception. Latest1/latest2 remain valid evidence
for the preceding frozen source. Status then correctly returned2/4, RC1.

A correctness-only patch now recognizes a drained closed channel and throws
the original producer exception. Valid buffered results are still drained
in order; a short stream without a recorded cause is a protocol error.
New source/test hashes are not relabelled as the preceding full-suite pass.
Focused tests cover both channel payloads, waiting/wakeup, buffered results,
actual worker failure, spool cleanup and BLAS restoration. Reproduction
uses unchanged default fit parameters to locate the underlying numerical
failure; no failed run can turn green by masking or loosening the certificate.

Root reproduced directly:2026-09-18 fits,2026-09-21 fails in an OOF fit with
`free_rms=1.4960563744338608e-6`, while the unchanged gate is1e-6.
This is a real numerical refusal, not startup overhead or a time limit.
The existing Riccati terminal candidate was reachable only after a positive-
ascent line search failed. The nonpositive-ascent ray branch threw before
that same candidate could be checked. A narrow trial routes this previously
failing branch through the ALREADY EXISTING full-certificate candidate;
accepted ray steps, cold reference, iteration budgets and all thresholds
remain identical. Keep this trial only if it resolves the real refusal
without breaking existing regressions; otherwise revert this math hunk.

The new dedicated channel-failure test command was tool-blocked, unrun.
No attempt is made to execute that same blocked test through another runner.
Direct fit reproduction is a separate numerical diagnostic, not its substitute.

The narrow terminal-candidate repair resolved the deterministic two-day
direct fit reproduction. Existing conditioned-EB tests passed145/145,
including the BLAS1/actual-BLAS6 frozen difficult fold, and existing timeblock
tests passed604/604. No tolerance, positive-ascent success branch, accepted
strict-rise step, alpha search or iteration budget changed.

The first repaired latest-window replay timed out before measurement.
An offline package-load command with768M heap hint also timed out; neither
is successful evidence and their time is not attributed wholly to JIT.
The subsequent package load at1024M completed in2s/640MiB. The repeated
latest-window command at DEFAULT GC then completed, compile/recompile0,
15.893177345s, and matched old portfolio/returns exactly.10 assertions pass.
This is correctness regression, not proof of an overall speedup.

Repaired earlier-window two-day warmup now succeeds. Its eight-day measured
call still hit the unchanged time guard, stopped in the numerical fixed-point
path (RC124,35s,1684MiB); it did not produce a complete window result.
No extra iterations, relaxed thresholds or larger command budget were used.
This remains a release blocker, not a successful generalization test.
