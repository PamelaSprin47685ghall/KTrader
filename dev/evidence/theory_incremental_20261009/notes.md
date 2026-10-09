# Incrementalization: theory-first investigation, 2026-10-09

User asks for the principal algorithmic problem, not another allocation
micro-optimization. Released HEAD fc54ce4 / v2.0.0 is not modified, committed
over or retagged. New files are development-only algebra/proof material.

The test script checks identities, derivatives and explicit counterexamples
with small deterministic matrices. It calls existing production cache
evaluation to compare the objective, not optimize_conditioned_eb or a full
fit/backtest. Original production/test files and numerical tolerances stay
unchanged. Research checks are not additions to the Final acceptance count.

Main claims being checked: masked metric transport has a commutator rank
bound; isotropic ridge is not preserved by a general congruence; fixed-alpha
negative evidence has a Sylvester base Hessian plus a rank-at-most106
moment correction; its Newton linear system reduces exactly to106 variables
when the base is invertible. This does not imply106 dimensions for the whole
alpha/floor-constrained/global-reference solve, or a quadratic-time daily
algorithm. The covariance floor and reference basin choice remain gates.

Numerical command budget: existing45s inner/55s outer/2048MiB RSS. No GPU,
long backtest or repeated fitting. Written proofs in the companion note
are the primary deliverable; numerical checks only catch algebra mistakes.

The first transport/eight-fit geometry audit completed (local_structure.log,
RC0,15s,1481MiB), but the expanded version adding support checks timed out
at a later ridge-spectrum computation (support_structure.log,RC124,35s,
1397MiB). It provides NO completed real support check. The next diagnostic
isolates only the first date's third fold, avoiding the already-completed
transport-rank work and seven unrelated spectral decompositions. Same
deadline/RSS guard; no posterior fit, changed tolerance or larger limit.

That isolated support audit completed8/8,RC0,10s,1104MiB. The observed
third-fold graph has r45/a19; original feature/target leakage is below
4e-13/4e-16 respectively. Null covariance equals the original floor block
to3.34e-16, the scalar null evidence gradient is negative, and the original
nonzero null coefficient mean matches its analytic trace-compensation term.
These identify an applicable structural face, not a new accepted solver.

The fixed-alpha Schur identity and its107-dimensional joint-block algebra
passed small deterministic checks. The joint test uses manufactured exact
mixed derivatives: it does not claim production alpha derivatives have been
implemented. The null-support test retains the original d, prior nullspace,
floor trace offset and nonzero conditional null mean in the full objective.
The toy floor value0.01 is only an algebra test parameter; actual model
audit uses the unchanged production1e-8. No production tolerance was edited.

Final research identity checks:710/710,RC0,19s,858MiB
(final_checks.log). One-fold real structure checks:8/8,RC0,10s,1104MiB
(support_face.log). Repeated intermediate checks are not summed into that
count. No optimize_conditioned_eb/fit/solve/backtest was launched.
The release verifier independently returnedSOURCE_VERIFIED for2.0.0 Final,
RC0,2s,347MiB (final_release_unchanged.log); it checks identity, not new
numerical acceptance. Existing production, tests, dependencies and tag remain
untouched. New uncommitted files are confined to the theory and evidence
directories. No GPU experiment, commit, reset, clean or version change.

Final process inspection saw an unrelated Julia fit already running outside
this work's scoped sessions. It was not started, stopped or modified here;
no claim is made that the entire machine has no Julia processes. All scoped
commands started in this investigation had returned before completion.
