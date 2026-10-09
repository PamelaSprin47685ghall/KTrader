# Contract test registry — the standard entry's fail-closed admission list.
#
# These files were delivered and certified in this mission (jcore / OOF
# diagnostics / contraction kernel / timed passthrough / solve-stage shadow
# forward / principal-root / model-API micro / ceiling admission / scoped-run
# guard / incremental primal prep). The standard test entry REFUSES to run
# if any of them is missing: a missing contract file must fail loudly before
# any numerical test executes, never be silently skipped by an isfile guard
# (the same silent-skip defect class previously hid a filename typo).
#
# verify_required_contract_files(dir; files) is the real primitive used by
# test/runtests.jl; contract tests and known-bad probes call the same function.
# M1 wiring notes — this registry reflects the REAL in-tree state and is
# never forced green:
#   - "prepared_problem_contract_tests.jl": delivered in-tree by the
#     PreparedProblem owner (typed PreparedProblem/FoldStatistics/
#     MacroStatistics/MacroFoldBlock; prepare_reference/prepare_incremental/
#     solve exported; the old fit path only forwards to the typed solve).
#     Required, as assigned.
#   - "oof_diagnostics_contract_tests.jl": REMOVED from required — the file
#     was deleted from the tree (it tested only the abolished OOF
#     diagnostics). If it returns under dev/, it becomes dev-optional, not
#     production-required.
#   - "solve_stage_shadow_contract_tests.jl": REMOVED from required — it
#     exercises the abolished OOF shadow machinery (KTrader.OOFCounters /
#     OOFShadowConfigError / oof_shadow kwarg), now deleted from production
#     response/predict; keeping it required would only surface
#     UndefVarError, not a meaningful gate. The file stays in-tree
#     untouched; migrating it to dev/ and rewiring it against the real dev
#     namespace (preserving its math-oracle assertions) belongs to the
#     Posterior follow-up package. This is an explicit registry decision,
#     not an isfile silent skip.
#   - "timed_passthrough_tests.jl": PERMANENTLY required — it guards the
#     ordinary timed/sink behavior that stays in production; a bare-return
#     regression there blocks real math output. It must NOT be demoted to
#     dev-optional when probes migrate.
#   - "ceiling_admission_tests.jl": stays REQUIRED — the probe behavior it
#     tests continues to exist in dev; the Posterior follow-up package
#     rewires its consumers to the explicit dev namespace and it keeps
#     executing in the standard entry (no dev-optional silent skip).
#   - test-side "ceiling_probes.jl": the probe dev-migration HAS landed; the
#     file is already rewired to the dev namespace (it includes
#     dev/probes.jl and does `using .DevProbes`; core math stays KTrader).
#     Its collector-accounting behavior is live in dev/, so it is REQUIRED —
#     a missing file fails loudly instead of being skipped by the old legacy
#     optional isfile block. The math gates (posterior_contract,
#     residual_oracle*, conditioned jcore/contraction kernel, principal
#     root, incremental primal prep, scoped run) stay required
#     unconditionally.
#   - Behavior-preserving suites PROMOTED from the legacy optional isfile
#     block in runtests.jl to REQUIRED (all four are delivered in-tree and
#     the behavior they test is live; an isfile guard would silently skip a
#     missing file — the exact defect class this registry exists to close;
#     promotion is the opposite of deleting tests to go green):
#       "conditioned_eb_tests.jl"    — conditioned EB certificate math
#                                      (live production solver behavior);
#       "relative_support_tests.jl"  — relative-support witness / degenerate
#                                      conditioned-EB gate math (live);
#       "timeblock_tests.jl"         — timeblock backtest equivalence
#                                      (live production scheduler behavior);
#       "ceiling_probes.jl"          — probe collector accounting, already
#                                      rewired to the dev namespace.
#   - "prepared_lifecycle_tests.jl": delivered in-tree by the Prepared
#     lifecycle owner (ReviewMath) and registered on that delivery report;
#     it sits in the standard entry's single-include sequence immediately
#     after prepared_problem_contract_tests.jl. The owner was instructed to
#     remove the parent file's trailing include of it (preventing a double
#     run); admission here stays fail-closed on the delivered file. The
#     core src field alive_now::Vector{Bool} it exercises is production
#     data semantics, not diagnostics — it is not on any forbidden list and
#     the AST architecture gate does not touch it.
#   - "cli_boundary_contract_tests.jl": delivered in-tree by ReviewEngine —
#     pure parsing + entry-source witness assertions at the supported
#     bin/ceiling_probes.jl entrance (explicit dev namespace load, no
#     command spawned, no second solver). Registered on the delivery report,
#     in the single-include sequence after prepared_lifecycle_tests.jl.
#     Its RUNTIME entry behavior still awaits DevOps's real CLI check: the
#     text witnesses prove source structure only and are NOT a record of an
#     executed CLI. Admission is fail-closed on the delivered file.
#   - Three admission-integration deliveries (independent src/dev owners,
#     REVISE'd assignment; each file has exactly one writer):
#       "incremental_budget_contract_tests.jl" — incremental resource-budget
#            routing contract (a budget may only choose the route, never
#            change any output);
#       "history_cache_contract_tests.jl" — history cache contract;
#       "artifact_replay_contract_tests.jl" — artifact replay contract.
#     They are REQUIRED from this registry change onward. During the
#     delivery window a file may not exist in-tree yet: the standard entry
#     then errors at admission (fail-closed red) — that is the intended
#     contract, and these entries must NEVER be demoted to optional or
#     hidden behind an isfile skip while pending. The single include of
#     each is the runtests.jl REQUIRED loop; delivering owners must not
#     add a trailing parent-file include (the double-run defect class
#     already closed for prepared_lifecycle_tests.jl).
#   - Only tests of genuinely deleted features (the abolished OOF
#     factory/counter machinery) leave required. When the Posterior
#     follow-up rewires the solve-stage shadow file against real dev
#     surfaces, the registry references the actual delivered file, and the
#     warmup-precondition rejection / observer-exception-preserves-failure /
#     model-passthrough gates keep their strength — none is lowered by the
#     rewiring.

# The architecture gate file itself is under fail-closed admission too: the
# standard entry refuses to run if it is missing. runtests.jl include()s it
# BEFORE the numeric suite (and as the whole run in --architecture-only
# mode), so it is verified through the combined file list rather than the
# REQUIRED_CONTRACT_TESTS include loop (which would double-include it).
const ARCHITECTURE_CONTRACT_TEST = "architecture_contract_tests.jl"

const REQUIRED_CONTRACT_TESTS = (
    "conditioned_jcore_tests.jl",
    "conditioned_deadwork_tests.jl",
    "fit_local_reuse_tests.jl",
    "conditioned_geometry_reuse_tests.jl",
    "conditioned_lazy_gradient_tests.jl",
    "scalar_tensor_tests.jl",
    "conditioned_certificate_reuse_tests.jl",
    "conditioned_certificate_cache_tests.jl",
    "conditioned_storage_tests.jl",
    "conditioned_contraction_kernel_tests.jl",
    "timed_passthrough_tests.jl",
    "principal_root_contract_tests.jl",
    "model_api_micro_tests.jl",
    "ceiling_admission_tests.jl",
    "scoped_run_tests.jl",
    "incremental_primal_prep_tests.jl",
    "incremental_pending_tests.jl",
    "kelly_packing_tests.jl",
    "backtest_target_tests.jl",
    "backtest_cache_tests.jl",
    "backtest_failure_tests.jl",
    "prepared_problem_contract_tests.jl",
    "prepared_lifecycle_tests.jl",
    "cli_boundary_contract_tests.jl",
    "posterior_contract_tests.jl",
    "residual_oracle_flow_tests.jl",
    "reference_isolation_tests.jl",
    # Promoted from the runtests.jl legacy optional isfile block (see notes
    # above): delivered and behavior-preserving, therefore fail-closed.
    "conditioned_eb_tests.jl",
    "relative_support_tests.jl",
    "timeblock_tests.jl",
    "ceiling_probes.jl",
    # Admission-integration deliveries (independent src/dev owners; see
    # notes above): REQUIRED immediately and fail-closed — a file not yet
    # written must make the standard entry error at admission, never be
    # isfile-skipped or demoted to optional during the delivery window.
    "incremental_budget_contract_tests.jl",
    "history_cache_contract_tests.jl",
    "artifact_replay_contract_tests.jl",
)

function verify_required_contract_files(dir; files = REQUIRED_CONTRACT_TESTS)
    for f in files
        isfile(joinpath(dir, f)) ||
            error("required contract test missing: $(repr(f)) (fail-closed: the standard test entry refuses to run without it; an isfile guard would have silently skipped this)")
    end
    true
end
