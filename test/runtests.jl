# Standard test entry — explicit admission, no silent skips.
#
#   julia --project=<pkg> test/runtests.jl                                  # full suite (default: no args)
#   julia --project=<pkg> test/runtests.jl --architecture-only             # M1 gate only, REAL src tree
#   julia --project=<pkg> test/runtests.jl --architecture-only --architecture-root PATH
#                                                                           # M1 gate only, against a fixture source root
#
# --architecture-only is a TEST-ENTRY-ONLY mode flag (not a production
# configuration): it keeps the same fail-closed admission
# (verify_required_contract_files, covering the architecture gate file
# itself) and runs the SAME M1 architecture gate (real-tree AST scan +
# runtime method metadata + known-bad fixture self-checks, via the same
# include of architecture_contract_tests.jl) while skipping the numeric
# suite — so DevOps can verify gate green/red within the <=60s scoped budget
# without running the full suite.
#
# --architecture-root PATH (valid ONLY together with --architecture-only)
# points the SAME tree-scan gate at a fixture source root, so the standard
# entry itself can be proven to exit non-zero (RC != 0) on a fake root with
# a planted forbidden edge (production->dev include via bare literal,
# joinpath(@__DIR__, ...), or qualified Base.include; oof_shadow kwarg; ...)
# and on an empty root, and green on a legal minimal root — without ever
# touching the real tree. Canonical on-disk fixture roots:
#   test/fixtures/arch_minimal_root   (expected green)
#   test/fixtures/arch_bad_root       (expected red: planted joinpath-dev,
#                                      qualified Base.include-dev, and
#                                      dynamic-variable include edges — the
#                                      exact forms the old walk silently
#                                      ignored)
#   test/fixtures/arch_nested_root    (expected green: nested subdirectory
#                                      includes whose ".." targets sit above
#                                      them; bare and @__DIR__ spellings
#                                      converge on one canonical key)
#
# Core-loading policy (static toplevel conditional; `if` introduces no
# scope in Julia, so `using` stays at legal toplevel load semantics):
#   - fixture-root mode: ONLY stdlib/Test is loaded — the floating
#     production module is never loaded, so the AST gate and the fail-closed
#     required admission cannot be blocked by production load state, and no
#     pkgimage/freeze equivalence is assumed;
#   - real-root --architecture-only: Test + KTrader are actually loaded, and
#     every runtime typed/method/namespace check executes against the real
#     module;
#   - default full mode: the full original loading set (unchanged).
# Runtime namespace checks (M1-D/E) apply only to the real module and are
# explicitly skipped in fixture-root mode (stated in output, never silent).
#
# Admission is EXACT: no args -> full; exactly --architecture-only; or
# exactly --architecture-only --architecture-root <existing dir>. Any other
# argument form — positional, mixed, duplicated, unknown, or
# --architecture-root without --architecture-only — is rejected immediately.
# (No pre-existing test-scope entry exists to reuse: bin/scoped_run.sh is a
# process resource guard, exercised by scoped_run_tests.jl — not a mode
# selector, so this is not a second check.)
function _parse_test_entry_args(args::Vector{String})
    isempty(args) && return (:full, nothing)
    args[1] == "--architecture-only" ||
        error("unknown test-entry argument(s) $(args): only --architecture-only (optionally followed by --architecture-root PATH) is accepted; no silent skips")
    length(args) == 1 && return (:architecture_only, nothing)
    if length(args) == 3 && args[2] == "--architecture-root"
        isdir(args[3]) ||
            error("--architecture-root must be an existing directory: $(args[3])")
        return (:architecture_only, args[3])
    end
    error("unknown test-entry argument combination $(args): expected exactly --architecture-only or --architecture-only --architecture-root PATH; no silent skips")
end

test_entry_mode, architecture_root_override = _parse_test_entry_args(ARGS)

if test_entry_mode == :architecture_only && architecture_root_override !== nothing
    # Fixture-root proof mode: stdlib/Test ONLY — never loads KTrader.
    using Test
elseif test_entry_mode == :architecture_only
    # Real-tree gate mode: actually load the real module (no freeze
    # equivalence assumed) so runtime typed/method/namespace checks run.
    using Test
    using KTrader
else
    using Test, Random, LinearAlgebra, Statistics, Dates, Convex, Clarabel
    using KTrader
end

# Fail-closed admission BEFORE any test executes (both modes): the delivered
# contract files must exist — including the architecture gate file itself —
# otherwise the entry errors out immediately (no silent isfile-skip). See
# test/contract_registry.jl for the registry + primitive.
include("contract_registry.jl")
verify_required_contract_files(@__DIR__;
                               files = (ARCHITECTURE_CONTRACT_TEST, REQUIRED_CONTRACT_TESTS...))

# M1 architecture contract gate (both modes): deterministic, mechanically
# checkable architecture invariants (AST include/export/identifier scan over
# the production include tree + runtime method metadata), plus the known-bad
# fixture self-checks that prove this gate can actually go red. The probe
# dev-migration HAS landed (src/ceiling_probes.jl deleted; no probes include
# or export in KTrader.jl), so the real tree is expected GREEN — any
# reported violation is a real regression. The include walk is fail-closed:
# only statically resolvable literal include forms are accepted (bare
# literal; literal-only joinpath/@__DIR__/@__FILE__/dirname chains; and the
# include(mod, path) / Base.include(mod_or_fn, path) forms); every other
# include shape — variable, interpolation, broadcast, foreign qualified
# callee, bare `include` as a value — is an explicit violation, never a
# silent skip. With --architecture-root PATH the same gate consumes the
# given fixture source root instead, loading no KTrader at all.
# See test/architecture_contract_tests.jl.
include("architecture_contract_tests.jl")

# Full numeric suite must run ALL reference-isolation phases: an
# inherited REFISO_STAGE=R2/R3 environment would silently skip the other
# phases' contracts (a false-green path). Reject partial environments here;
# the phase-split runner is for explicit single-file budgeted runs only.
# The standard full entry keeps refusing partial environments even while the
# wider suite is execution-paused: source-level admission stays exact.
if test_entry_mode == :full
    stage_env = get(ENV, "REFISO_STAGE", "ALL")
    stage_env == "ALL" ||
        error("full suite requires REFISO_STAGE=ALL (got $(stage_env)): partial phase environments must not silently skip reference-isolation contracts; unset it or set ALL")
    @testset "KTrader V1.0 Complete Rigorous Suite" begin
        include("v1_constitutional_tests.jl")
        include("numerical_tests.jl")
        include("data_tests.jl")
        include("execution_tests.jl")
        include("incremental_tests.jl")
        include("residual_oracle_tests.jl")
        # Delivered contract files: mandatory, verified above, included
        # unconditionally. conditioned_eb / relative_support / timeblock /
        # ceiling_probes were PROMOTED from the legacy optional isfile block
        # into the registry (see contract_registry.jl): they are delivered
        # and their behavior is live (ceiling_probes is rewired to the dev
        # namespace), so a missing file now fails closed at admission
        # instead of being silently skipped. The legacy optional block is
        # deleted — no isfile-guarded include remains in this entry.
        # The admission-integration deliveries (incremental_budget /
        # history_cache / artifact_replay contract tests) enter through this
        # SAME loop: each is included exactly once, here — no nested second
        # run from a parent file, no isfile skip. Until their owners land
        # them, admission above fails closed; that red during the delivery
        # window is the contract, not a regression to be worked around.
        for f in REQUIRED_CONTRACT_TESTS
            include(joinpath(@__DIR__, f))
        end
    end
end
