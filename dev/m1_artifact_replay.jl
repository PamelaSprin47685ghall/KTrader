# dev/m1_artifact_replay.jl
#
# Dev-only, artifacts-only, EXPLICIT-PHASE replay checker for the M1 typed
# PreparedProblem single-instance comparison. NOT part of the production
# module, NOT wired into any test suite. The explicit phases' RUN scope and
# records are in the unique manifest + dev note; NOT a full-suite or
# throughput certification. Five narrow entry points over direct primitives;
# no generic phase framework.
#
#   julia --project=. dev/m1_artifact_replay.jl --phase check
#   julia --project=. dev/m1_artifact_replay.jl --phase rewrap
#   julia --project=. dev/m1_artifact_replay.jl --phase solve-once
#   julia --project=. dev/m1_artifact_replay.jl --phase model-only
#   julia --project=. dev/m1_artifact_replay.jl --phase residual-sample
#
# THE ONE manifest is dev/m1_replay_manifest.toml (DevOps-owned; this
# script never writes or requires a second inventory). Hash verification
# uses the Julia SHA stdlib (SHA.sha256 + bytes2hex) against the
# manifest's full 64-hex sha256 values and byte sizes. Model loading uses
# the Serialization stdlib API.
#
# COMPARISON GATES (REVISE, 2026-10-08): every numeric comparison in the
# model-only / residual-sample / solve-once phases goes through THE ONE
# comparison owner below - pure functions that ERROR on failure. A
# printed recorded number is context, never a pass. The previous
# println-only behavior (differences displayed; phases still completed
# RC0 on over-tolerance diffs, finite/NaN mask mismatches, or a
# NaN-poisoned maximum(abs.(MA .- MB)) after materializing the full T×N
# residual matrix) is closed in source. solve-once now compares the
# deterministic 9-row residual block via residual_rows (finite-only,
# NaN-position-gated, Inf rejected outright) and never materializes T×N.
# Gate constants distinguish HISTORICAL thresholds from the Manager's
# numerical-verification decisions of THIS assignment (see TOL_OF_RECORD
# below; the new decisions are labeled as such, never masquerading as
# historical assertions).
#
# PATH-OWNER FIX (2026-10-08, 4th-review accepted): PATHS_OF_RECORD is
# THE ONE path owner. Historically verify_artifacts() hash-checked
# /tmp/prep_inc_t14309.jls (the manifest-registered path) while
# rewrap_prep / run_solve_once deserialized an UNREGISTERED
# /tmp/prep_inc_t14309.jl and recorded that same unbound string as
# source_prep - the verification object and the consumed object were
# DIFFERENT files: no hash ever covered the actually-consumed input.
# Fix, within the existing manifest and phase structure: every
# consumption point obtains its path via path_of_record(key),
# deserializes ONLY through load_verified (fails closed unless
# verify_artifacts hash-verified that exact path in THIS process), and
# persisted provenance passes assert_source_of_record. No new manifest
# entry, no '.jl' added to any inventory, no old artifact rewritten;
# phases / refusal order / tol / seed unchanged. Evidence timeline
# discipline: the archived check/model-only/residual-sample runs
# consumed '.jls' paths that already matched PATHS_OF_RECORD, and the
# archived solve-once RC1 refusal fired BEFORE any deserialize - none
# of those records consumed '.jl'. The historical rewrap RC0 does NOT
# license inferring that '/tmp/prep_inc_t14309.jl' existed then or that
# its content matched the '.jls': nothing ever bound that file, and the
# timeline forbids retroactive inference. The already-persisted
# /tmp/model_solveonce_t14309.jl2s embeds the OLD unbound source_prep
# string as a historical record - not rewritten, not re-certified.
#
# CONSUMED-BYTES RE-CHECK (same-day follow-up): registration alone
# would only prove 'this NAME verified in the past', not 'the bytes
# being consumed NOW are the certified bytes' - the file could be
# replaced between verify and load (TOCTOU). load_verified therefore
# reads the file ONCE, re-checks sha256+size of exactly those read
# bytes against the registered record, and deserializes THAT SAME
# byte string via IOBuffer - no second read, no replacement window.
#
# Library-safe: this file may be include'd (e.g. from
# test/artifact_replay_contract_tests.jl) without running main and
# without loading KTrader - the toplevel using below fires only when run
# AS A SCRIPT with a non-check phase; the check phase stays KTrader-free
# and deserialization-free (original contract, unchanged).
#
# Phase responsibilities (stated exactly):
#   * check      - NO KTrader load, NO deserialization: manifest parse,
#                  per-file existence + sha256 + byte-size verification
#                  for the four paths of record, static schema name
#                  cross-check (this script's constants vs the manifest's
#                  old_prep_schema records).
#   * rewrap     - loads KTrader, deserializes the old 5-key wrapper ->
#                  22-key prep, compares key names at runtime, builds the
#                  typed PreparedProblem via the PRODUCTION keyword
#                  constructor. No solve.
#   * solve-once - manifest verification, then the no-overwrite refusal
#                  for its recorded output path BEFORE any
#                  deserialization / rewrap / solve (the expensive
#                  actions). The recorded output artifact already exists
#                  and is registered in the ONE manifest, so NO new fit
#                  is required or authorized: the verification of this
#                  phase is exactly that the command FAILS with the
#                  refusal before any expensive action. No new output
#                  path, no new manifest record. 7.449 s remains the
#                  recorded single-instance comparison target, NOT a
#                  speedup claim.
#   * model-only / residual-sample - load-only comparisons on existing
#                  models (no re-fit), the preserved /tmp/m1_canon.jl and
#                  /tmp/m1_rescheck.jl logic (seed=1, S=300, independent
#                  MersenneTwister(1) streams) under fail-loud gates.
#
# Panel caveat (from the manifest): old_panel_hash = "unknown". The
# artifact-only local comparisons remain valid on their own (each
# artifact's sha256 is verified), but NOTHING here certifies the
# provenance of the original input panel. Source hashes in the manifest
# are the CURRENT tree at manifest time, not the old-measurement
# sources (see the manifest's own caveat).
#
# Success is never assumed; green requires an authorized run.
#
# World-age note: the KTrader load happens at TOPLEVEL (before main)
# for every non-check phase, so all later method dispatches see the
# bindings in the current world age; no function-internal include and no
# invokelatest framework patch.

using TOML, SHA, Serialization, LinearAlgebra, Random

const MANIFEST_PATH = joinpath(@__DIR__, "m1_replay_manifest.toml")

# Four artifact paths of record (the manifest carries 13 + the solveonce
# entry in total).
const PATHS_OF_RECORD = (
    ("prep_inc_t14309_jls",    "/tmp/prep_inc_t14309.jls"),
    ("model_m1_t14309_jls",    "/tmp/model_m1_t14309.jls"),
    ("model_t14309_inc_jls",   "/tmp/model_t14309_inc.jls"),
    ("model_t14309_final_jls", "/tmp/model_t14309_final.jls"),
)

# ---- Single path owner: verified-consume boundary ------------------------------
#
# PATHS_OF_RECORD is the sole path authority: every path this script
# deserializes, or records as provenance, is obtained through
# path_of_record(key). Literal path strings at consumption points are
# forbidden - that is exactly the historical bug (verify checked
# /tmp/prep_inc_t14309.jls while rewrap/solve-once deserialized the
# unregistered /tmp/prep_inc_t14309.jl).
#
# VERIFIED_PATHS: process-local registry of authentication records.
# Value = (sha256, bytes) of the VERIFIED byte string. Sole writer:
# verify_and_register (reached from verify_artifacts in production,
# and from the contract tests on small temp fixtures). Sole readers:
# load_verified / assert_source_of_record. Registration authorizes a
# path for CONSUMPTION-WITH-RE-CHECK: load_verified re-verifies the
# exact bytes it is about to deserialize against this record, so a
# file replaced after verification never reaches the consumer. An
# unregistered lookalike fails closed BEFORE any deserialize attempt
# (file existence, well-formedness, or payload similarity are all
# irrelevant to the refusal).
const VERIFIED_PATHS = Dict{String,Tuple{String,Int}}()

function path_of_record(key::AbstractString)
    for (k, path) in PATHS_OF_RECORD
        k == key && return path
    end
    error("no artifact path of record for key '", key, "'")
end

# Verify ONE artifact record against the actual file (existence +
# full sha256 + byte size) and, on success, register the path as
# consumable. On any mismatch this errors and registers NOTHING.
function verify_and_register(path::AbstractString, rec)
    isfile(path) || error("artifact of record missing: ", path)
    digest = bytes2hex(SHA.sha256(read(path)))
    digest == rec["sha256"] ||
        error("HASH MISMATCH for ", path, ": manifest ", rec["sha256"],
              " vs actual ", digest)
    filesize(path) == rec["bytes"] ||
        error("SIZE MISMATCH for ", path, ": manifest ", rec["bytes"],
              " vs actual ", filesize(path))
    VERIFIED_PATHS[abspath(path)] = (rec["sha256"], rec["bytes"])
    return rec
end

"""Deserialize ONLY bytes that match the registered authentication
record. ONE read: the bytes re-verified here (sha256 + size against
VERIFIED_PATHS) are exactly the bytes handed to deserialize via
IOBuffer - no second read of the path, so a file replaced after
verification can never reach the consumer. Unregistered lookalikes
are refused before any read/parse; registered-but-changed files are
refused before the bytes are deserialized."""
function load_verified(path::AbstractString)
    canon = abspath(path)
    rec = get(VERIFIED_PATHS, canon, nothing)
    rec === nothing &&
        error("REFUSING to deserialize unverified path ", canon,
              " - not hash-verified in this process ",
              "(verify_artifacts must pass for it first)")
    sha_rec, bytes_rec = rec
    # Single read: these exact bytes are both re-verified and consumed.
    data = read(canon)
    digest = bytes2hex(SHA.sha256(data))
    (digest == sha_rec && length(data) == bytes_rec) ||
        error("CONSUMED-BYTES MISMATCH for ", canon,
              ": registered (sha256 ", sha_rec, ", ", bytes_rec,
              " bytes) vs actual (sha256 ", digest, ", ", length(data),
              " bytes) - the file changed after verification; the new ",
              "bytes are NOT handed to the consumer")
    return Serialization.deserialize(IOBuffer(data))
end

"""Source-binding gate for persisted provenance: the source string must
BE the path of record for the consumed key AND that path must be
hash-verified in this process. The historical unbound lookalike
('/tmp/prep_inc_t14309.jl') fails this gate by construction."""
function assert_source_of_record(source::AbstractString,
                                 key::AbstractString)
    bound = path_of_record(key)
    source == bound ||
        error("[gate] source binding '", source,
              "' is NOT the path of record '", bound, "' for key '", key,
              "' (lookalike/unbound path refused)")
    haskey(VERIFIED_PATHS, abspath(bound)) ||
        error("[gate] source path '", bound,
              "' not hash-verified in this process")
    return bound
end

# Static schema names (DevOps evidence; cross-checked against the
# manifest's [old_prep_schema]).
const WRAPPER_KEYS = ("prep", "fast", "prepare_seconds", "limit", "peak_rss_kb")
const PREP_KEYS_22 = (
    "active_idx", "N_universe", "adj_act", "T", "N", "r", "s1", "field", "m",
    "relative_embedding", "X_rel", "s_m", "s_perp", "B_m", "ts_total", "n_res",
    "n_bands", "P_features", "X_rel_stacked", "Y_target_rel", "stats",
    "macro_stats",
)

# Recorded single-instance expectations (dev/m1_typed_replay.md §2).
const EXPECTED = (
    solve_seconds_of_record = 7.449,
    residual_rows = [1, 2342, 4684, 4685, 7026, 9368, 9369, 11710, 14052],
)

# ---- Tolerances of record (the gate constants) -------------------------------
#
# HISTORICAL thresholds (user-stated early / original checker records;
# nothing here is reverse-engineered from observed numbers):
#   mu 1e-10 / Sigma 1e-9 / alpha 1e-8 (ABSOLUTE difference) / w_L1 1e-7
#     - user-stated early tolerances.
#   residual 1e-9 - finite-cell maxdiff of /tmp/m1_rescheck.jl
#     ("tol 1e-9") and the manifest's [checker_commands] m1_rescheck.
#   kelly_cert 1e-8 - src/kelly.jl PRODUCTION default certificate
#     tolerance (kelly_weights_v1 / fast_kelly_solver /
#     clarabel_kelly_solver all default tol=1e-8); feasibility, KKT
#     residual and objective gap are gated at exactly this production
#     value.
# MANAGER numerical-verification decisions (2026-10-08, THIS assignment)
# - items with NO historical threshold. Recorded as NEW gate decisions
# of this assignment, NOT masquerading as historical assertions:
#   g_c            - Sigma_rel / G_c_mean matrix differences at 1e-9,
#                    elementwise max |a-b| (the metric is stated in the
#                    gate output).
#   d_v            - d_posterior / v_forecasts absolute 1e-10.
#   x_canonical    - same-seed canonical scenario X max 1e-10.
#   objective_diff - cross-model objective difference 1e-8: a
#                    numerical-reproduction criterion for this recorded
#                    same-input comparison, NOT a general theory claim
#                    that two different probabilistic laws share
#                    objectives.
#   swap_roundoff  - swap-loss difference 1e-12 pure-roundoff allowance
#                    (NOT relaxed to observed values), applying only
#                    together with the production certificates passing
#                    (the certificate gates run first).
# Also Manager-decided: the relative predictive covariance compares
# L*L' (never a bare factor body) at Frobenius 1e-9; residual blocks
# require shape / finite / NaN positions exactly equal with finite max
# 1e-9 and Inf rejected outright (even at matching positions).
const TOL_OF_RECORD = (
    mu = 1e-10,
    sigma = 1e-9,
    alpha = 1e-8,
    wl1 = 1e-7,
    residual = 1e-9,
    kelly_cert = 1e-8,
    g_c = 1e-9,
    d_v = 1e-10,
    x_canonical = 1e-10,
    objective_diff = 1e-8,
    swap_roundoff = 1e-12,
)

# ---- THE ONE comparison owner ------------------------------------------------
#
# Pure functions, no KTrader dependency (the contract tests exercise them
# on small arrays without loading the package). Every gate ERRORS on
# failure; return values exist for reporting only. The same numeric check
# is never re-implemented per phase.

"""Shape + finite/NaN mask identity + finite-cell-only maxdiff gate.
This is the residual-block semantics: NaN cells are legitimate missing
observations whose POSITIONS must match exactly, while Inf entries are
rejected outright - even at matching positions (Inf is never a
legitimate residual value)."""
function assert_masked_maxdiff(name::AbstractString, a::AbstractArray,
                               b::AbstractArray; tol::Real)
    size(a) == size(b) ||
        error("[gate] ", name, ": shape mismatch ", size(a), " vs ", size(b))
    fa = isfinite.(a); fb = isfinite.(b)
    isequal(fa, fb) ||
        error("[gate] ", name, ": finite/NaN/Inf mask mismatch (finite ",
              count(fa), "/", length(a), " vs ", count(fb), "/", length(b),
              ")")
    (any(isinf, a) || any(isinf, b)) &&
        error("[gate] ", name, ": Inf entries rejected outright (",
              count(isinf, a), " in a, ", count(isinf, b), " in b)")
    md = 0.0
    for j in eachindex(a)
        fa[j] || continue
        d = abs(a[j] - b[j])
        d > md && (md = d)
    end
    md <= tol ||
        error("[gate] ", name, ": finite-cell maxdiff ", md, " > tol ", tol)
    return md
end

"""Shape + all-finite + full elementwise maxdiff gate (model fields and
the canonical scenario matrix: posterior outputs and gross returns must
be all finite; a NaN/Inf entry is itself a red)."""
function assert_matrix_close(name::AbstractString, a::AbstractArray,
                             b::AbstractArray; tol::Real)
    size(a) == size(b) ||
        error("[gate] ", name, ": shape mismatch ", size(a), " vs ", size(b))
    all(isfinite, a) && all(isfinite, b) ||
        error("[gate] ", name, ": non-finite entries (",
              count(isfinite, a), "/", length(a), " finite vs ",
              count(isfinite, b), "/", length(b), ")")
    md = maximum(abs.(a .- b))
    md <= tol || error("[gate] ", name, ": maxdiff ", md, " > tol ", tol)
    return md
end

"""Shape + all-finite + Frobenius-difference gate. Used for the relative
predictive covariance comparison L*L' (never a bare factor body); the
metric is Frobenius and is stated in the gate output."""
function assert_frob_close(name::AbstractString, a::AbstractArray,
                           b::AbstractArray; tol::Real)
    size(a) == size(b) ||
        error("[gate] ", name, ": shape mismatch ", size(a), " vs ", size(b))
    all(isfinite, a) && all(isfinite, b) ||
        error("[gate] ", name, ": non-finite entries (",
              count(isfinite, a), "/", length(a), " finite vs ",
              count(isfinite, b), "/", length(b), ")")
    fr = norm(a .- b)
    fr <= tol ||
        error("[gate] ", name, ": Frobenius diff ", fr, " > tol ", tol)
    return fr
end

function assert_scalar_absdiff(name::AbstractString, a, b; tol::Real)
    (isfinite(a) && isfinite(b)) ||
        error("[gate] ", name, ": non-finite scalar (", a, " vs ", b, ")")
    d = abs(a - b)
    d <= tol || error("[gate] ", name, ": absdiff ", d, " > tol ", tol)
    return d
end

"""Kelly certificate gate at the production certificate tolerance:
feasibility, KKT residual and objective gap must all pass (mirrors the
src/kelly.jl certified predicate)."""
function assert_certified(name::AbstractString, c; tol::Real)
    (c.feasibility <= tol && c.kkt_residual <= tol &&
     c.objective_gap <= tol) ||
        error("[gate] ", name, ": kelly certificate NOT certified at tol ",
              tol, ": feasibility=", c.feasibility, " kkt_residual=",
              c.kkt_residual, " objective_gap=", c.objective_gap)
    return true
end

"""Kelly wealth boundary: every scenario wealth must stay positive."""
function assert_wealth_positive(name::AbstractString, minwealth)
    (isfinite(minwealth) && minwealth > 0) ||
        error("[gate] ", name, ": non-positive minimum wealth ", minwealth)
    return true
end

"""Swap-loss concavity boundary: F(w_own) - F(w_other) >= 0 up to a
1e-12 pure-roundoff allowance (Manager decision 2026-10-08; NOT relaxed
to observed values). Only meaningful together with the production
certificates passing - the certificate gates run before this one."""
function assert_swap_nonneg(name::AbstractString, swap; tol::Real)
    isfinite(swap) || error("[gate] ", name, ": non-finite swap loss ", swap)
    swap >= -tol ||
        error("[gate] ", name, ": swap loss ", swap,
              " violates the concavity non-negativity boundary -", tol)
    return true
end

"""Deterministic per-fold first/mid/last row picks - the preserved
/tmp/m1_rescheck.jl selection. Derived from hash-verified frozen fold
ranges, so it must reproduce the recorded row set exactly."""
function fold_sample_rows(ranges)
    rows = Int[]
    for rg in ranges
        push!(rows, first(rg), rg[div(length(rg), 2)], last(rg))
    end
    return sort!(unique!(rows))
end

# -- manifest verification -----------------------------------------------------

function verify_artifacts()
    m = TOML.parsefile(MANIFEST_PATH)
    meta = get(m, "meta", Dict{String,Any}())
    panel = get(meta, "old_panel_hash", "unknown")
    println("[manifest] ", MANIFEST_PATH, " (recorded ",
            get(meta, "recorded", "?"), "; old_panel_hash=", panel,
            panel == "unknown" ?
            " - artifact-only comparisons stand, panel provenance NOT certified)" :
            ")")
    arts = m["artifacts"]
    for (key, path) in PATHS_OF_RECORD
        a = get(arts, key, nothing)
        a === nothing && error("manifest has no [artifacts] entry '", key, "'")
        # Single verification+registration owner: on success the path
        # becomes load_verified-consumable; on failure it never does.
        verify_and_register(path, a)
        println("[manifest] verified ", path, " sha256 ", a["sha256"],
                " bytes ", a["bytes"], " (", a["role"], ")")
    end
    return m
end

# -- rewrap --------------------------------------------------------------------

function rewrap_from(p)
    names = sort!(collect(string.(keys(p))))
    names == sort!(collect(PREP_KEYS_22)) ||
        error("prep schema mismatch at runtime: ", names)
    return KTrader.PreparedProblem(
        active_idx = p.active_idx, N_universe = p.N_universe,
        T = p.T, N = p.N, s1 = p.s1, s_m = p.s_m, s_perp = p.s_perp,
        adj_act = p.adj_act, r = p.r, m = p.m,
        relative_embedding = p.relative_embedding,
        observed = p.field.observed,   # mask lives at prep.field.observed
        X_rel = p.X_rel, B_m = p.B_m, ts_total = p.ts_total,
        n_res = p.n_res, P_features = p.P_features,
        stats = KTrader.to_fold_statistics(p.stats),
        macro_stats = KTrader.to_macro_statistics(p.macro_stats),
        X_rel_stacked = p.X_rel_stacked,
        F_folds = length(p.stats.ranges))
    # alive_now auto-filled by the constructor default (owned copy);
    # ws_owner=nothing, ws_generation=0 (owner None).
end

# Consumed through the single path owner + the verified-consume gate
# (historical bug: this line deserialized the UNREGISTERED
# /tmp/prep_inc_t14309.jl while verify checked the '.jls').
rewrap_prep() = rewrap_from(
    load_verified(path_of_record("prep_inc_t14309_jls")).prep)

# -- check: no KTrader, no deserialization ------------------------------------

function run_check()
    m = verify_artifacts()
    schema = m["old_prep_schema"]
    sort!(collect(string.(schema["wrapper_keys"]))) ==
        sort!(collect(WRAPPER_KEYS)) ||
        error("static wrapper-key records disagree (script vs manifest)")
    sort!(collect(string.(schema["prep_keys_22"]))) ==
        sort!(collect(PREP_KEYS_22)) ||
        error("static prep-key records disagree (script vs manifest)")
    println("[check] static schema cross-check PASS: wrapper 5 keys, prep 22 keys")
    println("[check] checker scripts of record (manifest [checker_commands]):")
    for k in ("m1_convert", "m1_canon", "m1_rescheck", "i0_fit")
        c = m["checker_commands"][k]
        println("  ", k, " -> ", c["path"], " sha256 ", c["sha256"])
    end
    println("[check] NO deserialization in this phase: model artifacts contain ",
            "KTrader types and need those bindings - that is the ",
            "rewrap/solve-once/model-only/residual-sample phases' responsibility.")
    println("[check] scope: hash verification covers the FOUR core input ",
            "artifacts of record (of the 13 in the manifest); the checker ",
            "script sha256 values above are DISPLAYED only, not verified - ",
            "no extra verification framework is added to reach the full list.")
    nothing
end

# -- rewrap --------------------------------------------------------------------

function run_rewrap()
    verify_artifacts()
    prep = rewrap_prep()
    println("[rewrap] typed ", typeof(prep).name, ": T=", prep.T,
            " N=", prep.N, " P=", prep.P_features, " F=", prep.F_folds,
            " alive_now=", count(prep.alive_now),
            " ws_owner=", prep.ws_owner === nothing ? "nothing" : "set",
            " gen=", prep.ws_generation)
    println("[rewrap] runtime key-name comparison PASS; no solve in this phase")
    nothing
end

# -- solve-once: refuse-before-fit; no new output, no new fit ------------------

const SOLVE_ONCE_FIRED = Ref(false)
const SOLVE_ONCE_OUT = "/tmp/model_solveonce_t14309.jl2s"

# -- the one model-field comparison (used by solve-once; not duplicated) ------

function compare_model_fields(tag, model, mb)
    ri, rb = model.resp, mb.resp
    println("[F] vs ", tag, ": alpha_m absdiff=",
            assert_scalar_absdiff("alpha_macro vs " * tag, ri.alpha_macro,
                                  rb.alpha_macro;
                                  tol = TOL_OF_RECORD.alpha),
            " alpha_rel absdiff=",
            assert_scalar_absdiff("alpha_rel vs " * tag, ri.alpha_rel,
                                  rb.alpha_rel; tol = TOL_OF_RECORD.alpha),
            " [GATED abs tol ", TOL_OF_RECORD.alpha, "]")
    println("[F] vs ", tag, ": G_c maxdiff=",
            assert_matrix_close("G_c_mean vs " * tag, ri.G_c_mean,
                                rb.G_c_mean; tol = TOL_OF_RECORD.g_c),
            " [GATED 1e-9 elementwise max - Manager decision 2026-10-08]")
    println("[F] vs ", tag, ": Sigma maxdiff=",
            assert_matrix_close("Sigma_rel vs " * tag, ri.Sigma_rel,
                                rb.Sigma_rel; tol = TOL_OF_RECORD.sigma),
            " [GATED 1e-9 elementwise max]")
    pm, pb = model.pred_moments, mb.pred_moments
    println("[F] vs ", tag, ": mu maxdiff=",
            assert_matrix_close("mu_pred vs " * tag, model.mu_pred,
                                mb.mu_pred; tol = TOL_OF_RECORD.mu),
            " mu_rel maxdiff=",
            assert_matrix_close("mu_rel vs " * tag, pm.mu_rel, pb.mu_rel;
                                tol = TOL_OF_RECORD.mu),
            " var_m absdiff=",
            assert_scalar_absdiff("var_m vs " * tag, pm.var_m, pb.var_m;
                                  tol = TOL_OF_RECORD.mu),
            " [GATED tol ", TOL_OF_RECORD.mu, "]")
    LL = pm.L_rel * pm.L_rel'
    LLb = pb.L_rel * pb.L_rel'
    println("[F] vs ", tag, ": L*L' Frobenius=",
            assert_frob_close(
                "L*L' (relative predictive covariance) vs " * tag, LL, LLb;
                tol = TOL_OF_RECORD.sigma),
            " [GATED 1e-9 Frobenius - compares L*L', never the bare factor]")
    println("[F] vs ", tag, ": d_posterior maxdiff=",
            assert_matrix_close("d_posterior vs " * tag, model.d_posterior,
                                mb.d_posterior; tol = TOL_OF_RECORD.d_v),
            " v_forecasts maxdiff=",
            assert_matrix_close("v_forecasts vs " * tag, model.v_forecasts,
                                mb.v_forecasts; tol = TOL_OF_RECORD.d_v),
            " [GATED 1e-10 abs - Manager decision 2026-10-08]")
    nothing
end

function run_solve_once()
    verify_artifacts()
    # No-overwrite refusal BEFORE any deserialization / rewrap / solve (the
    # expensive actions). The recorded output artifact ALREADY EXISTS and is
    # registered in the ONE manifest, so no new fit is required or
    # authorized: the verification of this phase is exactly that the
    # command FAILS here, before any expensive action. No new output path,
    # no new manifest record - never a silent candidate switch.
    isfile(SOLVE_ONCE_OUT) &&
        error("refusing to overwrite existing artifact ", SOLVE_ONCE_OUT,
              " (registered in the ONE manifest); the recorded output ",
              "already exists, so this phase verifies the refusal-before-",
              "fit behavior ONLY - no new fit, no new output path, no ",
              "manifest change")
    SOLVE_ONCE_FIRED[] &&
        error("solve-once already ran in this process - at most ONE cold ",
              "solve per process; restart for another authorized run")
    src = path_of_record("prep_inc_t14309_jls")
    p_old = load_verified(src).prep
    rows = fold_sample_rows(p_old.stats.ranges)
    prep = rewrap_from(p_old)
    t0 = time()
    model = KTrader.solve(prep)   # cold, DEFAULT EB, no warmup, no repeats
    SOLVE_ONCE_FIRED[] = true
    println("[solve-once] elapsed=", round(time() - t0, digits=3),
            " s (record: ", EXPECTED.solve_seconds_of_record,
            " s - comparison target, NOT a speedup claim)")
    # Persist THIS run's model so later no-fit phases compare the ACTUAL new
    # fit, not a lost in-memory object or a stale older artifact. Payload
    # unchanged from the recorded run; never overwrites (refused above).
    # Persisted provenance is bound to the verified path of record via
    # the source-binding gate BEFORE serialization. (The already-persisted
    # artifact embeds the OLD unbound '/tmp/prep_inc_t14309.jl' string as
    # a historical record; it is not rewritten.)
    assert_source_of_record(src, "prep_inc_t14309_jls")
    open(SOLVE_ONCE_OUT, "w") do io
        Serialization.serialize(io,
            (; model, solve_seconds = time() - t0, mode = "cold-default-EB",
               source_prep = src,
               note = "solve-once single cold fit, current source tree"))
    end
    println("[solve-once] persisted ", SOLVE_ONCE_OUT)
    for (nm, key) in (("inc", "model_t14309_inc_jls"),
                      ("batch", "model_t14309_final_jls"))
        mb = load_verified(path_of_record(key)).model
        compare_model_fields(nm, model, mb)
        # Deterministic 9-row residual block (same gate as residual-sample);
        # the full T×N residual matrix is NEVER materialized here.
        ra = KTrader.residual_rows(model.res_history, rows)
        rb = KTrader.residual_rows(mb.res_history, rows)
        md = assert_masked_maxdiff("residual_rows[solve-once vs " * nm * "]",
                                   ra, rb; tol = TOL_OF_RECORD.residual)
        println("[R] solve-once vs ", nm, ": 9-row block ", size(ra),
                " finite cells ", count(isfinite.(ra)), "/", length(ra),
                " absdiff max=", md, " [GATED tol ",
                TOL_OF_RECORD.residual, "; Inf rejected]")
    end
    println("[F] references (§2, context only - pass/fail is decided by ",
            "the gates above): vs inc exact 0-diff on listed fields; vs ",
            "batch alpha_rel rel 8.53e-14, Gc 2.47e-13, Sigma 2.82e-13, ",
            "mu 1.13e-14, cov 6.13e-15")
    nothing
end

# -- model-only: canonical same-seed comparison, no re-fit ---------------------

function run_model_only()
    verify_artifacts()
    m_m1 = load_verified(path_of_record("model_m1_t14309_jls")).model
    # Strong model-field gates on the load-only path (Manager decision
    # 2026-10-08): the same compare_model_fields previously reachable only
    # from the solve-once success path must also run here against BOTH
    # preserved baselines. No new fit/prepare/model; tol/seed unchanged.
    for (nm, key) in (("inc", "model_t14309_inc_jls"),
                      ("batch", "model_t14309_final_jls"))
        compare_model_fields(nm, m_m1,
                             load_verified(path_of_record(key)).model)
    end
    m_old = load_verified(path_of_record("model_t14309_final_jls")).model
    # Exactly the preserved /tmp/m1_canon.jl flow: S=300, seed 1, two
    # independent MersenneTwister(1) streams. Seed/S/repeats unchanged.
    X1 = KTrader.generate_scenarios_v1(m_m1; S = 300, rng = MersenneTwister(1))
    X2 = KTrader.generate_scenarios_v1(m_old; S = 300, rng = MersenneTwister(1))
    xmd = assert_matrix_close("canonical X (seed 1, S=300) vs old", X1, X2;
                              tol = TOL_OF_RECORD.x_canonical)
    println("[S] X maxdiff = ", xmd, "  [GATED 1e-10 max - Manager decision ",
            "2026-10-08; recorded 9.77e-14]")
    tr = trues(Int(m_m1.N_universe))
    w1 = KTrader.scenario_weights(X1, m_m1.active_indices, tr)
    w2 = KTrader.scenario_weights(X2, m_old.active_indices, tr)
    wl1 = sum(abs.(w1 .- w2))
    (isfinite(wl1) && wl1 <= TOL_OF_RECORD.wl1) ||
        error("[gate] w_L1 ", wl1, " > tol ", TOL_OF_RECORD.wl1,
              " (or non-finite)")
    println("[S] w_L1 = ", wl1, "  [GATED tol ", TOL_OF_RECORD.wl1,
            "; recorded 2.31e-12]  sum(w) m1=", sum(w1), " old=", sum(w2))
    c1 = KTrader.kelly_certificate(X1, w1)
    c2 = KTrader.kelly_certificate(X2, w2)
    assert_certified("cert m1", c1; tol = TOL_OF_RECORD.kelly_cert)
    assert_certified("cert old", c2; tol = TOL_OF_RECORD.kelly_cert)
    println("[S] cert m1: feas=", c1.feasibility, " kkt=", c1.kkt_residual,
            " gap=", c1.objective_gap, " obj=", c1.objective,
            "  [GATED production tol ", TOL_OF_RECORD.kelly_cert, "; ",
            "recorded kkt~1e-10, feas 2.2e-16, gap 6.29e-9]")
    println("[S] cert old: feas=", c2.feasibility, " kkt=", c2.kkt_residual,
            " gap=", c2.objective_gap, " obj=", c2.objective)
    Fobj(w, X) = sum(log, X * w) / size(X, 1)
    assert_wealth_positive("min wealth m1", minimum(X1 * w1))
    assert_wealth_positive("min wealth old", minimum(X2 * w2))
    println("[S] minw m1=", minimum(X1 * w1), " old=", minimum(X2 * w2),
            "  [GATED positive; recorded ~0.804]")
    swap1 = Fobj(w1, X1) - Fobj(w2, X1)
    swap2 = Fobj(w2, X2) - Fobj(w1, X2)
    # Swap gate runs AFTER the certificate gates: the 1e-12 roundoff
    # allowance applies only together with the production certificates.
    assert_swap_nonneg("swap on X1", swap1; tol = TOL_OF_RECORD.swap_roundoff)
    assert_swap_nonneg("swap on X2", swap2; tol = TOL_OF_RECORD.swap_roundoff)
    println("[S] swap-loss on X1 = ", swap1, "  on X2 = ", swap2,
            "  [GATED concavity boundary, 1e-12 pure roundoff - Manager ",
            "decision 2026-10-08, NOT relaxed to observed values; ",
            "production certificates gated above; recorded ",
            "floating-point zero]")
    objdiff = abs(Fobj(w1, X1) - Fobj(w2, X2))
    (isfinite(objdiff) && objdiff <= TOL_OF_RECORD.objective_diff) ||
        error("[gate] cross-model objective diff ", objdiff, " > tol ",
              TOL_OF_RECORD.objective_diff,
              " [Manager decision 1e-8: numerical-reproduction criterion ",
              "for this recorded same-input comparison, NOT a general ",
              "claim that two different probabilistic laws share ",
              "objectives]")
    println("[S] objective diff = ", objdiff,
            "  [GATED 1e-8 numerical-reproduction criterion - Manager ",
            "decision 2026-10-08; recorded 0.0]")
    nothing
end

# -- residual-sample: 9-row block, no re-fit ------------------------------------

function run_residual_sample()
    verify_artifacts()
    m_m1 = load_verified(path_of_record("model_m1_t14309_jls")).model
    ranges = load_verified(
        path_of_record("prep_inc_t14309_jls")).prep.stats.ranges
    rows = fold_sample_rows(ranges)
    rows == EXPECTED.residual_rows ||
        error("[gate] deterministic row set ", rows,
              " disagrees with the recorded set ", EXPECTED.residual_rows,
              " (derived from hash-verified frozen fold ranges)")
    println("[R] deterministic rows: ", rows,
            "  (recorded: ", EXPECTED.residual_rows, ")")
    for (nm, key) in (("inc", "model_t14309_inc_jls"),
                      ("batch", "model_t14309_final_jls"))
        B = load_verified(path_of_record(key)).model
        ra = KTrader.residual_rows(m_m1.res_history, rows)
        rb = KTrader.residual_rows(B.res_history, rows)
        md = assert_masked_maxdiff("residual_rows vs " * nm, ra, rb;
                                   tol = TOL_OF_RECORD.residual)
        println("[R] vs ", nm, ": block ", size(ra), " finite cells ",
                count(isfinite.(ra)), "/", length(ra), " absdiff max=", md,
                " [GATED tol ", TOL_OF_RECORD.residual,
                "; shape/finite/NaN exactly equal; Inf rejected]")
        println("[R] vs ", nm, ": NaN positions gated identical ",
                "(isequal semantics, inside the mask gate)")
    end
    println("[R] SAMPLE-ROW fidelity only - never a full T×N cell claim")
    nothing
end

# -- CLI: one explicit phase per run --------------------------------------------

function main(args)
    phase = nothing
    i = 1
    while i <= length(args)
        if args[i] == "--phase" && i < length(args)
            phase = args[i + 1]; i += 2
        else
            error("unknown argument: ", args[i])
        end
    end
    phase === nothing &&
        error("no --phase given (check|rewrap|solve-once|model-only|residual-sample)")
    if phase == "check"
        run_check()
    elseif phase == "rewrap"
        run_rewrap()
    elseif phase == "solve-once"
        run_solve_once()
    elseif phase == "model-only"
        run_model_only()
    elseif phase == "residual-sample"
        run_residual_sample()
    else
        error("unknown phase '", phase, "'")
    end
    println("phase '", phase, "' completed - report the run honestly; ",
            "green requires an authorized run, never an assumption.")
end

# Toplevel conditional load BEFORE main: the four non-check phases need
# the KTrader bindings in the CURRENT world age. A function-internal
# include would leave later method calls dispatched in the pre-include
# world age; no invokelatest patch. The check phase stays KTrader-free
# and deserialization-free. When this file is include'd as a LIBRARY
# (e.g. from the contract tests), ARGS is empty, so no KTrader load
# happens either - the comparison gates are pure functions.
let phase_arg = nothing
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--phase" && i < length(ARGS)
            phase_arg = ARGS[i + 1]
        end
        i += 1
    end
    if phase_arg !== nothing && phase_arg != "check"
        using KTrader   # package PkgId/UUID registration: deserialize resolves
                       # serialized module identities through the PACKAGE map;
                       # a bare include() creates a Main-anonymous module whose
                       # identity does not match, causing KeyError PkgId. The
                       # old artifacts were serialized under the package UUID.
    end
end

# Library-safe entry: main runs ONLY when this file is the script target
# (running dev/m1_artifact_replay.jl directly). Including the file (the
# contract tests) does NOT run main and does NOT load KTrader.
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
