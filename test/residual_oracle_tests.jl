using Test, Random, LinearAlgebra, Statistics, Distributions
using KTrader

# The lazy ResidualOracle is now the production path (wired in KTrader.jl).
# The dense old formula survives ONLY as the explicit test reference
# `dense_oof_residuals`; every expectation below is built from that reference
# (or from the production path being compared AGAINST it), never from the new
# method grading itself.

const RO = KTrader

"Replicates _fit_prepared_v1's OOF fold refits from the same prep (same warm
start (1.0, 1.0), same full-minus-fold Grams), yielding oracle inputs."
function refit_fold_models(prep; F_folds=3, ridge_alpha=nothing)
    stats = prep.stats
    N = prep.N
    fold_models = ResponseOperator[]
    for fold in 1:F_folds
        fold_eval = stats.ranges[fold]
        S_xx_train = stats.xx === nothing ? nothing : stats.full_xx - stats.xx[fold]
        S_xy_train = stats.full_xy - stats.xy[fold]
        S_yy_train = stats.full_yy - stats.yy[fold]
        train_indices = [1:first(fold_eval)-1; last(fold_eval)+1:prep.n_res]
        train_ts = prep.ts_total[train_indices]
        train_design = length(train_indices) < prep.P_features ?
                        prep.X_rel_stacked[train_indices, :] : nothing
        resp_oof = fit_response_operator(prep.B_m, fill(Float64[], N), prep.m,
                                          prep.relative_embedding; ridge_alpha,
                                          ts = train_ts,
                                          S_xx_rel = S_xx_train, S_xy_rel = S_xy_train,
                                          S_yy_rel = S_yy_train, X_design = train_design,
                                          alpha_initial = (1.0, 1.0), need_uncertainty = false)
        push!(fold_models, resp_oof)
    end
    fold_models
end

"Old-formula scalar macro residual series, computed FROM the dense reference
matrix exactly as the pre-lazy predict.jl did."
function dense_macro_series(res)
    n, N = size(res)
    e = zeros(Float64, n)
    for idx in 1:n
        total = 0.0
        count = 0
        for j in 1:N
            v = res[idx, j]
            if isfinite(v)
                total += v
                count += 1
            end
        end
        e[idx] = count > 0 ? total / sqrt(count) : 0.0
    end
    e
end

"Old-formula scenario generation: replicates generate_scenarios_v1 exactly
(same rng consumption order, shared rows, own fallback, fractional scaling),
reading the DENSE truth matrix. The production generator passes L_rel
through principal_sqrt_root (the unique symmetric PSD principal square
root; Manager numerical-contract decision 2026-10-07) before multiplying z,
so the replica uses the same factor: an equal covariance under a different
factor still diverges draw-by-draw at fixed seed."
function old_scenario_draws(model, truth::AbstractMatrix{Float64}; S, rng)
    N = length(model.active_indices)
    T = size(truth, 1)
    z = Matrix{Float64}(undef, N, S)
    macro_draws = Vector{Float64}(undef, S)
    uniforms = rand(rng, S, N + 2)
    normal = Normal()
    randn!(rng, z)
    randn!(rng, macro_draws)
    relative = RO.principal_sqrt_root(model.pred_moments.L_rel) * z
    relative .+= model.pred_moments.mu_rel
    cumulative = cumsum(model.d_posterior)
    gross = fill(NaN, S, model.N_universe)
    for s in 1:S
        shift = mean(view(relative, :, s))
        mu_m = model.pred_moments.mu_m + sqrt(model.pred_moments.var_m) * macro_draws[s]
        ud = uniforms[s, 1]
        ur = uniforms[s, 2]
        d = min(searchsortedfirst(cumulative, ud), length(cumulative))
        scale = sqrt(model.v_forecasts[d] / model.v_bootstrap)
        row = min(floor(Int, ur * T) + 1, T)
        for j in 1:N
            residual = truth[row, j]
            if !isfinite(residual)
                own = model.own_res_rows[j]
                isempty(own) && error("active asset has no observed OOF residual")
                u = uniforms[s, j + 2]
                residual = truth[own[min(floor(Int, u * length(own)) + 1, length(own))], j]
            end
            loggross = (mu_m * model.e0[j] + relative[j, s] - shift) * model.s1[j] + residual * scale
            gross[s, model.active_indices[j]] = exp(loggross)
        end
    end
    gross
end

function dense_reference(P; F_folds = 3)
    model = fit_v1(P; F_folds)                          # production lazy path
    prep = KTrader._prepare_v1(P; F_folds)
    fold_models = refit_fold_models(prep; F_folds)
    oracle = RO.ResidualOracle(fold_models, prep.stats.ranges, prep.ts_total,
                               prep.r, prep.s1, prep.s_perp, prep.B_m, prep.X_rel)
    truth = RO.dense_oof_residuals(prep, fold_models)   # dense old-formula truth
    (; model, prep, fold_models, oracle, truth)
end

@testset "lazy residual oracle (production wiring)" begin
    # ---------- synchronous panel ----------
    rng = MersenneTwister(2027)
    T, N = 600, 3
    P = exp.(cumsum(0.01 .* randn(rng, T, N); dims = 1))
    ref = dense_reference(P)
    (; model, prep, fold_models, oracle, truth) = ref
    n_res = prep.n_res
    @test n_res == T - 1 - WARMUP == 343

    mo = RO.materialize(oracle)
    @test mo ≈ truth atol = 1e-10 rtol = 1e-8
    @test RO.materialize(model.res_history) ≈ truth atol = 1e-10 rtol = 1e-8
    @test RO.macro_residual_series(oracle) ≈ dense_macro_series(truth) atol = 1e-12 rtol = 1e-10
    own_truth = [findall(isfinite, view(truth, :, j)) for j in 1:N]
    @test RO.own_residual_rows(oracle) == own_truth
    @test model.own_res_rows == own_truth

    # fold boundaries (F=3, n_res=343 -> [1:114, 115:228, 229:343])
    @test RO.fold_of(oracle, 1) == 1
    @test RO.fold_of(oracle, 114) == 1
    @test RO.fold_of(oracle, 115) == 2
    @test RO.fold_of(oracle, 228) == 2
    @test RO.fold_of(oracle, 229) == 3
    @test RO.fold_of(oracle, 343) == 3
    @test RO.fold_of(oracle, 342) == 3

    # AbstractMatrix surface: size, column slicing, boundary single cells.
    @test size(oracle) == (n_res, N)
    perm = [3, 1, 2]
    @test oracle[:, perm] ≈ truth[:, perm] atol = 1e-10 rtol = 1e-8
    for (idx, j) in ((1, 1), (114, 3), (115, 1), (228, 2), (229, 3), (342, 1))
        @test oracle[idx, j] ≈ truth[idx, j] atol = 1e-10 rtol = 1e-8
    end

    # ---------- fit -> scenario -> Kelly, fixed seed, new vs old formula ----------
    Xnew = generate_scenarios_v1(model; S = 128, rng = MersenneTwister(77))
    Xold = old_scenario_draws(model, truth; S = 128, rng = MersenneTwister(77))
    @test isnan.(Xnew) == isnan.(Xold)
    msk = .!isnan.(Xnew)
    @test Xnew[msk] ≈ Xold[msk] atol = 1e-12 rtol = 1e-10
    tradable = trues(size(P, 2))
    wnew = KTrader.scenario_weights(Xnew, model.active_indices, tradable)
    wold = KTrader.scenario_weights(Xold, model.active_indices, tradable)
    @test wnew ≈ wold atol = 1e-9 rtol = 1e-9
    # Full production entry point against the old-formula chain.
    wpk = path_kelly_v1(P; S = 128, rng = MersenneTwister(77))
    @test wpk ≈ wold atol = 1e-9 rtol = 1e-9

    # ---------- cold/hot replay: same seed, same model, bit-identical ----------
    @test isequal(generate_scenarios_v1(model; S = 64, rng = MersenneTwister(14)),
                  generate_scenarios_v1(model; S = 64, rng = MersenneTwister(14)))
    @test isequal(RO.materialize(oracle), mo)
    rows = [5, 100, 100, 229, 1, 342]
    batch = RO.residual_rows(oracle, rows)
    @test isequal(RO.residual_rows(oracle, rows), batch)
    for idx in (1, 114, 115, 228, 229, 342)
        @test RO.residual_rows(oracle, [idx])[1, :] ≈ RO.residual_row(oracle, idx) atol = 1e-12 rtol = 1e-12
    end

    # ---------- ragged panel: masks, partial rows, a fully missing row ----------
    Pr = Matrix{Float64}(P)
    Pr[1:300, 3] .= NaN
    Pr[400:403, 2] .= NaN
    Pr[500, :] .= NaN
    refr = dense_reference(Pr)
    mor = RO.materialize(refr.oracle)
    @test isfinite.(mor) == isfinite.(refr.truth)
    mskr = isfinite.(mor)
    if any(mskr)
        @test mor[mskr] ≈ refr.truth[mskr] atol = 1e-10 rtol = 1e-8
    end
    # the production model's own oracle (model.res_history) also aligns with
    # the dense truth (mask-exact, values at the original tolerance).
    mtr = RO.materialize(refr.model.res_history)
    @test isfinite.(mtr) == isfinite.(refr.truth)
    mmask = isfinite.(mtr)
    if any(mmask)
        @test mtr[mmask] ≈ refr.truth[mmask] atol = 1e-10 rtol = 1e-8
    end
    er = RO.macro_residual_series(refr.oracle)
    @test er ≈ dense_macro_series(refr.truth) atol = 1e-12 rtol = 1e-10
    ownr = [findall(isfinite, view(refr.truth, :, j)) for j in 1:N]
    @test RO.own_residual_rows(refr.oracle) == ownr
    @test refr.model.own_res_rows == ownr
    # Fully missing price bar 500 kills returns r[499]=logP[500]-logP[499] and
    # r[500]=logP[501]-logP[500]; training row t's label is r[t+1], so the
    # all-NaN rows are t=498 (label r[499]) and t=499 (label r[500]).  Row
    # t=500's label r[501]=logP[502]-logP[501] does NOT touch price bar 500:
    # it stays a valid row in BOTH the oracle and the dense old formula.
    @test all(isnan, mor[498 - WARMUP + 1, :])
    @test all(isnan, mor[499 - WARMUP + 1, :])
    @test all(isfinite, mor[500 - WARMUP + 1, :])
    @test mor[500 - WARMUP + 1, :] ≈ refr.truth[500 - WARMUP + 1, :] atol = 1e-10 rtol = 1e-8
    @test er[498 - WARMUP + 1] == 0.0
    @test er[499 - WARMUP + 1] == 0.0
    @test er[500 - WARMUP + 1] ≈ dense_macro_series(refr.truth)[500 - WARMUP + 1] atol = 1e-12 rtol = 1e-10
    # scenario chain on the ragged panel too (own fallback exercised).
    Xr_new = generate_scenarios_v1(refr.model; S = 128, rng = MersenneTwister(78))
    Xr_old = old_scenario_draws(refr.model, refr.truth; S = 128, rng = MersenneTwister(78))
    @test isnan.(Xr_new) == isnan.(Xr_old)
    mskr2 = .!isnan.(Xr_new)
    @test Xr_new[mskr2] ≈ Xr_old[mskr2] atol = 1e-12 rtol = 1e-10

    # ---------- different s1 / band scales: an independent panel ----------
    Pv = exp.(cumsum(0.05 .* randn(MersenneTwister(31), T, N); dims = 1))
    refv = dense_reference(Pv)
    @test RO.materialize(refv.oracle) ≈ refv.truth atol = 1e-10 rtol = 1e-8
    @test RO.macro_residual_series(refv.oracle) ≈ dense_macro_series(refv.truth) atol = 1e-12 rtol = 1e-10

    # ---------- workspace reuse cannot pollute an existing lazy model ----------
    ws = KTrader.FitWorkspace()
    m1 = fit_v1(P; F_folds = 3, workspace = ws)
    snap = RO.materialize(m1.res_history)
    fit_v1(Pv; F_folds = 3, workspace = ws)   # same workspace, different panel
    @test isequal(RO.materialize(m1.res_history), snap)

    # ---------- frozen inputs: caller-side mutation cannot leak ----------
    snapshot = RO.materialize(oracle)
    prep.X_rel[1, 1] += 1.0
    prep.r[1, 1] += 1.0
    prep.B_m[1, 1] += 1.0
    prep.s1[1] *= 2.0
    prep.s_perp[2] *= 1.5
    @test isequal(RO.materialize(oracle), snapshot)
    @test RO.macro_residual_series(oracle) ≈ dense_macro_series(truth) atol = 1e-12 rtol = 1e-10

    # ---------- known-bad mutants: the same gates must reject these ----------
    # 1. wrong fold: ranges shifted by one row, still covering all of 1:343
    #    so only the fold-assignment error triggers, never a coverage check.
    wrong_ranges = [1:113, 114:227, 228:343]
    wrong_oracle = RO.ResidualOracle(fold_models, wrong_ranges, prep.ts_total,
                                     prep.r, prep.s1, prep.s_perp, prep.B_m, prep.X_rel)
    @test !isapprox(RO.materialize(wrong_oracle), truth; atol = 1e-8, rtol = 1e-8)

    # 2. stale fold models: panel A's refits against panel B's frozen inputs.
    stale_oracle = RO.ResidualOracle(refv.fold_models, refr.prep.stats.ranges,
                                     refr.prep.ts_total, refr.prep.r, refr.prep.s1,
                                     refr.prep.s_perp, refr.prep.B_m, refr.prep.X_rel)
    @test !isapprox(RO.materialize(stale_oracle), refr.truth; atol = 1e-6, rtol = 1e-6)

    # 3. NaN -> 0: treating unobserved cells as real values changes the mask.
    struct NanToZeroOracle <: AbstractMatrix{Float64}
        inner::RO.ResidualOracle
    end
    Base.size(o::NanToZeroOracle) = size(o.inner)
    Base.IndexStyle(::Type{NanToZeroOracle}) = Base.IndexCartesian()
    Base.getindex(o::NanToZeroOracle, i::Integer, j::Integer) =
        let v = o.inner[i, j]; isnan(v) ? 0.0 : v; end
    @test isfinite.(NanToZeroOracle(refr.oracle)) != isfinite.(refr.truth)

    # 4. wrong own support set: all rows instead of the observed rows only.
    @test [collect(1:refr.prep.n_res) for _ in 1:N] != ownr
end

# ---------------------------------------------------------------------------
# Off-by-one boundary regression (DevOps-verified fix, 2026-10-07).
#
# Index semantics that make this bound the ONLY legal one:
#   T_price = 600 bars  ->  T_ret = T_price - 1 = 599 return rows (r[1..599]).
#   A training row t needs the cumulative field X_rel[t] (return row t) AND
#   the label r[t+1] (return row t+1).  t+1 <= 599  =>  t <= 598 = T_ret - 1
#   = T_price - 2.  The constructor's T is size(X_rel,1) = T_ret = 599, and it
#   checks t <= T-1 = 598.  The OLD code conflated T_ret with T_price and
#   checked t <= T_ret - 2 = 597, wrongly rejecting the last legal row 598
#   (whose label r[599] is the final bar's return).
#
# If the old (T-2) bound is ever restored, the "t=598 constructs and aligns"
# test below must turn red: that row would be rejected at construction.
# ---------------------------------------------------------------------------
@testset "residual oracle off-by-one boundary (600 bars, last legal t=598)" begin
    rng = MersenneTwister(2033)
    Tb, Nb = 600, 3
    Pb = exp.(cumsum(0.01 .* randn(rng, Tb, Nb); dims = 1))
    prepb = KTrader._prepare_v1(Pb; F_folds = 3)
    fmb = refit_fold_models(prepb; F_folds = 3)

    # Caller contract: last legal training row is Tb-2 = 598 (label r[599]).
    @test prepb.ts_total[end] == Tb - 2 == 598
    @test length(prepb.ts_total) == 598 - WARMUP + 1
    # X_rel carries T_ret = Tb-1 = 599 return rows; the constructor's own T.
    @test size(prepb.X_rel, 1) == Tb - 1 == 599

    # The production-shaped oracle at the exact upper boundary constructs.
    oracle_b = RO.ResidualOracle(fmb, prepb.stats.ranges, prepb.ts_total,
                                 prepb.r, prepb.s1, prepb.s_perp, prepb.B_m,
                                 prepb.X_rel)
    @test size(oracle_b) == (length(prepb.ts_total), Nb)

    # Numerical alignment with the independent dense old formula, including
    # the final row (t=598, label r[599]) and the earliest WARMUP row (t=256,
    # where the tau=128 band is still disabled exactly as in the batch path).
    truthb = RO.dense_oof_residuals(prepb, fmb)
    gotb = RO.materialize(oracle_b)
    @test isfinite.(gotb) == isfinite.(truthb)
    @test gotb[isfinite.(gotb)] ≈ truthb[isfinite.(truthb)] atol = 1e-10 rtol = 1e-8
    last_idx = length(prepb.ts_total)
    @test prepb.ts_total[last_idx] == 598
    @test RO.residual_row(oracle_b, last_idx) ≈ truthb[last_idx, :] atol = 1e-10 rtol = 1e-8
    @test RO.residual_row(oracle_b, 1) ≈ truthb[1, :] atol = 1e-10 rtol = 1e-8
    @test RO.macro_residual_series(oracle_b) ≈ dense_macro_series(truthb) atol = 1e-12 rtol = 1e-10

    # t=599 (label would be r[600], which does not exist) must be rejected by
    # the ts_total range check.  All other shape parameters are made valid so
    # no earlier DimensionMismatch can mask the boundary failure.
    ts_bad = vcat(prepb.ts_total, 599)
    n_bad = length(ts_bad)
    q = div(n_bad, 3)
    ranges_bad = [1:q, (q + 1):2q, (2q + 1):n_bad]
    models_bad = vcat(fmb[1], fmb[2], fmb[3])   # 3 valid models for 3 ranges
    @test_throws ArgumentError RO.ResidualOracle(models_bad, ranges_bad, ts_bad,
                                                 prepb.r, prepb.s1, prepb.s_perp,
                                                 prepb.B_m, prepb.X_rel)

    # The WARMUP lower boundary is enforced by the same check: t=255 is too
    # early (the tau=2 band needs t >= 2*2+1 = 5, but the panel convention
    # starts training at WARMUP=256).
    ts_early = vcat(255, prepb.ts_total[1:(end - 1)])
    n_e = length(ts_early)
    qe = div(n_e, 3)
    ranges_e = [1:qe, (qe + 1):2qe, (2qe + 1):n_e]
    @test_throws ArgumentError RO.ResidualOracle(models_bad, ranges_e, ts_early,
                                                 prepb.r, prepb.s1, prepb.s_perp,
                                                 prepb.B_m, prepb.X_rel)
end
