using Test, Random, LinearAlgebra, Statistics, Distributions
using KTrader
const RO = KTrader

# Independent flow-contract evidence for the lazy ResidualOracle view.
#
# Contract under test: V1Model.res_history is an AbstractMatrix.  Production
# supplies a lazy ResidualOracle; a dense Matrix with identical value
# semantics must be interchangeable WITHOUT touching any other field of the
# model.  The scenario law (rng consumption order, uniform layout, shared
# row, own-row fallback, fractional scaling) is decided before the view is
# ever read, so same-seed oracle-view and dense-view draws must agree at the
# original numerical tolerance (NOT claimed bit-identical across views: the
# oracle's fold-grouped GEMM and the dense path's plain row indexing have
# different summation orders).  Same-implementation same-seed replay IS
# bit-identical (documented in generate_scenarios_v1).
#
# Truth source: dense_oof_residuals (the exported independent old-formula
# reference) — never the oracle grading itself via Matrix(oracle).

"Refit OOF fold models from a prep, mirroring _fit_prepared_v1's loop with
warm start (1.0,1.0) and full-minus-fold Grams (same as production)."
function flow_refit_fold_models(prep; F_folds=3, ridge_alpha=nothing)
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

"Same fitted model, ONLY the res_history view swapped for a dense matrix.
Every other field (posterior moments, innovation law, own-row support,
rulers, response operator) is the identical object."
function flow_dense_view(model::V1Model, dense_res::Matrix{Float64})
    V1Model(model.active_indices, model.N_universe, model.e0, model.s1,
            model.s_macro, model.s_perp, model.relative_observed, model.resp,
            model.mu_pred, dense_res, model.own_res_rows, model.pred_moments,
            model.d_posterior, model.v_forecasts, model.v_bootstrap)
end

@testset "lazy residual oracle flow contract" begin
    rng = MersenneTwister(2031)

    # ---------- fixtures: one synchronous panel, one ragged panel ----------
    T, N = 600, 3
    P = exp.(cumsum(0.01 .* randn(rng, T, N); dims = 1))
    # ragged: asset 3 missing its first 300 bars (own-fallback must fire),
    # asset 2 with a partial hole, one fully missing price bar (row 500).
    Pr = Matrix{Float64}(P)
    Pr[1:300, 3] .= NaN
    Pr[400:403, 2] .= NaN
    Pr[500, :] .= NaN

    for (name, panel) in (("synchronous", P), ("ragged", Pr))
        model = fit_v1(panel; F_folds = 3)          # production lazy path
        prep = KTrader._prepare_v1(panel; F_folds = 3)
        fold_models = flow_refit_fold_models(prep; F_folds = 3)
        truth = RO.dense_oof_residuals(prep, fold_models)  # independent dense reference
        dense_model = flow_dense_view(model, Matrix{Float64}(truth))
        active = model.active_indices
        free = trues(model.N_universe)

        @testset "$name panel: view interchange" begin
            # The lazy view's materialization equals the dense truth at the
            # original tolerance (this is the interchange precondition; the
            # scenario comparison below is the actual flow evidence).
            mo = RO.materialize(model.res_history)
            msk = isfinite.(mo)
            @test isfinite.(mo) == isfinite.(truth)
            if any(msk)
                @test mo[msk] ≈ truth[msk] atol = 1e-10 rtol = 1e-8
            end

            # Same seed, two views: identical rng stream, identical posterior
            # and innovation law; ONLY the residual read path differs.
            # Cross-view agreement is at roundoff scale (different summation
            # orders in the two batch paths) — never claimed bit-identical.
            S = 128
            X_lazy = generate_scenarios_v1(model; S, rng = MersenneTwister(77))
            X_dense = generate_scenarios_v1(dense_model; S, rng = MersenneTwister(77))
            @test isnan.(X_lazy) == isnan.(X_dense)
            m = .!isnan.(X_lazy)
            @test X_lazy[m] ≈ X_dense[m] atol = 1e-10 rtol = 1e-8

            # Same implementation, same seed, replayed: bit-identical
            # (documented: model never mutated, oracle holds no cache).
            X_again = generate_scenarios_v1(model; S, rng = MersenneTwister(77))
            @test isequal(X_again, X_lazy)
            Xd_again = generate_scenarios_v1(dense_model; S, rng = MersenneTwister(77))
            @test isequal(Xd_again, X_dense)

            # ---------- own-row fallback is genuinely exercised ----------
            # On the ragged panel, asset 3's early rows are unobserved in the
            # truth; whenever a scenario's shared row lands there, the draw
            # must come from asset 3's own observed rows (never NaN, never an
            # invented value).  With S=128 over T rows the shared-row sampler
            # hits unobserved rows with overwhelming probability.
            if name == "ragged"
                j3 = findfirst(==(3), active)
                @test j3 !== nothing
                @test !isempty(model.own_res_rows[j3])
                col = X_lazy[:, active[j3]]
                @test all(isfinite, col)   # fallback fired, no NaN leaked
                cold = X_dense[:, active[j3]]
                @test all(isfinite, cold)
                @test col ≈ cold atol = 1e-10 rtol = 1e-8  # same fallback row chosen
                own3 = model.own_res_rows[j3]
                @test all(r -> isfinite(truth[r, j3]), own3)

                # MUTATION PROOF that the real fallback path is ENTERED (not
                # merely that own_res_rows exists): strip asset 3's own-row
                # support to an empty list and replay the SAME seed.  The rng
                # stream is consumed before the residual view is read, so the
                # shared rows chosen are identical to the healthy run above.
                # On this fixture asset 3 is unobserved on truth rows 1..44
                # (44/343 of all rows); with S=128 uniform shared rows the
                # probability of never landing there is < e^-17, so a correct
                # implementation MUST enter the fallback at least once and
                # raise on the empty support.  An implementation that skips
                # the real fallback (fills NaN/0, ignores the mask) does NOT
                # raise and this assertion turns RED.  If the seed ever fails
                # to trigger, this assertion itself fails loudly: swap the
                # seed (one short DevOps run), never the assertion.
                broken_own = [j == j3 ? Int[] : model.own_res_rows[j] for j in 1:N]
                broken = V1Model(model.active_indices, model.N_universe, model.e0,
                    model.s1, model.s_macro, model.s_perp, model.relative_observed,
                    model.resp, model.mu_pred, Matrix{Float64}(truth), broken_own,
                    model.pred_moments, model.d_posterior, model.v_forecasts,
                    model.v_bootstrap)
                @test_throws ErrorException generate_scenarios_v1(broken;
                    S, rng = MersenneTwister(77))
            end

            # ---------- both views drive certified Kelly ----------
            w_lazy = RO.scenario_weights(X_lazy, active, free)
            w_dense = RO.scenario_weights(X_dense, active, free)
            @test all(isfinite, w_lazy) && all(isfinite, w_dense)
            @test sum(w_lazy) ≈ 1.0 atol = 1e-10
            @test sum(w_dense) ≈ 1.0 atol = 1e-10
            @test w_lazy ≈ w_dense atol = 1e-6 rtol = 1e-6  # identifiable weights
            # Objective values and certificates on each view's own scenarios.
            obj_lazy = mean(log.(X_lazy[:, active] * w_lazy[active]))
            obj_dense = mean(log.(X_dense[:, active] * w_dense[active]))
            @test obj_lazy ≈ obj_dense atol = 1e-8 rtol = 1e-8
            c_lazy = RO.kelly_certificate(X_lazy[:, active], w_lazy[active])
            c_dense = RO.kelly_certificate(X_dense[:, active], w_dense[active])
            @test c_lazy.objective_gap <= 1e-8
            @test c_dense.objective_gap <= 1e-8
            @test c_lazy.feasibility <= 1e-10
            @test c_dense.feasibility <= 1e-10
        end
    end

    # ---------- advance / checkpoint leave the OLD model and oracle frozen ----------
    @testset "advance and checkpoint do not disturb the old model" begin
        panel = Pr
        state = initialize_inference(panel[1:400, :]; ridge_alpha = 1.0, F_folds = 3)
        m_old = solve_current!(state)
        mu_snap = copy(m_old.mu_pred)
        res_snap = RO.materialize(m_old.res_history)
        gc_mean_snap = copy(m_old.resp.G_c_mean)
        # Freeze the caller-side inputs the oracle captured at construction.
        oracle_old = m_old.res_history

        cp = inference_checkpoint(state)
        # The two branches advance on DIFFERENT new bars: neither may disturb
        # the old model's already-computed outputs or the oracle's frozen inputs.
        advance_exact!(state, view(panel, 401, :))
        advance_exact!(cp, view(panel, 401, :) .* 1.01)   # divergent future

        m_a = solve_current!(state)
        m_b = solve_current!(cp)

        # Old model outputs are untouched by later solves.
        @test m_old.mu_pred == mu_snap
        @test m_old.resp.G_c_mean == gc_mean_snap
        # isequal, not ==: the ragged panel's residuals contain NaN cells and
        # NaN == NaN is false, so == can never express bit-identical-with-NaN.
        # isequal(NaN, NaN) is true — this is a NaN-semantics correction of the
        # fixture, NOT a tolerance change (still exact, bit-level).
        @test isequal(RO.materialize(m_old.res_history), res_snap)
        @test isequal(RO.materialize(oracle_old), res_snap)
        # Old model still generates its own scenarios identically: the active
        # columns are all finite (own-row fallback covers unobserved cells on
        # the ragged panel) and a same-seed replay is bit-identical.
        X_old_before = generate_scenarios_v1(m_old; S = 32, rng = MersenneTwister(5))
        @test all(isfinite, X_old_before[:, m_old.active_indices])
        X_old_replay = generate_scenarios_v1(m_old; S = 32, rng = MersenneTwister(5))
        @test isequal(X_old_replay, X_old_before)
        # The two branches produced their own models on their own prefixes.
        @test length(m_a.mu_pred) == length(m_b.mu_pred) == m_a.N_universe
        # T=401 prefix: ts_total = 256:399, so both views carry 144 rows.
        @test size(m_a.res_history) == size(m_b.res_history) == (144, 3)
    end
end
