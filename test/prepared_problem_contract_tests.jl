using Test, Random, LinearAlgebra
using KTrader

# M1 boundary contract: History -> prepare -> PreparedProblem -> solve.
# Both producers return the SAME typed object; solve is the single fit
# owner. Field set, shape rejection at the boundary, snapshot stability
# across a later state.advance, and numeric equivalence at the original
# tolerances (the existing suites exercise fit_v1/solve_current! which now
# route through solve; this file pins the boundary itself).

const PP_FIELDS = (:active_idx, :N_universe, :T, :N, :s1, :s_m, :s_perp,
    :adj_act, :alive_now, :r, :m, :relative_embedding, :observed, :X_rel, :B_m,
    :ts_total, :n_res, :P_features, :stats, :macro_stats, :X_rel_stacked,
    :F_folds, :ws_owner, :ws_generation)

function pp_fixture(;seed=731, T=320, N=2)
    rng=MersenneTwister(seed)
    ret=0.007 .* randn(rng, T, N)
    ret[280:286,:] .*= 12.0
    exp.(cumsum(ret; dims=1))
end

@testset "PreparedProblem typed boundary" begin
    prices=pp_fixture()
    F=3

    # [B1] reference producer returns the typed object with the full field
    # set (no field lost in the NamedTuple->struct migration, no extras).
    p_ref=prepare_reference(prices; F_folds=F)
    @test p_ref isa PreparedProblem
    @test Set(propertynames(p_ref)) == Set(PP_FIELDS)
    @test p_ref.stats isa FoldStatistics
    @test p_ref.macro_stats === nothing            # reference path: lazy macro
    @test p_ref.F_folds == F
    # Decision-day observation mask: owned Vector{Bool} frozen at prepare
    # time from the panel's last row (constructor default).
    @test p_ref.alive_now isa Vector{Bool}
    @test p_ref.alive_now == isfinite.(prices[end, p_ref.active_idx])

    # [B2] incremental producer returns the same type with identical
    # values on every consumed field (original preparation tolerances).
    state=initialize_inference(prices[1:size(prices,1)-1,:]; F_folds=F)
    advance_exact!(state, view(prices, size(prices,1), :))
    p_inc=prepare_incremental(state)
    @test p_inc isa PreparedProblem
    @test Set(propertynames(p_inc)) == Set(PP_FIELDS)
    @test p_inc.active_idx == p_ref.active_idx
    @test p_inc.ts_total == p_ref.ts_total
    @test p_inc.s1 ≈ p_ref.s1 atol=1e-12 rtol=1e-12
    @test p_inc.s_m ≈ p_ref.s_m atol=1e-12 rtol=1e-11
    @test p_inc.s_perp ≈ p_ref.s_perp atol=1e-12 rtol=1e-11
    @test p_inc.B_m ≈ p_ref.B_m atol=1e-10 rtol=1e-11
    @test p_inc.observed == p_ref.observed
    @test p_inc.alive_now == p_ref.alive_now
    @test p_inc.stats.ranges == p_ref.stats.ranges
    for name in (:full_xy,:full_yy)
        @test getproperty(p_inc.stats,name) ≈ getproperty(p_ref.stats,name) atol=1e-9 rtol=1e-11
    end
    p_ref.stats.full_xx === nothing || (@test p_inc.stats.full_xx ≈ p_ref.stats.full_xx atol=1e-9 rtol=1e-11)
    for f in 1:F
        @test p_inc.stats.xy[f] ≈ p_ref.stats.xy[f] atol=1e-9 rtol=1e-11
        @test p_inc.stats.yy[f] ≈ p_ref.stats.yy[f] atol=1e-9 rtol=1e-11
    end
    # incremental supplies typed macro blocks; reference defers them.
    @test p_inc.macro_stats isa MacroStatistics
    @test length(p_inc.macro_stats.folds) == F
    Xm=p_ref.B_m[p_ref.ts_total,:]; ym=p_ref.m[p_ref.ts_total .+ 1]
    @test p_inc.macro_stats.full.xx ≈ Xm'Xm atol=1e-9 rtol=1e-11
    @test p_inc.macro_stats.full.yy[1,1] ≈ dot(ym,ym) atol=1e-9 rtol=1e-11

    # [B3] Snapshot stability: a later advance does not disturb an already
    # prepared object (state.advance writes only the incremental core; the
    # prepared buffers are untouched). Solve after the advance must equal
    # the no-advance solve at model level.
    m1=solve(p_inc; ridge_alpha=1.0)
    nxt=exp.(log.(prices[end,:]) .+ 0.01 .* randn(MersenneTwister(7), 2))
    advance_exact!(state, nxt)
    m2=solve(p_inc; ridge_alpha=1.0)
    @test m1.mu_pred == m2.mu_pred
    @test m1.resp.G_c_mean == m2.resp.G_c_mean
    @test isequal(isfinite.(m1.res_history), isfinite.(m2.res_history))

    # [B4] solve is the single owner: fit_v1 routes through it and matches
    # the direct call at the original tolerances (alpha, prediction,
    # scenarios with a fixed seed, weights).
    m_fit=fit_v1(prices; ridge_alpha=1.0, F_folds=F)
    m_dir=solve(prepare_reference(prices; F_folds=F); ridge_alpha=1.0)
    @test m_fit.resp.alpha_macro ≈ m_dir.resp.alpha_macro atol=1e-9 rtol=1e-7
    @test m_fit.resp.alpha_rel ≈ m_dir.resp.alpha_rel atol=1e-9 rtol=1e-7
    @test m_fit.mu_pred ≈ m_dir.mu_pred atol=1e-10 rtol=1e-8
    A=generate_scenarios_v1(m_fit; S=64, rng=MersenneTwister(31))
    B=generate_scenarios_v1(m_dir; S=64, rng=MersenneTwister(31))
    @test A ≈ B atol=1e-9 rtol=1e-8
    wa=KTrader.scenario_weights(A, m_fit.active_indices, trues(m_fit.N_universe))
    wb=KTrader.scenario_weights(B, m_dir.active_indices, trues(m_dir.N_universe))
    @test wa ≈ wb atol=1e-6 rtol=1e-6

    # [B4b] PUBLIC OWNING BOUNDARY for the observation mask: the solved
    # model's relative_observed must be EXACTLY the prepared object's
    # observed BitMatrix (identity, not a zero-filled or re-derived mask).
    # Regression anchor for the solve-destructuring bug where the bare name
    # `observed` resolved to the module-level FUNCTION KTrader.observed and
    # V1Model construction received a function instead of the mask (real red
    # log preserved in the mission record: MethodError convert
    # typeof(KTrader.observed) -> BitMatrix at predict.jl V1Model field).
    # Ragged variant: a NaN block must stay NaN-masked end-to-end (the mask
    # is carried, never zero-imputed).
    let rng=MersenneTwister(733), T2=320, N2=2
        ret2=0.007 .* randn(rng, T2, N2)
        ret2[60:75, 2] .= NaN              # ragged: asset 2 unobserved block
        pr2=exp.(cumsum(ret2; dims=1))
        p2=prepare_reference(pr2; F_folds=F)
        m2b=solve(p2; ridge_alpha=1.0)
        # Value equivalence, NOT array identity: the mask VALUES must be
        # exact (missing stays unobserved, never zero-imputed); the owner may
        # legitimately copy the array in future without breaking this contract.
        @test isequal(m2b.relative_observed, p2.observed)
        @test any(.!p2.observed[:, 2])                       # mask really ragged
        @test p2.alive_now == isfinite.(pr2[end, p2.active_idx])

        @test !any(m2b.relative_observed[60:75, 2])          # NaN block stays unobserved
    end

    # [B5] Shape corruption fails at the boundary (constructor), not
    # inside the solver.
    bad=Dict(f => getfield(p_ref, f) for f in PP_FIELDS)
    @test_throws DimensionMismatch PreparedProblem(; bad..., alive_now=falses(3))
    @test_throws DimensionMismatch PreparedProblem(; bad..., r=zeros(3, 7))
    @test_throws DimensionMismatch PreparedProblem(; bad..., s1=[1.0])
    @test_throws DimensionMismatch PreparedProblem(; bad..., observed=falses(2, 2))
    @test_throws ArgumentError PreparedProblem(; bad..., F_folds=1)

    # [B6] solve rejects a mismatched F_folds (layout is owned by the
    # prepared object, not by the solver call).
    @test_throws ArgumentError solve(p_ref; F_folds=F+1)
    # The forward keeps working as a pure alias (dev-side compat only).
    m_fwd=KTrader._fit_prepared_v1(p_ref; ridge_alpha=1.0)
    @test m_fwd.mu_pred ≈ m_dir.mu_pred atol=0 rtol=0

    # [B7] Generation binding under an EXPLICITLY SHARED workspace: after a
    # second prepare on the same owner, the OLD prepared object must fail
    # loudly with a clear staleness error (use-after-reprepare is rejected,
    # never silently solved with the reused buffer's content), while the
    # NEW object solves normally.
    ws=KTrader.FitWorkspace()
    q1=prepare_reference(prices; F_folds=F, workspace=ws)
    @test q1.ws_owner === ws
    @test q1.ws_owner isa Union{Nothing,KTrader.FitWorkspace}   # concrete owner, not an Any bag
    q2=prepare_reference(prices[1:size(prices,1)-1,:]; F_folds=F, workspace=ws)
    @test q2.ws_owner === ws
    @test q2.ws_generation > q1.ws_generation
    @test_throws ErrorException solve(q1; ridge_alpha=1.0)   # stale lease: clear error
    m_new=solve(q2; ridge_alpha=1.0)
    @test length(m_new.mu_pred) == p_ref.N_universe         # new object solves fine

    # [B7b] Partial/failed prepare also invalidates the old lease: the
    # generation is bumped BEFORE any buffer use, so even a prepare that
    # throws mid-way (here: F_folds beyond the training rows) has already
    # retired q2's lease.
    @test_throws ArgumentError prepare_reference(prices; F_folds=10^6, workspace=ws)
    @test_throws ErrorException solve(q2; ridge_alpha=1.0)

    # [B7c] An UN-reprepared object may be solved repeatedly (solve never
    # bumps; same generation stays valid), and the DEFAULT reference path
    # (no shared workspace, independent owner) is never gated.
    ws2=KTrader.FitWorkspace()
    q3=prepare_reference(prices; F_folds=F, workspace=ws2)
    ma=solve(q3; ridge_alpha=1.0)
    mb=solve(q3; ridge_alpha=1.0)
    @test ma.mu_pred == mb.mu_pred
    p_indep=prepare_reference(prices; F_folds=F)
    @test p_indep.ws_owner === nothing
    m_indep=solve(p_indep; ridge_alpha=1.0)
    @test m_indep.mu_pred ≈ ma.mu_pred atol=1e-10 rtol=1e-8
    # B3 (advance stability) above already pins the incremental-side
    # snapshot semantics at the original tolerances.
end
