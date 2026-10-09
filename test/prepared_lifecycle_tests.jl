using Test, Random, LinearAlgebra
using KTrader

# Prepared lifecycle contract — the executable proof of the frozen
# decision-day observation mask (Manager architecture decision 2026-10-07):
# solve must not read the caller's mutable price panel; the alive support is
# an owned mask frozen at prepare time. The caller-side mutation below IS
# the known-bad injection: if solve ever re-reads adj_act[end, :] (the
# pre-fix silent aliasing), [L1f] turns red. A fresh prepare must still see
# the new information (per-object freeze, not a cache). The explicit-mask
# ownership, workspace lease and model-stability sections complete the
# lifecycle contract.

function lifecycle_panel(; seed=737, T=320, N=3)
    rng=MersenneTwister(seed)
    ret=0.006 .* randn(rng, T, N)
    exp.(cumsum(ret; dims=1))
end

@testset "Prepared lifecycle: frozen decision mask, lease, model stability" begin
    prices=lifecycle_panel()
    F=3

    # ---------- [L1f] frozen decision-day mask (both mutation directions) ----------
    p=prepare_reference(prices; F_folds=F)
    mask0=copy(p.alive_now)
    @test all(mask0)                                  # panel starts fully observed
    m_before=solve(p; ridge_alpha=1.0)
    e0_before=copy(m_before.e0)
    mu_before=copy(m_before.mu_pred)
    X_before=generate_scenarios_v1(m_before; S=64, rng=MersenneTwister(41))

    prices[end, 2]=NaN                                # known-bad: mutate AFTER prepare
    @test p.alive_now == mask0                        # the owned mask did not move
    m_after=solve(p; ridge_alpha=1.0)                 # SAME prepared object
    @test m_after.e0 == e0_before
    @test m_after.mu_pred == mu_before
    X_after=generate_scenarios_v1(m_after; S=64, rng=MersenneTwister(41))
    @test isequal(X_after, X_before)                  # scenarios bit-stable

    # fresh prepare reflects the new information (per-object freeze, not a cache)
    p_new=prepare_reference(prices; F_folds=F)
    @test !p_new.alive_now[2]
    m_new=solve(p_new; ridge_alpha=1.0)
    @test m_new.e0[2] == 0.0                          # NaN day leaves the e0 support
    @test m_new.e0[1] ≈ 1/sqrt(2) atol=1e-15
    @test m_new.e0[3] ≈ 1/sqrt(2) atol=1e-15

    # reverse direction: restore a finite last-row price; the OLD object is
    # still frozen, a fresh prepare picks the observation up again.
    prices[end, 2]=prices[end, 1]
    m_rev=solve(p; ridge_alpha=1.0)
    @test m_rev.e0 == e0_before
    @test m_rev.mu_pred == mu_before
    p_rev=prepare_reference(prices; F_folds=F)
    @test all(p_rev.alive_now)

    # ---------- [L1o] explicit mask supply is owned (no caller aliasing) ----------
    fields=propertynames(p)
    kw=Dict(f => getfield(p, f) for f in fields)
    bv=falses(length(p.alive_now)); bv[1]=true
    q=PreparedProblem(; kw..., alive_now=bv)
    @test q.alive_now isa Vector{Bool}
    @test q.alive_now == [true, false, false]
    bv[2]=true                                        # mutate the caller-side input
    @test q.alive_now == [true, false, false]         # the stored mask is a copy

    # ---------- [L2] workspace lease + already-solved model stability ----------
    ws=KTrader.FitWorkspace()
    pa=prepare_reference(prices; F_folds=F, workspace=ws)
    ma=solve(pa; ridge_alpha=1.0)
    snap=(e0=copy(ma.e0), mu=copy(ma.mu_pred),
          G=copy(ma.resp.G_c_mean), Sigma=copy(ma.resp.Sigma_rel),
          L=copy(ma.pred_moments.L_rel), d=copy(ma.d_posterior),
          res=KTrader.materialize(ma.res_history))
    pb=prepare_reference(lifecycle_panel(; seed=739); F_folds=F, workspace=ws)
    @test_throws ErrorException solve(pa; ridge_alpha=1.0)   # stale lease: fail loud
    mb=solve(pb; ridge_alpha=1.0)
    @test length(mb.mu_pred) == pa.N_universe               # the new lease solves
    # the already-solved model is untouched by the reuse (full-field snapshot,
    # not just res_history: the model owns every matrix it holds)
    @test ma.e0 == snap.e0
    @test ma.mu_pred == snap.mu
    @test ma.resp.G_c_mean == snap.G
    @test ma.resp.Sigma_rel == snap.Sigma
    @test ma.pred_moments.L_rel == snap.L
    @test ma.d_posterior == snap.d
    @test isequal(KTrader.materialize(ma.res_history), snap.res)
end
