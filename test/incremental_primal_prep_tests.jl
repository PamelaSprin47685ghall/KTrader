using Test, Random, LinearAlgebra
using KTrader

# Standalone primal-path preparation gate (NOT wired into runtests.jl; run
# this file directly). Scope: preparation only (_prepare_current!, no EB,
# no solve) on N=2 panels with T=300 so that n_res = 300-2-256+1 = 43 > P
# = 2*7*2 = 28: the statistics take the PRIMAL branch and full_xx is a real
# matrix. The dual branch (n_res < P) is already exercised by t=280 in the
# main incremental suite; this file closes the natural-data primal gap.
#
# Oracle: _prepare_v1 on the same prices (batch fold_sufficient_statistics
# over the same rows, same ruler path). Tolerances match the main suite
# (atol=1e-9, rtol=1e-11) - they are gates, not fitted to any
# implementation.
function primal_prep_compare(prices; F=3)
    state=initialize_inference(prices;ridge_alpha=1.0,F_folds=F)
    p=KTrader._prepare_current!(state)
    o=KTrader._prepare_v1(prices;F_folds=F)
    # Primal gate FIRST: a silent dual regression (full_xx === nothing) must
    # fail here instead of silently comparing nothing.
    @test p.stats.full_xx !== nothing
    @test o.stats.full_xx !== nothing
    @test length(p.ts_total)==length(o.ts_total)
    @test length(o.ts_total) > 2*7*2  # n_res exceeds P: genuinely primal
    @test p.stats.ranges==o.stats.ranges
    @test p.stats.full_xx ≈ o.stats.full_xx atol=1e-9 rtol=1e-11
    @test p.stats.full_xy ≈ o.stats.full_xy atol=1e-9 rtol=1e-11
    @test p.stats.full_yy ≈ o.stats.full_yy atol=1e-9 rtol=1e-11
    for f in 1:F
        @test p.stats.xx[f] ≈ o.stats.xx[f] atol=1e-9 rtol=1e-11
        @test p.stats.xy[f] ≈ o.stats.xy[f] atol=1e-9 rtol=1e-11
        @test p.stats.yy[f] ≈ o.stats.yy[f] atol=1e-9 rtol=1e-11
    end
    @test state.counters[:fast]==1
    @test state.counters[:fallback]==0
    (; state, p, o)
end

@testset "Primal preparation statistics vs batch (N=2, n_res>P)" begin
    rng=MersenneTwister(2026)
    ret=0.006 .* randn(rng,300,2)
    ret[120:126,:] .*= 9.0   # metric drift: never freeze yesterday's ruler
    prices=exp.(cumsum(ret;dims=1))
    result=primal_prep_compare(prices)
    @test result.state.core.async_seen==false
    diag=inference_diagnostics(result.state)
    @test diag.gram_bytes>0         # interior rows exercised the aggregated path
    @test diag.materialized_rows==0

    # IPO + synchronous gap + single-cell dropout on the primal path: the
    # activation embeds coordinates (permutation, no replay), the gap opens
    # an all-missing run, the dropout splits runs, and every training row's
    # 256-deep window crosses a transition here, so the exact materialized
    # path carries the whole statistics block.
    panel=copy(prices)
    panel[1:150,2].=NaN      # asset 2 activates: prices finite from row 151
    panel[200:202,:].=NaN    # synchronous all-missing gap
    panel[240,1]=NaN         # single-cell dropout
    result=primal_prep_compare(panel)
    @test result.state.core.async_seen  # ragged rows observed, diagnostic only
    @test result.state.counters[:embeds]==1   # one activation embed, no replay
    @test result.state.counters[:rebuilds]==0
    diag=inference_diagnostics(result.state)
    @test diag.runs>=5
    @test diag.materialized_rows>0  # boundary rows exercised the exact path
end
