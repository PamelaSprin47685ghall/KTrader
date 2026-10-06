using Test, LinearAlgebra, Statistics, Random
using KTrader

@testset "Path Kelly distribution invariants" begin
    rng=MersenneTwister(2026)
    T,N=600,4
    P=exp.(cumsum(0.01randn(rng,T,N),dims=1))
    model=fit_v1(P)
    @testset "Price units and asset coordinates" begin
        scaled=fit_v1(P.*exp.(randn(rng,N))')
        @test scaled.mu_pred ≈ model.mu_pred atol=1e-10
        @test scaled.pred_moments.L_rel*scaled.pred_moments.L_rel' ≈ model.pred_moments.L_rel*model.pred_moments.L_rel' atol=1e-9
        perm=[3,1,4,2]
        other=fit_v1(P[:,perm])
        @test other.mu_pred ≈ model.mu_pred[perm] atol=1e-9
        @test other.res_history ≈ model.res_history[:,perm] atol=1e-8
        covariance=model.pred_moments.L_rel*model.pred_moments.L_rel'
        @test other.pred_moments.L_rel*other.pred_moments.L_rel' ≈ covariance[perm,perm] atol=1e-8
        @test maximum(abs,model.mu_pred)<0.005
    end
    @testset "Trace-neutral support and macro decomposition" begin
        for cols in model.resp.constraint_cols
            @test abs(sum(model.resp.G_c_mean[j,cols[j]] for j in 1:N))<1e-9
        end
        decomposition=center_of_mass_decomposition(diff(log.(P);dims=1))
        @test vec(sum(decomposition.u_perp;dims=2)) ≈ zeros(T-1) atol=1e-14
    end
    @testset "Inactive assets cannot become risk-free competitors" begin
        ragged=hcat(P[:,1],fill(NaN,T),P[:,2:end],vcat(fill(NaN,T-1),1.0))
        fit=fit_v1(ragged)
        @test fit.active_indices==[1,3,4,5]
        @test fit.mu_pred[fit.active_indices] ≈ model.mu_pred atol=1e-10
        X=generate_scenarios_v1(fit;S=64,rng=MersenneTwister(4))
        @test all(isnan,view(X,:,[2,6]))
        w=KTrader.scenario_weights(X,fit.active_indices,trues(6))
        @test w[2]==w[6]==0.0
        @test sum(w) ≈ 1.0 atol=1e-10
        # Two consecutive prices are the activation boundary, not one observed bar.
        ragged[end-1,6]=0.99
        @test 6 in active_universe_indices(ragged)
        with_return=fit_v1(ragged)
        @test with_return.own_res_rows[end]==[size(with_return.res_history,1)]
    end
    @testset "Prefix ruler preserves missing pairs and universe mapping" begin
        ragged=hcat(P[:,1],fill(NaN,T),P[:,2:end])
        ragged[100:103,3].=NaN
        logs=log.(ragged)
        first=[1,T+1,1,1,1]
        stats=build_prefix_ruler_stats(logs,first)
        for prefix in (300,T)
            columns=[4,1,3,5]
            direct=ruler(logs[1:prefix,columns],first[columns],ones(prefix))
            cached=ruler_from_stats(stats,prefix,first[columns],columns)
            @test cached ≈ direct atol=1e-12
            @test stats.cnt[prefix,3,1]==count(t -> isfinite(logs[t,3])&&isfinite(logs[t-1,3]),2:prefix)
        end
        cached_model=fit_v1(ragged;ruler_stats=stats)
        direct_model=fit_v1(ragged)
        @test cached_model.mu_pred ≈ direct_model.mu_pred atol=1e-10
        @test cached_model.s1 ≈ direct_model.s1 atol=1e-12
    end
end
