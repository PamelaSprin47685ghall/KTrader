using KTrader, Test, Random, LinearAlgebra, Dates

@testset "Optional ruler cache removal retains actual prefix math" begin
    rng=MersenneTwister(1009037)
    for N in (2,3,5)
        prices=100exp.(cumsum(0.012randn(rng,365,N);dims=1))
        prices[1:19,N].=NaN; prices[281:283,1].=NaN
        cache=KTrader.PriceHistoryCache(prices)
        rs=KTrader.build_prefix_ruler_stats(cache.log_prices,cache.first_price)
        for T in (310,343)
            prefix=view(prices,1:T,:)
            old=KTrader.prepare_reference(prefix;F_folds=3,history_cache=cache,ruler_stats=rs)
            plain=KTrader.prepare_reference(prefix;F_folds=3,history_cache=cache)
            @test old.active_idx==plain.active_idx
            @test old.observed==plain.observed
            for k in (:s1,:s_m,:s_perp)
                @test isapprox(getproperty(old,k),getproperty(plain,k);atol=1e-12,rtol=1e-11)
            end
            for k in (:full_xx,:full_xy,:full_yy)
                x=getproperty(old.stats,k); y=getproperty(plain.stats,k)
                @test x===nothing ? y===nothing : isapprox(x,y;atol=1e-9,rtol=1e-11)
            end
            # Test the nonlinear output too, not a cached path against itself.
            a=KTrader.solve(old); b=KTrader.solve(plain)
            @test isapprox(a.resp.alpha_rel,b.resp.alpha_rel;atol=1e-9,rtol=1e-7)
            @test isapprox(a.resp.Sigma_rel,b.resp.Sigma_rel;atol=1e-12,rtol=1e-10)
            @test isapprox(a.mu_pred,b.mu_pred;atol=1e-12,rtol=1e-10)
            Xa=KTrader.generate_scenarios_v1(a;S=64,rng=MersenneTwister(9))
            Xb=KTrader.generate_scenarios_v1(b;S=64,rng=MersenneTwister(9))
            @test isequal(isfinite.(Xa),isfinite.(Xb))
            @test isapprox(Xa,Xb;atol=1e-12,rtol=1e-10)
            wa=KTrader.kelly_weights_v1(Xa); wb=KTrader.kelly_weights_v1(Xb)
            @test norm(wa-wb,1)<=1e-7
            @test KTrader.certified(KTrader.kelly_certificate(Xa,wa),1e-8)
            @test KTrader.certified(KTrader.kelly_certificate(Xb,wb),1e-8)
        end
        # The optional PUBLIC cache guard remains strict when used.
        corrupt=deepcopy(rs); corrupt.acc[343,1,1]+=1.0
        @test_throws ArgumentError KTrader.prepare_reference(view(prices,1:343,:);history_cache=cache,ruler_stats=corrupt)
    end
end

@testset "Incremental optional initialization caches do not define the state" begin
    prices=exp.(cumsum(0.01randn(MersenneTwister(53),345,3);dims=1))
    prices[1:20,3].=NaN; prices[282,1]=NaN
    history=KTrader.PriceHistoryCache(prices)
    rs=KTrader.build_prefix_ruler_stats(history.log_prices,history.first_price)
    a=KTrader.initialize_inference(view(prices,1:340,:);history_cache=history,ruler_stats=rs)
    b=KTrader.initialize_inference(view(prices,1:340,:))
    @test b.history_cache===nothing && b.ruler_stats===nothing
    for t in 341:345
        KTrader.advance_exact!(a,view(prices,t,:)); KTrader.advance_exact!(b,view(prices,t,:))
    end
    x=KTrader.prepare_incremental(a); y=KTrader.prepare_incremental(b)
    @test x.s1==y.s1 && x.s_perp==y.s_perp
    @test x.relative_embedding==y.relative_embedding
    for field in (:full_xx,:full_xy,:full_yy,:xx,:xy,:yy,:ranges)
        @test isequal(getproperty(x.stats,field),getproperty(y.stats,field))
    end
    @test KTrader.inference_diagnostics(a)==KTrader.inference_diagnostics(b)
end
