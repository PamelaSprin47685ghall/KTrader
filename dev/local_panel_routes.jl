# One preparation route on the authenticated full local prefix. No posterior
# solve, new market data, truncation, checkpoint copy, or default modification.
include(joinpath(@__DIR__,"local_panel_certificate.jl"))

function route_baseline(sources)
    meta=TOML.parsefile(joinpath(PANEL_EVIDENCE,"prepared.toml"))
    meta["sha256"]==PANEL_BASELINE_DIGESTS["prepared"] || error("baseline hash changed")
    before=meta["source_hashes"]
    before["src/response.jl"]=="c0d005c5c4d9abde3fb93397af6549f111268e73da0e5b034e85fb538b49ecc2" || error("wrong baseline response")
    before["src/incremental.jl"]=="bab652adfa74847a42033255a8559f1e46d0337313c19c130b5cd57a4a4b246d" || error("wrong baseline incremental")
    keys(before)==keys(sources) || error("runtime source set changed")
    # This before/after audit reviews precisely the response and pending-
    # filter edits. The preparation schema/data/geometry owners must match.
    all(k in ("src/response.jl","src/incremental.jl") || before[k]==sources[k] for k in keys(before)) ||
        error("unreviewed runtime source change")
    panel_load("prepared",before).prep
end

function route_comparison(actual,reference)
    @testset "N65 preparation route against authenticated batch fields" begin
        @test actual.active_idx==reference.active_idx
        @test actual.ts_total==reference.ts_total
        @test actual.alive_now==reference.alive_now
        @test isequal(actual.observed,reference.observed)
        @test isequal(actual.r,reference.r)
        @test actual.stats.ranges==reference.stats.ranges
        for key in (:s1,:s_m,:s_perp)
            a=getproperty(actual,key); b=getproperty(reference,key)
            @test isapprox(a,b;atol=1e-12,rtol=1e-11)
            println(key," max_abs=",maximum(abs.(a.-b)))
        end
        for key in (:m,:relative_embedding,:X_rel,:B_m)
            a=getproperty(actual,key); b=getproperty(reference,key)
            @test isapprox(a,b;atol=1e-10,rtol=1e-11)
        end
        for key in (:full_xx,:full_xy,:full_yy)
            a=getproperty(actual.stats,key); b=getproperty(reference.stats,key)
            @test isapprox(a,b;atol=1e-9,rtol=1e-11)
            println(key," relative_frobenius=",norm(a-b)/(1+norm(b)))
        end
        for key in (:xx,:xy,:yy), f in 1:reference.F_folds
            a=getproperty(actual.stats,key)[f]; b=getproperty(reference.stats,key)[f]
            @test isapprox(a,b;atol=1e-9,rtol=1e-11)
        end
        @test actual.N==reference.N==65
        @test actual.T==reference.T==14309
        @test actual.P_features==reference.P_features==910
    end
end

function panel_routes_main()
    length(ARGS)==1 && only(ARGS) in ("default","8192") || error("route: default | 8192")
    BLAS.set_num_threads(6)
    sources=panel_sources()
    reference=route_baseline(sources)
    @assert reference.N_universe==reference.N==65 && reference.active_idx==collect(1:65)
    limit=only(ARGS)=="default" ? KTrader.REGIME_MATERIALIZE_ROW_LIMIT_DEFAULT : 8192
    println("full authenticated prefix T=",reference.T," N=",reference.N,
            " F=",reference.F_folds," row_limit=",limit,
            " gram_budget=",KTrader.REGIME_GRAM_BUDGET_DEFAULT,
            " source=",sources["src/incremental.jl"])
    println("single init + single prepare; cold code allowed, no numerical warmup or fit")
    state=panel_measure("initialize_inference") do
        KTrader.initialize_inference(reference.adj_act;F_folds=reference.F_folds,
            materialize_row_limit=limit)
    end
    before=KTrader.inference_diagnostics(state)
    println("before=",before); flush(stdout)
    timing=KTrader.DecisionTiming()
    actual=panel_measure("prepare_incremental") do
        KTrader.prepare_incremental(state;timing)
    end
    after=KTrader.inference_diagnostics(state)
    println("after=",after)
    @test after.counters[:solves]==0
    @test after.counters[:fast]+after.counters[:fallback]==1
    @test after.counters[:rebuilds]==0
    route_comparison(actual,reference)
    for (bucket,seconds) in zip(KTrader.TIMING_BUCKETS,timing.seconds)
        println("timing ",bucket,"=",seconds)
    end
    panel_sources()==sources || error("runtime changed during route audit")
    println("source_after=",sources["src/incremental.jl"]," all runtime hashes unchanged")
end
panel_routes_main()
