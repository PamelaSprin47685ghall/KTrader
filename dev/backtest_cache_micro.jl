# Real-data prepare comparison, zero posterior fits. Explicitly measures the
# cached path WITH its existing validation, not just the table lookup.
include(joinpath(@__DIR__,"local_panel_stages.jl"))
using Test, Statistics

function bcm_sample(f)
    f()
    GC.gc() # keep only metadata; collector outside each primitive timing
    v=NamedTuple[]
    for _ in 1:3
        o=@timed f()
        push!(v,(;time=o.time,bytes=o.bytes,gctime=o.gctime,compile_time=o.compile_time))
        o=nothing
        GC.gc() # no retained prepared results across primitive samples
    end
    (;seconds=median(x.time for x in v),bytes=median(x.bytes for x in v),
      gc=median(x.gctime for x in v),compile=sum(x.compile_time for x in v))
end
function bcm_main()
    isempty(ARGS) || error("fixed N65 prepare-only probe")
    BLAS.set_num_threads(6)
    sources=panel_sources()
    for (p,h) in PANEL_INPUT_HASHES
        panel_digest(joinpath(PANEL_ROOT,p))==h || error("input changed")
    end
    bars=KTrader.load_bars(joinpath(PANEL_ROOT,"data"))
    signal=KTrader.signal_prices(bars); t=size(signal,1)-1
    cache=KTrader.PriceHistoryCache(signal)
    println("input N=",size(signal,2)," T=",size(signal,1)," full prefix=",t,
        " BLAS=",BLAS.get_num_threads()," same validated history cache on both sides"); flush(stdout)
    KTrader.build_prefix_ruler_stats(cache.log_prices,cache.first_price)
    GC.gc()
    build=@timed KTrader.build_prefix_ruler_stats(cache.log_prices,cache.first_price)
    stats=build.value
    println("prefix_table payload=",sizeof(stats.acc)+sizeof(stats.cnt),
        " allocation=",build.bytes," seconds=",build.time," compile=",build.compile_time); flush(stdout)
    input=view(signal,1:t,:)
    old=()->KTrader.prepare_reference(input;F_folds=3,history_cache=cache,ruler_stats=stats)
    new=()->KTrader.prepare_reference(input;F_folds=3,history_cache=cache)
    a=old(); b=new()
    @testset "Full local N65 cached/no-cache preparation at original gates" begin
        for field in (:s1,:s_m,:s_perp)
            x=getproperty(a,field); y=getproperty(b,field)
            @test isapprox(x,y;atol=1e-12,rtol=1e-11)
            println(field," max_abs=",maximum(abs.(x.-y)))
        end
        @test a.active_idx==b.active_idx && a.observed==b.observed
        @test a.stats.ranges==b.stats.ranges
        for field in (:full_xx,:full_xy,:full_yy)
            x=getproperty(a.stats,field); y=getproperty(b.stats,field)
            @test isapprox(x,y;atol=1e-9,rtol=1e-11)
            println(field," relative_frobenius=",norm(x-y)/max(norm(x),eps()))
        end
        for field in (:xx,:xy,:yy), fold in 1:3
            @test isapprox(getproperty(a.stats,field)[fold],getproperty(b.stats,field)[fold];atol=1e-9,rtol=1e-11)
        end
    end
    a=nothing; b=nothing; GC.gc()
    println("each primitive sample starts after explicit GC; not natural backtest GC throughput")
    println("prepare_cached_A ",bcm_sample(old))
    println("prepare_plain_B ",bcm_sample(new))
    println("prepare_plain_B2 ",bcm_sample(new))
    println("prepare_cached_A2 ",bcm_sample(old))
    sources==panel_sources() || error("source changed")
    println("runtime unchanged; no fit or backtest; this does not establish portfolio equality")
end
bcm_main()
