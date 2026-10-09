# Isolate the Kelly stage from the timed-out backtest. Uses the saved S300
# local-panel scenario matrix; no prepare/fit, scheduler or network request.
include(joinpath(@__DIR__,"local_panel_stages.jl"))
using Test, Statistics
include(joinpath(@__DIR__,"..","test","fixtures","kelly_packing_reference.jl"))

# The winning candidate is now production; no second dev implementation.

function packing_measure(f;repeats=5)
    observations=[@timed(f()) for _ in 1:repeats]
    (;seconds=median(o.time for o in observations),bytes=median(o.bytes for o in observations),
        compile_seconds=sum(o.compile_time for o in observations))
end

function packing_main()
    isempty(ARGS) || error("this probe has no workload overrides")
    BLAS.set_num_threads(1)
    meta=TOML.parsefile(joinpath(PANEL_EVIDENCE,"decision.toml"))
    meta["sha256"]=="b5a1374722e4ca0a494aa859c9583eaee97639d9146de2e4bc58f0cdca3c65a5" || error("wrong decision baseline")
    X=panel_load("decision",meta["source_hashes"]).X
    @assert size(X)==(300,65) && all(isfinite,X)
    sources=panel_sources(); active=collect(1:65); tradable=trues(65)
    indexed=view(X,:,active); packed=Matrix(indexed)
    println("S300/N65 actual saved scenarios, all-tradable controlled decision; BLAS=1")
    println("indexed_type=",typeof(indexed)," indexed_strided=",indexed isa StridedMatrix,
        " packed_strided=",packed isa StridedMatrix," cells_equal=",isequal(indexed,packed))
    before=()->kelly_indexed_reference(X,active,tradable)
    after=()->KTrader.scenario_weights(X,active,tradable)
    a=@timed before(); b=@timed after()
    println("first_reference seconds=",a.time," compile=",a.compile_time)
    println("first_production seconds=",b.time," compile=",b.compile_time); flush(stdout)
    @testset "Indexed/dense Kelly: same law, original certificates" begin
        @test isequal(indexed,packed)
        ca=KTrader.kelly_certificate(indexed,a.value)
        cb=KTrader.kelly_certificate(indexed,b.value)
        @test KTrader.certified(ca,1e-8)
        @test KTrader.certified(cb,1e-8)
        @test abs(ca.objective-cb.objective)<=1e-10
        @test norm(a.value-b.value,1)<=1e-7
        println("weight_L1=",norm(a.value-b.value,1)," objective_diff=",abs(ca.objective-cb.objective))
    end
    println("before_A ",packing_measure(before))
    println("after_B ",packing_measure(after))
    println("after_B2 ",packing_measure(after))
    println("before_A2 ",packing_measure(before))
    panel_sources()==sources || error("source changed during isolated measurement")
    println("kelly_source=",sources["src/kelly.jl"]," unchanged=true; no whole-backtest timing claim")
end
packing_main()
