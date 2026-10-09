# N65 primitive microbenchmark. No data download, fit, scenarios or backtest.
using KTrader, LinearAlgebra, Random, Statistics, SHA
include(joinpath(@__DIR__,"..","test","fixtures","scalar_tensor_reference.jl"))
include(joinpath(@__DIR__,"..","test","fixtures","conditioned_cpu_reference.jl"))

function scalar_measure(f;repeats=9)
    f() # explicit warm-up excluded
    times=Float64[]; bytes=Int[]
    for _ in 1:repeats
        v=@timed f(); push!(times,v.time); push!(bytes,v.bytes)
    end
    (;median_seconds=median(times),minimum_seconds=minimum(times),median_bytes=median(bytes))
end

function scalar_micro_main()
    length(ARGS)==1 && only(ARGS) in ("1","6") || error("BLAS threads must be 1 or 6")
    BLAS.set_num_threads(parse(Int,only(ARGS)))
    println("Julia=",VERSION," CPU=",Sys.CPU_NAME," BLAS=",BLAS.get_config()," threads=",BLAS.get_num_threads())
    source=joinpath(@__DIR__,"..","src","response.jl")
    before=bytes2hex(sha256(read(source)))
    println("source_before_sha256=",before)
    (;spectrum,YtY,n,gauge)=cpu_synthetic_spectrum(65)
    geometry=KTrader._conditioned_scalar_geometry(spectrum,YtY;gauge)
    S=Matrix{Float64}(I,64,64); Sigma=gauge*S*gauge'
    old=()->scalar_cache_reference(spectrum,n,Sigma,geometry;gauge)
    new=()->KTrader._conditioned_scalar_cache(spectrum,n,Sigma,geometry;gauge)
    ref=old(); current=new()
    nodes=collect(range(log(KTrader.EB_ALPHA_MIN),log(KTrader.EB_ALPHA_MAX);length=33))
    @assert isequal(ref.evidence.(nodes),current.evidence.(nodes))
    @assert isequal(ref.derivative.(nodes),current.derivative.(nodes))
    println("synthetic N65/P910/rank910; 33 node evidence and derivatives exact; fixed geometry excluded on BOTH sides")
    println("cache_before_A ",scalar_measure(old))
    println("cache_after_B ",scalar_measure(new))
    println("cache_after_B2 ",scalar_measure(new))
    println("cache_before_A2 ",scalar_measure(old))
    println("build_and_33_derivatives_before ",scalar_measure(()->old().derivative.(nodes)))
    println("build_and_33_derivatives_after ",scalar_measure(()->new().derivative.(nodes)))
    println("derivatives_only_before ",scalar_measure(()->ref.derivative.(nodes)))
    println("derivatives_only_after ",scalar_measure(()->current.derivative.(nodes)))
    after=bytes2hex(sha256(read(source))); @assert before==after
    println("source_after_sha256=",after)
end
scalar_micro_main()
