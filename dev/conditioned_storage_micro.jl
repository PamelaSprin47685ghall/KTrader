# Local work-cycle benchmark, including once-per-cycle storage construction.
# No price fitting or change to BLAS configuration during a measured cycle.
using KTrader, LinearAlgebra, Random, Statistics, Test, SHA
include(joinpath(@__DIR__,"..","test","fixtures","conditioned_cpu_reference.jl"))
function storage_cycle(f,geometry,ag,S,Sigma,pooled)
    buffers=pooled ? KTrader._conditioned_alpha_storage(ag) : nothing
    checksum=0.0
    for alpha in (0.1,0.3,0.7,1.0,2.0,5.0,11.0,0.1)
        cache=KTrader._conditioned_alpha_cache(f.spectrum,f.YtY,f.n,alpha,ag,buffers;gauge=f.gauge)
        state=KTrader._conditioned_cached_state(cache,S;direction=true)
        cert=KTrader._conditioned_eb_certificate(f.spectrum,f.YtY,f.n,alpha,Sigma,geometry,ag,cache;gauge=f.gauge)
        checksum+=state.value+cert.evidence+sum(state.J)
    end
    checksum
end
function storage_measure(f)
    f() # explicit specialization warm-up, no timing claim
    observations=[@timed(f()) for _ in 1:5]
    (;seconds=median(x.time for x in observations),bytes=median(x.bytes for x in observations),
      gc=median(x.gctime for x in observations),compile=sum(x.compile_time for x in observations))
end
function storage_main()
    length(ARGS)==1 && only(ARGS) in ("1","6") || error("one argument: BLAS 1 or 6")
    BLAS.set_num_threads(parse(Int,only(ARGS)))
    path=joinpath(@__DIR__,"..","src","response.jl")
    digest=bytes2hex(open(sha256,path))
    f=cpu_synthetic_spectrum(65)
    geometry=KTrader._conditioned_scalar_geometry(f.spectrum,f.YtY;gauge=f.gauge)
    ag=KTrader._conditioned_alpha_geometry(geometry)
    S=Matrix{Float64}(I,64,64); Sigma=f.gauge*S*f.gauge'
    before=()->storage_cycle(f,geometry,ag,S,Sigma,false)
    after=()->storage_cycle(f,geometry,ag,S,Sigma,true)
    @test isequal(before(),after())
    println("N65/P910/rank910 synthetic eight-alpha cycle; initial buffers INCLUDED; same covariance per step")
    println("CPU=",Sys.CPU_NAME," Julia=",VERSION," BLAS=",BLAS.get_config()," threads=",BLAS.get_num_threads())
    println("before_A ",storage_measure(before))
    println("after_B ",storage_measure(after))
    println("after_B2 ",storage_measure(after))
    println("before_A2 ",storage_measure(before))
    bytes2hex(open(sha256,path))==digest || error("source changed during measurement")
    println("checksum exact; source=",digest," unchanged=true; not a whole solver speedup")
end
storage_main()
