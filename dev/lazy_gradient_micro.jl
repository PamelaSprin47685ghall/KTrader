using KTrader, LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__,"..","test","fixtures","conditioned_cpu_reference.jl"))
include(joinpath(@__DIR__,"..","test","fixtures","conditioned_eager_state_reference.jl"))

function lazy_measure(f)
    f() # specialization warm-up excluded
    samples=[@timed f() for _ in 1:7]
    (;seconds=median(x.time for x in samples),bytes=median(x.bytes for x in samples))
end

function lazy_micro_main()
    length(ARGS)==1 || error("supply BLAS threads (1 or 6)")
    threads=parse(Int,only(ARGS)); threads in (1,6) || error("bounded topology only")
    BLAS.set_num_threads(threads)
    println("Julia=",VERSION," BLAS=",BLAS.get_config()," threads=",threads)
    f=cpu_synthetic_spectrum(65)
    cache=KTrader.conditioned_alpha_cache(f.spectrum,f.YtY,f.n,0.7;gauge=f.gauge)
    cache.jcore[]=KTrader.build_jacobian_cores(cache)
    rng=MersenneTwister(751); U=Matrix(qr(randn(rng,64,64)).Q)
    for floor_case in (false,true)
        values=collect(range(0.3,2.0;length=64))
        floor_case && (values[1:19].=1e-8)
        S=Matrix(Symmetric(U*Diagonal(values)*U'))
        before=()->eager_state_reference(cache,S)
        after=()->KTrader._conditioned_cached_state(cache,S;direction=true)
        old=before(); new=after()
        for key in (:value,:direction,:M,:J)
            @assert isequal(getproperty(old,key),getproperty(new,key))
        end
        @assert isequal(KTrader.conditioned_cached_state(cache,S;gradient=true),old)
        println("synthetic N65/rank910, cores prebuilt, floor=",floor_case," exact state and gradient agreement")
        println("state_before_A ",lazy_measure(before))
        println("state_after_B ",lazy_measure(after))
        println("state_after_B2 ",lazy_measure(after))
        println("state_before_A2 ",lazy_measure(before))
        println("full_gradient_public_api ",lazy_measure(()->KTrader.conditioned_cached_state(cache,S;gradient=true)))
        # Same certificate work included on both sides; no diagnostic omitted.
        println("state_and_certificate_before ",lazy_measure(()->begin
            x=before(); KTrader.covariance_certificate(S,x.direction)
        end))
        println("state_and_certificate_after ",lazy_measure(()->begin
            x=after(); KTrader.covariance_certificate(S,x.direction)
        end))
        flush(stdout)
    end
end
lazy_micro_main()
