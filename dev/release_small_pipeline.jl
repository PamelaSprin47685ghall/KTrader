# Bounded, synthetic 3-asset full/OOF -> scenarios -> Kelly regression.
# baseline runs BEFORE the source change; verify consumes that frozen output.
# No real market panel, no backtest and no alternate production solver.
using KTrader, LinearAlgebra, Random, Serialization, SHA, Test

function release_small_pipeline()
    rng=MersenneTwister(1080751)
    prices=100exp.(cumsum(0.01randn(rng,340,3);dims=1))
    prices[1:35,3].=NaN
    prices[280:281,2].=NaN
    prices[302,1]=NaN
    prep=KTrader.prepare_reference(prices;F_folds=3)
    KTrader.solve(prep) # same default cold numerical start; JIT warm-up only
    observation=@timed KTrader.solve(prep)
    model=observation.value
    X=KTrader.generate_scenarios_v1(model;S=64,rng=MersenneTwister(108))
    weights=KTrader.kelly_weights_v1(X)
    cert=KTrader.kelly_certificate(X,weights)
    @assert KTrader.certified(cert,1e-8)
    println("synthetic T340/N3/F3/S64 solve_seconds=",observation.time,
            " solve_bytes=",observation.bytes," Kelly=",cert); flush(stdout)
    (; alpha_m=model.resp.alpha_macro,alpha_r=model.resp.alpha_rel,
       G=model.resp.G_c_mean,Sigma=model.resp.Sigma_rel,mu=model.mu_pred,
       covariance=model.pred_moments.L_rel*model.pred_moments.L_rel',
       d=model.d_posterior,v=model.v_forecasts,X,weights,cert)
end

function release_small_main()
    length(ARGS)==1 && only(ARGS) in ("baseline","verify") || error("phase must be baseline or verify")
    BLAS.set_num_threads(1)
    path=joinpath(@__DIR__,"evidence","release_work_20261008","small_pipeline.jls")
    if only(ARGS)=="baseline"
        ispath(path) && error("refusing to overwrite baseline")
        output=release_small_pipeline()
        source_hash=bytes2hex(sha256(read(joinpath(@__DIR__,"..","src","response.jl"))))
        serialize(path,(;source_hash,output))
        println("baseline_source_sha256=",source_hash)
        println("baseline_sha256=",bytes2hex(sha256(read(path))))
    else
        data=read(path)
        bytes2hex(sha256(data))=="23a95350e58cacfa8cbb87231ad0b22512d5c432440e748a8428aa086dba174e" || error("baseline bytes changed")
        record=deserialize(IOBuffer(data))
        record.source_hash=="adde30b78d89e7298f3e33cbf2955eaa80afc4861bac968c0b770d45f2af0546" || error("unexpected baseline source")
        println("consumed_baseline_sha256=",bytes2hex(sha256(data)))
        actual=release_small_pipeline()
        @testset "Synthetic full-OOF predictive and Kelly against before-change output" begin
            for key in (:alpha_m,:alpha_r,:G,:Sigma,:mu,:covariance,:d,:v,:X)
                a=getproperty(actual,key); b=getproperty(record.output,key)
                @test isapprox(a,b;atol=1e-12,rtol=1e-10)
                delta=abs.(a.-b)
                println(key," max_abs=",delta isa Number ? delta : maximum(delta))
            end
            @test isapprox(actual.weights,record.output.weights;atol=1e-9,rtol=1e-9)
            @test KTrader.certified(actual.cert,1e-8)
            @test abs(actual.cert.objective-record.output.cert.objective)<=1e-10
        end
    end
end
release_small_main()
