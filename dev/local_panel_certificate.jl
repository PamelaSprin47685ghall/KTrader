# Explicit before/after boundary: only response.jl may differ from the three
# frozen CSV pipeline artifacts. All bytes are checked before deserialization.
include(joinpath(@__DIR__,"local_panel_stages.jl"))
using Test, Statistics
include(joinpath(@__DIR__,"..","test","fixtures","conditioned_certificate_reference.jl"))

const PANEL_BASELINE_DIGESTS=Dict(
    "prepared"=>"82b18e9b0283aa5aa9ef31e95e99d4870b5af49e70d800029d9fda8e2e4bfedc",
    "model"=>"691c3297813f2511efaffbee276c11274fb4ce03f32b19bf9cd5fa8eb0cd3577",
    "decision"=>"b5a1374722e4ca0a494aa859c9583eaee97639d9146de2e4bc58f0cdca3c65a5")
function panel_baseline(name,current)
    meta=TOML.parsefile(joinpath(PANEL_EVIDENCE,name*".toml"))
    meta["sha256"]==PANEL_BASELINE_DIGESTS[name] || error("baseline hash changed")
    before=meta["source_hashes"]
    before["src/response.jl"]=="c0d005c5c4d9abde3fb93397af6549f111268e73da0e5b034e85fb538b49ecc2" || error("wrong baseline source")
    keys(before)==keys(current) || error("source set changed")
    all(k=="src/response.jl" || before[k]==current[k] for k in keys(before)) || error("unreviewed source change")
    panel_load(name,before) # loads identical authenticated bytes, never blind-deserialize
end
function certificate_samples(f)
    f() # explicit primitive specialization warm-up
    times=Float64[]; bytes=Int[]
    for _ in 1:5
        obs=@timed f()
        push!(times,obs.time); push!(bytes,obs.bytes)
    end
    (;median_seconds=median(times),minimum_seconds=minimum(times),median_bytes=median(bytes))
end
function panel_certificate_main()
    length(ARGS)==1 && only(ARGS) in ("micro","verify") || error("phase: micro | verify")
    phase=only(ARGS)
    phase=="verify" && panel_refuse_existing("model_certificate")
    BLAS.set_num_threads(6)
    sources=panel_sources()
    println("source=",sources["src/response.jl"]," Julia=",VERSION," CPU=",Sys.CPU_NAME,
            " BLAS=",BLAS.get_config()," threads=",BLAS.get_num_threads())
    previous=panel_baseline("prepared",sources)
    prep=previous.prep
    reference=panel_baseline("model",sources).model
    if phase=="micro"
        println("actual full local-panel statistics T=",prep.T," N=",prep.N," F=",prep.F_folds,
                "; one spectral decomposition, zero posterior fit")
        spectrum=KTrader.ridge_spectrum(nothing,prep.stats.full_xx,prep.stats.full_xy)
        YtY=prep.stats.full_yy; n=prep.n_res; gauge=KTrader.relative_gauge(prep.N)
        alpha=reference.resp.alpha_rel; Sigma=reference.resp.Sigma_rel
        geometry=KTrader._conditioned_scalar_geometry(spectrum,YtY;gauge)
        ag=KTrader._conditioned_alpha_geometry(geometry)
        before=()->certificate_eager_reference(spectrum,YtY,n,alpha,Sigma;gauge)
        after=()->KTrader._conditioned_eb_certificate(spectrum,YtY,n,alpha,Sigma,geometry,ag;gauge)
        @assert isequal(before(),after()) && after().valid
        println("all certificate fields exact and valid; fit-owned geometry excluded on reused side")
        println("before_A ",certificate_samples(before))
        println("after_B ",certificate_samples(after))
        println("after_B2 ",certificate_samples(after))
        println("before_A2 ",certificate_samples(before))
        println("public_fresh ",certificate_samples(()->KTrader.conditioned_eb_certificate(spectrum,YtY,n,alpha,Sigma;gauge)))
    else
        println("EXPLICIT warm-up with the SAME DecisionTiming specialization; no alpha warm-start on either call")
        panel_measure("warmup_solve") do
            KTrader.solve(prep;timing=KTrader.DecisionTiming())
            nothing
        end
        GC.gc()
        timing=KTrader.DecisionTiming()
        model=panel_measure("measured_solve") do
            KTrader.solve(prep;timing)
        end
        for (bucket,seconds) in zip(KTrader.TIMING_BUCKETS,timing.seconds)
            println("timing ",bucket,"=",seconds)
        end
        X=KTrader.generate_scenarios_v1(model;S=300,rng=MersenneTwister(1))
        weights=KTrader.scenario_weights(X,model.active_indices,previous.tradable,nothing)
        free=[j for j in model.active_indices if previous.tradable[j]]
        cert=KTrader.kelly_certificate(view(X,:,free),weights[free])
        old=panel_baseline("decision",sources)
        @testset "N65 current-source full/OOF/scenario/Kelly vs frozen before-change pipeline" begin
            for field in (:alpha_macro,:alpha_rel,:G_c_mean,:Sigma_rel)
                a=getproperty(model.resp,field); b=getproperty(reference.resp,field)
                @test isapprox(a,b;atol=1e-12,rtol=1e-10)
                delta=abs.(a.-b)
                println(field," max_abs=",delta isa Number ? delta : maximum(delta))
            end
            for field in (:mu_pred,:d_posterior,:v_forecasts)
                a=getproperty(model,field); b=getproperty(reference,field)
                @test isapprox(a,b;atol=1e-12,rtol=1e-10)
                println(field," max_abs=",maximum(abs.(a.-b)))
            end
            @test isapprox(model.pred_moments.L_rel*model.pred_moments.L_rel',
                           reference.pred_moments.L_rel*reference.pred_moments.L_rel';atol=1e-12,rtol=1e-10)
            @test isequal(isfinite.(X),isfinite.(old.X))
            finite=isfinite.(X)
            @test isapprox(X[finite],old.X[finite];atol=1e-12,rtol=1e-10)
            @test isapprox(weights,old.weights;atol=1e-9,rtol=1e-9)
            @test KTrader.certified(cert,1e-8)
            @test abs(cert.objective-old.certificate.objective)<=1e-10
            println("X max_abs=",maximum(abs.(X[finite].-old.X[finite])),
                    " weight_L1=",norm(weights-old.weights,1)," certificate=",cert)
        end
        panel_sources()==sources || error("source changed during verification")
        panel_save("model_certificate",(;sources,model,date=previous.date,symbols=previous.symbols,
            tradable=previous.tradable,timings=copy(timing.seconds)))
    end
    panel_sources()==sources || error("source changed during command")
    println("source_after=",sources["src/response.jl"])
end
if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    panel_certificate_main()
end
