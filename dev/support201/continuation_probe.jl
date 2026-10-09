# Two adjacent complete third-fold problems; deterministic cold vs warm
# candidate experiment. No production override, no reference replacement.
using KTrader, LinearAlgebra, SHA, TOML, Serialization, Test, Statistics
include(joinpath(@__DIR__,"solver.jl"))
const SP=Support201
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
digest(p)=bytes2hex(open(sha256,p))
function read_day(index,sources)
    p=joinpath(ROOT,"dev","evidence","earlier_closure_20261009","fit_day_$(index).jls")
    meta=TOML.parsefile(p*".toml"); bytes=read(p)
    @assert bytes2hex(sha256(bytes))==meta["sha256"] && meta["sources"]==sources
    for (p,h) in meta["inputs"]; @assert digest(joinpath(ROOT,p))==h; end
    deserialize(IOBuffer(bytes))
end
function measure_three(f)
    rows=NamedTuple[]
    for _ in 1:3
        o=@timed f()
        push!(rows,(;seconds=o.time,bytes=o.bytes,gc=o.gctime,compile=o.compile_time))
    end
    (;median_seconds=median(x.seconds for x in rows),minimum_seconds=minimum(x.seconds for x in rows),
        maximum_seconds=maximum(x.seconds for x in rows),median_bytes=median(x.bytes for x in rows),
        median_gc=median(x.gc for x in rows),compile_sum=sum(x.compile for x in rows))
end
function main()
    isempty(ARGS) || error("fixed two-day candidate experiment")
    BLAS.set_num_threads(1)
    solver_digest=digest(joinpath(@__DIR__,"solver.jl"))
    driver_digest=digest(@__FILE__)
    println("candidate_solver_sha256=",solver_digest," driver_sha256=",driver_digest);flush(stdout)
    sources=Dict(joinpath("src",f)=>digest(joinpath(ROOT,"src",f)) for f in readdir(joinpath(ROOT,"src")) if endswith(f,".jl"))
    bars=KTrader.load_bars(joinpath(ROOT,"data")); signal=KTrader.signal_prices(bars)
    previous=nothing
    for index in 1:2
        old=read_day(index,sources)
        prep=KTrader.prepare_reference(view(signal,1:old.t,:);F_folds=3)
        train=1:(first(prep.stats.ranges[3])-1); witness=SP.support_basis(prep.r,maximum(prep.ts_total[train])+1)
        xx=prep.stats.full_xx-prep.stats.xx[3]; xy=prep.stats.full_xy-prep.stats.xy[3]
        yy=prep.stats.full_yy-prep.stats.yy[3]
        g=SP.prepare(xx,xy,yy,length(train),witness)
        cold_observation=@timed SP.solve_candidate(g;initial=old.initial[4][2],trace=false)
        cold=cold_observation.value
        println("date=",old.date," COLD seconds=",cold_observation.time," bytes=",cold_observation.bytes,
            " compile=",cold_observation.compile_time," accepted=",cold.accepted,
            " iterations=",get(cold,:iterations,-1)); flush(stdout)
        cold.accepted || error("candidate cold path rejected")
        if previous!==nothing
            @assert g.Z==previous.Z
            repeated=@timed SP.solve_candidate(g;initial=old.initial[4][2],trace=false)
            warm=@timed SP.solve_candidate(g;initial=previous.result.alpha,
                initial_state=previous.result.S,trace=false)
            println("REPEAT_COLD seconds=",repeated.time," bytes=",repeated.bytes," compile=",repeated.compile_time,
                " iterations=",get(repeated.value,:iterations,-1)); flush(stdout)
            println("WARM_CANDIDATE seconds=",warm.time," bytes=",warm.bytes," compile=",warm.compile_time,
                " accepted=",warm.value.accepted," iterations=",get(warm.value,:iterations,-1)); flush(stdout)
            warm.value.accepted || error("warm candidate rejected")
            repeat_warm=@timed SP.solve_candidate(g;initial=previous.result.alpha,
                initial_state=previous.result.S,trace=false)
            println("REPEAT_WARM seconds=",repeat_warm.time," bytes=",repeat_warm.bytes,
                " compile=",repeat_warm.compile_time," accepted=",repeat_warm.value.accepted,
                " iterations=",get(repeat_warm.value,:iterations,-1)); flush(stdout)
            @testset "Research warm/cold candidate agrees on this transition only" begin
                @test isapprox(cold.alpha,warm.value.alpha;atol=1e-9,rtol=1e-7)
                @test isapprox(cold.Sigma,warm.value.Sigma;atol=1e-12,rtol=1e-10)
                @test cold.certificate.valid && warm.value.certificate.valid
            end
            println("warm_cold_alpha_relative=",abs(cold.alpha-warm.value.alpha)/cold.alpha,
                " sigma_relative=",norm(cold.Sigma-warm.value.Sigma)/norm(cold.Sigma))
            spectrum=KTrader.ridge_spectrum(nothing,xx,xy)
            Q=KTrader.relative_gauge(prep.N)
            original=@timed KTrader.optimize_conditioned_eb(spectrum,yy,length(train);
                initial=old.initial[4][2],gauge=Q,return_certificate=true)
            repeated_original=@timed KTrader.optimize_conditioned_eb(spectrum,yy,length(train);
                initial=old.initial[4][2],gauge=Q,return_certificate=true)
            ref_now=repeated_original.value
            println("ORIGINAL_OPTIMIZER first_seconds=",original.time," first_compile=",original.compile_time,
                " repeat_seconds=",repeated_original.time," repeat_compile=",repeated_original.compile_time,
                " repeat_bytes=",repeated_original.bytes," iterations=",ref_now.iterations,
                " certificate=",ref_now.certificate); flush(stdout)
            println("candidate_vs_current_BLAS1_reference alpha_relative=",abs(cold.alpha-ref_now.alpha)/ref_now.alpha,
                " covariance_relative=",norm(cold.Sigma-ref_now.Sigma)/norm(ref_now.Sigma),
                " alpha_gate=",isapprox(cold.alpha,ref_now.alpha;atol=1e-9,rtol=1e-7),
                " covariance_gate=",isapprox(cold.Sigma,ref_now.Sigma;atol=1e-12,rtol=1e-10));flush(stdout)
            println("MEASUREMENTS same prepared/spectral inputs already resident, three samples each, natural in-call GC included")
            println("reference_A ",measure_three(()->KTrader.optimize_conditioned_eb(spectrum,yy,length(train);
                initial=old.initial[4][2],gauge=Q,return_certificate=true)));flush(stdout)
            println("candidate_cold_B ",measure_three(()->SP.solve_candidate(g;initial=old.initial[4][2])));flush(stdout)
            println("candidate_warm_C ",measure_three(()->SP.solve_candidate(g;initial=previous.result.alpha,
                initial_state=previous.result.S)));flush(stdout)
            println("reference_A2 ",measure_three(()->KTrader.optimize_conditioned_eb(spectrum,yy,length(train);
                initial=old.initial[4][2],gauge=Q,return_certificate=true)));flush(stdout)
        end
        ref=old.model.res_history.fold_models[3]
        println("vs2.0_original alpha_relative=",abs(cold.alpha-ref.alpha_rel)/ref.alpha_rel,
            " covariance_relative=",norm(cold.Sigma-ref.Sigma_rel)/norm(ref.Sigma_rel),
            " original_alpha_gate=",isapprox(cold.alpha,ref.alpha_rel;atol=1e-9,rtol=1e-7),
            " original_cov_gate=",isapprox(cold.Sigma,ref.Sigma_rel;atol=1e-12,rtol=1e-10));flush(stdout)
        previous=(;Z=g.Z,result=cold)
    end
    @assert sources==Dict(joinpath("src",f)=>digest(joinpath(ROOT,"src",f)) for f in readdir(joinpath(ROOT,"src")) if endswith(f,".jl"))
    @assert solver_digest==digest(joinpath(@__DIR__,"solver.jl")) && driver_digest==digest(@__FILE__)
    println("No general basin/OOF-isolation proof from one transition; production unchanged")
end
main()
