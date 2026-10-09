# One current-source proper-support OOF candidate vs an authenticated saved
# original fit; no whole backtest and no hidden repeated posterior baseline.
using KTrader, LinearAlgebra, SHA, TOML, Serialization, Test
include(joinpath(@__DIR__,"solver.jl"))
const SP=Support201
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
digest(p)=bytes2hex(open(sha256,p))
function main()
    length(ARGS)<=2 || error("optional saved day index 1:8 and reference-schedule mode")
    day=isempty(ARGS) ? 1 : parse(Int,ARGS[1])
    mode=length(ARGS)==2 ? ARGS[2] : "joint"
    mode in ("joint","reference-schedule") || error("invalid candidate mode")
    1<=day<=8 || error("day outside frozen small sequence")
    BLAS.set_num_threads(6)
    current=Dict(joinpath("src",f)=>digest(joinpath(ROOT,"src",f)) for f in readdir(joinpath(ROOT,"src")) if endswith(f,".jl"))
    p=joinpath(ROOT,"dev","evidence","earlier_closure_20261009","fit_day_$(day).jls")
    meta=TOML.parsefile(p*".toml"); bytes=read(p)
    @assert bytes2hex(sha256(bytes))==meta["sha256"] && meta["sources"]==current
    old=deserialize(IOBuffer(bytes)); bytes=nothing
    for (p,h) in meta["inputs"]; @assert digest(joinpath(ROOT,p))==h; end
    bars=KTrader.load_bars(joinpath(ROOT,"data")); prices=KTrader.signal_prices(bars)
    prep=KTrader.prepare_reference(view(prices,1:old.t,:);F_folds=3)
    train=1:(first(prep.stats.ranges[3])-1); cutoff=maximum(prep.ts_total[train])+1
    witness=SP.support_basis(prep.r,cutoff)
    xx=prep.stats.full_xx-prep.stats.xx[3]
    xy=prep.stats.full_xy-prep.stats.xy[3]
    yy=prep.stats.full_yy-prep.stats.yy[3]; n=length(train)
    timed_prepare=@timed SP.prepare(xx,xy,yy,n,witness); geometry=timed_prepare.value
    println("PREPARE date=",old.date," r=",geometry.r," d=",geometry.d," P=",length(geometry.values),
        " seconds=",timed_prepare.time," bytes=",timed_prepare.bytes," compile=",timed_prepare.compile_time,
        " support_leaks=",geometry.leaks); flush(stdout)
    observation=@timed (mode=="joint" ? SP.solve_candidate(geometry;initial=old.initial[4][2],trace=true) :
        SP.solve_reference_schedule(geometry;initial=old.initial[4][2],trace=true))
    result=observation.value
    println("CANDIDATE seconds=",observation.time," bytes=",observation.bytes," compile=",observation.compile_time)
    if !result.accepted
        println("REJECTED ",result); return
    end
    reference=old.model.res_history.fold_models[3]
    alpha=result.alpha; Sigma=result.Sigma; spectrum=result.spectrum
    G=(spectrum.B .* (1.0./(spectrum.values.+alpha)))'*spectrum.basis'
    posterior=KTrader.condition_trace_neutrality(G,KTrader.ridge_covariance(spectrum,alpha),Sigma,prep.N,length(KTrader.BANDS))
    # Evaluate ORIGINAL ambient Grams once; compressed certificate cannot
    # be its own independent equivalence oracle.
    original_spectrum=KTrader.ridge_spectrum(nothing,xx,xy)
    cert=KTrader.conditioned_eb_certificate(original_spectrum,yy,n,alpha,Sigma;gauge=KTrader.relative_gauge(prep.N))
    refcert=KTrader.conditioned_eb_certificate(original_spectrum,yy,n,reference.alpha_rel,
        reference.Sigma_rel;gauge=KTrader.relative_gauge(prep.N))
    println("REFERENCE_CERTIFICATE ",refcert); flush(stdout)
    fixed_alpha=reference.alpha_rel
    fixed_S=SP.sym(geometry.Z'*reference.Sigma_rel*geometry.Z)
    fixed_Sigma=SP.lift_covariance(geometry,fixed_S)
    fixed_G=(spectrum.B .* (1.0./(spectrum.values.+fixed_alpha)))'*spectrum.basis'
    fixed_posterior=KTrader.condition_trace_neutrality(fixed_G,KTrader.ridge_covariance(spectrum,fixed_alpha),
        fixed_Sigma,prep.N,length(KTrader.BANDS))
    println("SAME_REFERENCE_POINT compression_only covariance_relative=",
        norm(fixed_Sigma-reference.Sigma_rel)/norm(reference.Sigma_rel),
        " mean_relative=",norm(fixed_posterior.G_c-reference.G_c_mean)/norm(reference.G_c_mean),
        " covariance_gate=",isapprox(fixed_Sigma,reference.Sigma_rel;atol=1e-12,rtol=1e-10),
        " mean_gate=",isapprox(fixed_posterior.G_c,reference.G_c_mean;atol=1e-12,rtol=1e-10));flush(stdout)
    println("COMPARED alpha before=",reference.alpha_rel," after=",alpha,
        " relative_delta=",abs(alpha-reference.alpha_rel)/reference.alpha_rel,
        " Sigma relative=",norm(Sigma-reference.Sigma_rel)/norm(reference.Sigma_rel),
        " mean_relative=",norm(posterior.G_c-reference.G_c_mean)/norm(reference.G_c_mean),
        " original_cert=",cert); flush(stdout)
    checks=(;alpha=isapprox(alpha,reference.alpha_rel;atol=1e-9,rtol=1e-7),
        covariance=isapprox(Sigma,reference.Sigma_rel;atol=1e-12,rtol=1e-10),
        mean=isapprox(posterior.G_c,reference.G_c_mean;atol=1e-12,rtol=1e-10))
    println("ORIGINAL_GATES ",checks); flush(stdout)
    sh=digest(joinpath(@__DIR__,"solver.jl"))
    output=joinpath(ROOT,"dev","evidence","support201","$(mode)_day$(day)_$(sh[1:12]).jls")
    ispath(output) && error("refusing to overwrite candidate evidence")
    serialize(output,(;sources=current,solver_sha256=sh,day,alpha,Sigma,G=posterior.G_c,
        checks,certificate=cert,seconds=observation.time,compile=observation.compile_time,
        iterations=result.iterations,reference_alpha=reference.alpha_rel))
    println("SAVED ",basename(output)," sha256=",digest(output)); flush(stdout)
    @testset "Actual third-fold original certificate and reference posterior" begin
        @test cert.valid
        @test checks.alpha
        @test checks.covariance
        @test checks.mean
    end
    @assert current==Dict(joinpath("src",f)=>digest(joinpath(ROOT,"src",f)) for f in readdir(joinpath(ROOT,"src")) if endswith(f,".jl"))
end
main()
