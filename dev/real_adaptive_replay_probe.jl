# Bounded one-day numerical integration replay on independently captured
# *real* full-universe optimization/audit scenarios. No account backtest.
# This deliberately does not establish current source-signature acceptance.
using Dates, Serialization, Random, LinearAlgebra, Printf
BLAS.set_num_threads(parse(Int,get(ENV,"PROBE_BLAS_THREADS","4")))
module RealAdaptiveReplay
    using Random, LinearAlgebra
    include(joinpath(@__DIR__,"../src/gate0/geometry.jl"))
    include(joinpath(@__DIR__,"../src/gate0/modes.jl"))
    include(joinpath(@__DIR__,"../src/gate0/response.jl"))
    include(joinpath(@__DIR__,"../src/gate0/posterior.jl"))
    include(joinpath(@__DIR__,"../src/gate0/innovation.jl"))
    include(joinpath(@__DIR__,"../src/gate0/kelly.jl"))
    include(joinpath(@__DIR__,"../src/gate0/quadrature.jl"))
end
const R=RealAdaptiveReplay
path=get(ENV,"PROBE_CAPTURE_PATH",
    "/tmp/ktrader-two-year-20261011/one-day-qmc-65536-both-current.jls")
d=deserialize(path)
@assert d.timing_only && d.Mlow==32768 && d.Mhigh==65536 &&
        size(d.X8192,2)>=64 && size(d.audit8192)==size(d.X8192) &&
        isequal(d.X4096,d.X8192[1:d.Mlow,:])
seed=UInt64(0x00D05EED00000001) # src/gate0/driver.jl production seed
function gen(m::Int,rule::R.SobolOwenRule,_rng::AbstractRNG)
    1<=m<=d.Mhigh || error("replay scenario count outside captured full universe")
    X=rule.seed==seed ? d.X8192 :
      rule.seed==R.audit_seed(seed) ? d.audit8192 :
      error("unexpected optimisation/audit Sobol seed")
    copy(view(X,1:m,:))
end
src=R.RQMCScenarioSource(gen,size(d.X8192,2),2)
println("CAPTURED_REAL_ADAPTIVE_REPLAY date=$(d.date) N=$(src.n_assets) cap=$(d.Mhigh) " *
        "capture_signature_is_stale=$(d.timing_only)")
flush(stdout)
for round in 1:2
    local t0=time_ns()
    result=R.adaptive_scenario_kelly(src;
              rule_seed=seed,min_scenarios=16384,max_scenarios=65536,
              weight_tol=1e-4,utility_tol=1e-6,kelly_tol=1e-6)
    @assert result.converged && result.M==65536
    @assert result.certificate_A<=1e-4 &&
            abs(result.certificate_B)<=1e-6 &&
            result.certificate_B_audit<=1e-6 &&
            R.cash_kelly_certified(result.certificate_C;tol=1e-6)
    @printf("CAPTURED_ADAPTIVE_REPLAY_OK run=%d sec=%.4f M=%d A=%.9g Bopt=%.9g Baudit=%.9g conic_loaded=%s\n",
            round,(time_ns()-t0)/1e9,result.M,
            result.certificate_A,result.certificate_B,
            result.certificate_B_audit,isdefined(R,:Convex))
    flush(stdout)
end
