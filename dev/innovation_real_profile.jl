# Bounded, no-trading performance probe for one captured genuine prequential
# residual matrix. Rebuilds no synthetic residuals or portfolio positions.
# Use only for timing and exact-vs-reference comparisons, never to certify a
# completed backtest or a source signature that has since changed.
using Dates, Serialization, LinearAlgebra, FFTW, Printf, Profile
BLAS.set_num_threads(parse(Int,get(ENV,"PROBE_BLAS_THREADS","4")))
FFTW.set_num_threads(parse(Int,get(ENV,"PROBE_FFT_THREADS","1")))
include(joinpath(@__DIR__, "..", "gate0", "KTraderGate0.jl"))
const InnoPerf = KTraderGate0
path=get(ENV,"PROBE_INNOVATION_PATH",
         "/tmp/ktrader-two-year-20261011/real-one-day-innovation-input.jls")
data=deserialize(path)
data.version==1 && size(data.eps_R,2)>=64 ||
    error("input must be the real 64+ asset residual fixture")
R=data.eps_R
ids=data.u_J
println("REAL_INNOVATION_DIAGNOSTIC date=$(data.date) dim=$(size(R)) " *
        "fftw=$(FFTW.get_num_threads()) blas=$(BLAS.get_num_threads()) " *
        "original_timing_only=$(data.timing_only)")
flush(stdout)

measure(f,name) = begin
    t0=time_ns()
    result=f()
    @printf("MEASURE name=%s sec=%.6f\n",name,(time_ns()-t0)/1e9)
    flush(stdout)
    result
end

# Compile and verify a real single-node likelihood before measuring any time.
for d in (0.05,0.6)
    plan=measure("fft_prepare_$(d)") do
        InnoPerf._prepare_fractional_fft(R,ids)
    end
    plan === nothing && error("real risk domain unexpectedly rejects the FFT")
    prefix=measure("fft_prefix_$(d)") do
        InnoPerf._fft_prefix_covariances(R,ids,d;prepared=plan)
    end
    val=measure("quasi_parts_$(d)") do
        sum(-0.5*InnoPerf._quasi_measure(prefix(i),view(R,i,:),nothing)
            for i in 2:length(ids))
    end
    @assert isfinite(val)
    q=measure("quasi_direct_$(d)") do
        InnoPerf.quasi_loglik(R,ids,d;prepared_fft=plan)
    end
    @assert isapprox(val,q;atol=1e-6,rtol=1e-10)
    @printf("NODE_REPLAY d=%.3f ll=%.9g\n",d,q)
end

st=measure("adaptive_first") do
    InnoPerf.adaptive_d_quadrature(R,ids,data.target_t)
end
@assert st.converged && st.nodes==data.d_nodes
@assert isapprox(st.weights,data.d_weights;atol=1e-9,rtol=1e-7)
for i in 1:2
    measure("adaptive_steady_$i") do
        s=InnoPerf.adaptive_d_quadrature(R,ids,data.target_t)
        @assert s.converged && s.nodes==st.nodes
    end
end
if get(ENV,"PROBE_FULL_INNOVATION","false")=="true"
    N=size(R,2)
    E=InnoPerf.risk_domain_mode_basis(collect(1:N))
    assets=R*E'
    masks=[trues(N) for _ in ids]
    for i in 1:3
        measure("full_innovation_steady_$i") do
            full=InnoPerf.innovation_state(assets,ids,masks,collect(1:N),E;
                                          t=data.target_t,require_full_rank=true)
            @assert full.d_nodes==st.nodes
            @assert isapprox(full.d_weights,st.weights;atol=1e-8,rtol=1e-6)
        end
    end
end
if get(ENV,"PROBE_PROFILE","false")=="true"
    Profile.init(n=250000,delay=0.0005)
    Profile.clear()
    Profile.@profile InnoPerf.adaptive_d_quadrature(R,ids,data.target_t)
    Profile.print(format=:flat,sortedby=:count)
end
