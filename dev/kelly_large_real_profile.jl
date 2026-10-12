# Bounded native Kelly IPM performance study on one saved real 64-asset QMC
# scenario panel. Does not invoke Clarabel, does not calculate trading PnL.
using Dates, Serialization, LinearAlgebra, Printf
BLAS.set_num_threads(parse(Int,get(ENV,"PROBE_BLAS_THREADS","4")))
module NativeKelly
    using LinearAlgebra
    include(joinpath(@__DIR__,"../src/gate0/kelly.jl"))
end
path=get(ENV,"PROBE_CAPTURE_PATH",
         "/tmp/ktrader-two-year-20261011/one-day-qmc-32768.jls")
p=deserialize(path)
@assert p.version==2 && size(p.X8192,2)>=64
X=p.X8192
S,n=size(X)
b=zeros(S)
cash=ones(S)
println("REAL_FULL_DIM_KELLY_PROFILE S=$S assets=$n threads=$(BLAS.get_num_threads())")
flush(stdout)
for i in 1:parse(Int,get(ENV,"PROBE_REPEAT","2"))
    local t0=time_ns()
    r=NativeKelly._cash_kelly_ipm(X,b,cash,1.0,1e-6;blas_scan=true)
    @assert r.converged && r.w_val!==nothing
    c=NativeKelly.cash_kelly_certificate(X,r.w_val[1:n],r.w_val[end];base=b)
    @assert NativeKelly.cash_kelly_certified(c;tol=1e-6)
    @printf("IPM_PROFILE repeat=%d iters=%d seconds=%.5f gap=%.8g\n",
            i,r.iters,(time_ns()-t0)/1e9,c.objective_gap)
    flush(stdout)
end
