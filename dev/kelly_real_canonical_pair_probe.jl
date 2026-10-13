# Single-day, bounded replay of EXACT canonical full-dimensional Kelly on
# previously captured real optimisation and independent audit scenarios.
# Diagnostic provenance only. Never an account/backtest result.
using Dates, Serialization, LinearAlgebra, Printf
BLAS.set_num_threads(parse(Int,get(ENV,"PROBE_BLAS_THREADS","4")))
include(joinpath(@__DIR__, "..", "gate0", "KTraderGate0.jl"))
const CanonicalPair = KTraderGate0
const K=CanonicalPair
path=get(ENV,"PROBE_CAPTURE_PATH",
    "/tmp/ktrader-two-year-20261011/one-day-qmc-65536-both-current.jls")
data=deserialize(path)
@assert data.version==2 && data.timing_only && data.Mlow==32768 &&
        data.Mhigh==65536 && size(data.X8192,2)>=64 &&
        size(data.audit8192)==size(data.X8192) &&
        isequal(data.X4096,view(data.X8192,1:data.Mlow,:))
println("CANONICAL_REAL_PAIR date=$(data.date) full_risk=$(size(data.X8192,2)) diagnostic_only=true")
flush(stdout)
results=Any[]
for (M,X) in ((data.Mlow,data.X4096),(data.Mhigh,data.X8192))
    local started=time_ns()
    s=K._cash_kelly_staged_primary(X;budget=1.,tol=1e-6,tie_eps=1e-8)
    @assert K.cash_kelly_certified(s.cert;tol=1e-6)
    rw,c,cert=K._cash_kelly_staged_canonical(X,s)
    @assert K.cash_kelly_certified(cert;tol=1e-6)
    push!(results,vcat(rw,[c]))
    @printf("CANONICAL_PAIR_LAYER m=%d native=%s cash=%.10g support=%d sec=%.5f utility_diff=%.8g\n",
        M,!isdefined(K,:Convex),c,count(>(1e-8),rw),
        (time_ns()-started)/1e9,s.cert.objective-cert.objective)
    flush(stdout)
end
w0,w1=results
objective(X,w)=sum(log,X*w[1:end-1] .+ w[end])/size(X,1)
A=sum(abs,w1.-w0)
Bopt=objective(data.X8192,w1)-objective(data.X8192,w0)
Baudit=abs(objective(data.audit8192,w1)-objective(data.audit8192,w0))
pass=A<=1e-4 && abs(Bopt)<=1e-6 && Baudit<=1e-6
@printf("CANONICAL_REAL_PAIR_CERTIFIED A=%.9g Bopt=%.9g Baudit=%.9g pass=%s\n",
    A,Bopt,Baudit,pass)
flush(stdout)
exit(pass ? 0 : 45)
