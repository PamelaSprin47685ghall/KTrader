# Bounded one-date geometry proof on a CAPTURED genuine 64-asset QMC panel.
# No trading/account work. Never treats stale provenance as a production gate.
using Dates, Serialization, LinearAlgebra, Printf
BLAS.set_num_threads(4)
include(joinpath(@__DIR__, "..", "gate0", "KTraderGate0.jl"))
const RealVertex = KTraderGate0
path=get(ENV,"PROBE_CAPTURE_PATH",
         "/tmp/ktrader-two-year-20261011/one-day-qmc-32768-parallel-prefetch.jls")
dat=deserialize(path)
X=get(ENV,"PROBE_USE_LOW","false")=="true" ? dat.X4096 : dat.X8192
S,n=size(X)
@assert n>=64 && S>=8192
println("VERTEX_REAL_SINGLE_DAY_DIAGNOSTIC date=$(dat.date) S=$S n=$n timing_only=$(dat.timing_only)")
flush(stdout)
t0=time_ns()
ipm=RealVertex._cash_kelly_ipm(X,zeros(S),ones(S),1.0,1e-6;blas_scan=true)
@assert ipm.converged && ipm.w_val!==nothing
c=RealVertex.cash_kelly_certificate(X,ipm.w_val[1:n],ipm.w_val[end])
@assert RealVertex.cash_kelly_certified(c;tol=1e-6)
@printf("PRIMARY wmax=%.8g risk_fraction=%.8g cash=%.8g gap=%.8g sec=%.5f\n",
        maximum(ipm.w_val[1:n]),sum(ipm.w_val[1:n]),ipm.w_val[end],
        c.objective_gap,(time_ns()-t0)/1e9)
t0=time_ns()
v=RealVertex._cash_kelly_canonical_vertex(X,zeros(S),1.0,c.objective,
                                           ipm.w_val,1e-8,1e-6)
@printf("VERTEX_CERTIFIED=%s sec=%.5f\n",v!==nothing,(time_ns()-t0)/1e9)
if v!==nothing
    w,cc=v
    @printf("CANONICAL_CASH=%.10g support=%d utility_gap=%.8g\n",
            w[end],count(>(0.0),w[1:n]),c.objective-cc.objective)
else
    j=argmax(ipm.w_val[1:n])
    den=X[:,j].*ipm.w_val[j] .+ (1-ipm.w_val[j])
    g=X'*(1.0./den)/S
    @printf("AT_PRIMARY j=%d g_winner=%.8g g_cash=%.8g competitor_margin=%.8g\n",
            j,g[j],sum(1.0./den)/S,
            g[j]-maximum(g[k] for k in 1:n if k!=j))
    Ustar=c.objective
    threshold=Ustar-1e-8
    one_asset(a)=sum(log,1 .+a.*(X[:,j].-1))/S
    local lo,hi=0.,1.
    for _ in 1:66
        mid=(lo+hi)/2
        if one_asset(mid)>=threshold
            hi=mid
        else
            lo=mid
        end
    end
    a=min(1.,hi+512eps(Float64))
    candidate=zeros(n+1)
    candidate[j]=a
    candidate[end]=1-a
    cc=RealVertex.cash_kelly_certificate(X,candidate[1:n],candidate[end])
    invw=1.0./(1 .+ a.*(X[:,j].-1))
    gg=(X' * invw)./S
    margin=gg[j]-maximum(gg[k] for k in 1:n if k!=j)
    dc=gg[j]-sum(invw)/S
    @printf("VERTEX_DIAG Ustar=%.12g Ue=%.12g lower=%.12g slack=%.8g margin=%.8g cashmargin=%.8g riskupper=%.8g cashupper=%.8g certified=%s\n",
            Ustar,one_asset(1.),threshold,cc.objective-threshold,
            margin,dc,(cc.objective-threshold)/margin,
            (cc.objective-threshold)/dc,
            RealVertex.cash_kelly_certified(cc;tol=1e-6))
end
if get(ENV,"PROBE_FULL_NATIVE_KELLY","false")=="true"
    t0=time_ns()
    risky,cash,cert=RealVertex.cash_kelly(X;tol=1e-6,tie_eps=1e-8)
    @assert RealVertex.cash_kelly_certified(cert;tol=1e-6)
    @printf("REAL_FULL_KELLY_VALIDATED sec=%.5f cash=%.10g conic_loaded=%s\n",
            (time_ns()-t0)/1e9,cash,isdefined(RealVertex,:Convex))
    if v!==nothing
        @assert isapprox(cash,v[1][end];atol=1e-8,rtol=1e-6)
    end
    flush(stdout)
end
