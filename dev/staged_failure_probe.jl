# Bounded analytic two-gross-state regression diagnostic. Never a backtest.
using LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "gate0", "KTraderGate0.jl"))
const AnalyticStage = KTraderGate0
for M in (64,128,256,512,1024,2048)
    X=Matrix{Float64}(undef,M,2)
    for i in 1:M
        X[i,1],X[i,2] = i<=128 ? (1.08,1.01) : (0.80,1.18)
    end
    stage=AnalyticStage._cash_kelly_staged_primary(X;tol=1e-6)
    pre=stage.mid===nothing ? nothing : stage.mid.w_val
    v=stage.mid===nothing ? nothing : AnalyticStage._cash_kelly_canonical_vertex(
           X,zeros(M),1.0,stage.cert.objective,pre,1e-8,1e-6)
    @printf("STAGE M=%d risky=%s cash=%.8g cert_gap=%.7g vertex=%s\n",
            M,string(stage.risky),stage.cash,stage.cert.objective_gap,v!==nothing)
    if v===nothing
        j=argmax(stage.risky)
        th=stage.cert.objective-1e-8
        fun(a)=sum(log, 1 .+ a .* (X[:,j].-1))/M
        lo,hi=0.,1.
        for _ in 1:66
            mid=(lo+hi)/2
            if fun(mid)>=th
                hi=mid
            else
                lo=mid
            end
        end
        w=zeros(3);w[j]=hi;w[3]=1-hi
        c=AnalyticStage.cash_kelly_certificate(X,w[1:2],w[3])
        dw=X'*(1 ./ (1 .+hi.*(X[:,j].-1)))/M
        @printf("WHY M=%d Ustar=%.12g Uvert=%.12g f0=%.12g root_gap=%.7g cert_gap=%.7g g_w=%.7g g_other=%.7g g_cash=%.7g\n",
            M,stage.cert.objective,fun(1.),fun(0.),c.objective-th,
            c.objective_gap,dw[j],dw[3-j],
            sum(1 ./ (1 .+hi.*(X[:,j].-1)))/M)
    end
end
