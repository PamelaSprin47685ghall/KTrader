# Frozen compute_pending allocation pattern from incremental.jl SHA256
# bab652adfa74847a42033255a8559f1e46d0337313c19c130b5cd57a4a4b246d.
# Test-only: same raw triangular filter and iteration order, not a second
# production evaluator. Used to check removal of column-slice copies.
function pending_slice_reference(core,k)
    N=length(core.zsum)
    acc=Dict{Int,Tuple{Matrix{Float64},Matrix{Float64}}}()
    zu=zeros(N)
    for (b,tau) in enumerate(KTrader.BANDS)
        k>=2tau+1 || continue
        w1a=k-tau+1; w2a=k-2tau+1
        ri=KTrader.run_of_row_start(core,w2a)
        for u in w2a:k
            rho=core.runs[ri]
            if rho.last<u
                ri+=1; rho=core.runs[ri]
            end
            zu .= KTrader.zsum_at(core,u) .- KTrader.zsum_at(core,u-1)
            j=k-u
            slot=get!(acc,rho.maskid) do
                (zeros(N,length(KTrader.BANDS)),zeros(N,length(KTrader.BANDS)))
            end
            if u>=w1a
                wq=-(tau-1-j)/tau
                slot[1][:,b] .+= wq .* zu
            end
            wp=min(j+1,2tau-1-j)/tau
            slot[2][:,b] .+= wp .* zu
        end
    end
    isempty(acc) && return nothing
    regs=collect(keys(acc)); v=Vector{Matrix{Float64}}(undef,length(regs))
    for (i,mid) in enumerate(regs)
        q_m,p_m=acc[mid]
        mat=Matrix{Float64}(undef,N,KTrader.REGIME_CHANNELS)
        for b in eachindex(KTrader.BANDS)
            mat[:,2b-1] .= q_m[:,b]
            mat[:,2b] .= p_m[:,b]
        end
        v[i]=mat
    end
    KTrader.PendingPrototype(regs,v)
end

function pending_core_fixture(;N=5,T=420)
    rng=MersenneTwister(141837+N)
    core=KTrader.RawInferenceCore(collect(1:N),3)
    for t in 1:T
        r=0.01randn(rng,N)
        regime=mod(div(t-1,37),4)
        for j in 1:N
            (regime==1 && isodd(j) || regime==2 && iseven(j) || regime==3) && (r[j]=NaN)
        end
        KTrader.append_return!(core,r,3)
    end
    core
end
