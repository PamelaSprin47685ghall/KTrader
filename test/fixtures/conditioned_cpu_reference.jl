# Frozen pre-optimization formulas (response.jl SHA256
# 4e20646150641455752d515f460dd46f7773512a773c1791a42733a33a462437).
# Test/benchmark only; no production dependency or solver duplication.
function cpu_reference_alpha(spectrum,YtY,n,alpha;gauge=nothing)
    d=1.0 ./ (spectrum.values .+ alpha)
    Bd=spectrum.B .* d
    R=Matrix(Symmetric(YtY-spectrum.B'*Bd))
    V=KTrader.ridge_covariance(spectrum,alpha)
    cols=KTrader.get_constraint_columns(size(YtY,1),length(KTrader.BANDS))
    blocks=[gauge===nothing ? Matrix(view(V.basis,c,:)) : gauge'*view(V.basis,c,:) for c in cols]
    h=KTrader.ridge_constraint_traces(Bd,spectrum.basis,cols)
    compact_R=gauge===nothing ? R : gauge'*R*gauge
    rank=gauge===nothing ? size(YtY,1) : size(gauge,2)
    constant=-rank/2*sum(log1p.(spectrum.values./alpha))
    nc=length(cols); dd=size(blocks[1],1); rk=size(blocks[1],2)
    bwcat=Matrix{Float64}(undef,dd,nc*rk)
    bcat=Matrix{Float64}(undef,dd,nc*rk)
    for r in 1:nc
        lo=(r-1)*rk+1; hi=r*rk
        view(bcat,:,lo:hi) .= blocks[r]
        for l in 1:rk, j in 1:dd
            bwcat[j,lo+l-1]=blocks[r][j,l]*V.weights[l]
        end
    end
    jcore=Ref{Union{Nothing,Vector{Matrix{Float64}}}}(nothing)
    (; V,R=Matrix(Symmetric(compact_R)),blocks,h,n,alpha,rank,constant,bwcat,bcat,jcore)
end

function cpu_reference_cores(cache)
    nc=length(cache.blocks); rk=div(size(cache.bcat,2),nc)
    cores=Matrix{Float64}[]
    for r in 1:nc, s in r:nc
        A=view(cache.bwcat,:,(r-1)*rk+1:r*rk)*view(cache.bcat,:,(s-1)*rk+1:s*rk)'
        push!(cores,Matrix(Symmetric((A+A') ./ 2)))
    end
    cores
end

function cpu_synthetic_spectrum(N;dual=false,seed=20261008)
    rng=MersenneTwister(seed); P=2length(KTrader.BANDS)*N
    rk=dual ? min(P-1,37) : P
    U=Matrix(qr(randn(rng,P,rk)).Q)[:,1:rk]
    values=collect(range(1.0,40.0;length=rk))
    gauge=KTrader.relative_gauge(N)
    B=0.1randn(rng,rk,N)*gauge*gauge'
    n=dual ? rk : 14052
    YtY=B'*(B ./ values)+n*(gauge*gauge')
    (; spectrum=(;values,basis=U,B,dual),YtY,n,gauge)
end
