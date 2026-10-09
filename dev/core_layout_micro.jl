# Isolated synthetic N65 kernels: no inference, market data or backtest.
using KTrader, LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__,"..","test","fixtures","conditioned_cpu_reference.jl"))

function layout_tall(cache)
    d=size(cache.bcat,1); nc=length(cache.blocks); rk=div(size(cache.bcat,2),nc)
    reshape(permutedims(reshape(cache.bcat,d,rk,nc),(1,3,2)),d*nc,rk)
end

function layout_extract(A,d,nc)
    out=Vector{Matrix{Float64}}(undef,div(nc*(nc+1),2))
    p=0
    for r in 1:nc, s in r:nc
        p+=1
        core=Matrix(view(A,(r-1)*d+1:r*d,(s-1)*d+1:s*d))
        for j in 1:d, i in 1:j
            v=(core[i,j]+core[j,i])/2
            core[i,j]=core[j,i]=v
        end
        out[p]=core
    end
    out
end

function layout_gemm(cache,tall)
    weighted=tall .* cache.V.weights'
    layout_extract(weighted*tall',size(cache.bcat,1),length(cache.blocks))
end

function layout_syrk(cache,tall)
    w=cache.V.weights
    sign=all(>=(0),w) ? 1.0 : all(<=(0),w) ? -1.0 : error("mixed signs")
    weighted=tall .* sqrt.(abs.(w))'
    A=zeros(size(tall,1),size(tall,1))
    BLAS.syrk!('U','N',sign,weighted,0.0,A)
    LinearAlgebra.copytri!(A,'U')
    layout_extract(A,size(cache.bcat,1),length(cache.blocks))
end

function layout_measure(f)
    f()
    samples=[@timed f() for _ in 1:5]
    (;seconds=median(x.time for x in samples),bytes=median(x.bytes for x in samples))
end

function layout_main()
    length(ARGS)==1 || error("supply BLAS threads (1 or 6)")
    threads=parse(Int,only(ARGS)); threads in (1,6) || error("bounded topology only")
    BLAS.set_num_threads(threads)
    println("Julia=",VERSION," BLAS=",BLAS.get_config()," threads=",threads); flush(stdout)
    f=cpu_synthetic_spectrum(65)
    cache=KTrader.conditioned_alpha_cache(f.spectrum,f.YtY,f.n,0.7;gauge=f.gauge)
    tall=layout_tall(cache); expected=KTrader.build_jacobian_cores(cache)
    println("N=65 P=910 rank=910 synthetic; warm-up excluded, five samples")
    for (name,work) in (("existing",()->KTrader.build_jacobian_cores(cache)),
                        ("gemm_reuse_layout",()->layout_gemm(cache,tall)),
                        ("syrk_reuse_layout",()->layout_syrk(cache,tall)),
                        ("gemm_with_pack",()->layout_gemm(cache,layout_tall(cache))),
                        ("syrk_with_pack",()->layout_syrk(cache,layout_tall(cache))))
        actual=work()
        err=maximum(maximum(abs.(a-b)) for (a,b) in zip(actual,expected))
        @assert all(isapprox(a,b;atol=1e-12,rtol=1e-12) for (a,b) in zip(actual,expected))
        println(name," ",layout_measure(work)," max_abs=",err); flush(stdout)
    end
end
layout_main()
