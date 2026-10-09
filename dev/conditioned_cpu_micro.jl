# Bounded primitive benchmark; no price history, posterior fit or backtest.
# Usage: julia --project=. dev/conditioned_cpu_micro.jl [N] [BLAS threads]
using KTrader, LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__,"..","test","fixtures","conditioned_cpu_reference.jl"))

function cpu_measure(f;repeats=5)
    f() # explicit specialization warm-up, excluded from measurements
    times=Float64[]; bytes=Int[]
    for _ in 1:repeats
        v=@timed f()
        push!(times,v.time); push!(bytes,v.bytes)
    end
    (;median_seconds=median(times),median_bytes=median(bytes),minimum_seconds=minimum(times))
end

# Candidate ONLY: same weighted products, grouped by block row (14 rather
# than 105 GEMMs). Packing is included in its measured cost. Not production.
function cpu_blocked_cores(cache)
    nc=length(cache.blocks); d=size(cache.bcat,1); rk=div(size(cache.bcat,2),nc)
    tall=reshape(permutedims(reshape(cache.bcat,d,rk,nc),(1,3,2)),d*nc,rk)
    cores=Matrix{Float64}[]
    for r in 1:nc
        row=view(cache.bwcat,:,(r-1)*rk+1:r*rk)*view(tall,(r-1)*d+1:nc*d,:)'
        for s in r:nc
            A=Matrix(view(row,:,(s-r)*d+1:(s-r+1)*d))
            for j in 1:d, i in 1:j
                v=(A[i,j]+A[j,i])/2; A[i,j]=A[j,i]=v
            end
            push!(cores,A)
        end
    end
    cores
end

function cpu_micro_main()
    N=isempty(ARGS) ? 16 : parse(Int,ARGS[1])
    3<=N<=65 || error("N must be in 3:65; this is a bounded primitive probe")
    threads=length(ARGS)<2 ? 1 : parse(Int,ARGS[2])
    1<=threads<=6 || error("BLAS threads must be in 1:6")
    BLAS.set_num_threads(threads)
    println("Julia=",VERSION," CPU=",Sys.CPU_NAME," BLAS=",BLAS.get_config()," threads=",BLAS.get_num_threads())
    f=cpu_synthetic_spectrum(N); (;spectrum,YtY,n,gauge)=f
    println("synthetic dense spectrum N=",N," P=",size(spectrum.basis,1)," rank=",length(spectrum.values)," n=",n)
    geometry=KTrader._conditioned_scalar_geometry(spectrum,YtY;gauge)
    ageometry=isdefined(KTrader,:_conditioned_alpha_geometry) ? KTrader._conditioned_alpha_geometry(geometry) : nothing
    S=Matrix{Float64}(I,N-1,N-1); Sigma=gauge*S*gauge'
    oldalpha=()->cpu_reference_alpha(spectrum,YtY,n,0.7;gauge)
    newalpha=isdefined(KTrader,:_conditioned_alpha_cache) ?
        ()->KTrader._conditioned_alpha_cache(spectrum,YtY,n,0.7,ageometry;gauge) : oldalpha
    old=oldalpha(); new=newalpha()
    @assert old.h==new.h && old.R==new.R && old.bwcat==new.bwcat
    expected=cpu_reference_cores(old)
    actual=KTrader.build_jacobian_cores(new)
    @assert all(isequal.(expected,actual))
    println("equivalence: alpha h/R/bwcat and all cores exact")
    # ABBA order reduces first/second ordering bias; includes every allocation
    # in each primitive, excludes the once-per-fit geometry (reported separately).
    println("geometry_once ",cpu_measure(()->KTrader._conditioned_scalar_geometry(spectrum,YtY;gauge)))
    ageometry===nothing || println("alpha_pack_once ",cpu_measure(()->KTrader._conditioned_alpha_geometry(geometry)))
    println("alpha_before_A ",cpu_measure(oldalpha))
    println("alpha_after_B ",cpu_measure(newalpha))
    println("alpha_after_B2 ",cpu_measure(newalpha))
    println("alpha_before_A2 ",cpu_measure(oldalpha))
    println("cores_before ",cpu_measure(()->cpu_reference_cores(old)))
    println("cores_after ",cpu_measure(()->KTrader.build_jacobian_cores(new)))
    candidate=cpu_blocked_cores(new)
    println("blocked_core_candidate ",cpu_measure(()->cpu_blocked_cores(new)),
        " max_abs_diff=",maximum(maximum(abs.(a-b)) for (a,b) in zip(expected,candidate)))
    println("scalar_build ",cpu_measure(()->KTrader._conditioned_scalar_cache(spectrum,n,Sigma,geometry;gauge)))
    scalar=KTrader._conditioned_scalar_cache(spectrum,n,Sigma,geometry;gauge)
    nodes=collect(range(log(KTrader.EB_ALPHA_MIN),log(KTrader.EB_ALPHA_MAX);length=33))
    println("alpha_33_derivatives ",cpu_measure(()->scalar.derivative.(nodes)))
    new.jcore[]=actual
    println("state_with_cores ",cpu_measure(()->KTrader.conditioned_cached_state(new,S;gradient=true)))
    println("fixed_point_one_iteration ",cpu_measure(()->KTrader.sigma_stationary_fixed_point(new,S;iters=1)))
end
cpu_micro_main()
