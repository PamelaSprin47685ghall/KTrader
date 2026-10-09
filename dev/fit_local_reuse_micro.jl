# Synthetic, function-level measurement only. No prepare, EB optimization,
# scenario generation, backtest, external data, or package installation.
# Tests run first so the before/after formulas have an equality witness.
using LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__,"..","test","fit_local_reuse_tests.jl"))

function reuse_micro_measure(label,f)
    f() # explicit specialization warm-up, outside the reported samples
    bytes=minimum([@allocated(f()) for _ in 1:3])
    seconds=median([@elapsed(f()) for _ in 1:5])
    println(label," bytes=",bytes," median_seconds=",seconds)
    (;bytes,seconds)
end

function fit_local_reuse_micro()
    BLAS.set_num_threads(1)
    rng=MersenneTwister(8172)
    N=8; n=160; P=2length(KTrader.BANDS)*N
    gauge=KTrader.relative_gauge(N)
    X=randn(rng,n,P); Y=randn(rng,n,N)*gauge*gauge'
    spectrum=KTrader.ridge_spectrum(X,X'X,X'Y)
    YtY=Y'Y; Sigma=gauge*Matrix{Float64}(I,N-1,N-1)*gauge'
    geometry=KTrader._conditioned_scalar_geometry(spectrum,YtY;gauge)
    a=log(0.7)
    before=()->reuse_scalar_reference(spectrum,YtY,n,Sigma;gauge).evidence(a)
    after=()->KTrader._conditioned_scalar_cache(spectrum,n,Sigma,geometry;gauge).evidence(a)
    @assert isapprox(before(),after();atol=1e-11,rtol=1e-12)
    println("scope=synthetic per-Sigma cache plus one evidence; N=8 n=160 P=112 BLAS=1")
    println("reuse excludes one-time fit geometry construction; no whole-fit claim")
    reuse_micro_measure("scalar_before",before)
    reuse_micro_measure("scalar_after",after)

    oracle=reuse_residual_fixture(;nrows=1800,N=8)
    old=()->sum(reuse_macro_reference(oracle))
    new=()->sum(KTrader.macro_residual_series(oracle))
    @assert old()==new()
    println("scope=synthetic scalar residual series; rows=1800 N=8 F=3 recurrent masks")
    reuse_micro_measure("residual_before",old)
    reuse_micro_measure("residual_after",new)
end

fit_local_reuse_micro()
