# Bounded CPU-only log-sum Kelly kernel benchmark, without an account run.
# Tests the SAME full Hessian via BLAS vs the independent scalar reference,
# and re-certifies both solutions on the original unscaled objective.
using Random, LinearAlgebra, Test, Printf
BLAS.set_num_threads(parse(Int,get(ENV,"PROBE_BLAS_THREADS","4")))
include(joinpath(@__DIR__, "..", "gate0", "KTraderGate0.jl"))
const RealIPM = KTraderGate0
rng = MersenneTwister(20261012)
N = 64
S = parse(Int,get(ENV,"PROBE_S","512"))
S <= 8192 && S >= 128 || error("bounded scenario count must be 128..8192")
X = exp.(0.03 .* randn(rng,S,N))
b = zeros(S)
cash = ones(S)
@printf("LOGSUM_BENCH scenarios=%d dimension=%d BLAS=%d\n",S,N,BLAS.get_num_threads())
flush(stdout)
t0 = time_ns()
fast = RealIPM._cash_kelly_ipm(X,b,cash,1.0,1e-6;blas_scan=true)
tfast = (time_ns()-t0)/1e9
@printf("BLAS_FULL_HESSIAN converged=%s iters=%d time_sec=%.5f\n",
        fast.converged,fast.iters,tfast)
flush(stdout)
fast.converged || error("BLAS IPM failed on baseline: $(fast.reason)")
cf=RealIPM.cash_kelly_certificate(X,fast.w_val[1:N],fast.w_val[end];base=b)
@test RealIPM.cash_kelly_certified(cf;tol=1e-6)
if S <= 1024
    t0=time_ns()
    scalar=RealIPM._cash_kelly_ipm(X,b,cash,1.0,1e-6;blas_scan=false)
    ts=(time_ns()-t0)/1e9
    @printf("SCALAR_FULL_HESSIAN converged=%s iters=%d time_sec=%.5f\n",
            scalar.converged,scalar.iters,ts)
    flush(stdout)
    @test scalar.converged
    cs=RealIPM.cash_kelly_certificate(X,scalar.w_val[1:N],scalar.w_val[end];base=b)
    @test RealIPM.cash_kelly_certified(cs;tol=1e-6)
    @test isapprox(scalar.w_val,fast.w_val;atol=1e-5,rtol=1e-4)
    @printf("ORACLE_MATCH max_abs_weight_diff=%.9g speedup=%.3f\n",
            maximum(abs.(scalar.w_val.-fast.w_val)),ts/tfast)
end
@printf("CERTIFIED_IPM original_gap=%.8g kkt=%.8g\n",cf.objective_gap,cf.kkt_residual)
for iteration in 1:2
    local t0=time_ns()
    repeated=RealIPM._cash_kelly_ipm(X,b,cash,1.0,1e-6;blas_scan=true)
    @test repeated.converged
    @test isapprox(repeated.w_val,fast.w_val;atol=1e-10,rtol=1e-9)
    @printf("BLAS_STEADY repeat=%d sec=%.5f\n",iteration,(time_ns()-t0)/1e9)
    flush(stdout)
end
