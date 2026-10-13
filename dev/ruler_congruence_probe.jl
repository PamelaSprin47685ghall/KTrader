# Bounded mathematical falsification of the hypothesis that the fractional
# quasi-posterior over d is invariant to today's positive diagonal rulers.
# This is not a return/backtest/model run. Print each d likelihood difference
# and compare against the exact full-rank Gaussian volume correction.
using LinearAlgebra,Random,Printf
BLAS.set_num_threads(4)
include(joinpath(@__DIR__, "..", "gate0", "KTraderGate0.jl"))
const RulerInvariant = KTraderGate0
rng=MersenneTwister(20261012)
for N in (3,12,32),n in (N+3,100)
    E=RulerInvariant.domain_mode_basis(N)
    unscaled=randn(rng,n,N)*0.02
    ruler=exp.(0.5 .* randn(rng,N))
    reference=unscaled*E
    changed=(unscaled ./ reshape(ruler,1,:))*E
    ids=collect(1:n)
    values=[]
    for d in (0.05,0.2,0.6,1.0)
        original=RulerInvariant.quasi_loglik(reference,ids,d)
        scaled=RulerInvariant.quasi_loglik(changed,ids,d)
        push!(values,scaled-original)
    end
    @printf("CONGRUENCE N=%d n=%d differences=%s spread=%.8g fullrank_shift=%.8g\n",
            N,n,string(values),maximum(values)-minimum(values),
            -2*(n-N)*sum(log,ruler))
end
