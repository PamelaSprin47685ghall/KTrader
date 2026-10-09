# Independent reproduction of the t=14294 conditioned-EB certificate failure.
# Run:   OPENBLAS_NUM_THREADS=1 julia --project=. test/fixtures/replay_t14294.jl
# Expect: throws "conditioned EB has no certified ascent direction"
#         with free_rms = 1.4097897e-5 > tol = 1e-6, active_directions = 19.
# NOTE: OPENBLAS_NUM_THREADS=1 is part of the reproduction contract —
# multi-threaded BLAS changes summation order and flips this shallow-basin
# point to green. The fixture stores the exact captured call arguments
# (spectrum, YtY, n, initial, tol, gauge, iteration budgets) from the
# production fit at decision day t=14294 (fold 3 of 3, n=9358).
# Engineer: turn this into a permanent behavioral regression in
# test/conditioned_eb_tests.jl once the solver fix lands.
using Serialization
using KTrader
c = deserialize(joinpath(@__DIR__, "t14294_conditioned_fold3.jls"))
KTrader.optimize_conditioned_eb(c.spectrum, c.YtY, c.n; initial=c.initial, tol=c.tol,
    gauge=c.gauge, iters=c.iters, max_backtracks=c.max_backtracks,
    alpha_iters=c.alpha_iters, return_certificate=true,
    fp_iters=c.fp_iters, fp_tol=c.fp_tol)
println("UNEXPECTED: solve succeeded (fixture no longer reproduces)")
