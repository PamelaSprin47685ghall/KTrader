# Bounded performance-only probe for a 65-dimensional observed-joint-history
# size typical of 2024-2026. This DOES NOT simulate a trading decision,
# and does not consume / replace true causal residuals.
using Random, LinearAlgebra, Printf
BLAS.set_num_threads(parse(Int,get(ENV,"PROBE_BLAS_THREADS","1")))
include(joinpath(@__DIR__, "..", "gate0", "KTraderGate0.jl"))
const InnovationProbe = KTraderGate0
using .InnovationProbe
import FFTW; FFTW.set_num_threads(parse(Int,get(ENV,"PROBE_FFT_THREADS","1")))

rows = parse(Int,get(ENV,"PROBE_ROWS","400"))
assets = parse(Int,get(ENV,"PROBE_ASSETS","65"))
@assert 5 <= rows <= 600 && 1 <= assets <= 65
eps = 0.015 .* randn(MersenneTwister(74),rows,assets)
ids = collect(1:rows)
println("INNOVATION_BOUNDED rows=$rows assets=$assets")
flush(stdout)
started=time_ns()
v=InnovationProbe.V_direct(eps,ids,0.6,rows)
@printf("V_DIRECT seconds=%.5f\n",(time_ns()-started)/1e9)
flush(stdout)
for j in 1:2
    local t0=time_ns()
    @assert isapprox(InnovationProbe.V_direct(eps,ids,0.6,rows),v;atol=1e-12)
    @printf("V_STEADY run=%d seconds=%.5f\n",j,(time_ns()-t0)/1e9)
end
started=time_ns()
ll=quasi_loglik(eps,ids,0.6)
@printf("QUASI_LOGLIK seconds=%.5f finite=%s\n",(time_ns()-started)/1e9,isfinite(ll))
flush(stdout)
for j in 1:2
    local t0 = time_ns()
    @assert isapprox(quasi_loglik(eps,ids,0.6),ll;rtol=1e-12)
    @printf("QUASI_STEADY run=%d seconds=%.5f\n",j,(time_ns()-t0)/1e9)
end
if get(ENV,"PROBE_SHAPE","false")=="true"
    # Explicit one-node diagnostic, NOT an adaptive strategic decision.
    E=domain_mode_basis(assets)
    st=innovation_state(eps,ids,[trues(assets) for _ in ids],
                        collect(1:assets),E;t=rows,d_nodes=[0.6],d_cells=[1.0])
    local t0=time_ns()
    z=z_pool_rows(st,0.6)
    @assert size(z)==(rows-1,assets) && all(isfinite,z)
    @printf("SHAPE_POOL_STEADY rows=%d assets=%d sec=%.5f\n",
        rows,assets,(time_ns()-t0)/1e9)
end
if get(ENV,"PROBE_INITIAL_D","false")=="true"
    # Only the initial 7-node quadrature work as a *lower-bound cost* probe.
    # Not a converged adaptive-d posterior or a permitted production shortcut.
    E=domain_mode_basis(assets)
    nodes,cells=initial_mesh()
    local t0=time_ns()
    st=innovation_state(eps,ids,[trues(assets) for _ in ids],
                        collect(1:assets),E;t=rows,d_nodes=nodes,d_cells=cells)
    @printf("INITIAL_D_STATE_ONLY nodes=%d seconds=%.5f\n",
            length(nodes),(time_ns()-t0)/1e9)
    flush(stdout)
    t0=time_ns()
    for d in st.d_nodes
        Z=z_pool_rows(st,d)
        @assert size(Z)==(rows-1,assets) && all(isfinite,Z)
    end
    @printf("INITIAL_D_POOL_ONLY nodes=%d seconds=%.5f\n",
            length(nodes),(time_ns()-t0)/1e9)
    flush(stdout)
end
