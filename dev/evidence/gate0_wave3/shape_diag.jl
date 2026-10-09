# 诊断脚本（Wave 3）：9.7 的 M_z 异常——定位 z_pool 极端行的来源
# （行号、模长、该行 V_{s-1} 的最小正特征值）。
module DiagEnv3
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "geometry.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "modes.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "innovation.jl"))
end
using .DiagEnv3: innovation_state, z_pool_rows, domain_mode_basis,
    frac_weights_gate0, mp_sqrt_factors, V_at, shape_moments
using Random, LinearAlgebra, Statistics

SEED = 20261009
N_a = 5
n_big = 400
ids_big = collect(257:256 + n_big)
eps_big = 0.01 .* randn(MersenneTwister(SEED + 1), n_big, N_a)
masks_big = [trues(N_a) for _ in 1:n_big]
R_t = [1, 2, 3, 4]
E_active = domain_mode_basis(N_a)
st_big = innovation_state(eps_big, ids_big, masks_big, R_t, E_active; t = 256 + n_big)
Z = z_pool_rows(st_big, 0.4)
println("Z size = ", size(Z))
rownorm = [norm(Z[i, :]) for i in 1:size(Z, 1)]
println("row norms: max = ", maximum(rownorm), " at i = ", argmax(rownorm),
        "; p99 = ", sort(rownorm)[end - 4])
# 前 10 行的模长与 V_{s-1} 最小正特征值
for i in 1:8
    Jpos = st_big.L_t[i]
    uJ = st_big.row_ids[st_big.J_t]
    w = frac_weights_gate0(0.4, uJ[Jpos] - 1 - uJ[1] + 1)
    Vsm = zeros(4, 4)
    den = 0.0
    tsm = uJ[Jpos] - 1
    for k in 1:Jpos - 1
        τ = tsm + 1 - uJ[k]
        wk = frac_weights_gate0(0.4, τ)[τ]
        Vsm .+= wk .* (st_big.eps_R[k, :] * st_big.eps_R[k, :]')
        den += wk
    end
    Vsm ./= den
    F = mp_sqrt_factors(Vsm)
    println("L row $i (J=$Jpos): |z| = ", round(rownorm[i], digits=3),
            "  rank(V_sm) = ", F.rank,
            "  min pos λ = ", isempty(F.positive) ? "none" : round(F.positive[1], sigdigits=3))
end
Mz = shape_moments(st_big, 0.4)
println("Mz diag = ", round.(diag(Mz), digits=3))
println("Mz max offdiag = ", round(maximum(abs.(Mz - Diagonal(diag(Mz)))), digits=3))
# 剔除前 20 行后的 M_z 与 kurt
Z20 = Z[21:end, :]
M20 = (Z20' * Z20) ./ size(Z20, 1)
println("M_z (rows 21+) diag = ", round.(diag(M20), digits=3),
        "  max offdiag = ", round(maximum(abs.(M20 - Diagonal(diag(M20)))), digits=3))
k4(v) = begin
    m = mean(v); s2 = mean(abs2.(v .- m)); mean((v .- m) .^ 4) / (s2 * s2)
end
println("kurt col1 full = ", round(k4(Z[:, 1]), digits=2),
        "  rows21+ = ", round(k4(Z20[:, 1]), digits=2))
