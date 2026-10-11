# S10_gap_diag.jl — 深场景主解的零列 KKT gap 分布（读已有产物；轻量）
# 用法: julia S10_gap_diag.jl <t> [dir]
using Serialization, LinearAlgebra

function main()
    t = parse(Int, ARGS[1])
    D = length(ARGS) > 1 ? ARGS[2] :
        joinpath(@__DIR__, "..", "archive", "evidence",
                 "gate0_multiday_run_20261010", "batch_out")
    xp = deserialize(joinpath(D, "batch_t$(t)_X2.bin"))
    md = deserialize(joinpath(D, "batch_t$(t)_mid2.bin"))
    X = xp.X
    S = size(X, 1)
    Xf = X[:, md.free_pos]
    b = md.b
    w = md.w_val
    n = length(md.free_pos)
    if w === nothing
        println("t=", t, "  w_val is nothing; nothing to diagnose")
        return
    end
    wf = w[1:n]
    wc = w[n+1]
    wealth = Xf * wf .+ wc .+ b
    inv_w = 1.0 ./ wealth
    g = vcat(Xf' * inv_w, [sum(inv_w)]) ./ S
    nu = maximum(g)
    println("t=", t, "  S=", S, "  n_free=", n, "  w=", w, "  nu=", nu)
    zero_idx = findall(<=(1e-12), w)
    gaps = [nu - g[i] for i in zero_idx]
    println("zero cols: ", length(zero_idx), "  idx=", zero_idx)
    if !isempty(gaps)
        srt = sort(gaps)
        q(f) = srt[min(max(1, ceil(Int, f * length(srt))), length(srt))]
        println("gap: min=", minimum(srt), "  p25=", q(0.25), "  median=", q(0.5),
                "  p75=", q(0.75), "  p90=", q(0.90), "  max=", maximum(srt))
    end
    base_active = count(>(1e-12), w)
    for m in (1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1.0)
        extra = count(i -> w[i] <= 1e-12 && (nu - g[i]) < m, eachindex(w))
        println("margin=", m, ": active=", base_active + extra,
                "  (base=", base_active, " +", extra, ")")
    end
end

main()
