using Test, Random, LinearAlgebra, Statistics
using KTrader

# 主平方根契约（Manager 2026-10-07 裁决 + SVD 数值细节补充）：
# U·Diag(s)·U' 是 (L·Lᵀ) 的唯一对称 PSD 主根，与 L 的右正交变换
# （列符号、简并块内旋转、秩亏）解耦；高 κ/秩亏边界保 covariance
# fidelity；同 seed 场景逐点到 roundoff 一致；law 不变。
@testset "principal sqrt root canonical coupling (scenario contract)" begin
    rng = MersenneTwister(99)
    N = 6
    # Σ 谱：3 大 λ + 简并对 (0.7,0.7) + 小正 λ=1e-10（高 κ 边界）
    V = Matrix(qr(randn(rng, N, N)).Q)
    λ = [4.0, 2.5, 1.0, 0.7, 0.7, 1e-10]
    # L = V·Diag(√λ)·Wᵀ：右正交 W 任意（因子理论合法自由度）
    W = Matrix(qr(randn(rng, N, N)).Q)
    L = V * Diagonal(sqrt.(λ)) * W'

    R = KTrader.principal_sqrt_root(L)
    Σ = L * L'
    # covariance fidelity：R² = L·Lᵀ（相对容差，高 κ 下绝对精度受限）
    @test maximum(abs.(R * R .- Σ)) <= 1e-11 * norm(Σ)
    @test minimum(eigvals(Symmetric(R))) >= -1e-11 * norm(Σ)   # PSD
    @test R == R'                                                # 对称
    @test norm(R - Matrix(Symmetric(V * Diagonal(λ) .^ 0.5 * V'))) <= 1e-10 * norm(Σ)  # = 谱定理主根

    # --- 右正交不变性（因子理论的合法符号/简并变换不改数值 plan） ---
    # 列符号翻转：L·D（D=diag(±1)）
    D = diagm([1.0, -1, 1, -1, 1, -1])
    @test maximum(abs.(KTrader.principal_sqrt_root(L * D) .- R)) <= 1e-12 * norm(Σ)
    # 简并块内旋转：W 换成 W·Q_block（Q 只混合同奇异值方向——对 L 的
    # 列空间做块正交，Σ 不变）
    Qb = Matrix{Float64}(I, N, N); θ = 0.7
    Qb[4, 4] = cos(θ); Qb[4, 5] = -sin(θ); Qb[5, 4] = sin(θ); Qb[5, 5] = cos(θ)
    @test maximum(abs.(KTrader.principal_sqrt_root(L * Qb) .- R)) <= 1e-12 * norm(Σ)
    # 一般右正交 L·Q（Q 完全一般、所有奇异值互异）：右正交不改变
    # Σ=(LQ)(LQ)ᵀ=L·Lᵀ，主根不变——这是因子理论最一般的合法自由度
    # （非仅 diag(±1) 或简并块内旋转的特例）。
    Qg = Matrix(qr(randn(rng, N, N)).Q)
    @test maximum(abs.(KTrader.principal_sqrt_root(L * Qg) .- R)) <= 1e-12 * norm(Σ)
    # 一般右正交 + 符号混合
    @test maximum(abs.(KTrader.principal_sqrt_root(L * Qg * D) .- R)) <= 1e-12 * norm(Σ)
    # 秩亏：零奇异值保持零模（无 sqrt(eps) 假噪声）
    λz = [4.0, 2.0, 1.0, 0.5, 0.0, 0.0]
    Lz = V * Diagonal(sqrt.(λz)) * W'
    Rz = KTrader.principal_sqrt_root(Lz)
    @test maximum(abs.(Rz * Rz .- Lz * Lz')) <= 1e-12 * norm(Lz * Lz')
    @test minimum(eigvals(Symmetric(Rz))) >= -1e-12
    # 小正奇异值保留（不截秩、无阈值）
    @test minimum(eigvals(Symmetric(KTrader.principal_sqrt_root(V * Diagonal(sqrt.(λ)) * W')))) > 0

    # --- 场景级契约：同 seed 下 L 坐标差不改 X（真实 generator） ---
    # N=3：relative support 维数为 N-1=2（rank-2，非满秩）。此前 N=2
    # fixture 的 L 第二列恰为零列，翻转零列不可见——是 fixture 恰好
    # 翻到 zero 列的问题，不是 rank-1 永远抓不到 sign。N=3 使多列
    # 非零，右正交变换可观察。
    signal = exp.(cumsum(0.01 .* randn(rng, 300, 3); dims = 1))
    m = KTrader.fit_v1(signal)
    L0 = m.pred_moments.L_rel
    # 等价 L'：右正交（一般 Q + 符号混合）——Σ 同
    Q3 = Matrix(qr(randn(rng, size(L0, 2), size(L0, 2))).Q)
    Lflip = L0 * Q3 * diagm([1.0, -1.0, 1.0])
    m_flip = V1Model(m.active_indices, m.N_universe, m.e0, m.s1, m.s_macro,
        m.s_perp, m.relative_observed, m.resp, m.mu_pred, m.res_history,
        m.own_res_rows, merge(m.pred_moments, (L_rel = Lflip,)),
        m.d_posterior, m.v_forecasts, m.v_bootstrap)
    S = 40
    Xa = KTrader.generate_scenarios_v1(m; S, rng = MersenneTwister(7))
    Xb = KTrader.generate_scenarios_v1(m_flip; S, rng = MersenneTwister(7))
    @test Xa ≈ Xb atol = 1e-12 rtol = 1e-10   # 同 Σ 同 seed → 逐点一致
    # law 不变：大 S 均值一致
    Xc = KTrader.generate_scenarios_v1(m; S = 500, rng = MersenneTwister(3))
    Xd = KTrader.generate_scenarios_v1(m_flip; S = 500, rng = MersenneTwister(3))
    @test isapprox(vec(mean(Xc; dims = 1)), vec(mean(Xd; dims = 1)); atol = 1e-12)
    @test size(Xa) == size(Xb) == (S, 3)       # draw 数不变
    # RNG 后续状态一致：生成后两 rng 对象继续抽取产生相同序列
    ra, rb = MersenneTwister(7), MersenneTwister(7)
    KTrader.generate_scenarios_v1(m; S, rng = ra)
    KTrader.generate_scenarios_v1(m_flip; S, rng = rb)
    @test rand(ra, 5) == rand(rb, 5)           # 变量数/消耗一致 → 后续状态同
end
