# =============================================================================
# KTraderGate0 slow full-posterior reference — Step 8 tiny 级测试
# =============================================================================
#
# 测试对象：src/gate0/posterior.jl（sufficient_stats / s_alpha /
# log_evidence / log_prior_d035a / adaptive_quadrature_2d /
# fit_full_posterior / predict_mu / draw_mu / mu_asset /
# oof_residual_rows）。
#
# 覆盖映射（推导 §10 检查清单）：#1（A2 fixture）、#2（D-035a 下
# quadrature 应绿）、#3（S(α) 双构造 + 「不是 ridge 残差平方和」语义）、
# #4（matrix-t 边缘矩 vs dense）、#5（μ t 边缘矩双路径）、#6（先验退化
# 极限 vs Step 4 ridge）、#11（quadrature 收敛/重放/预算耗尽）、#13
# （within/between 非零可算）、#15（ν 绑定）。#7（OOF fold 隔离）归
# Step 9；#8/#9/#10 的参数恢复已在 Step 4/5/6 ridge 层覆盖，本文件加
# #10 的 posterior 层退化对照；#12 归 Step 4 已测层面；#14 为静态审查
# （Σ_R 不进 innovation——posterior.jl 无 innovation 对象，docstring 红线）。
#
# P0-8 补（2026-10-10 文档校准批次）：HalfCauchy-τ 先验表达式级断言
# （公式/归一化/_WEAKINFO 残留否定）——审计注记此前全 test/gate0/ 无
# HalfCauchy 直接数值断言；新增断言未运行（运行归 DevOps）。
#
# 纪律：合成数据固定 seed；不运行测试（运行归 DevOps）；断言全部与
# 回测收益无关（开发守则 §24）；dense 对照走与生产实现**独立**的公式
# 路径（evidence：quadgk 一维积分 + 直接构造 C 的 det，不经 Sylvester；
# matrix-t 矩：N=1/P=1 的显式一维积分）。

# 缺 using 修复（Wave 3 验证轮）：测试体 L261/L263 使用 Statistics 的
# mean/cov（原 using 面遗漏——与 Wave 2 modes_tests 同模式）；L80 使用
# SpecialFunctions 的 loggamma（非 Base 名——原遗漏，Project.toml 同轮
# 补依赖声明）。
using Test, Random, LinearAlgebra, Statistics, QuadGK, Distributions,
    SpecialFunctions

module Gate0PosteriorEnv
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "geometry.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "modes.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "response.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "posterior.jl"))
end
using .Gate0PosteriorEnv

# 错误文本断言 helper（@test_throws 只匹配类型，消息用 occursin）
_throws_msg(f, needle) = try
    f()
    false
catch e
    e isa ErrorException && occursin(needle, sprint(showerror, e))
end

@testset "gate0 Step 8: full-posterior reference" begin
    SEED = 20261009

    # -------------------------------------------------------------------
    # 1. S(α) 双构造一致性 + 「不是 ridge 残差平方和」语义（§10 #3）
    # -------------------------------------------------------------------
    @testset "S(alpha) two constructions" begin
        rng = MersenneTwister(SEED)
        N, P, n = 2, 3, 20
        X = randn(rng, n, P)
        Y = randn(rng, n, N)
        st = sufficient_stats(X, Y)
        # 独立构造路径：YᵀC⁻¹Y，C⁻¹ = I − X(Sxx+Λ)⁻¹Xᵀ（显式 Woodbury）
        for alpha in (1e-2, 1.0, 1e2)
            lam = fill(alpha, P)
            S_closed = s_alpha(st, lam)                       # 公式 1（生产路径）
            A = st.Sxx + Diagonal(lam)
            Cinv = Matrix(I, n, n) .- X * (A \ X')            # 公式 2（测试内独立）
            S_direct = Y' * Cinv * Y
            @test isapprox(Matrix(S_closed), S_direct, atol = 1e-10)
            # 语义区分（§4.2 实现警告）：S(α) ≠ ridge 残差平方和
            # (C⁻¹Y)ᵀ(C⁻¹Y)。断言限定 α ≥ 0.1（Wave 3 验证轮修正）：
            # docstring 钉死「二者仅在 α→0 的 OLS 投影极限重合」——
            # 小 α（1e-2）下区分度数学上不足（实测差 7.3e-4 <
            # 1e-3·‖Syy‖），语义区分在可区分的 α 上验证（1.0/1e2
            # 实测通过；断言语义与阈值不变）。
            if alpha >= 0.1
                R = Cinv * Y
                ridge_ss = R' * R
                @test norm(Matrix(S_closed) .- ridge_ss) > 1e-3 * norm(st.Syy)   # 中等 α 可区分
            end
        end
    end

    # -------------------------------------------------------------------
    # 2. marginal evidence vs dense 数值积分（N=1；§10 #2/#4 部分）
    # -------------------------------------------------------------------
    @testset "evidence vs dense (N=1)" begin
        rng = MersenneTwister(SEED + 1)
        n, P = 8, 2
        X = randn(rng, n, P)
        y = randn(rng, n)
        Y = reshape(y, :, 1)
        st = sufficient_stats(X, Y)
        # 省略的常数 K(n,1)（RP §4.3；只影响 log 偏移不影响形状）。
        # Manager 裁决（Wave 3 执行轮）：对齐 dense 直接积分（Jeffreys σ
        # 先验）的完整常数 2^{n/2−1}——原抄写（RP §4.3 的 K 含 2^{n/2}）
        # 多一个因子 2；dense 实测为准（两 α 偏移逐位一致 = logK−log2）。
        # RP 文档勘误由 Manager 登记。
        logK = (-(n / 2) * log(2π)) + (n / 2 - 1) * log(2.0) + loggamma(n / 2)
        for alpha in (0.3, 3.0)
            lam = fill(alpha, P)
            S = s_alpha(st, lam)[1, 1]
            C = Matrix(I, n, n) + X * (Diagonal(lam) \ X')    # 直接构造 C（不经 Sylvester）
            logdet_C = logdet(C)
            # dense：∫ σ^{-(n+1)}·exp(−S/(2σ²)) dσ（t = log σ 换元，quadgk）
            Sval = S
            integral = quadgk(t -> exp(-n * t - Sval * exp(-2t) / 2), -Inf, Inf; atol = 1e-12)[1]
            log_dense = -(n / 2) * log(2π) - logdet_C / 2 + log(integral)
            log_closed = log_evidence(st, lam)
            @test isapprox(log_dense - log_closed, logK, atol = 1e-8)   # 偏移 = K 且与 α 无关
        end
    end

    # -------------------------------------------------------------------
    # 3. matrix-t 边缘矩 vs dense（N=1、P=1；§10 #4）
    # -------------------------------------------------------------------
    @testset "matrix-t moments vs dense (N=1,P=1)" begin
        rng = MersenneTwister(SEED + 2)
        n = 8
        X = randn(rng, n, 1)
        y = randn(rng, n)
        alpha = 0.7
        st = sufficient_stats(X, reshape(y, :, 1))
        lam = fill(alpha, 1)
        V = 1.0 / (st.Sxx[1, 1] + alpha)
        b_hat = st.Sxy[1, 1] * V
        S = s_alpha(st, lam)[1, 1]
        # dense：核 [S + (b−b̂)²/V]^{−(n+P)/2}（一维 quadgk）
        kern(b) = (S + (b - b_hat)^2 / V)^(-(n + 1) / 2)
        Z0 = quadgk(kern, -Inf, Inf; atol = 1e-14)[1]
        Eb = quadgk(b -> b * kern(b), -Inf, Inf; atol = 1e-14)[1] / Z0
        Eb2 = quadgk(b -> b^2 * kern(b), -Inf, Inf; atol = 1e-14)[1] / Z0
        @test isapprox(Eb, b_hat, atol = 1e-9)                    # E[B] = B̂
        @test isapprox(Eb2 - b_hat^2, (S / (n - 2)) * V, rtol = 1e-9)  # Cov(B) = E[Σ]·V
    end

    # -------------------------------------------------------------------
    # 4. μ 的 t 边缘矩（双路径）+ 点质量 predict_mu（§10 #5）
    # -------------------------------------------------------------------
    @testset "mu t-edge moments" begin
        rng = MersenneTwister(SEED + 3)
        n, N = 8, 1
        P = 2
        X = randn(rng, n, P)
        y = randn(rng, n)
        Y = reshape(y, :, 1)
        alpha = 0.7
        post = fit_full_posterior(X, Y; prior = :point_mass, alpha_fixed = alpha)
        xt = randn(rng, P)
        pm = predict_mu(post, xt)
        c0 = post.conditional[1]
        c = dot(xt, c0.V_fact \ xt)
        # 路径 A（t 公式）：均值 = B̂x̃、within = (c/(ν−2))S
        @test isapprox(pm.mu, c0.B_hat * xt, atol = 1e-10)
        @test isapprox(pm.within[1, 1], (c / (post.nu - 2)) * c0.S_alpha[1, 1], atol = 1e-10)
        # 路径 B（独立推导：Cov(μ) = E[Σ]·c，E[Σ] = S/(n−N−1)）
        @test isapprox(pm.within[1, 1], (c0.S_alpha[1, 1] / (n - 2)) * c, atol = 1e-10)
        # 单节点：between = 0、within_computed = true、ν 绑定
        @test maximum(abs.(pm.between)) < 1e-12
        @test pm.within_computed
        @test post.nu == n + 1 - N && post.nu != n - N           # ν 绑定（G7；拒绝 n−N）
    end

    # -------------------------------------------------------------------
    # 5. 先验退化极限 vs Step 4 ridge（§10 #6）
    # -------------------------------------------------------------------
    @testset "point-mass limit vs Step 4 ridge" begin
        rng = MersenneTwister(SEED + 4)
        N = 2
        P = 14 * N
        n = 60
        X = randn(rng, n, P)
        G = 0.4 .* randn(rng, N, P)
        Y = X * G'
        alpha = 0.5
        post = fit_full_posterior(X, Y; prior = :point_mass, alpha_fixed = alpha)
        ridge = fit_full_block_ridge(X, Y; alpha = alpha)
        # Λ=αI（无 DC 单 group）⇒ B̂ 与 Step 4 ridge 解同式异路（充分统计 vs 直接 X）
        @test isapprox(post.conditional[1].B_hat, ridge.B, atol = 1e-10)
        @test post.prior == :point_mass && length(post.alpha_weights) == 1
    end

    # -------------------------------------------------------------------
    # 6. propriety gate（§10 #1：A2 fixture）
    # -------------------------------------------------------------------
    @testset "propriety gate (A2)" begin
        rng = MersenneTwister(SEED + 5)
        # n < N
        X1 = randn(rng, 2, 3)
        Y1 = randn(rng, 2, 3)
        @test _throws_msg(() -> fit_full_posterior(X1, Y1), "posterior improper")
        # 退化 Y（列线性相关 ⇒ rank(Syy) < N）
        n, N, P = 30, 3, 4
        X2 = randn(rng, n, P)
        Y2 = randn(rng, n, N)
        Y2[:, 3] = 2.0 .* Y2[:, 1]
        @test _throws_msg(() -> fit_full_posterior(X2, Y2), "posterior improper")
    end

    # -------------------------------------------------------------------
    # 7. quadrature：收敛 / 确定性重放 / 预算耗尽（§10 #11；#2 应绿）
    # -------------------------------------------------------------------
    @testset "quadrature convergence & determinism" begin
        rng = MersenneTwister(SEED + 6)
        N, P, n = 2, 3, 30
        X = randn(rng, n, P)
        G = 0.5 .* randn(rng, N, P)
        Y = X * G' .+ 0.3 .* randn(rng, n, N)          # 信号 + 噪声（evidence 有峰）
        xt = randn(rng, P)
        # (a) D-035a 修正先验下 quadrature 应绿（§10 #2：若红是实现 bug）
        post1 = fit_full_posterior(X, Y; tol = 1e-4)
        post2 = fit_full_posterior(X, Y; tol = 1e-4)
        @test post1.alpha_nodes == post2.alpha_nodes && post1.alpha_weights == post2.alpha_weights  # 逐位重放
        # (b) 网格细化收敛（tol 收紧时 μ 矩稳定）
        post3 = fit_full_posterior(X, Y; tol = 1e-6)
        pm1 = predict_mu(post1, xt)
        pm3 = predict_mu(post3, xt)
        @test isapprox(pm1.mu, pm3.mu, atol = 1e-4) && isapprox(pm1.cov, pm3.cov, atol = 1e-3)
        # (c) within/between 非零可算（§10 #13：between 是 plug-in 旧实现恒 0 的对照面）
        @test norm(pm1.between) > 0 && pm1.within_computed
        @test isapprox(pm1.cov, pm1.within .+ pm1.between, atol = 1e-12)
        # (d) ν 绑定（多节点 posterior）
        @test all(c -> c.nu == n + 1 - N, post1.conditional) && post1.nu == n + 1 - N
        # (e) 预算耗尽 fail loudly（D-067 文本）
        @test _throws_msg(() -> fit_full_posterior(X, Y; tol = 1e-14, max_cells = 2),
                          "Numerical integration did not converge")
    end

    # -------------------------------------------------------------------
    # 8. OOF 残差行（单 fold 全数据版接口）与 μ_asset
    # -------------------------------------------------------------------
    @testset "residual rows & mu_asset" begin
        rng = MersenneTwister(SEED + 7)
        N, P, n = 3, 5, 40
        X = randn(rng, n, P)
        Y = randn(rng, n, N)
        E = domain_mode_basis(N)
        post = fit_full_posterior(X, Y; prior = :point_mass, alpha_fixed = 0.5)
        # 手工：ε̃ = E·(y_s − B̂·x_s)（行 s；含 b₀ 扣除由 B̂·x 的 DC 列承担）
        Bmix = sum(c.weight .* c.B_hat for c in post.conditional)
        manual = (Y .- X * Bmix') * E'
        @test isapprox(oof_residual_rows(post, E), manual, atol = 1e-12)
        xt = randn(rng, P)
        @test isapprox(mu_asset(post, xt, E), E * (post.conditional[1].B_hat * xt), atol = 1e-10)
    end

    # -------------------------------------------------------------------
    # 9. DC 形态退化对照（§10 #10 弱版：alpha_fixed=(α,α) ⇒ Λ=αI 与 Step 5 对齐）
    # -------------------------------------------------------------------
    @testset "DC point-mass vs Step 5 ridge" begin
        rng = MersenneTwister(SEED + 8)
        N = 2
        P = 1 + 14 * N
        n = 60
        Xdc = hcat(ones(n), randn(rng, n, 14 * N))
        b0 = vcat(0.05, 0.02 .* randn(rng, N - 1))
        G = 0.3 .* randn(rng, N, 14 * N)
        Y = Xdc * hcat(b0, G)'
        post = fit_full_posterior(Xdc, Y; prior = :point_mass, alpha_fixed = (0.5, 0.5))
        ridge = fit_full_block_ridge(Xdc, Y; alpha = 0.5)
        @test post.has_a0
        @test isapprox(post.conditional[1].B_hat, ridge.B, atol = 1e-10)
    end

    # -------------------------------------------------------------------
    # 10. draw_mu：t 抽样矩 + 同 seed 重放（裁决 F）
    # -------------------------------------------------------------------
    @testset "draw_mu sampling" begin
        rng = MersenneTwister(SEED + 9)
        N, P, n = 2, 3, 40
        X = randn(rng, n, P)
        Y = randn(rng, n, N)
        xt = randn(rng, P)
        post = fit_full_posterior(X, Y; prior = :point_mass, alpha_fixed = 0.5)
        c0 = post.conditional[1]
        mu_hat = c0.B_hat * xt
        ck = dot(xt, c0.V_fact \ xt)
        scale = (ck / (post.nu - 2)) .* c0.S_alpha          # t 协方差（ν>2）
        # rng 共享修复（Wave 3 验证轮）：原 comprehension 每次迭代重建
        # MersenneTwister(42)——同 seed 同流使 2000 个 draw 全同、样本
        # 协方差恒为 0（innovation_tests.jl:369 的交付者注记正是此坑）。
        # 提出循环外共享，恢复「2000 个独立抽样做矩估计」的测试意图；
        # 断言语义不变（L 下方同 seed 逐位重放断言仍用独立构造的 rng）。
        rng42 = MersenneTwister(42)
        draws = [draw_mu(post, xt, rng42) for _ in 1:2000]
        mhat = mean(draws)                                   # 2000 抽样本均值（弱断言）
        @test norm(mhat .- mu_hat) < 0.15 * sqrt(maximum(diag(scale)))
        vhat = cov(stack(draws; dims = 1))
        @test norm(vhat .- scale) < 0.6 * norm(scale)        # 样本协方差量级（弱断言）
        @test draw_mu(post, xt, MersenneTwister(42)) == draw_mu(post, xt, MersenneTwister(42))  # 同 seed 逐位
    end

    # -------------------------------------------------------------------
    # 11. fail loudly（维度 / NaN / 参数域）
    # -------------------------------------------------------------------
    @testset "fail loudly" begin
        rng = MersenneTwister(SEED + 10)
        X = randn(rng, 20, 3)
        Y = randn(rng, 20, 2)
        @test_throws DimensionMismatch sufficient_stats(X, randn(rng, 19, 2))
        Xb = copy(X); Xb[3, 2] = NaN
        @test_throws DomainError sufficient_stats(Xb, Y)
        Yb = copy(Y); Yb[2, 1] = Inf
        @test_throws DomainError fit_full_posterior(X, Yb)
        post = fit_full_posterior(X, Y; prior = :point_mass, alpha_fixed = 0.5)
        @test_throws DimensionMismatch predict_mu(post, randn(rng, 2))
        @test_throws ArgumentError fit_full_posterior(X, Y; prior = :foo)
        @test_throws ArgumentError fit_full_posterior(X, Y; alpha_fixed = 0.5)   # 非 point_mass 拒绝
        # DC 形态必须二元组（防静默忽略 α₀）。fixture 列数修正（Wave 3
        # 验证轮）：原 randn(20, 27) 使总列数 1+27=28 恰等于 14N（N=2）
        # ——被 fit_full_posterior 当作无 DC 形态（P=14N），标量
        # alpha_fixed 合法、不抛错（断言「No exception thrown」红）。DC
        # 形态应为 1+14N=29 列：动态列 28。
        Xdc = hcat(ones(20), randn(rng, 20, 28))              # N=2 ⇒ P=29
        @test_throws ArgumentError fit_full_posterior(Xdc, randn(rng, 20, 2);
            prior = :point_mass, alpha_fixed = 0.5)
        @test_throws DimensionMismatch mu_asset(post, randn(rng, 3), randn(3, 3))
    end

    # -------------------------------------------------------------------
    # P0-8 补：proper HalfCauchy-τ 先验（表达式级断言）+ _WEAKINFO 残留否定
    # -------------------------------------------------------------------
    # 2026-10-10 文档校准批次新增（审计注记：此前全 test/gate0/ 无
    # HalfCauchy 直接数值断言）。语义：τ~HalfCauchy(0,1) ⇒
    # p(α)∝1/[√α(1+α)] ⇒ log 坐标密度 log p_u(u) = 0.5u − log(1+eᵘ)
    # （u = log α）；旧 improper 形式 ∝1/[α(1+α)] 的 log 密度 = u − log(1+eᵘ)。
    @testset "P0-8 HalfCauchy-tau prior (no weakinfo remnants)" begin
        # (a) 与 HalfCauchy-τ 公式逐点对照（up 坐标取 0：贡献 −log 2）
        for u in (-4.0, -1.0, 0.0, 2.0, 6.0)
            @test isapprox(log_prior_d035a(u, 0.0),
                           (0.5u - log1p(exp(u))) - log(2.0), atol = 1e-14)
        end
        # (b) 与旧 improper 形式（系数 1）可区分：差 = u/2（|u|=4 时 = 2）
        for u in (-4.0, 4.0)
            @test abs(log_prior_d035a(u, 0.0)
                      - ((u - log1p(exp(u))) - log(2.0))) > 1.0
        end
        # (c) proper 归一化：单坐标密度 ∫ e^{0.5u−log(1+eᵘ)} du = π
        lp1(u) = log_prior_d035a(u, 0.0) + log(2.0)      # = 0.5u − log(1+eᵘ)
        I1 = quadgk(u -> exp(lp1(u)), -Inf, Inf; atol = 1e-9)[1]
        @test isapprox(I1, π, rtol = 1e-7)
        # (d) 弱信息特例机器（回测 fixture 标定阈值 3/8）不得余留已定义名
        @test !isdefined(Gate0PosteriorEnv, :_WEAKINFO_GAP)
        @test !isdefined(Gate0PosteriorEnv, :_WEAKINFO_RATE)
        @test !isdefined(Gate0PosteriorEnv, :_WEAKINFO_SAFETY)
    end

    # -------------------------------------------------------------------
    # G-L 归一化溢出回归（真数据首日根因；2026-10-10）
    # -------------------------------------------------------------------
    # 根因（证据 archive/evidence/gate0_reallimit_firstday_20261010/）：
    # 旧 m 只取初始 cell 中心 max——G-L 节点（±1/√3、±√(3/5)）偏离中心，
    # 陡峭 logf 下节点可高出 m 数百（真数据大 P 实测 +502.7），
    # exp(fv−m) 溢出（阈值 exp(709.78)）→ Inf/NaN。本 fixture 构造同一
    # 机制：峰 A=750（>709.78）位于 cell (i=3,j=3) 的 2 点 G-L 节点
    # (1.25/√3, 1.25/√3)≈(0.7217,0.7217)；16 个 cell 中心距峰 ≥0.72
    # （最大中心贡献 ≈+8.6，远低于节点 +750）→ 旧机制 m≈base+8.6、
    # 节点 exp(741.4)=Inf，初始 16 cell 的 val/err 即 Inf/NaN → 旧实现
    # 必红（rel=NaN 或 Inf 触发 did-not-converge）。修复（m 覆盖全部
    # 求值节点）后应正常收敛。tol=1e-2 为 fixture 级成本控制（收敛性/
    # 有限性/权重和为 1 的语义断言不变；与 backtest_tests 的 fixture 降
    # 级同纪律）；m_ref=base+A 模拟调用方粗扫描 mode 参考（tail 证书
    # 契约：refval 需接近真峰，否则边界判据按设计红）。
    @testset "G-L normalization overflow regression" begin
        base = -1.0e6
        A = 750.0
        σ = 0.25
        u0p = 1.25 / sqrt(3.0)                 # cell (i=3,j=3) 的 +g2 节点
        upp = 1.25 / sqrt(3.0)
        logf_fix(u0, up) = base + A * exp(-((u0 - u0p)^2 + (up - upp)^2) / (2σ^2))
        # 构造性声明：峰恰在节点上；全部中心远低于峰（旧 m 不覆盖节点）
        @test logf_fix(u0p, upp) == base + A
        @test maximum(logf_fix(c, c) for c in (-3.75, -1.25, 1.25, 3.75)) < base + 10.0
        res = adaptive_quadrature_2d(logf_fix, (-5.0, 5.0), (-5.0, 5.0);
                                      tol = 1e-2, m_ref = base + A)
        @test isfinite(res.logZ)
        @test res.rel_err <= 1e-2
        @test res.n_cells < 8192                # 收敛而非撞预算
        @test all(isfinite, res.weights)
        @test isapprox(sum(res.weights), 1.0, atol = 1e-12)
        @test abs(res.logZ - (base + A)) < 50.0 # 峰主导量级（解析差 ~O(log A)）
        # 确定性重放（同输入逐位；无 RNG）
        res2 = adaptive_quadrature_2d(logf_fix, (-5.0, 5.0), (-5.0, 5.0);
                                       tol = 1e-2, m_ref = base + A)
        @test res.logZ == res2.logZ && res.weights == res2.weights
    end

    # -------------------------------------------------------------------
    # R4'''：Sα 鲁棒 logdet（eigen + floor）+ 防御软包装/防护（F1/F2；2026-10-10）
    # -------------------------------------------------------------------
    # 语义锚（docs/EARLY_ROW_FIT_FAILURE_DIAGNOSIS.md §8）：
    # - Sα 理论 PSD；F2 实测负特征值为 [-2.65e-12, -2.70e-13]（数值噪声级）
    #   → eigen 鲁棒 logdet：λ < -neg_tol 结构性负值 fail loudly；否则 floor。
    # - 求值点 PosDef（其它分解）→ 防御软包装 -Inf + 诊断；证书/refval 非有限
    #   → fail loudly；正常档零改变（eigen vs cholesky 差 ~roundoff 级）。
    # - n=2000/P=911 真实早期行不再失败由 DevOps 探针验证（本 testset 为语义锚）。
    @testset "R4''' robust PSD logdet & guards" begin
        # (a) 鲁棒 logdet：正常 PSD 与 cholesky 路径 roundoff 级一致
        Sbase = Symmetric([2.0 0.3; 0.3 1.5])
        @test isapprox(Gate0PosteriorEnv._robust_logdet_psd(Sbase),
                       logdet(cholesky(Sbase)), rtol = 1e-12)
        # 负噪声特征值（F2 量级）→ floor 后有限、不抛
        Snoise = Symmetric(diagm([1.0, -2.65e-12]))
        ld = Gate0PosteriorEnv._robust_logdet_psd(Snoise)
        @test isfinite(ld)
        @test isapprox(ld, log(1.0) + log(Gate0PosteriorEnv._PSD_FLOOR_REL), rtol = 1e-10)
        # 结构性负值 → fail loudly
        Sbad = Symmetric(diagm([1.0, -0.5]))
        @test _throws_msg(() -> Gate0PosteriorEnv._robust_logdet_psd(Sbad), "R4 fail-loud")
        # 确定性重放（同输入逐位；无 RNG）
        @test Gate0PosteriorEnv._robust_logdet_psd(Snoise) == Gate0PosteriorEnv._robust_logdet_psd(Snoise)
        # (b) 防御软包装：PosDef → -Inf + 记录；非 PosDef 不吞
        rawthrow = (lam) -> throw(PosDefException(2))
        rawok = (lam) -> 7.5
        dg = Gate0PosteriorEnv.PosDefDiagnostics()
        @test Gate0PosteriorEnv._soft_logf(rawthrow, [1.0, 1.0], -0.5, -12.0, -12.0, dg) == -Inf
        @test dg.attempts == 1 && dg.failures == 1
        @test dg.points == [(-12.0, -12.0)]
        @test Gate0PosteriorEnv._soft_logf(rawok, [1.0, 1.0], -0.5, 0.0, 0.0, dg) == 7.0
        @test dg.attempts == 2 && dg.failures == 1
        rawerr = (lam) -> error("boom")
        @test_throws ErrorException Gate0PosteriorEnv._soft_logf(rawerr, [1.0], 0.0, 0.0, 0.0, dg)
        dg2 = Gate0PosteriorEnv.PosDefDiagnostics()
        for k in 1:(Gate0PosteriorEnv._POSDEF_POINT_CAP + 5)
            Gate0PosteriorEnv._soft_logf(rawthrow, [1.0], 0.0, Float64(k), 0.0, dg2)
        end
        @test dg2.failures == Gate0PosteriorEnv._POSDEF_POINT_CAP + 5
        @test length(dg2.points) == Gate0PosteriorEnv._POSDEF_POINT_CAP
        # (c) 中域/占比负控 → fail loudly；远角不触发
        dg3 = Gate0PosteriorEnv.PosDefDiagnostics()
        dg3.attempts = 10; dg3.failures = 1
        push!(dg3.points, (0.5, 0.0))
        @test _throws_msg(() -> Gate0PosteriorEnv._assert_posdef_safe(dg3, 0.0, 0.0), "R4 fail-loud")
        dg4 = Gate0PosteriorEnv.PosDefDiagnostics()
        dg4.attempts = 1000; dg4.failures = 2
        push!(dg4.points, (-12.0, -12.0))
        @test Gate0PosteriorEnv._assert_posdef_safe(dg4, 0.0, 0.0) === nothing
        dg5 = Gate0PosteriorEnv.PosDefDiagnostics()
        dg5.attempts = 10; dg5.failures = 5
        push!(dg5.points, (-12.0, -12.0))
        @test _throws_msg(() -> Gate0PosteriorEnv._assert_posdef_safe(dg5, 0.0, 0.0), "R4 fail-loud")
        # (d) 证书/refval/全失败防护（2D；模拟非有限求值）
        logf_badcert = (u0, up) -> (u0 <= -5.0 + 1e-12 ? -Inf : -(u0^2 + up^2))
        @test _throws_msg(() -> adaptive_quadrature_2d(logf_badcert, (-5.0, 5.0), (-5.0, 5.0);
                                                       tol = 1e-4), "R4 fail-loud")
        logf_badref = (u0, up) -> (u0 == -3.0 && up == -3.0 ? -Inf : -(u0^2 + up^2))
        @test _throws_msg(() -> adaptive_quadrature_2d(logf_badref, (-5.0, 5.0), (-5.0, 5.0);
                                                       tol = 1e-4), "R4 fail-loud")
        logf_allbad = (u0, up) -> -Inf
        @test _throws_msg(() -> adaptive_quadrature_2d(logf_allbad, (-5.0, 5.0), (-5.0, 5.0);
                                                       tol = 1e-4), "R4 fail-loud")
        # (e) 正常档零改变：软包装不触发、诊断零失败、确定性重放
        rng = MersenneTwister(SEED + 11)
        N2, P2, n2 = 2, 3, 30
        X2 = randn(rng, n2, P2)
        G2 = 0.5 .* randn(rng, N2, P2)
        Y2 = X2 * G2' .+ 0.3 .* randn(rng, n2, N2)
        dgok = Gate0PosteriorEnv.PosDefDiagnostics()
        postA = fit_full_posterior(X2, Y2; tol = 1e-4, posdef_diag = dgok)
        postB = fit_full_posterior(X2, Y2; tol = 1e-4)
        @test postA.alpha_nodes == postB.alpha_nodes && postA.alpha_weights == postB.alpha_weights
        @test dgok.failures == 0 && dgok.attempts > 0
    end
end
