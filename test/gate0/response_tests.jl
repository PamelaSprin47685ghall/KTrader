# =============================================================================
# KTraderGate0 full block response reference — Step 4 tiny 级测试
# =============================================================================
#
# 测试对象：src/gate0/response.jl（build_mode_problem / fit_full_block_ridge /
# predict_mode / block_views——固定 ridge 版 full block response reference）。
#
# 纪律：
# - tiny 级（D-089 测试升级阶梯：static → tiny analytic → …）：合成数据、
#   固定 seed（MersenneTwister(20261009)）、无外部依赖、无 I/O；
# - 本任务不运行测试（Engineer 不执行命令；运行验证由 Manager 安排
#   DevOps 受控执行，≤60s/RSS 护栏）；
# - 断言对象全部与回测收益无关（开发守则 §24）；
# - 加载方式：经临时 module include 三个裸文件（主 module KTraderGate0
#   由后续施工步骤建立——裁决 H1/H2；此处不依赖旧线 KTrader）。
#
# D-092（核心测试，裁决书 §33）：known cross-mode world——人工 G* 的两个
# cross 块（G_m⊥：m 输出 ← relative 输入；G_⊥m：q 输出 ← macro 输入）均
# 非零；full-block 拟合必须恢复（无噪声 atol 1e-8）；block-diagonal 拟合
# （把 cross 块强制为零的分离回归——旧线两独立回归的等价物）残差显著
# 更大，证明 D-025 修复的正是被旧实现删除的 block。

using Test, Random, LinearAlgebra

module Gate0ResponseEnv
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "geometry.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "modes.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "response.jl"))
end
using .Gate0ResponseEnv

@testset "gate0 Step 4: full block response reference (fixed ridge)" begin
    SEED = 20261009

    # -------------------------------------------------------------------
    # 1. D-092 known cross-mode world（核心：统一回归代数 + cross 块恢复）
    #    直接合成 design X 与人工 G*——隔离验证回归层代数（pipeline 组装
    #    由第 2 组覆盖）。
    # -------------------------------------------------------------------
    @testset "D-092 cross-mode recovery" begin
        rng = MersenneTwister(SEED)
        N = 4
        P = 14 * N
        n = 300
        X = randn(rng, n, P)                       # 合成 design（固定 seed）

        # 人工 G*：四块全部非零；两个 cross 块是 D-092 的断言对象。
        # 块语义（response.jl 钉死）：行块 = 输出 mode（行 1 = m、行 2:N = q）、
        # 列块 = 输入 mode（列 1:14 = B_m、列 15:P = B_⊥）。
        G = zeros(N, P)
        G[1, 1:14] .= 0.30 .* randn(rng, 14)                    # G_mm（m ← macro path）
        G[1, 15:P] .= 0.50 .* randn(rng, 14 * (N - 1))          # G_m⊥ ≠ 0（m ← relative path：relative path 预测整体市场）
        G[2:N, 1:14] .= 0.40 .* randn(rng, (N - 1), 14)         # G_⊥m ≠ 0（q ← macro path：macro path 预测横截面 rotation）
        G[2:N, 15:P] .= 0.30 .* randn(rng, (N - 1), 14 * (N - 1))  # G_⊥⊥（q ← relative path）

        # 构造健全性：两个 cross 块确实携带显著信号
        @test norm(G[1, 15:P]) > 1.0
        @test norm(G[2:N, 1:14]) > 1.0

        # --- (a) 无噪声恢复（atol 1e-8）---
        Y = X * G'
        fit = fit_full_block_ridge(X, Y; alpha = 1e-10)
        @test isapprox(fit.B, G, atol = 1e-8)                   # 全矩阵恢复
        @test isapprox(vec(fit.B_mperp), G[1, 15:P], atol = 1e-8) # cross 块 1（D-092 点名；view 1×P' → vec 同形）
        @test isapprox(Matrix(fit.B_perpm), G[2:N, 1:14], atol = 1e-8)  # cross 块 2（D-092 点名）

        # --- (b) 小噪声：精确线性收敛语义 ---
        # 同一噪声场 Z 缩放 σ ⇒ B̂(σ) − G* 精确 ∝ σ（线性代数事实，
        # 固定 rng 保证逐位成立）——err(2σ) ≈ 2·err(σ)。
        Z = randn(rng, n, N)
        fit01 = fit_full_block_ridge(X, Y .+ 0.01 .* Z; alpha = 1e-10)
        fit02 = fit_full_block_ridge(X, Y .+ 0.02 .* Z; alpha = 1e-10)
        err01 = norm(fit01.B .- G)
        err02 = norm(fit02.B .- G)
        @test err01 < 0.05                                       # 噪声量级一致（弱绝对界）
        @test isapprox(err02, 2 * err01, rtol = 1e-6)            # 线性收敛语义

        # --- (c) block-diagonal 对照（旧实现等价物：cross 块强制为零）---
        # 分离回归：m 输出只用 B_m 列、q 输出只用 B_⊥ 列——即
        # G_m⊥ = 0、G_⊥m = 0 的拟合（旧线 macro/relative 两独立回归的
        # 等价约束形态）。残差必须显著更大（D-092 反例证据）。
        alpha_bd = 1e-10
        Xm = X[:, 1:14]
        Xp = X[:, 15:P]
        Bm_hat = (Xm' * Xm + alpha_bd * I) \ (Xm' * Y[:, 1:1])       # 14×1
        Bq_hat = (Xp' * Xp + alpha_bd * I) \ (Xp' * Y[:, 2:N])       # 14(N−1)×(N−1)
        Y_bd = hcat(Xm * Bm_hat, Xp * Bq_hat)
        resid_full = norm(Y .- X * fit.B')
        resid_bd = norm(Y .- Y_bd)
        @test resid_full < 1e-6                                   # full-block：无噪声下残差 ≈ 0
        @test resid_bd > 0.2 * norm(Y)                            # block-diagonal：cross 信号被删，残差显著
    end

    # -------------------------------------------------------------------
    # 2. design/target 代数验证（完整 pipeline：build_mode_problem）
    # -------------------------------------------------------------------
    @testset "design/target algebra (pipeline)" begin
        rng = MersenneTwister(SEED + 1)
        N = 4
        T = 320
        returns = 0.01 .* randn(rng, T, N)
        observed = trues(T, N)
        s1 = 0.01 .+ 0.02 .* rand(rng, N)
        E = domain_mode_basis(N)

        mp = build_mode_problem(returns, observed, s1, E)

        # 维度断言（14N 列、N 输出、T−1 配对行）
        @test size(mp.X) == (T - 1, 14 * N)
        @test size(mp.Y) == (T - 1, N)
        @test size(mp.X_full) == (T, 14 * N)
        @test size(mp.y_modes) == (T, N)
        @test size(mp.X_m) == (T,) && size(mp.X_rel) == (T, N)

        # 列布局与 mode_basis_design 输出一致（独立重调对照，逐位相等）
        des = mode_basis_design(mp.X_m, mp.X_rel, mp.Q, mp.s_m, mp.s_perp)
        @test mp.X_full[:, 1:14] == des.B_m
        @test mp.X_full[:, 15:end] == des.B_perp
        @test mp.X_full == hcat(mp.B_m, mp.B_perp)

        # 时序对齐（钉死语义的断言对象）：Y 行 s = mode_field(returns[s+1]) 的 y
        for s in (1, 5, 100, T - 1)
            mf = mode_field(view(returns, s + 1, :), view(observed, s + 1, :), s1, E)
            @test isapprox(mp.Y[s, :], mf.y, atol = 1e-12)
        end

        # 前缀不足行为 0（path_basis_1d：t < 2τ+1，最小 τ=2 ⇒ t < 5）
        @test all(iszero, mp.X_full[3, :])

        # 四块切分边界（块维度断言）
        fit = fit_full_block_ridge(mp.X, mp.Y; alpha = 1.0)
        @test size(fit.B) == (N, 14 * N)
        @test size(fit.B_mm) == (1, 14)
        @test size(fit.B_mperp) == (1, 14 * (N - 1))
        @test size(fit.B_perpm) == (N - 1, 14)
        @test size(fit.B_perpperp) == (N - 1, 14 * (N - 1))

        # view 语义：四块共享 B 的内存（非 copy）
        @test fit.B_mm[1, 3] == fit.B[1, 3]
        @test fit.B_mperp[1, 2] == fit.B[1, 16]
        @test fit.B_perpm[2, 5] == fit.B[3, 5]
        @test fit.B_perpperp[3, 7] == fit.B[4, 21]

        # predict_mode：ŷ = B·x_t
        x_now = mp.X_full[end, :]
        yhat = predict_mode(fit.B, x_now)
        @test length(yhat) == N
        @test isapprox(yhat, fit.B * x_now, atol = 1e-12)
    end

    # -------------------------------------------------------------------
    # 3. 零响应世界（Y 与 X 独立 → ridge 系数收缩；弱断言——精确收缩
    #    语义归 Step 5/8 的 posterior 层）
    # -------------------------------------------------------------------
    @testset "zero-response world" begin
        rng = MersenneTwister(SEED + 2)
        N = 4
        P = 14 * N
        n = 300
        X = randn(rng, n, P)
        Y0 = randn(rng, n, N)                     # 与 X 独立（零相关构造）
        Gsig = 0.4 .* randn(rng, N, P)            # 有信号对照的生成算子
        Ysig = X * Gsig'

        fit0 = fit_full_block_ridge(X, Y0; alpha = 1.0)
        fitsig = fit_full_block_ridge(X, Ysig; alpha = 1.0)
        @test norm(fit0.B) < 0.5 * norm(fitsig.B)  # 零响应系数显著小于有信号情形
        @test norm(fit0.B) < 3.5                   # 弱绝对阈值（ridge 下非零但小）
    end

    # -------------------------------------------------------------------
    # 4. permutation 协变（资产置换 π：asset 空间预测 π 协变——field 层面
    #    验证；mode 坐标 B 本身不协变是 gauge 坐标的语义，非 bug）
    # -------------------------------------------------------------------
    @testset "permutation covariance (asset field)" begin
        rng = MersenneTwister(SEED + 3)
        N = 4
        T = 320
        returns = 0.01 .* randn(rng, T, N)
        observed = trues(T, N)
        s1 = 0.01 .+ 0.02 .* rand(rng, N)
        E = domain_mode_basis(N)

        mp = build_mode_problem(returns, observed, s1, E)
        fit = fit_full_block_ridge(mp.X, mp.Y; alpha = 1.0)

        perm = randperm(rng, N)
        mp_p = build_mode_problem(returns[:, perm], trues(T, N), s1[perm],
                                  domain_mode_basis(N))
        fit_p = fit_full_block_ridge(mp_p.X, mp_p.Y; alpha = 1.0)

        # 决策行（t = T 的 feature）→ mode 条件均值 → asset 空间 normalized
        # field 预测（E·ŷ，输出契约 2 的 μ_asset 语义）
        yhat = predict_mode(fit.B, mp.X_full[end, :])
        yhat_p = predict_mode(fit_p.B, mp_p.X_full[end, :])
        mu = E * yhat
        mu_p = domain_mode_basis(N) * yhat_p
        @test isapprox(mu_p, mu[perm], atol = 1e-8)     # asset 预测 π 协变
        @test isapprox(yhat_p[1], yhat[1], atol = 1e-8) # macro 输出置换不变（全观测对称）
    end

    # -------------------------------------------------------------------
    # 5. fail loudly（维度 / NaN / Inf / alpha 域）
    # -------------------------------------------------------------------
    @testset "fail loudly" begin
        rng = MersenneTwister(SEED + 4)
        N = 4
        P = 14 * N
        n = 300
        X = randn(rng, n, P)
        Y = randn(rng, n, N)
        T = 60
        returns = 0.01 .* randn(rng, T, N)
        s1 = 0.01 .+ 0.02 .* rand(rng, N)
        E = domain_mode_basis(N)

        # build_mode_problem：维度不匹配 / T<2 / observed 坐标 NaN
        @test_throws DimensionMismatch build_mode_problem(returns, trues(T - 1, N), s1, E)
        @test_throws DimensionMismatch build_mode_problem(returns, trues(T, N), s1[1:(N - 1)], E)
        @test_throws ArgumentError build_mode_problem(returns[1:1, :], trues(1, N), s1, E)
        r_bad = copy(returns); r_bad[10, 2] = NaN
        @test_throws DomainError build_mode_problem(r_bad, trues(T, N), s1, E)

        # fit_full_block_ridge：P ≠ 14N / 行数不匹配 / alpha 域 / NaN / Inf
        @test_throws ArgumentError fit_full_block_ridge(X[:, 1:40], Y)
        @test_throws DimensionMismatch fit_full_block_ridge(X[1:50, :], Y)
        @test_throws ArgumentError fit_full_block_ridge(X, Y; alpha = 0.0)
        @test_throws ArgumentError fit_full_block_ridge(X, Y; alpha = -1.0)
        @test_throws ArgumentError fit_full_block_ridge(X, Y; alpha = Inf)
        Xbad = copy(X); Xbad[3, 5] = NaN
        @test_throws DomainError fit_full_block_ridge(Xbad, Y)
        Ybad = copy(Y); Ybad[2, 2] = Inf
        @test_throws DomainError fit_full_block_ridge(X, Ybad)

        # predict_mode / block_views：维度与 NaN
        fit_ok = fit_full_block_ridge(X, Y; alpha = 1.0)
        x_ok = randn(rng, P)
        @test_throws DimensionMismatch predict_mode(fit_ok.B, x_ok[1:(P - 1)])
        x_bad = copy(x_ok); x_bad[2] = NaN
        @test_throws DomainError predict_mode(fit_ok.B, x_bad)
        @test_throws ArgumentError block_views(zeros(N, 40))
    end

    # -------------------------------------------------------------------
    # 6. D-093 known DC world（Step 5；裁决书 §10 D-031/D-032/D-033）
    # -------------------------------------------------------------------
    @testset "D-093 known DC world" begin
        rng = MersenneTwister(SEED + 5)
        N = 4
        Pdc = 1 + 14 * N
        n = 300
        Xdc = hcat(ones(n), randn(rng, n, 14 * N))     # [1 | 动态列]（DC 形态）

        # --- (a) 纯 DC 世界：b₀* ≠ 0、动态系数全零 ---
        b0star = vcat(0.03, 0.01 .* randn(rng, N - 1))  # m 行 0.03、q 行小分量
        Bstar = hcat(b0star, zeros(N, 14 * N))          # N × (1+14N)
        Y = Xdc * Bstar'                                # 无噪声（每行 = b₀*）

        fitdc = fit_full_block_ridge(Xdc, Y; alpha = 1e-10)
        @test isapprox(fitdc.b0, b0star, atol = 1e-8)               # b₀ 恢复（D-093 核心）
        @test isapprox(fitdc.B[:, 2:end], zeros(N, 14 * N), atol = 1e-8)  # 动态系数保持 0
        @test isapprox(Matrix(fitdc.B_mm), zeros(1, 14), atol = 1e-8)     # 分块点名
        @test isapprox(Matrix(fitdc.B_perpperp), zeros(N - 1, 14 * (N - 1)), atol = 1e-8)

        # --- (b) 无 DC 列对照（Step 4 形式）：常数漂移被强制进动态系数/残差 ---
        # 同样的数据去掉 DC 列拟合——「所有 dynamic path feature 不解释收益时
        # 无条件均值必须为零」正是 D-031 要修复的旧缺陷（残差显著更大）。
        fitnodc = fit_full_block_ridge(Xdc[:, 2:end], Y; alpha = 1e-10)
        resid_dc = norm(Y .- Xdc * fitdc.B')
        resid_nodc = norm(Y .- Xdc[:, 2:end] * fitnodc.B')
        @test resid_dc < 1e-6
        @test resid_nodc > 0.2 * norm(Y)

        # --- (c) 混合世界：b₀* 与非零动态 G* 同时存在 → 两者都恢复（分离能力）---
        b0mix = vcat(0.02, 0.01 .* randn(rng, N - 1))
        Gmix = 0.4 .* randn(rng, N, 14 * N)
        Ymix = Xdc * hcat(b0mix, Gmix)'
        fitmix = fit_full_block_ridge(Xdc, Ymix; alpha = 1e-10)
        @test isapprox(fitmix.b0, b0mix, atol = 1e-8)
        @test isapprox(fitmix.B[:, 2:end], Gmix, atol = 1e-8)

        # --- (d) pipeline dc=true：维度 / 列布局 / b₀ 维度 / 块列偏移 ---
        rng2 = MersenneTwister(SEED + 6)
        T = 320
        returns = 0.01 .* randn(rng2, T, N)
        s1 = 0.01 .+ 0.02 .* rand(rng2, N)
        E = domain_mode_basis(N)
        mp0 = build_mode_problem(returns, trues(T, N), s1, E)             # dc=false（默认）
        mp1 = build_mode_problem(returns, trues(T, N), s1, E; dc = true)  # dc=true
        @test size(mp1.X) == (T - 1, Pdc) && size(mp1.X_full) == (T, Pdc)
        @test all(==(1.0), mp1.X_full[:, 1])                    # DC 列恒 1（D-031 x_{0,t}=1）
        @test mp1.X_full[:, 2:end] == mp0.X_full                # 其余列与 dc=false 逐位一致
        fit1 = fit_full_block_ridge(mp1.X, mp1.Y; alpha = 1.0)
        @test size(fit1.b0) == (N,)                             # b₀ ∈ R^N（统一 mode 输出空间）
        @test fit1.b0[2] == fit1.B[2, 1]                        # view 语义（b₀ 共享 B 内存）
        @test fit1.B_mm[1, 3] == fit1.B[1, 4]                   # 块列偏移：四块从列 2 起
        @test size(fit1.B_mm) == (1, 14) && size(fit1.B_mperp) == (1, 14 * (N - 1))

        # --- (e) 兼容性回归：dc=false 路径 b0 = nothing（Step 4 接口不破坏）---
        @test isnothing(fitnodc.b0)                        # 无 DC 拟合：b0 = nothing
        @test isnothing(block_views(fitnodc.B).b0)

        # --- (f) 零中心纪律（D-032）注释钉死：本步固定 ridge 无先验项；任何
        #     未来 prior 版本的 DC 先验均值必须为 0，禁止人工正收益 prior、
        #     禁止按历史赢家指定正先验——非零 posterior 只能来自价格证据
        #     （Step 8 义务；此处无可执行断言，纪律以 docstring + 本注释钉死）。

        # --- (g) fail loudly：DC 列非 1 值（构造校验）---
        Xbad_dc = hcat(2.0 .* ones(n), randn(rng, n, 14 * N))
        @test_throws ArgumentError fit_full_block_ridge(Xbad_dc, Y)
    end

    # -------------------------------------------------------------------
    # 7. D-094 trace-hypothesis counterexample（Step 6；裁决书 §33）
    # -------------------------------------------------------------------
    # 本测试是 D-094 的裁决内容——trace 约束从默认 core 移出的**反例证据
    # 闭环**：真实存在 diagonal common response 的世界里，unconstrained
    # 新 core 能恢复，而 trace-neutral 分支会系统性删除它。
    #
    # 纪律钉死：
    # - 新线**永不施加** trace 约束（D-028：默认 core 无 14 条约束；
    #   D-030：可达 support 由 gauge 坐标构造性处理，无需约束）；
    # - 旧 trace-conditioned solver 保留于冻结旧线 src/response.jl
    #   （D-029/D-084；H6 裁决不复制进新线）——本测试的 trace 对照是
    #   **测试内构造**，不进 src/；
    # - 对照采用欧氏投影 B_c = B − Cᵀ(CCᵀ)⁻¹C·vec(B) 的最小范数修正
    #   形式——只表达「trace 约束方向被移除」的语义，**不精确复刻**旧线
    #   optimize_conditioned_eb 的 Ω 度量 Gaussian conditioning（任务与
    #   TRACE_NEUTRALITY_DERIVATION.md A.2 语义对照即可）。
    #
    # 合成 X 的列布局：本测试采用旧线 **asset 坐标布局**（A.1：
    # 每 band 2N 列，col_q = (b−1)·2N + j、col_p = (b−1)·2N + N + j）——
    # 因为 D-094 的反例对象正是旧线 trace 约束所定义的对角方向
    # （G[i, asset_i 自己的列]）。fit_full_block_ridge 对列内容无假设
    # （P = 14N 即可），unconstrained 恢复不依赖布局。
    @testset "D-094 trace-hypothesis counterexample" begin
        rng = MersenneTwister(SEED + 7)
        N = 4
        P = 14 * N
        n = 300
        X = randn(rng, n, P)                    # 旧线 asset 坐标列布局（本 testset 语义）

        # 反例 G*：每 band/channel 的对角块非零且相等（c > 0 对全部 i）。
        # trA_b = trB_b = N·c ≠ 0——正是 trace 约束杀死的 diagonal common
        # response 方向（A.2：C 的行杀死对角和 Σᵢ G[i, asset_i 的列]）。
        c = 0.3
        G = zeros(N, P)
        for b in 1:7
            off = (b - 1) * 2 * N
            for i in 1:N
                G[i, off + i] = c               # Q 通道对角（asset_i 自己的 Q 列）
                G[i, off + N + i] = c           # P 通道对角（asset_i 自己的 P 列）
            end
        end
        # 本 testset 局部 helper（旧线 asset 坐标布局的对角和语义）
        trQ(B, b) = sum(B[i, (b - 1) * 2 * N + i] for i in 1:N)
        trP(B, b) = sum(B[i, (b - 1) * 2 * N + N + i] for i in 1:N)

        @test isapprox(trQ(G, 1), N * c, atol = 1e-12)   # 构造健全：trA_b = N·c ≠ 0

        # --- (a) unconstrained 新 core 恢复（D-094 第一半）---
        Y = X * G'
        fit = fit_full_block_ridge(X, Y; alpha = 1e-10)
        @test isapprox(fit.B, G, atol = 1e-8)                            # 全矩阵恢复
        @test isapprox([fit.B[i, 3 * 2 * N + i] for i in 1:N], fill(c, N), atol = 1e-8)      # b=4 Q 对角点名
        @test isapprox([fit.B[i, 6 * 2 * N + N + i] for i in 1:N], fill(c, N), atol = 1e-8)  # b=7 P 对角点名

        # --- (b) trace-neutral 对照的系统性删除（D-094 第二半）---
        # C：14 行（7 band × 2 channel，A.1 顺序 (b,Q),(b,P),…）、N·P 列；
        # 行 (b,ch) 在 asset_i 的 (b,ch) 列的 vec 位置（i + (col−1)·N）为 1。
        # 每列恰属一行（各行支撑不相交）⇒ CCᵀ = N·I₁₄（每行 N 个 1）。
        C = zeros(14, N * P)
        row = 0
        for b in 1:7, ch in 1:2
            row += 1
            for i in 1:N
                col = (b - 1) * 2 * N + (ch - 1) * N + i   # asset_i 的 (b,ch) 列
                C[row, i + (col - 1) * N] = 1.0            # vec 列主序（A.1 测试构造同式）
            end
        end
        bv = vec(fit.B)
        lam = (C * C') \ (C * bv)                           # 14 维（CCᵀ = N·I ⇒ lam = (C·bv)/N）
        B_c = reshape(bv .- C' * lam, N, P)                 # 欧氏投影：trace 方向被移除

        # 条件化后对角和 ≈ 0（系统性删除的直接证据）
        @test all(b -> abs(trQ(B_c, b)) < 1e-8, 1:7) && all(b -> abs(trP(B_c, b)) < 1e-8, 1:7)
        # 对角恢复误差：trace 版显著大于 unconstrained 版（删除了真实结构）
        @test norm(B_c .- G) > 100 * norm(fit.B .- G)

        # --- (c) 残差语义：同一数据上，删除真实结构付出拟合代价 ---
        resid_full = norm(Y .- X * fit.B')
        resid_trace = norm(Y .- X * B_c')
        @test resid_full < 1e-6                             # unconstrained：无噪声下 ≈ 0
        @test resid_trace > 0.2 * norm(Y)                   # trace-neutral：对角信号被删

        # --- (d) 小噪声鲁棒性：unconstrained 误差仍远小于被删除结构 ---
        Yn = Y .+ 0.01 .* randn(rng, n, N)
        fitn = fit_full_block_ridge(X, Yn; alpha = 1e-10)
        @test norm(fitn.B .- G) < 0.1 * norm(B_c .- G)
    end
end
