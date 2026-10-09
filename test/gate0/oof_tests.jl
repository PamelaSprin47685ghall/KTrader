# =============================================================================
# KTraderGate0 OOF full-mode folds — Step 9 tiny 级测试
# =============================================================================
#
# 测试对象：src/gate0/oof.jl（fold_grid / fold_stats / train_stats /
# fit_fold_posteriors / oof_residual_rows_full）。
#
# 覆盖（SPEC §27-§29 / RP §9 / D-043/D-044）：
# - fold 差分恒等式（train = full − fold，atol 1e-10）+ 差分 vs 直接
#   重算一致（差分路径的数值精度验证）；
# - fold 独立性（独立 quadrature 证据 + held-out 隔离：直接路径逐位、
#   差分路径 atol——差分浮点微扰语义见 train_stats docstring）；
# - OOF 残差行正确性（手工对照 + 全行覆盖）；
# - cross-block OOF（统一设计的构造性保证 + in-sample vs OOF 过拟合差
#   的可执行断言——泄漏会使 OOF 残差像 in-sample 一样小）；
# - 无观察行不进统计（rows 行集入口）；
# - fail loudly（F 非法 / fold propriety 违例含 fold 编号）。
#
# 纪律：固定 seed；不运行测试（运行归 DevOps）；断言与回测收益无关
# （D-044：OOF 不是模型选择工具——本文件无任何 Sharpe/收益断言）。

using Test, Random, LinearAlgebra

module Gate0OOFEnv
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "geometry.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "modes.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "response.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "posterior.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "oof.jl"))
end
using .Gate0OOFEnv

_throws_msg(f, needle) = try
    f()
    false
catch e
    e isa ErrorException && occursin(needle, sprint(showerror, e))
end

@testset "gate0 Step 9: OOF full-mode folds" begin
    SEED = 20261009
    rng = MersenneTwister(SEED)
    N, P, n, F = 2, 3, 30, 3
    X = randn(rng, n, P)
    Y = X * (0.5 .* randn(rng, N, P))' .+ 0.3 .* randn(rng, n, N)
    E = domain_mode_basis(N)
    folds = fold_grid(n; F_folds = F)

    # -------------------------------------------------------------------
    # 1. fold 差分恒等式 + 差分 vs 直接重算一致（SPEC §28）
    # -------------------------------------------------------------------
    @testset "fold difference identity" begin
        fs = fold_stats(X, Y, folds)
        @test vcat(folds...) == collect(1:n)                 # 覆盖完整（无遗漏/重复）
        for f in 1:F
            tr = setdiff(1:n, folds[f])
            st_train = train_stats(fs.full, fs.folds[f])     # 差分路径
            st_direct = sufficient_stats(X[tr, :], Y[tr, :]) # 直接重算路径
            @test st_train.n == st_direct.n
            @test isapprox(st_train.Sxx, st_direct.Sxx, atol = 1e-10)
            @test isapprox(st_train.Sxy, st_direct.Sxy, atol = 1e-10)
            @test isapprox(st_train.Syy, st_direct.Syy, atol = 1e-10)
        end
    end

    # -------------------------------------------------------------------
    # 2. fold 独立性与 held-out 隔离（D-043）
    # -------------------------------------------------------------------
    @testset "fold independence & held-out isolation" begin
        fp = fit_fold_posteriors(X, Y, folds; tol = 1e-4)
        # (a) 独立 quadrature 证据：不同 train 数据 ⇒ 不同节点/权重
        @test fp.posteriors[1].alpha_nodes != fp.posteriors[2].alpha_nodes ||
              fp.posteriors[1].alpha_weights != fp.posteriors[2].alpha_weights
        # (b) 直接路径逐位隔离：修改 fold 1 的 held-out 行 target 后，
        #     fold 1 的 train 行子矩阵逐位不变 ⇒ 直接重算 posterior 逐位不变
        s_hold = folds[1][2]
        Y_mod = copy(Y); Y_mod[s_hold, 1] += 5.0
        tr1 = setdiff(1:n, folds[1])
        p_before = fit_full_posterior(X[tr1, :], Y[tr1, :]; tol = 1e-4)
        p_after = fit_full_posterior(X[tr1, :], Y_mod[tr1, :]; tol = 1e-4)
        @test p_before.alpha_nodes == p_after.alpha_nodes &&
              p_before.alpha_weights == p_after.alpha_weights
        # (c) 差分路径隔离：修改 fold 1 的 held-out 行（贡献在 full 与
        #     fold1 中同现、差分相消）⇒ fold 1 posterior 数值不变
        #     （atol 承接差分路径的 eps 级浮点微扰——代数恒等式构造成立）
        fp_mod = fit_fold_posteriors(X, Y_mod, folds; tol = 1e-4)
        Bmix(a) = sum(c.weight .* c.B_hat for c in a.posteriors[1].conditional)
        @test isapprox(Bmix(fp), Bmix(fp_mod), atol = 1e-8)
    end

    # -------------------------------------------------------------------
    # 3. OOF 残差行正确性（RP §7.3(a)：完整后验均值 + b₀ 扣除 + asset 空间）
    # -------------------------------------------------------------------
    @testset "oof residual rows" begin
        fp = fit_fold_posteriors(X, Y, folds; tol = 1e-4)
        eps_full = oof_residual_rows_full(fp, X, Y, E)
        @test size(eps_full) == (n, N)                       # 全行覆盖（无遗漏）
        # 手工对照：行 s 用其所属 fold 的 train posterior 混合均值预测
        manual = Matrix{Float64}(undef, n, N)
        Bmix = [sum(c.weight .* c.B_hat for c in post.conditional) for post in fp.posteriors]
        for (f, rows) in enumerate(folds), s in rows
            manual[s, :] .= E * (Y[s, :] .- Bmix[f] * view(X, s, :))
        end
        @test isapprox(eps_full, manual, atol = 1e-10)
    end

    # -------------------------------------------------------------------
    # 4. cross-block OOF（统一设计的构造性保证——RP §9 第 4 条：macro/
    #    relative 交叉块在同一 fold 差分内，混合路径在结构上不可能；
    #    可执行断言 = in-sample vs OOF 的过拟合差：泄漏会使 OOF 残差
    #    像 in-sample 一样小）
    # -------------------------------------------------------------------
    @testset "cross-block OOF" begin
        rng4 = MersenneTwister(SEED + 1)
        N4 = 2
        P4 = 14 * N4                       # 统一设计（macro+relative 列同 fold 差分）
        n4 = 45
        X4 = randn(rng4, n4, P4)
        G4 = 0.8 .* randn(rng4, N4, P4)    # cross 块与对角块同幅度（统一信号）
        Y4 = X4 * G4' .+ 0.3 .* randn(rng4, n4, N4)
        E4 = domain_mode_basis(N4)
        folds4 = fold_grid(n4; F_folds = 3)
        fp4 = fit_fold_posteriors(X4, Y4, folds4; tol = 1e-4)
        oof4 = oof_residual_rows_full(fp4, X4, Y4, E4)
        # in-space 对照：full-data posterior 的单 fold 残差（Step 8 接口）
        post_full = fit_full_posterior(X4, Y4; tol = 1e-4)
        insample4 = oof_residual_rows(post_full, E4)
        @test norm(oof4) > 1.15 * norm(insample4)   # OOF 残差显著大于 in-sample（过拟合差）
    end

    # -------------------------------------------------------------------
    # 5. 无观察行不进统计（rows 行集入口；mask 判定归 prepare/consumer 层）
    # -------------------------------------------------------------------
    @testset "unobserved row exclusion" begin
        # 行 7 为「无观察」行（由调用方剔除——行集入口）。fixture 索引
        # 语义修正（Wave 3 裁决执行轮）：原 `fold_grid(rows)` 返回原值
        # 索引（1..30 除 7）配完整 30 行 X/Y——fold_stats 的 full 统计
        # 自然含行 7（n=30 ≠ 29 断言红）。「剔除」的正确形态 = 剔除后
        # 矩阵 + 新行索引（fold_grid(length(rows))）。
        rows = setdiff(1:n, [7])
        X5 = X[rows, :]
        Y5 = Y[rows, :]
        folds5 = fold_grid(length(rows); F_folds = F)
        fs5 = fold_stats(X5, Y5, folds5)
        @test fs5.full.n == n - 1          # n 不计
        # 字段级比较（struct == 走对象同一性；同输入同函数 → 逐位相同）
        st5 = sufficient_stats(X5[folds5[1], :], Y5[folds5[1], :])
        @test fs5.folds[1].n == st5.n
        @test fs5.folds[1].Sxx == st5.Sxx && fs5.folds[1].Sxy == st5.Sxy &&
              fs5.folds[1].Syy == st5.Syy
    end

    # -------------------------------------------------------------------
    # 6. fail loudly（F 非法 / fold propriety 含编号）
    # -------------------------------------------------------------------
    @testset "fail loudly" begin
        @test_throws ArgumentError fold_grid(n; F_folds = 1)
        @test_throws ArgumentError fold_grid(n; F_folds = n + 1)
        # fold propriety：full 层绿（rank 3）但 fold 1 的 train 退化
        # （行 4..9 的 Y 全共线 ⇒ rank(Syy^{(−1)}) = 1 < N=3）
        rng6 = MersenneTwister(SEED + 2)
        n6, N6, P6 = 9, 3, 4
        X6 = randn(rng6, n6, P6)
        Y6 = zeros(n6, N6)
        Y6[1:3, :] = randn(rng6, 3, N6)    # 行 1-3 张满秩
        Y6[4:9, :] .= randn(rng6, 1, N6)   # 行 4-9 共线（rank 1）
        folds6 = fold_grid(n6; F_folds = 3)
        @test _throws_msg(() -> fit_fold_posteriors(X6, Y6, folds6; tol = 1e-4),
                          "posterior improper (fold f=1)")
    end
end
