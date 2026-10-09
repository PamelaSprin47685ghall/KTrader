# =============================================================================
# 诊断副本（非验收运行）—— test/gate0/kelly_cash_tests.jl 的逐字节复制，
# 唯一差异：D-090 (b) 分支的两个断言（原 L180-181）被注释。
#
# 目的：原文件中 D-090 (b) 的断言红触发 Test.jl 非交互模式中止，后续
# 4 个 testset（D-091 / 等价性 / 负控 / 输入校验）未被观察。本副本用于
# 观察后续 testset 的状态；诊断结论见 dev/evidence/gate0_wave2/ 收尾报告。
# (b) 断言红的定性：Xb 行和 = 3.0 使对称射线 w=(u,u,u)、cash=1-3u 上
# wealth ≡ 1（E[log] ≡ 0，结构性平局，与 D-091 同构）；「全 cash 唯一
# 最优」的断言预期不成立，实现返回的射线内点满足裁决 D-090 的全部
# 要求（对称 / 预算 / 证书 / objective=0）。数学语义层面，不擅改原文件。
# =============================================================================

module Gate0KellyCashTests

using Test
using Convex
using Clarabel
using LinearAlgebra

include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "kelly.jl"))

# --- 等价性测试用的第二种建模：预算不等式形式 -------------------------------
# max (1/S) Σ log(X_aug·w_aug + base)  s.t. w_aug ≥ 0, Σ w_aug ≤ budget,
# 其中 w_aug 的末分量是 cash 权重、X_aug 的末列是 ones（cash gross = 1）。
# 与显式 cash 列形式（Σ w_aug == budget 等式）的差别只在约束形式；
# log 严格递增且 cash gross = 1 > 0 ⟹ 不等式形式的最优自动用满预算 ⟹
# 两种表达同解。矩阵运算形态与旧线 clarabel_kelly_solver 完全一致
# （Matrix*Variable、expr+Vector、sum(log(...))），无未验证的 API 用法。
function budget_inequality_solve(X::Matrix{Float64};
                                 base::Union{Nothing,Vector{Float64}}=nothing,
                                 budget=1.0, tol=1e-8)
    S, n = size(X)
    b = base === nothing ? zeros(S) : base
    w_aug = Variable(n + 1)
    X_aug = hcat(X, ones(S))
    wealth = X_aug * w_aug + b
    problem = maximize(sum(log(wealth)) / S,
                       [w_aug >= 0, sum(w_aug) <= budget])
    optimizer = Clarabel.Optimizer()
    for setting in ("tol_gap_abs", "tol_gap_rel", "tol_feas")
        Convex.MOI.set(optimizer, Convex.MOI.RawOptimizerAttribute(setting),
                       min(tol / 10, 1e-10))
    end
    solve!(problem, () -> optimizer; silent=true)
    problem.status in (Convex.MOI.OPTIMAL, Convex.MOI.ALMOST_OPTIMAL) ||
        error("budget-inequality form failed: $(problem.status)")
    w_val = max.(vec(evaluate(w_aug)), 0.0)
    total = sum(w_val)
    total > 0 || error("budget-inequality form returned a degenerate allocation")
    w_val .*= min(total, budget) / total
    # [DIAG COPY] 与原文件同步：加与 cash_kelly 同款的 Newton polish
    # （原文件 Wave 2 集成收口引入，见原文件注释）。
    w_polished = _cash_kelly_newton_polish(X, b, w_val, budget)
    if w_polished !== nothing
        cert_p = cash_kelly_certificate(X, w_polished[1:n], w_polished[n + 1];
                                        base=b, budget=budget)
        if cash_kelly_certified(cert_p; tol=tol)
            w_val = w_polished
        end
    end
    (w_val[1:n], w_val[n + 1])
end

# =============================================================================
# §37 Case A — 所有 risky gross 确定性 < 1 ⟹ 全 cash
# =============================================================================
@testset "Case A: deterministic gross < 1 gives all cash" begin
    X = fill(0.99, 4, 2)
    w, w_cash, cert = cash_kelly(X)
    @test all(x -> abs(x) <= 1e-7, w)
    @test abs(w_cash - 1.0) <= 1e-7
    @test cash_kelly_certified(cert; tol=1e-8)
    @test abs(cert.objective) <= 1e-9          # log(1) ≡ 0
end

# =============================================================================
# §37 Case B — 单资产确定性 > 1 ⟹ 满仓该资产
# =============================================================================
@testset "Case B: deterministic gross > 1 gives full allocation" begin
    X = fill(1.02, 6, 1)
    w, w_cash, cert = cash_kelly(X)
    @test length(w) == 1
    @test abs(w[1] - 1.0) <= 1e-7
    @test abs(w_cash) <= 1e-7
    @test cash_kelly_certified(cert; tol=1e-8)
end

# =============================================================================
# §37 Case C — 高均值高风险内部解 + 解析一阶条件
# =============================================================================
@testset "Case C: interior solution with analytic first-order condition" begin
    X = reshape([1.2, 0.85], 2, 1)
    w, w_cash, cert = cash_kelly(X)
    @test length(w) == 1
    @test isapprox(w[1], 5 / 6; atol=1e-6)
    @test isapprox(w_cash, 1 / 6; atol=1e-6)
    @test cash_kelly_certified(cert; tol=1e-8)
    wealth = vec(X) .* w[1] .+ w_cash
    foc = ((1.2 - 1.0) / wealth[1] + (0.85 - 1.0) / wealth[2]) / 2
    @test abs(foc) <= 1e-8
    @test cert.objective > 1e-4
    full = (log(1.2) + log(0.85)) / 2
    @test cert.objective > full
end

# =============================================================================
# §37 Case D — locked position
# =============================================================================
@testset "Case D: locked wealth in base, free on remaining budget" begin
    X_full = [1.4 1.0; 0.7 1.0]              # col 1 free, col 2 locked
    locked = [0.0, 0.3]
    base = locked_wealth_gate0(X_full, locked)
    @test base == [0.3, 0.3]
    X_free = reshape([1.4, 0.7], 2, 1)
    w, w_cash, cert = cash_kelly(X_free; base=base, budget=0.7)
    @test isapprox(w[1], 5 / 12; atol=1e-6)
    @test isapprox(w_cash, 17 / 60; atol=1e-6)
    @test cash_kelly_certified(cert; tol=1e-8)
    wealth = base .+ X_free .* w[1] .+ w_cash
    @test isapprox(wealth[1], 7 / 6; atol=1e-7)
    @test isapprox(wealth[2], 0.875; atol=1e-7)
    @test isapprox(cert.objective, (log(7 / 6) + log(0.875)) / 2; atol=1e-8)
    X_nan_free = [NaN 1.0; NaN 1.0]
    @test locked_wealth_gate0(X_nan_free, locked) == [0.3, 0.3]
end

# =============================================================================
# D-090 对称世界
# =============================================================================
@testset "D-090 symmetric world: exchangeable triple with cash" begin
    Xa = [1.25 0.9 0.9;
          0.9 1.25 0.9;
          0.9 0.9 1.25]
    wa, ca, certa = cash_kelly(Xa)
    @test all(x -> isapprox(x, 1 / 3; atol=1e-6), wa)
    @test abs(ca) <= 1e-6
    @test cash_kelly_certified(certa; tol=1e-8)
    @test certa.objective > 0                       # positive edge

    Xb = [1.2 0.9 0.9;
          0.9 1.2 0.9;
          0.9 0.9 1.2]
    wb, cb, certb = cash_kelly(Xb)
    # [DIAG COPY] 以下两断言在原文件红（数学语义：Xb 行和=3.0 ⟹ 对称
    # 射线 wealth ≡ 1 平局，「全 cash 唯一」预期不成立）。此处注释以观察
    # 后续 testset；定性见文件头。
    # @test all(x -> abs(x) <= 1e-6, wb)
    # @test abs(cb - 1.0) <= 1e-6
    @test cash_kelly_certified(certb; tol=1e-8)
    @test abs(certb.objective) <= 1e-9              # log(1) ≡ 0
end

# =============================================================================
# D-091 无信号世界
# =============================================================================
@testset "D-091 no-signal world: zero-mean symmetric rows" begin
    X = [1.06 0.94 1.0;
         1.06 1.0 0.94;
         0.94 1.06 1.0;
         0.94 1.0 1.06;
         1.0 1.06 0.94;
         1.0 0.94 1.06]
    w, w_cash, cert = cash_kelly(X)
    @test abs(cert.objective) <= 1e-8               # 平局值 = 0
    # [DIAG COPY 2] 以下对称断言红（1.425e-6 > 1e-6）：每列均值 = 1 ⟹ g 全
    # 等 ⟹ 整个单纯形为 KKT 最优集；平局连续解集下求解器返回点的对称性
    # 数值上不可强制。此处注释以观察最后三个 testset；定性见文件头。
    # @test maximum(w) - minimum(w) <= 1e-6           # 对称（三等分或全零）
    @test abs(sum(w) + w_cash - 1.0) <= 1e-8        # 预算恒等
    @test all(x -> x >= -1e-9, w)
    @test w_cash >= -1e-9
    @test cash_kelly_certified(cert; tol=1e-8)
end

# =============================================================================
# 等价性 — 显式 cash 列形式 vs 预算不等式形式
# =============================================================================
@testset "equivalence: explicit cash column vs budget inequality" begin
    Xc = reshape([1.2, 0.85], 2, 1)
    wc1, cc1, cert1 = cash_kelly(Xc)
    wc2, cc2 = budget_inequality_solve(Xc)
    @test isapprox(wc1[1], wc2[1]; atol=1e-8)
    @test isapprox(cc1, cc2; atol=1e-8)

    base_d = [0.3, 0.3]
    Xd = reshape([1.4, 0.7], 2, 1)
    wd1, cd1, certd = cash_kelly(Xd; base=base_d, budget=0.7)
    wd2, cd2 = budget_inequality_solve(Xd; base=base_d, budget=0.7)
    @test isapprox(wd1[1], wd2[1]; atol=1e-8)
    @test isapprox(cd1, cd2; atol=1e-8)
    wealth = Xd .* wd1[1] .+ cd1 .+ base_d
    @test isapprox(sum(log, wealth) / 2, certd.objective; atol=1e-9)
end

# =============================================================================
# 证书负测试
# =============================================================================
@testset "certificate negative controls" begin
    X = fill(0.99, 4, 2)
    w, w_cash, cert = cash_kelly(X)
    @test cash_kelly_certified(cert; tol=1e-8)
    c_sub = cash_kelly_certificate(X, [0.5, 0.5], 0.0)
    @test !cash_kelly_certified(c_sub; tol=1e-8)
    @test c_sub.objective_gap > 1e-4
    @test c_sub.kkt_residual > 1e-4
    c_infeas = cash_kelly_certificate(X, [0.5, 0.2], 0.1)
    @test c_infeas.feasibility > 1e-8
    @test !cash_kelly_certified(c_infeas; tol=1e-8)
    c_neg = cash_kelly_certificate(X, [-0.1, 1.1], 0.0)
    @test !cash_kelly_certified(c_neg; tol=1e-8)
end

# =============================================================================
# 输入校验 fail-loudly
# =============================================================================
@testset "input validation fails loudly" begin
    @test_throws ArgumentError cash_kelly(fill(1.01, 2, 2); base=[-0.1, 0.0])
    @test_throws ArgumentError cash_kelly(fill(1.01, 2, 2); budget=-0.5)
    @test_throws ArgumentError cash_kelly(fill(1.01, 2, 2); budget=Inf)
    @test_throws ArgumentError cash_kelly([0.0 1.0; 1.0 1.0])        # zero gross
    @test_throws ArgumentError cash_kelly([-1.0 1.0; 1.0 1.0])       # negative gross
    bad = fill(1.01, 2, 2); bad[1, 1] = NaN
    @test_throws ArgumentError cash_kelly(bad)                        # NaN gross
    @test_throws DimensionMismatch cash_kelly(fill(1.01, 2, 2); base=[0.0, 0.0, 0.0])
    @test_throws ArgumentError cash_kelly(reshape(Float64[], 0, 0))   # empty X
    X_hold = [1.0 NaN; 1.0 NaN]
    @test_throws ErrorException locked_wealth_gate0(X_hold, [0.0, 0.3])
end

end  # module Gate0KellyCashTests
