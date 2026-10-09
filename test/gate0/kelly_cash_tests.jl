# =============================================================================
# KTraderGate0 — Step 2 tests: cash feasible set exact log-Kelly.
# 所属 module: KTraderGate0 的 constitutional suite（test/gate0/ 独立入口，
# 裁决 H1）。module 骨架与统一入口（test/gate0/runtests.jl）由集成轮交付；
# 为使本文件可独立运行，此处将 src/gate0/kelly.jl include 进本地 module —
# 集成轮以 `using KTraderGate0` 替代此 include 块即可，断言本身不变。
#
# 覆盖（裁决书 §37 四 case + D-090/D-091 + 任务书附加项）：
#   Case A  全 risky 确定性 < 1        → w_cash = 1, w_risky = 0
#   Case B  单资产确定性 > 1           → w_asset = 1, cash = 0
#   Case C  高均值高风险内部解          → 解析一阶条件 E[(X-1)/wealth] = 0
#   Case D  locked position            → base_s 正确、剩余预算优化、总财富分解
#   D-090   对称世界 (a) 正 edge / (b) 无 edge
#   D-091   无信号世界（零均值对称）
#   等价性  显式 cash 列 vs 预算不等式（同一 X，容差 1e-8）
#   证书负测试 + 输入校验 fail-loudly
#
# 全部合成数据、完全确定（无 RNG、无 seed）。解析答案的推导以注释给出。
# =============================================================================

module Gate0KellyCashTests

using Test
using Convex
using Clarabel
using LinearAlgebra

include(joinpath(@__DIR__, "..", "..", "src", "gate0", "kelly.jl"))

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
    # Newton polish（与 cash_kelly 同款数值精修，Wave 2 集成收口）：被测
    # 函数 cash_kelly 引入 KKT 精修后解精确到 ~1e-12，而本辅助函数的原始
    # Clarabel 解带 ~1e-6 误差——等价性断言（atol=1e-8）对比的是两种
    # 建模形式，不应被求解器噪声淹没。加同款精修使两边都达到数学驻点
    # 精度；断言语义与容差不变（精修失败则保留原解，行为同未加精修）。
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
# 解析：全 cash 时 wealth ≡ 1，g_cash = 1 > g_risky = 0.99；KKT 在全 cash
# 角点成立（支撑 = {cash}，非支撑 g_risky < ν = 1），gap = 1 − 1 = 0。
# 任何 w_risky > 0 都使 E[log] < 0 < 全 cash 的 0 ⟹ 唯一最优。
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
# 解析：w = 1 时 wealth ≡ 1.02，g_asset = 1.02/1.02 = 1 > g_cash = 1/1.02；
# KKT 在 w = 1 角点成立（支撑 = {asset}），gap = 1 − 1 = 0。
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
# X = [1.2; 0.85]（S = 2，单资产，两状态等权）。wealth = 1 + w·(X − 1)。
# 内部一阶条件 g_asset = g_cash ⟺ E[(X − 1)/wealth] = 0：
#     0.2/(1 + 0.2w) = 0.15/(1 − 0.15w)
#     ⟹ 0.2 − 0.03w = 0.15 + 0.03w ⟹ w* = 5/6,  w_cash* = 1/6.
# 端点对照：E[log](w = 5/6) ≈ 0.01031 > E[log](w = 1) ≈ 0.00990 > 0 ⟹
# 内部最优成立（正 edge，cash 与 risky 同时持有）。
# =============================================================================
@testset "Case C: interior solution with analytic first-order condition" begin
    X = reshape([1.2, 0.85], 2, 1)
    w, w_cash, cert = cash_kelly(X)
    @test length(w) == 1
    @test isapprox(w[1], 5 / 6; atol=1e-6)
    @test isapprox(w_cash, 1 / 6; atol=1e-6)
    @test cash_kelly_certified(cert; tol=1e-8)
    # 数值验证一阶条件 E[(X-1)/wealth] = 0（任务书指定形式）：
    wealth = vec(X) .* w[1] .+ w_cash
    foc = ((1.2 - 1.0) / wealth[1] + (0.85 - 1.0) / wealth[2]) / 2
    @test abs(foc) <= 1e-8
    # 正 edge：目标值严格大于全 cash（0）与满仓（≈0.00990）。
    @test cert.objective > 1e-4
    full = (log(1.2) + log(0.85)) / 2
    @test cert.objective > full
end

# =============================================================================
# §37 Case D — locked position：base_s、剩余预算、总财富分解
# 全资产 scenario 矩阵 X_full = [1.4 1.0; 0.7 1.0]（列 1 free，列 2 locked
# 无风险 gross = 1）。locked 权重 0.3 ⟹ base = 0.3·1 = [0.3, 0.3]，
# budget = 1 − 0.3 = 0.7。
# wealth = base + w_f·X_f + w_c = 1 + w_f·(X_f − 1)（w_c = 0.7 − w_f 代入）。
# 一阶条件同 Case C 形式：0.4/(1+0.4w) = 0.3/(1−0.3w)
#     ⟹ 0.4 − 0.12w = 0.3 + 0.12w ⟹ w_f* = 5/12 ≈ 0.4167 < 0.7（内部）
#     ⟹ w_c* = 0.7 − 5/12 = 17/60.
# 总财富 = locked 贡献 + free 贡献 + cash：
#     行 1: 0.3 + (5/12)·1.4 + 17/60 = 7/6；行 2: 0.3 + (5/12)·0.7 + 17/60 = 0.875
# （与 1 ± 幅度·w 的解析式一致：1 + 0.4·5/12 = 7/6, 1 − 0.3·5/12 = 0.875）。
# 若 locked 被误当 cash（wealth ≡ 1），wealth 不会随行变化 — 本断言即
# locked-not-cash 的辨别性证据。
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
    # 总财富 = locked 贡献 + free 贡献 + cash（逐行对照解析值）
    wealth = base .+ X_free .* w[1] .+ w_cash
    @test isapprox(wealth[1], 7 / 6; atol=1e-7)
    @test isapprox(wealth[2], 0.875; atol=1e-7)
    @test isapprox(cert.objective, (log(7 / 6) + log(0.875)) / 2; atol=1e-8)
    # locked_wealth_gate0 的 0*NaN 防护（非持有侧）：locked = 0 的列即使
    # gross 为 NaN 也从不被触碰（0*NaN 不进入 base）；持有侧的 NaN 由
    # 下方 input-validation testset 的 ErrorException 断言覆盖。
    X_nan_free = [NaN 1.0; NaN 1.0]
    @test locked_wealth_gate0(X_nan_free, locked) == [0.3, 0.3]
end

# =============================================================================
# D-090 对称世界 — 三资产完全 exchangeable（行 = (a, b, b) 的全部置换）
# (a) a = 1.25：E[log] = (ln1.25 + 2ln0.9)/3 > 0 ⟹ 有正 edge。
#     对称解 w = (u,u,u) 下 wealth = 1 + u·(行和 − 3) = 1 + 0.05u（行和
#     恒 3.05）；一阶条件 Σ_s(X[s,i]−1)/wealth = (3.05−3)/wealth > 0 恒正
#     ⟹ 无内部 cash 解 ⟹ 角点 u = 1/3（cash = 0）。角点处
#     g_asset = 3.05/(3·(3.05/3)) = 1（精确）> g_cash = 3/3.05 ⟹ KKT 成立。
#     最优 w_risky 三等分 — 对称解（cash 边界是置换对称构造下的数学必然，
#     见 D-091 注释）。
# (b) a = 1.2：行和恒 3.0 ⟹ 对称射线 {(u,u,u), cash=1−3u} 上
#     wealth ≡ 1、objective ≡ 0——整条射线都是全局最优（结构性平局，
#     与 D-091 同构；原「全 cash 唯一最优」的推导漏了该射线）。
#     裁决书 D-090「允许全部 cash」是许可性语义（无正优势时不产生虚假
#     集中），非唯一性。断言（Manager 裁决 2026-10-09）：(i) 平局值
#     objective ≈ 0；(ii) 解在平局射线上（三分量对称、预算恒等、证书
#     绿）；(iii) 全 cash 端点 objective 对照——手工计算全 cash 点
#     （w=0, cash=1）的 objective 并断言与返回解相等，证明返回解无
#     虚假 edge、不劣于全 cash。
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
    # (i) 平局值：objective ≡ 0
    @test abs(certb.objective) <= 1e-9              # log(1) ≡ 0
    # (ii) 解在平局射线上：三分量对称、预算恒等、证书绿
    @test maximum(wb) - minimum(wb) <= 1e-6         # 对称射线上的点
    @test abs(sum(wb) + cb - 1.0) <= 1e-8           # 预算恒等
    @test cash_kelly_certified(certb; tol=1e-8)
    # (iii) 全 cash 端点对照（Manager 裁决 2026-10-09）：手工计算全 cash
    # 点（w_aug = [0,0,0,1]）的 objective，断言与返回解的 objective 相等
    # ——返回解无虚假 edge、不劣于全 cash 端点。
    wealth_cash = hcat(Xb, ones(3)) * [0.0, 0.0, 0.0, 1.0]   # ≡ [1,1,1]
    obj_cash = sum(log, wealth_cash) / 3
    @test isapprox(certb.objective, obj_cash; atol=1e-10)
end

# =============================================================================
# D-091 无信号世界 — 零均值对称（E[gross] = 1，行对称分布）
# 行 = (1.06, 0.94, 1.0) 的全部 6 个置换。E[gross_i] = 1 ∀i。
# Jensen：E[log(1 + wᵀ(X−1))] ≤ log(E[wᵀX]) = 0，等号 ⟺ wealth ≡ 1。
# 对称射线 w = (u,u,u)、cash = 1−3u 上 wealth ≡ 1 ⟹ 整条射线（含全
# cash 端点与等权满仓端点）目标恒 0 — 结构性平局；且每列均值 = 1 ⟹
# 全部 g_i 相等 ⟹ 整个单纯形都是 KKT 最优集（连续平局解集）。
# 断言（Manager 裁决 2026-10-09）：D-091 禁令对象是「稳定产生集中」而
# 非「严格对称」——连续平局解集上求解器返回点的严格对称数值不可强制
# （实测 max−min = 1.4e-6）。改为无集中断言 max(w) ≤ 1/3 + 0.01：若
# solver 产生 95% 单票（max ≈ 0.95）将远超此界而红——验证力保留。
# objective ≈ 0、预算、证书断言全部保留。
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
    @test maximum(w) <= 1 / 3 + 0.01                # 无集中（三资产世界）
    @test abs(sum(w) + w_cash - 1.0) <= 1e-8        # 预算恒等
    @test all(x -> x >= -1e-9, w)
    @test w_cash >= -1e-9
    @test cash_kelly_certified(cert; tol=1e-8)
end

# =============================================================================
# 等价性 — 显式 cash 列形式 vs 预算不等式形式（同一 X，容差 1e-8）
# 两种建模解同一决策问题（cash 列等式 simplex vs Σw + c ≤ budget 不等式；
# log 严格递增保证不等式形式自动用满预算）。
# =============================================================================
@testset "equivalence: explicit cash column vs budget inequality" begin
    # Case C 数据（内部解，两形式均非退化）
    Xc = reshape([1.2, 0.85], 2, 1)
    wc1, cc1, cert1 = cash_kelly(Xc)
    wc2, cc2 = budget_inequality_solve(Xc)
    @test isapprox(wc1[1], wc2[1]; atol=1e-8)
    @test isapprox(cc1, cc2; atol=1e-8)

    # Case D 数据（budget < 1 且 base 非零 — locked 语境下的等价性）
    base_d = [0.3, 0.3]
    Xd = reshape([1.4, 0.7], 2, 1)
    wd1, cd1, certd = cash_kelly(Xd; base=base_d, budget=0.7)
    wd2, cd2 = budget_inequality_solve(Xd; base=base_d, budget=0.7)
    @test isapprox(wd1[1], wd2[1]; atol=1e-8)
    @test isapprox(cd1, cd2; atol=1e-8)
    # 两形式的目标一致性（同一权重下逐行 wealth 相同）
    wealth = Xd .* wd1[1] .+ cd1 .+ base_d
    @test isapprox(sum(log, wealth) / 2, certd.objective; atol=1e-9)
end

# =============================================================================
# 证书负测试 — 正常解全绿；故意坏的 w 必须红（任务书允许的替代形态：
# 构造求解器次优场景困难时，改为证书函数对坏输入的拒绝性断言）
# =============================================================================
@testset "certificate negative controls" begin
    X = fill(0.99, 4, 2)
    # 正常解全绿
    w, w_cash, cert = cash_kelly(X)
    @test cash_kelly_certified(cert; tol=1e-8)
    # 可行但次优（w_risky = 0.5/0.5, cash = 0）：KKT 与 gap 必须红。
    # 解析：wealth ≡ 0.99, g_risky = 0.99 < g_cash = 1/0.99 ≈ 1.0101；
    # ν = 1.0101, complementarity ≈ 0.5·(1.0101−0.99) ≈ 0.01, gap ≈ 0.02.
    c_sub = cash_kelly_certificate(X, [0.5, 0.5], 0.0)
    @test !cash_kelly_certified(c_sub; tol=1e-8)
    @test c_sub.objective_gap > 1e-4
    @test c_sub.kkt_residual > 1e-4
    # 不可行（Σw_aug = 0.8 ≠ 1）：feasibility 必须红
    c_infeas = cash_kelly_certificate(X, [0.5, 0.2], 0.1)
    @test c_infeas.feasibility > 1e-8
    @test !cash_kelly_certified(c_infeas; tol=1e-8)
    # 负权重：feasibility 必须红
    c_neg = cash_kelly_certificate(X, [-0.1, 1.1], 0.0)
    @test !cash_kelly_certified(c_neg; tol=1e-8)
end

# =============================================================================
# 输入校验 fail-loudly（D-086 旧线语义沿用：log 域正性）
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
    # locked_wealth_gate0：持有资产缺预测律 → error（0*NaN 防护的持有侧）
    X_hold = [1.0 NaN; 1.0 NaN]
    @test_throws ErrorException locked_wealth_gate0(X_hold, [0.0, 0.3])
end

end  # module Gate0KellyCashTests
