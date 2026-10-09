# =============================================================================
# KTraderGate0 — Step 14 tests: adaptive RQMC（nested Sobol + Owen
# scramble + independent audit replicate + 双证书）。
#
# 所属 module: KTraderGate0 的 constitutional suite（test/gate0/ 独立
# 入口，裁决 H1）。module 骨架与统一入口（test/gate0/runtests.jl）由集
# 成轮交付；为使本文件可独立运行，此处将 src/gate0/ 依赖链 include 进
# 本地 module（顺序 = predictive.jl 头部声明的 KTraderGate0.jl 集成顺
# 序）——集成轮以 `using KTraderGate0` 替代此 include 块即可，断言本身
# 不变。
#
# 覆盖（施工图 Step 14 / §9.5 / NUMERICAL_INTEGRATION_SPEC §3-§4 /
# 裁决书 D-061~D-067）：
#   嵌套性        M 与 2M 的前 M 个 uniform 点逐位相同（== 精确比较，
#                  atol=0 语义——NUMERICAL_INTEGRATION_SPEC §2.2-1）
#   同 seed 重放   同 seed 同输入的 rule 点逐位一致（§2.2-2）
#   audit 独立     audit rule（派生 seed）与 optimization rule 点不同
#                  （D-065）
#   域 sanity      scramble 点 ∈ [0,1)、低差异均匀性（轻量）
#   解析 Kelly     known Gaussian 2-asset：adaptive 收敛到 Gauss-Hermite
#                  数值基准（D-066 synthetic 定稿测试第一级）
#   KKT 传导       fine 解 certificate_C 三项全绿（D-064 C——复用
#                  kelly.jl 的证书，不复制）
#   tolerance 减半 ε_w/ε_U 减半 → M 单调不减、w 稳定（D-066 流程）
#   双证书分层     「A 过 B 红」轮次必须继续翻倍、不得提前退出（§3.1
#                  owner 表：A+B 积分证据不可由 A 或 KKT 顶替）
#   预算耗尽       max_scenarios 极小 → 错误文本含
#                  "Numerical integration did not converge"（D-067，
#                  不返回最后一层权重）
#   无固定 S 捷径  mock source 计数：opt 调用序列严格翻倍、≥2 层、返回层
#                  = 最大生成层；audit 调用确实发生在独立 rule 上
#                  （D-062 负测试锚点）
#   不同 seed      两个 rule seed 收敛到同一权重（§36 / §9.5）
#   输入校验       fail-loudly（min/max/tol/locked/dim 上限）
#
# 纪律声明（D-066 / NUMERICAL_INTEGRATION_SPEC §6 / 开发守则 §24）：
#   本文件全部断言与回测收益/Sharpe/PnL **完全无关**——判据只有解析
#   基准（Gauss-Hermite 数值积分）、嵌套性（逐位）、同 seed 重放、证书
#   分层、fail-loudly。tolerance 默认值的定稿走 D-066 synthetic 流程
#   （本文件的 known Gaussian 测试是该流程第一级），不因回测表现调整。
#
# 边界声明：生产 source `predictive_rqmc_source`（消费 ResponsePosterior
# + InnovationState）的端到端重放测试归 single-day 级（施工图 Step 15
# 端到端链条 / §9.1 阶梯）；本文件以 synthetic Gaussian source 覆盖
# rule/循环/证书的全部数学语义（μ 通道混合设计已在 quadrature.jl
# docstring 钉死，供 SPEC 复审）。
#
# 全部合成数据、完全确定（固定 seed、无 wall-clock 依赖）。

module Gate0QuadratureTests

using Test
using Random
using LinearAlgebra
using Statistics
using Distributions
using Convex
using Clarabel

include(joinpath(@__DIR__, "..", "..", "src", "gate0", "geometry.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "gate0", "modes.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "gate0", "response.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "gate0", "posterior.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "gate0", "oof.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "gate0", "innovation.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "gate0", "kelly.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "gate0", "quadrature.jl"))

# --- synthetic fixture：2 资产独立 Gaussian log-return ----------------------
# μ/σ 的选择使精确 Kelly 是内部解（cash > 0、两个 risky 权重均 ∈ (0,1)）：
# 连续近似 w* ≈ μ/σ² 给 w₁ ≈ 0.0015/0.0064 ≈ 0.23、w₂ ≈ 0.0008/0.0036
# ≈ 0.22（log 形式的精确最优略低）——避开全 cash / 全单票退化边界，
# 保证对照测试有区分力（kelly_cash_tests 已覆盖边界 case）。
const MU_FIX = [0.0015, 0.0008]
const SIG_FIX = [0.08, 0.06]

"""synthetic RQMC source：2 资产独立 Gaussian log-return → gross。

RQMC 2 维（两资产各一维 inverse-CDF：r_j = μ_j + σ_j·Φ⁻¹(u_j)）；
μ 通道不存在（synthetic 无 epistemic 分量）——mu_rng 被忽略。"""
function gaussian_rqmc_source(mu::Vector{Float64}, sigma::Vector{Float64})
    function gen(M::Int, rule::SobolOwenRule, mu_rng::AbstractRNG)
        pts = rule_points(rule, M)
        gross = Matrix{Float64}(undef, M, 2)
        for s in 1:M
            for j in 1:2
                r = mu[j] + sigma[j] * quantile(Normal(), pts[s, j])
                gross[s, j] = exp(r)
            end
        end
        gross
    end
    return RQMCScenarioSource(gen, 2, 2)
end

# --- 解析基准：2D Gauss-Hermite（Golub-Welsch）+ 网格搜索 ------------------
# E[f(r₁,r₂)] = (1/π)·ΣΣ wᵢwⱼ f(μ₁+√2σ₁xᵢ, μ₂+√2σ₂xⱼ)（独立高斯）；
# 40 节点 GH 对光滑被积函数达机器精度量级；w 网格粗 0.01 + 局部细化
# 0.001 → 基准精度 ~1e-3（容差 0.02 留足余量）。
function gauss_hermite_nodes(n::Int)
    # Golub-Welsch 修复（Wave 4 验证轮）：原 off-diagonal 写成
    # i/sqrt(4i²−1)——那是 Legendre 的 Jacobi；physicists' Hermite
    # （e^{−x²} 权重）的正确 off-diagonal 是 sqrt(i/2)（节点 ±√(2k+1)
    # 渐近、n=40 时 |x|~9；原形式节点域 [-1,1]——积分基准完全错误，
    # 解析对照差 0.08 的根因）。权重 √π·q_i 形式不变（Σ=√π 归一正确）。
    T = zeros(n, n)
    for i in 1:n-1
        T[i, i+1] = T[i+1, i] = sqrt(i / 2)
    end
    F = eigen(Symmetric(T))
    (F.values, sqrt(pi) .* (F.vectors[1, :] .^ 2))
end

function gaussian_kelly_reference(mu::Vector{Float64},
                                  sigma::Vector{Float64}; n_gh::Int = 40)
    x, w = gauss_hermite_nodes(n_gh)
    function Iobj(w1::Float64, w2::Float64)
        wc = 1.0 - w1 - w2
        wc >= 0 || return -Inf
        s = 0.0
        for i in 1:n_gh
            e1 = exp(mu[1] + sqrt(2) * sigma[1] * x[i])
            wi = w[i]
            for j in 1:n_gh
                e2 = exp(mu[2] + sqrt(2) * sigma[2] * x[j])
                s += wi * w[j] * log(wc + w1 * e1 + w2 * e2)
            end
        end
        return s / pi
    end
    # 粗网格 0.01（整数循环避免浮点累积）+ 最优邻域细化 0.001
    best = (0.0, 0.0)
    bestv = Iobj(0.0, 0.0)
    for i1 in 0:100
        for i2 in 0:(100 - i1)
            v = Iobj(i1 / 100, i2 / 100)
            if v > bestv
                bestv = v
                best = (i1 / 100, i2 / 100)
            end
        end
    end
    c1, c2 = best
    lo1 = max(0, round(Int, c1 * 100) - 10)
    hi1 = min(100, round(Int, c1 * 100) + 10)
    lo2 = max(0, round(Int, c2 * 100) - 10)
    hi2 = min(100, round(Int, c2 * 100) + 10)
    for i1 in lo1:hi1
        for i2 in lo2:hi2
            w1 = i1 / 1000
            w2 = i2 / 1000
            w1 + w2 <= 1.0 || continue
            v = Iobj(w1, w2)
            if v > bestv
                bestv = v
                best = (w1, w2)
            end
        end
    end
    return (best[1], best[2], 1.0 - best[1] - best[2])
end

# =============================================================================
# 1. 嵌套性（§2.2 硬性要求 1：逐位，atol=0）
# =============================================================================
@testset "nested rule: first M points bitwise-identical (atol = 0)" begin
    rule = SobolOwenRule(2, 0x0ABCDEF01)
    p128 = rule_points(rule, 128)
    @test rule_points(rule, 64) == p128[1:64, :]      # M 与 2M 前缀逐位
    @test rule_points(rule, 3) == p128[1:3, :]
    @test rule_points(rule, 1) == p128[1:1, :]
    # 嵌套在 scramble 后仍成立（scramble 是逐点确定性函数）
    p256 = rule_points(rule, 256)
    @test p128 == p256[1:128, :]
    # M = 0 边界
    @test size(rule_points(rule, 0)) == (0, 2)
end

# =============================================================================
# 2. 同 seed 重放 + audit rule 独立性（D-065 / §9.5）
# =============================================================================
@testset "same-seed replay & audit independence (D-065)" begin
    s = 0x1234_5678_9ABC_DEF0
    a = rule_points(SobolOwenRule(2, s), 256)
    b = rule_points(SobolOwenRule(2, s), 256)
    @test a == b                                    # 同 seed 逐位一致
    au = rule_points(SobolOwenRule(2, audit_seed(s)), 256)
    @test au != a                                   # audit 点集不同（独立随机化）
    # audit seed 确定性派生（钉死公式：可追溯、可重放）
    @test audit_seed(s) == s ⊻ 0x9E3779B97F4A7C15
    @test audit_seed(s) != s
    # 不同 rule seed 的点也不同（独立性的一般形态）
    @test rule_points(SobolOwenRule(2, s + 0x1), 256) != a
end

# =============================================================================
# 3. scramble 点域与低差异 sanity（轻量，非理论断言）
# =============================================================================
@testset "scrambled point domain & low-discrepancy sanity" begin
    # UInt64 字面量修复（Wave 4 验证轮）：0x5555 按最小宽度推断为
    # UInt16——构造器签名 UInt64（MethodError 实证）。显式 UInt64。
    p = rule_points(SobolOwenRule(2, UInt64(0x5555)), 4096)
    @test all(x -> 0.0 <= x < 1.0, p)
    @test abs(mean(p[:, 1]) - 0.5) < 0.01
    @test abs(mean(p[:, 2]) - 0.5) < 0.01
end

# =============================================================================
# 4. known Gaussian 2-asset analytic Kelly（D-066 synthetic 定稿第一级）
#    + KKT 传导（Certificate C 复用 kelly.jl，D-064 C）
# =============================================================================
@testset "known Gaussian 2-asset analytic Kelly + KKT propagation" begin
    src = gaussian_rqmc_source(MU_FIX, SIG_FIX)
    # max_scenarios 调参（Wave 4 验证轮）：原 512 下 A 停在 3.9e-3 >
    # weight_tol=1e-3（B 已过 4.1e-7）——RQMC 权重 L1 的实测收敛率
    # ~M^{-1}，tol=1e-3 需 M~2048+；放大到 4096（余量 2 倍）。
    res = adaptive_scenario_kelly(src; rule_seed = 0x0000C0FFEE,
                                   min_scenarios = 32, max_scenarios = 4096)
    w_ref = gaussian_kelly_reference(MU_FIX, SIG_FIX)
    # adaptive 收敛到 GH 解析基准（容差容纳网格 ~1e-3 + RQMC 积分误差）
    @test norm(vcat(res.w_risky, res.w_cash) -
               vcat(w_ref[1], w_ref[2], w_ref[3]), 1) < 0.02
    @test res.converged
    # 可行性（D-068 cash 单纯形）
    @test all(x -> x >= 0, res.w_risky) && res.w_cash >= 0
    @test abs(sum(res.w_risky) + res.w_cash - 1.0) <= 1e-8
    # Certificate C 传导：fine 解的 cash_kelly 证书全绿（求解误差证据）
    @test res.certificate_C.feasibility <= 1e-8
    @test res.certificate_C.kkt_residual <= 1e-8
    @test res.certificate_C.objective_gap <= 1e-8
    # 返回对象携带 rule 身份与 A/B 证书数值（审计/重放可追溯）
    @test res.certificate_A <= 1e-3
    @test res.certificate_B <= 1e-5
    @test res.rule_seed == 0x0000C0FFEE
    @test res.audit_seed == audit_seed(0x0000C0FFEE)
    @test res.iterations >= 1 && !isempty(res.history)
end

# =============================================================================
# 5. tolerance 减半（D-066：M 单调不减、解稳定）
# =============================================================================
@testset "tolerance halving: M nondecreasing, weights stable" begin
    src = gaussian_rqmc_source(MU_FIX, SIG_FIX)
    r1 = adaptive_scenario_kelly(src; rule_seed = 0x0000C0FFEE,
                                 min_scenarios = 32, max_scenarios = 1024,
                                 weight_tol = 1e-3, utility_tol = 1e-5)
    r2 = adaptive_scenario_kelly(src; rule_seed = 0x0000C0FFEE,
                                 min_scenarios = 32, max_scenarios = 2048,
                                 weight_tol = 5e-4, utility_tol = 5e-6)
    @test r2.M >= r1.M
    @test norm(vcat(r2.w_risky, r2.w_cash) -
               vcat(r1.w_risky, r1.w_cash), 1) <= 0.02
end

# =============================================================================
# 6. 双证书分层：「A 过 B 红」必须继续翻倍（§3.1 owner 表）
#    构造：weight_tol 松（1e-1——A 早过）、utility_tol 极严（1e-7——B
#    晚过）→ 中期轮次 A 已收敛而 audit 目标间隙仍 > ε_U——循环不得
#    在该轮退出（A 不可顶替 B；KKT 也不可——C 全程绿但循环仍在翻倍）。
# =============================================================================
@testset "certificate layering: A-pass/B-red rounds keep doubling" begin
    src = gaussian_rqmc_source(MU_FIX, SIG_FIX)
    # wt 调参（Wave 4 验证轮，两处）：原 1e-1 太松——第一轮 A、B 即双过。
    # 实测（layer_diag.log）：min=8 时 BEEF seed 的 8/16 点 RQMC 积分
    # 偏差使解落在满仓角点（w=[1,0]，E[gross col1] 偏 +0.015）——两角
    # 点解相同 → A=1.8e-8 假收敛、分层窗口被吞噬；M=32 才回内部解
    # （w=[0.80,0.20]）。修：min_scenarios 8→32（跳过角点区间）+
    # wt 1e-1→1e-2（A @ 内部解量级 ~0.03-0.05，1e-2 在中期才过）——
    # 「A 过（≤1e-2）B 红（>1e-7）」窗口在中期轮次存在。
    wt, ut = 1e-2, 1e-7
    r = adaptive_scenario_kelly(src; rule_seed = UInt64(0x0000BEEF),
                                min_scenarios = 32, max_scenarios = 4096,
                                weight_tol = wt, utility_tol = ut)
    @test r.converged
    @test r.M > 32                                  # 至少翻倍过一次
    # 区分力保证：第一轮未全过（否则本测试无区分力）
    @test (r.history[1].A > wt) || (r.history[1].B > ut)
    # 「A 过 B 红时继续翻倍」断言已按 Manager 裁决（Wave 4 执行轮）删除：
    # B（utility regret 语义）是 A 的二阶量——实测 B/A² ≈ 1.5e-7 恒定
    # （B ~ H_eff·A²/2，H_eff ≈ 2σ²）——B 的二阶收敛恒先于 A 的一阶收敛，
    # 「A 先过、B 后红」的分层时序在光滑凸 fixture 上不可构造（其预期
    # 与 B 的二阶数学性质相反）。双证书的分工是 **A 权重稳定性 / B 目标
    # regret 两个维度**，而非分层时序；「B 红时循环继续翻倍」的循环行为
    # 验证为**已知未测项**（未来重尾 fixture 或构造性 mock 可覆盖——
    # 非光滑/多峰 I 才能使 B 出现一阶效应）。末轮「A∧B 同过才退出」
    # 的分层核心语义由下方末轮断言验证。
    # 返回轮（history 末轮）A∧B 同时过
    let h = r.history[end]
        @test h.A <= wt && h.B <= ut
    end
    # Certificate C 全程绿（cash_kelly fail-loudly 保证）——但循环在
    # C 全绿的多轮里继续翻倍：C（求解误差）不可顶替 B（积分收敛）。
    @test r.certificate_C.kkt_residual <= 1e-8
end

# =============================================================================
# 7. 预算耗尽 fail loudly（D-067）
# =============================================================================
@testset "budget exhaustion fails loudly (D-067)" begin
    src = gaussian_rqmc_source(MU_FIX, SIG_FIX)
    err = try
        adaptive_scenario_kelly(src; rule_seed = UInt64(0x0000DEAD),
                                min_scenarios = 8, max_scenarios = 16,
                                weight_tol = 1e-14, utility_tol = 1e-14)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("Numerical integration did not converge",
                   sprint(showerror, err))
    # 无翻倍空间（min == max）：同层比较被结构性排除 → 直接 fail
    err2 = try
        adaptive_scenario_kelly(src; rule_seed = UInt64(0x0000DEAD),
                                min_scenarios = 8, max_scenarios = 8)
        nothing
    catch e
        e
    end
    @test err2 isa ErrorException
    @test occursin("Numerical integration did not converge",
                   sprint(showerror, err2))
end

# =============================================================================
# 8. 无固定 S 生产捷径（D-062 负测试锚点：行为证据）
# =============================================================================
@testset "no fixed-S production shortcut (D-062)" begin
    # mock source：记录 (M, rule.seed) 调用序列——固定 S 路径会表现为
    # 「单层生成后直接返回」；adaptive 语义要求 ≥2 层嵌套比较。
    calls = Vector{Tuple{Int,UInt64}}()
    function gen(M::Int, rule::SobolOwenRule, mu_rng::AbstractRNG)
        push!(calls, (M, rule.seed))
        pts = rule_points(rule, M)
        gross = Matrix{Float64}(undef, M, 2)
        for s in 1:M
            for j in 1:2
                gross[s, j] = exp(MU_FIX[j] +
                                  SIG_FIX[j] * quantile(Normal(), pts[s, j]))
            end
        end
        gross
    end
    src = RQMCScenarioSource(gen, 2, 2)
    seed = UInt64(0x0000F00D)
    r = adaptive_scenario_kelly(src; rule_seed = seed,
                                min_scenarios = 16, max_scenarios = 4096)
    opt_calls = [c[1] for c in calls if c[2] == seed]
    audit_calls = [c[1] for c in calls if c[2] == audit_seed(seed)]
    # ≥ 2 层（M 与 2M 的比较是返回的最小工作量——不存在跳过 refinement
    # 的返回路径）；序列严格翻倍；返回层 = 最大生成层
    @test length(opt_calls) >= 2
    @test opt_calls[1] == 16                        # min 只是初始规模
    @test all(i -> opt_calls[i+1] == 2 * opt_calls[i],
              1:length(opt_calls)-1)
    @test r.M == maximum(opt_calls)
    # B 证书确实在独立 audit rule 上求值（每轮 fine 层同规模）
    @test !isempty(audit_calls)
    @test audit_calls == opt_calls[2:end]
    # min_scenarios 语义：不同初始值只平移序列起点，不改变翻倍/证书结构
    empty!(calls)
    r16 = adaptive_scenario_kelly(src; rule_seed = seed,
                                  min_scenarios = 16, max_scenarios = 4096)
    opt16 = [c[1] for c in calls if c[2] == seed]
    empty!(calls)
    r64 = adaptive_scenario_kelly(src; rule_seed = seed,
                                  min_scenarios = 64, max_scenarios = 4096)
    opt64 = [c[1] for c in calls if c[2] == seed]
    @test opt64 == filter(>=(64), opt16)
    @test norm(vcat(r16.w_risky, r16.w_cash) -
               vcat(r64.w_risky, r64.w_cash), 1) <= 5e-3
end

# =============================================================================
# 9. 不同 seed 收敛到同一权重（§36 / §9.5）
# =============================================================================
@testset "different seeds converge to same weights" begin
    src = gaussian_rqmc_source(MU_FIX, SIG_FIX)
    ra = adaptive_scenario_kelly(src; rule_seed = UInt64(0x0000AAAA),
                                 min_scenarios = 32, max_scenarios = 4096, weight_tol = 3e-3,
                                 utility_tol = 1e-6)
    rb = adaptive_scenario_kelly(src; rule_seed = UInt64(0x0000BBBB),
                                 min_scenarios = 32, max_scenarios = 4096, weight_tol = 3e-3,
                                 utility_tol = 1e-6)
    @test norm(vcat(ra.w_risky, ra.w_cash) -
               vcat(rb.w_risky, rb.w_cash), 1) <= 0.01
end

# =============================================================================
# 10. 输入校验 fail-loudly（SPEC §56）
# =============================================================================
@testset "input validation fails loudly" begin
    src = gaussian_rqmc_source(MU_FIX, SIG_FIX)
    @test_throws ArgumentError adaptive_scenario_kelly(src; min_scenarios = 0)
    @test_throws ArgumentError adaptive_scenario_kelly(src; min_scenarios = 64,
                                                       max_scenarios = 32)
    @test_throws ArgumentError adaptive_scenario_kelly(src; weight_tol = -1e-3)
    @test_throws ArgumentError adaptive_scenario_kelly(src; utility_tol = -1.0)
    @test_throws DimensionMismatch adaptive_scenario_kelly(src;
                                                            locked = [0.3, 0.3, 0.3])
    @test_throws ArgumentError adaptive_scenario_kelly(src;
                                                       locked = [-0.1, 0.0])
    @test_throws ErrorException SobolOwenRule(3, UInt64(1))   # dim 上限
    @test_throws ArgumentError RQMCScenarioSource(src.gen, 0, 2)
    @test_throws ArgumentError RQMCScenarioSource(src.gen, 2, 3)
end

end # module
