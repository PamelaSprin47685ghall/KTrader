# =============================================================================
# KTraderGate0 — Step 2: cash feasible set exact log-Kelly.
#
# 所属 module: KTraderGate0。本文件由 src/gate0/KTraderGate0.jl 在集成轮
# include；module 骨架由并行 Engineer 交付。为保持文件自含，依赖在此声明：
#     using Convex, Clarabel, LinearAlgebra
#
# 数学对象（AGENTS.md 裁决书 D-068~D-071，施工图 Step 2）：
#   cash 是显式 numeraire asset，无利息 gross return = 1：
#
#     max_{w ≥ 0, w_c ≥ 0, Σ_i w_i + w_c = budget}
#         (1/S) Σ_s log(base_s + X_s·w + w_c)
#
#   本实现采用显式 cash 列的等价形式（与 D-068「cash 是显式 numeraire
#   asset」字面一致；两种表达数学等价，等价性由 test/gate0/kelly_cash_tests.jl
#   的预算不等式对照测试覆盖）：
#
#     X_aug = [X  1_S],   w_aug = [w; w_c]
#     max (1/S) Σ_s log(X_aug·w_aug + base)   s.t.  w_aug ≥ 0, Σ w_aug = budget
#
#   wealth 正性：base ≥ 0、X 正有限、w_aug 在预算单纯形上 ⟹
#   wealth_s ≥ w_c ≥ 0，且 wealth_s = 0 仅当 base_s = 0 ∧ w_c = 0 ∧
#   X_s·w = 0 —— X 正有限使后者不可能（除非 w 全零且 base_s = 0，该情形
#   由输入校验与证书的 wealth 正性检查 fail loudly 拒绝）。
#
# locked 语义（D-017/D-068；旧 SPEC §38.1 语义参考，旧线 src/kelly.jl:137-
# 148 只读参照）：locked 资产不在 free 优化列中；其 scenario wealth 贡献
#   base_s = Σ_{j: locked} X[s,j]·w_locked[j]
# 由 locked_wealth_gate0 构造（0*NaN 永不进入 locked wealth）；budget =
# 1 − Σ w_locked 由调用方计算传入。locked 风险绝不当 cash。
#
# D-069: fully-invested risky simplex（Σw=1、无 cash）降级为显式实验约束，
#        不是本函数的默认理论。
# D-070: cash 是 exact feasible set 的一部分——不是 fractional Kelly、
#        不是置信度乘数、不是 0.5 Kelly、不是 volatility targeting。
# D-071: 不引入 max_weight / entropy penalty / diversification penalty /
#        risk parity blend——任何形式都不出现在本文件。
#
# fail loudly 纪律（裁决书 §56）：输入校验、求解器状态、证书三处全部
# error，不返回 heuristic 权重。
# =============================================================================

using Convex
using Clarabel
using LinearAlgebra

"""
    cash_kelly_inputs(X, budget, base) -> Vector{Float64}

Input validation for the cash-column Kelly problem. Fails loudly on:
- empty scenario matrix or zero free assets;
- non-finite or non-positive budget;
- any gross return that is not positive and finite (log domain, D-086 旧线
  语义沿用);
- locked wealth (base) that is not non-negative and finite, or whose
  scenario count does not match X.
Returns the effective base vector (zeros when `base === nothing`).
"""
function cash_kelly_inputs(X::Matrix{Float64}, budget::Real, base)
    S, n = size(X)
    S > 0 && n > 0 || throw(ArgumentError("cash Kelly needs scenarios and free assets"))
    isfinite(budget) && budget >= 0 ||
        throw(ArgumentError("invalid cash Kelly budget: $budget"))
    all(x -> isfinite(x) && x > 0, X) ||
        throw(ArgumentError("gross returns must be positive and finite"))
    b = base === nothing ? zeros(S) : base
    length(b) == S || throw(DimensionMismatch("locked wealth scenario count"))
    all(x -> isfinite(x) && x >= 0, b) ||
        throw(ArgumentError("invalid locked wealth: must be non-negative and finite"))
    b
end

"""
    cash_kelly_certificate(X, w, w_cash; base=nothing, budget=1.0)

Certificate for the explicit-cash-column problem, evaluated on the ORIGINAL
log objective (not any barrier reformulation). Three checks:

1. `feasibility`: |Σw_aug − budget| and min(w_aug) ≥ 0 violation.
2. `kkt_residual`: complementarity w_i·(ν − g_i) where
   g = ∇F on the augmented columns (g_cash = (1/S)Σ_s 1/wealth_s included)
   and ν = max(g). At the optimum, positive-weight columns satisfy g_i = ν
   and zero-weight columns satisfy g_i ≤ ν.
3. `objective_gap`: concavity gives F(w*) − F(w) ≤ budget·max(g) − gᵀw_aug.

Returns a NamedTuple `(feasibility, kkt_residual, objective_gap, objective,
dual)`. Non-finite weights or non-positive wealth short-circuit to an
all-Infinity certificate (which `cash_kelly_certified` rejects).
"""
function cash_kelly_certificate(X::Matrix{Float64}, w::Vector{Float64},
                                w_cash::Float64; base=nothing, budget=1.0)
    S, n = size(X)
    length(w) == n || throw(DimensionMismatch("cash Kelly weights"))
    b = cash_kelly_inputs(X, budget, base)
    w_aug = vcat(w, [Float64(w_cash)])
    feasibility = max(abs(sum(w_aug) - budget), max(0.0, -minimum(w_aug)))
    wealth = X * w .+ w_cash .+ b
    if !all(isfinite, w_aug) || !all(x -> isfinite(x) && x > 0, wealth)
        return (; feasibility=Inf, kkt_residual=Inf, objective_gap=Inf,
                 objective=-Inf, dual=Inf)
    end
    inv_wealth = 1.0 ./ wealth
    # Gradient of (1/S) Σ log(wealth) w.r.t. w_aug: risky columns plus the
    # cash column (d wealth_s / d w_cash = 1).
    g = vcat(X' * inv_wealth, [sum(inv_wealth)]) ./ S
    dual = maximum(g)
    complementarity = w_aug .* (dual .- g)
    gap = max(0.0, budget * dual - dot(w_aug, g))
    (; feasibility=feasibility,
       kkt_residual=maximum(abs, complementarity),
       objective_gap=gap,
       objective=sum(log, wealth) / S,
       dual=dual)
end

"""Certificate acceptance: all three checks within `tol` (default 1e-8)."""
cash_kelly_certified(c; tol=1e-8) =
    c.feasibility <= tol && c.kkt_residual <= tol && c.objective_gap <= tol

# =============================================================================
# Newton polish — Clarabel 解的 KKT 数值精修（同一 log 目标，非 heuristic）
# =============================================================================
# 动机（Wave 2 集成运行实证，archive/evidence/gate0_wave2/kelly_cash_tests.log）：
# Convex 的 exp-cone 建模 + Clarabel 在内部解上的可达 KKT 精度约 1e-7
# （Case C 实测 kkt_residual = objective_gap = 1.14e-7，解本身与解析值一致
# 到同一量级），低于证书默认门槛 tol = 1e-8。本函数对**同一数学对象**做
# 纯数值精修：固定 Clarabel 近似解的支撑集 S，在其上解 KKT 方程组
#     g_S(u) = ν·1,   Σ_{i∈S} u_i = budget,
# 其中 g_i = (1/S)Σ_s A[s,i]/wealth_s（A = [X 1] 增广列），Jacobian 使用
# 精确 Hessian H = (1/S)Σ_s a_s a_s^T / wealth_s²。对正确支撑 Newton 二次
# 收敛，一步内将 1e-7 残差打到机器精度。
# 纪律（D-060 / fail-loudly）：不改变目标、可行集或证书定义；仅在
# Clarabel 解未过证书时尝试，精修解通过同一证书才采纳，否则保留
# Clarabel 原解走原 error 路径——绝不返回 heuristic 权重，绝不放松 tol。
function _cash_kelly_newton_polish(X::Matrix{Float64}, b::Vector{Float64},
                                   w_val::Vector{Float64}, budget::Float64;
                                   max_iters::Int = 10,
                                   stationarity::Float64 = 1e-13)
    S, n = size(X)
    A = hcat(X, ones(S))
    m = n + 1
    length(w_val) == m || return nothing
    # 支撑判定阈值 1e-6：Clarabel（内点法）对真零分量返回 ~1e-8 量级的
    # 数值尘埃（D-090 实测 w_cash ≈ 5e-8），阈值过松会把尘埃列误纳入
    # Newton 支撑——该支撑上不存在驻点（尘埃列 g_i ≠ ν 恒成立），Newton
    # 无法收敛、精修失效。1e-6 区分「Clarabel 认为零」与「真实支撑」；
    # 若真解确有 < 1e-6 的合法小权重被误排除，精修解的证书会红、不
    # 被采纳（闭环防护），行为退回未精修路径。
    supp = findall(>(1e-6), w_val)
    isempty(supp) && return copy(w_val)
    k = length(supp)
    As = A[:, supp]                          # S × k（支撑上的增广列）
    u = w_val[supp]                          # 初值 = Clarabel 解的支撑分量
    best_u = copy(u)
    best_res = Inf
    for _ in 1:max_iters
        wealth = As * u .+ b
        all(x -> isfinite(x) && x > 0, wealth) || break
        g = (As' * (1.0 ./ wealth)) ./ S     # 支撑上的梯度（k 维）
        nu = sum(g) / k                      # 当前对偶估计（驻点处 g_i ≡ ν；
                                              #  不用 Statistics.mean——文件
                                              #  依赖面保持 Convex/Clarabel/
                                              #  LinearAlgebra 不变）
        r = vcat(g .- nu, [sum(u) - budget])
        res = maximum(abs, r)
        if res < best_res
            best_res = res
            best_u .= u
        end
        res <= stationarity && break
        # Newton 系统 [-H -1; 1' 0]·[du; dν] = -r（J·Δ = -F：残差 F 的
        # 线性化解，注意右侧为**负**残差——符号反了会朝远离驻点方向走）
        H = ((As ./ wealth)' * As) ./ S      # k×k 精确 Hessian（对称 PSD）
        M = zeros(k + 1, k + 1)
        M[1:k, 1:k] .= -H
        M[1:k, k + 1] .= -1.0
        M[k + 1, 1:k] .= 1.0
        sol = M \ (-r)
        du = sol[1:k]
        # 全步；若离开正 wealth 域则步长减半重试（精修场景下极少触发）
        step = 1.0
        advanced = false
        for _ in 1:4
            u_try = u .+ step .* du
            if all(x -> x >= -1e-12, u_try) &&
               all(x -> isfinite(x) && x > 0, As * u_try .+ b)
                u = u_try
                advanced = true
                break
            end
            step /= 2
        end
        advanced || break
    end
    best_res == Inf && return nothing        # 首次域检查即失败
    w_out = zeros(m)
    w_out[supp] .= best_u
    return w_out
end

"""
    cash_kelly(X; base=nothing, budget=1.0, tol=1e-8)
        -> (w_risky::Vector{Float64}, w_cash::Float64, certificate)

Exact log-Kelly with cash as an explicit numeraire asset (D-068).

Solves, via Convex.jl modelling + Clarabel (reference-grade solver, same
objective family as the legacy `clarabel_kelly_solver` but on the augmented
cash-column problem):

    max_{w_aug ≥ 0, Σ w_aug = budget} (1/S) Σ_s log(X_aug·w_aug + base)

with X_aug = [X ones(S)]. Returns the risky weights, the cash weight, and
the certificate. Fails loudly (error) when the solver status is not optimal,
when the returned allocation is degenerate, or when the certificate fails —
no heuristic weights are ever returned.

`budget` defaults to 1.0 (the full D-068 simplex w_c + Σw_i = 1). Callers
holding locked positions pass budget = 1 − Σ w_locked together with the
locked scenario wealth in `base` (see `locked_wealth_gate0`); this is the
D-017/D-068 locked semantics — locked wealth participates in wealth but not
in the free optimization.
"""
function cash_kelly(X::Matrix{Float64};
                    base::Union{Nothing,Vector{Float64}}=nothing,
                    budget=1.0, tol=1e-8)
    b = cash_kelly_inputs(X, budget, base)
    S, n = size(X)

    # Degenerate budget: the only feasible point is w_aug = 0. Its
    # certificate is trivially feasible (gap = 0·dual − 0 = 0, complementarity
    # vanishes); if some base_s = 0 makes the log domain ill-defined, the
    # certificate reports Infinity and we fail loudly below.
    if budget == 0
        w_risky = zeros(n)
        w_cash = 0.0
        cert = cash_kelly_certificate(X, w_risky, w_cash; base=b, budget=0.0)
        cash_kelly_certified(cert; tol=tol) ||
            error("cash Kelly certificate failed on degenerate budget: $cert")
        return (w_risky, w_cash, cert)
    end

    w_aug = Variable(n + 1)
    X_aug = hcat(X, ones(S))
    wealth = X_aug * w_aug + b
    problem = maximize(sum(log(wealth)) / S,
                       [w_aug >= 0, sum(w_aug) == budget])
    optimizer = Clarabel.Optimizer()
    # max_iter=1000（最后一轮修复，2026-10-09）：Clarabel 默认 200 迭代
    # 在个别数据形态下 SLOW_PROGRESS（60-day 段 6 首日 t=327 的 Kelly
    # 问题实测停滞——seg6_day_diag.log）。提高迭代上限是求解器配置
    # 鲁棒性修复：数学语义不动、fail-loudly 保留（真解不出仍红、
    # 证书门槛不放松）。
    for setting in ("tol_gap_abs", "tol_gap_rel", "tol_feas")
        Convex.MOI.set(optimizer, Convex.MOI.RawOptimizerAttribute(setting),
                       min(tol / 10, 1e-10))
    end
    Convex.MOI.set(optimizer, Convex.MOI.RawOptimizerAttribute("max_iter"),
                   1000)
    solve!(problem, () -> optimizer; silent=true)
    problem.status in (Convex.MOI.OPTIMAL, Convex.MOI.ALMOST_OPTIMAL) ||
        error("Clarabel cash log-Kelly failed: $(problem.status)")

    # Post-processing: clip solver-level negative dust to zero and restore
    # the exact budget. This touches only feasibility rounding of the SAME
    # solution — no objective change, no re-allocation rule.
    w_val = max.(vec(evaluate(w_aug)), 0.0)
    total = sum(w_val)
    total > 0 || error("cash Kelly solver returned a degenerate allocation")
    w_val .*= budget / total

    cert = cash_kelly_certificate(X, w_val[1:n], w_val[n + 1];
                                  base=b, budget=budget)
    if !cash_kelly_certified(cert; tol=tol)
        # Newton polish（同一 KKT 系统的数值精修，见函数 docstring）：仅当
        # 精修解通过同一证书（tol 不变）时采纳；否则保留 Clarabel 原解走
        # 原 error 路径——fail loudly 行为与未引入精修时完全一致。
        w_polished = _cash_kelly_newton_polish(X, b, w_val, budget)
        if w_polished !== nothing
            cert_p = cash_kelly_certificate(X, w_polished[1:n],
                                            w_polished[n + 1];
                                            base=b, budget=budget)
            if cash_kelly_certified(cert_p; tol=tol)
                w_val = w_polished
                cert = cert_p
            end
        end
    end

    w_risky = w_val[1:n]
    w_cash = w_val[n + 1]
    cash_kelly_certified(cert; tol=tol) ||
        error("cash Kelly certificate failed: $cert")
    (w_risky, w_cash, cert)
end

"""
    locked_wealth_gate0(X, locked) -> Vector{Float64}

Locked scenario wealth base_s = Σ_{j: locked[j] > 0} locked[j]·X[s,j].

`X` is the FULL scenario matrix (all risk-domain columns, free and locked);
`locked` is the held-weight vector over those columns. Only genuinely held
columns are multiplied — 0*NaN must never enter locked wealth (legacy
src/kelly.jl:137-148 semantics, D-017). A held asset without a finite
predictive gross return is an error, never a silent zero.
"""
function locked_wealth_gate0(X::Matrix{Float64}, locked::Vector{Float64})
    S, n = size(X)
    length(locked) == n || throw(DimensionMismatch("locked weight count"))
    base = zeros(S)
    for j in eachindex(locked)
        locked[j] > 0 || continue
        for s in 1:S
            gross = X[s, j]
            isfinite(gross) ||
                error("held asset $j has no predictive return law")
            base[s] += locked[j] * gross
        end
    end
    base
end
