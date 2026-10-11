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
# P0-9（裁决书 §P0-9）：cash Kelly canonical tie-break。当
#   U(w1) ≈ U(w2) 甚至完全相等时，返回哪个最优点不能由 Clarabel /
#   numerical support 决定。本文件在 cash_kelly 主求解后做二阶段
#   canonicalization：
#     第一阶段：U* = max_w U(w)（主求解给出，含 cash）；
#     第二阶段：在数值等价最优集 W_ε = {w : U(w) ≥ U* − tie_eps}
#               上最大化 cash weight w_c；
#     第三阶段：在相同最大 w_c 下最小化 ‖w_risky‖₂²。
#   两阶段都是凸问题，用与主求解相同的 Convex + Clarabel 技术栈。
#   tie_eps 默认 1e-8（初始候选，留待 D-066 数值 refinement 定稿，
#   禁止由回测选择——SPEC §95 / 开发守则 §24）。
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
# 精确 Hessian H = (1/S)Σ_s a_s a_s^T / wealth_s²。对正确支撑 Newton 通常
# 快速收敛（典型几步内到机器精度）；但病态尺度下实测可退化为近似线性
# （P1 判决：cap#3 的 kkt 1.98e-6 → 10 轮 1.08e-6 → 50 轮 7.2e-8 →
# 200 轮 2.8e-12；archive/evidence/gate0_multiday_run_20261010/P1_P4_*）。
# 故轮数上限提至 200、在 stationarity（1e-13）处早退——正常用例几步内
# 退出、零额外成本（每轮 k 小、微秒级）；病态用例获得所需深度。
# 纪律（D-060 / fail-loudly）：不改变目标、可行集或证书定义；仅在
# Clarabel 解未过证书时尝试，精修解通过同一证书才采纳，否则保留
# Clarabel 原解走原 error 路径——绝不返回 heuristic 权重，绝不放松 tol。
function _cash_kelly_newton_polish(X::Matrix{Float64}, b::Vector{Float64},
                                   w_val::Vector{Float64}, budget::Float64;
                                   max_iters::Int = 200,
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
    # 支撑列数不得超过场景数 S（防护恢复，来源：archive/evidence/
    # gate0_t327_fix_20261010/kelly_jl_pre_polish_rewrite.jl:182-190）：
    # k > S 时 H = (1/S)Σ a_s a_sᵀ/w_s² 的秩 ≤ S < k，KKT 矩阵必然
    # 奇异。按确定性规则把最小权重分量视为数值尘埃剔除，直到 k ≤ S；
    # 剩余奇异由下方安全求解兜底为「返回 nothing、保留 Clarabel 原解」
    # （精修既有契约；fail-safe、不改数学/证书）。
    while length(supp) > S
        deleteat!(supp, argmin(view(w_val, supp)))
    end
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
        # 安全求解（防护恢复，来源同上 :227-234）：M 在支撑列行空间
        # 线性相关时精确奇异——精修契约是「失败返回 nothing、保留
        # Clarabel 原解走原 error 路径」，故奇异/非有限解必须走 break，
        # 绝不让异常泄漏到调用方（fail-safe）。
        sol = nothing
        try
            sol = M \ (-r)
        catch err
            err isa LinearAlgebra.SingularException || rethrow()
        end
        sol === nothing && break
        all(isfinite, sol) || break
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

# =============================================================================
# KELLY-SCALE-RETRY-1（2026-10-10；设计 docs/KELLY_NUMERICAL_ROW_SCALING.md §8）
# =============================================================================
# 证书驱动的缩放重试：首选（未缩放）solve → 证书 → fail → 缩放路径重试一次
# → 原空间证书 → 过则采纳；首选 status 非白名单保持既有 error（不重试）；
# 已缩放仍 fail → 不重试 → polish → 既有失败链。数学对象/门禁/容差不变；
# 无 RNG；默认零行为变化（不给诊断 kw 时行为与改造前逐位一致）。
# 上下文（G2 实证，archive/evidence/gate0_multiday_run_20261010/）：t=330 的
# 1024 层输入 span=3.03e9（阈值下侧）原路径 ALMOST_OPTIMAL、kkt=1.9557e-6
# 不过；同一输入缩放路径 OPTIMAL、kkt=5.475e-9 过——中间窗口 [~1e8, 4.5e15)
# 存在。阈值不动（避免路径归属漂移）；以证书质量为触发条件覆盖该窗口。

"""
    KellyScaleDiagnostics — KELLY-SCALE-RETRY-1 诊断载体（不导出；经
    cash_kelly 的 scale_retry_diag kw 传入可变对象就地填充）。

字段：preferred（:scaled/:unscaled；默认构造为 :unset）、span
（max(cs)/min(cs)）、attempted（是否执行了缩放重试）、rescued（重试解是否
通过原空间证书并被采纳）；设计 A 追加 adopted/raw_quality/scaled_quality
（采纳路径与双路径证书质量 q = max(fes, kkt, gap)；无候选 = Inf）、
M2 追加 adopted_from_exit（采纳解是否源于 status 出口候选）。默认
nothing → 零行为变化。
"""
mutable struct KellyScaleDiagnostics
    preferred::Symbol
    span::Float64
    attempted::Bool
    rescued::Bool
    # 设计 A（§9）：采纳方与双路径证书质量（q=max(fes,kkt,gap)；无候选=Inf）。
    adopted::Symbol
    raw_quality::Float64
    scaled_quality::Float64
    # M2（§10）：采纳的解除法是否源于 status 出口候选（status 非白名单但解有限）。
    adopted_from_exit::Bool
end
KellyScaleDiagnostics() =
    KellyScaleDiagnostics(:unset, NaN, false, false, :unset, Inf, Inf, false)

# 纯决策 helper（可单测）：首选未缩放且证书 fail → 需要缩放重试。
_cash_kelly_needs_retry(preferred_scaled::Bool, cert, tol::Float64) =
    !preferred_scaled && !cash_kelly_certified(cert; tol = tol)

# TIE-FAST-1（§9.9 候选 a）：全 cash 精确快速路径的判据（纯函数，
# 直测阈值语义）。1e-12 是安全界：只捕获精确零/机器级尘埃。
_cash_kelly_skip_tiebreak(w_risky::AbstractVector{Float64}) =
    all(<(1e-12), w_risky)

# --- 设计 A（§9）辅助：证书质量 / 候选比较 / 路径序（确定性；无 RNG） ---
# q = max(feasibility, kkt, gap)；tie 阈值 _KELLY_CANDIDATE_TIE（待 D-066 定稿）。
const _KELLY_CANDIDATE_TIE = 1e-15
_kelly_cert_quality(c) = max(c.feasibility, c.kkt_residual, c.objective_gap)
_kelly_candidate_rank(p::Symbol) =
    p === :raw ? 1 : p === :raw_polished ? 2 : p === :scaled ? 3 : 4
function _pick_better_kelly_candidate(a, b)
    a.certified != b.certified && return a.certified ? a : b
    qa = _kelly_cert_quality(a.cert)
    qb = _kelly_cert_quality(b.cert)
    if abs(qa - qb) < _KELLY_CANDIDATE_TIE
        return _kelly_candidate_rank(a.path) <= _kelly_candidate_rank(b.path) ? a : b
    end
    qa < qb ? a : b
end


# 求解器装配 + 求解 + status 白名单 + clip/预算归一的内部 helper（原
# cash_kelly 主体抽出；数值语义逐字保留，含 max_iter=1000 注释语义）。
# 返回 (; w_val, status_ok, reason, status)。reason ∈ (:ok, :status,
# :status_exit, :degenerate)：:ok = 白名单且 w_val 有效；:status_exit
# （M2）= status 非白名单但提取到有限出口解（status_ok=false 且 w_val
# 非 nothing）；:status = 非白名单且提取失败；:degenerate = 归一失败。
# 由调用方决定采纳 / 放弃（重试）或 error（首选）。
function _cash_kelly_solve(Xt::Matrix{Float64}, bt::Vector{Float64},
                           cash_col::Vector{Float64}, budget::Float64,
                           tol::Float64)
    S, n = size(Xt)
    m = n + 1
    w_aug = Variable(m)
    X_aug = hcat(Xt, cash_col)
    wealth = X_aug * w_aug + bt
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
    # M2（§10）：status 非白名单时先尝试提取出口解（有限且 sum>0 → 归一）；
    # 提取失败/无解 → 既有 return 不变（fail-loud 由调用方按 reason 决定）。
    if !(problem.status in (Convex.MOI.OPTIMAL, Convex.MOI.ALMOST_OPTIMAL))
        w_exit = nothing
        try
            w_exit = max.(vec(evaluate(w_aug)), 0.0)
        catch
            w_exit = nothing
        end
        if w_exit !== nothing && all(isfinite, w_exit) && sum(w_exit) > 0
            w_exit .*= budget / sum(w_exit)
            return (; w_val = w_exit, status_ok = false, reason = :status_exit,
                      status = problem.status)
        end
        return (; w_val = nothing, status_ok = false, reason = :status,
                  status = problem.status)
    end

    # Post-processing: clip solver-level negative dust to zero and restore
    # the exact budget. This touches only feasibility rounding of the SAME
    # solution — no objective change, no re-allocation rule.
    w_val = max.(vec(evaluate(w_aug)), 0.0)
    total = sum(w_val)
    total > 0 ||
        return (; w_val = nothing, status_ok = false, reason = :degenerate,
                  status = nothing)
    w_val .*= budget / total
    (; w_val = w_val, status_ok = true, reason = :ok, status = problem.status)
end

function _cash_kelly_finish_main(X::Matrix{Float64}, pre)
    # B2B-SPLIT-1/2（§9.10）：本函数是 cash_kelly 的「出口链」段（段 II）——从首选
    # 求解的返回值起，含 status 处理/出口提取/设计 A 候选/缩放重试与 polish/取优/
    # 最终证书检查，并组装 mid 快照（含 skip_tb 判据）。tie-break 段已由
    # B2B-SPLIT-2 移出至 _cash_kelly_finish_tiebreak（本函数不再调用 tie-break）。
    # 组合（prelude + finish_main + finish_tiebreak）= 原 cash_kelly 逐点一致。
    pre.early === nothing || return pre.early
    b = pre.b
    budget = pre.budget
    tol = pre.tol
    tie_eps = pre.tie_eps
    scale_retry_diag = pre.scale_retry_diag
    cs = pre.cs
    scale_preferred = pre.scale_preferred
    res = pre.res
    S, n = size(X)
    # M2（§10）：status 非白名单但出口解存在 → 进入设计 A 候选集（证书验收
    # 决定去留）；exit status 记入诊断与错误文本。无出口解 → 既有 error。
    p0_status_exit = false
    exit_status_any = nothing
    if !res.status_ok
        if res.reason == :status_exit
            p0_status_exit = true
            exit_status_any = res.status
        elseif res.reason == :status
            error("Clarabel cash log-Kelly failed: $(res.status)")
        else
            error("cash Kelly solver returned a degenerate allocation")
        end
    end
    w_val = res.w_val

    cert = cash_kelly_certificate(X, w_val[1:n], w_val[n + 1];
                                  base=b, budget=budget)
    # --- KELLY-SCALE-RETRY-1 设计 A（§9）：双路径候选 + 各自 polish + 取优 ---
    # 候选：首选路径（raw 或已缩放，标签随装配）→ 未过者 polish；再按需缩放
    # 重试 + polish。统一比较键：过证书者优先；同档 q=max(fes,kkt,gap) 小者
    # 优先；|Δq| < _KELLY_CANDIDATE_TIE 用固定路径序（raw < raw_polished <
    # scaled < scaled_polished）。**采纳较优者，但较优者仍须过证书**——绝不
    # 因「取优」放宽验收；全不过 → fail-loud（双路径诊断，见下方 final 检查）。
    p0 = scale_preferred ? :scaled : :raw
    mk_cand(wv, cv, path::Symbol, from_exit::Bool) =
        (; w = wv, cert = cv, certified = cash_kelly_certified(cv; tol = tol),
           path = path, from_exit = from_exit)
    need_scl = _cash_kelly_needs_retry(scale_preferred, cert, tol)
    cands = [mk_cand(w_val, cert, p0, p0_status_exit)]
    if !cands[1].certified
        wp = _cash_kelly_newton_polish(X, b, w_val, budget)
        if wp !== nothing
            cp = cash_kelly_certificate(X, wp[1:n], wp[n + 1];
                                        base=b, budget=budget)
            push!(cands, mk_cand(wp, cp,
                                 p0 === :raw ? :raw_polished : :scaled_polished,
                                 p0_status_exit))
        end
    end
    if need_scl
        scale_retry_diag === nothing || (scale_retry_diag.attempted = true)
        res2 = _cash_kelly_solve(X ./ reshape(cs, :, 1), b ./ cs, 1.0 ./ cs,
                                 budget, tol)
        scl_status_exit = res2.reason == :status_exit
        if res2.status_ok || scl_status_exit
            scl_status_exit && exit_status_any === nothing &&
                (exit_status_any = res2.status)
            c2 = cash_kelly_certificate(X, res2.w_val[1:n], res2.w_val[n + 1];
                                        base=b, budget=budget)
            push!(cands, mk_cand(res2.w_val, c2, :scaled, scl_status_exit))
            if !cash_kelly_certified(c2; tol = tol)
                wp2 = _cash_kelly_newton_polish(X, b, res2.w_val, budget)
                if wp2 !== nothing
                    cp2 = cash_kelly_certificate(X, wp2[1:n], wp2[n + 1];
                                                 base=b, budget=budget)
                    push!(cands, mk_cand(wp2, cp2, :scaled_polished,
                                         scl_status_exit))
                end
            end
        end
    end
    best = cands[1]
    for c in cands[2:end]
        best = _pick_better_kelly_candidate(best, c)
    end
    w_val = best.w
    cert = best.cert
    rq_all = Inf; sq_all = Inf
    for c in cands
        q = _kelly_cert_quality(c.cert)
        if c.path in (:raw, :raw_polished)
            rq_all = min(rq_all, q)
        else
            sq_all = min(sq_all, q)
        end
    end
    if scale_retry_diag !== nothing
        scale_retry_diag.adopted = best.path
        scale_retry_diag.rescued =
            !scale_preferred && best.path in (:scaled, :scaled_polished)
        scale_retry_diag.raw_quality = rq_all
        scale_retry_diag.scaled_quality = sq_all
        scale_retry_diag.adopted_from_exit = best.from_exit
    end

    w_risky = w_val[1:n]
    w_cash = w_val[n + 1]
    if !cash_kelly_certified(cert; tol=tol)
        any_exit = any(c -> c.from_exit, cands)
        base_msg = "cash Kelly certificate failed (dual-path candidate selection): adopted=$(best.path) raw_quality=$rq_all scaled_quality=$sq_all"
        if any_exit
            error("Clarabel cash log-Kelly failed: $exit_status_any; exit candidate not certified: " *
                  base_msg * " — 出口解未过证书（含 polish）；不返回未认证权重（D-067 语义）")
        end
        error(base_msg * " — 双路径候选均未过证书；不返回未认证权重（D-067 语义）")
    end

    # P0-9 canonical tie-break（见 cash_kelly docstring）：主求解已给出
    # U*（cert.objective）；canonicalization 本体（tie_eps > 0 时的
    # cash-max / risky-L2-min 两阶段）已由 B2B-SPLIT-2 移出至
    # _cash_kelly_finish_tiebreak。tie-break 失败（求解器状态非 optimal、
    # 或 canonical 点未过同一证书）时**保留主解**并如实报告，绝不静默
    # 换入未验证点、也绝不因 tie-break 失败而 error 掉一个已经通过证书的
    # 合法主解（fail loudly 只约束主求解路径本身）。本段只计算 skip_tb
    # 判据（TIE-FAST-1）并组装 mid 快照；canonicalization 调用与采纳在
    # tie-break 段。
    # TIE-FAST-1（§9.9 候选 a）：全 cash 主解的精确快速路径。等价性：若主解
    # risky 分量全部 ≤ 1e-12（w_val 已由 _cash_kelly_solve clip 非负），则
    # w_cash = budget − Σw_risky 已是 ε-等价集内现金最大值、risky L2 = 0 已是
    # 最小——canonicalization（Stage 2/3）与主解逐位一致，跳过两次大 Clarabel
    # 不改变任何输出。仅此一路径改变；非全 cash 照旧走完整 tie-break
    # （canonicalization 定义不动）。阈值 1e-12 是安全界：只捕获精确零/机器
    # 级尘埃，不吞内点法 1e-8 级数值尘埃（那些照旧由 tie-break 规范）。
    skip_tb = _cash_kelly_skip_tiebreak(view(w_val, 1:n))
    (; b = b, budget = budget, tol = tol, tie_eps = tie_eps,
       w_val = w_val, cert = cert, w_risky = w_risky, w_cash = w_cash,
       skip_tb = skip_tb)
end

function _cash_kelly_finish_tiebreak(X::Matrix{Float64}, pre, mid)
    # B2B-SPLIT-2（§9.10）：tie-break 段——调用与采纳逻辑逐字保持；失败（nothing）
    # 或 skip_tb 时保留主解（mid.w_risky/mid.w_cash/mid.cert）。pre 保留供签名
    # 对称（当前全部输入经 mid 承载）。
    S, n = size(X)
    w_risky = mid.w_risky
    w_cash = mid.w_cash
    cert = mid.cert
    if mid.tie_eps > 0 && !mid.skip_tb
        tb = _cash_kelly_canonical_tiebreak(X, mid.b, mid.budget, mid.cert.objective,
                                            mid.w_val, Float64(mid.tie_eps), mid.tol)
        if tb !== nothing
            w_risky = tb[1][1:n]
            w_cash = tb[1][n + 1]
            cert = tb[2]
        end
    end
    (w_risky, w_cash, cert)
end

function _cash_kelly_finish(X::Matrix{Float64}, pre)
    # B2B-SPLIT-2 修复：early 快照（退化预算直通）不得进入 tie-break 段——
    # 恢复 B2B-SPLIT-1 的短路（重构时遗漏；与「组合 = 原 cash_kelly 逐点一致」
    # 的契约一致：early 路径本就直通返回三元组）。
    pre.early === nothing || return pre.early
    mid = _cash_kelly_finish_main(X, pre)
    _cash_kelly_finish_tiebreak(X, pre, mid)
end

function _cash_kelly_prelude_prep(X::Matrix{Float64};
                             base::Union{Nothing,Vector{Float64}}=nothing,
                             budget=1.0, tol=1e-8, tie_eps::Real=1e-8,
                             scale_retry_diag::Union{Nothing,KellyScaleDiagnostics}=nothing)
    # B2B-SPLIT-1/3（§9.10）：段 I（prep）——校验/退化分支/行尺度规范化/首选分流；
    # 求解已由 B2B-SPLIT-3 拆出至 _cash_kelly_prelude_solve（段 I′）。返回 prep
    # 快照供求解段；组合（prep + solve + finish）= 原 cash_kelly 逐点一致。
    tie_eps >= 0 || throw(ArgumentError("cash Kelly tie_eps must be ≥ 0 (got $tie_eps)"))
    b = cash_kelly_inputs(X, budget, base)
    S, n = size(X)
    if budget == 0
        w_risky = zeros(n)
        w_cash = 0.0
        cert = cash_kelly_certificate(X, w_risky, w_cash; base=b, budget=0.0)
        cash_kelly_certified(cert; tol=tol) ||
            error("cash Kelly certificate failed on degenerate budget: $cert")
        return (; early = (w_risky, w_cash, cert), b = b, budget = Float64(budget),
                  tol = Float64(tol), tie_eps = Float64(tie_eps),
                  scale_retry_diag = scale_retry_diag, cs = Float64[],
                  scale_preferred = false)
    end
    # 行尺度规范化（纯数值；数学恒等，见 KELLY_NUMERICAL §2.1）。
    cs = Vector{Float64}(undef, S)
    for s in 1:S
        v = abs(b[s])
        for j in 1:n
            a = abs(X[s, j]); a > v && (v = a)
        end
        cs[s] = v
    end
    # 条件启用（D-091：温和 span 走原路径；极端 span 走缩放装配）。
    span = maximum(cs) / minimum(cs)
    scale_preferred = span > 1.0 / eps(Float64)
    if scale_retry_diag !== nothing
        scale_retry_diag.preferred = scale_preferred ? :scaled : :unscaled
        scale_retry_diag.span = span
        scale_retry_diag.attempted = false
        scale_retry_diag.rescued = false
    end
    (; early = nothing, b = b, budget = Float64(budget), tol = Float64(tol),
       tie_eps = Float64(tie_eps), scale_retry_diag = scale_retry_diag,
       cs = cs, scale_preferred = scale_preferred)
end

function _cash_kelly_prelude_solve(X::Matrix{Float64}, prep)
    # B2B-SPLIT-3（§9.10）：主求解段——Xt/bt/cash_col 由 prep 与 X 重算（不落
    # 大矩阵；确定性）；early 快照直通。组合 = 原 _cash_kelly_prelude 逐点一致。
    if prep.early !== nothing
        return (; early = prep.early, b = prep.b, budget = prep.budget,
                  tol = prep.tol, tie_eps = prep.tie_eps,
                  scale_retry_diag = prep.scale_retry_diag, cs = prep.cs,
                  scale_preferred = prep.scale_preferred,
                  res = (; w_val = nothing, status_ok = false, reason = :early,
                           status = nothing))
    end
    S, n = size(X)
    if prep.scale_preferred
        Xt = X ./ reshape(prep.cs, :, 1)
        bt = prep.b ./ prep.cs
        cash_col = 1.0 ./ prep.cs
    else
        Xt = X
        bt = prep.b
        cash_col = ones(S)
    end
    res = _cash_kelly_solve(Xt, bt, cash_col, prep.budget, prep.tol)
    (; early = nothing, b = prep.b, budget = prep.budget, tol = prep.tol,
       tie_eps = prep.tie_eps, scale_retry_diag = prep.scale_retry_diag,
       cs = prep.cs, scale_preferred = prep.scale_preferred, res = res)
end

# B2B-SPLIT-1/3（§9.10）：段 I 组合——_cash_kelly_prelude_prep + _cash_kelly_prelude_solve
# （拆出前本函数为单一整段；组合与拆出前逐点一致）。
function _cash_kelly_prelude(X::Matrix{Float64};
                             base::Union{Nothing,Vector{Float64}}=nothing,
                             budget=1.0, tol=1e-8, tie_eps::Real=1e-8,
                             scale_retry_diag::Union{Nothing,KellyScaleDiagnostics}=nothing)
    prep = _cash_kelly_prelude_prep(X; base=base, budget=budget, tol=tol,
                                    tie_eps=tie_eps,
                                    scale_retry_diag=scale_retry_diag)
    _cash_kelly_prelude_solve(X, prep)
end

"""
    cash_kelly(X; base=nothing, budget=1.0, tol=1e-8, tie_eps=1e-8)
        -> (w_risky::Vector{Float64}, w_cash::Float64, certificate)

Exact log-Kelly with cash as an explicit numeraire asset (D-068).

B2B-SPLIT-1/2/3（§9.10）：本入口由 prelude 段（`_cash_kelly_prelude` =
`_cash_kelly_prelude_prep` + `_cash_kelly_prelude_solve`）与 finish 段
（`_cash_kelly_finish` = `_cash_kelly_finish_main` + `_cash_kelly_finish_tiebreak`）
组合而成；组合调用与拆分前逐点一致（kelly_cash_tests 的 B2B-SPLIT-1 testset
锁定；批量调度按段分批执行同一条链）。

Solves, via Convex.jl modelling + Clarabel (reference-grade solver, same
objective family as the legacy `clarabel_kelly_solver` but on the augmented
cash-column problem):

    max_{w_aug ≥ 0, Σ w_aug = budget} (1/S) Σ_s log(X_aug·w_aug + base)

with X_aug = [X ones(S)]. Returns the risky weights, the cash weight, and
the certificate. Fails loudly (error) when no admissible allocation can be
certified: a non-whitelisted solver status without an extractable exit
candidate, a degenerate allocation, or all candidate paths failing the
ORIGINAL certificate — no heuristic weights are ever returned. (M2: a
non-whitelisted status whose exit solution is finite enters the design-A
candidate set and is decided by the certificate.)

`budget` defaults to 1.0 (the full D-068 simplex w_c + Σw_i = 1). Callers
holding locked positions pass budget = 1 − Σ w_locked together with the
locked scenario wealth in `base` (see `locked_wealth_gate0`); this is the
D-017/D-068 locked semantics — locked wealth participates in wealth but not
in the free optimization.

# P0-9 canonical tie-break

When U(w1) ≈ U(w2) (or exactly equal), the raw solver's returned optimum is
decided by Clarabel / numerical support. P0-9 requires the canonical
representative of the optimum set instead. After the primary solve, when
`tie_eps > 0`, the finish chain (`_cash_kelly_finish_tiebreak`, B2B-SPLIT-2)
runs the two-stage canonicalization (`_cash_kelly_canonical_tiebreak`):

1. U* = max U(w) — already obtained as the primary certificate objective;
2. on the numerical tie set W_ε = {w_aug ≥ 0, Σw_aug = budget :
   U(w_aug) ≥ U* − tie_eps}, maximize the cash weight w_c;
3. among allocations with the maximal w_c, minimize ‖w_risky‖₂².

Both stages are convex (log-utility constraint is convex in w_aug; the
second-stage objective is convex quadratic) and are solved with the same
Convex + Clarabel stack. The final allocation is re-certified on the ORIGINAL
log objective; if the canonicalized point fails the certificate, the primary
solution is kept unchanged (the tie-break is a canonicalization refinement,
not a new decision authority — fail loudly applies to the primary solve only,
and a legal certified primary solution is never discarded because a
refinement stage did not converge).

TIE-FAST-1（§9.9 候选 a）：全 cash 主解（risky 分量全部 ≤ 1e-12）跳过
Stage 2/3 两次 Clarabel 求解，输出与 canonicalization 逐位一致。

`tie_eps` default 1e-8 is an INITIAL CANDIDATE — same order of magnitude as
the legacy utility tolerance (1e-5) but three orders tighter, so it only
activates on numerically indistinguishable optima. Its final value must be
fixed by the D-066 numerical refinement process (synthetic analytic laws →
single-day fixtures → halving → stability), never chosen by backtest
performance (SPEC §95 / 开发守则 §24).

KELLY-SCALE-RETRY-1（2026-10-10；设计 KELLY_NUMERICAL §8）：证书驱动的
缩放重试——首选（未缩放）路径通过 status 白名单但未过原空间证书时，以
行缩放装配重解一次（数学恒等、原空间证书、同一 tol）；首选 status 非
白名单保持既有 error（不重试）；已缩放仍 fail 不重试（落入 Newton polish
链）。可选 kw scale_retry_diag 接收 KellyScaleDiagnostics
（preferred/span/attempted/rescued）；默认 nothing 时行为逐位不变。
G2 证据：span=3.03e9（阈值下侧）原路径 kkt=1.9557e-6 不过、缩放路径
kkt=5.475e-9 过——中间窗口 [~1e8, 4.5e15)；条件缩放阈值不动。
"""
function cash_kelly(X::Matrix{Float64};
                    base::Union{Nothing,Vector{Float64}}=nothing,
                    budget=1.0, tol=1e-8, tie_eps::Real=1e-8,
                    scale_retry_diag::Union{Nothing,KellyScaleDiagnostics}=nothing)
    pre = _cash_kelly_prelude(X; base=base, budget=budget, tol=tol,
                              tie_eps=tie_eps, scale_retry_diag=scale_retry_diag)
    _cash_kelly_finish(X, pre)
end

"""
    _cash_kelly_canonical_tiebreak(X, b, budget, Ustar, w_primary, tie_eps, tol)
        -> Union{Nothing,Tuple{Vector{Float64},NamedTuple}}

P0-9 two-stage canonicalization on the numerical tie set
W_ε = {w_aug ≥ 0, Σw_aug = budget : U(w_aug) ≥ U* − tie_eps}, where
U(w_aug) = (1/S) Σ_s log(X_aug·w_aug + b) is the SAME log objective.

Stage 2 (cash-max):        max  w_c
                           s.t. w_aug ≥ 0, Σw_aug = budget,
                                U(w_aug) ≥ U* − tie_eps.
Stage 3 (risky L2-min):    min  ‖w_risky‖₂²
                           s.t. w_aug ≥ 0, Σw_aug = budget,
                                U(w_aug) ≥ U* − tie_eps,
                                w_c = w_c_max (from stage 2).

Both stages are convex: U is concave, so the constraint U ≥ U* − tie_eps
defines a convex upper level set; the stage-2 objective (w_c) is affine and
the stage-3 objective (‖w_risky‖₂²) is convex quadratic. Solved with the
same Convex + Clarabel stack as the primary problem.

`w_primary` is used only as a dimension guard: the canonicalization must
operate on the same augmented weight space (length n + 1) as the primary
solution; a mismatch returns `nothing` (the caller keeps the primary
solution) rather than silently solving on a different space.

Returns `nothing` when either stage fails to reach OPTIMAL/ALMOST_OPTIMAL
or when the canonical point fails the ORIGINAL certificate — the caller then
keeps the primary solution (tie-break is a canonicalization refinement, not
a new decision authority; fail loudly applies to the primary solve only).
The returned certificate is re-computed on the canonical point so downstream
consumers see the true objective of the returned allocation.
"""
function _cash_kelly_canonical_tiebreak(X::Matrix{Float64},
                                        b::Vector{Float64},
                                        budget::Float64,
                                        Ustar::Float64,
                                        w_primary::Vector{Float64},
                                        tie_eps::Float64,
                                        tol::Float64)
    S, n = size(X)
    m = n + 1
    length(w_primary) == m || return nothing
    X_aug = hcat(X, ones(S))
    U_lower = Ustar - tie_eps

    # --- Stage 2: maximize cash weight on W_ε ---
    w2 = Variable(m)
    wealth2 = X_aug * w2 + b
    U2 = sum(log(wealth2)) / S
    p2 = maximize(w2[m],
                  [w2 >= 0,
                   sum(w2) == budget,
                   U2 >= U_lower])
    opt2 = Clarabel.Optimizer()
    for setting in ("tol_gap_abs", "tol_gap_rel", "tol_feas")
        Convex.MOI.set(opt2, Convex.MOI.RawOptimizerAttribute(setting),
                       min(tol / 10, 1e-10))
    end
    Convex.MOI.set(opt2, Convex.MOI.RawOptimizerAttribute("max_iter"), 1000)
    solve!(p2, () -> opt2; silent=true)
    p2.status in (Convex.MOI.OPTIMAL, Convex.MOI.ALMOST_OPTIMAL) ||
        return nothing
    w2_val = max.(vec(evaluate(w2)), 0.0)
    wc_max = w2_val[m]

    # --- Stage 3: minimize risky L2 on W_ε ∩ {w_c = wc_max} ---
    w3 = Variable(m)
    wealth3 = X_aug * w3 + b
    U3 = sum(log(wealth3)) / S
    p3 = minimize(sumsquares(w3[1:n]),
                  [w3 >= 0,
                   sum(w3) == budget,
                   U3 >= U_lower,
                   w3[m] == wc_max])
    opt3 = Clarabel.Optimizer()
    for setting in ("tol_gap_abs", "tol_gap_rel", "tol_feas")
        Convex.MOI.set(opt3, Convex.MOI.RawOptimizerAttribute(setting),
                       min(tol / 10, 1e-10))
    end
    Convex.MOI.set(opt3, Convex.MOI.RawOptimizerAttribute("max_iter"), 1000)
    solve!(p3, () -> opt3; silent=true)
    p3.status in (Convex.MOI.OPTIMAL, Convex.MOI.ALMOST_OPTIMAL) ||
        return nothing
    w3_val = max.(vec(evaluate(w3)), 0.0)
    total3 = sum(w3_val)
    total3 > 0 || return nothing
    w3_val .*= budget / total3

    # Re-certify on the ORIGINAL log objective; only adopt when green.
    cert3 = cash_kelly_certificate(X, w3_val[1:n], w3_val[n + 1];
                                   base=b, budget=budget)
    cash_kelly_certified(cert3; tol=tol) || return nothing
    (w3_val, cert3)
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
