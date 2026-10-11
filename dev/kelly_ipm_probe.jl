#!/usr/bin/env julia
# =============================================================================
# dev/kelly_ipm_probe.jl
#
# cash Kelly（log-sum over simplex）专用求解核心 —— LOGSUM-SIMPLEX-IPM-1
#
# 状态：一次性独立验证器（2026-10-11 起冻结）
# ----
# 本探针的一次性验证使命已完成：其算法已由生产实现取代——src/gate0/
# kelly.jl 的 _cash_kelly_ipm（装配进 _cash_kelly_solve 双 backend，
# :auto 默认）。真实 t=338 输入（131072×2）验证事实：46 迭代、0.105s、
# 内部证书 kkt=gap=2.756e-9；生产 cash_kelly_certificate 复核差 9e-15 级。
# 本文件保留为**冻结的独立验证器**：不随生产演进、无需同步维护；任何修改
# 都必须重新走一遍验证（算法常量/tol/迭代上限与生产实现的一致性由人复核，
# 不再有自动同步）。
# 已知差异（生产 vs 本探针，2026-10-11 接线时点）：生产 _cash_kelly_ipm 的
# μ_floor 因子为 1e-3（本探针 0.01），且返回点绑定「证书过 AND μ≤floor」
# （本探针证书过即返）——为把角点残余小权重压到 1e-9 量级以对齐下游
# tie-break 的既有 1e-8 差值契约。两处均为数值深度选择，证书口径不变。
#
# 与生产的关系（同构输入、零耦合）
# ----
# * 输入约定与生产一致：增广矩阵 A = [X | cash_col] ∈ R^{S×m}（cash 为末列），
#   locked scenario wealth b（可省略），budget；行缩放（KELLY-SCALE-RETRY /
#   条件缩放）由调用方装配，本脚本只解「输入即求解空间」，并在大规模宽尾
#   案例中演示缩放前后对照。
# * 证书三量与 src/gate0/kelly.jl 的 cash_kelly_certificate 同式
#   （本脚本内联对照实现）；不加载 Convex/Clarabel 与生产模块。
# * 算法无 RNG；合成数据用确定性 splitmix64；同一输入逐位可复现。
# * 探针只打印，不写文件、不改生产状态。
#
# 用法
# ----
#   julia --startup-file=no dev/kelly_ipm_probe.jl
#   julia --startup-file=no dev/kelly_ipm_probe.jl <input.txt>
#
# 输入文件格式（可选）
# ----
#   第 1 行：S m budget [tol]
#   随后 S 行：每行 m 个数（增广矩阵行 a_s；cash 列须为 1 或已按行缩放后的
#              1/c_s，与生产装配一致）
#   可选随后 S 行：每行 1 个数（base b_s）
#   '#' 开头行与空行被忽略。
# =============================================================================

using LinearAlgebra
using Printf

# -----------------------------------------------------------------------------
# 0. 确定性伪随机（无 RNG）：splitmix64 + 均匀 [0,1)
# -----------------------------------------------------------------------------
function splitmix64(x::UInt64)::UInt64
    z = x + 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

const INV2P53 = 1.1102230246251565e-16  # 2^-53
uniform01(x::UInt64)::Float64 = Float64(x >> 11) * INV2P53

# -----------------------------------------------------------------------------
# 1. 证书（与 src/gate0/kelly.jl:102-126 同式；A 为增广矩阵）
# -----------------------------------------------------------------------------
function probe_certificate(A::Matrix{Float64}, b::AbstractVector{Float64},
                            w::AbstractVector{Float64}, budget::Float64)
    S, m = size(A)
    z = A * w .+ b
    feasibility = max(abs(sum(w) - budget), max(0.0, -minimum(w)))
    if !all(isfinite, w) || !all(x -> isfinite(x) && x > 0.0, z)
        return (; feasibility=Inf, kkt_residual=Inf, objective_gap=Inf,
                 objective=-Inf, dual=Inf)
    end
    invz = 1.0 ./ z
    g = (A' * invz) ./ S
    dual = maximum(g)
    kkt = maximum(abs, w .* (dual .- g))
    gap = max(0.0, budget * dual - dot(w, g))
    obj = sum(log, z) / S
    (; feasibility=feasibility, kkt_residual=kkt, objective_gap=gap,
       objective=obj, dual=dual)
end

probe_certified(c; tol::Float64=1e-8) =
    c.feasibility <= tol && c.kkt_residual <= tol && c.objective_gap <= tol

# 复用扫描所得梯度/目标值，避免二次 O(S) 扫描（主循环用）
function cert_from_g(g::Vector{Float64}, w::Vector{Float64},
                     budget::Float64, objective::Float64)
    dual = maximum(g)
    feasibility = max(abs(sum(w) - budget), max(0.0, -minimum(w)))
    kkt = maximum(abs, w .* (dual .- g))
    gap = max(0.0, budget * dual - dot(w, g))
    (; feasibility=feasibility, kkt_residual=kkt, objective_gap=gap,
       objective=objective, dual=dual)
end

# -----------------------------------------------------------------------------
# 2. 手写 Cholesky（m×m 对称正定；确定性、无 LAPACK 依赖）
# -----------------------------------------------------------------------------
function chol_solve(M::Matrix{Float64}, rhs::Vector{Float64})
    n = length(rhs)
    size(M) == (n, n) || return nothing
    L = zeros(n, n)
    for i in 1:n
        for j in 1:i
            s = M[i, j]
            for k in 1:(j - 1)
                s -= L[i, k] * L[j, k]
            end
            if i == j
                (isfinite(s) && s > 0.0) || return nothing
                L[i, j] = sqrt(s)
            else
                L[i, j] = s / L[j, j]
            end
        end
    end
    y = zeros(n)
    for i in 1:n
        s = rhs[i]
        for k in 1:(i - 1)
            s -= L[i, k] * y[k]
        end
        y[i] = s / L[i, i]
    end
    x = zeros(n)
    for i in n:-1:1
        s = y[i]
        for k in (i + 1):n
            s -= L[k, i] * x[k]
        end
        x[i] = s / L[i, i]
    end
    all(isfinite, x) ? x : nothing
end

# -----------------------------------------------------------------------------
# 3. 障碍值 φ_μ（Armijo 回溯判据用）
# -----------------------------------------------------------------------------
function barrier_value(A::Matrix{Float64}, b::Vector{Float64},
                       w::Vector{Float64}, mu::Float64)
    S, m = size(A)
    logsum = 0.0
    for s in 1:S
        zs = b[s]
        for i in 1:m
            zs += A[s, i] * w[i]
        end
        (isfinite(zs) && zs > 0.0) || return (NaN, false)
        logsum += log(zs)
    end
    phi = logsum / S
    for i in 1:m
        (isfinite(w[i]) && w[i] > 0.0) || return (NaN, false)
        phi += mu * log(w[i])
    end
    (phi, true)
end

# -----------------------------------------------------------------------------
# 4. 核心：原始对数障碍 IPM
#     max (1/S)Σ log(a_s·w + b_s)   s.t. w ≥ 0, 1'w = budget
#     返回 (; w, converged, reason, iters, cert, iter_ms, mu_final, trajectory)
#     终止判据 = 原空间证书三量 ≤ tol（与生产同一口径；无第二口径）。
# -----------------------------------------------------------------------------
function ipm_logsum_simplex(A::Matrix{Float64}, budget::Float64;
                             b::Union{Nothing,AbstractVector{Float64}}=nothing,
                             w0::Union{Nothing,AbstractVector{Float64}}=nothing,
                             tol::Float64=1e-8,
                             mu0::Float64=1.0,
                             sigma::Float64=0.2,
                             mu_floor_factor::Float64=0.01,
                             max_outer::Int=60,
                             max_newton::Int=4,
                             tau::Float64=0.99)
    S, m = size(A)
    S > 0 || throw(ArgumentError("empty scenario matrix"))
    m > 0 || throw(ArgumentError("empty column set"))
    (isfinite(budget) && budget > 0.0) ||
        throw(ArgumentError("invalid budget: $budget"))
    bb::Vector{Float64} = b === nothing ? zeros(S) : collect(Float64, b)
    length(bb) == S || throw(DimensionMismatch("base length"))

    # m == 1：唯一可行点 w = budget（全 cash）
    if m == 1
        w1 = [budget]
        c1 = probe_certificate(A, bb, w1, budget)
        return (; w=w1, converged=probe_certified(c1; tol=tol),
                reason=:m1, iters=0, cert=c1, iter_ms=Float64[],
                mu_final=NaN, trajectory=Float64[])
    end

    if w0 === nothing
        w = fill(budget / m, m)
    else
        w = collect(Float64, w0)
        length(w) == m || throw(DimensionMismatch("w0 length"))
    end
    floor_w = 1e-14 * max(budget, 1.0)
    for i in 1:m
        w[i] = max(w[i], floor_w)
    end
    w .*= budget / sum(w)

    mu = mu0
    mu_floor = max(mu_floor_factor * tol, 1e-12)
    iters = 0
    iter_ms = Float64[]
    traj = Float64[]
    nu = NaN

    for _outer in 1:max_outer
        for _inner in 1:max_newton
            t0 = time_ns()
            iters += 1

            # ---- 一遍扫描：g、H（下三角）、Σ log z ----
            g = zeros(m)
            H = zeros(m, m)
            logsum = 0.0
            scan_ok = true
            for s in 1:S
                zs = bb[s]
                for i in 1:m
                    zs += A[s, i] * w[i]
                end
                if !(isfinite(zs) && zs > 0.0)
                    scan_ok = false
                    break
                end
                logsum += log(zs)
                invz = 1.0 / zs
                for i in 1:m
                    ai = A[s, i]
                    g[i] += ai * invz
                    for j in 1:i
                        H[i, j] += ai * A[s, j] * invz * invz
                    end
                end
            end
            if !scan_ok
                push!(iter_ms, (time_ns() - t0) / 1e6)
                return (; w=w, converged=false, reason=:nonfinite_wealth,
                        iters=iters,
                        cert=probe_certificate(A, bb, w, budget),
                        iter_ms=iter_ms, mu_final=mu, trajectory=traj)
            end
            g ./= S
            for i in 1:m
                for j in (i + 1):m
                    H[i, j] = H[j, i]
                end
            end
            H ./= S
            objective_now = logsum / S
            phi_cur = objective_now
            for i in 1:m
                phi_cur += mu * log(w[i])
            end

            if isnan(nu)
                nu = maximum(g)
            end

            # ---- 证书检查（复用 g；原空间口径）----
            cert_now = cert_from_g(g, w, budget, objective_now)
            push!(traj, cert_now.objective_gap)
            if probe_certified(cert_now; tol=tol)
                push!(iter_ms, (time_ns() - t0) / 1e6)
                return (; w=w, converged=true, reason=:certified,
                        iters=iters, cert=cert_now, iter_ms=iter_ms,
                        mu_final=mu, trajectory=traj)
            end

            # ---- 牛顿方向（Schur 形式）----
            #   [H + μ W⁻²  1] [Δw]   [ r_dual ]
            #   [1'         0] [Δν] = [ −r_prim ]
            M = Matrix{Float64}(undef, m, m)
            for i in 1:m
                for j in 1:m
                    M[i, j] = H[i, j]
                end
                M[i, i] += mu / (w[i] * w[i])
            end
            r_dual = Vector{Float64}(undef, m)
            for i in 1:m
                r_dual[i] = g[i] + mu / w[i] - nu
            end
            r_prim = sum(w) - budget

            x1 = chol_solve(M, r_dual)
            x2 = chol_solve(M, ones(m))
            if x1 === nothing || x2 === nothing
                push!(iter_ms, (time_ns() - t0) / 1e6)
                return (; w=w, converged=false, reason=:singular_kkt,
                        iters=iters, cert=cert_now, iter_ms=iter_ms,
                        mu_final=mu, trajectory=traj)
            end
            denom = sum(x2)
            if !(isfinite(denom) && denom > 0.0)
                push!(iter_ms, (time_ns() - t0) / 1e6)
                return (; w=w, converged=false, reason=:singular_kkt,
                        iters=iters, cert=cert_now, iter_ms=iter_ms,
                        mu_final=mu, trajectory=traj)
            end
            dnu = (sum(x1) + r_prim) / denom
            dw = x1 .- dnu .* x2
            if !all(isfinite, dw)
                push!(iter_ms, (time_ns() - t0) / 1e6)
                return (; w=w, converged=false, reason=:nonfinite_direction,
                        iters=iters, cert=cert_now, iter_ms=iter_ms,
                        mu_final=mu, trajectory=traj)
            end

            # ---- 分数步长（保持 w > 0）----
            alpha = 1.0
            for i in 1:m
                if dw[i] < 0.0
                    tmax = -w[i] / dw[i]
                    tmax < alpha && (alpha = tmax)
                end
            end
            if alpha < 1.0
                alpha *= tau
            end

            # ---- Armijo 回溯（φ_μ 不下降）----
            accepted = false
            w_new = similar(w)
            for _ls in 1:12
                for i in 1:m
                    w_new[i] = w[i] + alpha * dw[i]
                end
                (phi_try, ok_try) = barrier_value(A, bb, w_new, mu)
                if ok_try && isfinite(phi_try) &&
                   phi_try >= phi_cur - 1e-10 * (1.0 + abs(phi_cur))
                    accepted = true
                    break
                end
                alpha *= 0.5
            end
            if !accepted
                push!(iter_ms, (time_ns() - t0) / 1e6)
                return (; w=w, converged=false, reason=:line_search_failed,
                        iters=iters, cert=cert_now, iter_ms=iter_ms,
                        mu_final=mu, trajectory=traj)
            end

            # ---- 接受 + 对偶更新 + 预算清理（O(m)）----
            for i in 1:m
                w[i] = w_new[i]
            end
            nu += alpha * dnu
            delta = budget - sum(w)
            if abs(delta) > 0.0
                w[m] += delta
                if w[m] <= 0.0
                    tot = sum(w)
                    if isfinite(tot) && tot > 0.0
                        w .*= budget / tot
                    end
                end
            end
            if !(all(isfinite, w) && sum(w) > 0.0)
                push!(iter_ms, (time_ns() - t0) / 1e6)
                return (; w=w, converged=false, reason=:nonfinite_state,
                        iters=iters, cert=cert_now, iter_ms=iter_ms,
                        mu_final=mu, trajectory=traj)
            end

            push!(iter_ms, (time_ns() - t0) / 1e6)
        end
        mu = max(mu * sigma, mu_floor)
    end

    (; w=w, converged=false, reason=:budget_exhausted, iters=iters,
       cert=probe_certificate(A, bb, w, budget), iter_ms=iter_ms,
       mu_final=mu, trajectory=traj)
end

# -----------------------------------------------------------------------------
# 5. tie-break Stage 2/3 的 m=2 专用化原型
#    W_ε = {u ∈ [0, budget] : h(u) ≥ U* − tie_eps}（u 为 risky 分量）：
#    Stage 2 max cash ⟺ min u = u_lo；Stage 3 min risky L2 同点。
#    候选点仍过原空间证书（与现状采纳语义一致）。
# -----------------------------------------------------------------------------
function tiebreak_m2(A::Matrix{Float64}, b::Vector{Float64}, budget::Float64,
                      Ustar::Float64, tie_eps::Float64,
                      w_primary::Vector{Float64};
                      bisect_iters::Int=80, tol::Float64=1e-8)
    S, m = size(A)
    m == 2 || throw(ArgumentError("tiebreak_m2 requires m == 2"))
    threshold = Ustar - tie_eps

    u_of = function (u::Float64)
        logsum = 0.0
        for s in 1:S
            zs = b[s] + A[s, 1] * u + A[s, 2] * (budget - u)
            (isfinite(zs) && zs > 0.0) || return NaN
            logsum += log(zs)
        end
        logsum / S
    end

    ulo = 0.0
    U0 = u_of(0.0)
    if !(isfinite(U0) && U0 >= threshold)
        ustar = w_primary[1]
        if isfinite(ustar) && ustar > 0.0
            lo, hi = 0.0, ustar
            for _ in 1:bisect_iters
                mid = 0.5 * (lo + hi)
                if u_of(mid) >= threshold
                    hi = mid
                else
                    lo = mid
                end
            end
            ulo = hi
        end
    end

    w_tb = [ulo, budget - ulo]
    cert = probe_certificate(A, b, w_tb, budget)
    (; u_lo=ulo, w=w_tb, cert=cert, certified=probe_certified(cert; tol=tol),
       threshold=threshold)
end

# -----------------------------------------------------------------------------
# 6. 合成数据（确定性；无 RNG）
# -----------------------------------------------------------------------------
function synth_large(S::Int, m_risky::Int; spread::Float64=0.04,
                     seed::UInt64=UInt64(0xC0FFEE))
    A = Matrix{Float64}(undef, S, m_risky + 1)
    st = seed
    for s in 1:S
        for j in 1:m_risky
            st = splitmix64(st)
            u = uniform01(st)
            A[s, j] = exp(spread * (2.0 * u - 1.0))
        end
        A[s, m_risky + 1] = 1.0
    end
    (A, zeros(S))
end

# -----------------------------------------------------------------------------
# 7. 运行与打印
# -----------------------------------------------------------------------------
function run_case(name::String, A::Matrix{Float64}, b::Vector{Float64},
                  budget::Float64; tol::Float64=1e-8,
                  analytic::Union{Nothing,AbstractVector{Float64}}=nothing)
    println()
    println("=== ", name, " ===")
    @printf("  input : S=%d m=%d budget=%.6g tol=%.1e\n",
            size(A, 1), size(A, 2), budget, tol)
    res = ipm_logsum_simplex(A, budget; b=b, tol=tol)
    n_it = length(res.iter_ms)
    total_ms = isempty(res.iter_ms) ? 0.0 : sum(res.iter_ms)
    mean_ms = n_it == 0 ? 0.0 : total_ms / n_it
    max_ms = isempty(res.iter_ms) ? 0.0 : maximum(res.iter_ms)
    @printf("  ipm   : iters=%d converged=%s reason=%s mu_final=%.3e\n",
            res.iters, res.converged, res.reason, res.mu_final)
    @printf("  cert  : feasibility=%.3e kkt=%.3e gap=%.3e objective=%.10e\n",
            res.cert.feasibility, res.cert.kkt_residual,
            res.cert.objective_gap, res.cert.objective)
    @printf("  time  : total_ms=%.3f mean_iter_ms=%.3f max_iter_ms=%.3f\n",
            total_ms, mean_ms, max_ms)
    @printf("  w     : %s\n", string(round.(res.w, digits=10)))
    if analytic !== nothing
        d = maximum(abs.(res.w .- analytic))
        @printf("  anly  : max|w - w_analytic| = %.3e\n", d)
    end
    res
end

function run_wide_with_scaling(S::Int; spread::Float64=40.0,
                               seed::UInt64=UInt64(0xBEEF), tol::Float64=1e-8)
    println()
    println("=== large S=$S wide-tail (spread=$spread): unscaled vs row-scaled ===")
    A, b = synth_large(S, 1; spread=spread, seed=seed)
    run_case("large S=$S wide-tail UNSCALED", A, b, 1.0; tol=tol)

    # 生产 cs 定义：c_s = max(|b_s|, max_j |X_sj|)（free risky 列；cash 列不参与），
    # cash 列同缩为 1/c_s（KELLY_NUMERICAL §2.1：不得保留 ones）。
    cs = Vector{Float64}(undef, S)
    for s in 1:S
        v = abs(b[s])
        a = abs(A[s, 1])
        a > v && (v = a)
        cs[s] = v
    end
    span = maximum(cs) / minimum(cs)
    A_s = hcat(A[:, 1] ./ cs, 1.0 ./ cs)
    b_s = b ./ cs
    res_s = run_case("large S=$S wide-tail ROW-SCALED", A_s, b_s, 1.0; tol=tol)
    @printf("  span  : max(cs)/min(cs) = %.3e\n", span)
    cert_orig = probe_certificate(A, b, res_s.w, 1.0)
    @printf("  xchk  : d(feasibility)=%.2e d(kkt)=%.2e d(gap)=%.2e  (orig vs scaled cert)\n",
            abs(cert_orig.feasibility - res_s.cert.feasibility),
            abs(cert_orig.kkt_residual - res_s.cert.kkt_residual),
            abs(cert_orig.objective_gap - res_s.cert.objective_gap))
end

function run_tiebreak_demo()
    println()
    println("=== tie-break m=2 demo (Stage 2/3 specialization) ===")

    # (a) 严格凹 Case C
    A = [1.2 1.0; 0.85 1.0]
    b = zeros(2)
    res = ipm_logsum_simplex(A, 1.0; b=b, tol=1e-8)
    tb = tiebreak_m2(A, b, 1.0, res.cert.objective, 1e-8, res.w; tol=1e-8)
    @printf("  [strict] U*=%.10e threshold=%.10e u_lo=%.10e cash_tb=%.10e\n",
            res.cert.objective, tb.threshold, tb.u_lo, 1.0 - tb.u_lo)
    @printf("           tb cert: feasibility=%.3e kkt=%.3e gap=%.3e certified=%s\n",
            tb.cert.feasibility, tb.cert.kkt_residual, tb.cert.objective_gap,
            tb.certified)
    @printf("           primary cash=%.10e delta_cash=%.3e\n",
            res.w[2], (1.0 - tb.u_lo) - res.w[2])

    # (b) 退化平局：risky 列恒 1 ⟹ wealth ≡ 1，整个单纯形平局（与 D-091 同构）
    A2 = [1.0 1.0; 1.0 1.0; 1.0 1.0]
    b2 = zeros(3)
    res2 = ipm_logsum_simplex(A2, 1.0; b=b2, tol=1e-8)
    tb2 = tiebreak_m2(A2, b2, 1.0, res2.cert.objective, 1e-8, res2.w; tol=1e-8)
    @printf("  [tie]    U*=%.3e threshold=%.3e u_lo=%.3e cash_tb=%.3e certified=%s\n",
            res2.cert.objective, tb2.threshold, tb2.u_lo, 1.0 - tb2.u_lo,
            tb2.certified)
end

# -----------------------------------------------------------------------------
# 8. 可选输入文件加载
# -----------------------------------------------------------------------------
function load_problem(path::String)
    raw = readlines(path)
    lines = String[]
    for ln in raw
        st = strip(ln)
        (isempty(st) || startswith(st, "#")) && continue
        push!(lines, st)
    end
    isempty(lines) && throw(ArgumentError("empty input file: $path"))
    header = split(lines[1])
    S = parse(Int, header[1])
    m = parse(Int, header[2])
    budget = parse(Float64, header[3])
    tol = length(header) >= 4 ? parse(Float64, header[4]) : 1e-8
    length(lines) >= 1 + S ||
        throw(ArgumentError("input file has fewer rows than S=$S"))
    A = Matrix{Float64}(undef, S, m)
    for s in 1:S
        vals = split(lines[1 + s])
        length(vals) == m ||
            throw(ArgumentError("row $s has $(length(vals)) values, expected $m"))
        for j in 1:m
            A[s, j] = parse(Float64, vals[j])
        end
    end
    b = zeros(S)
    if length(lines) >= 1 + 2 * S
        for s in 1:S
            b[s] = parse(Float64, split(lines[1 + S + s])[1])
        end
    end
    (; A=A, b=b, budget=budget, tol=tol)
end

# -----------------------------------------------------------------------------
# 9. main
# -----------------------------------------------------------------------------
function main()
    if !isempty(ARGS)
        prob = load_problem(ARGS[1])
        println("loaded problem from ", ARGS[1])
        run_case("file:" * ARGS[1], prob.A, prob.b, prob.budget; tol=prob.tol)
        return
    end

    println("kelly_ipm_probe: LOGSUM-SIMPLEX-IPM-1 prototype")
    println("(deterministic; no RNG; cert == cash_kelly_certificate formula)")

    println()
    println("--- small analytic fixtures ---")
    run_case("Case C (S=2, m=2, interior; analytic 5/6, 1/6)",
             [1.2 1.0; 0.85 1.0], zeros(2), 1.0;
             analytic=[5.0 / 6.0, 1.0 / 6.0])
    run_case("Case D (S=2, m=2, locked base [0.3,0.3], budget=0.7; analytic 5/12, 17/60)",
             [1.4 1.0; 0.7 1.0], [0.3, 0.3], 0.7;
             analytic=[5.0 / 12.0, 17.0 / 60.0])
    run_case("Case A (S=4, m=2, deterministic gross<1; analytic 0, 1)",
             [0.99 1.0; 0.99 1.0; 0.99 1.0; 0.99 1.0], zeros(4), 1.0;
             analytic=[0.0, 1.0])
    run_case("D-090(a) (S=3, m=4, symmetric triple; analytic 1/3,1/3,1/3,0)",
             [1.25 0.9 0.9 1.0; 0.9 1.25 0.9 1.0; 0.9 0.9 1.25 1.0],
             zeros(3), 1.0; analytic=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0])
    run_case("m=1 (all cash, budget=0.7; analytic 0.7)",
             fill(1.0, 4, 1), zeros(4), 0.7; analytic=[0.7])

    run_tiebreak_demo()

    println()
    println("--- large synthetic (deterministic, no RNG) ---")
    for S in (65536, 131072)
        A, b = synth_large(S, 1; spread=0.04, seed=UInt64(0xC0FFEE))
        run_case("large S=$S mild spread=0.04", A, b, 1.0; tol=1e-8)
    end
    run_wide_with_scaling(131072; spread=40.0, seed=UInt64(0xBEEF), tol=1e-8)

    println()
    println("done.")
end

main()
