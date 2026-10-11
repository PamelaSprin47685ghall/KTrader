# =============================================================================
# KTraderGate0 strict prequential residual history — P0-1 tiny 级测试
# =============================================================================
#
# 测试对象：src/gate0/oof.jl 的 `prequential_residual_rows`（production 残差
# 源，P0-1 裁决——不 export，同 module include 链可见）。
#
# 覆盖（P0-1 裁决）：
# - strictly causal：行 s 的预测只用行 1..s−1 拟合的 posterior——修改行 s
#   的 target 不影响行 s 的残差；修改行 s+1.. 不影响行 s 的残差（构造性
#   隔离的可执行断言）；
# - 早期行（train 不足 A2 propriety）输出 NaN；后续行 finite；
# - 输出形态与 legacy OOF 一致（n × N asset 空间；含 b₀ 扣除——DC 列在
#   B̂·x 内自动参与；D-041 完整后验均值）；
# - fail loudly：全部行 train 都退化时抛错（文本含 "prequential residual"）；
# - 与 legacy OOF 的对照（同一输入下 prequential 与 OOF 是不同对象——
#   OOF 行 s 可用 s+1..t，prequential 不可用；断言二者**不**逐位相同，
#   证明语义差异真实存在，而非实现巧合）；
# - 维度校验 fail loudly。
#
# 纪律：固定 seed；不运行测试（运行归 DevOps）；断言与回测收益无关
# （P0-1：prequential 是历史当时真正发生的 surprise，不是回测选择器）。

using Test, Random, LinearAlgebra

module Gate0PrequentialEnv
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "geometry.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "modes.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "response.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "posterior.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "oof.jl"))
end
using .Gate0PrequentialEnv
# prequential_residual_rows 按设计不 export（oof.jl 注释：同 module include
# 链可见）；测试经显式限定名访问。
import .Gate0PrequentialEnv: prequential_residual_rows

_throws_msg(f, needle) = try
    f()
    false
catch e
    e isa ErrorException && occursin(needle, sprint(showerror, e))
end

@testset "gate0 P0-1: strict prequential residual history" begin
    SEED = 20261010
    rng = MersenneTwister(SEED)
    N, P, n = 2, 3, 30
    X = randn(rng, n, P)
    Y = X * (0.5 .* randn(rng, N, P))' .+ 0.3 .* randn(rng, n, N)
    E = domain_mode_basis(N)

    # -------------------------------------------------------------------
    # 1. 输出形态（n × N asset 空间）+ 早期行 NaN + 后续行 finite
    # -------------------------------------------------------------------
    @testset "output shape & early-row NaN" begin
        eps_p = prequential_residual_rows(X, Y, E; tol = 1e-4)
        @test size(eps_p) == (n, N)
        # 早期行（train 不足 A2：n_tr = s−1 < N=2 ⇒ s ≤ 2 必 NaN）
        @test all(isnan, eps_p[1:2, :])
        # 后续行 finite：从首个满足 A2（n_tr ≥ N 且 rank(Syy) = N）的行起
        # 全部 finite——fixture 连续随机数据下 s ≥ N+1 即满足，但断言
        # 不假设具体 s0，只断言「存在 finite 行且一旦 finite 之后不再
        # 回到 NaN」（train 单调增长 ⇒ propriety 单调保持）。
        first_finite = findfirst(s -> all(isfinite, eps_p[s, :]), 1:n)
        @test first_finite !== nothing
        @test first_finite >= 3
        @test all(s -> all(isfinite, eps_p[s, :]), first_finite:n)
    end

    # -------------------------------------------------------------------
    # 2. strictly causal（构造性隔离的可执行断言）
    # -------------------------------------------------------------------
    @testset "strictly causal" begin
        eps_p = prequential_residual_rows(X, Y, E; tol = 1e-4)
        # (a) 修改行 s 的 target 不影响行 s 的预测（行 s 自身是目标，
        #     绝不进入自己的训练集 ⇒ B̂[1..s−1] 不变 ⇒ 预测不变）。
        # 残差 = E·(实际 − 预测)：实际 Y[s] 平移 +5、预测不变 ⇒
        # 残差平移 E·(5·1_N)（E 为 mode→asset 变换，非单位阵）。
        s = 10
        Y_mod = copy(Y); Y_mod[s, :] .+= 5.0
        eps_mod = prequential_residual_rows(X, Y_mod, E; tol = 1e-4)
        @test isapprox(eps_p[s, :] .+ E * fill(5.0, N), eps_mod[s, :], atol = 1e-8)
        # (b) 修改行 s+1..n 不影响行 s 的残差（行 s 的 train 只用 1..s−1）
        Y_tail = copy(Y); Y_tail[(s+1):end, :] .+= 5.0
        eps_tail = prequential_residual_rows(X, Y_tail, E; tol = 1e-4)
        @test isapprox(eps_p[s, :], eps_tail[s, :], atol = 1e-10)
        # (c) 但行 s+1 的残差会变（s+1 的 train 含行 s——正向控制）
        @test !isapprox(eps_p[s+1, :], eps_mod[s+1, :], atol = 1e-8)
    end

    # -------------------------------------------------------------------
    # 3. 与 legacy OOF 的语义对照（不同对象，非实现巧合）
    # -------------------------------------------------------------------
    @testset "vs legacy OOF (semantic distinction)" begin
        folds = fold_grid(n; F_folds = 3)
        fp = fit_fold_posteriors(X, Y, folds; tol = 1e-4)
        eps_oof = oof_residual_rows_full(fp, X, Y, E)
        eps_p = prequential_residual_rows(X, Y, E; tol = 1e-4)
        # OOF 行 s 可用 s+1..t（fold train = full − held-out fold），
        # prequential 不可用 ⇒ 二者不应逐位相同（语义差异真实存在）。
        # 取中段行（s ≥ 3 且非 fold 边界附近）比较。
        @test !isapprox(eps_p[10, :], eps_oof[10, :], atol = 1e-8)
    end

    # -------------------------------------------------------------------
    # 4. fail loudly（维度校验 + 全退化）
    # -------------------------------------------------------------------
    @testset "fail loudly" begin
        @test_throws DimensionMismatch prequential_residual_rows(X[1:5, :], Y, E)
        # E 尺寸不匹配（N=2 时 3×3 ≠ N×N）→ DimensionMismatch
        @test_throws DimensionMismatch prequential_residual_rows(X, Y, randn(3, 3))
        # 全退化：Y 全零 ⇒ rank(Syy) = 0 < N 恒不满足 ⇒ 无行有 proper
        # train posterior ⇒ 抛错（文本含 "prequential residual"）
        Y0 = zeros(n, N)
        @test _throws_msg(() -> prequential_residual_rows(X, Y0, E; tol = 1e-4),
                          "prequential residual")
    end
end


# =============================================================================
# 成本加速回归（2026-10-10 新增；Manager 裁决：(i) 求积循环内重复计算消除 +
# (ii) prequential 行级并行；见 docs/PREQUENTIAL_COST_ACCELERATION.md §8）。
# 语义：验证「增量维护 vs 全量重算」roundoff 级一致、「行级并行 vs 手动
# 逐行串行」逐位一致、以及同输入同结果的确定性；不改既有断言语义。
# =============================================================================
import .Gate0PrequentialEnv: fit_full_posterior, _row_undefined_error

# 全量重算参照（复刻修订前的循环路径；仅供对照，不进入生产）。
function _ref_quad_2d_full(logf, u0lo, u0hi, uplo, uphi; tol, init_grid = 4)
    vc = Dict{Tuple{Float64,Float64},Float64}()
    fv(u0, up) = get!(() -> logf(u0, up), vc, (u0, up))
    cells = Vector{Any}()
    du0 = (u0hi - u0lo) / init_grid; dup = (uphi - uplo) / init_grid
    for i in 1:init_grid, j in 1:init_grid
        push!(cells, (u0lo + (i - 1) * du0, u0lo + i * du0,
                      uplo + (j - 1) * dup, uplo + j * dup))
    end
    m = maximum(fv((a + b) / 2, (c + d) / 2) for (a, b, c, d) in cells)
    function cv(cell)
        a, b, c, d = cell
        mx = (a + b) / 2; hx = (b - a) / 2; my = (c + d) / 2; hy = (d - c) / 2
        g = 1.0 / sqrt(3.0)
        (exp(fv(mx - hx * g, my - hy * g) - m) + exp(fv(mx - hx * g, my + hy * g) - m) +
         exp(fv(mx + hx * g, my - hy * g) - m) + exp(fv(mx + hx * g, my + hy * g) - m)) *
         (b - a) * (d - c) / 4
    end
    function ce(cell)
        a, b, c, d = cell
        mx = (a + b) / 2; hx = (b - a) / 2; my = (c + d) / 2; hy = (d - c) / 2
        g2 = 1.0 / sqrt(3.0)
        i2 = (exp(fv(mx - hx * g2, my - hy * g2) - m) + exp(fv(mx - hx * g2, my + hy * g2) - m) +
              exp(fv(mx + hx * g2, my - hy * g2) - m) + exp(fv(mx + hx * g2, my + hy * g2) - m)) / 4
        g3 = sqrt(3.0 / 5.0); wl = 5.0 / 9.0; wh = 8.0 / 9.0
        xs = (mx - hx * g3, mx, mx + hx * g3); ys = (my - hy * g3, my, my + hy * g3)
        acc = 0.0
        for jj in 1:3, ii in 1:3
            acc += (ii == 2 ? wh : wl) * (jj == 2 ? wh : wl) * exp(fv(xs[ii], ys[jj]) - m)
        end
        abs(acc / 4 - i2) * ((b - a) * (d - c))
    end
    Z = sum(cv, cells); et = sum(ce, cells); rel = Z > 0 ? et / Z : Inf
    while rel > tol && length(cells) < 8192
        idx = 1; emax = ce(cells[1])
        for k in 2:length(cells)
            e = ce(cells[k]); if e > emax; emax = e; idx = k; end
        end
        a, b, c, d = cells[idx]
        if (b - a) >= (d - c)
            mid = (a + b) / 2; cells[idx] = (a, mid, c, d); push!(cells, (mid, b, c, d))
        else
            mid = (c + d) / 2; cells[idx] = (a, b, c, mid); push!(cells, (a, b, mid, d))
        end
        Z = sum(cv, cells); et = sum(ce, cells); rel = Z > 0 ? et / Z : Inf
    end
    weights = [cv(c) for c in cells]
    total = sum(weights)
    (; weights = weights ./ total, logZ = log(total) + m, rel_err = rel, n_cells = length(cells))
end

@testset "cost acceleration invariants (incremental & parallel)" begin
    # ---- A. 增量 vs 全量（求积对照；不逐位：roundoff 可差 1–2 cells）----
    g2d(u0, up) = -0.5 * (u0^2 + up^2)
    # fixture 域修正（首跑机械修正，2026-10-10；断言与 tail_rel 均不变）：
    # 原 ±6 域在 (0,±6) 边中点 logf=-18 > 阈值 -18.056（= m_ref 0 +
    # ln(1e-10) + ln(144)），tail 证书数学上不成立（1e-10 的 tail 需域 ≳6.5σ）；
    # 扩为 ±7：边中点 -24.5 < 阈值 -17.75，证书真实成立。
    pr = Gate0PrequentialEnv.adaptive_quadrature_2d(g2d, (-7.0, 7.0), (-7.0, 7.0);
                                                    tol = 1e-6, tail_rel = 1e-10, m_ref = 0.0)
    rf = _ref_quad_2d_full(g2d, -7.0, 7.0, -7.0, 7.0; tol = 1e-6)
    @test pr.rel_err <= 1e-6
    @test rf.rel_err <= 1e-6
    @test abs(pr.logZ - rf.logZ) <= 1e-8
    @test abs(sum(pr.weights) - 1.0) <= 1e-12
    @test abs(sum(rf.weights) - 1.0) <= 1e-12
    @test abs(pr.n_cells - rf.n_cells) <= 2
    @test abs(pr.logZ - log(2π)) <= 1e-4        # 解析锚（域截断 ~2e-9）

    # ---- B. 并行 vs 手动逐行串行（逐位；NaN 安全）----
    rng2 = MersenneTwister(20261010)
    N2, P2, n2 = 2, 3, 8
    X2 = randn(rng2, n2, P2)
    Y2 = X2 * (0.5 .* randn(rng2, N2, P2))' .+ 0.3 .* randn(rng2, n2, N2)
    E2 = domain_mode_basis(N2)
    eps_a = prequential_residual_rows(X2, Y2, E2; tol = 1e-4)
    eps_manual = Matrix{Float64}(undef, n2, N2)
    for s in 1:n2
        if s - 1 < 1
            eps_manual[s, :] .= NaN
            continue
        end
        try
            post = fit_full_posterior(X2[1:(s-1), :], Y2[1:(s-1), :]; tol = 1e-4)
            Bmix = sum(c.weight .* c.B_hat for c in post.conditional)
            eps_manual[s, :] .= E2 * (Y2[s, :] .- Bmix * view(X2, s, :))
        catch e
            if _row_undefined_error(e)
                eps_manual[s, :] .= NaN
            else
                rethrow()
            end
        end
    end
    @test all(isequal.(eps_a, eps_manual))     # 行级独立 ⇒ 与串行逐位一致
    eps_b = prequential_residual_rows(X2, Y2, E2; tol = 1e-4)
    @test all(isequal.(eps_a, eps_b))          # 确定性：同输入同结果
    @test size(eps_a) == (n2, N2)
end

# =============================================================================
# 阶段 1 秩一预分解对照（FIRST-DAY-FIT-1；见
# docs/FIRST_DAY_FIT_COST_ACCELERATION.md）：SM 路径 vs 逐点 cholesky 路径
# 的字段级对照 + fit_full_posterior rank1=true/false 端到端对照。
# 合成数据、固定 seed；不改既有断言语义；宿主复用本文件（加载链含 posterior.jl）。
# =============================================================================
import .Gate0PrequentialEnv: log_evidence, _log_evidence_rank1, _rank1_spec,
    sufficient_stats, lambda_diag, fit_full_posterior

@testset "stage-1 rank-one prespec (SM vs cholesky)" begin
    rng3 = MersenneTwister(20261010)
    N3 = 2; P3 = 1 + 14 * N3            # DC 形态（P=29）
    n3 = 40
    X3 = randn(rng3, n3, P3)
    Y3 = X3 * (0.3 .* randn(rng3, N3, P3))' .+ 0.5 .* randn(rng3, n3, N3)
    st3 = sufficient_stats(X3, Y3)
    spec3 = _rank1_spec(st3)
    @test spec3.W == spec3.Q' * st3.Sxy   # 载体自洽
    @test spec3.v == spec3.Q[1, :]        # Qᵀe1 = Q 第一行
    for (u0v, upv) in ((0.0, 0.0), (1.5, -0.5), (-2.0, 2.0), (3.0, 1.0))
        lam3 = lambda_diag(P3, u0v, upv, true)
        le_ref = log_evidence(st3, lam3)
        le_sm = _log_evidence_rank1(st3, lam3, spec3)
        # 误差界：双路径同为恒等式，差异仅浮点舍入（谱+SM vs cholesky）；
        # P=29 下 κ(Sxx)·ε·O(P) ≲ 1e-10；1e-8 留 100× 余量且远紧于任何
        # 真实公式错误（观测级 O(0.1)）。
        @test abs(le_sm - le_ref) <= 1e-8 * max(1.0, abs(le_ref))
    end
    # 端到端：rank1=true 与 false 的 posterior 摘要对照
    p_ref = fit_full_posterior(X3, Y3; tol = 1e-3)
    p_sm = fit_full_posterior(X3, Y3; tol = 1e-3, rank1 = true)
    # 断言强度（验证批次 5 复核）：自适应细分轨迹对 logf 的浮点微扰敏感，
    # 「节点数相同 / weights 逐元素 atol=1e-8」偏强——改为积分量对照：
    # evidence 绝对容差（两路径各自 tol=1e-3 收敛到同一积分；正常差 ≲1e-4，
    # 公式级 bug 会造 O(0.1+) 差异）；三个测试函数的后验加权均值（相对 5%）。
    @test abs(p_sm.evidence - p_ref.evidence) <= 0.05
    wq(p, f) = sum(p.alpha_weights[k] * f(p.alpha_nodes[k][1], p.alpha_nodes[k][2])
                   for k in eachindex(p.alpha_nodes))
    for ff in ((a, b) -> a, (a, b) -> b, (a, b) -> a^2 + b^2)
        wr = wq(p_ref, ff); ws = wq(p_sm, ff)
        @test abs(ws - wr) <= 5e-2 * max(1.0, abs(wr))
    end
    @test all(isfinite, p_sm.node_log_evidence)
end

# =============================================================================
# 阶段 1 回退路径定向测试（fail-soft）：SM 防护断言触发时，入口
# （_evidence_eval）捕获并逐点回退原路径——不抛异常外泄、结果与逐点
# 原路径同输入逐位一致。伪造 spec 仅作确定性触发手段（生产触发源为真实
# 数值病态；回退逻辑验证与触发源无关）。确定性、无随机。
# =============================================================================
import .Gate0PrequentialEnv: _evidence_eval, SxxRankOneSpec

@testset "stage-1 fallback path (fail-soft)" begin
    rng4 = MersenneTwister(20261010)
    Xs = randn(rng4, 8, 3)
    Ys = randn(rng4, 8, 2)
    st_s = sufficient_stats(Xs, Ys)
    spec_s = _rank1_spec(st_s)
    Qid = Matrix{Float64}(I, 3, 3)
    spec_d = SxxRankOneSpec(Qid, [-5.0, 1.0, 1.0], [1.0, 0.0, 0.0], zeros(3, 2))
    spec_n = SxxRankOneSpec(Qid, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], zeros(3, 2))
    # (1) 防护分支确定性触发（单元层）
    @test _throws_msg(() -> _log_evidence_rank1(st_s, [1.0, 2.0, 3.0], spec_s),
                     "_log_evidence_rank1")               # 形态分支
    @test _throws_msg(() -> _log_evidence_rank1(st_s, [1.0, 1e-3, 1e-3], spec_d),
                     "_log_evidence_rank1")               # 位移特征值 ≤ 0
    @test _throws_msg(() -> _log_evidence_rank1(st_s, [1e-3, 1.0, 1.0], spec_n),
                     "_log_evidence_rank1")               # SM 分母 ≤ 0
    # (2) 回退：无异常外泄 + 与逐点原路径同输入逐位一致
    @test _evidence_eval(st_s, [1.0, 2.0, 3.0], spec_s) == log_evidence(st_s, [1.0, 2.0, 3.0])
    @test _evidence_eval(st_s, [1.0, 1e-3, 1e-3], spec_d) == log_evidence(st_s, [1.0, 1e-3, 1e-3])
    @test _evidence_eval(st_s, [1e-3, 1.0, 1.0], spec_n) == log_evidence(st_s, [1e-3, 1.0, 1.0])
    # (3) 正常路径直通（SM 不被回退）+ nothing 直通
    lam_ok = [1.0, 1.0, 1.0]
    @test _evidence_eval(st_s, lam_ok, spec_s) == _log_evidence_rank1(st_s, lam_ok, spec_s)
    @test _evidence_eval(st_s, lam_ok, nothing) == log_evidence(st_s, lam_ok)
end

# =============================================================================
# 阶段 2 lazy fit_node reference equality（FIRST-DAY-FIT-1）：rank1=true 时
# _fit_node 全程经 spec（无 cholesky）——对同 (st, lam) 与默认物化路径做
# 字段级对照（B_hat / S_alpha / V_solve 二次型）；端到端用积分量对照
# （求积轨迹可因浮点微扰分叉，不做节点集/逐元素断言）。确定性、无随机。
# =============================================================================
import .Gate0PrequentialEnv: _fit_node, V_solve, predict_mu

@testset "stage-2 lazy fit_node (reference equality)" begin
    rng5 = MersenneTwister(20261010)
    N5 = 2; P5 = 1 + 14 * N5; n5 = 40
    X5 = randn(rng5, n5, P5)
    Y5 = X5 * (0.3 .* randn(rng5, N5, P5))' .+ 0.5 .* randn(rng5, n5, N5)
    st5 = sufficient_stats(X5, Y5)
    spec5 = _rank1_spec(st5)
    lam5 = lambda_diag(P5, 0.5, -0.5, true)
    c_ref = _fit_node(st5, lam5, 1.0)
    c_lz = _fit_node(st5, lam5, 1.0; rank1 = true, spec = spec5)
    @test c_lz.V === nothing
    @test c_ref.V isa Matrix{Float64}
    @test isapprox(c_lz.B_hat, c_ref.B_hat, atol = 1e-8)
    @test isapprox(c_lz.S_alpha, c_ref.S_alpha, atol = 1e-8)
    xt5 = collect(randn(rng5, P5))
    @test isapprox(dot(xt5, V_solve(c_lz, xt5)), dot(xt5, V_solve(c_ref, xt5)),
                   rtol = 1e-8)
    # 端到端：两路径各自 tol=1e-3 收敛；轨迹可因微扰分叉，故对照积分量
    # （mu/within）；atol 取收敛级余量（正常差 ≪0.05，公式级错误会远超）。
    p_ref5 = fit_full_posterior(X5, Y5; tol = 1e-3)
    p_lz5 = fit_full_posterior(X5, Y5; tol = 1e-3, rank1 = true)
    m_ref = predict_mu(p_ref5, xt5); m_lz = predict_mu(p_lz5, xt5)
    @test isapprox(m_lz.mu, m_ref.mu, atol = 0.05)
    @test isapprox(m_lz.within, m_ref.within, atol = 0.01)
    @test all(isfinite, p_lz5.node_log_evidence)
end