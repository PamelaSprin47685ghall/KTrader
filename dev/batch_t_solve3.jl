#!/usr/bin/env julia
# =============================================================================
# dev/batch_t_solve3.jl —— 批量两段式调度 · 段 B 的三段式拆分（B1'/B2'/B3'）
# =============================================================================
# 背景：段 B（min=65536→131072 一层对）稳定 ~50-58s（超 ~48s 窗口）；其组成可
# 分：w_65536 与 w_131072 独立计算（gen+solve 各自），A/B 只是后验统计量
# （用 131072 或 audit 场景矩阵评估）。
#
# 逐点等价性论证（与 adaptive_scenario_kelly 的单层对完全一致）：
# 1) rule 派生：opt_rule=SobolOwenRule(2, rule_seed)；a_seed=audit_seed(rule_seed)；
#    m_seed=_default_mu_seed(rule_seed)；am_seed=m_seed ⊻ 0x9E3779B97F4A7C15。
# 2) solve_layer(M)：X=src.gen(M, opt_rule, MersenneTwister(m_seed))；
#    base=locked_wealth_gate0(X, locked)；free_pos=[k: locked[k]==0]；
#    cash_kelly(X[:,free_pos]; base, budget, tol=kelly_tol)；w_full 填回 locked。
#    —— 与 quadrature.jl 的 solve_layer 同式（复刻；adaptive 变更须同步本脚本）。
# 3) A=‖vcat(w_2M,wc_2M)−vcat(w_M,wc_M)‖₁；B_opt 在 X_2M 上求值两权重目标差；
#    B_audit 在 X_a=gen(M2,audit_rule,MT(am_seed)) 上求绝对差——公式逐点复刻。
# 4) 默认容差与段 B（经便捷入口、未传参）一致：weight_tol=1e-3、utility_tol=1e-5、
#    utility_tol_audit=1e-5、kelly_tol=1e-8。
# 5) 确定性：gen/cash_kelly 均无新 RNG；三段各自原子落盘。
#
# 用法：
#   b1: julia ... dev/batch_t_solve3.jl b1 <prep.bin> [out_dir]
#   b2: julia ... dev/batch_t_solve3.jl b2 <prep.bin> [out_dir]
#   b3: julia ... dev/batch_t_solve3.jl b3 <prep.bin> [out_dir]   # 读同目录 seg1/seg2
#   细粒度（当三段式某段超窗时）：b1a gen(65536)->X1；b1b X1->seg1；b2a gen(131072)->X2；
#   b2b X2->seg2；随后 b3 拼装（seg 结构与三段式一致；等价性：b1a+b1b ≡ b1、b2a+b2b ≡ b2）。
#   b2b 出口链拆分（SPLIT-1/2/3）：b2b1 prelude->pre2；b2b2 finish_main->mid2；b2b3
#   tiebreak（读 mid2）->seg2；SPLIT-3：b2b1a prep->prep1；b2b1b solve（读 prep1）->pre2。
#   自检：julia ... dev/batch_t_solve3.jl --selftest
# 环境：GATE0_BATCH_MIN（默认 65536）、GATE0_BATCH_MAX（默认 131072）、
#       GATE0_BATCH_MU_QMC（默认 true）、GATE0_BATCH_CHISQ_QMC（默认 true）。
#
# 退出码：0 成功；2 参数；3 自检失败；20 prep/seg/X 读取或契约失败；31 B1'；32 B2'；
# 34 b1a gen；35 b1b kelly；36 b2a gen；37 b2b kelly；33 B3' 统计/归一；41/42 seg 落盘；
# 43 solve3 落盘；44/45 X 产物落盘。SPLIT-2/3（b2b 拆分链）：38/46 b2b1（prelude）、39/48
# b2b2（finish_main）、40/47 b2b3（finish_tiebreak）、50/52 b2b1a（prelude_prep）、51/53
# b2b1b（prelude_solve）——前者段失败、后者落盘失败。（旧行保留于下方注释时间线）
# 原表：0 成功；2 参数；3 自检失败；20 prep/seg 读取或契约失败；31 B1' 求解失败；
# 32 B2' 求解失败；33 B3' 统计/归一失败；41/42/43 各段落盘失败。
#
# A 对拍判据：三段拼的 A 应逐位复现 CQ 探针的 6.2585e-5；原 batch_t_solve.jl
# （整段 B）保留为对拍基准。
# =============================================================================

using Serialization, Random, LinearAlgebra

# 模块名必须与段 A（batch_t_capture.jl）完全一致：serialize 记录类型的模块路径
# （Main.Gate0BatchCaptureA.*），反序列化按名查找——不同名则 UndefVarError。
module Gate0BatchCaptureA
    include(joinpath(@__DIR__, "..", "src", "gate0", "geometry.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "modes.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "response.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "posterior.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "oof.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "innovation.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "predictive.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "kelly.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "quadrature.jl"))
end
using .Gate0BatchCaptureA
# 模块内函数未导出——显式导入本脚本用到的名字（反序列化类型仍按模块路径匹配）
using .Gate0BatchCaptureA: predictive_rqmc_source, SobolOwenRule,
    locked_wealth_gate0, cash_kelly, audit_seed,
    _cash_kelly_prelude, _cash_kelly_finish,
    _cash_kelly_finish_main, _cash_kelly_finish_tiebreak,
    _cash_kelly_prelude_prep, _cash_kelly_prelude_solve

const _PREP_VERSION = "batch-prep-v1"
const _SEG_VERSION = "batch-solve3-v1"
const _W_TOL = 1e-3
const _U_TOL = 1e-5
const _U_TOL_AUDIT = 1e-5
const _KELLY_TOL = 1e-8

_fail(code::Int, msg::String) = (println(stderr, "[batch-solve3] " * msg); exit(code))

function _atomic_serialize(path::String, payload)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        serialize(io, payload)
    end
    mv(tmp, path; force = true)
end

function _atomic_write_text(path::String, text::String)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        print(io, text)
    end
    mv(tmp, path; force = true)
end

const _PREP_FIELDS = (:version, :t, :seed, :held_mode, :data_note,
                      :rule_kwargs, :post, :st, :xt, :s1, :E_active,
                      :locked, :budget)

function _check_prep(d)::Union{Nothing,String}
    (d isa NamedTuple) || return "prep is not a NamedTuple"
    for f in _PREP_FIELDS
        hasproperty(d, f) || return "prep missing field: " * string(f)
    end
    d.version == _PREP_VERSION || return "prep version mismatch"
    nothing
end

const _SEG_FIELDS = (:version, :seg, :t, :seed, :M, :mu_qmc, :mu_chisq_qmc,
                     :w_full, :w_cash, :cert)

function _check_seg(s, want_seg::Int)::Union{Nothing,String}
    (s isa NamedTuple) || return "seg is not a NamedTuple"
    for f in _SEG_FIELDS
        hasproperty(s, f) || return "seg missing field: " * string(f)
    end
    s.version == _SEG_VERSION || return "seg version mismatch"
    s.seg == want_seg || return "seg tag mismatch: got $(s.seg), want $want_seg"
    nothing
end

function _src_from_prep(d; use_mu::Bool, use_chisq::Bool)
    predictive_rqmc_source(d.post, d.st, d.xt;
                           s1 = d.s1, E_active = d.E_active,
                           mu_qmc = use_mu, mu_chisq_qmc = use_chisq)
end

function _solve_layer(d, src, M::Int, rule_seed::UInt64, m_seed::UInt64)
    opt_rule = SobolOwenRule(src.rqmc_dim, rule_seed)
    X = src.gen(M, opt_rule, MersenneTwister(m_seed))
    locked = d.locked
    budget = 1.0 - sum(locked)
    base = locked_wealth_gate0(X, locked)
    free_pos = [k for k in 1:src.n_assets if locked[k] == 0]
    if isempty(free_pos)
        cert = (; feasibility = 0.0, kkt_residual = 0.0, objective_gap = 0.0,
                objective = isempty(X) ? 0.0 : mean(log.(budget .+ base)),
                dual = NaN)
        return (; X = X, w_full = zeros(src.n_assets), w_cash = budget, cert = cert)
    end
    X_free = X[:, free_pos]
    w_free, w_cash, cert = cash_kelly(X_free; base = base, budget = budget,
                                      tol = _KELLY_TOL)
    w_full = zeros(src.n_assets)
    w_full[free_pos] .= w_free
    for k in 1:src.n_assets
        locked[k] > 0 && (w_full[k] = locked[k])
    end
    (; X = X, w_full = w_full, w_cash = w_cash, cert = cert)
end

function _load_prep(path::String)
    isfile(path) || _fail(20, "prep not found: " * path)
    d = try
        deserialize(path)
    catch err
        _fail(20, "prep deserialize failed: " * sprint(showerror, err))
    end
    m = try
        _check_prep(d)
    catch err
        "prep contract check threw: " * sprint(showerror, err)
    end
    m === nothing || _fail(20, m)
    d
end

function _seg_path(out_dir::String, t::Int, seg::Int)
    joinpath(out_dir, "batch_t" * string(t) * "_seg" * string(seg) * ".bin")
end

function _run_layer(seg::Int, prep_path::String, out_dir::String)
    d = _load_prep(prep_path)
    use_mu = get(ENV, "GATE0_BATCH_MU_QMC", "true") == "true"
    use_chisq = get(ENV, "GATE0_BATCH_CHISQ_QMC", "true") == "true"
    min_s = parse(Int, get(ENV, "GATE0_BATCH_MIN", "65536"))
    max_s = parse(Int, get(ENV, "GATE0_BATCH_MAX", "131072"))
    M = seg == 1 ? min_s : max_s
    src = _src_from_prep(d; use_mu = use_mu, use_chisq = use_chisq)
    m_seed = Gate0BatchCaptureA._default_mu_seed(d.seed)
    res = try
        _solve_layer(d, src, M, d.seed, m_seed)
    catch err
        _fail(seg == 1 ? 31 : 32, "solve_layer(M=$M) failed: " * sprint(showerror, err))
    end
    payload = (; version = _SEG_VERSION, seg = seg, t = d.t, seed = d.seed, M = M,
                mu_qmc = use_mu, mu_chisq_qmc = use_chisq,
                w_full = res.w_full, w_cash = res.w_cash, cert = res.cert,
                X = res.X)
    path = _seg_path(out_dir, d.t, seg)
    try
        _atomic_serialize(path, payload)
    catch err
        _fail(seg == 1 ? 41 : 42, "write failed: " * sprint(showerror, err))
    end
    println("[batch-solve3] seg$seg ok t=$(d.t) M=$M -> " * path)
    exit(0)
end

function _run_combine(prep_path::String, out_dir::String)
    d = _load_prep(prep_path)
    s1p = _seg_path(out_dir, d.t, 1)
    s2p = _seg_path(out_dir, d.t, 2)
    isfile(s1p) || _fail(20, "seg1 not found: " * s1p)
    isfile(s2p) || _fail(20, "seg2 not found: " * s2p)
    s1 = deserialize(s1p)
    s2 = deserialize(s2p)
    for (s, w) in ((s1, 1), (s2, 2))
        m = _check_seg(s, w)
        m === nothing || _fail(20, m)
        (s.t == d.t && s.seed == d.seed) || _fail(20, "seg$w t/seed mismatch with prep")
    end
    (s1.mu_qmc == s2.mu_qmc && s1.mu_chisq_qmc == s2.mu_chisq_qmc) ||
        _fail(20, "seg1/seg2 switch mismatch")
    s2.M == 2 * s1.M || _fail(20, "seg2.M != 2*seg1.M")
    locked = d.locked
    N_R = length(locked)
    w_M = s1.w_full; wc_M = s1.w_cash
    w_2M = s2.w_full; wc_2M = s2.w_cash
    A = norm(vcat(w_2M, wc_2M) .- vcat(w_M, wc_M), 1)
    base_2M = locked_wealth_gate0(s2.X, locked)
    free_pos = [k for k in 1:N_R if locked[k] == 0]
    if isempty(free_pos)
        wealth_f_opt = wc_2M .+ base_2M
        wealth_c_opt = wc_M .+ base_2M
    else
        X_free_2M = s2.X[:, free_pos]
        wealth_f_opt = X_free_2M * w_2M[free_pos] .+ wc_2M .+ base_2M
        wealth_c_opt = X_free_2M * w_M[free_pos] .+ wc_M .+ base_2M
    end
    (all(x -> isfinite(x) && x > 0, wealth_f_opt) &&
     all(x -> isfinite(x) && x > 0, wealth_c_opt)) ||
        _fail(33, "B_opt objective nonpositive wealth")
    B_opt = sum(log, wealth_f_opt) / s2.M - sum(log, wealth_c_opt) / s2.M
    m_seed = Gate0BatchCaptureA._default_mu_seed(d.seed)
    am_seed = m_seed ⊻ 0x9E3779B97F4A7C15
    use_mu = s2.mu_qmc; use_chisq = s2.mu_chisq_qmc
    src = _src_from_prep(d; use_mu = use_mu, use_chisq = use_chisq)
    a_seed = audit_seed(d.seed)
    X_a = try
        src.gen(s2.M, SobolOwenRule(src.rqmc_dim, a_seed),
                MersenneTwister(am_seed))
    catch err
        _fail(33, "audit gen failed: " * sprint(showerror, err))
    end
    base_a = locked_wealth_gate0(X_a, locked)
    if isempty(free_pos)
        wealth_f = wc_2M .+ base_a
        wealth_c = wc_M .+ base_a
    else
        X_a_free = X_a[:, free_pos]
        wealth_f = X_a_free * w_2M[free_pos] .+ wc_2M .+ base_a
        wealth_c = X_a_free * w_M[free_pos] .+ wc_M .+ base_a
    end
    (all(x -> isfinite(x) && x > 0, wealth_f) &&
     all(x -> isfinite(x) && x > 0, wealth_c)) ||
        _fail(33, "audit objective nonpositive wealth")
    I_f = sum(log, wealth_f) / s2.M
    I_c = sum(log, wealth_c) / s2.M
    B_audit = abs(I_f - I_c)
    norm_err_2M = abs(sum(w_2M) + wc_2M - 1.0)
    norm_err_M = abs(sum(w_M) + wc_M - 1.0)
    (norm_err_2M <= 1e-8 && norm_err_M <= 1e-8) ||
        _fail(33, "normalization check failed: $norm_err_M / $norm_err_2M")
    cert = s2.cert
    text = join([
        "t=" * string(d.t),
        "seed=" * string(d.seed, base = 16),
        "M=" * string(s1.M) * "->" * string(s2.M),
        "A=" * string(A),
        "B_opt=" * string(B_opt),
        "B_audit=" * string(B_audit),
        "certificate_C_feasibility=" * string(cert.feasibility),
        "certificate_C_kkt=" * string(cert.kkt_residual),
        "certificate_C_gap=" * string(cert.objective_gap),
        "norm_err_M=" * string(norm_err_M),
        "norm_err_2M=" * string(norm_err_2M),
        "seg1_path=" * s1p,
        "seg2_path=" * s2p,
        ""
    ], "\n")
    out_path = joinpath(out_dir, "batch_t" * string(d.t) * "_solve3.txt")
    try
        _atomic_write_text(out_path, text)
    catch err
        _fail(43, "write failed: " * sprint(showerror, err))
    end
    println("[batch-solve3] b3 ok t=$(d.t) A=$(A) -> " * out_path)
    exit(0)
end

function _run_gen(seg::Int, prep_path::String, out_dir::String)
    d = _load_prep(prep_path)
    use_mu = get(ENV, "GATE0_BATCH_MU_QMC", "true") == "true"
    use_chisq = get(ENV, "GATE0_BATCH_CHISQ_QMC", "true") == "true"
    min_s = parse(Int, get(ENV, "GATE0_BATCH_MIN", "65536"))
    max_s = parse(Int, get(ENV, "GATE0_BATCH_MAX", "131072"))
    M = seg == 1 ? min_s : max_s
    src = _src_from_prep(d; use_mu = use_mu, use_chisq = use_chisq)
    m_seed = Gate0BatchCaptureA._default_mu_seed(d.seed)
    X = try
        src.gen(M, SobolOwenRule(src.rqmc_dim, d.seed),
                MersenneTwister(m_seed))
    catch err
        _fail(seg == 1 ? 34 : 36, "gen(M=$M) failed: " * sprint(showerror, err))
    end
    payload = (; version = _SEG_VERSION, kind = :X, seg = seg, t = d.t,
                seed = d.seed, M = M, mu_qmc = use_mu, mu_chisq_qmc = use_chisq,
                X = X)
    path = joinpath(out_dir, "batch_t" * string(d.t) * "_X" * string(seg) * ".bin")
    try
        _atomic_serialize(path, payload)
    catch err
        _fail(seg == 1 ? 44 : 45, "write failed: " * sprint(showerror, err))
    end
    println("[batch-solve3] b" * string(seg) * "a ok t=" * string(d.t) *
            " M=$M -> " * path)
    exit(0)
end

function _run_kelly_from_X(seg::Int, prep_path::String, out_dir::String)
    d = _load_prep(prep_path)
    xp = joinpath(out_dir, "batch_t" * string(d.t) * "_X" * string(seg) * ".bin")
    isfile(xp) || _fail(20, "X artifact not found: " * xp)
    xs = try
        deserialize(xp)
    catch err
        _fail(20, "X deserialize failed: " * sprint(showerror, err))
    end
    (xs isa NamedTuple && hasproperty(xs, :kind) && xs.kind == :X) ||
        _fail(20, "X artifact kind mismatch")
    for f in (:version, :seg, :t, :seed, :M, :mu_qmc, :mu_chisq_qmc, :X)
        hasproperty(xs, f) || _fail(20, "X missing field: " * string(f))
    end
    (xs.version == _SEG_VERSION && xs.seg == seg && xs.t == d.t &&
     xs.seed == d.seed) || _fail(20, "X contract mismatch")
    use_mu = get(ENV, "GATE0_BATCH_MU_QMC", "true") == "true"
    use_chisq = get(ENV, "GATE0_BATCH_CHISQ_QMC", "true") == "true"
    (xs.mu_qmc == use_mu && xs.mu_chisq_qmc == use_chisq) ||
        _fail(20, "X switch mismatch")
    locked = d.locked
    budget = 1.0 - sum(locked)
    base = locked_wealth_gate0(xs.X, locked)
    free_pos = [k for k in 1:length(locked) if locked[k] == 0]
    if isempty(free_pos)
        cert = (; feasibility = 0.0, kkt_residual = 0.0, objective_gap = 0.0,
                objective = isempty(xs.X) ? 0.0 : mean(log.(budget .+ base)),
                dual = NaN)
        w_full = zeros(length(locked)); w_cash = budget
    else
        X_free = xs.X[:, free_pos]
        w_free, w_cash, cert = try
            cash_kelly(X_free; base = base, budget = budget, tol = _KELLY_TOL)
        catch err
            _fail(seg == 1 ? 35 : 37, "cash_kelly failed: " * sprint(showerror, err))
        end
        w_full = zeros(length(locked))
        w_full[free_pos] .= w_free
        for k in 1:length(locked)
            locked[k] > 0 && (w_full[k] = locked[k])
        end
    end
    payload = (; version = _SEG_VERSION, seg = seg, t = d.t, seed = d.seed,
                M = xs.M, mu_qmc = use_mu, mu_chisq_qmc = use_chisq,
                w_full = w_full, w_cash = w_cash, cert = cert, X = xs.X)
    path = _seg_path(out_dir, d.t, seg)
    try
        _atomic_serialize(path, payload)
    catch err
        _fail(seg == 1 ? 41 : 42, "write failed: " * sprint(showerror, err))
    end
    println("[batch-solve3] b" * string(seg) * "b ok t=" * string(d.t) *
            " M=" * string(xs.M) * " -> " * path)
    exit(0)
end

function _load_X2(d, out_dir::String)
    xp = joinpath(out_dir, "batch_t" * string(d.t) * "_X2.bin")
    isfile(xp) || _fail(20, "X2 artifact not found: " * xp)
    xs = try
        deserialize(xp)
    catch err
        _fail(20, "X2 deserialize failed: " * sprint(showerror, err))
    end
    (xs isa NamedTuple && hasproperty(xs, :kind) && xs.kind == :X && xs.seg == 2) ||
        _fail(20, "X2 artifact kind/seg mismatch")
    use_mu = get(ENV, "GATE0_BATCH_MU_QMC", "true") == "true"
    use_chisq = get(ENV, "GATE0_BATCH_CHISQ_QMC", "true") == "true"
    (xs.mu_qmc == use_mu && xs.mu_chisq_qmc == use_chisq) ||
        _fail(20, "X2 switch mismatch")
    (xs.t == d.t && xs.seed == d.seed) || _fail(20, "X2 t/seed mismatch")
    xs
end

function _run_b2b1(prep_path::String, out_dir::String)
    d = _load_prep(prep_path)
    xs = _load_X2(d, out_dir)
    locked = d.locked
    budget = 1.0 - sum(locked)
    base = locked_wealth_gate0(xs.X, locked)
    free_pos = [k for k in 1:length(locked) if locked[k] == 0]
    X_free = xs.X[:, free_pos]
    pre = try
        _cash_kelly_prelude(X_free; base = base, budget = budget, tol = _KELLY_TOL)
    catch err
        _fail(38, "prelude failed: " * sprint(showerror, err))
    end
    payload = (; version = _SEG_VERSION, kind = :pre, seg = 2, t = d.t,
                seed = d.seed, M = xs.M, mu_qmc = xs.mu_qmc,
                mu_chisq_qmc = xs.mu_chisq_qmc, free_pos = free_pos,
                b = pre.b, budget = pre.budget, tol = pre.tol,
                tie_eps = pre.tie_eps, cs = pre.cs,
                scale_preferred = pre.scale_preferred, res = pre.res,
                early = pre.early)
    path = joinpath(out_dir, "batch_t" * string(d.t) * "_pre2.bin")
    try
        _atomic_serialize(path, payload)
    catch err
        _fail(46, "write failed: " * sprint(showerror, err))
    end
    println("[batch-solve3] b2b1 ok t=" * string(d.t) * " -> " * path)
    exit(0)
end

function _run_b2b2(prep_path::String, out_dir::String)
    # B2B-SPLIT-2：b2b2 改为「候选构建段」（_cash_kelly_finish_main）→ mid 产物；
    # tie-break 段移到 b2b3（读 mid → _cash_kelly_finish_tiebreak → seg2）。
    d = _load_prep(prep_path)
    xs = _load_X2(d, out_dir)
    pp = joinpath(out_dir, "batch_t" * string(d.t) * "_pre2.bin")
    isfile(pp) || _fail(20, "pre2 artifact not found: " * pp)
    pre = try
        deserialize(pp)
    catch err
        _fail(20, "pre2 deserialize failed: " * sprint(showerror, err))
    end
    (pre isa NamedTuple && hasproperty(pre, :kind) && pre.kind == :pre) ||
        _fail(20, "pre2 kind mismatch")
    locked = d.locked
    budget = 1.0 - sum(locked)
    free_pos = [k for k in 1:length(locked) if locked[k] == 0]
    pre.free_pos == free_pos || _fail(20, "pre2 free_pos mismatch")
    abs(pre.budget - budget) <= 0 || _fail(20, "pre2 budget mismatch")
    base = locked_wealth_gate0(xs.X, locked)
    (length(pre.b) == length(base) && all(pre.b .== base)) ||
        _fail(20, "pre2 base mismatch")
    X_free = xs.X[:, free_pos]
    pre2 = (; early = pre.early, b = pre.b, budget = budget, tol = pre.tol,
            tie_eps = pre.tie_eps, scale_retry_diag = nothing,
            cs = pre.cs, scale_preferred = pre.scale_preferred, res = pre.res)
    mid = try
        _cash_kelly_finish_main(X_free, pre2)
    catch err
        _fail(39, "finish_main failed: " * sprint(showerror, err))
    end
    payload = (; version = _SEG_VERSION, kind = :mid, seg = 2, t = d.t,
                seed = d.seed, M = xs.M, mu_qmc = xs.mu_qmc,
                mu_chisq_qmc = xs.mu_chisq_qmc, free_pos = free_pos,
                b = mid.b, budget = mid.budget, tol = mid.tol,
                tie_eps = mid.tie_eps, w_val = mid.w_val, cert = mid.cert,
                w_risky = mid.w_risky, w_cash = mid.w_cash, skip_tb = mid.skip_tb)
    path = joinpath(out_dir, "batch_t" * string(d.t) * "_mid2.bin")
    try
        _atomic_serialize(path, payload)
    catch err
        _fail(48, "write failed: " * sprint(showerror, err))
    end
    println("[batch-solve3] b2b2 ok t=" * string(d.t) * " -> " * path)
    exit(0)
end

function _run_b2b3(prep_path::String, out_dir::String)
    d = _load_prep(prep_path)
    xs = _load_X2(d, out_dir)
    mp = joinpath(out_dir, "batch_t" * string(d.t) * "_mid2.bin")
    isfile(mp) || _fail(20, "mid2 artifact not found: " * mp)
    m2 = try
        deserialize(mp)
    catch err
        _fail(20, "mid2 deserialize failed: " * sprint(showerror, err))
    end
    for f in (:version, :kind, :seg, :t, :seed, :M, :mu_qmc, :mu_chisq_qmc,
              :free_pos, :b, :budget, :tol, :tie_eps, :w_val, :cert,
              :w_risky, :w_cash, :skip_tb)
        hasproperty(m2, f) || _fail(20, "mid2 missing field: " * string(f))
    end
    (m2.kind == :mid && m2.seg == 2 && m2.t == d.t && m2.seed == d.seed) ||
        _fail(20, "mid2 contract mismatch")
    locked = d.locked
    budget = 1.0 - sum(locked)
    free_pos = [k for k in 1:length(locked) if locked[k] == 0]
    m2.free_pos == free_pos || _fail(20, "mid2 free_pos mismatch")
    base = locked_wealth_gate0(xs.X, locked)
    (length(m2.b) == length(base) && all(m2.b .== base)) ||
        _fail(20, "mid2 base mismatch")
    X_free = xs.X[:, free_pos]
    mid = (; b = m2.b, budget = budget, tol = m2.tol, tie_eps = m2.tie_eps,
           w_val = m2.w_val, cert = m2.cert, w_risky = m2.w_risky,
           w_cash = m2.w_cash, skip_tb = m2.skip_tb)
    r = try
        _cash_kelly_finish_tiebreak(X_free, nothing, mid)
    catch err
        _fail(40, "finish_tiebreak failed: " * sprint(showerror, err))
    end
    w_full = zeros(length(locked))
    w_full[free_pos] .= r[1]
    for k in 1:length(locked)
        locked[k] > 0 && (w_full[k] = locked[k])
    end
    payload = (; version = _SEG_VERSION, seg = 2, t = d.t, seed = d.seed,
                M = xs.M, mu_qmc = xs.mu_qmc, mu_chisq_qmc = xs.mu_chisq_qmc,
                w_full = w_full, w_cash = r[2], cert = r[3], X = xs.X)
    path = _seg_path(out_dir, d.t, 2)
    try
        _atomic_serialize(path, payload)
    catch err
        _fail(47, "write failed: " * sprint(showerror, err))
    end
    println("[batch-solve3] b2b3 ok t=" * string(d.t) * " -> " * path)
    exit(0)
end

function _run_b2b1a(prep_path::String, out_dir::String)
    # B2B-SPLIT-3：prep 段（校验/退化/cs/分流）→ prep1 快照；不落 Xt 大矩阵。
    d = _load_prep(prep_path)
    xs = _load_X2(d, out_dir)
    locked = d.locked
    budget = 1.0 - sum(locked)
    base = locked_wealth_gate0(xs.X, locked)
    free_pos = [k for k in 1:length(locked) if locked[k] == 0]
    X_free = xs.X[:, free_pos]
    prep = try
        _cash_kelly_prelude_prep(X_free; base = base, budget = budget,
                                 tol = _KELLY_TOL)
    catch err
        _fail(50, "prelude_prep failed: " * sprint(showerror, err))
    end
    payload = (; version = _SEG_VERSION, kind = :prep1, seg = 1, t = d.t,
                seed = d.seed, M = xs.M, mu_qmc = xs.mu_qmc,
                mu_chisq_qmc = xs.mu_chisq_qmc, free_pos = free_pos,
                b = prep.b, budget = prep.budget, tol = prep.tol,
                tie_eps = prep.tie_eps, cs = prep.cs,
                scale_preferred = prep.scale_preferred,
                scale_retry_diag = prep.scale_retry_diag, early = prep.early)
    path = joinpath(out_dir, "batch_t" * string(d.t) * "_prep1.bin")
    try
        _atomic_serialize(path, payload)
    catch err
        _fail(52, "write failed: " * sprint(showerror, err))
    end
    println("[batch-solve3] b2b1a ok t=" * string(d.t) * " -> " * path)
    exit(0)
end

function _run_b2b1b(prep_path::String, out_dir::String)
    # B2B-SPLIT-3：主求解段（_cash_kelly_prelude_solve）→ pre2 快照（含 res）。
    d = _load_prep(prep_path)
    xs = _load_X2(d, out_dir)
    pp1 = joinpath(out_dir, "batch_t" * string(d.t) * "_prep1.bin")
    isfile(pp1) || _fail(20, "prep1 artifact not found: " * pp1)
    prep = try
        deserialize(pp1)
    catch err
        _fail(20, "prep1 deserialize failed: " * sprint(showerror, err))
    end
    (prep isa NamedTuple && hasproperty(prep, :kind) && prep.kind == :prep1) ||
        _fail(20, "prep1 kind mismatch")
    locked = d.locked
    budget = 1.0 - sum(locked)
    free_pos = [k for k in 1:length(locked) if locked[k] == 0]
    prep.free_pos == free_pos || _fail(20, "prep1 free_pos mismatch")
    abs(prep.budget - budget) <= 0 || _fail(20, "prep1 budget mismatch")
    X_free = xs.X[:, free_pos]
    sol = try
        _cash_kelly_prelude_solve(X_free, prep)
    catch err
        _fail(51, "prelude_solve failed: " * sprint(showerror, err))
    end
    payload = (; version = _SEG_VERSION, kind = :pre, seg = 2, t = d.t,
                seed = d.seed, M = xs.M, mu_qmc = xs.mu_qmc,
                mu_chisq_qmc = xs.mu_chisq_qmc, free_pos = free_pos,
                b = sol.b, budget = sol.budget, tol = sol.tol,
                tie_eps = sol.tie_eps, cs = sol.cs,
                scale_preferred = sol.scale_preferred, res = sol.res,
                early = sol.early)
    path = joinpath(out_dir, "batch_t" * string(d.t) * "_pre2.bin")
    try
        _atomic_serialize(path, payload)
    catch err
        _fail(53, "write failed: " * sprint(showerror, err))
    end
    println("[batch-solve3] b2b1b ok t=" * string(d.t) * " -> " * path)
    exit(0)
end

function _selftest()
    ok = true
    try
        good = (; version = _PREP_VERSION, t = 330, seed = UInt64(1),
                held_mode = :zeros, data_note = "d", rule_kwargs = (;),
                post = nothing, st = nothing, xt = [0.0], s1 = [1.0],
                E_active = zeros(1, 1), locked = [0.0], budget = 1.0)
        @assert _check_prep(good) === nothing
        @assert _check_prep((; version = _PREP_VERSION, t = 330)) !== nothing
        seg = (; version = _SEG_VERSION, seg = 1, t = 330, seed = UInt64(1),
                M = 65536, mu_qmc = true, mu_chisq_qmc = true,
                w_full = [0.0], w_cash = 1.0, cert = (;), X = zeros(1, 1))
        @assert _check_seg(seg, 1) === nothing
        @assert _check_seg(seg, 2) !== nothing
        @assert _check_seg(merge(seg, (; version = "x")), 1) !== nothing
    catch err
        ok = false
        println(stderr, "[batch-solve3 selftest] FAILED: " * sprint(showerror, err))
    end
    ok || exit(3)
    println("[batch-solve3 selftest] ok (6 checks; kelly not invoked)")
    exit(0)
end

function main()
    length(ARGS) >= 2 || _fail(2, "usage: b1|b2|b3 <prep.bin> [out_dir]")
    mode = ARGS[1]
    prep_path = ARGS[2]
    out_dir = length(ARGS) >= 3 ? ARGS[3] : dirname(prep_path)
    mkpath(out_dir)
    if mode == "b1"
        _run_layer(1, prep_path, out_dir)
    elseif mode == "b2"
        _run_layer(2, prep_path, out_dir)
    elseif mode == "b1a"
        _run_gen(1, prep_path, out_dir)
    elseif mode == "b1b"
        _run_kelly_from_X(1, prep_path, out_dir)
    elseif mode == "b2a"
        _run_gen(2, prep_path, out_dir)
    elseif mode == "b2b"
        _run_kelly_from_X(2, prep_path, out_dir)
    elseif mode == "b2b1"
        _run_b2b1(prep_path, out_dir)
    elseif mode == "b2b1a"
        _run_b2b1a(prep_path, out_dir)
    elseif mode == "b2b1b"
        _run_b2b1b(prep_path, out_dir)
    elseif mode == "b2b2"
        _run_b2b2(prep_path, out_dir)
    elseif mode == "b2b3"
        _run_b2b3(prep_path, out_dir)
    elseif mode == "b3"
        _run_combine(prep_path, out_dir)
    else
        _fail(2, "unknown mode: " * mode)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1 && ARGS[1] == "--selftest"
        _selftest()
    else
        main()
    end
end