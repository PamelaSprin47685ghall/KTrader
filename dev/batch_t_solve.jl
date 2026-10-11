#!/usr/bin/env julia
# =============================================================================
# dev/batch_t_solve.jl —— 批量两段式调度 · 段 B（离线 kelly 求解）
# =============================================================================
# 读段 A 的 prep 捕获（batch_t<T>_prep.bin）-> adaptive_scenario_kelly（过门配置：
# mu_qmc=true + chisq=true + min=65536 + max=131072）-> 落盘 batch_t<T>_solve.txt。
#
# 用法：julia --startup-file=no --project=. dev/batch_t_solve.jl <prep.bin> [out.txt]
#       可选环境：GATE0_BATCH_MIN（默认 65536）、GATE0_BATCH_MAX（默认 131072）、
#                 GATE0_BATCH_MU_QMC（默认 true）、GATE0_BATCH_CHISQ_QMC（默认 true）。
# 自检：julia --startup-file=no --project=. dev/batch_t_solve.jl --selftest
#
# 退出码（独立可辨）：0 成功；2 参数错误；3 自检失败；20 读取/契约校验失败；
# 21 求解失败（含 D-067 fail-loud）；22 落盘失败；23 解完整性（归一性）失败。
#
# 等价性说明（min 近似）：min=65536 跳过 64..65536 的中间层；t=330 已知该区间
# 全部不过。若某日中间层可早停，min=65536 会跳过该早停——首版以深段判定运行，
# min 可配置，该近似登记于 docs/MU_CHANNEL_QMC_DESIGN.md 与 batch_two_stage.md。
# =============================================================================

using Serialization

# 模块名必须与段 A（batch_t_capture.jl）完全一致：Julia serialize 记录类型的
# 模块路径（Main.Gate0BatchCaptureA.*），反序列化时按名查找——不同名则 UndefVarError。
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

const _PREP_VERSION = "batch-prep-v1"

_fail(code::Int, msg::String) = (println(stderr, "[batch-solve] " * msg); exit(code))

const _REQUIRED_FIELDS = (:version, :t, :seed, :held_mode, :data_note,
                          :rule_kwargs, :post, :st, :xt, :s1, :E_active,
                          :locked, :budget)

function _check_prep(d)::Union{Nothing,String}
    (d isa NamedTuple) || return "prep is not a NamedTuple"
    for f in _REQUIRED_FIELDS
        hasproperty(d, f) || return "prep missing field: " * string(f)
    end
    d.version == _PREP_VERSION ||
        return "prep version mismatch: got " * string(d.version) *
               ", want " * _PREP_VERSION
    d.held_mode == :zeros ||
        return "unsupported held_mode: " * string(d.held_mode) *
               " (first version supports :zeros only)"
    nothing
end

function _norm_check(w::AbstractVector{Float64}, w_cash::Float64)::Union{Nothing,String}
    e = abs(sum(w) + w_cash - 1.0)
    e <= 1e-8 ? nothing : "normalization check failed: |sum(w)+cash-1| = $e"
end

function _atomic_write_text(path::String, text::String)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        print(io, text)
    end
    mv(tmp, path; force = true)
end

function main()
    length(ARGS) >= 1 || _fail(2, "usage: batch_t_solve.jl <prep.bin> [out.txt]")
    prep_path = ARGS[1]
    isfile(prep_path) || _fail(20, "prep not found: " * prep_path)
    d = try
        deserialize(prep_path)
    catch err
        _fail(20, "prep deserialize failed (corrupt/incomplete): " * prep_path *
                  " — " * sprint(showerror, err))
    end
    msg = try
        _check_prep(d)
    catch err
        "prep contract check threw: " * sprint(showerror, err)
    end
    msg === nothing || _fail(20, msg)
    min_s = parse(Int, get(ENV, "GATE0_BATCH_MIN", "65536"))
    max_s = parse(Int, get(ENV, "GATE0_BATCH_MAX", "131072"))
    use_mu = get(ENV, "GATE0_BATCH_MU_QMC", "true") == "true"
    use_chisq = get(ENV, "GATE0_BATCH_CHISQ_QMC", "true") == "true"
    t0 = time()
    res = try
        adaptive_scenario_kelly(d.post, d.st, d.xt;
                                s1 = d.s1, E_active = d.E_active,
                                rule_seed = d.seed, locked = d.locked,
                                min_scenarios = min_s, max_scenarios = max_s,
                                mu_qmc = use_mu, mu_chisq_qmc = use_chisq)
    catch err
        _fail(21, "kelly solve failed: " * sprint(showerror, err))
    end
    elapsed = time() - t0
    nmsg = _norm_check(res.w_risky, res.w_cash)
    nmsg === nothing || _fail(23, nmsg)
    out_path = length(ARGS) >= 2 ? ARGS[2] :
        joinpath(dirname(prep_path), "batch_t" * string(d.t) * "_solve.txt")
    text = join([
        "t=" * string(d.t),
        "seed=" * string(d.seed, base = 16),
        "min_scenarios=" * string(min_s),
        "max_scenarios=" * string(max_s),
        "mu_qmc=" * string(use_mu),
        "mu_chisq_qmc=" * string(use_chisq),
        "M=" * string(res.M),
        "w_risky=" * join(res.w_risky, ","),
        "w_cash=" * string(res.w_cash),
        "certificate_A=" * string(res.certificate_A),
        "certificate_B_opt=" * string(res.certificate_B),
        "certificate_B_audit=" * string(res.certificate_B_audit),
        "certificate_C_feasibility=" * string(res.certificate_C.feasibility),
        "certificate_C_kkt=" * string(res.certificate_C.kkt_residual),
        "certificate_C_gap=" * string(res.certificate_C.objective_gap),
        "elapsed_s=" * string(elapsed),
        "norm_err=" * string(abs(sum(res.w_risky) + res.w_cash - 1.0)),
        ""
    ], "\n")
    try
        _atomic_write_text(out_path, text)
    catch err
        _fail(22, "write failed: " * sprint(showerror, err))
    end
    println("[batch-solve] ok t=" * string(d.t) * " M=" * string(res.M) *
            " elapsed=" * string(round(elapsed; digits = 1)) * "s -> " * out_path)
    exit(0)
end

function _selftest()
    ok = true
    good = (; version = _PREP_VERSION, t = 330, seed = UInt64(1),
            held_mode = :zeros, data_note = "d", rule_kwargs = (;),
            post = nothing, st = nothing, xt = [0.0], s1 = [1.0],
            E_active = zeros(1, 1), locked = [0.0], budget = 1.0)
    try
        @assert _check_prep(good) === nothing
        @assert _check_prep(merge(good, (; version = "wrong"))) !== nothing
        @assert _check_prep(merge(good, (; held_mode = :nonzero))) !== nothing
        @assert _check_prep((; version = _PREP_VERSION, t = 330)) !== nothing
        @assert _check_prep(42) !== nothing
        @assert _norm_check([0.5], 0.5) === nothing
        @assert _norm_check([0.5], 0.4) !== nothing
    catch err
        ok = false
        println(stderr, "[batch-solve selftest] FAILED: " * sprint(showerror, err))
    end
    ok || exit(3)
    println("[batch-solve selftest] ok (7 checks; kelly not invoked)")
    exit(0)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1 && ARGS[1] == "--selftest"
        _selftest()
    else
        main()
    end
end