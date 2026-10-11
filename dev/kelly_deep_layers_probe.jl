#!/usr/bin/env julia
# =============================================================================
# dev/kelly_deep_layers_probe.jl —— A 判据加深复核前置实验（批次 F）
# =============================================================================
#
# 目的
# ----
# 对给定决策日 t，在多个 scenario 规模层（默认 131072 → 262144 → 524288，
# 可含 1048576）上，用**生产 IPM 核心**（src/gate0/kelly.jl 的
# _cash_kelly_ipm；即 _cash_kelly_solve 的 :ipm 装配所调用的同一函数）求解
# cash Kelly，输出各层证书与相邻层对的 A = ‖w_2M − w_M‖₁、B_opt、各层解与
# 基准层解的 maxdiff，以及嵌套前缀逐位检查——供「336/340/343/… 类未过门日
# 的加深是否收缩」判定使用。
#
# 静态事实（引用；本脚本不改任何生产语义）
# ----
# [事实] 层配置面：生产入口 bin/backtest.jl 的
#   GATE0_MODE=adaptive + GATE0_MAX_SCENARIOS（默认 512；过门档 131072）
#   + GATE0_MU_QMC / GATE0_CHISQ_QMC（过门配置 true/true）
#   + GATE0_WEIGHT_TOL（默认 1e-4，过门口径）+ GATE0_KELLY_TOL（默认 1e-6）。
#   driver 的 adaptive 循环从 min_scenarios（driver.jl:250 默认 64）逐层
#   翻倍到 max_scenarios（quadrature.jl:576-598），**每层对检查 A/B/C 四证书**，
#   只在当前对全过时返回；批量链（dev/batch_two_stage.md「等价性近似」节）
#   跳过 64..32768 的中间层（登记：t=330 已知该区间不早停），只复现最终对。
#   因此未过门日的加深语义 = 把 max 提到 262144，使收敛检查落在
#   A(131072→262144) 上（CQ 表：32768→65536 A=2.618e-4 不过、
#   65536→131072 A=6.26e-5 过——引用 archive/.../CQ_summary.md）。
# [事实] 嵌套性硬保证：quadrature.jl:204-232（rule_points docstring）
#   ——「rule_points(rule, M)[1:M,:] 与更大 M' 的前 M 行逐位相同」；同加
#   quadrature.jl:600-603（adaptive docstring）与 :457-505（gen 内固定 seed
#   重建的 μ 流、顺序消费）——**262144 层 X 的前 131072 行与 131072 层 X
#   逐位一致**。本脚本在运行时对每个层对做 nest_maxdiff 自检（应为 0）。
# [事实] X 生成机制：predictive_rqmc_source(...).gen(M, SobolOwenRule(dim,
#   rule_seed), MersenneTwister(m_seed))，rule_seed = prep.seed，
#   m_seed = _default_mu_seed(prep.seed)（quadrature.jl:149）；与
#   dev/batch_t_solve3.jl:126-148 的 _solve_layer 逐点同构。
# [事实] deep336 生成方式（批次 F 调查问题 3）：`GATE0_BATCH_MAX=262144`
#   下运行 `dev/batch_t_solve3.jl b2a <prep> <out_dir>` 生成
#   batch_t336_X2.bin（262144 层；证据 S3_t336d_b2a.log 第 1 行 +
#   scoped_run 尾行 rc=0 elapsed=11s rss_peak=1058MiB）；随后 b2b1
#   （prelude@262144，44s）与 b2b2（finish_main@262144，124×2 出窗）——
#   tiebreak/seg2 未产出，A(131072→262144) 历史未取得（S3_summary.md §1）。
#   产物在 archive/evidence/gate0_multiday_run_20261010/batch_out/deep336/。
# [估计] 成本外推（静态；含冷启动的实测段见括号）：
#   X 大小 = S×N_R×8B（N_R=1 时 131072→1.05MB、262144→2.1MB、524288→4.2MB、
#   1048576→8.4MB；增广视角 S×(N_R+1) 翻倍）；
#   gen：b2a 段实测 11s@131072、12s@262144（含 include 冷启动 ~9-10s）→
#   纯 gen 增量估计 ~1-3s/翻倍；solve（生产 IPM 实测 131072 层 0.105-0.159s）
#   → 线性外推估计 262144 ~0.2-0.3s、524288 ~0.4-0.6s、1048576 ~0.8-1.2s、
#   2097152 ~1.6-2.4s。单命令预算建议上限：1048576 层（总估 <40s）；
#   2097152 层超窗风险高——拆分或登记（不改任何调度脚本）。
#
# 用法
# ----
#   julia --startup-file=no --project=. dev/kelly_deep_layers_probe.jl <t> [layers] [out_dir]
#
#   layers  逗号分隔、严格递增；默认 "131072,262144,524288"（可含 1048576）
#   out_dir prep / X2 产物所在目录；默认 dev/batch_out。
#           历史产物目录：archive/evidence/gate0_multiday_run_20261010/batch_out
#           （直接作为第三个参数传入即可读 batch_t<T>_prep.bin 与 _X2.bin）
#
# 环境（与 batch 工具同名开关，保证与过门配置逐点一致）
# ----
#   GATE0_BATCH_MU_QMC      默认 true
#   GATE0_BATCH_CHISQ_QMC   默认 true
#   KELLY_DEEP_LOAD_X2      默认 true：基准层优先加载 out_dir/batch_t<T>_X2.bin
#                           （校验 kind/seg/M/t/seed/开关；不匹配则重新 gen）
#   KELLY_DEEP_KELLY_TOL    默认 1e-6（生产 bin/backtest.jl 默认口径；
#                           复现 batch 链历史 A 时用 1e-8）
#   KELLY_DEEP_WEIGHT_TOL   默认 1e-4（A 参考判据；正式门仍归生产链）
#
# 输出
# ----
#   stdout + <out_dir>/deep_layers_t<T>.txt（原子写）：
#     每层 DEEP_LAYER 行：origin/gen_s/solve_s/iters/ok/reason + 证书三量
#     每个层对 DEEP_PAIR 行：A / A_pass(参考) / B_opt / maxdiff_vs_base /
#                            nest_maxdiff
#
# 退出码（独立可辨）
# ----
#   0 全层认证；2 参数；20 prep/X/gen/契约；41 任一层求解未认证（数据仍落盘）；
#   42 落盘失败；43 装配与核心不一致（不应发生——确定性同函数）。
#
# 依赖说明（如实；复用既有非公开路径，不改生产源码迁就实验）
# ----
# * 模块结构复刻 dev/batch_t_solve3.jl（module Gate0BatchCaptureA + 同一
#   9 文件链），以便反序列化 batch_t<T>_prep.bin（版本 batch-prep-v1）与
#   batch_t<T>_X2.bin（契约 batch-solve3-v1：kind=:X、seg=2）。
# * 未导出名字（_cash_kelly_ipm / _cash_kelly_solve / _default_mu_seed /
#   cash_kelly_certificate 等）按 batch 工具先例显式导入。
# * 求解不引入 RNG；gen 用 MersenneTwister(m_seed)（与生产同 seed 机制）。
#
# 边界
# ----
# 本脚本是受控实验工具，不是生产门：A 的正式判定仍在生产链 / batch b3 口径。
# 不改任何 src/test/调度脚本；不放松任何容差；结果供加深复核批次使用。
# =============================================================================

using Serialization, Random, LinearAlgebra, Printf

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
using .Gate0BatchCaptureA: predictive_rqmc_source, SobolOwenRule,
    locked_wealth_gate0, cash_kelly_certificate, _cash_kelly_ipm,
    _cash_kelly_solve, _default_mu_seed

const _DEEP_VERSION = "deep-layers-probe-v1"
const _PREP_VERSION = "batch-prep-v1"

_fail(code::Int, msg::String) = (println(stderr, "[deep-layers] " * msg); exit(code))

function _atomic_write_text(path::String, text::String)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        print(io, text)
    end
    mv(tmp, path; force = true)
end

"""嵌套前缀检查：X_new[1:M_old, :] 与 X_old 的逐元素最大绝对差。
硬保证下应为 0.0（bitwise 相同）；非 0 视为重大异常（打印、不掩盖）。"""
function _nest_maxdiff(X_new::Matrix{Float64}, X_old::Matrix{Float64})
    size(X_new, 2) == size(X_old, 2) || return NaN
    M_old = size(X_old, 1)
    d = 0.0
    for j in 1:size(X_old, 2)
        for s in 1:M_old
            v = abs(X_new[s, j] - X_old[s, j])
            v > d && (d = v)
        end
    end
    d
end

"""尝试加载既有基准层 X2 产物（batch-solve3-v1 契约；不匹配返回原因）。"""
function _try_load_x2(out_dir::String, t::Int, M::Int, d_seed::UInt64,
                      use_mu::Bool, use_chisq::Bool)
    p = joinpath(out_dir, "batch_t$(t)_X2.bin")
    isfile(p) || return (nothing, "not found: " * p)
    xs = try
        deserialize(p)
    catch err
        return (nothing, "deserialize failed: " * sprint(showerror, err))
    end
    (xs isa NamedTuple && hasproperty(xs, :kind) && xs.kind == :X) ||
        return (nothing, "not an X artifact")
    (hasproperty(xs, :seg) && xs.seg == 2) || return (nothing, "seg != 2")
    (hasproperty(xs, :t) && xs.t == t) || return (nothing, "t mismatch")
    (hasproperty(xs, :seed) && xs.seed == d_seed) || return (nothing, "seed mismatch")
    (hasproperty(xs, :M) && xs.M == M) ||
        return (nothing, "M mismatch ($(get(xs, :M, -1)) != $M)")
    (hasproperty(xs, :mu_qmc) && xs.mu_qmc == use_mu &&
     hasproperty(xs, :mu_chisq_qmc) && xs.mu_chisq_qmc == use_chisq) ||
        return (nothing, "switch mismatch")
    (getproperty(xs, :X), "")
end

"""从 X 求解一层（生产 IPM 核心）+ 原空间证书；可选装配对拍。"""
function _solve_layer_from_X(d, X::Matrix{Float64}, M::Int, kelly_tol::Float64;
                             assembly_check::Bool=false)
    locked = d.locked
    N_R = length(locked)
    budget = 1.0 - sum(locked)
    base = locked_wealth_gate0(X, locked)
    free_pos = [k for k in 1:N_R if locked[k] == 0]
    n_free = length(free_pos)
    Xf = X[:, free_pos]
    t0 = time_ns()
    core = _cash_kelly_ipm(Xf, base, ones(M), budget, kelly_tol)
    solve_s = (time_ns() - t0) / 1e9

    w_val = core.w_val
    w_full = zeros(N_R)
    w_cash = NaN
    cert = nothing
    if core.converged && w_val !== nothing
        w_full[free_pos] .= w_val[1:n_free]
        w_cash = w_val[n_free + 1]
        if n_free > 0
            cert = cash_kelly_certificate(Xf, w_val[1:n_free], w_val[n_free + 1];
                                          base = base, budget = budget)
        else
            cert = (; feasibility = 0.0, kkt_residual = 0.0, objective_gap = 0.0,
                    objective = isempty(X) ? 0.0 : mean(log.(budget .+ base)),
                    dual = NaN)
        end
    end
    for k in 1:N_R
        locked[k] > 0 && (w_full[k] = locked[k])
    end

    asm_maxdiff = NaN
    if assembly_check && n_free > 0
        asm = _cash_kelly_solve(Xf, base, ones(M), budget, kelly_tol;
                                backend = :ipm)
        if asm.w_val !== nothing && w_val !== nothing
            asm_maxdiff = maximum(abs.(asm.w_val .- w_val))
        end
    end
    (; w_full = w_full, w_cash = w_cash, cert = cert, converged = core.converged,
       reason = core.reason, iters = core.iters, solve_s = solve_s,
       base = base, free_pos = free_pos, asm_maxdiff = asm_maxdiff,
       w_aug = vcat(w_full, w_cash))
end

"""B_opt：在 fine 层 X 上求两个权重的目标差 I_fine(w_fine) − I_fine(w_coarse)
（与 batch_t_solve3.jl b3 的 B_opt 公式同式；非正 wealth 返回 NaN）。"""
function _B_opt(X_fine::Matrix{Float64}, base_fine::Vector{Float64},
                free_pos::Vector{Int}, w_fine_aug::Vector{Float64},
                w_coarse_aug::Vector{Float64})
    Xf = X_fine[:, free_pos]
    nf = Xf * w_fine_aug[free_pos] .+ w_fine_aug[end] .+ base_fine
    nc = Xf * w_coarse_aug[free_pos] .+ w_coarse_aug[end] .+ base_fine
    (all(x -> isfinite(x) && x > 0, nf) &&
     all(x -> isfinite(x) && x > 0, nc)) || return NaN
    sum(log, nf) / size(Xf, 1) - sum(log, nc) / size(Xf, 1)
end

function main()
    length(ARGS) >= 1 ||
        _fail(2, "usage: kelly_deep_layers_probe.jl <t> [layers] [out_dir]")
    t = try
        parse(Int, ARGS[1])
    catch
        _fail(2, "cannot parse t: " * ARGS[1])
    end
    layers_str = length(ARGS) >= 2 ? ARGS[2] :
                 get(ENV, "KELLY_DEEP_LAYERS", "131072,262144,524288")
    layers = try
        [parse(Int, strip(s)) for s in split(layers_str, ',')]
    catch
        _fail(2, "cannot parse layers: " * layers_str)
    end
    isempty(layers) && _fail(2, "empty layer list")
    for k in 2:length(layers)
        layers[k] > layers[k - 1] ||
            _fail(2, "layers must be strictly increasing: $layers_str")
    end
    out_dir = length(ARGS) >= 3 ? ARGS[3] :
              get(ENV, "KELLY_DEEP_OUT", joinpath(@__DIR__, "..", "dev", "batch_out"))
    mkpath(out_dir)
    use_mu = get(ENV, "GATE0_BATCH_MU_QMC", "true") == "true"
    use_chisq = get(ENV, "GATE0_BATCH_CHISQ_QMC", "true") == "true"
    load_x2 = get(ENV, "KELLY_DEEP_LOAD_X2", "true") == "true"
    kelly_tol = parse(Float64, get(ENV, "KELLY_DEEP_KELLY_TOL", "1e-6"))
    weight_tol = parse(Float64, get(ENV, "KELLY_DEEP_WEIGHT_TOL", "1e-4"))

    prep_path = joinpath(out_dir, "batch_t$(t)_prep.bin")
    isfile(prep_path) ||
        _fail(20, "prep not found: $prep_path（历史产物目录：archive/evidence/gate0_multiday_run_20261010/batch_out —— 作为 out_dir 传入；或先跑 batch_t_capture.jl）")
    d = try
        deserialize(prep_path)
    catch err
        _fail(20, "prep deserialize failed: " * sprint(showerror, err))
    end
    for f in (:version, :t, :seed, :post, :st, :xt, :s1, :E_active, :locked, :budget)
        hasproperty(d, f) || _fail(20, "prep missing field: " * string(f))
    end
    d.version == _PREP_VERSION || _fail(20, "prep version mismatch: $(d.version)")
    d.t == t || _fail(20, "prep t mismatch: $(d.t) != $t")
    d.seed isa UInt64 || _fail(20, "prep seed is not UInt64")

    src = try
        predictive_rqmc_source(d.post, d.st, d.xt;
                               s1 = d.s1, E_active = d.E_active,
                               mu_qmc = use_mu, mu_chisq_qmc = use_chisq)
    catch err
        _fail(20, "predictive_rqmc_source failed: " * sprint(showerror, err))
    end
    m_seed = _default_mu_seed(d.seed)

    lines = String[]
    push!(lines, "# " * _DEEP_VERSION * " t=" * string(t) *
                 " seed=0x" * string(d.seed, base = 16) *
                 " mu_qmc=" * string(use_mu) * " chisq_qmc=" * string(use_chisq) *
                 " kelly_tol=" * string(kelly_tol) *
                 " weight_tol(ref)=" * string(weight_tol))
    push!(lines, "# layers=" * join(layers, ",") * " out_dir=" * out_dir)
    println(lines[1]); println(lines[2])
    println("[deep-layers] prep=" * prep_path * " n_assets=" * string(src.n_assets) *
            " N_R=" * string(length(d.locked)) * " locked_sum=" * string(sum(d.locked)))

    X_prev = nothing
    res_prev = nothing
    res_base = nothing
    M_prev = 0
    any_fail = false
    for (li, M) in enumerate(layers)
        # ---- X：基准层优先加载既有 X2 产物（契约校验），否则 gen ----
        X = nothing
        origin = "gen"
        gen_s = 0.0
        if li == 1 && load_x2
            X_loaded, note = _try_load_x2(out_dir, t, M, d.seed, use_mu, use_chisq)
            if X_loaded !== nothing
                X = X_loaded
                origin = "load:X2"
            else
                note_line = "# note: X2 load skipped ($note); regenerating"
                push!(lines, note_line); println(note_line)
            end
        end
        if X === nothing
            t0 = time_ns()
            X = try
                src.gen(M, SobolOwenRule(src.rqmc_dim, d.seed),
                        MersenneTwister(m_seed))
            catch err
                _fail(20, "gen(M=$M) failed: " * sprint(showerror, err))
            end
            gen_s = (time_ns() - t0) / 1e9
        end

        # ---- 嵌套前缀检查（层对） ----
        nest = NaN
        if X_prev !== nothing
            nest = _nest_maxdiff(X, X_prev)
        end

        # ---- 求解 + 证书 ----
        res = _solve_layer_from_X(d, X, M, kelly_tol; assembly_check = (li == 1))
        res.converged || (any_fail = true)
        if li == 1 && isfinite(res.asm_maxdiff) && res.asm_maxdiff > 0.0
            _fail(43, "assembly/core mismatch at M=$M: maxdiff=$(res.asm_maxdiff)")
        end

        layer_line = @sprintf("DEEP_LAYER t=%d M=%d origin=%s gen_s=%.3f solve_s=%.3f iters=%d ok=%s reason=%s",
                              t, M, origin, gen_s, res.solve_s, res.iters,
                              res.converged, res.reason)
        if res.cert !== nothing
            layer_line *= @sprintf(" feas=%.3e kkt=%.3e gap=%.3e obj=%.10e",
                                   res.cert.feasibility, res.cert.kkt_residual,
                                   res.cert.objective_gap, res.cert.objective)
        end
        push!(lines, layer_line)
        println(layer_line)
        if li == 1 && isfinite(res.asm_maxdiff)
            asm_line = @sprintf("# assembly(:ipm) vs core maxdiff = %.3e", res.asm_maxdiff)
            push!(lines, asm_line); println(asm_line)
        end

        # ---- 层对统计（当前层 relative 上一层） ----
        if li >= 2
            A = norm(res.w_aug .- res_prev.w_aug, 1)
            Bopt = _B_opt(X, res.base, res.free_pos, res.w_aug, res_prev.w_aug)
            maxdiff = maximum(abs.(res.w_aug .- res_base.w_aug))
            pair_line = @sprintf("DEEP_PAIR t=%d M=%d->%d A=%.10e A_pass_ref=%s B_opt=%.3e maxdiff_vs_base=%.3e nest_maxdiff=%.3e",
                                 t, M_prev, M, A,
                                 (A <= weight_tol ? "true" : "false"),
                                 Bopt, maxdiff, nest)
            push!(lines, pair_line)
            println(pair_line)
        end

        X_prev = X
        res_prev = res
        M_prev = M
        li == 1 && (res_base = res)
    end

    out_path = joinpath(out_dir, "deep_layers_t$(t).txt")
    try
        _atomic_write_text(out_path, join(lines, "\n") * "\n")
    catch err
        _fail(42, "write failed: " * sprint(showerror, err))
    end
    println("[deep-layers] wrote " * out_path)
    exit(any_fail ? 41 : 0)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
