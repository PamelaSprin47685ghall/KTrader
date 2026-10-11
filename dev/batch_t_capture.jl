#!/usr/bin/env julia
# =============================================================================
# dev/batch_t_capture.jl —— 批量两段式调度 · 段 A（prep 捕获）
# =============================================================================
# 背景：过门配置完整单日链 85-95s（> 单命令有效窗口 ~48s）；分段：prep 28.6s +
# kelly 40.8s（已验证收敛：A=6.26e-5、三项证书全过，证据 CQ_*）。本脚本在
# kelly 入口前捕获全部所需对象、原子落盘并正常退出，供段 B 离线求解。
#
# 用法：julia --startup-file=no --project=. dev/batch_t_capture.jl <t> [out_dir]
# 环境：GATE0_DATA_DIR（默认 <repo>/data）、GATE0_SEED（十六进制串，默认 driver 默认）。
#
# 退出码（独立可辨，不得合并）：0 成功；2 参数错误；10 prep 构造失败；11 落盘失败。
# 产物：<out_dir>/batch_t<T>_prep.bin（临时文件 + 原子重命名，要么完整要么不存在）。
#
# 模式来源：与 driver.jl 的 single_day_decision 步骤 0-11 同式（复刻；driver 变更
# 须同步本脚本）。held 首版固定 zeros（跨日持仓推进登记为后续）。
# =============================================================================

using Serialization, Dates, Statistics, LinearAlgebra, Random

module Gate0BatchCaptureA
    using Dates
    include(joinpath(@__DIR__, "..", "src", "gate0", "market.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "geometry.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "modes.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "response.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "posterior.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "oof.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "innovation.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "predictive.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "kelly.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "quadrature.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "driver.jl"))
end
using .Gate0BatchCaptureA
import KTrader

const _PREP_VERSION = "batch-prep-v1"

_fail(code::Int, msg::String) = (println(stderr, "[batch-capture] " * msg); exit(code))

function _atomic_serialize(path::String, payload)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        serialize(io, payload)
    end
    mv(tmp, path; force = true)
end

function _build_prep(t::Int, seed::UInt64, data_dir::String)
    b = KTrader.load_bars(data_dir)
    mf = from_bars_arrays(b.dates, b.symbols, b.close, b.adj, b.bar)
    el = Eligibility(mf)
    held = zeros(size(mf.observed, 2))
    posterior_tol = 1e-6; posterior_max_cells = 2048; u_span = 5.0
    # --- 步骤 0：因果截断（构造性） ---
    mf_t = MarketFacts(mf.dates[1:t], mf.symbols, mf.close[1:t, :],
                       mf.adj[1:t, :], mf.observed[1:t, :])
    el_t = Eligibility(el.model_admitted[1:t, :], el.trade_eligible[1:t, :],
                       el.executable[1:t, :])
    # --- 步骤 1：active 域 ---
    act = findall(el_t.model_admitted[t, :])
    N_act = length(act)
    N_act == 0 && throw(ArgumentError("no active assets at t=$t"))
    # --- 步骤 2：signal -> log/return ---
    signal = signal_prices(mf_t)[:, act]
    x_log = log.(signal)
    f_first = [findfirst(view(el_t.executable, 1:t, j)) for j in act]
    all(i -> i !== nothing, f_first) || throw(ArgumentError("active asset without observation"))
    s1_act = vec(ruler(x_log, Int[i for i in f_first])[:, 1])
    # --- 步骤 3：mode 坐标 ---
    r = Matrix{Float64}(undef, t, N_act)
    r[1, :] .= NaN
    r[2:t, :] .= x_log[2:t, :] .- x_log[1:t-1, :]
    obs_ret = isfinite.(r)
    E_active = domain_mode_basis(N_act)
    mp = build_mode_problem(r, Matrix{Bool}(obs_ret), s1_act, E_active; dc = true)
    # --- 步骤 4：训练行 ---
    t - 1 >= WARMUP || throw(ArgumentError("t=$t below WARMUP"))
    rows = Int[s for s in WARMUP:(t-1) if any(view(obs_ret, s + 1, :))]
    length(rows) >= 2 || throw(ArgumentError("insufficient training rows"))
    row_ids = Int[s + 1 for s in rows]
    row_masks = Vector{BitVector}([BitVector(view(obs_ret, u, :)) for u in row_ids])
    # --- 步骤 5：free/locked ---
    free_univ = free(el_t, t)
    locked_univ = locked(held .> 0, el_t, t)
    free_act = Int[i for (i, j) in enumerate(act) if free_univ[j]]
    locked_act = Int[i for (i, j) in enumerate(act) if locked_univ[j]]
    # --- 步骤 6：R 域判定 ---
    R_act, dropped_free_act = resolve_risk_domain(row_ids, row_masks, free_act, locked_act, t)
    N_R = length(R_act)
    # --- 步骤 7：full posterior ---
    X_tr = mp.X[rows, :]
    Y_tr = mp.Y[rows, :]
    post = fit_full_posterior(X_tr, Y_tr; tol = posterior_tol,
                              max_cells = posterior_max_cells)
    # --- 步骤 8：prequential 残差（strictly causal） ---
    resid_all = Gate0BatchCaptureA.prequential_residual_rows(X_tr, Y_tr, E_active;
                                          tol = posterior_tol,
                                          max_cells = posterior_max_cells,
                                          u_span = u_span)
    keep_res = Int[i for i in 1:size(resid_all, 1) if all(isfinite, resid_all[i, :])]
    isempty(keep_res) && throw(ArgumentError("no definable prequential residual row"))
    resid = resid_all[keep_res, :]
    row_ids_ok = row_ids[keep_res]
    row_masks_ok = row_masks[keep_res]
    # --- 步骤 8b：二次 resolve（P0-3 满秩 gate） ---
    free_after_first = setdiff(free_act, dropped_free_act)
    R_act2, dropped2 = resolve_risk_domain(row_ids_ok, row_masks_ok,
                                           free_after_first, locked_act, t;
                                           residual_rows = resid)
    R_act = R_act2
    N_R = length(R_act)
    # --- 步骤 9：innovation state ---
    st = innovation_state(resid, row_ids_ok, row_masks_ok, R_act, E_active;
                          t = t, require_full_rank = true)
    # --- 步骤 10：决策行 feature ---
    x_now = vec(mp.X_full[t, :])
    # --- 步骤 11：locked 权重（held=zeros 首版） ---
    locked_set = Set(locked_act)
    locked_w = zeros(N_R)
    for (k, j) in enumerate(R_act)
        if j in locked_set
            locked_w[k] = held[act[j]]
        end
    end
    budget = 1.0 - sum(locked_w)
    (; version = _PREP_VERSION, t = t, seed = seed, held_mode = :zeros,
       data_note = data_dir, rule_kwargs = (; posterior_tol, posterior_max_cells, u_span),
       post = post, st = st, xt = x_now, s1 = s1_act, E_active = E_active,
       locked = locked_w, budget = budget)
end

function main()
    length(ARGS) >= 1 || _fail(2, "usage: batch_t_capture.jl <t> [out_dir]")
    t = try
        parse(Int, ARGS[1])
    catch
        _fail(2, "cannot parse t: " * ARGS[1])
    end
    out_dir = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "dev", "batch_out")
    mkpath(out_dir)
    data_dir = get(ENV, "GATE0_DATA_DIR", joinpath(@__DIR__, "..", "data"))
    seed = let e = get(ENV, "GATE0_SEED", "")
        if isempty(e)
            Gate0BatchCaptureA._DRIVER_DEFAULT_SEED
        else
            s = startswith(lowercase(e), "0x") ? e[3:end] : e
            parse(UInt64, s; base = 16)
        end
    end
    payload = try
        _build_prep(t, seed, data_dir)
    catch err
        _fail(10, "prep build failed at t=$t: " * sprint(showerror, err))
    end
    path = joinpath(out_dir, "batch_t" * string(t) * "_prep.bin")
    try
        _atomic_serialize(path, payload)
    catch err
        _fail(11, "write failed: " * sprint(showerror, err))
    end
    println("[batch-capture] ok t=$t seed=0x" * string(seed, base = 16) *
            " -> " * path)
    exit(0)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end