#!/usr/bin/env julia
# =============================================================================
# dev/tol_trace.jl —— D-066 数值容差定稿实验（F2）执行工具：三段式 gen/solve/join
# =============================================================================
# 规范来源：docs/NUMERICAL_TOLERANCE_FINALIZATION.md（预注册设计）§3.6（分段执行与
# 新工具规格）、§8（受控执行分段总表/规格要求）、§7（统一定义与证据规范）、
# §2（预注册纪律）。本工具只复刻生产调用链与离线拼装统计：不引入新数学对象、
# 不改任何生产容差、不缓存跨调用、不消耗调用方 rng（全部随机性由显式 seed
# 确定性派生并每次重建局部流）。
#
# 三段式（对齐设计 §3.6/§8；CLI 以位置参数承载 world/M——语义逐项一致）：
#   gen   <world> <M> [--seed HEX] [--audit] [--out DIR]   # 生成 X_M 并原子落盘
#   solve <world> <M> [--seed HEX] [--out DIR]             # 读 X_M → cash_kelly → seg_M
#   join  <world> <M> <M2> [--seed HEX] [--out DIR]        # 读 seg_M/seg_2M（X_2M 独立
#                                                          #   产物优先、seg 内嵌回退）
#                                                          #   + 现场 audit → A/B_opt/
#                                                          #   B_audit → CSV 追加一行
# 附加（设计 §3.6 的「小层可合并」与解析基准入口）：
#   run   <world> <Mmin> <Mmax> [--seed HEX] [--out DIR]   # 串行 gen+solve+join（同函数、
#                                                          #   同落盘、同 CSV；Mmin/Mmax 为
#                                                          #   2 的幂，层对为相邻两层）
#   ref   <world> [--out DIR]                              # 解析基准（GH+KKT / 解析锚），
#                                                          #   自洽检查入口；落盘 tolref_*.bin
#   --selftest                                             # 结构锚（不调 kelly/GH 求根）
#
# world 清单（设计 §3.1；参数为预注册值）：
#   A1  known Gaussian 2-asset   μ=[0.0015,0.0008] σ=[0.08,0.06]（复用
#       quadrature_tests.jl 的 MU_FIX/SIG_FIX）；inverse-CDF 2D source
#   A2  同 A1、μ 减半 [0.00075,0.0004]
#   A3  known Gaussian 3-asset   μ=[0.0012,0.0009,0.0006] σ=[0.09,0.07,0.05]
#   A4  D-090(a) 三行置换世界（1.25/0.9）；解析角点 w*=(1/3,1/3,1/3)
#   A5a D-090(b) 三行置换世界（1.2/0.9）——平局射线（设计 §3.1 的 A5 之一）
#   A5b D-091 六行置换世界（1.06/0.94/1.0）——无信号（设计 §3.1 的 A5 之二）
#   real:<t>  真实 fixture：读 <prep-dir>/batch_t<t>_prep.bin（batch_t_capture.jl
#             产物）与 batch_out 的 X/seg 产物复用（设计 §4.1/§4.6）
# 说明（诚实登记）：设计 §3.1 的 A5 引用两个矩阵（kelly_cash_tests.jl 的 Xb 与
# 六行 X）；一个 world 标签对应一个 gen 矩阵，故拆为 A5a/A5b——二者合并即 A5 的
# 引用范围。
#
# A1–A3 的结构边界（诚实登记）：设计 §3.2 要求 inverse-CDF source；A1/A2 的 2D
# 直接由 rule 点驱动（与 quadrature_tests.jl 的 gaussian_rqmc_source 同模式）。
# A3 需要第三个独立随机通道，而 src 的 SobolOwenRule 上限 dim=2
# （src/gate0/quadrature.jl）；本工具**不复用也不猜测** Joe-Kuo 第三维方向数
# （凭记忆内联违反证据纪律），改为主 rule seed 确定性派生的 1 维子 rule
# （常数 _A3_DIM3_SEED_XOR，任意确定、无优化含义）驱动第三资产。性质：嵌套保持
# （两子 rule 各自从同一 seed 重建）、确定、可重放；第三维随机化与资产 1/2 独立
# （不同 scramble）。若 SPEC 要求严格单 rule 3 维，须先扩展 src 的 _SOBOL_DIRV
# （src 变更，超出本工具范围）。A1–A3 均无 μ epistemic 通道（mu_rng 被忽略）——
# 与真实链的结构差异（真实链有 matrix-t μ 通道），边界由 B 组真实轨迹承担，
# 不得由合成外推（设计 §10.3-4）。
#
# 逐点等价性论证（与 adaptive_scenario_kelly / batch_t_solve3.jl 同式复刻）：
# 1) rule 派生：opt_rule=SobolOwenRule(src.rqmc_dim, rule_seed)；a_seed=
#    audit_seed(rule_seed)；m_seed=_default_mu_seed(rule_seed)；am_seed=m_seed ⊻
#    0x9E3779B97F4A7C15（src/gate0/quadrature.jl 的四条派生；audit 的 rule/流
#    分别用 a_seed/am_seed）。
# 2) gen：X=src.gen(M, opt_rule, MersenneTwister(m_seed))（与 batch_t_solve3.jl 的
#    _solve_layer/_run_gen 同式；real 的 rule_seed=prep.seed）。
# 3) solve：free 列 Kelly + base_locked（方案 2 表示）：base=locked_wealth_gate0(X,
#    locked)；X_free=X[:,free_pos]；cash_kelly(X_free; base, budget,
#    tol=_KELLY_TOL)；w_full 填回 locked。**禁止** X_full·w_full + base 双重计入
#    （P0-5；quadrature.jl 的 solve_layer 同式）。
# 4) join：A=‖vcat(w_2M,wc_2M)−vcat(w_M,wc_M)‖₁（含 cash 增广）；B_opt 在 X_2M 上
#    求 fine 解 minus coarse 解的目标差；B_audit 在 audit 矩阵（现场 gen 或预生成
#    audit 产物）上求绝对差——与 quadrature.jl:777/798/819 与 batch b3 逐点同式。
# 5) 开关：real 的 gen 用 GATE0_BATCH_MU_QMC/GATE0_BATCH_CHISQ_QMC（默认 true，
#    与 batch 同 env 同默认；P-batch 口径）；solve/join 沿用 X/seg 产物记录的开关。
#
# 产物与命名（设计 §7.2）：
#   X（opt）：<out>/tolX_<tag>_<seedhex>_M<M>.bin
#   X（audit，gen --audit 预生成；join 优先读取、否则现场 gen）：
#             <out>/tolX_<tag>_<seedhex>_M<M>_audit.bin
#   seg：      <out>/tolseg_<tag>_<seedhex>_M<M>.bin（含 X 内嵌，与 batch seg 同构）
#   ref：      <out>/tolref_<tag>.bin
#   CSV：      <out>/tolA_<tag>_<seedhex>.csv（合成）；<out>/tolB_t<t>_layers.csv
#              （real；P-default 子集加 _pdefault 后缀）
# CSV 列（设计 §3.3/§3.5；任务书列单 + 设计 §3.3 的 objective(M2)）：
#   M,M2,A,Bopt,Baudit,w1..wN,wc,dist_star,secs,objective_M2
#   - dist_star：|vcat(w_2M,wc_2M) − vcat(w*,wc*)|₁（A1–A4）；A5a/A5b/real 留空
#     （无唯一 w*——A5 为平局集，real 无解析锚）。
#   - secs：seg_M.solve + seg_2M.solve + X_2M.gen + join 段自身（含 audit 获取）
#     的耗时之和；缺失分量（batch 历史产物无该字段）记空（不伪造 0）。
#   - batch 复用的 X1/X2/seg1/seg2 直接可读（M/t/seed 字段匹配时）；M 不匹配的
#     候选跳过、同 M 候选契约错 fail（不静默换用）。
# 禁止字段（设计 §7.3 重申）：任何收益派生列（pnl/sharpe/return/net_value/
# drawdown 等）全产物禁止——本工具字段表不含、也不得添加任何此类列；全部判据
# 与回测收益无关（SPEC §95 / 开发守则 §24）。
#
# 原子落盘：全部 .bin 与 CSV 走 tmp + rename（要么完整要么不存在）；CSV 追加为
# 「读旧 + 拼接 + 原子替换」，header 逐字校验。契约校验：读取端显式校验
# 字段/版本/world/seed/M/开关，不匹配即 rc=20（不静默接受）。
#
# 环境变量：GATE0_BATCH_MU_QMC / GATE0_BATCH_CHISQ_QMC（默认 "true"，real 的 gen
#   开关；与 batch_t_solve3.jl 同名同默认）。
# 目录：--out 默认 archive/evidence/gate0_tol_finalize_<yyyymmdd>（设计 §7.2）；
#   --prep-dir 默认 archive/evidence/gate0_multiday_run_20261010/batch_out
#   （设计 §4.1；可用 flag 覆盖）。
#
# 退出码（独立可辨，不得合并）：
#   0  成功；2  参数/用法错误；3  --selftest 失败
#   10 gen：source 构造或场景生成失败；11 gen：X 落盘失败
#   20 读取/契约校验失败（prep/X/seg/ref/CSV header）
#   21 solve：cash_kelly 失败（含 D-067 fail-loud）；22 solve：seg 落盘失败；
#   23 solve：归一性检查失败
#   30 join：audit 获取/生成失败；31 join：A/B 统计求值失败；
#   32 join：归一性检查失败；33 join：CSV 追加落盘失败；34 join：ref 读取/计算失败
#   40 ref：基准计算/自洽检查失败；41 ref：落盘失败
#
# 执行批次分段（预注册草案；本工具交付时不运行；每段 ≤60s/RSS2048，由
# scoped 封装承担 log 尾行）：
#   A 组（合成）：A1–A3 × 4 预注册 seed（0xC0FFEE/0xBEEF/0xAAAA/0xBBBB）——
#     小层 {64..4096} 用 run 合并（如 run A1 64 4096）、8192..65536 逐层
#     gen/solve + 逐对 join；A4/A5a/A5b 结构检查各 1-2 段（run A4 64 256 等）。
#   B 组（真实，P-batch 主口径）：核心 6 日（330/331/335/336/344/346）——
#     小层 run real:<t> 64 4096；32768/65536 逐层 gen+solve；层对 join 复用
#     batch 的 seg1/seg2/X1/X2（65536→131072 直接 join）；P-default 对照
#     （330/336）以 GATE0_BATCH_MU_QMC=false 重跑全部层。
#   C/D 组：离线为主（本工具只提供行数据；M_req/假收敛率/包络比等分析离线）。
#   首行前置：pre_snapshot（设计 §2.4；由执行批次记录 SHA256 清单，工具不代劳）。
# =============================================================================

using Serialization, Random, LinearAlgebra, Statistics, Dates
using Distributions: quantile, Normal

# 模块名必须与 batch_t_capture.jl / batch_t_solve3.jl 完全一致：serialize 记录
# 类型的模块路径（Main.Gate0BatchCaptureA.*），反序列化按名查找——不同名则
# UndefVarError（batch 产物：prep 含 post/st 等 src 类型）。
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
using .Gate0BatchCaptureA: SobolOwenRule, rule_points, audit_seed,
    predictive_rqmc_source, locked_wealth_gate0, cash_kelly

include(joinpath(@__DIR__, "tol_ref.jl"))

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

const _TOL_VERSION = "tol-trace-v1"
const _KELLY_TOL = 1e-8                      # 与 batch_t_solve3.jl / 生产同值
const _AM_AUDIT_XOR = 0x9E3779B97F4A7C15     # audit 流 seed 派生（与 src 同式）
const _A3_DIM3_SEED_XOR = 0xBF58476D1CE4E5B9 # A3 第三维子 rule 派生（任意确定常数，
                                             # 无优化含义；避开 src 既有派生常数）

# world 表（设计 §3.1 预注册值）
const _A1_MU = [0.0015, 0.0008]
const _A1_SIG = [0.08, 0.06]
const _A2_MU = [0.00075, 0.0004]
const _A3_MU = [0.0012, 0.0009, 0.0006]
const _A3_SIG = [0.09, 0.07, 0.05]
const _A4_X = [1.25 0.9 0.9;
               0.9 1.25 0.9;
               0.9 0.9 1.25]
const _A5A_X = [1.2 0.9 0.9;
                0.9 1.2 0.9;
                0.9 0.9 1.2]
const _A5B_X = [1.06 0.94 1.0;
                1.06 1.0 0.94;
                0.94 1.06 1.0;
                0.94 1.0 1.06;
                1.0 1.06 0.94;
                1.0 0.94 1.06]

const _USAGE = "usage: julia --startup-file=no --project=. dev/tol_trace.jl " *
               "gen|solve|join|run|ref ... | --selftest（详见文件头注释）"

# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

_fail(code::Int, msg::String) = (println(stderr, "[tol-trace] " * msg); exit(code))

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

_seedhex(seed::UInt64) = lpad(string(seed, base = 16), 16, '0')

function _parse_M_soft(s::String)::Union{Int,Nothing}
    M = tryparse(Int, s)
    (M === nothing || M < 1) && return nothing
    M
end

_parse_M_or_fail(s::String) = let M = _parse_M_soft(s)
    M === nothing ? _fail(2, "cannot parse M（正整数）: " * s) : M
end

function _parse_seed_soft(s::String)::Union{UInt64,Nothing}
    t = s
    if length(s) >= 2 && lowercase(s[1:2]) == "0x"
        t = length(s) == 2 ? "" : s[3:end]
    end
    isempty(t) && return nothing
    tryparse(UInt64, t; base = 16)
end

_parse_seed_or_fail(s::String) = let r = _parse_seed_soft(s)
    r === nothing ? _fail(2, "cannot parse seed（十六进制）: " * s) : r
end

_is_pow2(x::Int) = x >= 1 && (x & (x - 1)) == 0

function _default_out_dir()
    joinpath("archive", "evidence",
             "gate0_tol_finalize_" * Dates.format(Dates.today(), "yyyymmdd"))
end

_default_prep_dir() =
    joinpath("archive", "evidence", "gate0_multiday_run_20261010", "batch_out")

_out_dir(parsed) = get(parsed.opts, "--out", _default_out_dir())
_prep_dir(parsed) = get(parsed.opts, "--prep-dir", _default_prep_dir())

# ---------------------------------------------------------------------------
# world 解析与 spec
# ---------------------------------------------------------------------------

function _world_spec(tag::String)::Union{Nothing,NamedTuple}
    tag == "A1" && return (; kind = :gauss, n = 2, dim = 2, mu = _A1_MU,
                           sigma = _A1_SIG, rows = nothing)
    tag == "A2" && return (; kind = :gauss, n = 2, dim = 2, mu = _A2_MU,
                           sigma = _A1_SIG, rows = nothing)
    tag == "A3" && return (; kind = :gauss, n = 3, dim = 2, mu = _A3_MU,
                           sigma = _A3_SIG, rows = nothing)
    tag == "A4" && return (; kind = :rows, n = 3, dim = 1, mu = nothing,
                           sigma = nothing, rows = _A4_X)
    tag == "A5a" && return (; kind = :rows, n = 3, dim = 1, mu = nothing,
                            sigma = nothing, rows = _A5A_X)
    tag == "A5b" && return (; kind = :rows, n = 3, dim = 1, mu = nothing,
                            sigma = nothing, rows = _A5B_X)
    nothing
end

_world_spec_or_fail(tag::String) = let spec = _world_spec(tag)
    spec === nothing &&
        _fail(2, "未知 world: " * tag * "（支持 A1/A2/A3/A4/A5a/A5b/real:<t>）")
    spec
end

"""real 的开关：GATE0_BATCH_MU_QMC/GATE0_BATCH_CHISQ_QMC（默认 true，与 batch 同）；
合成世界无 μ 通道（不适用；产物标记 false）。"""
function _switch_pair(ctx)::Tuple{Bool,Bool}
    if ctx.kind == :real
        (get(ENV, "GATE0_BATCH_MU_QMC", "true") == "true",
         get(ENV, "GATE0_BATCH_CHISQ_QMC", "true") == "true")
    else
        (false, false)
    end
end

_locked_for(ctx) = ctx.kind == :real ? collect(Float64, ctx.prep.locked) :
                                       zeros(ctx.n_assets)

function _resolve_world_or_fail(world_arg::String, seed_opt::Union{Nothing,String},
                                prep_dir::String)
    if startswith(world_arg, "real:")
        t = tryparse(Int, world_arg[6:end])
        (t !== nothing && t >= 1) ||
            _fail(2, "real 标签格式须为 real:<t>（t ≥ 1；got " * world_arg * "）")
        prep_path = joinpath(prep_dir, "batch_t" * string(t) * "_prep.bin")
        d = _load_prep_or_fail(prep_path)
        d.t == t || _fail(20, "prep.t=$(d.t) 与标签 t=$t 不一致: " * prep_path)
        seed = if seed_opt === nothing
            d.seed
        else
            s = _parse_seed_or_fail(seed_opt)
            s == d.seed ||
                _fail(20, "--seed (0x" * string(s, base = 16) *
                           ") 与 prep.seed (0x" * string(d.seed, base = 16) *
                           ") 不一致（real 的 rule seed 以 prep 为唯一真源）")
            s
        end
        (; kind = :real, tag = "t" * string(t), t = t, seed = seed, prep = d,
           prep_path = prep_path, n_assets = length(d.locked), spec = nothing)
    else
        spec = _world_spec_or_fail(world_arg)
        seed_opt !== nothing ||
            _fail(2, "合成世界需要 --seed（预注册 seed 清单，见设计 §2.3）")
        seed = _parse_seed_or_fail(seed_opt)
        (; kind = spec.kind, tag = world_arg, t = -1, seed = seed, prep = nothing,
           prep_path = "", n_assets = spec.n, spec = spec)
    end
end

# ---------------------------------------------------------------------------
# prep 加载（real；字段表与 batch_t_solve3.jl 同源）
# ---------------------------------------------------------------------------

const _PREP_VERSION = "batch-prep-v1"
const _PREP_FIELDS = (:version, :t, :seed, :held_mode, :data_note,
                      :rule_kwargs, :post, :st, :xt, :s1, :E_active,
                      :locked, :budget)

function _prep_error(d)::Union{Nothing,String}
    (d isa NamedTuple) || return "prep is not a NamedTuple"
    for f in _PREP_FIELDS
        hasproperty(d, f) || return "prep missing field: " * string(f)
    end
    d.version == _PREP_VERSION || return "prep version mismatch"
    d.held_mode == :zeros ||
        return "prep held_mode=$(d.held_mode)（本工具按 zeros 语义消费）"
    nothing
end

function _load_prep_or_fail(path::String)
    isfile(path) || _fail(20, "prep 不存在: " * path)
    d = try
        deserialize(path)
    catch err
        _fail(20, "prep 反序列化失败: " * path * ": " * sprint(showerror, err))
    end
    msg = try
        _prep_error(d)
    catch err
        "prep 契约检查抛错: " * sprint(showerror, err)
    end
    msg === nothing || _fail(20, "prep 契约失败: " * path * ": " * msg)
    d
end

# ---------------------------------------------------------------------------
# scenario source（合成 / real）
# ---------------------------------------------------------------------------

"""A1/A2（2D）：r_j = μ_j + σ_j·Φ^{-1}(pts[s,j])——与 quadrature_tests.jl 的
gaussian_rqmc_source 同模式（inverse-CDF；mu_rng 忽略）。"""
function _make_gauss_gen(mu::Vector{Float64}, sigma::Vector{Float64})
    N = length(mu)
    N == length(sigma) ||
        throw(DimensionMismatch("_make_gauss_gen: μ/σ 长度不等"))
    if N == 2
        return function (M::Int, rule::SobolOwenRule, mu_rng::AbstractRNG)
            pts = rule_points(rule, M)          # M×2（rqmc_dim=2）
            gross = Matrix{Float64}(undef, M, 2)
            @inbounds for s in 1:M, j in 1:2
                gross[s, j] = exp(mu[j] + sigma[j] * quantile(Normal(), pts[s, j]))
            end
            gross
        end
    elseif N == 3
        # 第三维：主 rule seed 派生的 1 维子 rule（见文件头 A3 边界登记）
        return function (M::Int, rule::SobolOwenRule, mu_rng::AbstractRNG)
            pts = rule_points(rule, M)          # M×2（资产 1、2）
            sub = SobolOwenRule(1, rule.seed ⊻ _A3_DIM3_SEED_XOR)
            pts3 = rule_points(sub, M)          # M×1（资产 3）
            gross = Matrix{Float64}(undef, M, 3)
            @inbounds for s in 1:M
                gross[s, 1] = exp(mu[1] + sigma[1] * quantile(Normal(), pts[s, 1]))
                gross[s, 2] = exp(mu[2] + sigma[2] * quantile(Normal(), pts[s, 2]))
                gross[s, 3] = exp(mu[3] + sigma[3] * quantile(Normal(), pts3[s, 1]))
            end
            gross
        end
    end
    throw(ArgumentError("_make_gauss_gen: 支持 N=2,3（got " * string(N) * "）"))
end

"""A4/A5：从 1 维 rule 点做行选择（idx = ceil(u·R)，clamp 到 1..R——u=0 的
测度 2^{-32} 归入首行；设计 §3.2 的行世界驱动）。"""
function _make_rows_gen(rows::Matrix{Float64})
    R = size(rows, 1)
    R >= 1 || throw(ArgumentError("_make_rows_gen: 行数 ≥ 1"))
    function gen(M::Int, rule::SobolOwenRule, mu_rng::AbstractRNG)
        pts = rule_points(rule, M)              # M×1（rqmc_dim=1）
        gross = Matrix{Float64}(undef, M, size(rows, 2))
        @inbounds for s in 1:M
            idx = min(max(ceil(Int, pts[s, 1] * R), 1), R)
            gross[s, :] .= view(rows, idx, :)
        end
        gross
    end
    gen
end

function _synthetic_source(spec)
    if spec.kind == :gauss
        return RQMCScenarioSource(_make_gauss_gen(spec.mu, spec.sigma),
                                  spec.n, spec.dim)
    elseif spec.kind == :rows
        return RQMCScenarioSource(_make_rows_gen(spec.rows),
                                  size(spec.rows, 2), 1)
    end
    error("_synthetic_source: 未知 spec.kind=" * string(spec.kind))
end

function _source_for(ctx; use_mu::Bool, use_chisq::Bool)
    if ctx.kind == :real
        predictive_rqmc_source(ctx.prep.post, ctx.prep.st, ctx.prep.xt;
                               s1 = ctx.prep.s1, E_active = ctx.prep.E_active,
                               mu_qmc = use_mu, mu_chisq_qmc = use_chisq)
    else
        _synthetic_source(ctx.spec)
    end
end

# ---------------------------------------------------------------------------
# 产物命名与读取
# ---------------------------------------------------------------------------

_x_name(ctx, M::Int, role::Symbol) =
    "tolX_" * ctx.tag * "_" * _seedhex(ctx.seed) * "_M" * string(M) *
    (role === :audit ? "_audit" : "") * ".bin"
_seg_name(ctx, M::Int) =
    "tolseg_" * ctx.tag * "_" * _seedhex(ctx.seed) * "_M" * string(M) * ".bin"

function _x_candidates(ctx, M::Int, out_dir::String, prep_dir::String)
    paths = String[joinpath(out_dir, _x_name(ctx, M, :opt))]
    if ctx.kind == :real
        push!(paths, joinpath(prep_dir, "batch_t" * string(ctx.t) * "_X1.bin"))
        push!(paths, joinpath(prep_dir, "batch_t" * string(ctx.t) * "_X2.bin"))
    end
    paths
end

function _seg_candidates(ctx, M::Int, out_dir::String, prep_dir::String)
    paths = String[joinpath(out_dir, _seg_name(ctx, M))]
    if ctx.kind == :real
        push!(paths, joinpath(prep_dir, "batch_t" * string(ctx.t) * "_seg1.bin"))
        push!(paths, joinpath(prep_dir, "batch_t" * string(ctx.t) * "_seg2.bin"))
    end
    paths
end

function _x_payload_error(p, ctx, M::Int)::Union{Nothing,String}
    (p isa NamedTuple) || return "not a NamedTuple"
    hasproperty(p, :kind) || return "missing kind"
    p.kind == :X || return "kind mismatch"
    for f in (:version, :M, :seed, :X)
        hasproperty(p, f) || return "missing field: " * string(f)
    end
    p.M == M || return "M mismatch"
    p.seed == ctx.seed || return "seed mismatch"
    if p.version == _TOL_VERSION
        for f in (:world, :role, :rqmc_dim, :n_assets, :mu_qmc, :mu_chisq_qmc, :secs)
            hasproperty(p, f) || return "missing field: " * string(f)
        end
        p.world == ctx.tag || return "world mismatch: $(p.world) ≠ $(ctx.tag)"
        (p.role === :opt || p.role === :audit) ||
            return "role 非法（须 :opt/:audit）: $(p.role)"
        p.n_assets == ctx.n_assets ||
            return "n_assets mismatch: $(p.n_assets) ≠ $(ctx.n_assets)"
        size(p.X) == (M, ctx.n_assets) ||
            return "X size $(size(p.X)) ≠ ($M, $(ctx.n_assets))"
    elseif p.version == "batch-solve3-v1"
        for f in (:seg, :t, :mu_qmc, :mu_chisq_qmc)
            hasproperty(p, f) || return "missing field: " * string(f)
        end
        p.t == ctx.t || return "t mismatch: $(p.t) ≠ $(ctx.t)"
        size(p.X) == (M, ctx.n_assets) ||
            return "X size $(size(p.X)) ≠ ($M, $(ctx.n_assets))"
    else
        return "unknown version: " * string(p.version)
    end
    all(x -> isfinite(x) && x > 0, p.X) || return "X 含非正/非有限 cell"
    nothing
end

"""读取器的角色匹配：tol 格式要求 role 精确等于 want_role；batch 格式无 role
字段（单一角色，仅 opt 路径消费）。返回 nothing 或错误串。"""
function _x_role_error(p, want_role::Symbol)::Union{Nothing,String}
    hasproperty(p, :version) || return "missing version"
    p.version == _TOL_VERSION || return nothing
    hasproperty(p, :role) || return "missing role"
    p.role == want_role || return "role mismatch（want $want_role, got $(p.role)）"
    nothing
end

function _seg_payload_error(p, ctx, M::Int)::Union{Nothing,String}
    (p isa NamedTuple) || return "not a NamedTuple"
    hasproperty(p, :version) || return "missing version"
    for f in (:M, :w_full, :w_cash, :cert)
        hasproperty(p, f) || return "missing field: " * string(f)
    end
    p.M == M || return "M mismatch"
    if p.version == _TOL_VERSION
        hasproperty(p, :kind) || return "missing kind"
        p.kind == :seg || return "kind mismatch"
        for f in (:world, :seed, :rqmc_dim, :n_assets, :mu_qmc, :mu_chisq_qmc, :secs)
            hasproperty(p, f) || return "missing field: " * string(f)
        end
        p.world == ctx.tag || return "world mismatch"
        p.seed == ctx.seed || return "seed mismatch"
        p.n_assets == ctx.n_assets || return "n_assets mismatch"
        hasproperty(p, :X) || return "missing X"
        size(p.X) == (M, ctx.n_assets) ||
            return "X size $(size(p.X)) ≠ ($M, $(ctx.n_assets))"
    elseif p.version == "batch-solve3-v1"
        for f in (:seg, :t, :seed, :mu_qmc, :mu_chisq_qmc)
            hasproperty(p, f) || return "missing field: " * string(f)
        end
        p.t == ctx.t || return "t mismatch"
        p.seed == ctx.seed || return "seed mismatch"
    else
        return "unknown version: " * string(p.version)
    end
    length(p.w_full) == ctx.n_assets || return "w_full length"
    all(isfinite, p.w_full) || return "w_full 非有限"
    isfinite(p.w_cash) || return "w_cash 非有限"
    nothing
end

function _adapt_seg(p, path::String)
    (; M = p.M, w_full = p.w_full, w_cash = p.w_cash, cert = p.cert,
       X = (hasproperty(p, :X) ? p.X : nothing),
       secs = (hasproperty(p, :secs) ? p.secs : missing),
       mu_qmc = p.mu_qmc, mu_chisq_qmc = p.mu_chisq_qmc, source = path)
end

function _try_load_X_candidates(cands, ctx, M::Int; want_role::Symbol = :opt)
    for path in cands
        isfile(path) || continue
        p = try
            deserialize(path)
        catch err
            _fail(20, "X 反序列化失败: " * path * ": " * sprint(showerror, err))
        end
        (p isa NamedTuple && hasproperty(p, :M)) ||
            _fail(20, "X payload 形态错误（非 NamedTuple/缺 M）: " * path)
        p.M == M || continue                      # 不相关的候选（M 不匹配）跳过
        msg = _x_payload_error(p, ctx, M)
        msg === nothing || _fail(20, "X 契约失败: " * path * ": " * msg)
        rmsg = _x_role_error(p, want_role)
        rmsg === nothing || _fail(20, "X 角色失败: " * path * ": " * rmsg)
        return (; X = p.X, secs = (hasproperty(p, :secs) ? p.secs : missing),
                rqmc_dim = (hasproperty(p, :rqmc_dim) ? p.rqmc_dim : 2),
                mu_qmc = p.mu_qmc, mu_chisq_qmc = p.mu_chisq_qmc, source = path)
    end
    nothing
end

function _try_load_X_any(ctx, M::Int, out_dir::String, prep_dir::String)
    _try_load_X_candidates(_x_candidates(ctx, M, out_dir, prep_dir), ctx, M)
end

function _load_X_any(ctx, M::Int, out_dir::String, prep_dir::String)
    r = _try_load_X_any(ctx, M, out_dir, prep_dir)
    r === nothing &&
        _fail(20, "X 产物缺失（M=$M, world=$(ctx.tag)）；候选: " *
                   join(_x_candidates(ctx, M, out_dir, prep_dir), ", "))
    r
end

# audit X 预生成产物（gen --audit；role=:audit 的 tol 产物）——join 优先读取；
# 缺失返回 nothing（回退现场 gen）；存在但契约错 fail（不静默跳过）。
function _try_load_audit_X(ctx, M::Int, out_dir::String)
    path = joinpath(out_dir, _x_name(ctx, M, :audit))
    isfile(path) || return nothing
    r = _try_load_X_candidates(String[path], ctx, M; want_role = :audit)
    r === nothing &&
        _fail(20, "audit X 存在但 M 不匹配（损坏?）: " * path)
    r
end

function _load_seg_any(ctx, M::Int, out_dir::String, prep_dir::String)
    cands = _seg_candidates(ctx, M, out_dir, prep_dir)
    for path in cands
        isfile(path) || continue
        p = try
            deserialize(path)
        catch err
            _fail(20, "seg 反序列化失败: " * path * ": " * sprint(showerror, err))
        end
        (p isa NamedTuple && hasproperty(p, :M)) ||
            _fail(20, "seg payload 形态错误（非 NamedTuple/缺 M）: " * path)
        p.M == M || continue
        msg = _seg_payload_error(p, ctx, M)
        msg === nothing || _fail(20, "seg 契约失败: " * path * ": " * msg)
        return _adapt_seg(p, path)
    end
    _fail(20, "seg 产物缺失（M=$M, world=$(ctx.tag)）；候选: " *
               join(cands, ", "))
end

# ---------------------------------------------------------------------------
# 段函数（gen / solve / join 核心）
# ---------------------------------------------------------------------------

function _gen_segment(ctx, M::Int, out_dir::String; audit::Bool)
    use_mu, use_chisq = _switch_pair(ctx)
    src = try
        _source_for(ctx; use_mu = use_mu, use_chisq = use_chisq)
    catch err
        _fail(10, "source 构造失败: " * sprint(showerror, err))
    end
    m_seed = Gate0BatchCaptureA._default_mu_seed(ctx.seed)
    am_seed = m_seed ⊻ _AM_AUDIT_XOR
    rule_seed = audit ? audit_seed(ctx.seed) : ctx.seed
    rng_seed = audit ? am_seed : m_seed
    t0 = time()
    X = try
        src.gen(M, SobolOwenRule(src.rqmc_dim, rule_seed), MersenneTwister(rng_seed))
    catch err
        _fail(10, "gen 失败（M=$M）: " * sprint(showerror, err))
    end
    secs = time() - t0
    size(X) == (M, ctx.n_assets) ||
        _fail(10, "gen 尺寸错误: $(size(X)) ≠ ($M, $(ctx.n_assets))")
    all(x -> isfinite(x) && x > 0, X) || _fail(10, "gen 产出含非正/非有限 gross")
    role = audit ? :audit : :opt
    payload = (; version = _TOL_VERSION, kind = :X, role = role,
               world = ctx.tag, seed = ctx.seed, M = M, rqmc_dim = src.rqmc_dim,
               n_assets = ctx.n_assets, mu_qmc = use_mu,
               mu_chisq_qmc = use_chisq, X = X, secs = secs)
    path = joinpath(out_dir, _x_name(ctx, M, role))
    try
        _atomic_serialize(path, payload)
    catch err
        _fail(11, "X 落盘失败: " * sprint(showerror, err))
    end
    println("[tol-trace] gen ok world=$(ctx.tag) role=$(role) M=$M " *
            "secs=$(round(secs, digits = 3)) -> $path")
    (; path = path, M = M, secs = secs)
end

function _solve_segment(ctx, M::Int, out_dir::String, prep_dir::String)
    x = _load_X_any(ctx, M, out_dir, prep_dir)
    locked = _locked_for(ctx)
    budget = 1.0 - sum(locked)
    budget >= 0 || _fail(21, "locked 总和超预算: Σlocked=$(sum(locked))")
    base = locked_wealth_gate0(x.X, locked)
    free_pos = [k for k in 1:ctx.n_assets if locked[k] == 0]
    t0 = time()
    if isempty(free_pos)
        cert = (; feasibility = 0.0, kkt_residual = 0.0, objective_gap = 0.0,
                objective = sum(log, budget .+ base) / M, dual = NaN)
        w_full = zeros(ctx.n_assets)
        w_cash = budget
    else
        X_free = x.X[:, free_pos]
        w_free, w_cash, cert = try
            cash_kelly(X_free; base = base, budget = budget, tol = _KELLY_TOL)
        catch err
            _fail(21, "cash_kelly 失败（M=$M）: " * sprint(showerror, err))
        end
        w_full = zeros(ctx.n_assets)
        w_full[free_pos] .= w_free
        for k in 1:ctx.n_assets
            locked[k] > 0 && (w_full[k] = locked[k])
        end
    end
    secs = time() - t0
    norm_err = abs(sum(w_full) + w_cash - 1.0)
    norm_err <= 1e-8 || _fail(23, "归一性失败: norm_err=$norm_err（M=$M）")
    payload = (; version = _TOL_VERSION, kind = :seg, world = ctx.tag,
               seed = ctx.seed, M = M, rqmc_dim = x.rqmc_dim,
               n_assets = ctx.n_assets, mu_qmc = x.mu_qmc,
               mu_chisq_qmc = x.mu_chisq_qmc, w_full = w_full, w_cash = w_cash,
               cert = cert, X = x.X, secs = secs)
    path = joinpath(out_dir, _seg_name(ctx, M))
    try
        _atomic_serialize(path, payload)
    catch err
        _fail(22, "seg 落盘失败: " * sprint(showerror, err))
    end
    println("[tol-trace] solve ok world=$(ctx.tag) M=$M " *
            "secs=$(round(secs, digits = 3)) norm_err=$(norm_err) -> $path")
    (; path = path, M = M, secs = secs)
end

"""A/B 统计（与 quadrature.jl:777/798/819 与 batch b3 同式；方案 2 表示）。

wealth 统一为 X_free·w_free + w_cash + base_locked（禁止在 X_full·w_full 之上
再加 base——P0-5 的 locked double count）。返回 (; A, B_opt, B_audit,
objective_M2, norm_err_M, norm_err_2M)；非正 wealth 与尺寸错 throw。"""
function _ab_stats(X_2M::Matrix{Float64}, X_a::Matrix{Float64},
                   w_M::Vector{Float64}, wc_M::Float64,
                   w_2M::Vector{Float64}, wc_2M::Float64,
                   locked::Vector{Float64})
    n = length(locked)
    M2 = size(X_2M, 1)
    (size(X_2M, 2) == n && size(X_a, 2) == n) ||
        throw(DimensionMismatch("_ab_stats: X 列数 ≠ locked 长度"))
    size(X_a, 1) == M2 || throw(DimensionMismatch("_ab_stats: X_a 行数 ≠ M2"))
    (length(w_M) == n && length(w_2M) == n) ||
        throw(DimensionMismatch("_ab_stats: w 长度 ≠ locked 长度"))
    A = norm(vcat(w_2M, wc_2M) .- vcat(w_M, wc_M), 1)
    free_pos = [k for k in 1:n if locked[k] == 0]
    base_2M = locked_wealth_gate0(X_2M, locked)
    base_a = locked_wealth_gate0(X_a, locked)
    if isempty(free_pos)
        wealth_f_opt = wc_2M .+ base_2M
        wealth_c_opt = wc_M .+ base_2M
        wealth_f = wc_2M .+ base_a
        wealth_c = wc_M .+ base_a
    else
        X2f = X_2M[:, free_pos]
        Xaf = X_a[:, free_pos]
        wealth_f_opt = X2f * w_2M[free_pos] .+ wc_2M .+ base_2M
        wealth_c_opt = X2f * w_M[free_pos] .+ wc_M .+ base_2M
        wealth_f = Xaf * w_2M[free_pos] .+ wc_2M .+ base_a
        wealth_c = Xaf * w_M[free_pos] .+ wc_M .+ base_a
    end
    for (tag, wealth) in (("B_opt/fine", wealth_f_opt), ("B_opt/coarse", wealth_c_opt),
                          ("B_audit/fine", wealth_f), ("B_audit/coarse", wealth_c))
        all(x -> isfinite(x) && x > 0, wealth) ||
            throw(ErrorException("非正/非有限 wealth：$tag"))
    end
    B_opt = sum(log, wealth_f_opt) / M2 - sum(log, wealth_c_opt) / M2
    B_audit = abs(sum(log, wealth_f) / M2 - sum(log, wealth_c) / M2)
    objective_M2 = sum(log, wealth_f_opt) / M2
    (; A = A, B_opt = B_opt, B_audit = B_audit, objective_M2 = objective_M2,
       norm_err_M = abs(sum(w_M) + wc_M - 1.0),
       norm_err_2M = abs(sum(w_2M) + wc_2M - 1.0))
end

function _obtain_audit_X(ctx, M2::Int, out_dir::String, seg_2M)
    pre = _try_load_audit_X(ctx, M2, out_dir)
    if pre !== nothing
        (pre.mu_qmc == seg_2M.mu_qmc && pre.mu_chisq_qmc == seg_2M.mu_chisq_qmc) ||
            _fail(20, "预生成 audit X 的开关与 seg_2M 不一致（mu_qmc/chisq）")
        return (; X = pre.X, from_artifact = true)
    end
    use_mu = seg_2M.mu_qmc
    use_chisq = seg_2M.mu_chisq_qmc
    src = try
        _source_for(ctx; use_mu = use_mu, use_chisq = use_chisq)
    catch err
        _fail(30, "audit source 构造失败: " * sprint(showerror, err))
    end
    m_seed = Gate0BatchCaptureA._default_mu_seed(ctx.seed)
    am_seed = m_seed ⊻ _AM_AUDIT_XOR
    X_a = try
        src.gen(M2, SobolOwenRule(src.rqmc_dim, audit_seed(ctx.seed)),
                MersenneTwister(am_seed))
    catch err
        _fail(30, "audit 现场 gen 失败（M2=$M2）: " * sprint(showerror, err))
    end
    size(X_a) == (M2, ctx.n_assets) ||
        _fail(30, "audit gen 尺寸错误: $(size(X_a))")
    (; X = X_a, from_artifact = false)
end

function _join_core(ctx, M1::Int, M2::Int, out_dir::String, prep_dir::String)
    M2 == 2 * M1 || _fail(2, "join 层对必须 M2 == 2*M1（got $M1→$M2）")
    t0 = time()
    seg_M = _load_seg_any(ctx, M1, out_dir, prep_dir)
    seg_2M = _load_seg_any(ctx, M2, out_dir, prep_dir)
    (seg_M.mu_qmc == seg_2M.mu_qmc &&
     seg_M.mu_chisq_qmc == seg_2M.mu_chisq_qmc) ||
        _fail(20, "seg_M/seg_2M 开关不一致（mu_qmc/chisq）")
    locked = _locked_for(ctx)
    # X_2M：独立产物优先（tol 或 batch），seg 内嵌回退
    x2 = _try_load_X_any(ctx, M2, out_dir, prep_dir)
    X_2M = nothing
    x2_secs = missing
    if x2 !== nothing
        (x2.mu_qmc == seg_2M.mu_qmc && x2.mu_chisq_qmc == seg_2M.mu_chisq_qmc) ||
            _fail(20, "独立 X_2M 的开关与 seg_2M 不一致（mu_qmc/chisq）")
        X_2M = x2.X
        x2_secs = x2.secs
    elseif seg_2M.X !== nothing
        size(seg_2M.X) == (M2, ctx.n_assets) ||
            _fail(20, "seg_2M 内嵌 X 尺寸错误: $(size(seg_2M.X))")
        X_2M = seg_2M.X
    else
        _fail(20, "X_2M 无法获得（独立产物缺失且 seg_2M 无内嵌 X）")
    end
    size(X_2M) == (M2, ctx.n_assets) ||
        _fail(20, "X_2M 尺寸错误: $(size(X_2M))")
    # audit（预生成产物优先，否则现场 gen——两种路径同一派生链）
    audit = _obtain_audit_X(ctx, M2, out_dir, seg_2M)
    stats = try
        _ab_stats(X_2M, audit.X, seg_M.w_full, seg_M.w_cash,
                  seg_2M.w_full, seg_2M.w_cash, locked)
    catch err
        _fail(31, "A/B 统计求值失败: " * sprint(showerror, err))
    end
    (stats.norm_err_M <= 1e-8 && stats.norm_err_2M <= 1e-8) ||
        _fail(32, "归一性失败: norm_err_M=$(stats.norm_err_M), " *
                  "norm_err_2M=$(stats.norm_err_2M)")
    join_secs = time() - t0
    secs_final = _compose_secs((seg_M.secs, seg_2M.secs, x2_secs, join_secs))
    dist = _dist_star(ctx, seg_2M.w_full, seg_2M.w_cash, out_dir)
    csv_path = _csv_path(ctx, out_dir, seg_2M.mu_qmc, seg_2M.mu_chisq_qmc)
    header = _csv_header(ctx.n_assets)
    row = _csv_row(M1, M2, stats.A, stats.B_opt, stats.B_audit,
                   seg_2M.w_full, seg_2M.w_cash, dist, secs_final,
                   stats.objective_M2)
    try
        _csv_append(csv_path, header, row)
    catch err
        _fail(33, "CSV 追加失败: " * csv_path * ": " * sprint(showerror, err))
    end
    ds = dist === missing ? "NA" : string(dist)
    ss = secs_final === missing ? "NA" : string(round(secs_final, digits = 3))
    println("[tol-trace] join ok world=$(ctx.tag) M=$M1->$M2 " *
            "A=$(stats.A) Bopt=$(stats.B_opt) Baudit=$(stats.B_audit) " *
            "dist_star=$ds secs=$ss audit=$(audit.from_artifact ? "artifact" : "in-situ") " *
            "-> $csv_path")
    (; stats = stats, csv = csv_path)
end

# ---------------------------------------------------------------------------
# ref（解析基准）
# ---------------------------------------------------------------------------

function _build_ref(tag::String, spec; fail_code::Int = 40)
    t0 = time()
    if spec.kind == :gauss
        mu = spec.mu
        sigma = spec.sigma
        if length(mu) == 2
            rc = try
                gh_refcheck_2d(mu, sigma; n_a = 80, n_b = 120, tol = 1e-6)
            catch err
                _fail(fail_code, "GH 2D 基准失败: " * sprint(showerror, err))
            end
            rc.consistent ||
                _fail(fail_code, "GH 2D 自洽失败: 80→120 w* L1 diff=" *
                                 string(rc.diff) * " ≥ 1e-6")
            return (; version = _TOL_VERSION, kind = :ref, world = tag,
                    has_wref = true, w_ref = rc.rb.w, wc_ref = rc.rb.wc,
                    objective = rc.rb.I, grad_res = rc.rb.grad_res,
                    consistency_diff = rc.diff,
                    meta = (; method = :gh2d, n_gh_a = 80, n_gh_b = 120),
                    secs = time() - t0)
        else
            rc = try
                gh_refcheck_3d(mu, sigma; n_gh = 40, tol = 1e-6)
            catch err
                _fail(fail_code, "GH 3D 基准失败: " * sprint(showerror, err))
            end
            rc.consistent ||
                _fail(fail_code, "GH 3D 一阶条件残差 " * string(rc.ra.grad_res) *
                                 " ≥ 1e-6")
            return (; version = _TOL_VERSION, kind = :ref, world = tag,
                    has_wref = true, w_ref = rc.ra.w, wc_ref = rc.ra.wc,
                    objective = rc.ra.I, grad_res = rc.ra.grad_res,
                    consistency_diff = missing,
                    meta = (; method = :gh3d, n_gh = 40,
                             note = "3D 只做局部最优性验证（设计 §3.2）；无节点翻倍自洽"),
                    secs = time() - t0)
        end
    else
        a = try
            analytic_world_ref(tag)
        catch err
            _fail(fail_code, "解析锚失败: " * sprint(showerror, err))
        end
        return (; version = _TOL_VERSION, kind = :ref, world = tag,
                has_wref = a.has_wref, w_ref = a.w_ref, wc_ref = a.wc_ref,
                objective = a.objective, grad_res = 0.0, consistency_diff = 0.0,
                meta = (; method = :analytic, note = a.note),
                secs = time() - t0)
    end
end

function _load_or_compute_ref(ctx, out_dir::String)
    path = joinpath(out_dir, "tolref_" * ctx.tag * ".bin")
    if isfile(path)
        r = try
            deserialize(path)
        catch err
            _fail(34, "ref 反序列化失败: " * path * ": " * sprint(showerror, err))
        end
        (r isa NamedTuple && hasproperty(r, :kind) && r.kind == :ref &&
         hasproperty(r, :world) && r.world == ctx.tag) ||
            _fail(34, "ref 契约失败: " * path)
        hasproperty(r, :has_wref) || _fail(34, "ref 缺 has_wref: " * path)
        return r
    end
    _build_ref(ctx.tag, ctx.spec; fail_code = 34)   # 缺产物现场计算（不落盘）
end

# ---------------------------------------------------------------------------
# CSV 与 dist_star
# ---------------------------------------------------------------------------

function _csv_header(n::Int)
    cols = String["M", "M2", "A", "Bopt", "Baudit"]
    for i in 1:n
        push!(cols, "w" * string(i))
    end
    append!(cols, ["wc", "dist_star", "secs", "objective_M2"])
    join(cols, ",")
end

function _csv_row(M::Int, M2::Int, A::Float64, Bopt::Float64, Baudit::Float64,
                  w_full::Vector{Float64}, w_cash::Float64,
                  dist_star::Union{Missing,Float64},
                  secs::Union{Missing,Float64}, objective::Float64)
    parts = String[string(M), string(M2), string(A), string(Bopt), string(Baudit)]
    for x in w_full
        push!(parts, string(x))
    end
    push!(parts, string(w_cash))
    push!(parts, dist_star === missing ? "" : string(dist_star))
    push!(parts, secs === missing ? "" : string(secs))
    push!(parts, string(objective))
    join(parts, ",")
end

function _csv_append(path::String, header::String, row::String)
    if isfile(path)
        old = read(path, String)
        firstline = split(chomp(old), '\n')[1]
        firstline == header ||
            throw(ErrorException("CSV header 不匹配（拒绝混写）: " * path))
        new = endswith(old, "\n") ? old : old * "\n"
        new *= row * "\n"
    else
        new = header * "\n" * row * "\n"
    end
    _atomic_write_text(path, new)
end

_csv_path(ctx, out_dir::String, use_mu::Bool, use_chisq::Bool) =
    ctx.kind == :real ?
        joinpath(out_dir, "tolB_t" * string(ctx.t) * "_layers" *
                           ((use_mu && use_chisq) ? "" : "_pdefault") * ".csv") :
        joinpath(out_dir, "tolA_" * ctx.tag * "_" * _seedhex(ctx.seed) * ".csv")

function _compose_secs(parts::Tuple)
    total = 0.0
    for p in parts
        p === missing && return missing
        total += Float64(p)
    end
    total
end

function _dist_star(ctx, w_full::Vector{Float64}, w_cash::Float64, out_dir::String)
    ctx.kind == :real && return missing
    (ctx.tag in ("A5a", "A5b")) && return missing
    ref = _load_or_compute_ref(ctx, out_dir)
    (ref.has_wref && length(ref.w_ref) == ctx.n_assets) || return missing
    norm(vcat(w_full, w_cash) .- vcat(ref.w_ref, ref.wc_ref), 1)
end

# ---------------------------------------------------------------------------
# 命令
# ---------------------------------------------------------------------------

function _parse_cli(args::Vector{String})
    sub = args[1]
    pos = String[]
    opts = Dict{String,String}()
    flags = Set{String}()
    i = 2
    while i <= length(args)
        a = args[i]
        if a in ("--seed", "--out", "--prep-dir")
            i + 1 <= length(args) || _fail(2, "缺少 " * a * " 的值")
            opts[a] = args[i + 1]
            i += 2
        elseif a == "--audit"
            push!(flags, a)
            i += 1
        elseif startswith(a, "--")
            _fail(2, "未知选项: " * a * "\n" * _USAGE)
        else
            push!(pos, a)
            i += 1
        end
    end
    (; sub = sub, pos = pos, opts = opts, flags = flags)
end

function _cmd_gen(parsed)
    length(parsed.pos) == 2 ||
        _fail(2, "gen <world> <M> [--seed HEX] [--audit] [--out DIR] [--prep-dir DIR]")
    world_arg, Ms = parsed.pos[1], parsed.pos[2]
    M = _parse_M_or_fail(Ms)
    out_dir = _out_dir(parsed)
    mkpath(out_dir)
    prep_dir = _prep_dir(parsed)
    ctx = _resolve_world_or_fail(world_arg, get(parsed.opts, "--seed", nothing),
                                 prep_dir)
    _gen_segment(ctx, M, out_dir; audit = ("--audit" in parsed.flags))
    exit(0)
end

function _cmd_solve(parsed)
    length(parsed.pos) == 2 ||
        _fail(2, "solve <world> <M> [--seed HEX] [--out DIR] [--prep-dir DIR]")
    "--audit" in parsed.flags && _fail(2, "--audit 仅 gen 支持")
    world_arg, Ms = parsed.pos[1], parsed.pos[2]
    M = _parse_M_or_fail(Ms)
    out_dir = _out_dir(parsed)
    mkpath(out_dir)
    prep_dir = _prep_dir(parsed)
    ctx = _resolve_world_or_fail(world_arg, get(parsed.opts, "--seed", nothing),
                                 prep_dir)
    _solve_segment(ctx, M, out_dir, prep_dir)
    exit(0)
end

function _cmd_join(parsed)
    length(parsed.pos) == 3 ||
        _fail(2, "join <world> <M> <M2> [--seed HEX] [--out DIR] [--prep-dir DIR]")
    "--audit" in parsed.flags && _fail(2, "--audit 仅 gen 支持")
    world_arg, M1s, M2s = parsed.pos[1], parsed.pos[2], parsed.pos[3]
    M1 = _parse_M_or_fail(M1s)
    M2 = _parse_M_or_fail(M2s)
    out_dir = _out_dir(parsed)
    mkpath(out_dir)
    prep_dir = _prep_dir(parsed)
    ctx = _resolve_world_or_fail(world_arg, get(parsed.opts, "--seed", nothing),
                                 prep_dir)
    _join_core(ctx, M1, M2, out_dir, prep_dir)
    exit(0)
end

function _cmd_run(parsed)
    length(parsed.pos) == 3 ||
        _fail(2, "run <world> <Mmin> <Mmax> [--seed HEX] [--out DIR] [--prep-dir DIR]")
    "--audit" in parsed.flags && _fail(2, "--audit 仅 gen 支持")
    world_arg, mins, maxs = parsed.pos[1], parsed.pos[2], parsed.pos[3]
    Mmin = _parse_M_or_fail(mins)
    Mmax = _parse_M_or_fail(maxs)
    (_is_pow2(Mmin) && _is_pow2(Mmax)) ||
        _fail(2, "run: Mmin/Mmax 须为 2 的幂（got $Mmin/$Mmax）")
    Mmin <= Mmax || _fail(2, "run: Mmin ≤ Mmax 需要（got $Mmin/$Mmax）")
    out_dir = _out_dir(parsed)
    mkpath(out_dir)
    prep_dir = _prep_dir(parsed)
    ctx = _resolve_world_or_fail(world_arg, get(parsed.opts, "--seed", nothing),
                                 prep_dir)
    layers = Int[]
    m = Mmin
    while m <= Mmax
        push!(layers, m)
        m *= 2
    end
    for M in layers
        _gen_segment(ctx, M, out_dir; audit = false)
        _solve_segment(ctx, M, out_dir, prep_dir)
    end
    for i in 1:(length(layers) - 1)
        _join_core(ctx, layers[i], layers[i + 1], out_dir, prep_dir)
    end
    println("[tol-trace] run ok world=$(ctx.tag) layers=" * join(layers, ","))
    exit(0)
end

function _cmd_ref(parsed)
    length(parsed.pos) == 1 || _fail(2, "ref <world> [--out DIR]")
    "--audit" in parsed.flags && _fail(2, "--audit 仅 gen 支持")
    world_arg = parsed.pos[1]
    startswith(world_arg, "real:") &&
        _fail(2, "ref 不支持 real（真实链无解析 w*——dist_star 留空）")
    spec = _world_spec_or_fail(world_arg)
    out_dir = _out_dir(parsed)
    mkpath(out_dir)
    payload = _build_ref(world_arg, spec; fail_code = 40)
    path = joinpath(out_dir, "tolref_" * world_arg * ".bin")
    try
        _atomic_serialize(path, payload)
    catch err
        _fail(41, "ref 落盘失败: " * sprint(showerror, err))
    end
    wstr = payload.has_wref ? string(payload.w_ref) : "tie-only"
    dstr = payload.consistency_diff === missing ? "NA" :
           string(payload.consistency_diff)
    println("[tol-trace] ref ok world=$world_arg w=$wstr consistency=$dstr -> $path")
    exit(0)
end

# ---------------------------------------------------------------------------
# selftest（结构锚；不调 kelly、不调 GH 求根）
# ---------------------------------------------------------------------------

function _selftest()
    checks = 0
    try
        # 1) world 解析
        @assert _world_spec("A1") !== nothing
        @assert _world_spec("A2") !== nothing
        @assert _world_spec("A3") !== nothing
        @assert _world_spec("A4") !== nothing
        @assert _world_spec("A5a") !== nothing
        @assert _world_spec("A5b") !== nothing
        @assert _world_spec("Z9") === nothing
        @assert _world_spec("real:330") === nothing
        checks += 1
        # 2) seed 解析
        @assert _parse_seed_soft("0x0000C0FFEE") == UInt64(0x0000C0FFEE)
        @assert _parse_seed_soft("BEEF") == UInt64(0xBEEF)
        @assert _parse_seed_soft("0x") === nothing
        @assert _parse_seed_soft("zz") === nothing
        @assert _parse_M_soft("64") == 64
        @assert _parse_M_soft("0") === nothing
        @assert _is_pow2(64) && _is_pow2(4096) && !_is_pow2(96)
        checks += 1
        # 3) rule 嵌套锚（前缀逐位）
        rp8 = rule_points(SobolOwenRule(2, UInt64(0xC0FFEE)), 8)
        rp4 = rule_points(SobolOwenRule(2, UInt64(0xC0FFEE)), 4)
        @assert rp8[1:4, :] == rp4
        checks += 1
        # 4) 派生常数一致性（与 src 定义逐式）
        @assert audit_seed(UInt64(0x1234)) == (UInt64(0x1234) ⊻ 0x9E3779B97F4A7C15)
        @assert Gate0BatchCaptureA._default_mu_seed(UInt64(0x1234)) ==
                (UInt64(0x1234) + 0x2545F4914F6CDD1D)
        checks += 1
        # 5) GH 节点（n=2 解析：±1/√2、权重 √π/2、和 √π）
        xg, wg = gh_nodes(2)
        @assert isapprox(sum(wg), sqrt(pi); atol = 1e-14)
        @assert isapprox(abs(xg[1]), 1 / sqrt(2); atol = 1e-14)
        @assert isapprox(wg[1], sqrt(pi) / 2; atol = 1e-14)
        checks += 1
        # 6) A1 source：确定性 + 首行手算（inverse-CDF）
        spec1 = _world_spec("A1")
        src1 = _synthetic_source(spec1)
        r1 = SobolOwenRule(2, UInt64(0x11))
        pts1 = rule_points(r1, 4)
        X1a = src1.gen(4, r1, MersenneTwister(UInt64(0x22)))
        X1b = src1.gen(4, r1, MersenneTwister(UInt64(0x22)))
        @assert X1a == X1b
        @assert size(X1a) == (4, 2)
        for j in 1:2
            @assert X1a[1, j] ==
                    exp(spec1.mu[j] + spec1.sigma[j] * quantile(Normal(), pts1[1, j]))
        end
        checks += 1
        # 7) A3 source：确定性 + 第三维派生手算
        spec3 = _world_spec("A3")
        src3 = _synthetic_source(spec3)
        r3 = SobolOwenRule(2, UInt64(0x55))
        X3a = src3.gen(4, r3, MersenneTwister(UInt64(0x66)))
        X3b = src3.gen(4, r3, MersenneTwister(UInt64(0x66)))
        @assert X3a == X3b && size(X3a) == (4, 3)
        sub3 = SobolOwenRule(1, UInt64(0x55) ⊻ _A3_DIM3_SEED_XOR)
        pts3 = rule_points(sub3, 4)
        @assert X3a[2, 3] ==
                exp(spec3.mu[3] + spec3.sigma[3] * quantile(Normal(), pts3[2, 1]))
        checks += 1
        # 8) A4 source：行选择 ∈ 行集
        spec4 = _world_spec("A4")
        src4 = _synthetic_source(spec4)
        X4 = src4.gen(8, SobolOwenRule(1, UInt64(0x33)), MersenneTwister(UInt64(0x44)))
        @assert size(X4) == (8, 3)
        for s in 1:8
            @assert any(t -> X4[s, :] == _A4_X[t, :], 1:3)
        end
        checks += 1
        # 9) 解析锚 A4/A5
        a4 = analytic_world_ref("A4")
        @assert a4.has_wref && a4.w_ref == [1 / 3, 1 / 3, 1 / 3] && a4.wc_ref == 0.0
        a5a = analytic_world_ref("A5a")
        a5b = analytic_world_ref("A5b")
        @assert !a5a.has_wref && !a5b.has_wref
        checks += 1
        # 10) _ab_stats 例 1（无 locked；对数比恒等）
        st1 = _ab_stats(reshape([2.0, 1.0], 2, 1), fill(1.0, 2, 1),
                        [0.25], 0.75, [0.5], 0.5, [0.0])
        @assert st1.A == 0.5
        @assert isapprox(st1.B_opt, log(1.2) / 2; atol = 1e-15)
        @assert st1.B_audit == 0.0
        @assert isapprox(st1.objective_M2, log(1.5) / 2; atol = 1e-15)
        @assert st1.norm_err_M == 0.0 && st1.norm_err_2M == 0.0
        checks += 1
        # 11) _ab_stats 例 2（部分 locked；恒一财富 → B 全零）
        st2 = _ab_stats(fill(1.0, 2, 2), fill(1.0, 2, 2),
                        [0.25, 0.4], 0.35, [0.5, 0.4], 0.1, [0.0, 0.4])
        @assert st2.A == 0.5
        @assert abs(st2.B_opt) <= 1e-15 && abs(st2.B_audit) <= 1e-15 &&
                abs(st2.objective_M2) <= 1e-15
        checks += 1
        # 12) _ab_stats 例 3（全 locked 分支）
        st3 = _ab_stats(reshape([1.25, 0.8], 2, 1), reshape([1.1, 0.9], 2, 1),
                        [0.3], 0.7, [0.3], 0.7, [0.3])
        @assert st3.A == 0.0 && st3.B_opt == 0.0 && st3.B_audit == 0.0
        checks += 1
        # 13) X/seg 契约（好 + 坏）
        ctxD = (; kind = :gauss, tag = "A1", t = -1, seed = UInt64(0xAB),
                n_assets = 2, prep = nothing, spec = nothing)
        goodX = (; version = _TOL_VERSION, kind = :X, role = :opt, world = "A1",
                 seed = UInt64(0xAB), M = 4, rqmc_dim = 2, n_assets = 2,
                 mu_qmc = false, mu_chisq_qmc = false, X = ones(4, 2), secs = 0.0)
        @assert _x_payload_error(goodX, ctxD, 4) === nothing
        # role 值域（:opt/:audit 均过）；具体角色匹配由读取器 want_role 执行
        @assert _x_payload_error(merge(goodX, (; role = :audit)), ctxD, 4) === nothing
        @assert _x_payload_error(merge(goodX, (; role = :bogus)), ctxD, 4) !== nothing
        @assert _x_role_error(goodX, :opt) === nothing
        @assert _x_role_error(goodX, :audit) !== nothing
        @assert _x_role_error(merge(goodX, (; version = "batch-solve3-v1")), :audit) === nothing
        @assert _x_payload_error(merge(goodX, (; world = "A2")), ctxD, 4) !== nothing
        @assert _x_payload_error(merge(goodX, (; version = "x")), ctxD, 4) !== nothing
        @assert _x_payload_error(merge(goodX, (; seed = UInt64(1))), ctxD, 4) !== nothing
        @assert _x_payload_error(merge(goodX, (; M = 8)), ctxD, 4) !== nothing
        @assert _x_payload_error((; version = _TOL_VERSION, kind = :X), ctxD, 4) !== nothing
        goodS = (; version = _TOL_VERSION, kind = :seg, world = "A1",
                 seed = UInt64(0xAB), M = 4, rqmc_dim = 2, n_assets = 2,
                 mu_qmc = false, mu_chisq_qmc = false, w_full = [0.4, 0.4],
                 w_cash = 0.2, cert = (;), X = ones(4, 2), secs = 0.0)
        @assert _seg_payload_error(goodS, ctxD, 4) === nothing
        @assert _seg_payload_error(merge(goodS, (; kind = :X)), ctxD, 4) !== nothing
        @assert _seg_payload_error(merge(goodS, (; M = 8)), ctxD, 4) !== nothing
        checks += 1
        # 14) CSV header/row roundtrip
        hdr = _csv_header(2)
        @assert hdr == "M,M2,A,Bopt,Baudit,w1,w2,wc,dist_star,secs,objective_M2"
        row = _csv_row(64, 128, 0.125, 0.25, 0.5, [0.3, 0.6], 0.1,
                       missing, 1.25, 0.75)
        f = split(row, ',')
        @assert length(f) == 11
        @assert f[1] == "64" && f[2] == "128"
        @assert f[6] == "0.3" && f[7] == "0.6" && f[8] == "0.1"
        @assert f[9] == "" && f[10] == "1.25" && f[11] == "0.75"
        row2 = _csv_row(64, 128, 0.1, 0.2, 0.3, [0.5, 0.5], 0.0, 0.9, 2.0, 0.4)
        f2 = split(row2, ',')
        @assert f2[9] == "0.9" && f2[10] == "2.0"
        checks += 1
        # 15) CSV 路径与 secs 组合
        ctxS = (; kind = :gauss, tag = "A1", t = -1, seed = UInt64(0xAB),
                n_assets = 2, prep = nothing, spec = nothing)
        @assert basename(_csv_path(ctxS, "d", false, false)) ==
                "tolA_A1_" * _seedhex(UInt64(0xAB)) * ".csv"
        ctxR = (; kind = :real, tag = "t330", t = 330, seed = UInt64(1),
                n_assets = 65, prep = nothing, spec = nothing)
        @assert basename(_csv_path(ctxR, "d", true, true)) == "tolB_t330_layers.csv"
        @assert basename(_csv_path(ctxR, "d", false, false)) ==
                "tolB_t330_layers_pdefault.csv"
        @assert _compose_secs((1.0, 2.0, 3.0)) == 6.0
        @assert _compose_secs((1.0, missing, 3.0)) === missing
        checks += 1
    catch err
        println(stderr, "[tol-trace selftest] FAILED: " * sprint(showerror, err))
        exit(3)
    end
    println("[tol-trace selftest] ok (" * string(checks) *
            " check groups; kelly/GH 求根 not invoked)")
    exit(0)
end

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

function main()
    isempty(ARGS) && _fail(2, _USAGE)
    if ARGS[1] == "--selftest"
        _selftest()
    end
    parsed = _parse_cli(ARGS)
    sub = parsed.sub
    if sub == "gen"
        _cmd_gen(parsed)
    elseif sub == "solve"
        _cmd_solve(parsed)
    elseif sub == "join"
        _cmd_join(parsed)
    elseif sub == "run"
        _cmd_run(parsed)
    elseif sub == "ref"
        _cmd_ref(parsed)
    else
        _fail(2, "unknown subcommand: " * sub * "\n" * _USAGE)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
