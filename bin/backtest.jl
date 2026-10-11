# =============================================================================
# bin/backtest.jl —— KTrader 唯一当前入口：Gate-0 回测（2026-10-10 切换收口）
# =============================================================================
#
# 身份（Gate-0 验收结论第 7 步）：本脚本是 KTrader 当前唯一的「回测/决策」
# 生产入口，驱动独立新线模块 KTraderGate0（src/gate0/，裁决 H1）：
#
#   MarketFacts + Eligibility
#     → single_day_decision（Step 15 单日 driver：统一 mode 坐标 → 完整
#        response posterior → strict prequential 残差 → vector innovation
#        → adaptive RQMC 双证书收敛 → cash Kelly）
#     → run_gate0_backtest（sequential 多日 + cash / EW-daily / EW-buyhold
#        三 benchmark，D-072）
#
# 历史 release：旧线 `KTrader`（2.0-RC legacy implementation）冻结为历史
# fixture；其对比/报告/执行工具（bin/bench.jl、bin/report.jl、bin/live.jl）
# 保留但标注为历史线工具，不属于当前入口；两线互不 include（裁决 H1）。
#
# 运行成本现状（如实，必须随入口一起读）：
# - Gate-0 是 D-082/D-083 slow reference：单线程、无 incremental、无
#   scheduler、无缓存；**多日窗口是批量负载**——每个决策日重跑一条完整
#   链条（response posterior + prequential + innovation + quadrature +
#   Kelly）。
# - 默认末 20 天短窗口（见下）。60/501 天与完整两年在单命令 60s 护栏
#   之外，归 owner 按 D-088/D-089 分批调度；**不得把短窗口计时外推为长
#   回测耗时**。
# - 已知限制（2026-10-10 最终）：kelly 收敛鲁棒性缺口已修复（条件行缩放
#   + polish 防护；见 AGENTS.md 第14任段）；最终受控验证 984/984 全绿
#   （archive/evidence/gate0_final_verify_20261010/）。
#
# 数据桥（H1 互不 include 的直接推论）：`import KTrader` 只取旧线数据层
# `load_bars`（src/data.jl），把 Bars 的字段数组提取出来，经
# `from_bars_arrays`（src/gate0/market.jl）构造 MarketFacts——桥只搬裸
# 矩阵，不跨 module 依赖数学层。
#
# 为什么用 `import KTrader` 而非 `using KTrader`：旧线 KTrader 与
# KTraderGate0 存在同名导出（signal_prices / ruler / path_basis_1d /
# principal_sqrt_root / WARMUP 等）。`using` 两个 module 会让同名导出
# ambiguous（调用即报错）；`import KTrader` 只引入模块名，旧线数据层经
# `KTrader.load_bars` 限定名访问，KTraderGate0 的导出零冲突。
#
# Eligibility 构造（D-013/D-014/D-016/D-020）：
# - 默认政策：`Eligibility(mf)`——model_admitted/executable 从 MarketFacts
#   派生，trade_eligible 全 true；
# - 可选 252 有效交易日政策：`Eligibility(mf, effective_bars_252(mf))`
#   （D-018/H4：累计第 252 个有效 bar 当日解除；GATE0_TRADE_252=true）。
#
# 参数面（全部为 GATE0_*；不保留旧线回测参数——旧参数语义（固定
# S=300、ADAPTIVE_SCENARIOS=false、F_FOLDS=3、RIDGE_ALPHA、ENGINE、
# DATE_TASKS 等）已在 Gate-0 合同中废止，不在此入口消费）：
#
#   GATE0_DATA_DIR          数据目录（默认 <repo>/data）。
#   GATE0_MODE              adaptive（默认，生产路径：A/B_opt/B_audit/C 双
#                           证书收敛）| reference（固定 S 对照——D-062 的
#                           合法用途：unit test / artifact replay /
#                           microbenchmark / historical fixture，不是生产
#                           默认）。
#   GATE0_S_REFERENCE       reference 模式的固定 S（默认 256）。
#   GATE0_T_START / GATE0_T_END
#                           决策区间（行索引；默认末 20 天窗口，t_end 默认
#                           T；t_start 必须 ≥ WARMUP+1 = 257）。
#   GATE0_SEED              决策随机流 seed（十六进制字符串，可带 0x 前缀；
#                           默认 = driver 的 _DRIVER_DEFAULT_SEED，同一
#                           真源）。
#   GATE0_TRADE_252         true → 252 有效交易日 Eligibility 政策（H4；
#                           默认 false）。
#   GATE0_POSTERIOR_TOL / GATE0_POSTERIOR_MAX_CELLS
#                           posterior 2D 求积口径（默认 1e-6 / 2048，P0-7
#                           严格起点；运行超时的正确行为是 fail / 缩小
#                           fixture，不是放宽 tolerance）。
#   GATE0_MAX_SCENARIOS     adaptive 积分预算（默认 512）。
#   GATE0_WEIGHT_TOL / GATE0_UTILITY_TOL / GATE0_KELLY_TOL
#                           证书容差（D-066 初始候选；默认与 driver 的
#                           生产口径一致：1e-4 / 1e-6 / 1e-6）。
#
# 示例（smoke 与最小规模运行由 DevOps 受控调度；t_start/t_end 按数据调整）：
#   默认末 20 天：  julia --startup-file=no --project=. bin/backtest.jl
#   5 天最小窗口：  GATE0_T_START=14001 GATE0_T_END=14006 \
#                     julia --startup-file=no --project=. bin/backtest.jl
#   reference 对照：GATE0_MODE=reference GATE0_S_REFERENCE=64 \
#                     julia --startup-file=no --project=. bin/backtest.jl
#
# 输出：Gate0BacktestResult 关键字段——账户净值曲线、cash / EW-daily /
# EW-buy-hold 三 benchmark 曲线（D-072）、ew_rebalance_gap（D-073 只报告
# 不做 volatility pump 因果解释）、逐日记录（Gate0DayRecord：决策、持仓、
# 财富因子、净值）。
# =============================================================================

using Dates, Statistics, LinearAlgebra, Printf

# BLAS 单线程（Manager 裁决 U2，2026-10-10；见 docs/MULTIDAY_COST_OPTIMIZATION.md
# §7）：gate0 负载实测最优配置——小 GEMM 在多线程 BLAS 下被线程开销放大
# ~3×（单点 log_evidence 8.36ms@默认 6 线程 → 2.52ms@1 线程；证据
# archive/evidence/gate0_merged_verify8_20261010/ 分段微基准）。这是性能/
# 确定性配置，不改变任何数学对象、门禁或容差（P0-7/D-066 不受影响）。
BLAS.set_num_threads(1)

# 旧线 module 只经模块名访问数据层（避免同名导出 ambiguity，见文件头）
import KTrader

# ---------------------------------------------------------------------------
# 加载 Gate-0 独立顶层 module（与 test/gate0/ 各测试文件同模式）
# ---------------------------------------------------------------------------
include(joinpath(@__DIR__, "..", "src", "gate0", "KTraderGate0.jl"))
using .KTraderGate0

# ---------------------------------------------------------------------------
# 数据桥：旧线数据层 → MarketFacts（H1；只搬裸矩阵）
# ---------------------------------------------------------------------------
data_dir = get(ENV, "GATE0_DATA_DIR", joinpath(@__DIR__, "..", "data"))
b = KTrader.load_bars(data_dir)
mf = from_bars_arrays(b.dates, b.symbols, b.close, b.adj, b.bar)

# ---------------------------------------------------------------------------
# 参数解析（参数面 = GATE0_*；默认值与 driver 合同一致）
# ---------------------------------------------------------------------------
function _parse_seed(text::AbstractString)
    t = strip(text)
    s = startswith(lowercase(t), "0x") ? t[3:end] : t
    return parse(UInt64, s; base = 16)
end

mode = Symbol(get(ENV, "GATE0_MODE", "adaptive"))
mode in (:adaptive, :reference) ||
    error("GATE0_MODE must be adaptive or reference (got $mode)")
s_reference = parse(Int, get(ENV, "GATE0_S_REFERENCE", "256"))
seed = _parse_seed(get(ENV, "GATE0_SEED",
                       string(KTraderGate0._DRIVER_DEFAULT_SEED, base = 16)))
trade252 = get(ENV, "GATE0_TRADE_252", "false") == "true"
posterior_tol = parse(Float64, get(ENV, "GATE0_POSTERIOR_TOL", "1e-6"))
posterior_max_cells = parse(Int, get(ENV, "GATE0_POSTERIOR_MAX_CELLS", "2048"))
max_scenarios = parse(Int, get(ENV, "GATE0_MAX_SCENARIOS", "512"))
weight_tol = parse(Float64, get(ENV, "GATE0_WEIGHT_TOL", "1e-4"))
utility_tol = parse(Float64, get(ENV, "GATE0_UTILITY_TOL", "1e-6"))
kelly_tol = parse(Float64, get(ENV, "GATE0_KELLY_TOL", "1e-6"))
# b′（μ 通道部分 QMC）：E3 判据已成立（A 越过）；默认关闭（false）时逐位
# 不变；60-day 前置待验证。经 GATE0_MU_QMC=true 开启。
mu_qmc = get(ENV, "GATE0_MU_QMC", "false") == "true"
# CHISQ-QMC（A 门过门配置，证据 CQ_*）：mu_qmc=true + GATE0_CHISQ_QMC=true +
# GATE0_MAX_SCENARIOS=131072 → A=6.2585e-5、三项证书全过。默认关闭时逐位
# 不变；成本：131072 档仅 kelly 层约 41s / ~1.6GiB；60-day 为批量负载（分批调度）。
mu_chisq_qmc = get(ENV, "GATE0_CHISQ_QMC", "false") == "true"

# ---------------------------------------------------------------------------
# 决策区间：默认末 20 天短窗口（Gate-0 slow reference；D-088/D-089）
# ---------------------------------------------------------------------------
T = size(mf.observed, 1)
t_end = parse(Int, get(ENV, "GATE0_T_END", string(T)))
t_start = parse(Int, get(ENV, "GATE0_T_START",
                         string(max(WARMUP + 1, t_end - 20))))

# ---------------------------------------------------------------------------
# Eligibility（默认政策或 252 有效交易日政策，D-018/H4）
# ---------------------------------------------------------------------------
el = trade252 ? Eligibility(mf, effective_bars_252(mf)) : Eligibility(mf)

println("KTrader current entry: bin/backtest.jl -> KTraderGate0 (Gate-0 line; 2026-10-10)")
println("Julia=$(VERSION) threads=$(Threads.nthreads()) panel=$(size(mf.observed)) input=$(first(mf.dates)):$(last(mf.dates))")
println("window=$t_start:$t_end mode=$mode S_reference=$s_reference seed=0x$(string(seed, base = 16)) trade252=$trade252")
println("posterior_tol=$posterior_tol posterior_max_cells=$posterior_max_cells max_scenarios=$max_scenarios weight_tol=$weight_tol utility_tol=$utility_tol kelly_tol=$kelly_tol mu_qmc=$mu_qmc mu_chisq_qmc=$mu_chisq_qmc")
println("Gate-0 slow reference (D-083): single-threaded, no incremental/cache; multi-day windows are batch loads.")

# ---------------------------------------------------------------------------
# 主入口：run_gate0_backtest
# ---------------------------------------------------------------------------
println("Running KTrader Gate-0 backtest from $t_start to $t_end...")
res = run_gate0_backtest(mf, el;
                         t_start = t_start, t_end = t_end,
                         mode = mode,
                         seed = seed,
                         S_reference = s_reference,
                         posterior_tol = posterior_tol,
                         posterior_max_cells = posterior_max_cells,
                         max_scenarios = max_scenarios,
                         weight_tol = weight_tol,
                         utility_tol = utility_tol,
                         kelly_tol = kelly_tol,
                         mu_qmc = mu_qmc,
                         mu_chisq_qmc = mu_chisq_qmc)

# ---------------------------------------------------------------------------
# 输出 Gate0BacktestResult 关键字段
# ---------------------------------------------------------------------------
println("\n==========================================================================")
println("KTrader Gate-0 Backtest: $(mf.dates[t_start]) to $(mf.dates[t_end-1]) ($(length(res.days)) daily decisions, $(length(mf.symbols)) symbols)")
println("==========================================================================")

row(n, r) = @printf("%-30s final %6.2fx\n", n, r)

row("Gate-0 Path Kelly (cash feasible)", res.equity[end])
row("Cash Benchmark", res.equity_cash[end])
row("EW Daily Benchmark", res.equity_ew_daily[end])
row("EW Buy-Hold Benchmark", res.equity_ew_buyhold[end])
@printf("EW rebalance gap (D-073: report only, no causal claim): %+.4f\n",
        res.equity_ew_daily[end] - res.equity_ew_buyhold[end])

println("\nDaily records:")
for d in res.days
    dec = d.decision
    @printf("  t=%-4d mode=%-15s M=%-4d cash=%6.3f locked=%6.3f wealth=%7.4f equity=%7.4f\n",
            d.t, dec.mode, dec.M, dec.w_cash, dec.locked_exposure,
            d.wealth_factor, d.equity_after)
end

println("\nConcentration diagnostics (last decision day):")
last_dec = res.days[end].decision
c = last_dec.concentration
@printf("  top_asset=%d top_weight=%5.2f%% risky_weight=%5.2f%% cash_weight=%5.2f%%\n",
        c.top_asset, 100c.top_weight, 100c.risky_weight, 100c.cash_weight)
@printf("  integration_M=%d converged=%s utility_margin_1pct=%s\n",
        c.integration_M, c.converged,
        isnan(c.utility_margin_1pct) ? "NOT COMPUTED" : string(c.utility_margin_1pct))
@printf("  epistemic_variance=%s innovation_variance=%.6g\n",
        isnan(c.epistemic_variance) ? "NOT COMPUTED" : string(c.epistemic_variance),
        c.innovation_variance)

# =============================================================================
# 历史 release 指引（旧线 2.0.0 / 2.0-RC legacy implementation）
# =============================================================================
# 旧线回测（backtest_v1 与其旧参数面 SCENARIOS / ADAPTIVE_SCENARIOS /
# F_FOLDS / RIDGE_ALPHA / ENGINE / DATE_TASKS / …）不再由本入口消费；
# 历史调用形态保留在 git 历史与 README「历史 release」段。当前入口只认
# GATE0_* 参数。旧线代码与测试保持冻结（裁决 H1 / §46），本文件不引用
# 其回测逻辑。
# =============================================================================
