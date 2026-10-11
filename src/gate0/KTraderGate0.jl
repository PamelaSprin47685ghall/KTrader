# ============================================================================
# src/gate0/KTraderGate0.jl —— Gate-0 纠偏新线 module 骨架（Wave 2, Step 1）
# ============================================================================
#
# 身份（裁决 H1，docs/GATE0_MANAGER_ADJUDICATIONS.md）：独立顶层 module
# `KTraderGate0`，与旧 `src/`（KTrader 2.0 RC legacy）互不 include——旧线
# 冻结为 historical fixture（裁决书 §46 / D-082），本新线是 Gate-0 纠偏的
# slow reference 实现（D-082/D-083：单线程、无 incremental、无 scheduler、
# 无 GPU、无 workspace）。
#
# 规范来源（优先级，裁决 I1）：AGENTS.md Gate-0 裁决书（D-001~D-097）>
# docs/GATE0_MANAGER_ADJUDICATIONS.md（终审裁决 A~I）>
# docs/GATE0_IMPLEMENTATION_PLAN.md（施工图 Step 0-16）。
#
# 依赖纪律：骨架层只 using Dates（market.jl 的需求）；geometry.jl 与
# kelly.jl 按文件自含模式声明各自依赖（geometry: LinearAlgebra；kelly:
# Convex/Clarabel/LinearAlgebra/Random/Printf 中前三个实际使用）——include
# 时这些 using 在本 module 作用域执行。kelly.jl 的 Convex/Clarabel 是
# reference solver 依赖（Project.toml 已登记，旧线同源）；除此之外不引入
# FFTW / Distributions / HTTP 等旧线重依赖（D-082/D-083 reference 定位）。
#
# include 列表（Wave 2 集成收口，依赖链顺序）：
#   1. market.jl   —— Step 1 四 mask（O/A/E/T），无内部依赖
#   2. geometry.jl —— 常量 / ruler / Helmert gauge / path_basis_1d /
#                     principal_sqrt_root（domain_mode_basis 定义于此）
#   3. modes.jl    —— Step 3 mode 坐标；消费 geometry 的 BANDS/TAUS/
#                     BANDCOL/path_basis_1d/domain_mode_basis（故必须后于
#                     geometry include；domain_mode_basis 的 export 声明在
#                     本文件，定义在 geometry.jl——include 后名字已定义）
#   4. response.jl —— Step 4/5 固定 ridge full block response reference；
#                     消费 modes 的 mode_field / cumulative_path_coordinates /
#                     mode_basis_design 与 geometry 的 fast_s_m /
#                     compute_s_perp（故必须后于 modes include）；文件内
#                     自带 export（build_mode_problem / fit_full_block_ridge
#                     / predict_mode / block_views）与 using LinearAlgebra
#   5. posterior.jl —— Step 8 slow full-posterior reference；自身只依赖
#                     LinearAlgebra/Distributions/Random（E 由调用方传入，
#                     不反向依赖 producer 层）；文件内自带 using 与 export
#                     14 名
#   6. oof.jl      —— Step 9 OOF full-mode folds；消费 posterior.jl 的
#                     同作用域件（SufficientStats/sufficient_stats/
#                     lambda_diag/log_evidence/log_prior_d035a/
#                     adaptive_quadrature_2d/_fit_node/ResponsePosterior/
#                     ProprietyCertificate——_fit_node 未 export，同 module
#                     include 链可见；故必须后于 posterior include）；
#                     文件内自带 export 5 名
#   7. innovation.jl —— Step 10-12 vector innovation reference；消费 modes
#                     的 risk_domain_mode_basis（故必须后于 modes include）；
#                     文件内自带 using Random, LinearAlgebra 与 export 16 名
#   8. predictive.jl —— Step 13 PredictiveLaw 组装；消费 posterior 的
#                     predict_mu/draw_mu/ResponsePosterior 与 innovation 的
#                     InnovationState/draw_innovation（故必须后于二者
#                     include）；文件内自带 using Random, LinearAlgebra 与
#                     export 2 名（PredictiveLawResult / predictive_law）
#   9. kelly.jl    —— Step 2 cash Kelly；独立于前述文件，文件内自带
#                     using Convex / Clarabel / LinearAlgebra
# 10. quadrature.jl —— Step 14 adaptive RQMC（nested Sobol + Owen
#                     scramble + audit replicate + 三证书）；消费 posterior
#                     的 draw_mu/ResponsePosterior、innovation 的
#                     InnovationState/z_pool_rows/mp_sqrt_factors、kelly 的
#                     cash_kelly/locked_wealth_gate0（故必须后于三者
#                     include）；文件内自带 using Random, LinearAlgebra
#                     与 export 7 名
# 11. driver.jl   —— Step 15 端到端单日决策 driver（single_day_decision
#                     / equal_weight_benchmark）；消费全部前述模块的
#                     公共接口 + market 的 MarketFacts/Eligibility（依赖
#                     链末端）；文件内自带 using Random, LinearAlgebra,
#                     Statistics 与 export 3 名
# 12. backtest.jl —— Step 15 多日段：sequential 多日回测 driver
#                     （run_gate0_backtest——D-089 短窗口 5→20 天级；
#                     消费 driver 的 single_day_decision/Gate0DayDecision
#                     与非导出常量 _DRIVER_DEFAULT_SEED——同 module 链
#                     可见，故必须后于 driver include）；文件内自带
#                     using Random, LinearAlgebra, Statistics 与 export 3 名
# 后续：Gate-0 全部 Step（1-15）已收口（单日 driver + 多日 backtest）。
#
# export 面收口（90 名，无重名）：market 11 名（本骨架层 export）+
# geometry 11 名（文件内自带 export）+ modes 6 名（文件内自带 export）+
# response 4 名（文件内自带 export）+ posterior 14 名（文件内自带
# export）+ oof 5 名（文件内自带 export）+ innovation 20 名（文件内
# 自带 export；P0-2 新增 initial_mesh/adaptive_d_quadrature、P0-3 新增
# residual_rank/rank_sufficient）+ predictive 2 名（文件内自带
# export）+ quadrature 7 名（文件内自带 export）+ driver 3 名（文件内
# 自带 export）+ backtest 3 名（文件内自带 export）+ kelly 4 名（本
# 骨架层 export；kelly_cash_tests.jl 消费面——cash_kelly_inputs 为
# 内部校验函数，不导出）。

module KTraderGate0

using Dates

include("market.jl")
include("geometry.jl")
include("modes.jl")
include("response.jl")
include("posterior.jl")
include("oof.jl")
include("innovation.jl")
include("predictive.jl")
include("kelly.jl")
include("quadrature.jl")
include("driver.jl")
include("backtest.jl")

# market.jl（Step 1 四 mask）公共 API
export MarketFacts, Eligibility, from_bars_arrays, signal_prices,
    model_admitted, trade_eligible, effective_bars_252, executable,
    free, locked, benchmark_universe

# kelly.jl（Step 2 cash Kelly）公共 API
export cash_kelly, locked_wealth_gate0, cash_kelly_certificate,
    cash_kelly_certified

end # module KTraderGate0
