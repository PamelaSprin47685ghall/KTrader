# =============================================================================
# test/gate0/runtests.jl —— KTraderGate0 统一测试入口（唯一入口第一阶段）
# =============================================================================
#
# 本文件是 Gate-0 测试的统一入口：按顺序 include test/gate0/ 下的 12 个
# 测试文件。每个测试文件是独立 module（Gate0MarketEnv / Gate0ModesEnv /
# Gate0ResponseEnv / Gate0PosteriorEnv / Gate0OofEnv / Gate0PrequentialEnv /
# Gate0InnovationEnv / Gate0PredictiveEnv / Gate0KellyCashTests /
# Gate0QuadratureTests / Gate0DriverEnv / Gate0BacktestEnv），各自 include
# 自己的裸文件链，include 顺序无跨文件依赖冲突。
#
# 运行（归 DevOps，Engineer 不执行命令）：
#   julia --startup-file=no --project=. test/gate0/runtests.jl
#
# ENV 控制：
#   GATE0_MODULES —— 模块选择（逗号分隔序号，默认 "all"）。例：
#       GATE0_MODULES="1,2,3,4,5,6,7" 跑模块 1..7；
#       GATE0_MODULES="8,9,10,11,12" 跑 8..12；GATE0_MODULES="all" 跑全部。
#       12 模块累积超单命令预算，需分组调度（如 "1..7" 与 "8..12" 两组）。
#   BACKTEST_SCENARIO —— backtest_tests.jl 的场景选择。默认 "1"（5-day
#       end-to-end，最便宜场景）。全量 7 场景累积超单命令预算，需分组
#       调度：BACKTEST_SCENARIO="1,2" / "3,4" / "5,6,7" 分别运行（同一
#       进程编译一次摊薄冷启动）。**默认只跑场景 1**——multi-day ×
#       prequential 全量跑需单独调度，不在本统一入口的默认预算内。
#       ENV 必须在 include backtest_tests.jl **之前**设置——该文件的
#       `const _BT_SCENARIO = get(ENV, ...)` 在 include 时求值。
#
# 超时边界（必须明说）：本入口默认跑 11 个模块 + backtest 场景 1；全量
# backtest 7 场景、或全量 prequential 多日窗口，单命令 ≤60s/RSS2048 预算
# 内不可行——需按上述分组单独调度。
#
# 结构约束（为什么用包裹 module 而非裸 include）：
# 每个测试文件的顶层结构是「module XxxEnv … end」+ 顶层 `using .XxxEnv` +
# 顶层 `@testset`。若在 Main 直接 include，各测试文件的 `using .XxxEnv`
# 会把同名导出（MarketFacts / ruler / signal_prices 等——都来自同一批
# gate0 裸文件）叠加到 Main，导致同名函数/类型 ambiguous、任何调用报错。
# 因此每个测试文件包进一个独立 module（`Gate0Runtests_<name>`）：测试文件
# 里的 `module XxxEnv` 成为该包裹 module 的嵌套 module、`using .XxxEnv` 与
# `@testset` 都在包裹 module 作用域内执行，各测试文件互不污染 Main。
# 嵌套 module 声明在 Julia 中合法（module 内可定义 module）。
#
# 模块选择实现（为什么用 @eval）：Julia 的 `module` 定义必须是顶层语句，
# 不能嵌套在 `if` 块内（"module expression not at top level"）。因此每个
# 包裹 module 用 `@eval` 在 if 内动态定义——@eval 把 module 定义到当前
# module（Main）的顶层作用域，if 控制是否执行。这是合法的模块选择机制。
#
# 本文件是测试入口，不是生产代码。
# =============================================================================

using Test

# ---------------------------------------------------------------------------
# 模块选择（GATE0_MODULES）：逗号分隔模块序号，默认 "all"。
#   例：GATE0_MODULES="1,2,3,4,5,6,7" 跑模块 1..7；
#       GATE0_MODULES="8,9,10,11,12" 跑 8..12；GATE0_MODULES="all" 跑全部。
# 与 BACKTEST_SCENARIO 机制同风格（backtest_tests.jl 的 ENV 选择）。
# ---------------------------------------------------------------------------
const _GATE0_MODULES_SPEC = get(ENV, "GATE0_MODULES", "all")
const _GATE0_MODULES = _GATE0_MODULES_SPEC == "all" ?
    Set(string.(1:12)) : Set(split(_GATE0_MODULES_SPEC, ','))

# @eval 内 @__DIR__ 指向 Main 目录，需在 @eval 前捕获 runtests.jl 所在目录
const _TDIR = @__DIR__

# ---------------------------------------------------------------------------
# backtest_tests.jl 的 ENV 机制：默认场景 1（5-day end-to-end，最便宜）。
# 用户显式设置时尊重用户值（透传）。必须在 include backtest_tests.jl 之前
# 设置（其 `const _BT_SCENARIO = get(ENV, ...)` 在 include 时求值）。
# ---------------------------------------------------------------------------
if !haskey(ENV, "BACKTEST_SCENARIO")
    ENV["BACKTEST_SCENARIO"] = "1"
end

# ---------------------------------------------------------------------------
# 按依赖从轻到重顺序 include（各自独立包裹 module，无跨文件依赖冲突）：
# market → modes → response → posterior → oof → prequential → innovation →
# predictive → kelly_cash → quadrature → driver → backtest
# ---------------------------------------------------------------------------
if "1" in _GATE0_MODULES
    @eval module Gate0Runtests_Market
        println("\n=== include test/gate0/market_tests.jl ===")
        include(joinpath(Main._TDIR, "market_tests.jl"))        # Step 1 四 mask
    end
end

if "2" in _GATE0_MODULES
    @eval module Gate0Runtests_Modes
        println("\n=== include test/gate0/modes_tests.jl ===")
        include(joinpath(Main._TDIR, "modes_tests.jl"))         # Step 3 mode 坐标
    end
end

if "3" in _GATE0_MODULES
    @eval module Gate0Runtests_Response
        println("\n=== include test/gate0/response_tests.jl ===")
        include(joinpath(Main._TDIR, "response_tests.jl"))      # Step 4/5 full block response
    end
end

if "4" in _GATE0_MODULES
    @eval module Gate0Runtests_Posterior
        println("\n=== include test/gate0/posterior_tests.jl ===")
        include(joinpath(Main._TDIR, "posterior_tests.jl"))     # Step 8 slow full-posterior
    end
end

if "5" in _GATE0_MODULES
    @eval module Gate0Runtests_Oof
        println("\n=== include test/gate0/oof_tests.jl ===")
        include(joinpath(Main._TDIR, "oof_tests.jl"))           # Step 9 OOF full-mode folds
    end
end

if "6" in _GATE0_MODULES
    @eval module Gate0Runtests_Prequential
        println("\n=== include test/gate0/prequential_tests.jl ===")
        include(joinpath(Main._TDIR, "prequential_tests.jl"))   # P0-1 strict prequential
    end
end

if "7" in _GATE0_MODULES
    @eval module Gate0Runtests_Innovation
        println("\n=== include test/gate0/innovation_tests.jl ===")
        include(joinpath(Main._TDIR, "innovation_tests.jl"))    # Step 10/11/12 vector innovation
    end
end

if "8" in _GATE0_MODULES
    @eval module Gate0Runtests_Predictive
        println("\n=== include test/gate0/predictive_tests.jl ===")
        include(joinpath(Main._TDIR, "predictive_tests.jl"))    # Step 13 PredictiveLaw
    end
end

if "9" in _GATE0_MODULES
    @eval module Gate0Runtests_KellyCash
        println("\n=== include test/gate0/kelly_cash_tests.jl ===")
        include(joinpath(Main._TDIR, "kelly_cash_tests.jl"))    # Step 2 cash feasible set
    end
end

if "10" in _GATE0_MODULES
    @eval module Gate0Runtests_Quadrature
        println("\n=== include test/gate0/quadrature_tests.jl ===")
        include(joinpath(Main._TDIR, "quadrature_tests.jl"))    # Step 14 adaptive RQMC
    end
end

if "11" in _GATE0_MODULES
    @eval module Gate0Runtests_Driver
        println("\n=== include test/gate0/driver_tests.jl ===")
        include(joinpath(Main._TDIR, "driver_tests.jl"))        # Step 15 端到端单日 driver
    end
end

if "12" in _GATE0_MODULES
    @eval module Gate0Runtests_Backtest
        println("\n=== include test/gate0/backtest_tests.jl ===")
        include(joinpath(Main._TDIR, "backtest_tests.jl"))      # Step 15 多日回测（默认场景 1）
    end
end