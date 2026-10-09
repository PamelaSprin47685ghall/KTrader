# ============================================================================
# src/gate0/market.jl —— MarketFacts / Eligibility / free / locked / benchmark
# ============================================================================
#
# Gate-0 纠偏新线 Step 1（施工图 docs/GATE0_IMPLEMENTATION_PLAN.md §2.1/§2.2
# 与 Step 1 施工段；终审裁决 H1/H4/H7）：
# 四 mask 拆分——observation（D-012）/ model admission（D-013）/
# trade eligibility（D-014/D-018）/ physical execution（D-015）；
# free = E^trade ∧ T^exec（D-016）；held/locked 单独处理（D-017）；
# benchmark universe 外生化（D-020）。
#
# 旧线对照（施工图 §7.2 的语义混用面，本文件是其改写方案）：
# - 旧 src/data.jl:2-5 把「有真实 bar」与「可交易」混写为同义（"i.e. the
#   asset was tradable"）——本文件以 MarketFacts.observed 与 Eligibility
#   三 mask 在字段名与 docstring 层面拆开这三者。
# - 旧 src/backtest.jl:242-245 的 free = active ∩ bar 把 model admission
#   混进可交易集合——本文件的 free 只含 trade_eligible ∧ executable。
#
# 本文件不引用旧线 KTrader（裁决 H1：主 module 与新线互不 include）；
# 与旧 Bars 的字段对应转换由 from_bars_arrays 裸矩阵桥承担（施工图 §1.3）。

# 文件内 export（Wave 4 执行轮补齐——与 geometry/modes/kelly 等后续交付
# 文件的自含模式统一；原为骨架层 export 模式，测试的独立加载路径经
# `using .Gate0Module` 拿不到非导出名。与骨架层 export 同名单、重复声明
# 无害）。
export MarketFacts, Eligibility, from_bars_arrays, signal_prices,
    model_admitted, trade_eligible, effective_bars_252, executable,
    free, locked, benchmark_universe

"""
    MarketFacts(dates, symbols, close, adj, observed)

价格历史的市场事实层（D-010/D-012）。

- `dates::Vector{Date}`：交易日历。
- `symbols::Vector{String}`：universe 符号。
- `close`/`adj::Matrix{Float64}`：账户 marking 序列（上市前 NaN、缺 bar
  处 carry forward）——只用于 marking，不进入理论 signal。
- `observed::BitMatrix`：O_{t,i}。唯一语义：「日期 t 资产 i 有真实市场
  bar」（D-012）。它不表示可交易性、不表示模型 admission、不表示风控
  许可——政策性语义一律由 `Eligibility` 承担。市场事实不是策略配置。

构造不变量（fail loudly，绝不静默改写 observed——「因价格缺失/不想交易/
历史不足/模型不 admission/风控而改观测」同属 D-012 禁止的写入方向）：
- 各字段维度一致（T×N，T ≥ 1，N ≥ 1）；
- `observed[t,i] == true` 蕴含 `close[t,i]` 与 `adj[t,i]` 均 finite
  （观测蕴含真实价格；旧 4 参构造 `bar = isfinite.(rawclose) .&
  isfinite.(rawadj)`（src/data.jl:28）的推导逻辑由本不变量承接）；
- `observed[t,i] == false` 处价格任意（NaN = 上市前；有限值 = carry
  forward marking，合法——marking 与 signal 的分界见 `signal_prices`）。

只读约定（D-012 四禁的落地形态）：`observed` 构造后只读——本 module 不
提供任何针对 `MarketFacts` 的 mutation 接口，构造函数是唯一入口，且构造
时对全部字段做防御性 copy（外部持有的数组与本对象解耦）。
"""
struct MarketFacts
    dates::Vector{Date}
    symbols::Vector{String}
    close::Matrix{Float64}
    adj::Matrix{Float64}
    observed::BitMatrix
    function MarketFacts(dates::Vector{Date}, symbols::Vector{String},
                         close::Matrix{Float64}, adj::Matrix{Float64},
                         observed::BitMatrix)
        T = length(dates)
        N = length(symbols)
        T >= 1 || error("MarketFacts: dates 为空，市场事实层至少需要一行")
        N >= 1 || error("MarketFacts: symbols 为空，至少需要一个资产")
        size(close) == (T, N) ||
            error("MarketFacts: close 应为 $(T)×$(N)，实际 $(size(close))")
        size(adj) == (T, N) ||
            error("MarketFacts: adj 应为 $(T)×$(N)，实际 $(size(adj))")
        size(observed) == (T, N) ||
            error("MarketFacts: observed 应为 $(T)×$(N)，实际 $(size(observed))")
        for j in 1:N, t in 1:T
            if observed[t, j]
                isfinite(close[t, j]) ||
                    error("MarketFacts: observed[$(t),$(j)] = true 但 close = $(close[t, j]) 非 finite（D-012：观测蕴含真实价格）")
                isfinite(adj[t, j]) ||
                    error("MarketFacts: observed[$(t),$(j)] = true 但 adj = $(adj[t, j]) 非 finite（D-012：观测蕴含真实价格）")
            end
        end
        new(copy(dates), copy(symbols), copy(close), copy(adj), copy(observed))
    end
end

"""
    from_bars_arrays(dates, symbols, close, adj, observed) -> MarketFacts

数据桥（施工图 §1.3；裁决 H1「互不 include」的直接推论）：从旧线 `Bars`
的字段对应转换构造 `MarketFacts`。调用方（未来的 gate0 backtest 驱动器）
自己从旧线 `Bars` 提取 `dates/symbols/close/adj/bar` 数组传入；本 module
不 `using KTrader`、不引用旧 `Bars` 类型。

对应关系：`observed` ≡ 旧 `Bars.bar`（逐元素一致）。语义修正（D-012）：
旧 docstring「i.e. the asset was tradable」的观测/可交易混写在新线废除
——`observed` 只表示物理观测事实，可交易性由 `Eligibility` 承担。
"""
from_bars_arrays(dates::Vector{Date}, symbols::Vector{String},
                 close::Matrix{Float64}, adj::Matrix{Float64},
                 observed::BitMatrix) =
    MarketFacts(dates, symbols, close, adj, observed)

"""
    signal_prices(mf::MarketFacts) -> Matrix{Float64}

D-012 语义下的理论 signal 恢复：`ifelse.(observed, adj, NaN)`——
`observed = false` 处价格为 NaN（缺失观测），绝不把 carry-forward marking
冒充为 signal 价格（等价旧线 src/data.jl:37 的 signal_prices；新线以
`MarketFacts` 方法重新拥有，施工图 §2.1）。

不交易 ≠ 零收益、缺 bar ≠ 零收益：缺失就是缺失（NaN），由上层（Step 3
的 mode producer）按 observation mask 处理。
"""
signal_prices(mf::MarketFacts)::Matrix{Float64} = ifelse.(mf.observed, mf.adj, NaN)

"""
    Eligibility(model_admitted, trade_eligible, executable)
    Eligibility(mf::MarketFacts)                # 默认政策（E^trade 全 true）
    Eligibility(mf::MarketFacts, trade_mask)    # 自定义 E^trade 政策

政策层：三 mask 独立（D-013/D-014/D-015）。

- `model_admitted::BitMatrix`：A^model_{t,i}（判定函数 `model_admitted`）。
- `trade_eligible::BitMatrix`：E^trade_{t,i}——是否允许当日新建/增加风险
  仓位。政策入口：默认 `trade_eligible(mf)` 全 true；252 有效交易日规则
  实例 `effective_bars_252(mf)`。
- `executable::BitMatrix`：T^exec_{t,i}（判定函数 `executable`）。

构造纪律（D-020 benchmark 外生性的根基）：三个 mask 只能由
`MarketFacts.observed` 与用户政策参数派生——任何把 posterior/模型输出
写回 Eligibility 的路径都是类型错误。裸构造对三个输入做同形校验与
防御性 copy；便捷构造从 `MarketFacts` 派生 admitted/executable，并接收
调用方选定的 E^trade 政策 mask。
"""
struct Eligibility
    model_admitted::BitMatrix
    trade_eligible::BitMatrix
    executable::BitMatrix
    function Eligibility(model_admitted::BitMatrix, trade_eligible::BitMatrix,
                         executable::BitMatrix)
        (size(model_admitted) == size(trade_eligible) == size(executable)) ||
            error("Eligibility: 三个 mask 尺寸必须一致，实际 $(size(model_admitted)) / $(size(trade_eligible)) / $(size(executable))")
        (size(model_admitted, 1) >= 1 && size(model_admitted, 2) >= 1) ||
            error("Eligibility: mask 不能为空")
        new(copy(model_admitted), copy(trade_eligible), copy(executable))
    end
end

"""
    Eligibility(mf::MarketFacts)

默认政策构造：model_admitted / executable 从 `MarketFacts` 派生，
trade_eligible 取无限制政策（全 true，`trade_eligible(mf)`）。
"""
Eligibility(mf::MarketFacts) =
    Eligibility(model_admitted(mf), trade_eligible(mf), executable(mf))

"""
    Eligibility(mf::MarketFacts, trade_mask)

自定义 E^trade 政策构造：`trade_mask` 是调用方选定的政策实例（如
`effective_bars_252(mf)` 的返回值），尺寸须与 `mf.observed` 一致。
"""
function Eligibility(mf::MarketFacts, trade_mask::BitMatrix)
    size(trade_mask) == size(mf.observed) ||
        error("Eligibility: trade 政策 mask 应为 $(size(mf.observed))，实际 $(size(trade_mask))")
    return Eligibility(model_admitted(mf), trade_mask, executable(mf))
end

"""
    model_admitted(mf::MarketFacts) -> BitMatrix

D-013：A^model[t,j] = 1 当且仅当 prefix 1:t 中资产 j 至少存在一条有效
daily return——即存在 s ∈ 2:t 使 observed[s-1,j] 与 observed[s,j] 均为
true 且 adj[s-1,j] 与 adj[s,j] 均 finite（return 属于相邻对的第二日 s）。

性质：
- 对 t 单调不减（prefix 语义）；
- 全 NaN dummy 资产（无任何观测）全列 false，构造性排除（§11/D-060）；
- 单日 missing 不回退已 admitted 的资产（prefix 内更早的 return 仍在）；
- IPO 资产在首条有效 return 之前不 admitted：首观测日 s₀ 自身没有
  return（需要相邻对 (s₀, s₀+1)），故首个 admitted 日是 s₀+1。这是
  D-013 的 return-pair 口径；它与 252 规则的 O 计数口径（H4）分属两个
  mask 的两种裁决字面，不得混同或「统一口径」。

弱证据由 posterior 表达，不用 MINROWS 类门槛（D-013）。
"""
function model_admitted(mf::MarketFacts)::BitMatrix
    T, N = size(mf.observed)
    admitted = falses(T, N)
    if T >= 2
        for j in 1:N
            first_return = 0  # 0 = prefix 中尚未找到有效 return
            for t in 2:T
                if mf.observed[t-1, j] && mf.observed[t, j] &&
                   isfinite(mf.adj[t-1, j]) && isfinite(mf.adj[t, j])
                    first_return = t
                    break
                end
            end
            if first_return > 0
                admitted[first_return:T, j] .= true
            end
        end
    end
    return admitted
end

"""
    trade_eligible(mf::MarketFacts) -> BitMatrix

D-014 默认政策：无限制——全 true。E^trade 回答「当日是否允许新建/增加
风险仓位」；无政策约束的基准情形即无限制。252 有效交易日规则的政策
实例见 `effective_bars_252`。
"""
trade_eligible(mf::MarketFacts)::BitMatrix = trues(size(mf.observed))

"""
    effective_bars_252(mf::MarketFacts) -> BitMatrix

D-018（D-014 政策入口之一）：252 有效交易日规则——
`trade_eligible[t,j] = 1 ⟺ Σ_{τ≤t} observed[τ,j] ≥ 252`。

口径（裁决 H4）：计数按 O 累计（观测 bar 数——不是 return 数、不是
日历行数）；累计第 252 个有效 bar 的当日解除限制（当日 E^trade = 1；
第 251 个有效 bar 的当日仍为 0）。252 是本政策实例的固定常数（D-018
裁决原文），不是可调策略参数。

本 mask 只约束新建/增加风险仓位，不触碰 `MarketFacts.observed`——
252 日以前模型仍然可以看见该资产的真实价格、该资产仍可作为预测信息
（裁决书 §6 的 PONY 场景：可见、可预测、不可开仓）。
"""
function effective_bars_252(mf::MarketFacts)::BitMatrix
    T, N = size(mf.observed)
    eligible = falses(T, N)
    for j in 1:N
        count = 0
        for t in 1:T
            count += mf.observed[t, j]
            eligible[t, j] = count >= 252
        end
    end
    return eligible
end

"""
    executable(mf::MarketFacts) -> BitMatrix

D-015 理论 backtest 的简单形式：T^exec = O_{t,i}——当天有真实市场 bar
即物理可执行。live/broker 层的报价级可执行性（bid/ask/last > 0，
src/broker.jl:136 tradable 语义）是独立的 T^exec 供给源，与本函数各自
拥有、不互相校验（docs/MODEL_LEDGER.md §5 的分层）。

返回 `observed` 的独立副本：调用方对返回值的任何修改不会污染
`MarketFacts.observed`（mask 分离的防御边界）。
"""
executable(mf::MarketFacts)::BitMatrix = copy(mf.observed)

"""
    free(el::Eligibility) -> BitMatrix
    free(el::Eligibility, t::Int) -> BitVector

D-016：新建风险仓位的自由集合 free = E^trade ∧ T^exec（整矩阵 / 第 t
日行）。model_admitted 不参与 free——「当前模型是否预测该资产」是另一
个问题（admission 只影响模型坐标空间，D-013）；本定义直接纠正旧线
free = active ∩ bar（src/backtest.jl:242-245）把 admission 混进可交易
集合的语义违规。
"""
free(el::Eligibility)::BitMatrix = el.trade_eligible .& el.executable

free(el::Eligibility, t::Int)::BitVector =
    el.trade_eligible[t, :] .& el.executable[t, :]

"""
    locked(held::AbstractVector{Bool}, el::Eligibility, t::Int) -> BitVector

D-017：当日已持有（w^held_j > 0；持仓权重到 Bool 的判定由调用方负责）
但 free_j = 0 的资产集合：locked = held ∧ ¬free。

语义钉死：locked 资产是 locked risk，**不是 cash**——其 scenario wealth
贡献必须保留在 Kelly 的 base_s（Step 2 落地；语义沿旧 src/kelly.jl:137-148
locked_wealth：held 资产乘其预测 gross return，缺预测律即 error）。本
函数只负责判定，财富语义由 Kelly 层拥有。held 是当日持仓状态（输入），
不是 Eligibility 的字段。
"""
function locked(held::AbstractVector{Bool}, el::Eligibility, t::Int)::BitVector
    N = size(el.executable, 2)
    length(held) == N ||
        error("locked: held 长度 $(length(held)) 与资产数 $(N) 不符")
    # BitVector(...) 显式构造：held（Vector{Bool} 或 BitVector）与
    # .!free(el,t)（BitVector）的广播结果类型不承担返回注解的契约。
    return BitVector(held .& .!free(el, t))
end

"""
    benchmark_universe(el::Eligibility) -> BitMatrix
    benchmark_universe(el::Eligibility, t::Int) -> BitVector

D-020：等权 benchmark 的候选集合 = E^trade ∧ T^exec，与 model_admitted
完全无关——实验组（模型的 admission）不能决定对照组（benchmark）有哪
些资产。函数签名只接收 `Eligibility`（其构造纪律保证 mask 只依赖
`MarketFacts.observed` 与用户政策），不接收任何 model / active /
posterior 概念。

与 `free` 数学同式（D-016 与 D-020 是两条独立裁决），但 owner 语义不同：
`free` 是实验组 Kelly 的「新建仓位自由集合」，本函数是外生对照组的
「benchmark 候选集合」。二者独立实现、独立测试——合并实现会让 D-020
的外生性测试依赖 D-016 的实现路径。
"""
benchmark_universe(el::Eligibility)::BitMatrix =
    el.trade_eligible .& el.executable

benchmark_universe(el::Eligibility, t::Int)::BitVector =
    el.trade_eligible[t, :] .& el.executable[t, :]
