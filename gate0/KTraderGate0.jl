# gate0/KTraderGate0.jl —— 旧入口兼容层（只做重导出，不含任何算法）
#
# Gate-0 正文已抽吸到 KTraderNew 的 13 个库（版本钉在根 Project.toml 的
# [sources]）。本模块把它们拼回旧名字 `KTraderGate0`，让 bin/ 与 dev/ 的
# 旧调用（`G.single_day_decision`、`G._ForecastCache` 之类内部名）不改语义。
#
# 规则：按依赖拓扑序遍历各库的全部名字（含未导出的 `_xxx`），先到先得地
# import 进来；各库的公开名字再原样 export。同名只可能是下游 import 上游的
# 同一绑定，先到的就是上游本体。
#
# 用法：include(joinpath(<repo>, "gate0", "KTraderGate0.jl")); using .KTraderGate0
# 解除条件：bin/ 与 dev/ 全部改为直接 using 各库后删除本文件。
module KTraderGate0

using KMarket, KMarketIO, KGeometry, KModes, KResponse, KPosterior, KOOF,
      KInnovation, KPredictive, KKelly, KQuadrature, KDriver, KBacktest

const LIBS = (KMarket, KMarketIO, KGeometry, KModes, KResponse, KPosterior, KOOF,
              KInnovation, KPredictive, KKelly, KQuadrature, KDriver, KBacktest)

const _SKIP = (:eval, :include)

for M in LIBS, n in names(M; all=true, imported=true)
    s = String(n)
    (startswith(s, '#') || n in _SKIP || n === nameof(M)) && continue
    isdefined(@__MODULE__, n) && continue
    isdefined(M, n) || continue
    Core.eval(@__MODULE__, Expr(:import, Expr(:(:), Expr(:., fullname(M)...), Expr(:., n))))
end

for M in LIBS, n in names(M)
    n === nameof(M) && continue
    Core.eval(@__MODULE__, Expr(:export, n))
end

end # module KTraderGate0
