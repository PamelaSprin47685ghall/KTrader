# gate0/KTraderGate0.jl —— 旧入口兼容层（只做重导出，不含任何算法）
#
# Gate-0 正文在 KTraderNew；本仓库只依赖组装层 KTraderGate0（版本钉在根 Project.toml）。
# 组装层只导出公开名字；bin/ 与 dev/ 还用内部名（G._ForecastCache 之类），所以这里在
# 加载组装层之后，从每个名字的所有者库把它接回来（跳过各库 CONTRACTS 里的契约桩）。
# 本模块在 Main 里以 KTraderGate0 的名字出现，旧脚本的 `using .KTraderGate0` / `G = KTraderGate0` 照旧可用。
#
# 用法：include(joinpath(<repo>, "gate0", "KTraderGate0.jl")); using .KTraderGate0
# 解除条件：bin/ 与 dev/ 全部改为直接 using 组装层后删除本文件。
module KTraderGate0Compat

import KTraderGate0 as Assembly

const LIBS = Assembly.LIBS

for M in LIBS
    Core.eval(@__MODULE__, :(const $(nameof(M)) = $M))
end
for M in LIBS
    stubs = isdefined(M, :CONTRACTS) ? M.CONTRACTS : ()
    for n in names(M; all = true)
        s = String(n)
        (startswith(s, '#') || n in (:eval, :include, :CONTRACTS) || n === nameof(M) || n in stubs) && continue
        isdefined(@__MODULE__, n) && continue
        Core.eval(@__MODULE__, Expr(:import, Expr(:(:), Expr(:., :., nameof(M)), Expr(:., n))))
    end
end
for n in names(Assembly)
    n === nameof(Assembly) && continue
    isdefined(@__MODULE__, n) || Core.eval(@__MODULE__, Expr(:import, Expr(:(:), Expr(:., :Assembly), Expr(:., n))))
    Core.eval(@__MODULE__, Expr(:export, n))
end

end # module KTraderGate0Compat

const KTraderGate0 = KTraderGate0Compat
