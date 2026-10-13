# src/libs.jl —— 只做导入，不含算法。
# v1 正文全部在 KTraderNew；本仓库只依赖组装层 KTraderV1（版本钉在 Project.toml）。
# 各库单独加载时契约名是空桩，所以先加载组装层，再从各库取名字：
#   - 每个名字（含 _xxx 内部名）取自它的所有者库，跳过各库 CONTRACTS 里的契约桩；
#   - ruler 取自 KRulerV1（KGeometry.ruler 是 gate0 版，短历史回退不同，见 KTraderNew E5），
#     所以 KGeometry 排在最后，只补其余几何名。
import KTraderV1
for M in (KTraderV1.REEXPORT..., KTraderV1.KGeometry)
    Core.eval(@__MODULE__, :(const $(nameof(M)) = $M))
    stubs = isdefined(M, :CONTRACTS) ? M.CONTRACTS : ()
    for n in names(M; all = true)
        s = String(n)
        (startswith(s, '#') || n in (:eval, :include, :CONTRACTS) || n === nameof(M) || n in stubs) && continue
        isdefined(@__MODULE__, n) && continue
        Core.eval(@__MODULE__, Expr(:import, Expr(:(:), Expr(:., :., nameof(M)), Expr(:., n))))
    end
end
