# src/libs.jl —— 只做导入，不含算法。
# v1 正文全部在 KTraderNew；本仓库只依赖组装层 KTraderV1（经私有 registry KTraderRegistry）。
# 名字的来源顺序：
#   1. 组装层自己定义的名字优先（它把 geom / RULER 等构造数据绑进各库消息，例如 ruler、
#      build_X_rel_stacked、fit_response_operator、initialize_inference）；
#   2. 其余名字（含 _xxx 内部名）取自所有者库，跳过各库 CONTRACTS 里的契约桩；
#   3. KGeometry 最后，只补剩下的几何常量。
import KTraderV1
const _ASSEMBLY = KTraderV1
let skip(M, n) = (s = String(n); startswith(s, '#') || n in (:eval, :include, :CONTRACTS, :LIBS, :REEXPORT) ||
                  n === nameof(M) || n in (isdefined(M, :CONTRACTS) ? M.CONTRACTS : ()))
    for n in names(KTraderV1; all = true)
        (skip(KTraderV1, n) || isdefined(@__MODULE__, n) || n === :KTraderV1) && continue
        Core.eval(@__MODULE__, Expr(:import, Expr(:(:), Expr(:., :KTraderV1), Expr(:., n))))
    end
    for M in KTraderV1.LIBS
        isdefined(@__MODULE__, nameof(M)) || Core.eval(@__MODULE__, :(const $(nameof(M)) = $M))
    end
    for M in (KTraderV1.REEXPORT..., KTraderV1.KGeometry), n in names(M; all = true)
        (skip(M, n) || isdefined(@__MODULE__, n)) && continue
        Core.eval(@__MODULE__, Expr(:import, Expr(:(:), Expr(:., :., nameof(M)), Expr(:., n))))
    end
end
