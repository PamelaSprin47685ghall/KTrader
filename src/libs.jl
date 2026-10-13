# src/libs.jl —— 只做导入，不含算法；KTrader 与 test/reference_isolation_tests.jl 共用。
# data / geometry / numerics / response / residual_oracle 已抽吸到 KTraderNew：
# KBars, KRulerV1, KNumerics, KResponseV1, KResidualOracle（版本钉在 Project.toml）。
# 这里只把它们的全部名字（含 _xxx 内部名）接回本模块，供余下文件按原名使用。
# ruler 必须来自 KRulerV1（短历史回退与 KGeometry.ruler 不同，见 KTraderNew E5）。
import KBars, KRulerV1, KNumerics, KResponseV1, KResidualOracle
import KGeometry: TAUS, BANDS, BANDCOL, WARMUP, center_of_mass, compute_s_perp,
    fast_s_m, path_basis_1d, principal_sqrt_root, relative_gauge
for M in (KBars, KRulerV1, KNumerics, KResponseV1, KResidualOracle),
        n in names(M; all = true, imported = true)
    s = String(n)
    (startswith(s, '#') || n in (:eval, :include) || n === nameof(M)) && continue
    (isdefined(@__MODULE__, n) || !isdefined(M, n)) && continue
    Core.eval(@__MODULE__, Expr(:import, Expr(:(:), Expr(:., fullname(M)...), Expr(:., n))))
end
