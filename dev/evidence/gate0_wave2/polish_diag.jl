# 诊断脚本：隔离 _cash_kelly_newton_polish 的行为（Wave 2 集成，Case C 数据）。
# 问题：cash_kelly 的证书数值与引入 polish 前逐位一致，说明 polish 未生效。
# 本脚本直接以「解析最优 + 1.1e-7 扰动」（模仿 Clarabel 实测误差量级）
# 喂给 polish，观察它是否收敛到机器精度驻点。
module PolishDiag
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "kelly.jl"))
end
using .PolishDiag: cash_kelly_certificate, _cash_kelly_newton_polish,
    cash_kelly_certified

X = reshape([1.2, 0.85], 2, 1)
b = zeros(2)
# 解析最优 (5/6, 1/6)；加 1.1e-7 扰动（Clarabel 实测误差量级与方向未知，
# 此处取对称扰动作代表）
w_val = [5 / 6 + 1.1e-7, 1 / 6 - 1.1e-7]
println("input  w = ", w_val)

c0 = cash_kelly_certificate(X, w_val[1:1], w_val[2]; base=b, budget=1.0)
println("input  cert: kkt=", c0.kkt_residual, "  gap=", c0.objective_gap,
        "  certified=", cash_kelly_certified(c0))

wp = _cash_kelly_newton_polish(X, b, w_val, 1.0)
if wp === nothing
    println("polish returned NOTHING")
else
    println("polished w = ", wp)
    c1 = cash_kelly_certificate(X, wp[1:1], wp[2]; base=b, budget=1.0)
    println("polished cert: kkt=", c1.kkt_residual, "  gap=", c1.objective_gap,
            "  certified=", cash_kelly_certified(c1))
end
