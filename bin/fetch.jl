# DEPRECATED：本入口已迁到 KTraderNew（https://github.com/PamelaSprin47685ghall/KTraderNew）。
# 新用法：在 KTraderNew 根目录 `julia bin/ktrader.jl fetch`。本文件只是薄转发：
# 默认数据目录 / universe 仍指向本仓库的 data/ 与 universe.txt，其余参数（环境变量、ARGS）原样传给 KTraderCLI。
using KTraderCLI
root = joinpath(@__DIR__, "..")
haskey(ENV, "KTRADER_DATA_DIR") || haskey(ENV, "GATE0_DATA_DIR") || (ENV["KTRADER_DATA_DIR"] = joinpath(root, "data"))
haskey(ENV, "KTRADER_UNIVERSE") || (ENV["KTRADER_UNIVERSE"] = joinpath(root, "universe.txt"))
exit(KTraderCLI.main(["fetch"; ARGS]))
