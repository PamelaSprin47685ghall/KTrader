# 数据层工具（双线共用；不含模型决策）：下载/保存行情，供 Gate-0
# 当前入口（bin/backtest.jl，经数据桥读 data/）与历史 release 线共用。
using KTrader
syms = load_universe(joinpath(@__DIR__, "..", "universe.txt"))
b = download_bars(syms)
save_bars(b, joinpath(@__DIR__, "..", "data"))
println(length(b.dates), " days ", b.dates[1], " → ", b.dates[end], " ", length(syms), " symbols")
