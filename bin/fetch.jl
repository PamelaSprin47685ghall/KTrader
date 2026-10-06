using KTrader
syms = load_universe(joinpath(@__DIR__, "..", "universe.txt"))
b = download_bars(syms)
save_bars(b, joinpath(@__DIR__, "..", "data"))
println(length(b.dates), " days ", b.dates[1], " → ", b.dates[end], " ", length(syms), " symbols")
