module KTrader
include(joinpath(@__DIR__, "dev", "probes.jl"))
Base.include(Main, joinpath("dev", "more_probes.jl"))
f = "data.jl"
include(f)
end
