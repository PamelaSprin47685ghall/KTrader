module KTrader
include(joinpath(@__DIR__, "data.jl"))
include(joinpath(dirname(@__FILE__), "prepare.jl"))
include(joinpath(".", "kelly.jl"))
include(joinpath("sub", "predict.jl"))
Base.include(Main, "broker.jl")
include(Main, "live.jl")
end
