using Test, Random, LinearAlgebra, Statistics, Dates
using KTrader

@testset "KTrader V1.0 Complete Rigorous Suite" begin
    include("v1_constitutional_tests.jl")
    include("v1_invariant_tests.jl")
    include("v1_convergence_tests.jl")
    include("data_tests.jl")
    include("execution_tests.jl")
end
