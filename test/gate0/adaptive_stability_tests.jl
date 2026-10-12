using Test, Random, LinearAlgebra
module AdaptiveStability
    using Random, LinearAlgebra
    include(joinpath(@__DIR__,"../../src/gate0/geometry.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/modes.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/response.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/posterior.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/innovation.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/kelly.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/quadrature.jl"))
end
using .AdaptiveStability

@testset "false 64→128 convergence cannot certify a jump at 256" begin
    calls=Int[]
    function source(M,rule,rng)
        push!(calls,M)
        X=fill(0.95,M,1)
        if M>128
            X[129:end,1].=1.25
        end
        X
    end
    s=RQMCScenarioSource(source,1,2)
    kw=(; min_scenarios=64,max_scenarios=256,weight_tol=1e-4,
           utility_tol=1e-6,utility_tol_audit=1e-6,kelly_tol=1e-6)
    naive=adaptive_scenario_kelly(s; kw...,confirmation_pairs=1)
    @test naive.M==128
    @test naive.w_cash==1.0
    empty!(calls)
    err=try
        adaptive_scenario_kelly(s;kw...)
        nothing
    catch ex
        ex
    end
    @test err isa ErrorException
    @test occursin("Numerical integration did not converge",sprint(showerror,err))
    @test 256 in calls
    @test_throws ArgumentError adaptive_scenario_kelly(s;
        kw...,confirmation_pairs=0)
end
