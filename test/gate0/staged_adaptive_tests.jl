using Test, Random, LinearAlgebra
module StagedAdaptive
    using Random, LinearAlgebra
    include(joinpath(@__DIR__,"../../src/gate0/geometry.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/modes.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/response.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/posterior.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/innovation.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/kelly.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/quadrature.jl"))
end

@testset "staged certification refuses early equal-corner plateau" begin
    calls=Tuple{Int,UInt64}[]
    function make_scenarios(M,rule,rng)
        push!(calls,(M,rule.seed))
        X=Matrix{Float64}(undef,M,2)
        for i in 1:M
            if i<=128
                X[i,1],X[i,2]=1.08,1.01
            else
                X[i,1],X[i,2]=0.80,1.18
            end
        end
        X
    end
    src=StagedAdaptive.RQMCScenarioSource(make_scenarios,2,2)
    @test_throws ErrorException StagedAdaptive.adaptive_scenario_kelly(src;
           min_scenarios=64,max_scenarios=128,
           weight_tol=1e-4,utility_tol=1e-6,kelly_tol=1e-6)
    empty!(calls)
    result=StagedAdaptive.adaptive_scenario_kelly(src;
           min_scenarios=64,max_scenarios=4096,
           weight_tol=1e-4,utility_tol=1e-6,kelly_tol=1e-6)
    @test result.converged
    @test result.M>=1024
    @test result.history[1].A<=1e-4
    @test result.history[2].A>0.5
    @test all(h -> h.A<=1e-4 && abs(h.B_opt)<=1e-6 &&
                   h.B_audit<=1e-6, result.history[end-1:end])
    @test result.w_risky[2]>0.9
    @test isapprox(result.w_cash+sum(result.w_risky),1.0;atol=1e-9)
    @test StagedAdaptive.cash_kelly_certified(result.certificate_C;tol=1e-6)
    opts=[c[1] for c in calls if c[2]==result.rule_seed]
    @test opts==[2^k for k in 6:trailing_zeros(result.M)]
end
