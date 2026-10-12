# Flat-ridge certificate (user adjudication 2026-10-12;
# docs/KELLY_ADAPTIVE_CERTIFICATE_DRIVEN.md §FR). Synthetic source whose two
# risky assets are near-duplicates: the Kelly optimum is a flat ridge, the
# primary split jitters across layers (A fails) while utility is flat.
push!(LOAD_PATH,joinpath(@__DIR__,"..","..","gate0pkg"))
using KTraderGate0Precompiled, Test, Random, Distributions
const G=KTraderGate0Precompiled.KTraderGate0

function ridge_source(eta)
    gen=function(M,rule,mu_rng)
        P=G.rule_points(rule,M)
        z1=quantile.(Normal(),clamp.(P[:,1],1e-12,1-1e-12))
        z2=quantile.(Normal(),clamp.(P[:,2],1e-12,1-1e-12))
        g1=exp.(0.004 .+ 0.08 .* z1)
        hcat(g1, g1 .* exp.(eta .* z2))
    end
    G.RQMCScenarioSource(gen,2,2)
end
kw=(;min_scenarios=64,max_scenarios=8192,weight_tol=1e-10,utility_tol=1e-6,
     utility_tol_audit=1e-6,kelly_tol=1e-6)

@testset "flat-ridge certificate" begin
    src=ridge_source(1e-4)
    r=G.adaptive_scenario_kelly(src;kw...)
    @test r.certificate_path===:flat_ridge_utility
    @test r.converged
    @test r.certificate_A > kw.weight_tol                 # A reported, not waived silently
    h=r.history[end]
    @test h.certificate_path===:flat_ridge_utility
    @test h.ridge_utility_spread <= kw.utility_tol
    @test h.ridge_utility_spread_audit <= kw.utility_tol
    @test abs(r.certificate_B) <= kw.utility_tol && r.certificate_B_audit <= kw.utility_tol
    # disabled => original contract: no weights, loud failure
    @test_throws ErrorException G.adaptive_scenario_kelly(src;kw...,flat_ridge_certificate=false)
    # utility gap above tol => rejected (B / spread fail)
    @test_throws ErrorException G.adaptive_scenario_kelly(src;kw...,utility_tol=1e-15,utility_tol_audit=1e-15)
    # weight-stable source (single asset, A converges) keeps the original path
    s1=G.RQMCScenarioSource((M,rule,rng)->begin
        P=G.rule_points(rule,M); reshape(exp.(0.004 .+ 0.08 .* quantile.(Normal(),clamp.(P[:,1],1e-12,1-1e-12))),:,1) end,1,2)
    r1=G.adaptive_scenario_kelly(s1;kw...,weight_tol=1e-3,max_scenarios=65536)
    @test r1.certificate_path===:weight_stable
end
