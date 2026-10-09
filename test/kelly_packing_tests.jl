using KTrader, Test, LinearAlgebra, Random
include(joinpath(@__DIR__,"fixtures","kelly_packing_reference.jl"))

function check_packed_kelly(X,active,tradable,held=nothing)
    snapshot=copy(X); held_snapshot=held===nothing ? nothing : copy(held)
    expected=kelly_indexed_reference(X,active,tradable,held)
    actual=KTrader.scenario_weights(X,active,tradable,held)
    @test isequal(X,snapshot)
    @test isequal(held,held_snapshot)
    @test actual!==held
    @test all(isfinite,actual)
    @test norm(actual-expected,1)<=1e-7
    mask=falses(size(X,2)); mask[active].=true
    free=tradable .& mask; indices=findall(free)
    current=held===nothing ? zeros(size(X,2)) : held
    locked=current .* .!free
    @test actual[.!free]==locked[.!free]
    budget=1-sum(locked)
    if !isempty(indices) && budget>1e-12
        base=KTrader.locked_wealth(X,locked)
        # BOTH candidates are judged on the original indexed scenario law.
        a=KTrader.kelly_certificate(view(X,:,indices),actual[indices];budget,base)
        b=KTrader.kelly_certificate(view(X,:,indices),expected[indices];budget,base)
        @test KTrader.certified(a,1e-8)
        @test KTrader.certified(b,1e-8)
        @test abs(a.objective-b.objective)<=1e-10
    else
        @test actual==current
    end
end

@testset "Packed free scenarios preserve certified Kelly and locked risk" begin
    rng=MersenneTwister(10091840)
    for N in (3,6,12)
        X=exp.(0.025randn(rng,96,N))
        active=collect(1:N)
        check_packed_kelly(X,active,trues(N))
        check_packed_kelly(X,reverse(active),fill(true,N),fill(1/N,N))
        # Positive untradable holdings must contribute scenario wealth.
        tradable=trues(N); tradable[2]=false
        held=fill(0.75/(N-1),N); held[2]=0.25
        check_packed_kelly(X,active,tradable,held)
        check_packed_kelly(X,active,falses(N),held)
        onlyone=falses(N); onlyone[1]=true
        check_packed_kelly(X,active,onlyone,held)
        full_locked=zeros(N); full_locked[2]=1.0
        check_packed_kelly(X,active,tradable,full_locked)
        # Noncontiguous active set; NaN inactive columns are never packed.
        sparse=[1,3]; X[:,2].=NaN
        N>3 && (X[:,4:end].=NaN)
        check_packed_kelly(X,sparse,trues(N))
        check_packed_kelly(view(X,1:2:96,:),sparse,trues(N))
        badheld=zeros(N); badheld[2]=0.2
        @test_throws ErrorException KTrader.scenario_weights(X,sparse,trues(N),badheld)
        invalid=copy(X); invalid[1,1]=NaN
        @test_throws ArgumentError KTrader.scenario_weights(invalid,sparse,trues(N))
    end
end
