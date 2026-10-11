using Test, Dates
module MarketIOEnv
    using Dates
    include(joinpath(@__DIR__, "../../src/gate0/market.jl"))
    include(joinpath(@__DIR__, "../../src/gate0/market_io.jl"))
end
@testset "market CSV adapter: alignment and signal/mark separation" begin
    mktempdir() do dir
        close_text = "date,A,B\n2024-01-02,100,NaN\n2024-01-03,NaN,50\n2024-01-04,110,51\n"
        adj_text = "date,A,B\n2024-01-02,90,NaN\n2024-01-03,NaN,45\n2024-01-04,99,46\n"
        write(joinpath(dir, "close.csv"), close_text)
        write(joinpath(dir, "adj.csv"), adj_text)
        mf = MarketIOEnv.load_market_facts(dir)
        @test mf.symbols == ["A", "B"]
        @test size(mf.observed) == (3, 2)
        @test mf.adj[2, 1] == 90.0
        @test !mf.observed[2, 1]
        @test isnan(MarketIOEnv.signal_prices(mf)[2, 1])
        @test isnan(mf.adj[1, 2])
        write(joinpath(dir, "adj.csv"), replace(adj_text, "2024-01-03" => "2024-01-05"))
        @test_throws ErrorException MarketIOEnv.load_market_facts(dir)
        write(joinpath(dir, "adj.csv"), replace(adj_text, "date,A,B" => "date,B,A"))
        @test_throws ErrorException MarketIOEnv.load_market_facts(dir)
        write(joinpath(dir, "adj.csv"), replace(adj_text, ",99," => ",0,"))
        @test_throws ErrorException MarketIOEnv.load_market_facts(dir)
        write(joinpath(dir, "adj.csv"), replace(adj_text, ",99," => ",Inf,"))
        @test_throws ErrorException MarketIOEnv.load_market_facts(dir)
    end
end
