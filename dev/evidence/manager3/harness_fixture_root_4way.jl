using Test
@testset "fixture-root 4-way (stdlib only, no KTrader load)" begin
    mk(pairs...) = joinpath("/tmp/fx4", string(pairs...))
    # minimal: legal 9-file tree
    for d in ("minimal","nested","bad","empty"); mkpath(mk(d)); end
    for d in ("minimal","nested")
        open(mk(d,"KTrader.jl"),"w") do io
            println(io,"module KTrader")
            for f in ("data","geometry","numerics","response","residual_oracle","prepare","predict","kelly","backtest")
                println(io,"include(\"$f.jl\")")
                open(mk(d,"$f.jl"),"w") do io2; println(io2,"f_$f() = 1"); end
            end
            println(io,"end")
        end
    end
    open(mk("bad","KTrader.jl"),"w") do io
        println(io,"module KTrader"); println(io,"include(\"dev/probes.jl\")"); println(io,"end")
    end
    # minimal green
    @test filesize(mk("minimal","KTrader.jl")) > 0
    @test count(occursin("include(\"", l) for l in eachline(mk("minimal","KTrader.jl"))) == 9
    # nested green (same structure)
    @test filesize(mk("nested","KTrader.jl")) > 0
    # bad red: contains dev/ include
    s = read(mk("bad","KTrader.jl"),String)
    @test occursin("dev/probes", s)
    # empty red: no KTrader.jl
    @test !isfile(mk("empty","KTrader.jl"))
end
println("# FX4 DONE")
