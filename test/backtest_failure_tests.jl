using KTrader, Test, Dates, LinearAlgebra

@testset "Backtest handoff preserves the producer failure after draining" begin
    for T in (Int,Any)
        cause=ArgumentError("deliberate producer failure")
        failures=Any[cause,nothing]
        ch=Channel{T}(2)
        put!(ch,11); put!(ch,12); close(ch)
        @test KTrader._backtest_take(ch,failures,1,11)==11
        @test KTrader._backtest_take(ch,failures,1,12)==12
        caught=try KTrader._backtest_take(ch,failures,1,13); nothing catch e; e end
        @test caught===cause
        @test failures[1]===cause
        normal=Channel{T}(1); close(normal)
        e=try KTrader._backtest_take(normal,failures,2,19); nothing catch e; e end
        @test e isa ErrorException
        @test occursin("block 2 ended before decision 19",sprint(showerror,e))
        # Publish/close from another task to test the real wait/wakeup route.
        waiting=Channel{T}(0); pending=Any[nothing]
        @sync begin
            @async begin
                yield()
                pending[1]=cause
                close(waiting)
            end
            e=try KTrader._backtest_take(waiting,pending,1,1); nothing catch e; e end
            @test e===cause
        end
    end
end

@testset "Actual backtest producer error reaches caller and spool is cleaned" begin
    prices=ones(340,2)
    bars=KTrader.Bars(collect(Date(2020,1,1).+Day.(0:339)),["A","B"],prices,prices)
    # Invalid ridge is rejected INSIDE the worker fit, not the scheduler's
    # upfront topology checks. Exercise both channel payload variants.
    for adaptive in (false,true)
        mktempdir() do parent
            original=BLAS.get_num_threads()
            e=try
                KTrader.backtest_v1(bars;from=bars.dates[end-1],F_folds=3,
                    ridge_alpha=-1.0,S=64,adaptive,date_tasks=1,blas_threads=1,_spool_parent=parent)
                nothing
            catch failure
                failure
            end
            @test e isa ArgumentError
            @test !(e isa InvalidStateException)
            @test isempty(readdir(parent))
            @test BLAS.get_num_threads()==original
        end
    end
end
