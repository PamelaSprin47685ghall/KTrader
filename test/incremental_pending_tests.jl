using KTrader, Test, Random
include(joinpath(@__DIR__,"fixtures","incremental_pending_reference.jl"))

@testset "Pending filter column views: exact old arithmetic across mask runs" begin
    for N in (2,5,17)
        core=pending_core_fixture(;N)
        z_before=deepcopy(core.zsum_hist)
        rows_before=length(core.rows)
        for k in (1,4,5,8,9,16,17,32,33,64,65,128,129,256,257,296,333,370,420)
            old=pending_slice_reference(core,k)
            new=KTrader.compute_pending(core,k)
            if old===nothing
                @test new===nothing
            else
                @test old.regs==new.regs
                @test isequal(old.v,new.v)
                @test all(m->size(m)==(N,KTrader.REGIME_CHANNELS),new.v)
                new.v[1][1,1]+=1.0
                @test isequal(KTrader.compute_pending(core,k).v,old.v)
            end
        end
        @test isequal(core.zsum_hist,z_before)
        @test length(core.rows)==rows_before
    end
end
