using KTrader, Random, Statistics, LinearAlgebra
include(joinpath(@__DIR__,"..","test","fixtures","incremental_pending_reference.jl"))

function pending_micro_measure(f)
    f() # explicit specialization warm-up, excluded
    observations=[@timed(f()) for _ in 1:7]
    (;median_seconds=median(o.time for o in observations),
      minimum_seconds=minimum(o.time for o in observations),
      median_bytes=median(o.bytes for o in observations))
end

function pending_micro_main()
    BLAS.set_num_threads(1)
    core=pending_core_fixture(;N=65,T=420)
    old=()->pending_slice_reference(core,420)
    new=()->KTrader.compute_pending(core,420)
    a=old(); b=new()
    @assert a.regs==b.regs && isequal(a.v,b.v)
    println("synthetic N65/T420 recurring masks; one pending-filter call; exact values")
    println("before_A ",pending_micro_measure(old))
    println("after_B ",pending_micro_measure(new))
    println("after_B2 ",pending_micro_measure(new))
    println("before_A2 ",pending_micro_measure(old))
end
pending_micro_main()
