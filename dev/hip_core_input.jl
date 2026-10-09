# Export actual first-day alpha-cache blocks and the original CPU core result.
# This is preparation plus ONE spectrum, zero additional posterior fits.
include(joinpath(@__DIR__,"local_panel_stages.jl"))
using Statistics
const HCI_OUT=joinpath(@__DIR__,"evidence","earlier_closure_20261009")
function hci_main()
    isempty(ARGS) || error("fixed actual N65 core export")
    output=joinpath(HCI_OUT,"hip_core_input.bin")
    ispath(output) && error("refusing to overwrite GPU input")
    BLAS.set_num_threads(6); sources=panel_sources()
    p=joinpath(HCI_OUT,"fit_day_1.jls")
    meta=TOML.parsefile(p*".toml"); bytes=read(p)
    bytes2hex(sha256(bytes))==meta["sha256"] && meta["sources"]==sources || error("model identity mismatch")
    day=deserialize(IOBuffer(bytes)); bytes=nothing
    for (p,h) in PANEL_INPUT_HASHES
        panel_digest(joinpath(PANEL_ROOT,p))==h || error("input changed")
    end
    bars=KTrader.load_bars(joinpath(PANEL_ROOT,"data"))
    prices=KTrader.signal_prices(bars)
    prep=KTrader.prepare_reference(view(prices,1:day.t,:);F_folds=3)
    spectrum=KTrader.ridge_spectrum(nothing,prep.stats.full_xx,prep.stats.full_xy)
    alpha=day.model.resp.alpha_rel
    cache=KTrader.conditioned_alpha_cache(spectrum,prep.stats.full_yy,prep.n_res,alpha;gauge=KTrader.relative_gauge(prep.N))
    cores=KTrader.build_jacobian_cores(cache)
    KTrader._fill_jacobian_cores!(cores,cache) # explicitly warm the reusable CPU kernel
    observations=NamedTuple[]
    for _ in 1:5
        o=@timed KTrader._fill_jacobian_cores!(cores,cache)
        push!(observations,(;seconds=o.time,bytes=o.bytes,compile=o.compile_time))
    end
    d=size(cache.bcat,1); nc=length(cache.blocks); rk=div(size(cache.bcat,2),nc)
    println("actual date=",day.date," alpha=",alpha," d=",d," rank=",rk," constraints=",nc," pairs=",length(cores))
    println("CPU_BLAS6_reused_core_median_seconds=",median(x.seconds for x in observations),
        " allocated_bytes=",median(x.bytes for x in observations)," compile_sum=",sum(x.compile for x in observations))
    sources==panel_sources() || error("source changed")
    open(output,"w") do io
        write(io,codeunits("KTRJC001"))
        write(io,UInt64[d,rk,nc,length(cores)])
        write(io,cache.bcat); write(io,cache.bwcat)
        foreach(A->write(io,A),cores)
    end
    digest=panel_digest(output)
    open(output*".toml","w") do io
        TOML.print(io,Dict("sources"=>sources,"inputs"=>PANEL_INPUT_HASHES,"sha256"=>digest,
            "bytes"=>filesize(output),"alpha"=>alpha,"date"=>string(day.date),
            "cpu_median_seconds"=>median(x.seconds for x in observations)))
    end
    println("GPU_INPUT sha256=",digest," bytes=",filesize(output),"; no GPU backend or solver change")
end
hci_main()
