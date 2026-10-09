# Locate long/failed dates without restarting a whole backtest. Exact public
# fit_v1 and original full/fold alpha chain, persisted after successful days.
include(joinpath(@__DIR__,"local_panel_stages.jl"))
const EFD_OUT=joinpath(@__DIR__,"evidence","earlier_closure_20261009")
function efd_main()
    length(ARGS)==1 || error("one day index 1:8")
    index=parse(Int,only(ARGS)); 1<=index<=8 || error("index must be 1:8")
    stem="fit_day_"*string(index)
    path=joinpath(EFD_OUT,stem*".jls"); meta=path*".toml"
    !ispath(path) && !ispath(meta) || error("refusing to overwrite diagnostic")
    BLAS.set_num_threads(6); sources=panel_sources()
    for (p,h) in PANEL_INPUT_HASHES
        panel_digest(joinpath(PANEL_ROOT,p))==h || error("CSV changed")
    end
    bars=KTrader.load_bars(joinpath(PANEL_ROOT,"data"))
    tmax=size(bars.adj,1)-8
    signal=KTrader.signal_prices(bars)[1:tmax,:]
    t=tmax-8+index-1
    warm=fill((1.0,1.0),4)
    if index>1
        p=joinpath(EFD_OUT,"fit_day_"*string(index-1)*".jls")
        m=TOML.parsefile(p*".toml"); bytes=read(p)
        bytes2hex(sha256(bytes))==m["sha256"] && length(bytes)==m["bytes"] || error("prior artifact changed")
        m["sources"]==sources && m["inputs"]==PANEL_INPUT_HASHES || error("prior source/input mismatch")
        prior=deserialize(IOBuffer(bytes))
        prior.sources==sources && prior.t==t-1 || error("wrong preceding day")
        copyto!(warm,prior.warm)
        println("previous_sha256=",m["sha256"])
    end
    initial=copy(warm); history=KTrader.PriceHistoryCache(signal)
    println("BEGIN index=",index," date=",bars.dates[t]," prefix=1:",t,
        " default EB/F3 BLAS6; warm=",initial," source=",sources["src/response.jl"]); flush(stdout)
    count=Ref(0)
    sink=(event,bucket,stamp)->begin
        if event==:start && bucket==:OOF_fit
            count[]+=1
        end
        println(event," ",bucket," fold=",count[]," ns=",stamp); flush(stdout)
    end
    timing=KTrader.DecisionTiming(zeros(length(KTrader.TIMING_BUCKETS)),sink,nothing)
    o=@timed KTrader.fit_v1(view(signal,1:t,:);F_folds=3,history_cache=history,
        alpha_initial=warm,workspace=KTrader.FitWorkspace(),timing)
    println("END index=",index," seconds=",o.time," bytes=",o.bytes," gc=",o.gctime,
        " compile=",o.compile_time," warm_out=",warm); flush(stdout)
    for (b,v) in zip(KTrader.TIMING_BUCKETS,timing.seconds)
        println("bucket ",b,"=",v)
    end
    sources==panel_sources() || error("source changed during diagnostic")
    for (p,h) in PANEL_INPUT_HASHES
        panel_digest(joinpath(PANEL_ROOT,p))==h || error("CSV changed during diagnostic")
    end
    !ispath(path) && !ispath(meta) || error("output appeared")
    serialize(path,(;sources,t,index,date=bars.dates[t],warm,initial,model=o.value,
        seconds=o.time,bytes=o.bytes,gc=o.gctime,compile=o.compile_time,timings=timing.seconds))
    digest=panel_digest(path)
    open(meta,"w") do io
        TOML.print(io,Dict("sources"=>sources,"inputs"=>PANEL_INPUT_HASHES,
            "sha256"=>digest,"bytes"=>filesize(path),"date"=>string(bars.dates[t])))
    end
    println("SAVED ",basename(path)," sha256=",digest); flush(stdout)
end
efd_main()
