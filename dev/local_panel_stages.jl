# One bounded stage per command, on the COMPLETE checked-in CSV prefix.
# prepare -> solve -> decision; never a backtest, hidden warm-up or repeat fit.
using KTrader, LinearAlgebra, Random, Dates, Serialization, SHA, TOML

const PANEL_ROOT = normpath(joinpath(@__DIR__, ".."))
const PANEL_EVIDENCE = joinpath(@__DIR__, "evidence", "local_panel_20261008")
const PANEL_INPUT_HASHES = Dict(
    "data/adj.csv" => "e0c25e04a079947b1c5fbc3879703b76d9d35c20300dccd969fedcb268a29322",
    "data/close.csv" => "4b9b17c8d9dbf81f4274187e82c13f43ab094cbddc9d1b3be804b03a4b55aced")

panel_digest(path) = open(sha256, path) |> bytes2hex
function panel_sources()
    Dict(joinpath("src",f) => panel_digest(joinpath(PANEL_ROOT,"src",f))
         for f in sort(readdir(joinpath(PANEL_ROOT,"src"))) if endswith(f,".jl"))
end
function panel_measure(f, name)
    println("BEGIN ",name); flush(stdout)
    obs=@timed f()
    println("END ",name," seconds=",obs.time," allocated_bytes=",obs.bytes,
            " gc_seconds=",obs.gctime)
    # Native accounting is reported as provided, never inferred from modes.
    for key in (:compile_time,:recompile_time)
        hasproperty(obs,key) && println(name," ",key,"=",getproperty(obs,key))
    end
    flush(stdout)
    obs.value
end
function panel_refuse_existing(name)
    for suffix in (".jls",".toml")
        ispath(joinpath(PANEL_EVIDENCE,name*suffix)) && error("refusing to overwrite $name$suffix")
    end
end
function panel_save(name,record)
    panel_refuse_existing(name)
    path=joinpath(PANEL_EVIDENCE,name*".jls")
    serialize(path,record)
    digest=panel_digest(path)
    open(joinpath(PANEL_EVIDENCE,name*".toml"),"w") do io
        TOML.print(io,Dict("file"=>basename(path),"sha256"=>digest,
                         "bytes"=>filesize(path),"source_hashes"=>record.sources,
                         "input_hashes"=>PANEL_INPUT_HASHES))
    end
    println("artifact=",basename(path)," sha256=",digest," bytes=",filesize(path)); flush(stdout)
end
function panel_load(name,sources)
    meta=TOML.parsefile(joinpath(PANEL_EVIDENCE,name*".toml"))
    meta["file"]==name*".jls" || error("artifact filename mismatch")
    meta["source_hashes"]==sources || error("sources changed since preceding stage")
    meta["input_hashes"]==PANEL_INPUT_HASHES || error("input record mismatch")
    # Deserialize the SAME bytes that are hashed; no second artifact read.
    data=read(joinpath(PANEL_EVIDENCE,meta["file"]))
    length(data)==meta["bytes"] && bytes2hex(sha256(data))==meta["sha256"] || error("artifact bytes changed")
    record=deserialize(IOBuffer(data))
    record.sources==sources || error("artifact source identity mismatch")
    println("consumed=",meta["file"]," sha256=",meta["sha256"]); flush(stdout)
    record
end
function panel_timing()
    sink=(event,bucket,stamp)->begin
        println("bucket ",event," ",bucket," monotonic_ns=",stamp); flush(stdout)
    end
    KTrader.DecisionTiming(zeros(length(KTrader.TIMING_BUCKETS)),sink,nothing)
end
function panel_main()
    length(ARGS)==1 && only(ARGS) in ("prepare","solve","decision") || error("phase: prepare | solve | decision")
    phase=only(ARGS)
    name=phase=="prepare" ? "prepared" : phase=="solve" ? "model" : "decision"
    panel_refuse_existing(name) # before loading inputs or starting any fit
    BLAS.set_num_threads(6)
    sources=panel_sources()
    println("phase=",phase," Julia=",VERSION," CPU=",Sys.CPU_NAME,
            " Julia_threads=",Threads.nthreads()," BLAS_threads=",BLAS.get_num_threads(),
            " BLAS=",BLAS.get_config())
    println("single numerical-cold call; no hidden warm-up; not steady-state throughput")
    if phase=="prepare"
        for (p,h) in PANEL_INPUT_HASHES
            panel_digest(joinpath(PANEL_ROOT,p))==h || error("CSV input changed: $p")
            println("input=",p," sha256=",h)
        end
        bars=panel_measure("load_bars") do
            KTrader.load_bars(joinpath(PANEL_ROOT,"data"))
        end
        signal=KTrader.signal_prices(bars)
        # Last bar is withheld: decision at T-1, no future price in preparation.
        t=size(signal,1)-1
        @assert length(bars.dates)==size(signal,1) && length(bars.symbols)==65
        @assert issorted(bars.dates) && allunique(bars.dates)
        @assert all(x->isnan(x)||(isfinite(x)&&x>0),signal)
        @assert isequal(isfinite.(signal),bars.bar)
        println("loader=load_bars -> signal_prices -> prefix[1:",t,"]",
                " full_rows=",size(signal,1)," decision_date=",bars.dates[t],
                " observed_prefix=",count(isfinite,view(signal,1:t,:)))
        timing=panel_timing()
        prep=panel_measure("prepare_reference") do
            KTrader.prepare_reference(view(signal,1:t,:);F_folds=3,timing)
        end
        @assert prep.N==65 && prep.P_features==910 && prep.T==t && prep.F_folds==3
        @assert prep.ws_owner===nothing
        for (p,h) in PANEL_INPUT_HASHES
            panel_digest(joinpath(PANEL_ROOT,p))==h || error("CSV changed during prepare: $p")
        end
        record=(;sources,prep,date=bars.dates[t],symbols=bars.symbols,
                 tradable=copy(bars.bar[t,:]),timings=copy(timing.seconds))
    elseif phase=="solve"
        previous=panel_load("prepared",sources)
        prep=previous.prep; timing=panel_timing()
        println("prepared T=",prep.T," N=",prep.N," P=",prep.P_features," F=",prep.F_folds)
        model=panel_measure("solve_once") do
            KTrader.solve(prep;timing) # default independent EB; same production owner
        end
        for (bucket,seconds) in zip(KTrader.TIMING_BUCKETS,timing.seconds)
            println("timing ",bucket,"=",seconds)
        end
        record=(;sources,model,date=previous.date,symbols=previous.symbols,
                 tradable=previous.tradable,timings=copy(timing.seconds))
    else
        previous=panel_load("model",sources); model=previous.model
        X=panel_measure("scenarios_S300") do
            KTrader.generate_scenarios_v1(model;S=300,rng=MersenneTwister(1))
        end
        weights=panel_measure("Kelly") do
            KTrader.scenario_weights(X,model.active_indices,previous.tradable,nothing)
        end
        free=[j for j in model.active_indices if previous.tradable[j]]
        certificate=KTrader.kelly_certificate(view(X,:,free),weights[free])
        @assert KTrader.certified(certificate,1e-8)
        println("Kelly certificate=",certificate)
        record=(;sources,date=previous.date,symbols=previous.symbols,X,weights,certificate)
    end
    panel_sources()==sources || error("runtime source changed during stage")
    panel_measure("persist_"*name) do
        panel_save(name,record)
    end
end
if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    panel_main()
end
