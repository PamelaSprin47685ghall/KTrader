# CPU-only release evidence audit; no fit/backtest/GPU calls or package writes.
using SHA, TOML, Dates, Serialization, LinearAlgebra, Test
const C20_ROOT=normpath(joinpath(@__DIR__,".."))
const C20_OUT=joinpath(@__DIR__,"evidence","cpu20_acceptance_20261009")
const C20_QDIR=joinpath(@__DIR__,"evidence","earlier_closure_20261009","qualification")
const C20_QHASH="31e1c48729147d5022a0b5f894c23afece80cd0bb9a596161fd7619c359e3f17"
const C20_INPUTS=Dict(
    "data/adj.csv"=>"e0c25e04a079947b1c5fbc3879703b76d9d35c20300dccd969fedcb268a29322",
    "data/close.csv"=>"4b9b17c8d9dbf81f4274187e82c13f43ab094cbddc9d1b3be804b03a4b55aced")
const C20_WINDOW_HASHES=(
    "99cf04c814ebfbfe888edb32be15216f5b32fb6738fab442f114902a1c05e8cc",
    "9b92105494e982686ca91a1f1f2f75d014287888ce214b910e458e449fdff6b8")
c20_hash(p)=bytes2hex(open(sha256,p))
c20_require(ok,message)=ok ? nothing : throw(ArgumentError(message))

function c20_receipt(r,entries,digest)
    c20_require(get(r,"passed",false)===true,"receipt not passed")
    c20_require(get(r,"snapshot",nothing)==digest,"wrong source snapshot")
    c20_require(get(r,"entries",nothing)==entries,"wrong or partial entries")
    nothing
end
function c20_sources(expected)
    for (p,h) in expected
        full=joinpath(C20_ROOT,p)
        c20_require(isfile(full) && c20_hash(full)==h,"qualified file changed/missing: $p")
    end
    # Source/test additions must not escape the previous coverage inventory.
    for dir in ("src","test"), (root,_,files) in walkdir(joinpath(C20_ROOT,dir)), f in files
        endswith(f,".jl") || endswith(f,".jls") || continue
        p=relpath(joinpath(root,f),C20_ROOT)
        c20_require(haskey(expected,p),"new unqualified production/test file: $p")
    end
    nothing
end
function c20_load(path,digest)
    bytes=read(path)
    c20_require(bytes2hex(sha256(bytes))==digest,"artifact byte identity changed: $path")
    deserialize(IOBuffer(bytes)) # only known local bytes after authentication
end
function c20_success_log(path)
    text=read(path,String)
    footers=collect(eachmatch(r"scoped_run: rc=(\d+) elapsed=(\d+)s deadline=(\d+) killed=(\d+).*rss_peak=(\d+)MiB",text))
    c20_require(length(footers)==1,"missing/ambiguous resource footer: $path")
    rc,elapsed,deadline,killed,rss=parse.(Int,only(footers).captures)
    c20_require(rc==0 && killed==0 && elapsed<=deadline<=45 && rss<=2048,
                "command did not pass its original resource scope: $path")
    (;elapsed,rss)
end
function c20_main()
    length(ARGS)==1 && only(ARGS) in ("audit","status") || error("action: audit | status")
    action=only(ARGS); output=joinpath(C20_OUT,"acceptance.toml")
    action=="audit" && ispath(output) && error("refusing to overwrite CPU acceptance")
    qpath=joinpath(C20_QDIR,"qualification_snapshot.toml")
    c20_require(c20_hash(qpath)==C20_QHASH,"qualification snapshot changed")
    q=TOML.parsefile(qpath); c20_sources(q["sources"])
    runtime=Dict(k=>v for (k,v) in q["sources"] if startswith(k,"src/"))
    standard=q["standard_files"]; groups=q["groups"]
    entries=vcat(collect(values(groups))...)
    c20_require(length(standard)==41 && length(groups)==22,"unexpected qualified inventory")
    c20_require(Set(first(split(x,"::")) for x in entries)==Set(standard),"incomplete standard-file coverage")
    c20_require(length(entries)==length(unique(entries)),"duplicate group entry")
    evidence=Dict{String,String}()
    for (name,targets) in groups
        path=joinpath(C20_QDIR,"pass_"*name*".toml")
        receipt=TOML.parsefile(path)
        c20_receipt(receipt,targets,C20_QHASH)
        c20_require(get(receipt,"opt_level",nothing)==(name=="numerics" ? 1 : 2),"unreported test optimization profile")
        evidence[relpath(path,C20_ROOT)]=c20_hash(path)
    end
    for (p,h) in C20_INPUTS
        c20_require(c20_hash(joinpath(C20_ROOT,p))==h,"CSV input changed")
    end
    windows=Any[]; earlier=joinpath(@__DIR__,"evidence","earlier_closure_20261009")
    for i in 1:2
        receiptpath=joinpath(earlier,"verified_"*string(i)*".toml")
        receipt=TOML.parsefile(receiptpath)
        c20_require(receipt["passed"] && receipt["sources"]==runtime,"wrong window verification receipt")
        c20_require(receipt["sha256"]==C20_WINDOW_HASHES[i],"wrong window artifact")
        c20_require(receipt["producer_command_rc"]==(i==1 ? 124 : 0),"historical producer status changed")
        log=joinpath(earlier,"verify"*string(i)*"_cpu20.log"); c20_success_log(log)
        w=c20_load(joinpath(earlier,"window_"*string(i)*".jls"),C20_WINDOW_HASHES[i])
        c20_require(w.sources==runtime && w.inputs==C20_INPUTS,"window source/input mismatch")
        push!(windows,w)
        for p in (receiptpath,log)
            evidence[relpath(p,C20_ROOT)]=c20_hash(p)
        end
    end
    # The second producer completed within the guard, unlike recovered run1.
    producer=c20_success_log(joinpath(earlier,"window2.log"))
    evidence[relpath(joinpath(earlier,"window2.log"),C20_ROOT)]=c20_hash(joinpath(earlier,"window2.log"))
    # Preserve authenticated individual chronological fits as independent
    # references; do not deserialize or recompute their posterior here.
    for i in 1:8
        p=joinpath(earlier,"fit_day_"*string(i)*".jls"); meta=TOML.parsefile(p*".toml")
        c20_require(meta["sources"]==runtime && meta["inputs"]==C20_INPUTS,"chronological fit identity changed")
        c20_require(filesize(p)==meta["bytes"] && c20_hash(p)==meta["sha256"],"chronological fit bytes changed")
        evidence[relpath(p*".toml",C20_ROOT)]=c20_hash(p*".toml")
    end
    recentpath=joinpath(@__DIR__,"evidence","release_freeze_20261009","repaired_latest.jls")
    recenthash="f41ff40fadd14e41d7321dbfd8e896a3876a5e805c2e3c9ae9dfe272bbdf25fa"
    recent=c20_load(recentpath,recenthash)
    c20_require(recent.sources==runtime && recent.inputs==C20_INPUTS,"recent-window identity changed")
    recentlog=joinpath(dirname(recentpath),"repaired_latest_loaded.log")
    c20_success_log(recentlog); evidence[relpath(recentlog,C20_ROOT)]=c20_hash(recentlog)
    r1=windows[1].measured.result; r2=windows[2].measured.result
    weight_delta=maximum(sum(abs.(r1.weights-r2.weights);dims=2))
    return_delta=maximum(abs.(r1.ret-r2.ret))
    @testset "CPU acceptance audit: saved outputs and fail-closed receipts" begin
        @test r1.dates==r2.dates
        @test first(r1.dates)==Date(2026,9,10) && last(r1.dates)==Date(2026,9,21)
        @test isempty(intersect(r1.dates,recent.result.dates))
        @test first(recent.result.dates)==Date(2026,9,22) && last(recent.result.dates)==Date(2026,10,1)
        @test weight_delta<=1e-7
        @test isapprox(r1.ret,r2.ret;atol=1e-12,rtol=1e-10)
        @test r1.weights_ew==r2.weights_ew && r1.locked_days==r2.locked_days
        for w in windows
            @test w.measured.compile==0.0 && w.measured.recompile==0.0
            @test size(w.measured.result.weights)==(8,65)
            @test w.measured.result.scenario_counts==fill(300,8)
        end
        @test recent.compile==0.0
        @test size(recent.result.weights)==(8,65)
        @test recent.result.scenario_counts==fill(300,8)
        good=Dict("passed"=>true,"snapshot"=>C20_QHASH,"entries"=>["test.jl"])
        @test isnothing(c20_receipt(good,["test.jl"],C20_QHASH))
        @test_throws ArgumentError c20_receipt(merge(good,Dict("passed"=>false)),["test.jl"],C20_QHASH)
        @test_throws ArgumentError c20_receipt(merge(good,Dict("snapshot"=>"wrong")),["test.jl"],C20_QHASH)
        @test_throws ArgumentError c20_receipt(good,["test.jl","missing.jl"],C20_QHASH)
        @test_throws ArgumentError c20_receipt(Dict{String,Any}(),["test.jl"],C20_QHASH)
    end
    # Check qualified files again before certifying this local delivery.
    c20_sources(q["sources"])
    for (p,h) in C20_INPUTS
        c20_require(c20_hash(joinpath(C20_ROOT,p))==h,"CSV changed during audit")
    end
    report=Dict(
        "milestone"=>"Path Kelly 2.0 CPU",
        "cpu_engineering_acceptance_passed"=>true,
        "gpu_release"=>"2.1","gpu_blocks_cpu20"=>false,
        "published"=>false,"tag_created"=>false,
        "package_metadata_version"=>TOML.parsefile(joinpath(C20_ROOT,"Project.toml"))["version"],
        "qualification_sha256"=>C20_QHASH,"standard_files"=>41,"groups_passed"=>22,
        "single_process_full_suite_run"=>false,
        "audit_driver_sha256"=>c20_hash(@__FILE__),"runtime_sha256"=>runtime,"evidence_sha256"=>evidence,
        "input_sha256"=>C20_INPUTS,
        "profile"=>Dict("engine"=>"batch","date_tasks"=>1,"blas_threads"=>6,
            "julia"=>"1.12.7","cpu"=>"Ryzen 5 7500F","assets"=>65,"folds"=>3,"scenarios"=>300,
            "precision"=>"Float64","gc"=>"default","formal_compile_seconds"=>0.0),
        "windows"=>Dict("earlier_sha256"=>collect(C20_WINDOW_HASHES),"recent_sha256"=>recenthash,
            "earlier_seconds"=>[w.measured.seconds for w in windows],"recent_seconds"=>recent.seconds,
            "independent_weight_l1_max"=>weight_delta,"independent_return_max"=>return_delta,
            "completed_earlier_producer_peak_rss_mib"=>producer.rss,"first_producer_rc_preserved"=>124),
        "limits"=>["Evidence is from local N65 windows, not all-history convergence or long-run throughput.",
            "Two-date-task memory headroom is inadequate as a default; no parallel guarantee.",
            "42 days/s mission is not met and is not asserted as a CPU release gate.",
            "This engineering acceptance does not publish a package, create a tag, or claim all CPU optimizations are exhausted."])
    if action=="audit"
        ispath(output) && error("acceptance file appeared during audit")
        open(output,"w") do io; TOML.print(io,report;sorted=true); end
        println("SAVED ",relpath(output,C20_ROOT)," sha256=",c20_hash(output))
    else
        c20_require(TOML.parsefile(output)==report,"saved CPU acceptance differs from current evidence")
    end
    println("CPU20_ACCEPTANCE=PASS groups=22/22 standard_files=41 earlier_replays=2/2 independent_weight_L1=",
        weight_delta," return_max=",return_delta," GPU_RELEASE=2.1 GPU_BLOCKER=false published=false")
end
c20_main()
