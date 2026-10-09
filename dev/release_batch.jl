# Budgeted qualification of the standard suite's actual file inventory.
# Each invocation runs ONE group. Never changes the standard full entry.
using SHA, TOML
# Match the unchanged standard test entry: some legacy files deliberately
# depend on its imports instead of importing Random/Dates/Convex themselves.
using Test, Random, LinearAlgebra, Statistics, Dates, Convex, Clarabel
using KTrader
const RB_ROOT=normpath(joinpath(@__DIR__,".."))
const RB_OUT=joinpath(@__DIR__,"evidence","earlier_closure_20261009","qualification")
include(joinpath(RB_ROOT,"test","contract_registry.jl"))
const RB_GROUPS=Dict(
    "architecture"=>[ARCHITECTURE_CONTRACT_TEST],
    "constitution"=>["v1_constitutional_tests.jl"],
    "numerics"=>["numerical_tests.jl","data_tests.jl","execution_tests.jl","kelly_packing_tests.jl"],
    "response_a"=>["conditioned_jcore_tests.jl","conditioned_deadwork_tests.jl","fit_local_reuse_tests.jl","conditioned_geometry_reuse_tests.jl"],
    "response_b"=>["conditioned_lazy_gradient_tests.jl","scalar_tensor_tests.jl","conditioned_certificate_reuse_tests.jl","conditioned_certificate_cache_tests.jl","conditioned_storage_tests.jl","conditioned_contraction_kernel_tests.jl"],
    "eb"=>["conditioned_eb_tests.jl","relative_support_tests.jl"],
    "prepared"=>["incremental_primal_prep_tests.jl","incremental_pending_tests.jl","incremental_budget_contract_tests.jl","history_cache_contract_tests.jl","prepared_problem_contract_tests.jl","prepared_lifecycle_tests.jl"],
    "posterior"=>["posterior_contract_tests.jl","residual_oracle_flow_tests.jl","principal_root_contract_tests.jl"],
    "residuals"=>["residual_oracle_tests.jl"],
    "timeblocks"=>["timeblock_tests.jl","backtest_target_tests.jl"],
    "cache_elision"=>["backtest_cache_tests.jl"],
    "failure_delivery"=>["backtest_failure_tests.jl"],
    "interfaces"=>["timed_passthrough_tests.jl","model_api_micro_tests.jl","ceiling_admission_tests.jl","cli_boundary_contract_tests.jl","ceiling_probes.jl","artifact_replay_contract_tests.jl"],
    "reference"=>["reference_isolation_tests.jl"],
    "inc_prefix"=>["incremental_tests.jl::prefix"],
    "inc_ragged"=>["incremental_tests.jl::ragged"],
    "inc_ownership"=>["incremental_tests.jl::ownership"],
    "inc_kernels"=>["incremental_tests.jl::kernels"],
    "guard_a"=>["scoped_run_tests.jl::selftest","scoped_run_tests.jl::baseline"],
    "guard_b"=>["scoped_run_tests.jl::owner-term","scoped_run_tests.jl::owner-int","scoped_run_tests.jl::owner-hup","scoped_run_tests.jl::clock-mutant"],
    "guard_c"=>["scoped_run_tests.jl::clock-mono","scoped_run_tests.jl::clock-fail","scoped_run_tests.jl::clock-saturate","scoped_run_tests.jl::clock-start"],
    "guard_d"=>["scoped_run_tests.jl::clock-cleanup","scoped_run_tests.jl::clock-elapsed","scoped_run_tests.jl::clock-transient","scoped_run_tests.jl::clock-window-poll","scoped_run_tests.jl::clock-negctl","scoped_run_tests.jl::command-admission"])

rb_hash(path)=bytes2hex(open(sha256,path))
function rb_sources()
    paths=["Project.toml","Manifest.toml","bin/scoped_run.sh"]
    for dir in ("src","test","bin","dev")
        for (root,_,files) in walkdir(joinpath(RB_ROOT,dir))
            occursin(joinpath("dev","evidence"),root) && continue
            for f in files
                (endswith(f,".jl") || endswith(f,".cpp") || (dir=="test" && endswith(f,".jls"))) || continue
                push!(paths,relpath(joinpath(root,f),RB_ROOT))
            end
        end
    end
    Dict(p=>rb_hash(joinpath(RB_ROOT,p)) for p in sort!(unique(paths)))
end
function rb_validate_plan()
    source=read(joinpath(RB_ROOT,"test","runtests.jl"),String)
    full=split(source,"@testset \"KTrader V1.0 Complete Rigorous Suite\" begin";limit=2)[2]
    legacy=[m.captures[1] for m in eachmatch(r"include\(\"([^\"]+\.jl)\"\)",full)]
    targets=vcat(collect(REQUIRED_CONTRACT_TESTS),legacy,[ARCHITECTURE_CONTRACT_TEST])
    entries=vcat(collect(values(RB_GROUPS))...)
    length(unique(entries))==length(entries) || error("duplicate planned entry")
    Set(first(split(x,"::")) for x in entries)==Set(targets) || error("plan does not cover standard file inventory exactly")
    verify_required_contract_files(joinpath(RB_ROOT,"test");files=targets)
    for (file,regex,quoted) in (
        ("scoped_run_tests.jl",r"const PHASES\s*=\s*\[([^\]]+)\]"s,true),
        ("incremental_tests.jl",r"const INCREMENTAL_TEST_PHASES\s*=\s*\(([^)]+)\)"s,false))
        text=match(regex,read(joinpath(RB_ROOT,"test",file),String)).captures[1]
        phases=quoted ? [m.captures[1] for m in eachmatch(r"\"([^\"]+)\"",text)] :
                        [m.captures[1] for m in eachmatch(r":([a-z_]+)",text)]
        actual=[last(split(x,"::")) for x in entries if startswith(x,file*"::")]
        Set(phases)==Set(actual) || error("missing or unknown phase in $file")
    end
    targets
end
function rb_write(path,record)
    ispath(path) && error("refusing to overwrite $path")
    open(path,"w") do io
        TOML.print(io,record;sorted=true)
    end
end
function rb_main()
    length(ARGS)==1 || error("one action: snapshot | status | group name")
    action=only(ARGS); empty!(ARGS)
    get(ENV,"REFISO_STAGE","ALL")=="ALL" || error("partial reference environment forbidden")
    targets=rb_validate_plan()
    snapshot=joinpath(RB_OUT,"qualification_snapshot.toml")
    if action=="snapshot"
        rb_write(snapshot,Dict("sources"=>rb_sources(),"groups"=>RB_GROUPS,"standard_files"=>sort!(unique(targets))))
        println("SNAPSHOT ",rb_hash(snapshot)," files=",length(unique(targets))," groups=",length(RB_GROUPS))
        return
    end
    record=TOML.parsefile(snapshot); digest=rb_hash(snapshot)
    record["sources"]==rb_sources() || error("source changed since qualification snapshot")
    record["groups"]==RB_GROUPS || error("plan changed since snapshot")
    if action=="status"
        missing=String[]
        for group in sort!(collect(keys(RB_GROUPS)))
            path=joinpath(RB_OUT,"pass_"*group*".toml")
            if !isfile(path)
                push!(missing,group); println("MISSING ",group); continue
            end
            r=TOML.parsefile(path)
            r["snapshot"]==digest && r["entries"]==RB_GROUPS[group] && r["passed"] || error("invalid receipt $group")
            println("PASS ",group," ",r["counts"])
        end
        println("COVERAGE groups=",length(RB_GROUPS)-length(missing),"/",length(RB_GROUPS),
            " standard_files=",length(unique(targets))," source_unchanged=true")
        isempty(missing) || error("qualification incomplete: "*join(missing,", "))
        println("BOUNDED STANDARD-FILE COVERAGE COMPLETE; not single-process suite or performance sign-off")
        return
    end
    haskey(RB_GROUPS,action) || error("unknown group")
    receipt=joinpath(RB_OUT,"pass_"*action*".toml")
    ispath(receipt) && error("group already recorded; no silent rerun")
    if action!="architecture"
        a=TOML.parsefile(joinpath(RB_OUT,"pass_architecture.toml"))
        a["snapshot"]==digest && a["passed"] || error("architecture admission missing")
    end
    println("BEGIN group=",action," snapshot=",digest," entries=",RB_GROUPS[action]); flush(stdout)
    counts="child process assertions in log"
    if action=="architecture"
        push!(ARGS,"--architecture-only")
        include(joinpath(RB_ROOT,"test","runtests.jl")); empty!(ARGS)
        counts="standard architecture entry in log"
    elseif all(occursin("::",x) for x in RB_GROUPS[action])
        for entry in RB_GROUPS[action]
            file,phase=split(entry,"::")
            println("CHILD ",entry); flush(stdout)
            run(`$(Base.julia_cmd()) --startup-file=no --compiled-modules=existing --project=$RB_ROOT $(joinpath(RB_ROOT,"test",file)) --phase=$phase`)
        end
    else
        # These are the actual standard files, not copied mathematical tests.
        ts=@testset "Release group $action" begin
            for file in RB_GROUPS[action]
                println("FILE ",file); flush(stdout)
                include(joinpath(RB_ROOT,"test",file))
            end
        end
        counts=repr(Test.get_test_counts(ts))
    end
    record["sources"]==rb_sources() || error("source changed during group")
    rb_write(receipt,Dict("snapshot"=>digest,"entries"=>RB_GROUPS[action],"passed"=>true,"counts"=>counts,
        "julia"=>string(VERSION),"opt_level"=>Int(Base.JLOptions().opt_level)))
    println("END group=",action," passed=true counts=",counts); flush(stdout)
end
rb_main()
