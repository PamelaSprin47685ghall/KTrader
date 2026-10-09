using Test
using KTrader
# M1: the probe CLI's parse/admission core lives in the dev namespace and is
# shared VERBATIM with bin/ceiling_probes.jl — this file is the witness at
# that supported entrance. It never spawns a command and never builds a
# second solver: the boundary under test is the argument/admission contract
# the bin script actually consumes, plus source-level guards against the
# audited CLI regressions (stale KTrader.-qualified probe calls, the removed
# production probe keyword, Date-before-include, S/seed on the sequential
# interface).
isdefined(@__MODULE__, :DevProbes) || include(joinpath(@__DIR__, "..", "dev", "probes.jl"))
using .DevProbes

# Mutation guidance (how this suite proves it can go red): change
# PROBE_CLI_FLAGS, PROBE_CLI_KNOWN_PROBES or any admission rule in
# dev/probes.jl, or reintroduce a stale call shape in
# bin/ceiling_probes.jl, and the corresponding test below fails — the bin and
# the tests share the same decision path, so a drift on either side shows up
# here rather than only at a failed CLI run.
#
# UNVERIFIED BOUNDARY (kept open on purpose): parse-level and source-text
# assertions cannot by themselves prove that the real bin script executes —
# include order, module loading, runtime errors and the actual stage flow
# are outside this file's reach. The minimal real-CLI positive/negative
# checks (one admitted command, one rejected command line) belong to DevOps;
# this suite must never be cited as evidence that the script has been run.

@testset "CLI boundary contract (bin/ceiling_probes.jl entrance)" begin
    # ---------- parse: supported forms are accepted verbatim ----------
    let parsed = DevProbes.parse_probe_cli(String[])
        @test parsed["from"] == "2018-01-01"
        @test parsed["t-star"] == ""
        @test parsed["phase"] == "prepare"
        @test parsed["mode"] == "cold"
        @test parsed["folds"] == "3"
        @test parsed["probes"] == "B"
        @test parsed["out"] == ""
    end
    let parsed = DevProbes.parse_probe_cli(["--from", "2020-06-01", "--phase", "solve",
                                            "--mode", "warm", "--folds", "4",
                                            "--ridge-alpha", "2.5",
                                            "--probes", "B,D,E",
                                            "--t-star", "14000", "--out", "/tmp/x.txt"])
        @test parsed["from"] == "2020-06-01"
        @test parsed["phase"] == "solve"
        @test parsed["probes"] == "B,D,E"
        @test parsed["t-star"] == "14000"
    end

    # ---------- parse: fail-closed BEFORE anything is loaded ----------
    @test_throws ArgumentError DevProbes.parse_probe_cli(["--bogus", "1"])
    @test_throws ArgumentError DevProbes.parse_probe_cli(["--from", "2020-01-01", "--from", "2021-01-01"])
    @test_throws ArgumentError DevProbes.parse_probe_cli(["--from"])          # missing value
    @test_throws ArgumentError DevProbes.parse_probe_cli(["--probes", "  "])  # empty value

    # ---------- admission: legal shapes ----------
    # Default: probe B only, phase prepare — zero solves.
    let a = DevProbes.validate_probe_cli(DevProbes.parse_probe_cli(String[]); K = 2500)
        @test a.phase === :prepare
        @test a.mode === :cold
        @test a.probe_list == [:B]
        @test a.runs_ac == false
    end
    # D/E are readings of the phase stages: E requires --phase solve.
    let a = DevProbes.validate_probe_cli(DevProbes.parse_probe_cli(["--phase", "solve", "--probes", "B,D,E"]); K = 2500)
        @test a.phase === :solve
        @test a.probe_list == [:B, :D, :E]
        @test a.runs_ac == false
    end
    let a = DevProbes.validate_probe_cli(DevProbes.parse_probe_cli(["--probes", "D"]); K = 2500)
        @test a.phase === :prepare          # D's prepare reading is legal
        @test a.runs_ac == false
    end
    # A/C: the single-day sequential run, K=1 only, replacing the phases.
    let a = DevProbes.validate_probe_cli(DevProbes.parse_probe_cli(["--probes", "A"]); K = 1)
        @test a.runs_ac == true
        @test a.phase === :prepare          # default phase; the A/C run replaces it
    end
    let a = DevProbes.validate_probe_cli(DevProbes.parse_probe_cli(["--probes", "A,C"]); K = 1)
        @test a.runs_ac == true             # A and C SHARE the one sequential run
    end

    # ---------- admission: conflicting shapes are rejected fail-closed ----------
    base(args; K) = DevProbes.validate_probe_cli(DevProbes.parse_probe_cli(args); K = K)
    # A/C with K>1: the multi-day hidden workload.
    @test_throws ArgumentError base(["--probes", "A"]; K = 30)
    @test_throws ArgumentError base(["--probes", "C"]; K = 2500)
    # A/C with --phase solve: a SECOND solve in one command.
    @test_throws ArgumentError base(["--probes", "A", "--phase", "solve"]; K = 1)
    # A/C with D/E: no stage readings exist for the replaced phases.
    @test_throws ArgumentError base(["--probes", "A,D"]; K = 1)
    @test_throws ArgumentError base(["--probes", "C,E"]; K = 1)
    # A/C with --t-star: not read by the sequential interface — refuse silently.
    @test_throws ArgumentError base(["--probes", "A", "--t-star", "14000"]; K = 1)
    # E without --phase solve: E is a reading OF the solve.
    @test_throws ArgumentError base(["--probes", "E"]; K = 2500)
    @test_throws ArgumentError base(["--probes", "B,E", "--phase", "prepare"]; K = 2500)
    # Bad values.
    @test_throws ArgumentError base(["--phase", "both"]; K = 1)
    @test_throws ArgumentError base(["--mode", "hot"]; K = 1)
    @test_throws ArgumentError base(["--probes", "X"]; K = 1)
    @test_throws ArgumentError base(["--probes", ""]; K = 1)

    # ---------- bin source guards: the audited regressions stay dead ----------
    bin_src = read(joinpath(@__DIR__, "..", "bin", "ceiling_probes.jl"), String)
    # The script must consume the SAME parse/admission core as these tests.
    @test occursin("DevProbes.parse_probe_cli", bin_src)
    @test occursin("DevProbes.validate_probe_cli", bin_src)
    @test occursin("DevProbes.sequential_ceiling_probe", bin_src)
    # Stale KTrader.-qualified probe calls (the functions live in DevProbes
    # only since M1) must not come back.
    @test !occursin("KTrader.prepare_stage_probe", bin_src)
    @test !occursin("KTrader.solve_stage_probe", bin_src)
    # The production probe keyword is gone from backtest_v1; passing it is a
    # MethodError, so the script must not attempt it.
    @test !occursin(", probe)", bin_src)
    @test !occursin("probe = probe", bin_src)
    # Dates must be loaded at the top: Date(...) runs after `using Dates`.
    @test something(findfirst("using Dates", bin_src), 0:-1)[1] <
          something(findfirst("Date(", bin_src), 0:-1)[1]
    # The sequential interface has no S/seed knobs — passing them is a
    # MethodError, so the script must not attempt it.
    @test !occursin(Regex("sequential_ceiling_probe\\([^\\n)]*;[^\\n)]*S\\s*="), bin_src)
    @test !occursin(Regex("sequential_ceiling_probe\\([^\\n)]*;[^\\n)]*seed\\s*="), bin_src)
    # The single-authorized-solve rule is stated in the admission core and
    # honored by the script's phase replacement.
    @test occursin("replaced by the A/C sequential run", bin_src)
end
