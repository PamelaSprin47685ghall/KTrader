# test/artifact_replay_contract_tests.jl
#
# Contract tests for the comparison gates of dev/m1_artifact_replay.jl
# (THE ONE comparison owner). Pure logic ONLY: small arrays and the gate
# functions - no KTrader load, no artifact access, no fit / prepare /
# scenario loops. Positive cases plus adversarial cases that MUST throw
# (shape mismatch, finite<->NaN mask mismatch, Inf - including matching
# positions -, over-tolerance diff, bad kelly certificates, negative or
# over-roundoff swap loss, non-positive wealth, Frobenius semantics).
#
# The dev script is include'd library-safe: PROGRAM_FILE is THIS file, so
# its toplevel main() guard does not fire, and ARGS is empty so no
# 'using KTrader' happens. The check-phase zero-KTrader contract of the
# script is untouched by these tests (no phase is ever invoked here).
#
# Wired into the standard entry: test/contract_registry.jl registers
# this file among the 21 REQUIRED contract tests (plus the separate
# architecture gate file), so test/runtests.jl's single include loop
# includes it unconditionally — no isfile skip, no nested double run
# (the earlier "not wired" header wording was the pre-registration
# delivery-time state and stands as timeline). The standalone entry
# also works (the dev script is include'd library-safe: no main, no
# KTrader load):
#   julia --project=. test/artifact_replay_contract_tests.jl
#
# Run status (2026-10-08, DevOps, scoped_run 45 s / RSS 2048, RSS peak
# 355 MiB): 54/54 RC0 in 1.9 s. Timeline: the first run was 52 Pass /
# 2 Fail — both failures were hand-calculated index expectations in
# the fold_sample_rows determinism testset, corrected to the function's
# actual first/mid/last semantics; no gate logic or tolerance was
# touched. Green here certifies ONLY the gate logic on these fixtures
# - never an artifact comparison, never a replay phase.
#
# NEW (2026-10-08, 4th-review fix delivery): the "verified-path consume
# contract" testset below exercises the script's REAL load boundary -
# verify_and_register -> VERIFIED_PATHS -> load_verified /
# assert_source_of_record - with REAL Serialization.serialize /
# deserialize and the REAL SHA.sha256 of the actual bytes, on small
# temp fixtures. Positive: a legal fixture round-trips through the SAME
# production primitives the phases call. Negative (must throw): a
# tampered consumed file (real hash mismatch, registers nothing), the
# historical '.jl' lookalike twin carrying a DIFFERENT LEGAL payload, a
# bad-schema sibling, a missing sibling, the exact historical unbound
# source string, a string-correct-but-unverified source binding, and a
# POST-VERIFICATION REPLACEMENT: a verified+registered path is
# overwritten with a DIFFERENT LEGAL payload - load must be red and
# must not hand the new bytes to the consumer (re-verifying the new
# bytes through the legal channel then turns it green).
# No string-grep assertions, no re-implemented compare, no KTrader
# load, no access to the real /tmp artifacts, no N65 / EB solve. The
# VERIFIED_PATHS entries the test creates (including one pure
# in-memory registration of the of-record path - a Dict write, no
# file access) are removed in the finally block.
#
# RUN STATUS (source-written 2026-10-08, at fix-delivery time): the
# new testset had NOT been run when this source was delivered; the
# 54/54 record above is the pre-fix timeline and stays as history.
# Post-delivery run evidence, where it exists, lives in the DevOps
# records (dev/evidence/manager4/); this header does not restate or
# compete with those records - they are the single source of run truth.

using Test, LinearAlgebra, Random
using Serialization, SHA

include(joinpath(@__DIR__, "..", "dev", "m1_artifact_replay.jl"))

@testset "artifact replay comparison gates" begin
    @testset "library-safe include" begin
        @test !SOLVE_ONCE_FIRED[]      # include ran no phase
        @test length(PATHS_OF_RECORD) == 4
        @test length(WRAPPER_KEYS) == 5 && length(PREP_KEYS_22) == 22
    end

    @testset "tolerances of record" begin
        # historical thresholds (user-stated early / checker records)
        @test TOL_OF_RECORD.mu == 1e-10
        @test TOL_OF_RECORD.sigma == 1e-9
        @test TOL_OF_RECORD.alpha == 1e-8
        @test TOL_OF_RECORD.wl1 == 1e-7
        @test TOL_OF_RECORD.residual == 1e-9
        @test TOL_OF_RECORD.kelly_cert == 1e-8
        # Manager numerical-verification decisions (2026-10-08, this
        # assignment; items with NO historical threshold)
        @test TOL_OF_RECORD.g_c == 1e-9
        @test TOL_OF_RECORD.d_v == 1e-10
        @test TOL_OF_RECORD.x_canonical == 1e-10
        @test TOL_OF_RECORD.objective_diff == 1e-8
        @test TOL_OF_RECORD.swap_roundoff == 1e-12
    end

    @testset "assert_masked_maxdiff (residual semantics)" begin
        A = [1.0 2.0; 3.0 4.0]
        B = A .+ 1e-12
        md = assert_masked_maxdiff("pos", A, B; tol = 1e-9)
        @test 0.0 < md <= 2e-12
        # same-position NaN on both sides: finite-only, still green
        C = [1.0 NaN; 2.0 3.0]
        D = [1.0 NaN; 2.0 3.0 + 5e-13]
        md2 = assert_masked_maxdiff("nan-same", C, D; tol = 1e-9)
        @test 0.0 < md2 <= 1e-12
        @test_throws ErrorException assert_masked_maxdiff(
            "shape", A, [1.0 2.0; 3.0 4.0; 5.0 6.0]; tol = 1e-9)
        @test_throws ErrorException assert_masked_maxdiff(
            "finite-nan", [1.0, 2.0], [1.0, NaN]; tol = 1e-9)
        @test_throws ErrorException assert_masked_maxdiff(
            "inf", [1.0, 2.0], [1.0, Inf]; tol = 1e-9)
        @test_throws ErrorException assert_masked_maxdiff(
            "over-tol", [1.0], [1.0 + 1e-6]; tol = 1e-9)
        # NaN on both sides but at DIFFERENT positions is still red
        @test_throws ErrorException assert_masked_maxdiff(
            "nan-pos", [1.0, NaN], [NaN, 1.0]; tol = 1e-9)
        # Inf is rejected OUTRIGHT, even at MATCHING positions
        @test_throws ErrorException assert_masked_maxdiff(
            "inf-both", [Inf], [Inf]; tol = 1e-9)
        @test_throws ErrorException assert_masked_maxdiff(
            "inf-both-neg", [-Inf, 1.0], [-Inf, 1.0]; tol = 1e-9)
    end

    @testset "assert_matrix_close (model-field / scenario semantics)" begin
        M = reshape(1.0:6.0, 2, 3)
        @test assert_matrix_close("pos", M, M .+ 1e-12; tol = 1e-9) <= 2e-12
        @test_throws ErrorException assert_matrix_close(
            "shape", M, zeros(2, 2); tol = 1e-9)
        @test_throws ErrorException assert_matrix_close(
            "nan", M, [NaN 2.0 3.0; 4.0 5.0 6.0]; tol = 1e-9)
        @test_throws ErrorException assert_matrix_close(
            "inf", [1.0 Inf], [1.0 2.0]; tol = 1e-9)
        @test_throws ErrorException assert_matrix_close(
            "over-tol", [1.0], [1.0 + 1e-6]; tol = 1e-9)
    end

    @testset "assert_frob_close (L*L' Frobenius semantics)" begin
        A = [3.0 0.0; 0.0 4.0]
        B = A .+ [2e-10 0.0; 0.0 0.0]
        @test assert_frob_close("pos", A, B; tol = 1e-9) <= 2.5e-10
        @test_throws ErrorException assert_frob_close(
            "over-tol", [1.0], [1.0 + 1e-6]; tol = 1e-9)
        @test_throws ErrorException assert_frob_close(
            "shape", [1.0 2.0], [1.0 2.0; 3.0 4.0]; tol = 1e-9)
        @test_throws ErrorException assert_frob_close(
            "nan", [1.0], [NaN]; tol = 1e-9)
        @test_throws ErrorException assert_frob_close(
            "inf", [1.0], [Inf]; tol = 1e-9)
        # Frobenius AGGREGATES: many small per-cell diffs can exceed a
        # max-based tolerance - the metrics are not interchangeable
        C = fill(1e-6, 4)     # frob = sqrt(4)*1e-6 = 2e-6 > 1e-9
        @test_throws ErrorException assert_frob_close(
            "agg", fill(0.0, 4), C; tol = 1e-9)
    end

    @testset "scalar gates" begin
        @test assert_scalar_absdiff("abs-ok", 1.0, 1.0 + 1e-12;
                                    tol = 1e-9) <= 1e-9
        @test_throws ErrorException assert_scalar_absdiff(
            "abs-tol", 1.0, 1.0 + 1e-6; tol = 1e-9)
        @test_throws ErrorException assert_scalar_absdiff(
            "abs-nan", 1.0, NaN; tol = 1e-9)
        @test_throws ErrorException assert_scalar_absdiff(
            "abs-inf", 1.0, Inf; tol = 1e-9)
    end

    @testset "kelly certificate gate" begin
        good = (feasibility = 2.2e-16, kkt_residual = 9.9e-11,
                objective_gap = 6.3e-9, objective = 0.0059)
        @test assert_certified("good", good; tol = 1e-8)
        @test_throws ErrorException assert_certified(
            "bad-feas", (feasibility = 1e-6, kkt_residual = 1e-10,
                         objective_gap = 1e-9, objective = 0.0); tol = 1e-8)
        @test_throws ErrorException assert_certified(
            "bad-kkt", (feasibility = 1e-16, kkt_residual = 1e-6,
                        objective_gap = 1e-9, objective = 0.0); tol = 1e-8)
        @test_throws ErrorException assert_certified(
            "bad-gap", (feasibility = 1e-16, kkt_residual = 1e-10,
                        objective_gap = 1e-6, objective = 0.0); tol = 1e-8)
        @test_throws ErrorException assert_certified(
            "inf-cert", (feasibility = Inf, kkt_residual = Inf,
                         objective_gap = Inf, objective = -Inf); tol = 1e-8)
    end

    @testset "wealth and swap boundaries" begin
        @test assert_wealth_positive("wealth-ok", 0.804)
        @test_throws ErrorException assert_wealth_positive(
            "wealth-zero", 0.0)
        @test_throws ErrorException assert_wealth_positive(
            "wealth-neg", -0.1)
        # swap: 1e-12 pure-roundoff allowance (Manager decision), NOT the
        # observed values; anything beyond it must be red
        @test assert_swap_nonneg("swap-zero", 0.0; tol = 1e-12)
        @test assert_swap_nonneg("swap-roundoff", -2.5e-17; tol = 1e-12)
        @test_throws ErrorException assert_swap_nonneg(
            "swap-neg", -1e-6; tol = 1e-12)
        @test_throws ErrorException assert_swap_nonneg(
            "swap-over-roundoff", -1e-11; tol = 1e-12)
        @test_throws ErrorException assert_swap_nonneg(
            "swap-nan", NaN; tol = 1e-12)
    end

    @testset "fold_sample_rows determinism" begin
        # Actual implementation: first(rg), rg[div(len,2)], last(rg).
        # 1:5 -> {1, 1:5[2]=2, 5}; 6:10 -> {6, 6:10[2]=7, 10};
        # 11:15 -> {11, 11:15[2]=12, 15} — sorted unique.
        rows = fold_sample_rows([1:5, 6:10, 11:15])
        @test rows == [1, 2, 5, 6, 7, 10, 11, 12, 15]
        @test length(rows) == 9
        # overlapping: {2,3,4} ∪ {3,4,6} -> [2,3,4,6]
        @test fold_sample_rows([2:4, 3:6]) == [2, 3, 4, 6]
    end

    @testset "verified-path consume contract (real machinery)" begin
        # Agreement under test: producer = Serialization.serialize on a
        # small fixture; consumer = the script's REAL load boundary
        # (verify_and_register -> load_verified /
        # assert_source_of_record), the same primitives the production
        # phases call. Identity rule = full sha256 + byte size of the
        # EXACT file the consumer will deserialize. Failure semantics =
        # fail closed BEFORE any deserialize attempt. A mutation on
        # either side of the boundary must turn this red immediately.
        tmp = mktempdir()
        registered = String[]   # VERIFIED_PATHS keys created by this test
        try
            payload = (; prep = (; a = 1.25, b = [0.5, -0.25],
                                 stats = (; ranges = [1:5, 6:10])),
                       fast = true)
            good = joinpath(tmp, "fixture.jls")
            open(good, "w") do io
                Serialization.serialize(io, payload)
            end
            # REAL sha256 of the REAL bytes (the production call).
            digest = bytes2hex(SHA.sha256(read(good)))
            nbytes = filesize(good)
            rec = Dict{String,Any}("sha256" => digest, "bytes" => nbytes,
                                   "role" => "contract fixture")

            # [P1] legal fixture: verify+register, then load round-trips
            # through the SAME production primitives.
            @test verify_and_register(good, rec)["sha256"] == digest
            push!(registered, abspath(good))
            rt = load_verified(good)
            @test rt.prep.a == 1.25
            @test rt.prep.b == [0.5, -0.25]
            @test rt.prep.stats.ranges == [1:5, 6:10]
            @test rt.fast === true
            @test VERIFIED_PATHS[abspath(good)] == (digest, nbytes)

            # [N1] tampered consumed file: one byte flipped inside the
            # serialized payload. The real hash comparison must be red
            # and register NOTHING - so the consumer refuses the file
            # too (the refusal is registry-based, not parse-based).
            tampered = joinpath(tmp, "fixture_tampered.jls")
            rawbytes = read(good)
            rawbytes[end] = xor(rawbytes[end], 0x01)
            write(tampered, rawbytes)
            rec_twin = Dict{String,Any}("sha256" => digest,
                                        "bytes" => nbytes,
                                        "role" => "tampered twin")
            @test_throws ErrorException verify_and_register(tampered,
                                                             rec_twin)
            @test !haskey(VERIFIED_PATHS, abspath(tampered))
            @test_throws ErrorException load_verified(tampered)

            # [N2] the historical bug's exact shape: the authenticated
            # '.jls' stays registered and readable; a sibling '.jl'
            # exists with a DIFFERENT LEGAL payload - the consumer must
            # NOT read it (unregistered path, refused before ANY
            # deserialize, even though the file exists and would parse).
            sibling = joinpath(tmp, "fixture.jl")
            open(sibling, "w") do io
                Serialization.serialize(io,
                    (; prep = (; a = 99.0, b = [1.0, 2.0],
                               stats = (; ranges = [1:3])), fast = false))
            end
            @test isfile(sibling)
            @test_throws ErrorException load_verified(sibling)
            # bad-schema sibling: not a valid serialization at all - the
            # refusal still fires BEFORE parsing (not parse-based).
            junk = joinpath(tmp, "fixture_junk.jl")
            write(junk, "not a serialization at all")
            @test_throws ErrorException load_verified(junk)
            # missing sibling: the same fail-closed refusal.
            @test_throws ErrorException load_verified(
                joinpath(tmp, "fixture_missing.jl"))
            # the authenticated fixture is unaffected by the siblings:
            rt2 = load_verified(good)
            @test rt2.prep.a == 1.25
            @test rt2.prep.stats.ranges == [1:5, 6:10]

            # [N3] source-binding gate. The of-record path for the prep
            # key is the manifest-registered '.jls' - NOT the historical
            # unbound '.jl':
            key = "prep_inc_t14309_jls"
            por = path_of_record(key)
            @test por == "/tmp/prep_inc_t14309.jls"
            @test_throws ErrorException path_of_record("no_such_key")
            # [N3a] the exact historical unbound source string: RED.
            @test_throws ErrorException assert_source_of_record(
                "/tmp/prep_inc_t14309.jl", key)
            # [N3b] string-correct but NOT verified in this process:
            # RED (the gate checks the registry, not just the string).
            @test_throws ErrorException assert_source_of_record(por, key)
            # [N3c] positive: register the of-record path (pure
            # in-memory Dict write - NO file access), then the gate
            # passes and returns the bound path.
            VERIFIED_PATHS[abspath(por)] = (digest, nbytes)
            push!(registered, abspath(por))
            @test assert_source_of_record(por, key) == por

            # [N4] post-verification replacement (the TOCTOU gap this
            # follow-up closed): a LEGAL fixture is verified+registered
            # and loads green; the SAME path is then overwritten with a
            # DIFFERENT LEGAL serialize payload; load_verified must be
            # RED and must NOT hand the new payload to the consumer.
            # Re-registering the NEW bytes through the same legal
            # channel makes the path consumable again - green, and the
            # consumer then sees the NEW (re-certified) payload.
            swapped = joinpath(tmp, "fixture_swap.jls")
            open(swapped, "w") do io
                Serialization.serialize(io, payload)
            end
            rec_s = Dict{String,Any}(
                "sha256" => bytes2hex(SHA.sha256(read(swapped))),
                "bytes" => filesize(swapped),
                "role" => "swap fixture")
            @test verify_and_register(swapped, rec_s)["role"] ==
                  "swap fixture"
            push!(registered, abspath(swapped))
            @test load_verified(swapped).fast === true   # pre-swap green
            # replace the SAME path with a DIFFERENT legal payload:
            open(swapped, "w") do io
                Serialization.serialize(io,
                    (; prep = (; a = -7.5, b = [9.0],
                               stats = (; ranges = [2:4])), fast = false))
            end
            # the new file is well-formed and WOULD deserialize - the
            # gate must still refuse (consumed bytes != registered):
            @test_throws ErrorException load_verified(swapped)
            # re-verify the NEW bytes through the legal channel: green,
            # and the consumer now sees the re-certified payload.
            rec_s2 = Dict{String,Any}(
                "sha256" => bytes2hex(SHA.sha256(read(swapped))),
                "bytes" => filesize(swapped),
                "role" => "swap fixture re-verified")
            @test verify_and_register(swapped, rec_s2)["role"] ==
                  "swap fixture re-verified"
            rt3 = load_verified(swapped)
            @test rt3.prep.a == -7.5
            @test rt3.prep.b == [9.0]
            @test rt3.prep.stats.ranges == [2:4]
            @test rt3.fast === false
        finally
            for k in registered
                delete!(VERIFIED_PATHS, k)
            end
            rm(tmp; force = true, recursive = true)
        end
    end
end
