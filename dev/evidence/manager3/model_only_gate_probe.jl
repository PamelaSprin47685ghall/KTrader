using KTrader, Serialization, Random, LinearAlgebra
include(joinpath("/home/kunweiz/Desktop/vibe/KTrader", "dev", "m1_artifact_replay.jl")); using .Main: compare_model_fields, verify_artifacts
function main()
    verify_artifacts()
    m = deserialize("/tmp/model_m1_t14309.jls").model
    inc = deserialize("/tmp/model_t14309_inc.jls").model
    bat = deserialize("/tmp/model_t14309_final.jls").model
    println("[G] positive: compare_model_fields vs inc + batch on load-only path")
    compare_model_fields("inc", m, inc)
    compare_model_fields("batch", m, bat)
    println("[G] POSITIVE PASS")
    # In-memory negative: mutate one finite mu_pred cell, same consumer must throw.
    idx = findfirst(isfinite, m.mu_pred)
    println("[N] mutating mu_pred[", idx, "] in memory")
    orig = m.mu_pred[idx]
    m.mu_pred[idx] = orig + 1.0
    threw = try compare_model_fields("inc", m, inc); false
    catch e; occursin("mu", sprint(showerror, e)) || (showerror(stdout, e); println()); true end
    m.mu_pred[idx] = orig   # restore in-memory
    println("[N] NEGATIVE threw=", threw)
    threw || error("negative case did not throw")
    println("# DONE")
end
main()
