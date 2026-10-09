include(joinpath("/home/kunweiz/Desktop/vibe/KTrader","test","contract_registry.jl"))

# Negative: temp dir missing ONE required file -> verify must error (fail-closed)
tmp = mktempdir()
for f in REQUIRED_CONTRACT_TESTS[1:end-1]
    touch(joinpath(tmp, f))
end
threw = try verify_required_contract_files(tmp); false
catch e; occursin("fail-closed", sprint(showerror, e)); end
println("[NEG] missing-last-required threw=", threw)
threw || error("negative case did not throw")
# Positive: all present in real test dir
@assert verify_required_contract_files(dirname(abspath("test/contract_registry.jl")))
println("[POS] all-present true")
println("# REQNEG DONE")
