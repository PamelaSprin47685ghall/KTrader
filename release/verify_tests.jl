# Tests for release tooling, not replacement mathematical assertions.
using Test, SHA, TOML
include(joinpath(@__DIR__,"..","bin","verify_release.jl"))
@testset "Final release metadata and integrity boundaries" begin
    root=normpath(joinpath(@__DIR__,".."))
    result=verify_final_release(root)
    @test result.version=="2.0.0"
    @test result.groups==22 && result.standard_files==41
    @test result.manifest_unchanged
    metadata=TOML.parsefile(joinpath(root,"RELEASE.toml"))
    text=read(joinpath(root,"Project.toml"),String)
    baseline=metadata["acceptance"]["baseline_project_sha256"]
    @test isnothing(final_project_check(text,baseline))
    @test_throws ArgumentError final_project_check(replace(text,"2.0.0"=>"2.0.1"),baseline)
    @test_throws ArgumentError final_project_check(text*"\n# unreviewed change\n",baseline)
    for unsafe in ("../Project.toml","/etc/passwd","src/../Project.toml","src\\KTrader.jl","")
        @test_throws ArgumentError final_path(root,unsafe)
    end
    @test_throws ArgumentError final_hash_check(root,"Project.toml",repeat("0",64))
    @test_throws ArgumentError final_hash_check(root,"Project.toml","not-a-sha256")
    @test_throws ArgumentError final_path(root,"release/not-present.toml")
    mktempdir() do dir
        write(joinpath(dir,"file"),"test bytes")
        @test isnothing(final_hash_check(dir,"file",bytes2hex(sha256("test bytes"))))
        symlink(joinpath(dir,"file"),joinpath(dir,"alias"))
        @test_throws ArgumentError final_path(dir,"alias")
    end
end
