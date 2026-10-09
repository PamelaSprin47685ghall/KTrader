#!/usr/bin/env julia
# Source release verification only: no package loading, deserialization,
# network, fitted data, trading action or retrospective numerical claim.
using SHA, TOML

const FINAL_VERSION = "2.0.0"
const FINAL_ACCEPTANCE_HASH = "dd1fe993b2ad5020e46db8f634fdb9254b816d114ce9dd7c38a64fb188c820df"
const FINAL_QUALIFICATION_HASH = "31e1c48729147d5022a0b5f894c23afece80cd0bb9a596161fd7619c359e3f17"
const FINAL_TEST_LIBRARIES = ("dev/probes.jl", "dev/m1_artifact_replay.jl")

final_require(ok, message) = ok ? nothing : throw(ArgumentError(message))
final_digest(path) = bytes2hex(open(sha256, path))
function final_path(root, relative)
    parts=split(relative,'/')
    final_require(!isempty(relative) && !isabspath(relative) && !occursin('\\',relative) &&
        all(p->!(p in ("", ".", "..")),parts),"unsafe release path: $relative")
    path=root
    for part in parts
        path=joinpath(path,part)
        final_require(!islink(path),"symlink is not a release input: $relative")
    end
    final_require(isfile(path),"missing release file: $relative")
    path
end
function final_hash_check(root, relative, expected)
    final_require(occursin(r"^[0-9a-f]{64}$",expected),"invalid SHA256 for $relative")
    final_require(final_digest(final_path(root,relative))==expected,"release bytes changed: $relative")
end
function final_project_check(text, baseline_hash)
    line="version = \"2.0.0\""
    final_require(count(==(line),split(text,'\n'))==1,"expected exact 2.0.0 version line")
    original=replace(text,line=>"version = \"0.1.0\"";count=1)
    final_require(bytes2hex(sha256(original))==baseline_hash,
        "Project changed beyond the approved version line")
    final_require(TOML.parse(text)["version"]==FINAL_VERSION,"wrong package version")
end
function verify_final_release(root=normpath(joinpath(@__DIR__,"..")))
    metadata=TOML.parsefile(final_path(root,"RELEASE.toml"))
    final_require(metadata["name"]=="KTrader" && metadata["version"]==FINAL_VERSION &&
        metadata["status"]=="Final" && metadata["edition"]=="CPU" &&
        metadata["gpu_release"]=="2.1","wrong release identity")
    provenance=metadata["acceptance"]
    final_require(provenance["sha256"]==FINAL_ACCEPTANCE_HASH &&
        provenance["qualification_sha256"]==FINAL_QUALIFICATION_HASH,"wrong accepted baseline")
    final_hash_check(root,provenance["path"],FINAL_ACCEPTANCE_HASH)
    final_hash_check(root,provenance["qualification_path"],FINAL_QUALIFICATION_HASH)
    accepted=TOML.parsefile(final_path(root,provenance["path"]))
    qualified=TOML.parsefile(final_path(root,provenance["qualification_path"]))
    final_require(accepted["cpu_engineering_acceptance_passed"] &&
        accepted["standard_files"]==41 && accepted["groups_passed"]==22,
        "CPU engineering acceptance is incomplete")
    final_require(accepted["qualification_sha256"]==FINAL_QUALIFICATION_HASH,
        "acceptance and qualification disagree")
    hashes=qualified["sources"]
    final_require(provenance["baseline_project_sha256"]==hashes["Project.toml"] &&
        provenance["manifest_sha256"]==hashes["Manifest.toml"],"metadata baseline mismatch")
    final_project_check(read(final_path(root,"Project.toml"),String),hashes["Project.toml"])
    final_hash_check(root,"Manifest.toml",hashes["Manifest.toml"])
    checked=0
    for (path,digest) in hashes
        if startswith(path,"src/") || startswith(path,"test/") || startswith(path,"bin/") ||
            path in FINAL_TEST_LIBRARIES
            final_hash_check(root,path,digest); checked+=1
        end
    end
    # Additional production/test code cannot hide outside the accepted set.
    for dir in ("src","test"), (base,_,files) in walkdir(joinpath(root,dir)), file in files
        endswith(file,".jl") || endswith(file,".jls") || continue
        path=replace(relpath(joinpath(base,file),root),'\\'=>'/')
        final_require(haskey(hashes,path),"unqualified production/test addition: $path")
    end
    for (path,digest) in accepted["evidence_sha256"]
        final_hash_check(root,path,digest)
    end
    qdir=dirname(provenance["qualification_path"])
    final_require(length(qualified["standard_files"])==41 && length(qualified["groups"])==22,
        "qualification inventory changed")
    for (group,entries) in qualified["groups"]
        path=replace(joinpath(qdir,"pass_"*group*".toml"),'\\'=>'/')
        receipt=TOML.parsefile(final_path(root,path))
        final_require(get(receipt,"passed",false) && receipt["snapshot"]==FINAL_QUALIFICATION_HASH &&
            receipt["entries"]==entries,"invalid qualification receipt: $group")
    end
    # The source archive adds a complete payload checksum inventory. It is
    # separate from the accepted mathematics and is not a new numerical test.
    inventory=joinpath(root,"release","SHA256SUMS")
    count_payload=0
    if isfile(inventory)
        seen=Set{String}()
        for line in eachline(inventory)
            m=match(r"^([0-9a-f]{64})  (.+)$",line)
            final_require(m!==nothing,"invalid archive checksum line")
            digest,path=m.captures
            final_require(!(path in seen),"duplicate archive path: $path")
            push!(seen,path); final_hash_check(root,path,digest)
        end
        actual=Set{String}()
        for (base,dirs,files) in walkdir(root)
            filter!(!=(".git"),dirs)
            for file in files
                path=replace(relpath(joinpath(base,file),root),'\\'=>'/')
                path=="release/SHA256SUMS" || push!(actual,path)
            end
        end
        final_require(actual==seen,"archive has unlisted or missing files")
        count_payload=length(seen)
    end
    (;version=FINAL_VERSION,checked_cpu_files=checked,groups=22,standard_files=41,
      payload_files=count_payload,manifest_unchanged=true)
end
function final_main()
    isempty(ARGS) || throw(ArgumentError("verify_release takes no arguments"))
    result=verify_final_release()
    println("KTrader ",result.version," Final CPU: SOURCE_VERIFIED",
        " accepted_groups=22/22 standard_files=41 manifest_unchanged=true",
        " archive_payload_files=",result.payload_files," GPU=2.1")
    println("Source/evidence identity check only; no numerical tests, network or trading actions executed.")
end
if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    final_main()
end
