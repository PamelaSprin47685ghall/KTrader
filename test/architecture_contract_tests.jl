# M1 Architecture Contract Tests — deterministic, mechanically checkable only.
# Wiring owner: this file plus test/contract_registry.jl and test/runtests.jl
# (exclusive per the M1 wiring assignment; no other engineer writes these).
#
# Mechanically gated here (real AST include/export/identifier scan + real
# Julia runtime method metadata — no repo-wide grep guessing, no line counts):
#   [A] The production include tree (KTrader.jl, recursively) contains no
#       dev/ path and no default ceiling_probes include.
#   [B] KTrader exports no forbidden profiling symbols.
#   [C] No forbidden profiling identifiers (oof_shadow / OOFCounters /
#       OOFShadowConfigError / oof_shadow_identity_check!) appear as AST
#       identifiers anywhere in the production include tree. Comments and
#       string literals cannot match, so this is not a text grep.
#   [D] PreparedProblem is a typed boundary with the single solve entry:
#       isdefined(KTrader, :PreparedProblem) and
#       hasmethod(KTrader.solve, Tuple{KTrader.PreparedProblem}).
#   [E] Profiling types are absent from the runtime production namespace
#       (OOFShadowConfigError, CeilingProbeCollector must be undefined).
#
# INCLUDE SYNTAX BOUNDARY (fail-closed; untrusted Julia is NEVER evaluated):
# The tree walk statically resolves ONLY these include forms:
#     include("file.jl")                          bare string literal
#     include(joinpath("a", "b.jl"))              literal-only joinpath
#     include(joinpath(@__DIR__, "b.jl"))         @__DIR__ = dirname(current file)
#     include(joinpath(dirname(@__FILE__), ...))  @__FILE__ = current file
#     include(mod, path)                          2-argument module form
#     Base.include(mod_or_fn, path)               qualified Base.include
# (joinpath/dirname are accepted bare or as Base.joinpath/Base.dirname, and
# only over the forms above). Bare literals resolve relative to the
# INCLUDING file's directory — Julia's real include semantics — so
# include("../data.jl") in sub/a.jl and
# include(joinpath(@__DIR__, "..", "data.jl")) in that same file converge on
# the SAME canonical target key (exactly one rebase, never two). ".."
# segments are checked as written FIRST (the raw segment list before any
# folding: a dev segment anywhere in the written path is a dev edge even if
# folding would remove it), and only then folded into a canonical filemap
# key ("." drops, ".." pops its parent; absolute paths, and any target whose
# ".." escapes the scanned source root, are NOT claimed as supported — both
# are explicit fail-closed violations). Every OTHER include shape is an explicit
# violation: variable paths, string interpolation, broadcast `include.`,
# foreign qualified callees (Foo.include), a bare `include` used as a value
# (e.g. map(include, files)), and wrong arity. Nothing is silently ignored —
# a form the gate cannot prove static is a red gate, not a skipped edge. The
# old walk silently ignored joinpath/qualified/dynamic includes; this file
# closes that hole. Resolved paths are checked segment-as-written (no
# normalization that could erase a dev/ segment; "." segments are dropped
# because they cannot change what a path points at, ".." segments are KEPT
# because they can): any path containing a "dev" segment is rejected, and
# every resolved include target must exist in the source map.
#
# NOT mechanically gated — recorded invariants, enforced by the math
# regression suites and review, with NO fabricated regex gates:
#   - "one owner per math concept" / algorithmic equivalence of accelerators;
#   - "incremental consumes only the single solve" (consumption semantics);
#   - "reference runs without accel/probes loaded" beyond include shape.
# These cannot be decided from syntax without false positives; they stay
# human/math-regression territory by design. The gate deliberately draws NO
# solver-uniqueness conclusion from string call counts.
#
# Core-loading policy (matches the standard entry's parsed mode):
#   fixture-root mode (--architecture-only --architecture-root PATH) loads
#   ONLY stdlib/Test here — the floating production module is never loaded,
#   so the AST gate and the fail-closed required admission cannot be blocked
#   by production load state, and no pkgimage/freeze equivalence is assumed.
#   Real-tree mode (--architecture-only, no root) and default full mode DO
#   load KTrader here so the runtime typed/method/namespace checks (M1-D/E)
#   execute against the real module. The AST consumer rules are IDENTICAL in
#   every mode — no rule is weakened for fixtures.
#
# The SAME primitive m1_architecture_violations consumes (i) the real src
# tree (default), (ii) a fixture source root injected by the standard test
# entry via `--architecture-only --architecture-root PATH` — canonical
# on-disk roots live under test/fixtures/arch_minimal_root (expected green)
# and test/fixtures/arch_bad_root (expected red: planted joinpath-dev,
# qualified Base.include-dev, and dynamic-variable include edges) — and
# (iii) in-memory known-bad fixtures.
#
# CURRENT STATE: the probe dev-migration HAS landed (src/ceiling_probes.jl
# is physically deleted; KTrader.jl includes no probes file and exports no
# probe symbols; probes live in dev/probes.jl, loaded explicitly by
# test-side consumers). The real-tree checks are therefore expected GREEN —
# any violation reported below is a real regression, not an expected-red
# window (the earlier "red until the migration lands" notes described the
# pre-landing window and are obsolete). The fixture self-checks prove the
# gate itself can go red. In fixture-root mode the runtime namespace checks
# (M1-D/E) are explicitly out of scope (they apply to the real module only)
# and are skipped with a stated @info — never silently.

using Test

const M1_DEFAULT_SRC_DIR = joinpath(dirname(@__DIR__), "src")

# Source root consumed by the tree scan: the real src tree by default, or the
# fixture root set by runtests.jl (--architecture-only --architecture-root
# PATH). Read defensively so this file also behaves when included without an
# entry-parsed override present.
_m1_src_root = M1_DEFAULT_SRC_DIR
if @isdefined(architecture_root_override) &&
   architecture_root_override isa AbstractString
    _m1_src_root = String(architecture_root_override)
end
const M1_SRC_ROOT = _m1_src_root
const M1_FIXTURE_ROOT_MODE = M1_SRC_ROOT != M1_DEFAULT_SRC_DIR

# Core loading: real-tree and full modes actually load KTrader (no
# pkgimage/freeze equivalence is assumed); fixture-root mode never loads it.
# `if` introduces no scope in Julia, so `using` stays at legal toplevel load
# semantics inside this static conditional.
if !M1_FIXTURE_ROOT_MODE
    using KTrader
end

if M1_FIXTURE_ROOT_MODE
    @info "M1 fixture-root mode: tree scan consumes $(M1_SRC_ROOT); KTrader is NOT loaded; runtime namespace checks (M1-D/E) apply only to the real module and are out of scope here"
end
m1_tree_label = M1_FIXTURE_ROOT_MODE ? "fixture root: $(M1_SRC_ROOT)" :
                                      "real tree (expected GREEN: the probe dev-migration has landed)"

# ---------------------------------------------------------------------------
# Static include-path resolution — the ONLY supported forms (see the syntax
# boundary at the top of this file). These helpers never evaluate untrusted
# Julia: they fold string literals, @__DIR__ / @__FILE__ (relative to the
# file currently being scanned), and literal-only joinpath / dirname chains
# into a filemap key. Anything else resolves to `nothing`, which the walk
# records as an explicit violation.
# ---------------------------------------------------------------------------

# dirname() normalized into the filemap's relative-key space ("." -> "").
_static_dir_of(path::AbstractString) =
    let d = dirname(path)
        (isempty(d) || d == ".") ? "" : String(d)
    end

# Callee name for statically supported calls: bare `f` or `Base.f`.
function _static_callee_name(c)
    c isa Symbol && return c
    if c isa Expr && c.head == :. && length(c.args) == 2 &&
       c.args[1] == :Base && c.args[2] isa QuoteNode && c.args[2].value isa Symbol
        return c.args[2].value
    end
    return nothing
end

function _resolve_static_path(x, cur_file::String)
    x isa String && return x
    x isa Expr || return nothing
    if x.head == :macrocall && !isempty(x.args)
        # Both macros are expressed in the INCLUDING file's own directory
        # basis ("." / the file's basename); the caller rebases exactly once,
        # so @__DIR__-based and bare-literal paths can never double-rebase.
        x.args[1] == Symbol("@__DIR__") && return "."
        x.args[1] == Symbol("@__FILE__") && return basename(cur_file)
        return nothing
    elseif x.head == :call && !isempty(x.args)
        name = _static_callee_name(x.args[1])
        if name == :joinpath
            parts = String[]
            for a in x.args[2:end]
                p = _resolve_static_path(a, cur_file)
                p === nothing && return nothing
                if startswith(p, "/")   # joinpath resets on absolute members
                    empty!(parts)
                end
                push!(parts, p)
            end
            joined = join(filter(p -> !isempty(p) && p != ".", parts), "/")
            return isempty(joined) ? nothing : joined
        elseif name == :dirname
            length(x.args) == 2 || return nothing
            p = _resolve_static_path(x.args[2], cur_file)
            p === nothing && return nothing
            return _static_dir_of(p)
        end
    end
    return nothing
end

# Fold a root-relative raw path into a canonical filemap key: "." segments
# drop, ".." pops its parent, absolute paths and root-escaping targets
# resolve to nothing (NOT supported — the walk reports them explicitly).
function _canonical_root_key(p::AbstractString)
    startswith(p, "/") && return nothing
    parts = String[]
    for seg in split(p, '/')
        isempty(seg) && continue
        seg == "." && continue
        if seg == ".."
            isempty(parts) && return nothing   # escapes the source root
            pop!(parts)
        else
            push!(parts, String(seg))
        end
    end
    return isempty(parts) ? nothing : join(parts, '/')
end

# Recursively collect every *.jl under root, keyed by root-relative path
# ("KTrader.jl", "dev/probes.jl", ...). Subdirectories are collected so a
# planted production->dev edge resolves to the precise dev/ violation
# instead of a generic missing-file violation; collected-but-unreached
# orphan files are never walked.
function _collect_source_map(root::String)
    srcmap = Dict{String,String}()
    isdir(root) || return srcmap
    for (dir, _subdirs, names) in walkdir(root)
        d = relpath(dir, root)
        for n in names
            endswith(n, ".jl") || continue
            key = d == "." ? n : joinpath(d, n)
            srcmap[key] = read(joinpath(dir, n), String)
        end
    end
    return srcmap
end

# The single helper of this gate (no framework beyond it): parse a module
# source map, walk the recursive include tree, and report violations.
# filemap: include-relative filename -> source text.
function m1_architecture_violations(module_filename::AbstractString,
                                    filemap::Dict{String,String})
    violations = String[]
    forbidden_default_includes = ("ceiling_probes.jl",)
    forbidden_exports = (:CeilingProbeCollector, :print_probe_report,
                         :mask_transition_coverage, :solve_floor_probe,
                         :sequential_ceiling_probe, :PROBE_COVERAGE_WINDOWS)
    forbidden_idents = (:OOFCounters, :oof_shadow, :OOFShadowConfigError,
                        :oof_shadow_identity_check!)

    includes_found = String[]
    exports_found = Symbol[]
    idents_found = Symbol[]
    seen = Set{String}()

    function walk(x, cur_file::String)
        if x isa Symbol
            if x == :include
                # Supported callees never reach here (the include call branch
                # does not recurse into the callee), so a bare `include`
                # symbol at this point is a dynamic use of include as a VALUE.
                push!(violations,
                      "bare `include` used as a value (dynamic include form) in $(cur_file)")
            end
            push!(idents_found, x)
        elseif x isa QuoteNode
            x.value isa Symbol && push!(idents_found, x.value)
        elseif x isa Expr
            if x.head == :export
                for a in x.args
                    a isa Symbol && push!(exports_found, a)
                end
            elseif x.head == :call && !isempty(x.args)
                if _static_callee_name(x.args[1]) == :include
                    nargs = length(x.args) - 1
                    if !(nargs in (1, 2))
                        push!(violations,
                              "unsupported include arity ($(nargs) argument(s)) in $(cur_file): only include(path) or include(mod, path) are statically supported")
                    else
                        path_expr = x.args[end]   # the path is the last argument
                        path = _resolve_static_path(path_expr, cur_file)
                        if path === nothing
                            push!(violations,
                                  "unsupported include form (path is not a statically resolvable literal; accepted: string literal, @__DIR__/@__FILE__, literal-only joinpath/dirname chains) in $(cur_file): $(path_expr)")
                        else
                            # Single rebase from the including file's own
                            # directory into the root-relative raw key; the
                            # canonical fold happens in scan_file, AFTER the
                            # as-written dev-segment check consumes raw keys.
                            curdir = _static_dir_of(cur_file)
                            push!(includes_found,
                                  isempty(curdir) ? path : joinpath(curdir, path))
                        end
                    end
                    # Recurse the non-callee arguments (module argument and
                    # path expression) so identifier collection still sees
                    # them; string literals record nothing as identifiers.
                    for a in x.args[2:end]
                        walk(a, cur_file)
                    end
                    return nothing
                end
            elseif x.head == :. && length(x.args) == 2
                if x.args[1] isa Symbol && x.args[1] == :include
                    # include in dot position: broadcast include.(...) or a
                    # property access on the include function itself.
                    push!(violations,
                          "unsupported include form (`include` in dot position: broadcast `include.` or property access on `include`) in $(cur_file)")
                elseif x.args[2] isa QuoteNode && x.args[2].value == :include
                    # A qualified X.include OUTSIDE a supported call callee
                    # position (supported Base.include callees are never
                    # recursed into, so this cannot false-positive on them):
                    # e.g. Foo.include(...) as a callee, or Base.include
                    # passed around as a value.
                    push!(violations,
                          "unsupported include form (qualified `$(x.args[1]).include` outside a supported call callee position; only Base.include is accepted) in $(cur_file)")
                end
            end
            for a in x.args
                walk(a, cur_file)
            end
        end
        return nothing
    end

    function scan_file(raw_path::String)
        key = _canonical_root_key(raw_path)
        if key === nothing
            push!(violations,
                  "include target escapes the source root or is absolute (not statically supported; every target must resolve inside the scanned source map): $(raw_path)")
            return
        end
        key in seen && return
        push!(seen, key)
        src = get(filemap, key, nothing)
        if src === nothing
            push!(violations,
                  "include target not present in source map (path must exist): $(raw_path) -> $(key)")
            return
        end
        walk(Meta.parseall(src), key)
    end

    scan_file(module_filename)
    i = 1
    while i <= length(includes_found)
        scan_file(includes_found[i])
        i += 1
    end

    for inc in includes_found
        if basename(inc) in forbidden_default_includes
            push!(violations,
                  "production include tree has default dev probes include: $(inc)")
        end
        if any(seg -> seg == "dev", split(inc, '/'))
            push!(violations, "production module includes dev/ path: $(inc)")
        end
    end
    for e in exports_found
        if e in forbidden_exports
            push!(violations,
                  "KTrader exports forbidden profiling symbol: $(e)")
        end
    end
    for s in idents_found
        if s in forbidden_idents
            push!(violations,
                  "forbidden profiling identifier in production include tree: $(s)")
        end
    end
    return violations
end

@testset "M1-A/B/C: production include tree AST scan — $m1_tree_label" begin
    srcmap = _collect_source_map(M1_SRC_ROOT)
    @testset "src tree non-empty and module file present (fail-closed)" begin
        @test length(srcmap) > 0
        @test haskey(srcmap, "KTrader.jl")
    end
    violations = m1_architecture_violations("KTrader.jl", srcmap)
    if !isempty(violations)
        @warn "M1 architecture violations (the probe dev-migration has landed; the real tree is expected GREEN — these are real regressions)" violations
    end
    @test violations == String[]  # failure output lists each violation
end

if !M1_FIXTURE_ROOT_MODE
    @testset "M1-D: PreparedProblem typed boundary with single solve (runtime metadata)" begin
        @test isdefined(KTrader, :PreparedProblem)
        if isdefined(KTrader, :PreparedProblem)
            @test hasmethod(KTrader.solve, Tuple{KTrader.PreparedProblem})
        end
    end

    @testset "M1-E: profiling types absent from production runtime namespace" begin
        @test !isdefined(KTrader, :OOFShadowConfigError)
        @test !isdefined(KTrader, :CeilingProbeCollector)
    end
end

@testset "M1 gate self-check: known-bad fixtures go red via the same primitive" begin
    # planted production -> dev include (bare literal)
    bad_dev = Dict{String,String}(
        "KTrader.jl" => "module KTrader\ninclude(\"dev/probes.jl\")\nend\n",
        "dev/probes.jl" => "module DevProbes\nend\n",
    )
    v = m1_architecture_violations("KTrader.jl", bad_dev)
    @test any(s -> occursin("dev/ path", s), v)

    # planted production -> dev include via joinpath(@__DIR__, ...) — the old
    # walk silently IGNORED this form; it must now be CAUGHT as a dev/ edge.
    bad_dev_joinpath = Dict{String,String}(
        "KTrader.jl" =>
            "module KTrader\ninclude(joinpath(@__DIR__, \"dev\", \"probes.jl\"))\nend\n",
        "dev/probes.jl" => "module DevProbes\nend\n",
    )
    v = m1_architecture_violations("KTrader.jl", bad_dev_joinpath)
    @test any(s -> occursin("dev/ path", s), v)

    # planted production -> dev include via qualified Base.include with a
    # module argument — also silently ignored by the old walk.
    bad_qualified = Dict{String,String}(
        "KTrader.jl" =>
            "module KTrader\nBase.include(Main, joinpath(\"dev\", \"probes.jl\"))\nend\n",
        "dev/probes.jl" => "module DevProbes\nend\n",
    )
    v = m1_architecture_violations("KTrader.jl", bad_qualified)
    @test any(s -> occursin("dev/ path", s), v)

    # dynamic include (variable path) — must be REJECTED, not silently ignored.
    bad_dynamic_var = Dict{String,String}(
        "KTrader.jl" =>
            "module KTrader\nf = \"data.jl\"\ninclude(f)\nend\n",
    )
    v = m1_architecture_violations("KTrader.jl", bad_dynamic_var)
    @test any(s -> occursin("unsupported include form", s), v)

    # interpolated path — not a literal, must be rejected.
    bad_interp = Dict{String,String}(
        "KTrader.jl" =>
            "module KTrader\nname = \"data\"\ninclude(\"\$(name).jl\")\nend\n",
    )
    v = m1_architecture_violations("KTrader.jl", bad_interp)
    @test any(s -> occursin("unsupported include form", s), v)

    # broadcast include — must be rejected.
    bad_broadcast = Dict{String,String}(
        "KTrader.jl" =>
            "module KTrader\nfiles = (\"data.jl\",)\ninclude.(files)\nend\n",
    )
    v = m1_architecture_violations("KTrader.jl", bad_broadcast)
    @test any(s -> occursin("unsupported include form", s), v)

    # bare `include` passed as a value — must be rejected.
    bad_bare_value = Dict{String,String}(
        "KTrader.jl" =>
            "module KTrader\nfiles = (\"data.jl\",)\nmap(include, files)\nend\n",
    )
    v = m1_architecture_violations("KTrader.jl", bad_bare_value)
    @test any(s -> occursin("dynamic include form", s), v)

    # foreign qualified callee (Foo.include) — only Base.include is supported.
    bad_foreign_qualified = Dict{String,String}(
        "KTrader.jl" =>
            "module KTrader\nFoo.include(\"data.jl\")\nend\n",
    )
    v = m1_architecture_violations("KTrader.jl", bad_foreign_qualified)
    @test any(s -> occursin("outside a supported call callee position", s), v)

    # planted default probes include
    bad_probes = Dict{String,String}(
        "KTrader.jl" => "module KTrader\ninclude(\"ceiling_probes.jl\")\nend\n",
        "ceiling_probes.jl" => "module Probes\nend\n",
    )
    v = m1_architecture_violations("KTrader.jl", bad_probes)
    @test any(s -> occursin("default dev probes", s), v)

    # planted forbidden profiling keyword (oof_shadow kw arg)
    bad_kw = Dict{String,String}(
        "KTrader.jl" => "module KTrader\ninclude(\"predict.jl\")\nend\n",
        "predict.jl" => "fit(; oof_shadow = nothing) = 1\n",
    )
    v = m1_architecture_violations("KTrader.jl", bad_kw)
    @test any(s -> occursin("oof_shadow", s), v)

    # planted forbidden profiling type
    bad_type = Dict{String,String}(
        "KTrader.jl" => "module KTrader\ninclude(\"response.jl\")\nend\n",
        "response.jl" => "struct OOFShadowConfigError <: Exception\nend\n",
    )
    v = m1_architecture_violations("KTrader.jl", bad_type)
    @test any(s -> occursin("OOFShadowConfigError", s), v)

    # planted forbidden profiling export
    bad_export = Dict{String,String}(
        "KTrader.jl" => "module KTrader\nexport CeilingProbeCollector\nend\n",
    )
    v = m1_architecture_violations("KTrader.jl", bad_export)
    @test any(s -> occursin("CeilingProbeCollector", s), v)

    # missing include target (every include path must truly exist — no fail-open)
    bad_missing = Dict{String,String}(
        "KTrader.jl" => "module KTrader\ninclude(\"ghost.jl\")\nend\n",
    )
    v = m1_architecture_violations("KTrader.jl", bad_missing)
    @test any(s -> occursin("not present in source map", s), v)

    # Nested dev escape via "..": a subdirectory file reaching a dev/ target
    # through parent segments — the raw written path carries the dev segment,
    # so the edge is identified BEFORE canonical folding (folding must never
    # hide it); the root-escaping variant additionally reports unsupported.
    bad_nested_dev = Dict{String,String}(
        "KTrader.jl" =>
            "module KTrader\ninclude(\"sub/escape.jl\")\ninclude(\"sub/escape2.jl\")\nend\n",
        "sub/escape.jl" =>
            "include(joinpath(@__DIR__, \"..\", \"..\", \"dev\", \"probes.jl\"))\n",
        "sub/escape2.jl" =>
            "include(joinpath(@__DIR__, \"..\", \"dev\", \"more.jl\"))\ninclude(\"../dev/more.jl\")\n",
        "dev/more.jl" => "module DevMore\nend\n",
    )
    v = m1_architecture_violations("KTrader.jl", bad_nested_dev)
    @test any(s -> occursin("dev/ path", s), v)

    # Root-escaping non-dev target: NOT claimed as supported — explicit reject.
    bad_escape_root = Dict{String,String}(
        "KTrader.jl" =>
            "module KTrader\ninclude(\"../outside.jl\")\nend\n",
    )
    v = m1_architecture_violations("KTrader.jl", bad_escape_root)
    @test any(s -> occursin("escapes the source root", s), v)
end

@testset "M1 gate self-check: legal minimal structure stays green" begin
    ok = Dict{String,String}(
        "KTrader.jl" =>
            "module KTrader\ninclude(\"data.jl\")\ninclude(\"prepare.jl\")\ninclude(\"predict.jl\")\nend\n",
        "data.jl" => "struct Bars\nend\n",
        "prepare.jl" => "struct PreparedProblem\nend\nsolve(p::PreparedProblem) = 1\n",
        "predict.jl" => "fit(p) = 2\n",
    )
    @test m1_architecture_violations("KTrader.jl", ok) == String[]

    # Every statically supported include form, exercised together: bare
    # literal, joinpath over literals / @__DIR__ / dirname(@__FILE__), a
    # "."-segment joinpath, a subdirectory joinpath, the 2-argument module
    # form, and qualified Base.include. All must resolve and stay green.
    ok_forms = Dict{String,String}(
        "KTrader.jl" =>
            "module KTrader\n" *
            "include(joinpath(@__DIR__, \"data.jl\"))\n" *
            "include(joinpath(dirname(@__FILE__), \"prepare.jl\"))\n" *
            "include(joinpath(\".\", \"kelly.jl\"))\n" *
            "include(joinpath(\"sub\", \"predict.jl\"))\n" *
            "Base.include(Main, \"broker.jl\")\n" *
            "include(Main, \"live.jl\")\n" *
            "end\n",
        "data.jl" => "struct Bars\nend\n",
        "prepare.jl" => "struct PreparedProblem\nend\nsolve(p::PreparedProblem) = 1\n",
        "kelly.jl" => "k(p) = 3\n",
        "sub/predict.jl" => "fit(p) = 2\n",
        "broker.jl" => "b(p) = 4\n",
        "live.jl" => "l(p) = 5\n",
    )
    @test m1_architecture_violations("KTrader.jl", ok_forms) == String[]

    # Nested legal structure: a subdirectory include whose targets sit ABOVE
    # it (".." to the root) — the two semantically identical spellings
    # (bare "../data.jl" and joinpath(@__DIR__, "..", "data.jl")) must
    # converge on the SAME canonical key (no double rebase, no false
    # missing-target), alongside a deeper nested include below it.
    ok_nested = Dict{String,String}(
        "KTrader.jl" =>
            "module KTrader\ninclude(\"sub/a.jl\")\nend\n",
        "sub/a.jl" =>
            "include(\"../data.jl\")\ninclude(joinpath(@__DIR__, \"..\", \"data.jl\"))\ninclude(joinpath(@__DIR__, \"..\", \"shared.jl\"))\ninclude(joinpath(\"inner\", \"leaf.jl\"))\n",
        "data.jl" => "d() = 1\n",
        "shared.jl" => "s() = 2\n",
        "sub/inner/leaf.jl" => "l() = 3\n",
    )
    @test m1_architecture_violations("KTrader.jl", ok_nested) == String[]
end
