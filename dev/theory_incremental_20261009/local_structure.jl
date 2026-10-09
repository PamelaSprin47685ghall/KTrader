# Two authenticated adjacent dates: prepare/spectrum/certificate geometry
# only, NO optimize_conditioned_eb, solve, fit_v1 or backtest.
using KTrader, LinearAlgebra, SHA, TOML, Serialization
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
digest(p)=bytes2hex(open(sha256,p))
function sources()
    Dict(joinpath("src",f)=>digest(joinpath(ROOT,"src",f)) for f in readdir(joinpath(ROOT,"src")) if endswith(f,".jl"))
end
function saved_day(i,current)
    p=joinpath(ROOT,"dev","evidence","earlier_closure_20261009","fit_day_$(i).jls")
    meta=TOML.parsefile(p*".toml"); bytes=read(p)
    @assert bytes2hex(sha256(bytes))==meta["sha256"] && length(bytes)==meta["bytes"]
    @assert meta["sources"]==current
    for (p,h) in meta["inputs"]; @assert digest(joinpath(ROOT,p))==h; end
    record=deserialize(IOBuffer(bytes)); @assert record.sources==current
    record
end
function observation_support(r,cutoff)
    N=size(r,2); parent=collect(1:N)
    function root(i)
        while parent[i]!=i; i=parent[i]; end
        i
    end
    for t in 1:cutoff
        seen=findall(isfinite,view(r,t,:))
        isempty(seen) && continue
        for j in seen[2:end]; parent[root(j)]=root(first(seen)); end
    end
    groups=Dict{Int,Vector{Int}}()
    for j in 1:N; push!(get!(groups,root(j),Int[]),j); end
    blocks=Matrix{Float64}[]
    for group in sort!(collect(values(groups));by=first)
        length(group)==1 && continue
        Z=zeros(N,length(group)-1); Z[group,:]=KTrader.relative_gauge(length(group))
        push!(blocks,Z)
    end
    isempty(blocks) ? zeros(N,0) : reduce(hcat,blocks)
end
function structure_main()
    isempty(ARGS) || error("fixed two-date geometry diagnostic")
    BLAS.set_num_threads(6); current=sources()
    a=saved_day(1,current); b=saved_day(2,current)
    bars=KTrader.load_bars(joinpath(ROOT,"data")); signal=KTrader.signal_prices(bars)
    old=KTrader.prepare_reference(view(signal,1:a.t,:);F_folds=3)
    new=KTrader.prepare_reference(view(signal,1:b.t,:);F_folds=3)
    @assert old.N==new.N==65 && new.T==old.T+1
    d0=1.0./max.(old.s1,1e-6); d1=1.0./max.(new.s1,1e-6)
    ell=d1./d0
    channel_ratio=repeat(max.(old.s_perp,1e-6)./max.(new.s_perp,1e-6);inner=2)
    diagonal=vcat([r.*ell for r in channel_ratio]...)
    common=view(new.X_rel_stacked,1:old.n_res,:)
    E=common-old.X_rel_stacked.*diagonal'
    singular=sqrt.(max.(eigvals(Symmetric(E'*E)),0.0))
    # Eigenvalues of E'E lose precision at tiny singular values: this count
    # uses a conservative1e-6 relative singular threshold, NOT exact rank.
    numerical=count(>(maximum(singular)*1e-6),singular)
    masks=unique([Tuple(isfinite.(view(old.r,t,:))) for t in axes(old.r,1)])
    nonempty=filter(m->any(m),masks)
    M=hcat([Float64.(collect(m)) for m in nonempty]...)
    span=hcat(M,ell.*M)
    println("DATES ",a.date," -> ",b.date," T=",old.T," -> ",new.T," N65 P910; zero posterior fits")
    println("mask_count=",length(nonempty)," mask_span_rank=",rank(M;rtol=1e-10),
        " transported_mask_span_rank=",rank(span;rtol=1e-10))
    println("transported_design_residual_relative=",norm(E)/norm(common),
        " numerical_rank_at_1e-6_singular=",numerical,
        " general_bound=",min(new.P_features,14*rank(span;rtol=1e-10)))
    println("raw-coordinate rank is not a proof of low transported rank; no rank is truncated")
    E=nothing; common=nothing; GC.gc()
    Q=KTrader.relative_gauge(65)
    for (record,prep) in ((a,old),(b,new))
        models=vcat([record.model.resp],record.model.res_history.fold_models)
        for f in 0:3
            response=models[f+1]; S=Matrix(Symmetric(Q'*response.Sigma_rel*Q))
            lambda=eigvals(Symmetric(S))
            roundoff=128eps(Float64)*max(opnorm(S),KTrader.EB_COVARIANCE_FLOOR)
            active=count(<=(KTrader.EB_COVARIANCE_FLOOR+roundoff),lambda)
            xx=f==0 ? prep.stats.full_xx : prep.stats.full_xx-prep.stats.xx[f]
            xy=f==0 ? prep.stats.full_xy : prep.stats.full_xy-prep.stats.xy[f]
            yy=f==0 ? prep.stats.full_yy : prep.stats.full_yy-prep.stats.yy[f]
            n=f==0 ? prep.n_res : prep.n_res-length(prep.stats.ranges[f])
            spectrum=KTrader.ridge_spectrum(nothing,xx,xy)
            cache=KTrader.conditioned_alpha_cache(spectrum,yy,n,response.alpha_rel;gauge=Q)
            L=Matrix(cholesky(Symmetric(S)).L)
            rvalues=eigvals(Symmetric(L\cache.R/L'))
            normalized_base=(rvalues .+ rvalues' .- n)./(2n)
            println("date=",record.date," fit=",f," alpha=",response.alpha_rel,
                " floor_active=",active," of64",
                " base_hessian_min_over_n=",minimum(normalized_base),
                " base_min_abs_denominator_over_n=",minimum(abs,normalized_base))
            if f==3
                training=setdiff(1:prep.n_res,prep.stats.ranges[f])
                cutoff=maximum(prep.ts_total[training])+1
                Z=observation_support(prep.r,cutoff)
                Pz=Z*Z'; Pa=Q*Q'-Pz; a=64-size(Z,2)
                feature_projection=kron(Matrix{Float64}(I,14,14),Pz)
                xtrain=prep.X_rel_stacked[training,:]
                ytrain=prep.relative_embedding[prep.ts_total[training].+1,:]
                println("CONSTRUCTIVE_SUPPORT r=",size(Z,2)," null_relative=",a,
                    " feature_leak=",norm(xtrain-xtrain*feature_projection)/norm(xtrain),
                    " target_leak=",norm(ytrain-ytrain*Pz)/norm(ytrain),
                    " sigma_cross=",norm(Pa*response.Sigma_rel*Pz)/norm(response.Sigma_rel),
                    " null_floor_error=",norm(Pa*response.Sigma_rel*Pa-KTrader.EB_COVARIANCE_FLOOR*Pa))
                M=KTrader._conditioned_constraint_direct(cache,S)
                multiplier=cholesky(Symmetric(M))\cache.h
                B=Matrix(Symmetric(Z'*response.Sigma_rel*Z))
                Rz=Matrix(Symmetric(Z'*Q*cache.R*Q'*Z))
                Lz=Matrix(cholesky(Symmetric(B)).L)
                restricted=eigvals(Symmetric(Lz\Rz/Lz'))
                null_score=-n/(2KTrader.EB_COVARIANCE_FLOOR)-
                    (tr(inv(Symmetric(M)))-dot(multiplier,multiplier))/(2response.alpha_rel)+
                    length(cache.h)/(2tr(S))
                println("REDUCED_FACE base_min_over_n=",minimum(restricted)/n-0.5,
                    " likelihood_feature_dimension=",14size(Z,2),
                    " original_output_dimension_kept=64",
                    " null_evidence_gradient=",null_score,
                    " null_one_sided_KKT=",null_score<=0)
                mean_error=0.0; null_mean=0.0
                for c in eachindex(response.constraint_cols)
                    actual=Pa*response.G_c_mean[:,response.constraint_cols[c]]*Pa
                    expected=-(KTrader.EB_COVARIANCE_FLOOR/response.alpha_rel)*multiplier[c].*Pa
                    mean_error=max(mean_error,norm(actual-expected))
                    null_mean=max(null_mean,norm(actual))
                end
                println("NULL_PRIOR trace-compensation mean norm=",null_mean,
                    " analytic block error=",mean_error,
                    "; these coefficients are NOT dropped or set to zero")
            end
        end
    end
    @assert current==sources()
    println("runtime hashes unchanged; structural audit only, not an incremental solver benchmark")
end
if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    structure_main()
end
