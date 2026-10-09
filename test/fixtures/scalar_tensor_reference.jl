# Frozen pre-change _conditioned_scalar_cache, from response SHA256
# 69733860bb42bd18482619b81262976b95bcbcd625ee3b34f482caac21265ac5.
# Test/dev only. Same fit geometry on both sides, no geometry-cost hiding.
function scalar_tensor_reference(blocks,S)
    nc=length(blocks); rank=size(first(blocks),2)
    tensors=zeros(rank,nc,nc)
    for r in 1:nc
        transformed=S*blocks[r]
        for s in r:nc,k in 1:rank
            tensors[k,r,s]=tensors[k,s,r]=dot(view(blocks[s],:,k),view(transformed,:,k))
        end
    end
    tensors
end

function scalar_cache_reference(spectrum,n,Sigma,geometry;gauge=nothing)
    (; B,R,blocks,hcoef)=geometry
    nc=length(blocks); rank=length(spectrum.values)
    S=gauge===nothing ? Sigma : gauge'*Sigma*gauge
    chol=cholesky(Symmetric(S)); solved=chol\B'
    quadratic=[dot(view(B,k,:),view(solved,:,k)) for k in 1:rank]
    tensors=scalar_tensor_reference(blocks,S)
    constant=-n/2*logdet(chol)-0.5tr(chol\R)
    function moments(logalpha)
        alpha=exp(logalpha); d=1.0 ./ (spectrum.values .+ alpha)
        baseline=spectrum.dual ? 1/alpha : 0.0
        weights=spectrum.dual ? -spectrum.values .* d ./ alpha : d
        M=zeros(nc,nc)
        for r in 1:nc,s in r:nc
            M[r,s]=M[s,r]=dot(weights,view(tensors,:,r,s))
        end
        for r in 1:nc
            M[r,r]+=baseline*tr(S)
        end
        (; alpha,d,baseline,M,h=hcoef'*d)
    end
    function evidence(logalpha)
        state=moments(logalpha); cholM=cholesky(Symmetric(state.M))
        constant-size(S,1)/2*sum(log1p.(spectrum.values./state.alpha))+0.5dot(state.d,quadratic)-
            0.5logdet(cholM)-0.5dot(state.h,cholM\state.h)+nc/2*log(tr(S)/state.alpha)
    end
    function derivative(logalpha)
        state=moments(logalpha); alpha=state.alpha; d=state.d
        cholM=cholesky(Symmetric(state.M)); z=cholM\state.h
        dd=-alpha.*d.^2
        wd=spectrum.dual ? spectrum.values .* d .* (d .+ 1/alpha) : dd
        Md=zeros(nc,nc)
        for r in 1:nc,s in r:nc
            Md[r,s]=Md[s,r]=dot(wd,view(tensors,:,r,s))
        end
        for r in 1:nc
            Md[r,r]-=state.baseline*tr(S)
        end
        size(S,1)/2*sum(spectrum.values.*d)+0.5dot(dd,quadratic)-
            0.5tr(cholM\Md)+0.5dot(z,Md*z)-dot(z,hcoef'*dd)-nc/2
    end
    (; evidence,derivative)
end
