# Frozen certificate owner at response SHA256 c0d005c5c4d9abde3fb93397af6549f111268e73da0e5b034e85fb538b49ecc2.
# The direct scalar/evidence APIs below retain their own implementation;
# this reference deliberately performs the old duplicate geometry/full grad.
function certificate_eager_reference(spectrum,YtY,n,alpha,Sigma;gauge=nothing,tol=1e-6)
    scalar=KTrader.conditioned_evidence_cache(spectrum,YtY,n,Sigma;gauge)
    slope=scalar.derivative(log(alpha))
    rank=gauge===nothing ? size(Sigma,1) : size(gauge,2)
    scale=max(rank*sum(spectrum.values./(spectrum.values.+alpha)),2length(KTrader.BANDS),1.0)
    ac=KTrader.bounded_alpha_certificate(alpha,slope,scale;tol)
    state=KTrader.conditioned_evidence(spectrum,YtY,n,alpha,Sigma;gradient=true,gauge)
    sc=KTrader.covariance_certificate(Sigma,state.direction;gauge,tol)
    (;valid=ac.valid && sc.valid && isfinite(state.value),alpha=ac,covariance=sc,evidence=state.value)
end
