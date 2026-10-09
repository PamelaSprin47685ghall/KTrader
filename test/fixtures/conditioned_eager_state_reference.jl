# Before-change eager formula, response.jl SHA256 adde30b78d89e7298f3e33cbf2955eaa80afc4861bac968c0b770d45f2af0546.
# Only the evaluation order/unused gradient changes; M/J owners are unchanged.
function eager_state_reference(cache,S;M=nothing)
    M===nothing && (M=KTrader.conditioned_constraint_contraction(cache,S))
    cholM=cholesky(Symmetric(M)); cholS=cholesky(Symmetric(S))
    z=cholM\cache.h; nc=length(cache.h)
    value=cache.constant-cache.n/2*logdet(cholS)-0.5tr(cholS\cache.R)-
          0.5logdet(cholM)-0.5dot(cache.h,z)+nc/2*log(tr(S)/cache.alpha)
    J=KTrader.conditioned_constraint_jacobian(cache,cholM,z)
    evS=eigen(Symmetric(S))
    Λ=evS.values; V=evS.vectors
    VRV=V'*cache.R*V
    VJV=V'*J*V
    G=(-cache.n ./ Λ) .* Matrix{Float64}(I,size(S,1),size(S,1))
    G .+= VRV ./ (Λ .* Λ')
    G .-= VJV
    G ./= 2
    trS=tr(S)
    for j in axes(G,1)
        G[j,j]+=nc/(2trS)
    end
    grad=V*G*V'
    direction=(-cache.n.*S+cache.R-S*J*S+(nc/trS).*(S*S))./cache.n
    (; value,grad=Matrix(Symmetric(grad)),direction=Matrix(Symmetric(direction)),M,J)
end
