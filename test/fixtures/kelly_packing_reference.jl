# Frozen original scenario_weights boundary: indexed view reaches the
# original certified Kelly solver. Test/benchmark oracle only.
function kelly_indexed_reference(X,active_indices,tradable,held=nothing;tol=1e-8)
    N=size(X,2)
    active=falses(N); active[active_indices].=true
    free=tradable .& active
    current=held===nothing ? zeros(N) : held
    locked=current .* .!free
    budget=1.0-sum(locked)
    indices=findall(free)
    (isempty(indices) || budget<=1e-12) && return copy(current)
    base=KTrader.locked_wealth(X,locked)
    out=copy(locked)
    out[indices].=KTrader.kelly_weights_v1(view(X,:,indices);budget,base,tol)
    out
end
