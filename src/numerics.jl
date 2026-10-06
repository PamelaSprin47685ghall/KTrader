const TIMING_BUCKETS = (:prep, :basis, :gram, :eigen, :EB, :condition, :fracFFT, :scenario, :Kelly)

mutable struct DecisionTiming
    seconds::Vector{Float64}
end
DecisionTiming() = DecisionTiming(zeros(length(TIMING_BUCKETS)))
mutable struct FitWorkspace
    matrices::Dict{Tuple{Symbol,Int},Matrix{Float64}}
end
FitWorkspace() = FitWorkspace(Dict{Tuple{Symbol,Int},Matrix{Float64}}())
function matrix_buffer!(workspace::Union{Nothing,FitWorkspace},name::Symbol,rows::Int,columns::Int;
                        slot=0,grow_rows=false)
    workspace === nothing && return zeros(rows,columns)
    key=(name,slot)
    storage=get(workspace.matrices,key,nothing)
    # Allocate exact dimensions so memory stride strictly equals rows (leading dimension = rows).
    # A non-matching leading dimension causes OpenBLAS Level-3 GEMM/SYRK to degrade by 4.7x!
    if storage === nothing || size(storage, 2) != columns || size(storage, 1) != rows
        storage = Matrix{Float64}(undef, rows, columns)
        workspace.matrices[key]=storage
    end
    storage
end

function timed(f, timing::Union{Nothing,DecisionTiming}, bucket::Symbol)
    timing === nothing && return f()
    i = something(findfirst(==(bucket), TIMING_BUCKETS))
    start = time_ns()
    try
        f()
    finally
        timing.seconds[i] += (time_ns() - start) * 1e-9
    end
end

struct PriceHistoryCache
    log_prices::Matrix{Float64}
    returns::Matrix{Float64}
    first_price::Vector{Int}
    first_return::Vector{Int}
    alive_returns::Vector{Int}
    inv_sqrt_alive::Vector{Float64}
end
function PriceHistoryCache(prices::AbstractMatrix{Float64})
    T, N = size(prices)
    logs = log.(prices)
    returns = diff(logs; dims=1)
    first_price = [something(findfirst(isfinite, view(logs, :, j)), T + 1) for j in 1:N]
    first_return = [something(findfirst(isfinite, view(returns, :, j)), T) for j in 1:N]
    alive_returns = [count(isfinite, view(returns, t, :)) for t in 1:(T - 1)]
    inv_sqrt_alive = [cnt > 0 ? 1.0 / sqrt(cnt) : 0.0 for cnt in alive_returns]
    PriceHistoryCache(logs, returns, first_price, first_return, alive_returns, inv_sqrt_alive)
end

# A spectral representation of (X'X + alpha I)^-1, including the dual null space.
struct RidgeCovariance
    basis::Matrix{Float64}
    weights::Vector{Float64}
    baseline::Float64
end

const RELATIVE_GAUGE_LOCK = ReentrantLock()
const RELATIVE_GAUGES = Dict{Int,Matrix{Float64}}()
function relative_gauge(N::Int)
    lock(RELATIVE_GAUGE_LOCK) do
        get!(RELATIVE_GAUGES,N) do
            Q=zeros(N,max(N-1,0))
            for k in 1:N-1
                scale=sqrt(k*(k+1))
                Q[1:k,k].=1/scale
                Q[k+1,k]=-k/scale
            end
            Q
        end
    end
end
function covariance_product(V::RidgeCovariance, x::AbstractVecOrMat{Float64})
    z = V.basis' * x
    z .*= V.weights
    out = V.basis * z
    V.baseline == 0 || (out .+= V.baseline .* x)
    out
end
function covariance_right_product(x::AbstractMatrix{Float64}, V::RidgeCovariance)
    z = x * V.basis
    z .*= V.weights'
    out = z * V.basis'
    V.baseline == 0 || (out .+= V.baseline .* x)
    out
end

# Per-call FFT storage is leased by a task, not by threadid(): Julia tasks may migrate.
mutable struct FractionalFFTWorkspace{F,I}
    input::Vector{Float64}
    spectrum::Vector{ComplexF64}
    product::Vector{ComplexF64}
    output::Vector{Float64}
    forward::F
    inverse::I
    kernels::Vector{Vector{ComplexF64}}
    counts::Vector{Vector{Float64}}
end
const FRACTIONAL_FFT_LOCK = ReentrantLock()
const FRACTIONAL_FFT_POOL = Dict{Int,Vector{FractionalFFTWorkspace}}()
function fractional_fft_workspace(L::Int)
    lock(FRACTIONAL_FFT_LOCK) do
        pool = get!(FRACTIONAL_FFT_POOL, L, FractionalFFTWorkspace[])
        !isempty(pool) && return pop!(pool)
        input = zeros(L)
        forward = plan_rfft(input; flags=FFTW.ESTIMATE)
        spectrum = forward * input
        product = similar(spectrum)
        inverse = plan_irfft(spectrum, L; flags=FFTW.ESTIMATE)
        kernels = Vector{Vector{ComplexF64}}()
        counts = Vector{Vector{Float64}}()
        for d in DGRID_V1
            weights = frac_weights(d, L ÷ 2)
            fill!(input, 0.0)
            input[1:length(weights)] .= weights
            push!(kernels, forward * input)
            push!(counts, cumsum(weights))
        end
        FractionalFFTWorkspace(input, spectrum, product, zeros(L), forward, inverse, kernels, counts)
    end
end
function release_fractional_fft_workspace(work::FractionalFFTWorkspace)
    # Do not serialize native plan pointers in a precompiled module image.
    ccall(:jl_generating_output,Cint,()) == 1 && return nothing
    lock(FRACTIONAL_FFT_LOCK) do
        push!(FRACTIONAL_FFT_POOL[length(work.input)], work)
    end
end
