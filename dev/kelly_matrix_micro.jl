# Diagnostic GEMM/GEMV cost on real archived 32k×64 scenarios. No solver/PnL.
using Dates, Serialization, LinearAlgebra, Printf
BLAS.set_num_threads(parse(Int,get(ENV,"PROBE_BLAS_THREADS","4")))
p=deserialize(get(ENV,"PROBE_CAPTURE_PATH",
                  "/tmp/ktrader-two-year-20261011/one-day-qmc-32768.jls"))
X=p.X8192
S,n=size(X)
A=hcat(X,ones(S))
m=n+1
w=fill(1/m,m)
z=A*w
invz=1.0 ./z
scaled=similar(A)
@inbounds for j in 1:m, i in 1:S
    scaled[i,j]=A[i,j]*invz[i]
end
H=zeros(m,m)
g=zeros(m)

function micro(label, reps, f)
    f()  # JIT warm-up outside timing
    t0=time_ns()
    for i in 1:reps
        f()
    end
    @printf("MATRIX_MICRO op=%s reps=%d mean_ms=%.6f\n",
            label,reps,(time_ns()-t0)/1e6/reps)
    flush(stdout)
end

println("MATRIX_MICRO_SETUP S=$S m=$m blas=$(BLAS.get_num_threads())")
micro("GEMM",20,()->mul!(H,transpose(scaled),scaled))
micro("GEMV",60,()->mul!(g,transpose(A),invz))
micro("FORWARD",60,()->mul!(z,A,w))
micro("SCALE",30,()->(@inbounds for j in 1:m, i in 1:S
                          scaled[i,j]=A[i,j]*invz[i]
                       end))
