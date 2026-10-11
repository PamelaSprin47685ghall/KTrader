# =============================================================================
# KTraderGate0 几何原语层 — Gate-0 纠偏新线（施工图 Step 3 交付物）
# =============================================================================
#
# 文件地位（裁决 H1/H2，docs/GATE0_MANAGER_ADJUDICATIONS.md）：本文件是未来
# 顶层 module `KTraderGate0`（src/gate0/ 目录线九文件切分）的组成文件；主
# module 文件由后续施工步骤建立。当前按「裸文件 + 来源标注」交付，测试经
# 临时 module include 加载（test/gate0/modes_tests.jl）。与旧线 src/ 的关系：
# 互不 include、不跨 module 依赖。
#
# 复制裁决 H7：ruler、relative_gauge、principal_sqrt_root、path_basis_1d、
# center_of_mass、TAUS/BANDS/WARMUP 等确定性纯函数复制进入 src/gate0/ 并
# 标注来源（file:line）；不跨 module 依赖——旧线冻结为历史快照不是活跃
# owner，复制不构成双 owner；单一真源由「来源标注 + 静态审查逐函数确认
# 纯函数性」承担（施工图 §1.3）。带旧数学耦合者（fit_response_operator、
# condition_trace_neutrality、optimize_conditioned_eb 等）一律不进新线。
#
# 语义地位（D-024）：本文件全部对象是固定模型定义下的确定性历史泛函 /
# 数值 gauge，不是 posterior 随机对象。
#
# 简化声明（如实标注，唯一简化点）：`relative_gauge` 去除旧线的
# ReentrantLock 与全局 Dict 缓存（旧线的并发性能设施；新线 reference
# 单线程、无 workspace、无缓存——D-082/D-083）。Helmert 构造逐字节一致，
# 数值输出与旧线完全相同。

export TAUS, BANDS, BANDCOL, WARMUP,
       center_of_mass, ruler, relative_gauge, principal_sqrt_root,
       compute_s_perp, path_basis_1d, fast_s_m

# 文件自含依赖声明（与 kelly.jl 的模式一致）：principal_sqrt_root 使用
# svd / Symmetric（LinearAlgebra 导出名）。本文件以「裸文件 + include」形态
# 被两种加载路径消费（顶层 module KTraderGate0 与测试的临时 module），
# include 时此语句在加载者作用域执行——两条路径同时获得所需名字。
using LinearAlgebra

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 来源：src/geometry.jl:11-14（旧线 KTrader V1.0 geometry 层常量）。
# 复制裁决 H7；语义不变：TAUS 是 ruler 拟合尺度、BANDS 是路径 basis 时间
# 频带、WARMUP 是最大 band 的两倍（路径 basis 需要的前缀长度）。
const TAUS  = 2 .^ (0:8)      # ruler fit horizons (1, 2, 4, 8, 16, 32, 64, 128, 256)
const BANDS = 2 .^ (1:7)      # multi-scale bands (2, 4, 8, 16, 32, 64, 128)
const BANDCOL = [findfirst(==(τ), TAUS) for τ in BANDS]  # path_basis_1d 的 s 长度分派依赖
const WARMUP = 2 * maximum(BANDS)

# ---------------------------------------------------------------------------
# 域级等权 center-of-mass 向量
# ---------------------------------------------------------------------------

"""
    center_of_mass(N::Int) -> Vector{Float64}

域级等权 COM 单位向量 `e₀ = (1,…,1)ᵀ/√N`。

- 来源：src/geometry.jl:20（旧线函数 `center_of_mass`）。
- 复制裁决 H7；语义不变。
- 域级语义（裁决 E1）：`e₀` 以**域大小** N 等权归一，与具体行的观察集
  无关。行级 field 的 m_s 归一（`1/√|O_s|`）是另一个概念（见
  src/gate0/modes.jl 的 `mode_field`）——域基与行级 field 定义并存不矛盾。
"""
center_of_mass(N::Int) = fill(1.0 / sqrt(N), N)

# ---------------------------------------------------------------------------
# Helmert 固定数值 gauge（D-023）
# ---------------------------------------------------------------------------

"""
    relative_gauge(N::Int) -> Matrix{Float64}

Helmert 固定数值 gauge：`Q ∈ R^{N×(N-1)}`，满足 `QᵀQ = I`、`Qᵀ𝐭 = 0`。

- 来源：src/numerics.jl:368-380（旧线函数 `relative_gauge`）。
- 复制裁决 H7；构造逐字节一致（第 k 列：前 k 个坐标 1/√(k(k+1))、第
  k+1 个坐标 −k/√(k(k+1))）。
- 语义不变（D-023）：纯数值 gauge，不是板块、不是 factor、不是市场
  模式；自然结构来自算子谱。
- 简化点（唯一）：去除旧线的 `RELATIVE_GAUGE_LOCK`/`RELATIVE_GAUGES`
  全局缓存（并发性能设施）；新线 reference 单线程无缓存（D-082/D-083），
  数值输出与旧线完全相同。
"""
function relative_gauge(N::Int)
    Q = zeros(N, max(N - 1, 0))
    for k in 1:N-1
        scale = sqrt(k * (k + 1))
        Q[1:k, k] .= 1 / scale
        Q[k+1, k] = -k / scale
    end
    Q
end

"""
    domain_mode_basis(n::Int) -> Matrix{Float64}

域级正交 mode 基 `E = [e₀, Q] ∈ R^{n×n}`：`e₀ = center_of_mass(n)`（域级
等权），`Q = relative_gauge(n)`（Helmert）。满足 `EᵀE = I`。

- 裁决 E1（域级基）与裁决 A（E_active/E_{R_t} 域实例化）的构造落点。
- `E_active = domain_mode_basis(N_a)`（active 域，N_a = active 资产数）；
  `E_{R_t} = domain_mode_basis(N_R)`（risk 域，N_R = |R_t|，供 innovation
  层后续消费——见 `risk_domain_mode_basis`）。
- 语义参考：docs/GATE0_VECTOR_INNOVATION.md §1.1（mode 坐标与正交变换）。
"""
function domain_mode_basis(n::Int)
    n >= 1 || throw(ArgumentError("domain_mode_basis: domain size must be ≥ 1"))
    hcat(center_of_mass(n), relative_gauge(n))
end

# ---------------------------------------------------------------------------
# 唯一对称 PSD 主平方根
# ---------------------------------------------------------------------------

"""
    principal_sqrt_root(L::AbstractMatrix{Float64}) -> Matrix{Float64}

协方差因子 `L·Lᵀ` 的唯一对称 PSD 主平方根：`R = U·Diag(s)·Uᵀ`
（`SVD(L) = U·Diag(s)·Vᵀ`）。

- 来源：src/numerics.jl:452-460（旧线函数 `principal_sqrt_root`）。
- 复制裁决 H7；实现逐字节一致（SVD 版本）。
- 语义不变：与底层分解器的基选择/符号自由度解耦（谱定理唯一性）；
  不平方条件数、零奇异值保持零模、保留全部 s（不截秩、无阈值抹小
  正值）。数值契约（Manager 2026-10-07 裁决）随复制一并有效。
"""
function principal_sqrt_root(L::AbstractMatrix{Float64})
    F = svd(L)
    Matrix(Symmetric((F.U .* F.S') * F.U'))
end

# ---------------------------------------------------------------------------
# Per-asset fractal ruler（D-024：确定性历史泛函）
# ---------------------------------------------------------------------------

"""
    ruler(x::AbstractMatrix{Float64}, f::AbstractVector{<:Integer},
          w::Union{Nothing,AbstractVector{Float64}} = nothing)
        -> Matrix{Float64}    # N × |TAUS|

Per-asset fractal ruler：对 log 价格 `x`（T×N）与首观测行 `f`，在 TAUS
尺度上做 power-law 拟合的加权 RMS τ-increment 平滑尺度。返回
`s[j, a] = s_j(TAUS[a])`；`s1 = s[:, 1]` 即一日 ruler（价格尺度正规化，
不是 alpha feature）。

- 来源：src/geometry.jl:26-76（旧线函数 `ruler`）。
- 复制裁决 H7；函数体逐字节一致（含 `4τ ≤ T − f_j + 1` 门槛、
  `n_use ≥ 2` 否则全 1 的退化分支、`max(·, 1e-12)` 数值下限）。
- 语义不变（D-024：确定性历史泛函，非 posterior 对象）。
"""
function ruler(x::AbstractMatrix{Float64}, f::AbstractVector{<:Integer},
               w::Union{Nothing,AbstractVector{Float64}} = nothing)
    T, N = size(x)
    Ntaus = length(TAUS)
    out = Matrix{Float64}(undef, N, Ntaus)
    lτ = log.(TAUS)
    L_buf = zeros(Float64, Ntaus)
    tau_buf = similar(L_buf)
    for j in 1:N
        n_use = 0; sum_l = 0.0; sum_log_tau = 0.0; fj = f[j]
        for a in 1:Ntaus
            τ = TAUS[a]
            if 4τ <= T - fj + 1
                acc = 0.0; ws = 0.0
                @inbounds @simd for t in (fj + τ):T
                    x1 = x[t, j]; x0 = x[t - τ, j]
                    if isfinite(x1) && isfinite(x0)
                        d = x1 - x0
                        wt = w === nothing ? 1.0 : w[t]
                        acc += wt * d * d
                        ws += wt
                    end
                end
                ws > 0 || continue
                n_use += 1
                val = log(sqrt(max(acc / max(ws, 1e-12), 1e-12)))
                L_buf[n_use] = val
                tau_buf[n_use] = lτ[a]
                sum_l += val
                sum_log_tau += lτ[a]
            end
        end
        if n_use >= 2
            m_l = sum_l / n_use
            m_tau = sum_log_tau / n_use
            num = 0.0; den = 0.0
            for a in 1:n_use
                d_tau = tau_buf[a] - m_tau
                num += (L_buf[a] - m_l) * d_tau
                den += d_tau * d_tau
            end
            H = num / max(den, 1e-12)
            for a in 1:Ntaus
                out[j, a] = exp(m_l + H * (lτ[a] - m_tau))
            end
        else
            # 短历史回退（n_use < 2；登记供 SPEC 审查——见
            # docs/RULER_SHORT_HISTORY_FALLBACK.md）：用该资产自身的
            # 一阶（τ=1）增量 RMS 作为所有 τ 的平坦外推（H=0——单点
            # 无法识别 scaling 斜率，不发明趋势；不设 4τ 门槛：这是
            # 「无门槛内数据可用」时唯一的真实尺度信息）。
            # 可达性：active 资产在 prefix 内必有 ≥1 个相邻观测对 ⇒
            # ws ≥ 1 恒成立；ws == 0 保留 1.0 作最后兜底（不可达防御，
            # 见文档 §4）。旧行为（s ≡ 1.0，≈日波动 100% 的荒谬尺度）
            # 曾使极短历史资产预测律病态（testset (8) t=271 红；证据
            # archive/evidence/gate0_t4_recovery_20261010/）。
            acc = 0.0; ws = 0.0
            @inbounds @simd for t in (fj + 1):T
                x1 = x[t, j]; x0 = x[t - 1, j]
                if isfinite(x1) && isfinite(x0)
                    d = x1 - x0
                    wt = w === nothing ? 1.0 : w[t]
                    acc += wt * d * d
                    ws += wt
                end
            end
            out[j, :] .= ws > 0 ? sqrt(max(acc / max(ws, 1e-12), 1e-12)) : 1.0
        end
    end
    out
end

# ---------------------------------------------------------------------------
# Per-TAU macro ruler（macro 累积路径的尺度）
# ---------------------------------------------------------------------------

"""
    fast_s_m(X::AbstractVector{Float64}) -> Vector{Float64}    # |TAUS|

macro 累积路径 `X_m` 的 per-TAU ruler（对单序列跑 `ruler`）。

- 来源：src/response.jl:34（旧线函数 `fast_s_m`）。
- 复制裁决 H7；语义不变：返回 |TAUS| 维（9 维），`path_basis_1d` 经
  `BANDCOL` 分派按 band 取列（与旧线 `B_m = path_basis_1d(X_m_vec, s_m)`
  的调用语义一致，src/predict.jl:281）。
"""
fast_s_m(X::AbstractVector{Float64}) = vec(ruler(reshape(X, :, 1), [1]))

# ---------------------------------------------------------------------------
# Per-band gauge-invariant scalar relative ruler（SPEC §15）
# ---------------------------------------------------------------------------

"""
    compute_s_perp(X::AbstractMatrix{Float64}) -> Vector{Float64}    # |BANDS|

每 temporal band 一个 gauge-invariant 标量 relative ruler：对累积 relative
路径 `X_⊥`（T×N）的历史平方位移平均，乘 `N/(N−1)` 归一。

- 来源：src/response.jl:18-33（旧线函数 `compute_s_perp`）。
- 复制裁决 H7；函数体逐字节一致。
- 语义不变（SPEC §15）：per-band 标量（非逐坐标）——逐坐标归一会破坏
  `Q → QR` 旋转协变性，禁止。
"""
function compute_s_perp(X::AbstractMatrix{Float64})
    T, N = size(X)
    out = zeros(length(BANDS))
    for (b, tau) in enumerate(BANDS)
        count = 0; acc = 0.0
        for j in 1:N, t in (tau+1):T
            a = X[t,j]; z = X[t-tau,j]
            if isfinite(a) && isfinite(z)
                acc += (a-z)^2
                count += 1
            end
        end
        out[b] = count == 0 ? 1e-3 : sqrt(max(acc / count * N / max(N-1,1), 1e-6))
    end
    out
end

# ---------------------------------------------------------------------------
# Paired Q/P path basis（SPEC §16）
# ---------------------------------------------------------------------------

"""
    path_basis_1d(X::AbstractVector{Float64}, s::AbstractVector{Float64})
        -> Matrix{Float64}    # T × 2|BANDS|

单序列 paired Q/P path basis（level 上构造，SPEC §14/§16 语义）：

- `c₀(t,τ) = mean(X[t-τ+1:t])`（近期窗口均值）、
  `c₁(t,τ) = mean(X[t-2τ+1:t-τ])`（前一期窗口均值）；
- `Q(t,τ) = −(X[t] − c₀)/s(τ)`（位置偏离）、
  `P(t,τ) = (c₀ − c₁)/s(τ)`（方向/确认）；
- 列序：`(band 1 Q, band 1 P, band 2 Q, band 2 P, …)`，即列 `2b−1` 为
  Q 通道、列 `2b` 为 P 通道；
- `t < 2τ+1` 的行保持 0（前缀不足）；
- `s` 长度为 |BANDS| 时按 band 索引、否则经 `BANDCOL` 映射（与旧线
  `s_m` 9 维输入的分派一致）。

- 来源：src/response.jl:2-16（旧线函数 `path_basis_1d`）。
- 复制裁决 H7；函数体逐字节一致。
- 语义不变：时间滤波是线性算子——与 `Qᵀ` 投影可交换（B_⊥ 构造语义，
  GATE0_RESPONSE_POSTERIOR.md §1.4）。
"""
function path_basis_1d(X::AbstractVector{Float64}, s::AbstractVector{Float64})
    T = length(X)
    B = zeros(T, 2length(BANDS))
    sums = cumsum(vcat(0.0, X))
    for (b, tau) in enumerate(BANDS)
        scale = max(s[length(s) == length(BANDS) ? b : BANDCOL[b]], 1e-6)
        for t in (2tau+1):T
            c0 = (sums[t+1] - sums[t+1-tau]) / tau
            c1 = (sums[t+1-tau] - sums[t+1-2tau]) / tau
            B[t,2b-1] = -(X[t] - c0) / scale
            B[t,2b] = (c0 - c1) / scale
        end
    end
    B
end
