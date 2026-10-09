# =============================================================================
# KTraderGate0 统一 mode coordinate producer — Gate-0 纠偏新线（施工图 Step 3）
# =============================================================================
#
# 文件地位（裁决 H1/H2）：未来顶层 module `KTraderGate0` 的组成文件；当前
# 按裸文件交付（同 src/gate0/geometry.jl），由测试经临时 module include。
# 本文件依赖 geometry.jl 先行 include（常量 BANDS 与 path_basis_1d）。
#
# 职责边界（施工图 Step 3 / §2.3）：**只做** `[m; Qᵀe]`（统一 mode 输出，
# N 维）与 `[B_m; B_⊥]`（统一 mode 输入，14N 维，DC 列位预留）的 producer；
# 暂不改 posterior（§40 Step 3 原文）。
#
# 规范来源：
# - 维度与坐标：docs/GATE0_RESPONSE_POSTERIOR.md §1（记号与维度总表：
#   y = [m; q] ∈ R^N、x̃ = [1; B_m; B_⊥] ∈ R^{14N+1}、B_m ∈ R^{14}、
#   B_⊥ ∈ R^{14(N−1)} gauge 坐标）。
# - 行级归一（裁决 E1）：m_s 按该行观察集 O_s 归一（1/√|O_s|，SPEC §12
#   现状语义）；E_active 用 active 集等权、E_{R_t} 用 R_t 等权——域基
#   （domain basis，随域等权）与行级 field 定义（随行观察集归一）是两
#   个概念，并存不矛盾。
# - 零嵌入（D-022）：未观测坐标在 field 代数中嵌入为 0 只表示「当前
#   active 坐标空间中 relative field 的嵌入值为零」，不表示 r = 0；
#   observation mask 永久单独保留。
# - gauge 坐标（D-030）：relative 部分直接在 Qᵀe 坐标工作——构造性删除
#   input common 不可达方向与 output common/relative 冗余表示。
# - 累积语义（SPEC §14）：X_m、X_⊥ 在 level 上累积；Q/P basis 在 level
#   上构造（时间滤波与 Qᵀ 投影可交换——GATE0_RESPONSE_POSTERIOR §1.4）。
#
# DC 列位预留（Step 5）：统一输入 x̃ = [1; B_m; B_⊥] 的列 1 为常数列
# （D-031 恢复的 intercept 通道）。**本步不实现 DC 列本身**（按施工图
# §40 顺序：Step 4 固定 ridge → Step 5 加 DC）；本文件钉死的列布局为：
# 列 1 = DC（预留）、列 2:15 = B_m（macro path）、列 16:14N+1 = B_⊥
# （relative path gauge 坐标）。

export domain_mode_basis, risk_domain_mode_basis,
       mode_field, asset_field_reconstruct,
       cumulative_path_coordinates, mode_basis_design

# ---------------------------------------------------------------------------
# E_{R_t}：risk 域 mode 基（供 innovation 层后续消费）
# ---------------------------------------------------------------------------

"""
    risk_domain_mode_basis(R_t::AbstractVector{Int}) -> Matrix{Float64}

risk 域 mode 基 `E_{R_t} = domain_mode_basis(|R_t|)`（裁决 A/B2'：R_t =
free ∪ locked 的域实例化）。`E_{R_t}` 作用在 R_t 子向量（N_R 维）上；
子向量提取（asset 空间 OOF 残差行 → R_t 子向量）由调用方（innovation
层）执行——mode 坐标无资产子集索引，一律走 asset 空间提取路径（裁决
A3）。本步只提供构造函数。
"""
risk_domain_mode_basis(R_t::AbstractVector{Int}) = domain_mode_basis(length(R_t))

# ---------------------------------------------------------------------------
# 行级 mode field：[m_s; q_s]
# ---------------------------------------------------------------------------

"""
    mode_field(returns_row, observed_row, s1, E) -> (; y, e, observed)

单行统一 mode 输出 `y = [m_s; q_s] ∈ R^N`（GATE0_RESPONSE_POSTERIOR §1）。

输入：
- `returns_row ∈ R^N`：当日 log returns（缺失坐标为 NaN，仅作占位）；
- `observed_row ∈ Bool^N`：当日观察 mask O_s（D-022：永久保留）；
- `s1 ∈ R^N`：per-asset 一日 ruler（`ruler(x,f)[:,1]` 语义）；
- `E ∈ R^{N×N}`：域级 mode 基（`domain_mode_basis(N)`，如 E_active）。

计算：
1. `u_i = returns_row[i] / s1[i]`（observed 处；SPEC §12 ruler 标准化）；
2. `m_s = (1/√|O_s|) · Σ_{i∈O_s} u_i`（**行级归一**，裁决 E1——注意与
   E 的域级等权 e₀ 是两个概念：|O_s| ≠ N 时 `m_s ≠ e₀ᵀu`）；
3. `e_s`：observed 坐标 `u_i − (1/|O_s|)Σ_{j∈O_s} u_j`、缺失坐标 **0**
   （D-022 零嵌入——field 代数，不是收益 imputation）；行和构造性为零
   （`e_s ∈ 𝟏⊥`，故 `Qᵀe_s` 无损、`Q·q_s = e_s` 精确成立）；
4. `q_s = Qᵀ e_s`（Q = E 的 relative 块，D-030 gauge 坐标）。

返回 `(; y, e, observed)`：
- `y`：`[m_s; q_s] ∈ R^N`（统一 mode 输出向量）；
- `e`：零嵌入 relative field（N 维）；
- `observed`：输入 mask 的 owned copy。

边界与 fail-loudly：`|O_s| = 0`（空观察行）时 `m_s = 0`、`e = 0`（零
嵌入语义；该行是否进入统计由消费层行集判定——VI §1.4，非本层职责）。
observed 坐标的 return 必须 finite；s1 必须 finite 且 > 0；维度不一致即
抛错（DimensionMismatch / DomainError，SPEC §56）。
"""
function mode_field(returns_row::AbstractVector{Float64},
                    observed_row::AbstractVector{Bool},
                    s1::AbstractVector{Float64},
                    E::AbstractMatrix{Float64})
    N = length(returns_row)
    (length(observed_row) == N && length(s1) == N) ||
        throw(DimensionMismatch("mode_field: returns_row/observed_row/s1 lengths must all equal N"))
    (size(E, 1) == N && size(E, 2) == N) ||
        throw(DimensionMismatch("mode_field: E must be the N×N domain mode basis"))
    for j in 1:N
        (isfinite(s1[j]) && s1[j] > 0) ||
            throw(DomainError(s1[j], "mode_field: s1 must be finite and positive (ruler scale)"))
        if observed_row[j]
            isfinite(returns_row[j]) ||
                throw(DomainError(returns_row[j],
                    "mode_field: observed coordinate must carry a finite return (D-022: mask is the observation fact)"))
        end
    end
    c = 0
    sum_u = 0.0
    u = Vector{Float64}(undef, N)
    for j in 1:N
        if observed_row[j]
            u[j] = returns_row[j] / s1[j]
            sum_u += u[j]
            c += 1
        end
    end
    m = c == 0 ? 0.0 : sum_u / sqrt(c)          # 行级归一 1/√|O_s|（裁决 E1）
    e = zeros(Float64, N)                       # 零嵌入（D-022）
    if c > 0
        shift = sum_u / c
        for j in 1:N
            if observed_row[j]
                e[j] = u[j] - shift
            end
        end
    end
    Q = view(E, :, 2:N)                         # 域基的 relative 块（Helmert）
    q = Q' * e
    y = Vector{Float64}(undef, N)
    y[1] = m
    if N >= 2
        y[2:N] .= q
    end
    (; y = y, e = e, observed = BitVector(observed_row))
end

# ---------------------------------------------------------------------------
# mode → asset 重构
# ---------------------------------------------------------------------------

"""
    asset_field_reconstruct(mode_vec, E) -> Vector{Float64}

逆变换 `u = E · mode_vec`（`[m; q] → m·e₀ + Q·q`，reconstruct 验证用）。

- E 为域级基（裁决 E1）。**全观测行**（|O_s| = N）上 `m_s = e₀ᵀu`，故
  `E·y = u` 精确无损（reconstruct 精度测试的断言对象）。
- **ragged 行**上 `m_s`（行级归一 1/√|O_s|）与 e₀（域级等权 1/√N）是
  两个概念：`E·y` 的 observed 坐标 = `u_i − (1 − √(|O_s|/N))·mean_{O_s}`，
  不等于 u_i——这是裁决 E1「域基与行级 field 定义并存」的直接后果，
  不是 bug；ragged 行的 field 语义由 `mode_field` 返回的 `e`/`observed`
  承担。
"""
function asset_field_reconstruct(mode_vec::AbstractVector{Float64},
                                  E::AbstractMatrix{Float64})
    n = size(E, 1)
    size(E, 2) == n ||
        throw(DimensionMismatch("asset_field_reconstruct: E must be square (domain mode basis)"))
    length(mode_vec) == n ||
        throw(DimensionMismatch("asset_field_reconstruct: mode_vec length must match E"))
    all(isfinite, mode_vec) ||
        throw(DomainError(mode_vec, "asset_field_reconstruct: mode_vec must be finite"))
    E * collect(mode_vec)
end

# ---------------------------------------------------------------------------
# 累积路径坐标（SPEC §14：level 上累积）
# ---------------------------------------------------------------------------

"""
    cumulative_path_coordinates(m_series, e_embedded) -> (X_m, X_rel)

level 累积路径坐标（SPEC §14 语义；独立实现，参考旧线
src/predict.jl:256-269 的 X_m/X_rel 构造语义）：

- `X_m[t] = Σ_{s≤t} m_s`（macro 累积 level，Vector{T}）；
- `X_rel[t, j] = Σ_{s≤t} e_s[j]`（relative 累积 level，T×N）。

Q/P basis 必须在 level 上构造（不能 numerator 用 returns、denominator 用
integrated levels——历史已修正的关键 bug）。输入必须 finite（NaN/Inf 即
抛错）；`e_embedded` 是 mode_field 产生的零嵌入 field（缺失坐标已嵌 0，
故 X_rel 无 NaN）。
"""
function cumulative_path_coordinates(m_series::AbstractVector{Float64},
                                      e_embedded::AbstractMatrix{Float64})
    T = length(m_series)
    size(e_embedded, 1) == T ||
        throw(DimensionMismatch("cumulative_path_coordinates: m_series rows must match e_embedded rows"))
    N = size(e_embedded, 2)
    X_m = Vector{Float64}(undef, T)
    X_rel = Matrix{Float64}(undef, T, N)
    running_m = 0.0
    for t in 1:T
        isfinite(m_series[t]) ||
            throw(DomainError(m_series[t], "cumulative_path_coordinates: m_series must be finite"))
        running_m += m_series[t]
        X_m[t] = running_m
    end
    for j in 1:N
        running_rel = 0.0
        for t in 1:T
            isfinite(e_embedded[t, j]) ||
                throw(DomainError(e_embedded[t, j], "cumulative_path_coordinates: e_embedded must be finite"))
            running_rel += e_embedded[t, j]
            X_rel[t, j] = running_rel
        end
    end
    (X_m, X_rel)
end

# ---------------------------------------------------------------------------
# [B_m; B_⊥] 设计（统一 mode 输入 producer）
# ---------------------------------------------------------------------------

"""
    mode_basis_design(X_m, X_rel, Q, s_m, s_perp) -> (; B_m, B_perp, X_q)

统一 mode 输入 `[B_m; B_⊥]` 的 producer（GATE0_RESPONSE_POSTERIOR §1
维度表的落点；**暂不含 DC 列**——Step 5 在头部加常数列 1，本步钉死
预留列位）。

构造：
- `B_m = path_basis_1d(X_m, s_m)`：macro 累积路径的 paired Q/P features，
  **T × 14**（7 bands × Q/P 两通道，列序 (b,Q),(b,P)——与维度表 B_m ∈
  R^{14} 逐项一致）；归一 `s_m`（per-TAU macro ruler，`fast_s_m` 语义，
  9 维输入经 BANDCOL 分派）。
- `X_q = X_rel · Q`：**先累积再投影**（T × (N−1)；线性交换性使
  `X_rel·Q = Σ_{s≤t}Qᵀe_s`——先投影再累积的恒等结果）。
- `B_perp`：对 X_q 每 gauge 坐标 k 做 `path_basis_1d(X_q[:,k], s_perp)`，
  **T × 14(N−1)**；归一 `s_perp`（per-band gauge-invariant 标量 ruler，
  SPEC §15——逐坐标归一破坏 Q→QR 协变性，禁止）。

**B_⊥ 列布局**（与 GATE0_RESPONSE_POSTERIOR §1 维度表逐项一致；与旧线
asset 坐标布局 src/response.jl:55-75 同构、坐标换成 gauge）：

```
列偏移 offset = (b−1)·2(N−1)
Q 通道（位置偏离）：列 offset + k          （k = 1..N−1，逐 gauge 坐标）
P 通道（方向确认）：列 offset + (N−1) + k  （k = 1..N−1）
```

即每 band 占 2(N−1) 列，Q 通道块在前、P 通道块在后。统一输入拼接（由
调用方/未来 ModeProblem 组装）：`x̃ = [1; B_m; B_⊥] ∈ R^{14N+1}`——列 1
DC（Step 5）、列 2:15 B_m、列 16:14N+1 B_⊥。

时间滤波与 Qᵀ 投影可交换（线性算子）：`B_⊥` 的每 (band, channel) 块 =
asset 坐标同块 × Q（`F_τ ∘ Qᵀ = Qᵀ ∘ F_τ`）——permutation/gauge 不变量
的可测契约（GATE0_RESPONSE_POSTERIOR §1.4）。
"""
function mode_basis_design(X_m::AbstractVector{Float64},
                           X_rel::AbstractMatrix{Float64},
                           Q::AbstractMatrix{Float64},
                           s_m::AbstractVector{Float64},
                           s_perp::AbstractVector{Float64})
    T = length(X_m)
    size(X_rel, 1) == T ||
        throw(DimensionMismatch("mode_basis_design: X_m rows must match X_rel rows"))
    N = size(X_rel, 2)
    (size(Q, 1) == N && size(Q, 2) == N - 1) ||
        throw(DimensionMismatch("mode_basis_design: Q must be N×(N−1) (Helmert gauge of the domain)"))
    length(s_perp) == length(BANDS) ||
        throw(DimensionMismatch("mode_basis_design: s_perp must have |BANDS| entries (per-band ruler)"))
    (length(s_m) == length(BANDS) || length(s_m) == length(TAUS)) ||
        throw(DimensionMismatch("mode_basis_design: s_m must have |BANDS| or |TAUS| entries"))
    n_bands = length(BANDS)
    B_m = path_basis_1d(X_m, s_m)
    X_q = X_rel * Q                            # 先累积再投影（交换性的落点）
    Nm1 = N - 1
    B_perp = Matrix{Float64}(undef, T, 2 * n_bands * Nm1)
    col = Vector{Float64}(undef, T)
    for k in 1:Nm1
        copyto!(col, view(X_q, :, k))
        Bk = path_basis_1d(col, s_perp)        # T × 14（列序 (b,Q),(b,P)）
        for b in 1:n_bands, ch in 1:2
            src_col = 2 * (b - 1) + ch
            dst_col = (b - 1) * 2 * Nm1 + (ch - 1) * Nm1 + k
            for t in 1:T
                B_perp[t, dst_col] = Bk[t, src_col]
            end
        end
    end
    (; B_m = B_m, B_perp = B_perp, X_q = X_q)
end
