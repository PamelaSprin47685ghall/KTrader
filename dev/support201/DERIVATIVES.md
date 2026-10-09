# Exact reduced derivatives and acceptance scope

Let `a=log(alpha)`, original relative output dimension `d`, retained support
dimension `r`, null dimension `k=d-r`, constraints `m=14`, and original
floor `delta`. The candidate does not replace `d` with `r` in evidence.

For data eigenvalues `lambda`, set

```
v   = 1/(lambda+alpha)
v_a = -alpha*v^2
v_aa = -alpha*v^2 + 2*alpha^2*v^3
R = YY - B' diag(v) B
h = hcoef' v
M = A_alpha(S) + k*delta/alpha * I
t = tr(S) + k*delta
```

The loss is the negative ORIGINAL evidence:

```
f = d/2 sum(log1p(lambda/alpha))
  + n/2 (logdet(S)+k*log(delta)) + tr(S^-1 R)/2
  + logdet(M)/2 + h'M^-1 h/2 - m/2 log(t/alpha).
```

With `P=S^-1`, `V=M^-1`, `z=Vh`, `K=V-zz'`, the alpha derivatives include
the null term (`M_a` has `-k*delta/alpha*I`; `M_aa` has the positive term).

```
z_a = V(h_a-M_a z)
K_a = -V M_a V - z_a z' - z z_a'
f_a = -d/2 sum(lambda*v) + tr(P R_a)/2
    + tr(K M_a)/2 + z'h_a + m/2
f_aa = d/2 sum(lambda*alpha*v^2) + tr(P R_aa)/2
     + tr(K_a M_a)/2 + tr(K M_aa)/2 + z_a'h_a + z'h_aa
f_Sa = -P R_a P/2 + A_alpha^*(K_a)/2 + (dA_alpha/da)^*(K)/2.
```

The implementation contracts derivative cores directly using14 GEMMs rather
than forming105 first- and105 second-derivative matrices. Second alpha
moments use the scalar spectral tensors; all terms above remain present.
Line search uses exact scalar evidence, not a Taylor approximation or
stale previous-day value.

The covariance Hessian is `H0+A*G A`, with106 moment coordinates. The joint
alpha direction uses its107-dimensional Schur system. Local descent is
checked; feasibility and the original ambient certificate are not replaced
by positive base curvature. Unsupported floors or failed steps reject.

## Sufficient fixed-alpha uniqueness enclosure

`A(S)` is positive, and for `X=S^-1/2 H S^-1/2` the normalized derivative
of `M` has operator norm at most `||X||op`. Thus the negative logdet-M
curvature is bounded below by `-(m/2)||X||F^2`; the h-quadratic and trace
parts have nonnegative curvature. The loss Hessian is positive on

```
0 < S < beta * R/n,  beta=2n/(n+m), n>m.
```

Outside that domain, the Gaussian covariance loss exceeds its unconstrained
minimum by at least `n/2*(log(beta)+1/beta-1)`. Since the ridge covariance
has lower bound `1/(lambda_max+alpha)`, the non-Gaussian loss has global
lower bound `-(m/2)*log1p(lambda_max/alpha)` in the retained space.
The implemented test compares an actual point's sublevel against those
bounds with a separate floating-point margin. It does not use the seed to
declare uniqueness, does not assert joint-alpha convexity, and does not
settle general off-face covariance branches.

## Why the present acceptance is still red

The joint corrector refines to a different finite stopping point. Both the
original and candidate can pass their original1e-6 KKT certificate yet
differ more than the independently specified posterior-output gate. Fixed
alpha profiling alone also does not reproduce the original algorithm's
finite inner stopping trajectory. Changing a test threshold, shifting a
return toward a saved golden output, or rounding alpha is not a fix.

There is also an ill-conditioned toy objective mismatch: the ambient and
compressed covariance trace evaluation differ around2.13e-7, dominating
the total2.17e-7 evidence gap. All analytic derivative tests pass, but that
does not turn the original objective-value gate green. Real fixed-parameter
posterior compression checks pass, separating the structural algebra from
the solver stopping-point issue. Both observations are retained.

Until these are resolved, this is a quarantined candidate, not2.0.1 Final.
