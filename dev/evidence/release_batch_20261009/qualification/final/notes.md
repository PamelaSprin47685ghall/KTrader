Qualification after preserving the certificate's original M summation order.
The preceding cache-reuse test exposed roundoff differences between fresh
GEMM M and warmed core-dot M. This revision does NOT relax those exact-field
assertions; it explicitly uses the original direct M route for certificates,
while reusing alpha-only cores. Prior failed logs/snapshots stay untouched.
