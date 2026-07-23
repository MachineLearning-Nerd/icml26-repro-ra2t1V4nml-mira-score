# Claim 4 method

The primary verifier evaluates the finite-N identity with `fractions.Fraction`
for several analytic monotone maps, including the null map, disjoint-support
limit, and strongly nonlinear alternatives. It evaluates the exact N values
used by the paper's two lensing applications (`N=20,000` and `N=64`).

An independent checker does not reuse the analytic formulas. It exhaustively
enumerates 6,435 nondecreasing step maps on an eight-point rational grid and
checks the discrete Chebyshev inequality exactly.

The negative control deliberately replaces the nondecreasing radial-mass map
with `T(u)=1-u`. It must produce a score below 1/2. This is expected rejection,
not a counterexample, because a decreasing map cannot arise from two CDFs on
the same nested balls.

All computations are deterministic and use no random seed. The verifier exits
nonzero if any contracted inequality, paper-scale cell, independent result, or
negative-control behavior differs from its expectation.
