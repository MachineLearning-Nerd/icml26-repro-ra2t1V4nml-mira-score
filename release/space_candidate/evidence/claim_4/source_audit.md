# Claim 4 source audit

- Source: arXiv `2605.02014v1`, retrieved from
  `https://export.arxiv.org/e-print/2605.02014v1` on
  `2026-07-23T08:29:30Z` with an explicit User-Agent.
- Source archive SHA-256:
  `93f93308136b7078018639dd75ab58d977994817e2cda7946c96477a3240a184`.
- Main statement: Proposition 3.5, `prop:lower_bound`, TeX lines 343–354.
- Proof: Appendix `app:conv_lower`, TeX lines 731–827.
- Region definition: TeX lines 245–268.

The quantified phrase is “For any candidate model M.” The proof is not
assumption-free: it uses a continuous radial candidate CDF so that
`lambda_n ~ Uniform[0,1]`, and it uses nested metric balls to express the true
mass as a nondecreasing map `lambda_k = T(lambda_n)`. The contract records these
assumptions explicitly. It tests the paper-defined unnormalized score. The
released code's final calibration factor only raises the finite-N floor.

The proof reduces the claim to Chebyshev's integral inequality:

`integral u T(u) du >= integral u du * integral T(u) du`

for two nondecreasing functions. Therefore
`2 E[U T(U)] - E[T(U)] >= 0`, which substituted into the exact finite-N score
identity gives the bound.
