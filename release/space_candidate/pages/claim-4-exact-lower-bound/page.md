# Claim 4 — exact lower-bound certificate

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_mira_claim4_exact", "created_at": "2026-07-23T09:15:00+00:00", "title": "Proposition 3.5 exact certificate", "pinned": true, "pinned_at": "2026-07-23T09:15:00+00:00"}
-->
## VERIFIED

This replaces—not deletes—the preserved Gaussian toy page. Under the
continuous radial-CDF assumptions used in Proposition 3.5, the exact identity is

`μ = 1/2 + N/(N+2) × [2 E(U T(U)) - E(T(U))]`.

The radial-mass map `T` is nondecreasing, so Chebyshev's integral inequality
makes the bracket nonnegative. Exact rational cells pass at `N=1`, `N=64`, and
`N=20,000`. The released normalized finite floors are `0.75`, `0.507692`, and
`0.500025`.

An independent checker exhausts 6,435 nondecreasing rational-grid maps and
finds minimum gap zero. The decreasing control `T(u)=1-u` violates the
assumptions and falls below `1/2`, demonstrating verifier sensitivity.

Evidence:
[contract](../../evidence/claim_4/claim_contract.json),
[exact results](../../evidence/claim_4/raw_exact_results.json),
[independent checker](../../evidence/claim_4/independent_checker_output.json),
[negative control](../../evidence/claim_4/negative_control_output.json), and
[evaluation](../../evidence/claim_4/EVAL.md).
