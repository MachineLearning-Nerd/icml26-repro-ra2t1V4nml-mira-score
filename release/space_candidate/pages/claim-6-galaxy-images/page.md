# Claim 6 — exact released galaxy tensors

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_mira_claim6", "created_at": "2026-07-23T09:15:00+00:00", "title": "Claim 6 verdict", "pinned": true, "pinned_at": "2026-07-23T09:15:00+00:00"}
-->
## VERIFIED

The exact author tensors were fetched from
`SammyS15/MIRA_Paper_Plots@3bc229222cbcf72bd470267175d9a6dff6689ce0`.
SHA-256, shape, and dtype checks establish the paper scale: `L=16`, `N=64`,
and `64×64×3 = 12,288` dimensions.

| Model | Paper | Observed |
| --- | ---: | ---: |
| Spiral prior, σ=2 (true) | 0.6442 | 0.6831 |
| Spiral prior, σ=0.5 | 0.5783 | 0.5097 |
| Elliptical prior, σ=2 | 0.5298 | 0.5032 |
| Elliptical prior, σ=0.5 | 0.5056 | 0.4953 |

The complete model order agrees with the claim. Every paired 95% two-axis
bootstrap interval for true minus misspecified excludes zero. A full-scale
released-code check has absolute delta `0.0`; rolling the truth pairing lowers
the true score from `0.6831` to `0.5183`.

The numerical values are not claimed identical: the maximum absolute difference
from the hardcoded plot scores is `0.0686`, reported as **DIVERGENT**. The paper
does not publish its MIRA seed or center configuration.

Evidence:
[contract](../../evidence/claim_6/claim_contract.json),
[raw summary](../../evidence/claim_6/raw_summary.json),
[raw 6,400-cell CSV](../../evidence/claim_6/raw_per_region_truth.csv),
[implementation parity](../../evidence/claim_6/implementation_parity.json),
[independent checker](../../evidence/claim_6/independent_checker_output.json),
[negative control](../../evidence/claim_6/negative_control_output.json), and
[evaluation](../../evidence/claim_6/EVAL.md).
