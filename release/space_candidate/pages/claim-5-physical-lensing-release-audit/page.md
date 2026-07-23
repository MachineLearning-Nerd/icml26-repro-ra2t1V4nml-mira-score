# Claim 5 — 13-D physical-lensing release audit

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_mira_claim5", "created_at": "2026-07-23T09:15:00+00:00", "title": "Claim 5 verdict", "pinned": true, "pinned_at": "2026-07-23T09:15:00+00:00"}
-->
## BLOCKED

The paper reports `L=100`, `N=20,000`, 13 dimensions, and four candidate
models: EPL+3 Sérsic, SIE+3, EPL+1, and SIE+1. Its pinned plotting notebook
hardcodes MIRA scores `0.6320 > 0.5788 > 0.5394 > 0.5223`.

Complete recursive Git-tree audits at the pinned author commits found:

- no `L=100`, 13-D truth parameter set;
- no four `100×20,000×13` posterior sample sets;
- no MALA implementation, step size, initialization, or tuning;
- no executable caustics EPL/SIE–Sérsic model construction;
- no values for parameters the paper calls “held constant”; and
- no dependency versions or random seeds.

An aggregate TARP `.npz` exists, but it cannot regenerate MIRA. Therefore this
campaign neither verifies nor falsifies the claim. The prior GMM proxy is not
counted.

A clean-room physical reconstruction then tested three distinct posterior
approaches at the exact `N=20,000` sample scale: preconditioned HMC,
affine-invariant ensemble sampling, and adaptive multiscale importance
sampling. Every approach failed a predeclared R̂/ESS/weight gate for at least
one of the four candidates. Favorable rankings from those rejected profiles
are not counted. [See the illustrated diagnostic audit](../../reports/claim5-three-approach/report.md).

Evidence:
[contract](../../evidence/claim_5/claim_contract.json),
[raw release audit](../../evidence/claim_5/raw_release_audit.json),
[independent checker](../../evidence/claim_5/independent_checker_output.json),
[negative control](../../evidence/claim_5/negative_control_output.json), and
[evaluation](../../evidence/claim_5/EVAL.md).
