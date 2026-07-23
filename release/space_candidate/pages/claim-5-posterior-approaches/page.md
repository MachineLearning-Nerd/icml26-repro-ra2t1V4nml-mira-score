# Claim 5 — three posterior approaches

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_mira_claim5_posterior_approaches", "created_at": "2026-07-23T13:18:32+00:00", "title": "Three fail-closed posterior profiles", "pinned": true, "pinned_at": "2026-07-23T13:18:32+00:00"}
-->
## BLOCKED after three distinct attempts

The clean-room 13-D physical-lensing reconstruction tested three materially
different posterior estimators at exactly `N=20,000`:

| Method | Reliability result |
| --- | --- |
| Preconditioned HMC | True posterior passed, but three misspecified posteriors had split-R̂ `75.59–193.97` |
| Affine-invariant ensemble | A longer run still had split-R̂ `1.587` and `2.941` for the two 3-Sérsic targets |
| Adaptive multiscale importance | Weight collapse; ESS only `1.09–10.26` of 20,000 |

The gate was split-R̂ ≤ 1.20 and ESS ≥ 400 for chains, or importance ESS ≥
1,000 with maximum normalized weight ≤ 0.01. Every method failed for at least
one candidate. Several rejected profiles ranked the true EPL+3 model first, but
nonconverged samples are not promoted as Claim 5 evidence.

[Read the visual audit and exact experiment lineage](../../reports/claim5-three-approach/report.md).
