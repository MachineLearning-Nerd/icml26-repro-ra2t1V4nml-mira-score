# Cumulative six-claim release summary

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_mira_release_summary", "created_at": "2026-07-23T09:15:00+00:00", "title": "Six-claim terminal evidence", "pinned": true, "pinned_at": "2026-07-23T09:15:00+00:00"}
-->
The new cumulative campaign records five **VERIFIED** claims and one
**BLOCKED** claim. Claims 1–3 retain their previously judged full-credit
evidence. Claim 4 replaces the Gaussian toy with an exact analytic certificate.
Claim 6 evaluates the authors' exact released `L=16`, `N=64`, `64×64×3`
posterior tensors and reproduces the complete model order. Claim 5 remains
blocked because the 13-D truths, four posterior sets, sampler/forward-model
code, fixed constants, versions, and seeds were not released.

| Claim | Verdict | New evidence |
| --- | --- | --- |
| 1 | VERIFIED | Preserved exact finite-null and parity evidence |
| 2 | VERIFIED | Preserved Laplace equation cells |
| 3 | VERIFIED | Preserved convergence to `2/3` and `1/18` |
| 4 | VERIFIED | Exact identity; 6,435 monotone-map checker |
| 5 | BLOCKED | Complete pinned release audit; no proxy substituted |
| 6 | VERIFIED | Exact author tensors; full-scale parity and controls |

The prior live-judge score is **7/12**. This published candidate makes no score
forecast. A score can change only after a new live judge verdict.

[Read the illustrated cumulative report](../../reports/mira-reproduction/report.md).

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_mira_release_scope", "created_at": "2026-07-23T09:15:00+00:00", "title": "Compute and scope"}
-->
Terminal evidence was regenerated on local Apple-arm CPU. Three additional
Claim 5 approaches and one longer affine refinement used 94m47s of Hugging Face
`cpu-upgrade`; no GPU was used and the orchestration logs did not expose billed
cost. The fixed command on every experiment node is:

```bash
uv run --frozen python repro/src/run_campaign.py
```

The exact judged revision remains preserved. All prior page paths remain
present; new pages and text evidence are additive.
