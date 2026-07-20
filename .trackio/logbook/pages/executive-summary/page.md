# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_24d5d7365a6e", "created_at": "2026-07-20T09:10:22+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-20T09:10:22+00:00"}
-->
All three scored claims pass the local publication gate: **6 / 6 points**.
The clean-room estimator reproduces the released score (mean delta `0.0`; s.d.
delta `2.24e-8`), an exact finite-null calculation reaches the paper's reference
values at `N=5,000`, and five retained source-scale GMM jobs rank the correct
zero-shift candidate first. The multi-seed nearest-shift margin is `0.0021722`
(`8.30` standard errors), with every seed preserving both strict ordering tests.
The result is appropriately scoped: the source omits GMM construction details,
which are made deterministic and explicit here, and a separate control shows
that arithmetic high-dimensional density-ratio averages are unstable.

## Scope & cost

| Item | Value |
| --- | --- |
| GPU / compute | Five `t4-medium` Jobs; full protocol: 100-D, 20 components, `L=N=5,000`, 100 regions |
| Wall time | Mean Job runtime: 228.7 s; local verification: 9 tests in 6.2 s |
| Feasibility | Full protocol reproduced with streamed batches; all raw JSON is retained and hash-checked |


---
<!-- trackio-cell
{"type": "figure", "id": "cell_47eb587eddf3", "created_at": "2026-07-20T09:10:22+00:00", "title": "Reproduction poster (poster_embed.html)", "pinned": true, "pinned_at": "2026-07-20T09:10:22+00:00"}
-->
````html
<iframe src="poster_embed.html" title="MIRA reproduction poster" style="width:100%;height:510px;border:0;border-radius:12px"></iframe>
````
