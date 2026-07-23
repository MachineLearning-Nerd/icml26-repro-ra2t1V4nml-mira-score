# Claim 5 evaluation

Verdict: **BLOCKED**

The paper reports the full `L=100`, `N=20,000`, 13-dimensional experiment and
the pinned plotting notebook contains aggregate scores
`0.6320, 0.5788, 0.5394, 0.5223`. Those numbers rank EPL+3 Sérsic first, but
they are hardcoded presentation values, not independently regenerable data.

Recursive, non-truncated Git tree audits at both pinned author commits found no
physical-model truth parameter set, no four posterior sample sets, no MALA
implementation or tuning, and no executable EPL/SIE–Sérsic forward-model
construction. The paper additionally omits values for parameters it calls
"held constant," dependency versions, and all relevant seeds. An aggregated
TARP `.npz` is present but cannot regenerate MIRA.

Accordingly, this campaign neither verifies nor falsifies the ranking. It marks
the exact claim **BLOCKED** and does not substitute the earlier GMM proxy.
