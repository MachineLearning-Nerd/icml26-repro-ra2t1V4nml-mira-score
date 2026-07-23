# Claim 6 evaluation

Verdict: **VERIFIED**

The exact author-released tensors have `L=16`, `N=64`, and 12,288 dimensions
(`64×64×3`). With 100 regions, the observed model scores were
`0.6831, 0.5097, 0.5032, 0.4953` in the paper's expected
order: correct prior/noise, noise-only misspecification, prior-only
misspecification, then both misspecified. Every paired 95% bootstrap interval
for the true-model advantage excludes zero. The independent CSV checker
recomputed the ranking without importing torch or the scoring implementation.

The paper's hardcoded plot scores are
`0.6442, 0.5783, 0.5298, 0.5056`; the largest absolute numerical difference is
`0.0686`. This numerical reference is
reported as **DIVERGENT**, not silently
treated as aligned. The exact source claim is detection and ranking, while the
paper omits the random seed and MIRA center configuration required for exact
number parity. A full-scale one-region check against the released scorer is
within one discrete count quantum.

The negative control rolls the fiducial truth-to-observation pairing by one,
deliberately breaking conditional correspondence; it must lower the intact
true-model score. This is a published-data evaluation, not a Gaussian or
low-dimensional proxy. Model training was not repeated because the released
posterior samples are the paper's evaluation inputs and training requires GPU
hardware outside the authorized CPU-only campaign.
