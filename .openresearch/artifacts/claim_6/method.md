# Claim 6 method

The fixed campaign command downloads the five exact author tensors into a
shared content-addressed cache, checks commit-pinned URLs, byte sizes, SHA-256
hashes, dtypes, and shapes, then evaluates all four models on CPU.

The implementation follows released `mira(..., norm=True)` semantics:
per-pixel min–max normalization is computed from the 16 truths, 100 centers are
sampled uniformly in `[0,1]^12288`, one of 64 posterior draws defines each
radius, strict interior counts exclude the reference through the strict
comparison, and the released finite-sample probability is divided by its
maximum `64/65`. Squared Euclidean distances preserve the same comparisons.
A batched matrix multiplication computes the exact full-dimensional geometry
without allocating a four-model displacement tensor.

All 6,400 per-region/per-model/per-truth score cells are retained as CSV.
Uncertainty is a paired, deterministic, 10,000-replicate nonparametric
bootstrap that resamples both the 100 regions and 16 fiducials. A separate
standard-library checker recomputes aggregates, ranking, and paper deltas from
the CSV without importing torch or scoring code.

The negative control rolls truth-to-observation correspondence by one
fiducial, while retaining the posterior regions, and requires the broken
pairing to lower the true-model score.
