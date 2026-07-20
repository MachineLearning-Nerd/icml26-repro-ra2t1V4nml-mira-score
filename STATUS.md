# Status

- Paper: `ra2t1V4nml` — *MIRA: A Score for Conditional Distribution Accuracy and Model Comparison*
- Owner: `codex-mira-three-claims`
- State: `publication_ready`
- Effective contract: 3 live claims / 6 possible points
- Primary source: arXiv `2605.02014v1`, source archive SHA-256
  `93f93308136b7078018639dd75ab58d977994817e2cda7946c96477a3240a184`
- Official implementation: `SammyS15/mira-score@c57487198ac30711783b78ac2af6a76758544483`

## Current step

**FULL THREE-CLAIM / SIX-POINT LOCAL PUBLICATION GATE PASSED.** The fail-closed
verifier reports `3/3` claims, `6/6` points, and `9` passing tests. Claim 1 has
clean-room parity with the released estimator (mean delta `0.0`, s.d. delta
`2.24e-8`). Claim 2 has an exact finite-null calculation at `N=5,000` (mean
error `6.66e-5` from `2/3`; variance error `1.11e-5` from `1/18`). Claim 3 has
five durable source-scale T4 Job readbacks; every seed ranks zero shift first
and preserves the absolute-shift order. The aggregate nearest-shift margin is
`0.00217222`, or `8.30` standard errors.

The raw arithmetic density-ratio Bayes-factor average is retained as a
negative-control caveat: it is unstable in the independent 100,000-observation
high-dimensional audit. The verified claim is the direct MIRA ranking plus a
correctly ranked log-density control. The primary source omits GMM construction
details, so the released dimensions/counts are reproduced exactly while the
seeded isotropic construction is explicit rather than overstated as a literal
experiment rerun.

## Next action

Create and push the public GitHub evidence repository. Only after the first
push succeeds, atomically enqueue this gate-complete directory through the
canonical shared Hugging Face backlog; the shared drain, not this session, owns
Space publication and public readback.
