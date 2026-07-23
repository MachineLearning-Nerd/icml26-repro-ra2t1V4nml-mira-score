# Status — six-claim release candidate

- Baseline live-judge score: `7/12`
- Current terminal evidence: Claims 1–4 and 6 `VERIFIED`; Claim 5 `BLOCKED`
- Publication state: **not published; explicit approval required**
- Score increase claimed: **no**
- Compute for new campaign: local Apple-arm CPU, $0; no GPU, no HF CPU fallback
- Winning lineage: `orx/cumulative-release-candidate-and-report`
- Fixed command: `uv run --frozen python repro/src/run_campaign.py`
- Detailed report: `reports/mira-reproduction/report.md`
- Tutorial: `notebooks/mira_reproduction.py`
- Space candidate: `release/space_candidate/`
- Text upload allowlist: `release/text_upload_allowlist.txt`

The prior three-claim gate below is retained as historical context and is
superseded by `.openresearch/artifacts/campaign_verdicts.json`.

# Prior status

- Paper: `ra2t1V4nml` — *MIRA: A Score for Conditional Distribution Accuracy and Model Comparison*
- Owner: `codex-mira-three-claims`
- State: `publication_queued`
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

Public GitHub evidence repository: `MachineLearning-Nerd/icml26-repro-ra2t1V4nml-mira-score`
at initial evidence commit `1dd53e8`. The complete gate was atomically added to
the canonical shared Hugging Face backlog as entry `67` (68 entries currently
in the backlog). The shared drain owns Space creation and will publish to the
canonical `DineshAI/ra2t1V4nml` target when quota permits; this session must not
publish directly. Next: await the drain's public Space readback and record its
commit, required tags, and artifact-bucket verification here.
