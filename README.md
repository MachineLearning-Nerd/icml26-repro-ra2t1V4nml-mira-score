# MIRA reproduction: six-claim CPU audit

This repository reproduces the judged claims of *MIRA: A Score for Conditional
Distribution Accuracy and Model Comparison* (arXiv `2605.02014`) on CPU.
The campaign replaces the prior Claim 4 toy with an exact certificate, runs the
exact released `64×64×3` galaxy tensors for Claim 6, and audits why the 13-D
physical-lensing Claim 5 cannot be faithfully rerun from the public release.

Result: Claims 1–4 and 6 are **VERIFIED**; Claim 5 is **BLOCKED**, not failed,
because essential raw inputs and protocol values are unavailable. The Claim 6
paper scores are `[0.6442, 0.5783, 0.5298, 0.5056]`; the exact released-default
evaluation observes `[0.6831, 0.5097, 0.5032, 0.4953]`, reproducing the complete
order while retaining the maximum numerical delta (`0.0686`) as a divergence.
The terminal evidence was regenerated on local Apple-arm CPU. Three additional
Claim 5 posterior approaches used 94m47s of Hugging Face `cpu-upgrade`; none
passed the predeclared reliability gate. No GPU was used.

[Read the illustrated technical report](reports/mira-reproduction/report.md) ·
[Read the Claim 5 three-approach audit](reports/claim5-three-approach/report.md) ·
[Open the self-contained marimo tutorial](notebooks/mira_reproduction.py)

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-ra2t1V4nml-mira-score/blob/main/notebooks/mira_reproduction.py)

Open the notebook locally with
`uv run marimo edit notebooks/mira_reproduction.py` or serve it with
`uv run marimo run notebooks/mira_reproduction.py`.

## Experiment log

Every formal node uses the exact inherited command
`uv run --frozen python repro/src/run_campaign.py`.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
| --- | --- | --- | --- | --- |
| `main` | Publication surface | Not run as an experiment (publication surface) | Validated starting SHA `9af5d4e` | — |
| [`orx/frozen-judged-baseline-7-12`](https://github.com/MachineLearning-Nerd/icml26-repro-ra2t1V4nml-mira-score/tree/orx/frozen-judged-baseline-7-12) | Freeze judged baseline and uv lock | `uv run --frozen python repro/src/run_campaign.py` | 8/9; exposed one-float32-ULP tolerance issue | local CPU |
| [`orx/float32-parity-tolerance`](https://github.com/MachineLearning-Nerd/icml26-repro-ra2t1V4nml-mira-score/tree/orx/float32-parity-tolerance) | Explicit two-ULP parity tolerance | `uv run --frozen python repro/src/run_campaign.py` | 9/9 regression checks | local CPU |
| [`orx/claim-4-analytic-lower-bound-certificate`](https://github.com/MachineLearning-Nerd/icml26-repro-ra2t1V4nml-mira-score/tree/orx/claim-4-analytic-lower-bound-certificate) | Exact Proposition 3.5 certificate | `uv run --frozen python repro/src/run_campaign.py` | VERIFIED; 6,435-map checker | local CPU, 28.07 s |
| [`orx/claim-6-exact-released-galaxy-tensors`](https://github.com/MachineLearning-Nerd/icml26-repro-ra2t1V4nml-mira-score/tree/orx/claim-6-exact-released-galaxy-tensors) | `norm=True` ambiguity sibling | `uv run --frozen python repro/src/run_campaign.py` | True model first; remaining order differs | local CPU, 75.03 s |
| [`orx/claim-6-released-default-without-normalization`](https://github.com/MachineLearning-Nerd/icml26-repro-ra2t1V4nml-mira-score/tree/orx/claim-6-released-default-without-normalization) | Exact released tensors with API default | `uv run --frozen python repro/src/run_campaign.py` | VERIFIED; full order recovered | local CPU, 41.00 s |
| [`orx/claim-5-exact-release-completeness-audit`](https://github.com/MachineLearning-Nerd/icml26-repro-ra2t1V4nml-mira-score/tree/orx/claim-5-exact-release-completeness-audit) | Pinned raw-input/protocol audit | `uv run --frozen python repro/src/run_campaign.py` | BLOCKED by six documented omissions | local CPU, 39.47 s |
| [`orx/cumulative-release-candidate-and-report`](https://github.com/MachineLearning-Nerd/icml26-repro-ra2t1V4nml-mira-score/tree/orx/cumulative-release-candidate-and-report) | Cumulative gate, report, notebook, additive Space candidate | `uv run --frozen python repro/src/run_campaign.py` | All 19 fail-closed steps passed | local CPU, 53.03 s |
| [`orx/claim-5-preconditioned-hmc-posterior`](https://github.com/MachineLearning-Nerd/icml26-repro-ra2t1V4nml-mira-score/tree/orx/claim-5-preconditioned-hmc-posterior) | Exact-MH HMC posterior profile | `uv run --frozen python repro/src/run_campaign.py` | Rejected: 3/4 R̂ failures | HF `cpu-upgrade`, 29m03s |
| [`orx/claim-5-affine-ensemble-posterior`](https://github.com/MachineLearning-Nerd/icml26-repro-ra2t1V4nml-mira-score/tree/orx/claim-5-affine-ensemble-posterior) | Affine-invariant ensemble profile | `uv run --frozen python repro/src/run_campaign.py` | Rejected: true-model R̂ 2.289 | HF `cpu-upgrade`, 20m04s |
| [`orx/claim-5-adaptive-importance-posterior`](https://github.com/MachineLearning-Nerd/icml26-repro-ra2t1V4nml-mira-score/tree/orx/claim-5-adaptive-importance-posterior) | Adaptive multiscale importance profile | `uv run --frozen python repro/src/run_campaign.py` | Rejected: ESS 1.09–10.26 of 20,000 | HF `cpu-upgrade`, 4m29s |
| [`orx/claim-5-long-affine-convergence`](https://github.com/MachineLearning-Nerd/icml26-repro-ra2t1V4nml-mira-score/tree/orx/claim-5-long-affine-convergence) | Four-times-longer affine production | `uv run --frozen python repro/src/run_campaign.py` | Rejected: EPL+3 R̂ 1.587; SIE+3 R̂ 2.941 | HF `cpu-upgrade`, 41m11s |

The live judge’s starting score is **7/12**. This repository does not predict
or claim a score increase; only a future live verdict can establish one.

## Run the cumulative suite

```bash
uv sync --frozen
uv run --frozen python repro/src/run_campaign.py
uv run marimo edit notebooks/mira_reproduction.py
```

## Prior baseline documentation

# MIRA reproduction

Claim-by-claim reproduction of *MIRA: A Score for Conditional Distribution
Accuracy and Model Comparison* (ICML 2026, OpenReview `ra2t1V4nml`; arXiv
`2605.02014v1`).

The effective contract has three jury claims (six possible points):

1. MIRA estimates conditional-distribution match probability from joint samples.
2. Its correctly specified null has the reported theoretical reference values.
3. It supports direct Bayesian model comparison without numerical evidence
   estimation.

### Prior result

The local publication gate passes all three claims (6/6 points). Claim 1 has a
clean-room implementation parity check; Claim 2 has an exact finite-null
enumeration; Claim 3 has five retained full-scale T4 Job readbacks. The full
protocol uses 100 dimensions, 20 components, `L=N=5,000`, and 100 regions,
streamed to limit memory use.

The source does not release the GMM seed, component means/covariances, or the
experiment script. Those choices are therefore deterministic and disclosed in
`repro/src/gmm_full_protocol.py`, rather than presented as an exact byte-level
rerun. See `outputs/claim_verification.json` and the Trackio logbook for the
claim-level verdicts and caveat.

### Prior re-run

```bash
source .venv/bin/activate
python repro/src/run_tests.py
python repro/src/verify_claims.py
python repro/src/build_evidence_bundle.py
python repro/src/build_reproduction_bundle.py
```
