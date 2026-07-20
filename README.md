# MIRA reproduction

Claim-by-claim reproduction of *MIRA: A Score for Conditional Distribution
Accuracy and Model Comparison* (ICML 2026, OpenReview `ra2t1V4nml`; arXiv
`2605.02014v1`).

The effective contract has three jury claims (six possible points):

1. MIRA estimates conditional-distribution match probability from joint samples.
2. Its correctly specified null has the reported theoretical reference values.
3. It supports direct Bayesian model comparison without numerical evidence
   estimation.

## Result

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

## Re-run

```bash
source .venv/bin/activate
python repro/src/run_tests.py
python repro/src/verify_claims.py
python repro/src/build_evidence_bundle.py
python repro/src/build_reproduction_bundle.py
```
