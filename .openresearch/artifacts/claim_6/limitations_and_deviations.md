# Claim 6 limitations and deviations

- The evaluation uses the authors' exact released posterior tensors rather
  than retraining the diffusion posterior. This directly reproduces the score
  claim, but does not independently reproduce model training. The paper says
  training/inference used A100 GPUs, which are prohibited in this CPU-only
  campaign.
- The paper does not publish the random seed used for its 100 MIRA regions.
  This reproduction pre-registers seed `260502014`; a score tolerance of 0.03
  is narrower than the paper's reported per-region standard deviations.
- The paper does not report the MIRA normalization flag. This node uses the
  released API default `norm=False`. A retained sibling evaluates `norm=True`;
  that path still identifies the true model but does not recover the remaining
  published ordering.
- The paper's plotted error bars are hardcoded in its plotting notebook and
  are not labeled with a precise estimator there. We report both the
  across-region standard deviation and a paired two-axis bootstrap interval.
- The independent checker verifies aggregation and the claimed ranking. Small
  implementation parity cells for the released MIRA routine remain covered by
  the cumulative Claim 1–3 regression suite.
