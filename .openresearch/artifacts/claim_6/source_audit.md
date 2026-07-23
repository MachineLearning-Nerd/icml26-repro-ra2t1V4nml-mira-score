# Claim 6 source audit

Paper source: `https://export.arxiv.org/e-print/2605.02014v1`, retrieved
2026-07-23 with an explicit `OpenResearch-MIRA-Reproduction/1.0` User-Agent,
SHA-256 `93f93308136b7078018639dd75ab58d977994817e2cda7946c96477a3240a184`.

The main-text strong-lensing image experiment is in Section 5.3 of the TeX
source (lines 498–510); implementation details appear at lines 1083–1106. The
claim uses 16 fiducial images, 64 posterior samples per image, 100 random
regions, and RGB images of shape 64×64 (12,288 dimensions). The four conditions
cross a spiral versus elliptical source prior with Gaussian noise standard
deviation 2 versus 0.5. Spiral/2 is the data-generating condition.

The authors' plotting repository is pinned at
`SammyS15/MIRA_Paper_Plots@3bc229222cbcf72bd470267175d9a6dff6689ce0`.
Its `data/lens_exp` directory releases the exact truth and four posterior
tensors. `notebooks/main_plotting.ipynb` records the paper scores
`[0.6442, 0.5783, 0.5298, 0.5056]` and errors
`[0.0606, 0.0728, 0.0748, 0.0690]` in the condition order above.

Quantifier tested: on these exact released evaluation tensors at the paper's
full scale and `L=16`, the correctly specified condition ranks first and each
stated prior/noise misspecification reduces MIRA.
