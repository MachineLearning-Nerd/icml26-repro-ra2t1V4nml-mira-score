# Claim 5 source audit

Primary source: `arXiv:2605.02014v1`, source SHA-256
`93f93308136b7078018639dd75ab58d977994817e2cda7946c96477a3240a184`.
The exact statement appears in Section 5.3, TeX lines 480–486; protocol details
appear in Appendix lines 1025–1080.

The true model is EPL + three Sérsic sources. Candidates are EPL+3, SIE+3,
EPL+1, and SIE+1. The paper draws `L=100` truths, obtains `N=20,000` MALA
posterior samples per observation/model, evaluates all models in 13 dimensions,
and states that source-count agreement matters more than lens-profile
agreement. MALA uses 100 walkers, 200 burn-in steps, and 200 sampling steps.
Images are 100×100 at 0.05 arcsec/pixel with unit Gaussian noise.

The paper gives priors for four EPL lens parameters and three parameters per
Sérsic source. It says all other parameters are held constant but does not give
their values. It gives no MALA step size, implementation, initialization,
software versions, random seeds, or exact MIRA center configuration.

The author plotting repository at
`SammyS15/MIRA_Paper_Plots@3bc229222cbcf72bd470267175d9a6dff6689ce0`
hardcodes scores `[0.6319946647, 0.5787630081, 0.5393556356,
0.5222544074]`. It releases an aggregate TARP `.npz`, but no truth parameters,
physical-model posterior samples, forward-model source, or MALA source.
The score implementation repository is pinned at
`SammyS15/mira-score@c57487198ac30711783b78ac2af6a76758544483`.
