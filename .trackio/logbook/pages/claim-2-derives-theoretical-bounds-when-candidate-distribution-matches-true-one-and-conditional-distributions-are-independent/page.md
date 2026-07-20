# Claim 2: Derives theoretical bounds when candidate distribution matches true one and conditional distributions are independent.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_bdb98cb7a8f5", "created_at": "2026-07-20T09:10:22+00:00", "title": "Claim 2: Derives theoretical bounds when candidate distribution matches true one and conditional distributions are independent."}
-->
The primary source and official implementation are hash-pinned. Independently,
the latent uniform mass is integrated with exact rational arithmetic for seven
finite sample counts through `N=5,000`: every joint probability normalizes to
one, while the mean and variance are respectively within `6.66e-5` of `2/3`
and `1.11e-5` of `1/18` at the source-scale count. This is separate from the
released Monte Carlo code.


---
<!-- trackio-cell
{"type": "code", "id": "cell_3bf556ed1e6d", "created_at": "2026-07-20T09:11:35+00:00", "title": "Run: python audit_sources.py (exit 0)", "command": ["python", "repro/src/audit_sources.py"], "exit_code": 0, "duration_s": 0.092}
-->
````bash
$ python repro/src/audit_sources.py
````

exit 0 · 0.1s


````python title=audit_sources.py
"""Fail-closed source/protocol audit for the three MIRA claims."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    pins = json.loads((ROOT / "docs" / "source_pins.json").read_text(encoding="utf-8"))
    archive = ROOT / "docs" / "arxiv_source.tar"
    official = ROOT / "upstream" / "src" / "mira_score" / "mira.py"
    with tarfile.open(archive) as bundle:
        tex_member = next(member for member in bundle.getmembers() if member.name.endswith("icml_2026_main_conference.tex"))
        tex = bundle.extractfile(tex_member).read().decode("utf-8")
    code = official.read_text(encoding="utf-8")
    source_checks = {
        "primary_source_hash_matches": digest(archive) == pins["arxiv_source_sha256"],
        "official_statistic": "(counts + 1) / (N + 2)" in code and "(N - counts + 1) / (N + 2)" in code,
        "official_calibration": "calib = prob / max_val" in code,
        "paper_null_beta_law": "Beta$(2,1)$" in tex and "\\tfrac23" in tex and "\\tfrac{1}{18}" in tex,
        "paper_full_gmm_dimensions": "100$ dimensions" in tex,
        "paper_full_gmm_components": "20$ mixture components" in tex,
        "paper_full_gmm_samples": "$N=5\\,000$ samples" in tex and "$L=5\\,000$ true samples" in tex,
        "paper_full_gmm_regions": "100$ regions per true sample" in tex,
        "paper_bayes_factor_claim": "Bayesian model comparison" in tex and "bypassing the challenging evidence computation" in tex,
    }
    payload = {
        "source_pins": pins,
        "source_checks": source_checks,
        "all_static_checks_pass": all(source_checks.values()),
        "undocumented_gmm_details": ["seed", "component means", "component covariance", "released experiment script"],
    }
    output = ROOT / "outputs" / "source_audit.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["all_static_checks_pass"]:
        raise SystemExit("source audit failed")


if __name__ == "__main__":
    main()

````


````output
{
  "source_pins": {
    "openreview_id": "ra2t1V4nml",
    "arxiv_id": "2605.02014v1",
    "arxiv_source_sha256": "93f93308136b7078018639dd75ab58d977994817e2cda7946c96477a3240a184",
    "primary_pdf_sha256": "430adb66f9a8c8b4b9db5f868e5bf65aab423d1d05e26873f71718b443b03419",
    "official_repository": "https://github.com/SammyS15/mira-score",
    "official_commit": "c57487198ac30711783b78ac2af6a76758544483",
    "official_claims_sha256": "af5ab2d62f786ae36861957cbd08b4188f6d4c86e67152becc661a9c5bbb9d57"
  },
  "source_checks": {
    "primary_source_hash_matches": true,
    "official_statistic": true,
    "official_calibration": true,
    "paper_null_beta_law": true,
    "paper_full_gmm_dimensions": true,
    "paper_full_gmm_components": true,
    "paper_full_gmm_samples": true,
    "paper_full_gmm_regions": true,
    "paper_bayes_factor_claim": true
  },
  "all_static_checks_pass": true,
  "undocumented_gmm_details": [
    "seed",
    "component means",
    "component covariance",
    "released experiment script"
  ]
}

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_312005efb739", "created_at": "2026-07-20T09:11:37+00:00", "title": "Run: python verify_claims_1_2.py (exit 0)", "command": ["python", "repro/src/verify_claims_1_2.py"], "exit_code": 0, "duration_s": 1.998}
-->
````bash
$ python repro/src/verify_claims_1_2.py
````

exit 0 · 2.0s


````python title=verify_claims_1_2.py
"""Independent checks for MIRA's statistic and correctly specified null law."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from mira_score import mira

from null_law import finite_null_moments, successor_probability


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"


def direct_score(truth: torch.Tensor, posterior: torch.Tensor, runs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Clean-room direct implementation of the released random-ball estimator."""
    models, truths, samples, dimensions = posterior.shape
    assert truth.shape == (truths, dimensions)
    scores = []
    for _ in range(runs):
        centers = torch.rand((truths, dimensions), dtype=truth.dtype)
        distances = torch.linalg.vector_norm(centers[None, :, None, :] - posterior, dim=-1)
        indices = torch.randint(0, samples, (models, truths))
        model_index = torch.arange(models)[:, None]
        truth_index = torch.arange(truths)[None, :]
        radii = distances[model_index, truth_index, indices]
        keep = torch.ones((models, truths, samples), dtype=torch.bool)
        keep[model_index, truth_index, indices] = False
        counts = (distances.masked_fill(~keep, float("inf")) < radii[..., None]).sum(dim=-1)
        hit = (torch.linalg.vector_norm(centers - truth, dim=-1)[None, :] <= radii).to(truth.dtype)
        probability = torch.where(
            hit.bool(),
            (counts + 1) / samples,
            (samples - counts) / samples,
        )
        scores.append(probability.mean(dim=1))
    stacked = torch.stack(scores)
    return stacked.mean(dim=0), stacked.std(dim=0, unbiased=True)


def main() -> None:
    torch.manual_seed(20260720)
    truth = torch.rand((19, 3))
    posterior = torch.rand((2, 19, 37, 3))
    runs = 11

    torch.manual_seed(314159)
    official_mean, official_std = mira(
        truth, posterior, num_runs=runs, disable_tqdm=True, device=torch.device("cpu")
    )
    torch.manual_seed(314159)
    independent_mean, independent_std = direct_score(truth, posterior, runs)
    source_delta = float(torch.max(torch.abs(official_mean - independent_mean)))
    source_std_delta = float(torch.max(torch.abs(official_std - independent_std)))

    cells = {str(n): finite_null_moments(n) for n in (1, 2, 5, 25, 100, 1000, 5000)}
    equation_cells = [
        {"N": n_total, "n": count, "k": hit, "value": successor_probability(count, hit, n_total)}
        for n_total, count, hit in ((5, 0, 0), (5, 2, 1), (5, 5, 0), (100, 37, 1))
    ]
    payload = {
        "official_independent_max_abs_delta": source_delta,
        "official_independent_std_delta": source_std_delta,
        "source_parity_passed": source_delta < 1e-12 and source_std_delta < 1e-7,
        "finite_null_cells": cells,
        "finite_null_probability_normalized": all(abs(cell["probability_total"] - 1.0) < 1e-15 for cell in cells.values()),
        "asymptotic_reference_convergence": cells["5000"]["mean_error_to_two_thirds"] < 1e-4
        and cells["5000"]["variance_error_to_one_over_18"] < 1e-4,
        "equation_cells": equation_cells,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "claims_1_2.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not (payload["source_parity_passed"] and payload["finite_null_probability_normalized"]):
        raise SystemExit("claim 1/2 verification failed")


if __name__ == "__main__":
    main()

````


````output
{
  "official_independent_max_abs_delta": 0.0,
  "official_independent_std_delta": 2.2351741790771484e-08,
  "source_parity_passed": true,
  "finite_null_cells": {
    "1": {
      "probability_total": 1.0,
      "mean": 0.5555555555555556,
      "variance": 0.024691358024691357,
      "mean_error_to_two_thirds": 0.11111111111111105,
      "variance_error_to_one_over_18": 0.030864197530864196
    },
    "2": {
      "probability_total": 1.0,
      "mean": 0.5833333333333334,
      "variance": 0.034722222222222224,
      "mean_error_to_two_thirds": 0.08333333333333326,
      "variance_error_to_one_over_18": 0.02083333333333333
    },
    "5": {
      "probability_total": 1.0,
      "mean": 0.6190476190476191,
      "variance": 0.045351473922902494,
      "mean_error_to_two_thirds": 0.04761904761904756,
      "variance_error_to_one_over_18": 0.010204081632653059
    },
    "25": {
      "probability_total": 1.0,
      "mean": 0.654320987654321,
      "variance": 0.053345526596555407,
      "mean_error_to_two_thirds": 0.012345679012345623,
      "variance_error_to_one_over_18": 0.002210028959000146
    },
    "100": {
      "probability_total": 1.0,
      "mean": 0.6633986928104575,
      "variance": 0.055000213593062494,
      "mean_error_to_two_thirds": 0.0032679738562091387,
      "variance_error_to_one_over_18": 0.0005553419624930583
    },
    "1000": {
      "probability_total": 1.0,
      "mean": 0.666333998669328,
      "variance": 0.05550000022133599,
      "mean_error_to_two_thirds": 0.0003326679973386648,
      "variance_error_to_one_over_18": 5.555533421956055e-05
    },
    "5000": {
      "probability_total": 1.0,
      "mean": 0.6666000266560043,
      "variance": 0.0555444444462208,
      "mean_error_to_two_thirds": 6.664001066236658e-05,
      "variance_error_to_one_over_18": 1.1111109334750735e-05
    }
  },
  "finite_null_probability_normalized": true,
  "asymptotic_reference_convergence": true,
  "equation_cells": [
    {
      "N": 5,
      "n": 0,
      "k": 0,
      "value": 0.8571428571428571
    },
    {
      "N": 5,
      "n": 2,
      "k": 1,
      "value": 0.42857142857142855
    },
    {
      "N": 5,
      "n": 5,
      "k": 0,
      "value": 0.14285714285714285
    },
    {
      "N": 100,
      "n": 37,
      "k": 1,
      "value": 0.37254901960784315
    }
  ]
}

````
