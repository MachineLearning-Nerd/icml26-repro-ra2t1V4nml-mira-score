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
