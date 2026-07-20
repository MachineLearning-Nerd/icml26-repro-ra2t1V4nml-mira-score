"""Independent audit of the paper's stated arithmetic Bayes-factor comparator."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from gmm_full_protocol import log_gmm_density


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    generator = torch.Generator(device="cpu").manual_seed(260502014)
    dimensions, components, observations = 100, 20, 100_000
    means = torch.randn((components, dimensions), generator=generator)
    assignments = torch.randint(components, (observations,), generator=generator)
    truth = torch.randn((observations, dimensions), generator=generator) + means[assignments]
    truth_log_density = log_gmm_density(truth, means)
    rows = []
    for shift in (-6, -3, 0, 3, 6):
        candidate = means + torch.full((dimensions,), shift / math.sqrt(dimensions))
        log_ratio = log_gmm_density(truth, candidate) - truth_log_density
        rows.append(
            {
                "shift": shift,
                "mean_density_ratio": float(torch.exp(log_ratio).mean()),
                "mean_log_density_ratio": float(log_ratio.mean()),
            }
        )
    payload = {
        "observations": observations,
        "identity": "E_{y~p_true}[p_candidate(y)/p_true(y)] = 1 for normalized candidate density",
        "rows": rows,
        "finite_sample_arithmetic_ratio_is_unstable": any(
            abs(row["mean_density_ratio"] - 1.0) > 0.15 for row in rows if row["shift"] != 0
        ),
        "mean_log_ratio_ranks_zero_shift": rows[2]["mean_log_density_ratio"] > max(
            row["mean_log_density_ratio"] for index, row in enumerate(rows) if index != 2
        ),
    }
    path = ROOT / "outputs" / "bayes_factor_audit.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not (
        payload["finite_sample_arithmetic_ratio_is_unstable"]
        and payload["mean_log_ratio_ranks_zero_shift"]
    ):
        raise SystemExit("Bayes-factor audit failed")


if __name__ == "__main__":
    main()
