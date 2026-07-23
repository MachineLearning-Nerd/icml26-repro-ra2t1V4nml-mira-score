"""Fail-closed checks for the Claim 5 feasibility profile."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".openresearch" / "artifacts" / "claim_5_profile"


def main() -> None:
    summary = json.loads((ARTIFACTS / "profile_summary.json").read_text())
    with (ARTIFACTS / "raw_mira_cells.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        cells = list(csv.DictReader(handle))
    protocol = summary["protocol"]
    diagnostics = [
        row
        for model_rows in summary["diagnostics"].values()
        for row in model_rows
    ]
    assertions = {
        "explicitly_not_claim_evidence": summary["stage"]
        == "FEASIBILITY_PROFILE_NOT_CLAIM_EVIDENCE",
        "physical_image_shape_exact": protocol["pixels"] == [100, 100],
        "parameter_space_exact": protocol["dimensions"] == 13,
        "all_four_named_candidates": summary["models"]
        == [
            "EPL+3_Sersic_true",
            "SIE+3_Sersic",
            "EPL+1_Sersic",
            "SIE+1_Sersic",
        ],
        "posterior_sampler_configured": protocol["sampler"]
        == "preconditioned_hmc"
        and protocol["burnin_steps"] > 0
        and protocol["sampling_steps"] > 0
        and protocol["walkers"] > 1,
        "posterior_shapes": all(
            row["sample_shape"]
            == [protocol["posterior_samples_N"], 13]
            for row in diagnostics
        ),
        "finite_posteriors": all(row["finite_samples"] for row in diagnostics),
        "acceptance_non_degenerate": all(
            0.01 <= row["production_acceptance"] <= 0.99
            for row in diagnostics
        ),
        "paper_N_chain_convergence": (
            protocol["posterior_samples_N"] != 20_000
            or all(
                row["rhat_max"] is not None
                and row["rhat_max"] <= 1.20
                and row["ess_min"] is not None
                and row["ess_min"] >= 400
                for row in diagnostics
            )
        ),
        "raw_cell_count": len(cells)
        == protocol["regions"] * len(summary["models"]) * protocol["truths_L"],
        "cpu_only": summary["environment"]["device"] == "cpu",
        "fixed_command": summary["environment"]["command"]
        == "uv run --frozen python repro/src/run_campaign.py",
        "negative_control_is_rejection_case": summary["negative_control"][
            "expected_rejection"
        ],
        "profile_not_mislabeled_terminal": all(
            word not in summary["stage"] for word in ("VERIFIED", "FALSIFIED")
        ),
    }
    payload = {
        "claim_id": 5,
        "stage": "FEASIBILITY_PROFILE",
        "assertions": assertions,
        "passed": all(assertions.values()),
        "terminal_verdict": None,
    }
    (ARTIFACTS / "profile_verifier_output.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit("Claim 5 feasibility profile failed")


if __name__ == "__main__":
    main()
