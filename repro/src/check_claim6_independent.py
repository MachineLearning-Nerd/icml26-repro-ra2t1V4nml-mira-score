"""Independent standard-library audit of Claim 6 raw CSV evidence."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".openresearch" / "artifacts" / "claim_6"
PAPER_SCORES = [0.6442, 0.5783, 0.5298, 0.5056]


def main() -> None:
    cells: dict[tuple[int, int], list[float]] = defaultdict(list)
    names: dict[int, str] = {}
    with (ARTIFACTS / "raw_per_region_truth.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            run_index = int(row["region_run"])
            model_index = int(row["model_index"])
            cells[(run_index, model_index)].append(float(row["score"]))
            names[model_index] = row["model"]
    row_count = sum(len(values) for values in cells.values())
    if row_count != 100 * 4 * 16 or len(cells) != 100 * 4:
        raise SystemExit(f"unexpected raw evidence dimensions: rows={row_count}, cells={len(cells)}")

    run_scores = {
        key: statistics.fmean(values)
        for key, values in cells.items()
    }
    means = []
    standard_deviations = []
    for model_index in range(4):
        values = [run_scores[(run_index, model_index)] for run_index in range(100)]
        means.append(statistics.fmean(values))
        standard_deviations.append(statistics.stdev(values))
    order = sorted(range(4), key=lambda index: means[index], reverse=True)
    maximum_paper_delta = max(abs(means[index] - PAPER_SCORES[index]) for index in range(4))
    payload = {
        "checker": "Python standard-library CSV aggregation, independent of torch scoring code",
        "raw_row_count": row_count,
        "region_model_cell_count": len(cells),
        "models": [
            {
                "model_index": index,
                "model": names[index],
                "mean": means[index],
                "region_std": standard_deviations[index],
                "paper_score": PAPER_SCORES[index],
                "absolute_paper_delta": abs(means[index] - PAPER_SCORES[index]),
            }
            for index in range(4)
        ],
        "observed_model_order": [names[index] for index in order],
        "model_order_matches_paper": order == [0, 1, 2, 3],
        "maximum_absolute_paper_delta": maximum_paper_delta,
        "within_0_03_numeric_reference_tolerance": maximum_paper_delta <= 0.03,
        "numeric_reference_alignment": (
            "ALIGNED" if maximum_paper_delta <= 0.03 else "DIVERGENT"
        ),
        "numeric_reference_is_claim_gate": False,
    }
    (ARTIFACTS / "independent_checker_output.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    if not payload["model_order_matches_paper"]:
        raise SystemExit("independent checker found a different ranking")


if __name__ == "__main__":
    main()
