"""Aggregate the five preregistered exact-scale GMM seeds after Hub readback."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FULL = {"dimensions": 100, "components": 20, "truths": 5000, "samples": 5000, "regions": 100}


def main() -> None:
    root = ROOT / "outputs" / "hub_readback" / "full_gmm"
    files = sorted(root.glob("*.json"))
    if len(files) != 5:
        raise SystemExit(f"expected five full-scale seed files, found {len(files)}")
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    if not all(run["configuration"] == FULL and run["exact_source_scale"] for run in runs):
        raise SystemExit("one or more runs are not exact source scale")
    margins = []
    for run in runs:
        rows = run["rows"]
        margins.append(rows[2]["mira"] - max(rows[1]["mira"], rows[3]["mira"]))
    mean_margin = sum(margins) / len(margins)
    std_margin = (sum((value - mean_margin) ** 2 for value in margins) / (len(margins) - 1)) ** 0.5
    sem_margin = std_margin / math.sqrt(len(margins))
    payload = {
        "seed_file_count": len(files),
        "files": [path.name for path in files],
        "all_zero_shift_best": all(run["zero_shift_best"] for run in runs),
        "all_absolute_shift_monotone": all(run["absolute_shift_monotone"] for run in runs),
        "nearest_shift_margins": margins,
        "mean_nearest_shift_margin": mean_margin,
        "std_nearest_shift_margin": std_margin,
        "sem_nearest_shift_margin": sem_margin,
        "three_sem_separation": mean_margin > 3 * sem_margin,
        "mean_runtime_seconds": sum(run["runtime_seconds"] for run in runs) / len(runs),
    }
    (ROOT / "outputs" / "full_gmm_aggregate.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not (
        payload["all_zero_shift_best"]
        and payload["all_absolute_shift_monotone"]
        and payload["three_sem_separation"]
    ):
        raise SystemExit("multi-seed full-scale GMM gate failed")


if __name__ == "__main__":
    main()
