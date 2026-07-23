"""Independent standard-library recomputation of Claim 5 release absence."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".openresearch" / "artifacts" / "claim_5"


def main() -> None:
    manifest = json.loads((ARTIFACTS / "release_tree_manifest.json").read_text())
    entries = [
        row
        for repository in manifest["repositories"]
        for row in repository["entries"]
    ]
    paths = [str(row["path"]).lower() for row in entries]
    code_suffixes = (".py", ".ipynb", ".jl", ".cpp", ".c")
    mala_code = [path for path in paths if "mala" in path and path.endswith(code_suffixes)]
    physical_posteriors = [
        path
        for path in paths
        if "posterior" in path
        and any(token in path for token in ("model_misspec", "sersic", "epl", "mala"))
    ]
    physical_truth = [
        path
        for path in paths
        if "truth" in path
        and any(token in path for token in ("model_misspec", "sersic", "epl", "mala"))
    ]
    aggregate_tarp = [path for path in paths if "model_misspecification_tarp_data.npz" in path]
    payload = {
        "checker": "independent path-level audit over pinned Git tree manifests",
        "repository_count": len(manifest["repositories"]),
        "total_tree_entries": len(entries),
        "all_trees_complete": all(
            repository["truncated"] is False for repository in manifest["repositories"]
        ),
        "mala_code_paths": mala_code,
        "physical_model_posterior_paths": physical_posteriors,
        "physical_model_truth_paths": physical_truth,
        "aggregate_tarp_paths": aggregate_tarp,
        "exact_inputs_absent": not mala_code and not physical_posteriors and not physical_truth,
        "aggregate_tarp_is_not_raw_posterior_evidence": bool(aggregate_tarp),
    }
    (ARTIFACTS / "independent_checker_output.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    if not (
        payload["all_trees_complete"]
        and payload["exact_inputs_absent"]
        and payload["aggregate_tarp_is_not_raw_posterior_evidence"]
    ):
        raise SystemExit("independent release-absence audit failed")


if __name__ == "__main__":
    main()
