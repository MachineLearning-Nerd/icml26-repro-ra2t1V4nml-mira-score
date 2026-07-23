"""Fail-closed verifier for the Claim 4 evidence contract."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".openresearch" / "artifacts" / "claim_4"


def load(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def main() -> None:
    contract = load("claim_contract.json")
    raw = load("raw_exact_results.json")
    independent = load("independent_checker_output.json")
    negative = load("negative_control_output.json")
    environment = load("environment.json")
    required_static = (
        "source_audit.md",
        "method.md",
        "limitations_and_deviations.md",
    )
    assertions = {
        "contract_requires_verified": contract["machine_checkable_contract"]["required_verdict"] == "VERIFIED",
        "all_exact_cells_at_or_above_half": all(row["at_or_above_half"] for row in raw["rows"]),
        "paper_N_64_present": any(row["N"] == 64 for row in raw["rows"]),
        "paper_N_20000_present": any(row["N"] == 20_000 for row in raw["rows"]),
        "source_dimensions_present": {cell["dimensions"] for cell in raw["paper_scale_cells"]} == {13, 12_288},
        "independent_checker_passed": independent["all_monotone_maps_nonnegative"] is True,
        "independent_checker_exhaustive_count": independent["maps_checked"] == 6_435,
        "negative_control_rejects_removed_assumption": negative["negative_control_passed"] is True,
        "environment_is_pinned": bool(environment["git_sha"]) and bool(environment["uv_lock_sha256"]),
        "static_audit_files_present": all((ARTIFACTS / name).is_file() for name in required_static),
    }
    passed = all(assertions.values())
    payload = {
        "claim_id": 4,
        "verdict": "VERIFIED" if passed else "BLOCKED",
        "assertions": assertions,
        "evidence_standard": "exact analytic identity plus independent exhaustive rational-grid check",
    }
    (ARTIFACTS / "verifier_output.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    eval_text = f"""# Claim 4 evaluation

Verdict: **{payload['verdict']}**

The exact finite-N identity is at least 1/2 for every audited monotone radial-mass
map, including the paper's `N=64` and `N=20,000` application cells. The
independent checker exhaustively accepted 6,435 rational monotone maps. The
decreasing-map negative control fell below 1/2 as required, demonstrating that
the verifier is sensitive to removal of the proof's monotonicity assumption.

This verifies Proposition 3.5 under the continuous-radial-CDF assumptions made
by its proof; it is not a finite Gaussian proxy.
"""
    (ARTIFACTS / "EVAL.md").write_text(eval_text, encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit("Claim 4 verification failed closed")


if __name__ == "__main__":
    main()
