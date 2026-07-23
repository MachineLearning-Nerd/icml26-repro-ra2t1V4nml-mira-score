"""Cumulative, claim-numbered campaign verdicts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OPENRESEARCH = ROOT / ".openresearch" / "artifacts"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    null_evidence = load(ROOT / "outputs" / "claims_1_2.json")
    claim4 = load(OPENRESEARCH / "claim_4" / "verifier_output.json")
    claim6 = load(OPENRESEARCH / "claim_6" / "verifier_output.json")
    laplace_cells_pass = all(
        abs(
            row["value"]
            - (
                (row["n"] + 1) / (row["N"] + 2)
                if row["k"] == 1
                else (row["N"] - row["n"] + 1) / (row["N"] + 2)
            )
        )
        < 1e-15
        for row in null_evidence["equation_cells"]
    )
    claims = [
        {
            "claim_id": 1,
            "verdict": "VERIFIED",
            "passed": (
                null_evidence["source_parity_passed"]
                and null_evidence["finite_null_probability_normalized"]
            ),
        },
        {"claim_id": 2, "verdict": "VERIFIED", "passed": laplace_cells_pass},
        {
            "claim_id": 3,
            "verdict": "VERIFIED",
            "passed": null_evidence["asymptotic_reference_convergence"],
        },
        {"claim_id": 4, "verdict": claim4["verdict"], "passed": claim4["verdict"] == "VERIFIED"},
        {"claim_id": 6, "verdict": claim6["verdict"], "passed": claim6["verdict"] == "VERIFIED"},
    ]
    payload = {
        "paper": "2605.02014",
        "claims": claims,
        "evaluated_claim_count": len(claims),
        "all_evaluated_claims_terminal": all(
            row["verdict"] in {"VERIFIED", "FALSIFIED", "BLOCKED"} for row in claims
        ),
        "all_evaluated_claim_checks_passed": all(row["passed"] for row in claims),
    }
    (OPENRESEARCH / "campaign_verdicts.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    if not (payload["all_evaluated_claims_terminal"] and payload["all_evaluated_claim_checks_passed"]):
        raise SystemExit("cumulative campaign verifier failed")


if __name__ == "__main__":
    main()
