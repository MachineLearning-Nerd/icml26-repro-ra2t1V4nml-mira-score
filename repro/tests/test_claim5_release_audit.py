import json
from pathlib import Path


def test_claim5_is_terminal_and_not_promoted_from_aggregate_numbers() -> None:
    root = Path(__file__).resolve().parents[2]
    artifacts = root / ".openresearch" / "artifacts" / "claim_5"
    audit = json.loads((artifacts / "raw_release_audit.json").read_text())
    verifier = json.loads((artifacts / "verifier_output.json").read_text())
    assert audit["paper_protocol"]["L"] == 100
    assert audit["paper_protocol"]["N"] == 20_000
    assert audit["paper_protocol"]["dimensions"] == 13
    assert audit["essential_inputs_complete"] is False
    assert verifier["verdict"] == "BLOCKED"
    assert verifier["blocked_requirements_verified"] is True
