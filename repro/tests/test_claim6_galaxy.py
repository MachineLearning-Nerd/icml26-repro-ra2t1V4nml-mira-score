import json
from pathlib import Path


def test_claim6_evidence_is_full_scale_and_verified() -> None:
    root = Path(__file__).resolve().parents[2]
    artifacts = root / ".openresearch" / "artifacts" / "claim_6"
    summary = json.loads((artifacts / "raw_summary.json").read_text())
    verifier = json.loads((artifacts / "verifier_output.json").read_text())
    assert summary["protocol"] == {
        "L": 16,
        "N": 64,
        "dimensions": 12_288,
        "image_shape": [64, 64, 3],
        "regions": 100,
        "norm": False,
        "center_distribution": "Uniform[0,1]^12288",
        "scoring_seed": 260502014,
    }
    assert verifier["verdict"] == "VERIFIED"
