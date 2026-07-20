import json
from pathlib import Path
import subprocess
import sys


def test_all_claims_gate() -> None:
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, "repro/src/verify_claims.py"], cwd=root, check=True)
    payload = json.loads((root / "outputs" / "claim_verification.json").read_text())
    assert payload["all_claims_complete"]
    assert payload["earned_points"] == 6
