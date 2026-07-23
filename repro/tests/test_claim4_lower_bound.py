import json
from pathlib import Path
import subprocess
import sys


def test_claim4_exact_lower_bound() -> None:
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, "repro/src/claim4_lower_bound.py"], cwd=root, check=True)
    subprocess.run([sys.executable, "repro/src/check_claim4_independent.py"], cwd=root, check=True)
    subprocess.run([sys.executable, "repro/src/verify_claim4.py"], cwd=root, check=True)
    payload = json.loads(
        (root / ".openresearch" / "artifacts" / "claim_4" / "verifier_output.json").read_text()
    )
    assert payload["verdict"] == "VERIFIED"
