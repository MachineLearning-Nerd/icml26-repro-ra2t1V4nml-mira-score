import json
from pathlib import Path
import subprocess
import sys


def test_bayes_factor_identity_audit() -> None:
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, "repro/src/bayes_factor_audit.py"], cwd=root, check=True)
    payload = json.loads((root / "outputs" / "bayes_factor_audit.json").read_text())
    assert payload["finite_sample_arithmetic_ratio_is_unstable"]
    assert payload["mean_log_ratio_ranks_zero_shift"]
