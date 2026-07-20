import json
from pathlib import Path
import subprocess
import sys


def test_source_audit_passes() -> None:
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, "repro/src/audit_sources.py"], cwd=root, check=True)
    result = json.loads((root / "outputs" / "source_audit.json").read_text())
    assert result["all_static_checks_pass"]
