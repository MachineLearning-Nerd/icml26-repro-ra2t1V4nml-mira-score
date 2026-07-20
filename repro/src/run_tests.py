"""Run the complete test suite and retain its exact exit status for the gate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "repro/tests"], cwd=ROOT, check=False, text=True, capture_output=True
    )
    payload = {
        "tests_passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    (ROOT / "outputs" / "test_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
