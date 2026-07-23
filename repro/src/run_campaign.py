"""Run the complete, fail-closed reproduction contract.

The OpenResearch command is intentionally fixed at this wrapper. Experiment
children may add checks here, but they may not change the command or environment.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
FIXED_COMMAND = "uv run --frozen python repro/src/run_campaign.py"
BASELINE_STEPS = (
    "repro/src/audit_sources.py",
    "repro/src/verify_claims_1_2.py",
    "repro/src/bayes_factor_audit.py",
    "repro/src/aggregate_full_gmm.py",
    "repro/src/claim4_lower_bound.py",
    "repro/src/check_claim4_independent.py",
    "repro/src/verify_claim4.py",
    "repro/src/claim6_galaxy_images.py",
    "repro/src/check_claim6_independent.py",
    "repro/src/verify_claim6.py",
    "repro/src/claim5_release_audit.py",
    "repro/src/check_claim5_independent.py",
    "repro/src/verify_claim5.py",
    "repro/src/run_tests.py",
    "repro/src/verify_claims.py",
    "repro/src/verify_campaign.py",
    "repro/src/build_evidence_bundle.py",
    "repro/src/build_reproduction_bundle.py",
)


def git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run_step(relative_path: str) -> dict[str, object]:
    started = perf_counter()
    print(f"\n=== RUN {relative_path} ===", flush=True)
    completed = subprocess.run(
        [sys.executable, relative_path],
        cwd=ROOT,
        check=False,
    )
    elapsed = perf_counter() - started
    record = {
        "path": relative_path,
        "returncode": completed.returncode,
        "runtime_seconds": elapsed,
    }
    print(f"=== END {relative_path}: rc={completed.returncode}, seconds={elapsed:.3f} ===", flush=True)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, relative_path)
    return record


def main() -> None:
    started = perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    sha = git_sha()
    print(
        json.dumps(
            {
                "event": "campaign_start",
                "command": FIXED_COMMAND,
                "git_sha": sha,
                "python": sys.version,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
                "started_utc": started_utc,
            },
            indent=2,
        ),
        flush=True,
    )

    records = [run_step(step) for step in BASELINE_STEPS]
    payload = {
        "command": FIXED_COMMAND,
        "git_sha": sha,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": perf_counter() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "steps": records,
        "all_steps_passed": all(record["returncode"] == 0 for record in records),
    }
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "campaign_run.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n=== CAMPAIGN SUMMARY ===")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
