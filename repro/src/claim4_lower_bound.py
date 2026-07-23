"""Exact finite-N evidence for the MIRA lower bound."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".openresearch" / "artifacts" / "claim_4"
SAMPLE_COUNTS = (1, 64, 20_000)


def fraction_payload(value: Fraction) -> dict[str, object]:
    return {
        "fraction": f"{value.numerator}/{value.denominator}",
        "decimal": float(value),
    }


def score(sample_count: int, gap: Fraction) -> Fraction:
    return Fraction(1, 2) + Fraction(sample_count, sample_count + 2) * gap


def power_map_gap(exponent: Fraction) -> Fraction:
    """2 E[U*T(U)] - E[T(U)] for T(u)=u**exponent."""
    return Fraction(2, 1) / (exponent + 2) - Fraction(1, 1) / (exponent + 1)


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    started = perf_counter()
    analytic_maps = {
        "disjoint_limit_T_zero": Fraction(0),
        "sqrt_T": power_map_gap(Fraction(1, 2)),
        "null_T_identity": power_map_gap(Fraction(1)),
        "quadratic_T": power_map_gap(Fraction(2)),
        "eighth_power_T": power_map_gap(Fraction(8)),
    }
    rows = []
    for sample_count in SAMPLE_COUNTS:
        for name, gap in analytic_maps.items():
            value = score(sample_count, gap)
            rows.append(
                {
                    "N": sample_count,
                    "map": name,
                    "chebyshev_gap": fraction_payload(gap),
                    "paper_score": fraction_payload(value),
                    "at_or_above_half": value >= Fraction(1, 2),
                }
            )

    normalized_floors = [
        {
            "N": sample_count,
            "released_normalized_floor": fraction_payload(
                Fraction(sample_count + 2, 2 * (sample_count + 1))
            ),
        }
        for sample_count in SAMPLE_COUNTS
    ]

    decreasing_gap = Fraction(-1, 6)
    negative_rows = [
        {
            "N": sample_count,
            "map": "T(u)=1-u",
            "chebyshev_gap": fraction_payload(decreasing_gap),
            "paper_score": fraction_payload(score(sample_count, decreasing_gap)),
            "below_half": score(sample_count, decreasing_gap) < Fraction(1, 2),
            "expected_rejection_reason": "T is decreasing and violates the nested-radial-CDF contract",
        }
        for sample_count in SAMPLE_COUNTS
    ]

    raw = {
        "claim_id": 4,
        "arithmetic": "fractions.Fraction exact rational arithmetic",
        "identity": "mu = 1/2 + N/(N+2) * gap",
        "gap": "2 E[U T(U)] - E[T(U)]",
        "rows": rows,
        "released_implementation_normalized_floors": normalized_floors,
        "paper_scale_cells": [
            {"dimensions": 13, "L": 100, "N": 20_000, "regions": 100},
            {"dimensions": 12_288, "L": 16, "N": 64, "regions": 100},
        ],
    }
    negative = {
        "claim_id": 4,
        "control_type": "out-of-contract negative control",
        "expected_result": "all cells fall below 1/2",
        "rows": negative_rows,
        "negative_control_passed": all(row["below_half"] for row in negative_rows),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "raw_exact_results.json").write_text(
        json.dumps(raw, indent=2) + "\n",
        encoding="utf-8",
    )
    (ARTIFACTS / "negative_control_output.json").write_text(
        json.dumps(negative, indent=2) + "\n",
        encoding="utf-8",
    )
    environment = {
        "command": "uv run --frozen python repro/src/run_campaign.py",
        "git_sha": git_sha(),
        "uv_lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "deterministic_seeds": [],
        "runtime_seconds": perf_counter() - started,
    }
    (ARTIFACTS / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n",
        encoding="utf-8",
    )
    if not all(row["at_or_above_half"] for row in rows):
        raise SystemExit("analytic lower-bound cell failed")
    if not negative["negative_control_passed"]:
        raise SystemExit("negative control did not reject the removed monotonicity assumption")
    print(json.dumps({"raw": raw, "negative_control": negative, "environment": environment}, indent=2))


if __name__ == "__main__":
    main()
