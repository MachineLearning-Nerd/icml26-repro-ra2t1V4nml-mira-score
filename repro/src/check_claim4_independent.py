"""Independent exhaustive rational-grid checker for Proposition 3.5."""

from __future__ import annotations

import itertools
import json
from fractions import Fraction
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".openresearch" / "artifacts" / "claim_4"


def mean(values: tuple[Fraction, ...] | list[Fraction]) -> Fraction:
    return sum(values, Fraction(0)) / len(values)


def main() -> None:
    started = perf_counter()
    grid_size = 8
    x = tuple(Fraction(index, grid_size - 1) for index in range(grid_size))
    levels = tuple(Fraction(index, grid_size) for index in range(grid_size + 1))
    checked = 0
    equality_cases = 0
    minimum_gap: Fraction | None = None
    minimum_sequence: tuple[Fraction, ...] | None = None

    # T(0)=0. The remaining values exhaust every nondecreasing grid map.
    for suffix in itertools.combinations_with_replacement(levels, grid_size - 1):
        values = (Fraction(0),) + suffix
        gap = 2 * mean([left * right for left, right in zip(x, values, strict=True)]) - mean(values)
        checked += 1
        if gap == 0:
            equality_cases += 1
        if minimum_gap is None or gap < minimum_gap:
            minimum_gap = gap
            minimum_sequence = values
        if gap < 0:
            raise SystemExit(f"monotone-map counterexample found: {values!r}, gap={gap}")

    assert minimum_gap is not None and minimum_sequence is not None
    payload = {
        "checker": "exhaustive discrete Chebyshev inequality with fractions.Fraction",
        "grid_size": grid_size,
        "level_count": len(levels),
        "maps_checked": checked,
        "equality_cases": equality_cases,
        "minimum_gap": f"{minimum_gap.numerator}/{minimum_gap.denominator}",
        "minimum_sequence": [f"{value.numerator}/{value.denominator}" for value in minimum_sequence],
        "all_monotone_maps_nonnegative": minimum_gap >= 0,
        "runtime_seconds": perf_counter() - started,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "independent_checker_output.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
