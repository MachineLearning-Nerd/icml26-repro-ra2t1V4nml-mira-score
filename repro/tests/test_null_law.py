from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from null_law import finite_null_moments, successor_probability


def test_successor_formula() -> None:
    assert successor_probability(2, 1, 5) == 3 / 7
    assert successor_probability(2, 0, 5) == 4 / 7


def test_finite_null_distribution_normalizes() -> None:
    result = finite_null_moments(100)
    assert abs(result["probability_total"] - 1.0) < 1e-15


def test_null_moments_approach_paper_reference() -> None:
    result = finite_null_moments(5000)
    assert result["mean_error_to_two_thirds"] < 1e-4
    assert result["variance_error_to_one_over_18"] < 1e-4
