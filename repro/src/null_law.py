"""Independent finite-N law for the MIRA statistic under the null."""

from __future__ import annotations

from fractions import Fraction


def finite_null_moments(sample_count: int) -> dict[str, float]:
    """Integrate the uniform latent mass exactly, without the released code.

    With a uniform region mass lambda, n has the beta-binomial uniform law and
    k|n has the Laplace-successor probabilities used in the paper.  This
    enumerates their joint law with exact rational arithmetic.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    mean = Fraction(0)
    second = Fraction(0)
    total = Fraction(0)
    for n in range(sample_count + 1):
        denominator = (sample_count + 1) * (sample_count + 2)
        p_k1 = Fraction(n + 1, denominator)
        p_k0 = Fraction(sample_count - n + 1, denominator)
        stat_k1 = Fraction(n + 1, sample_count + 2)
        stat_k0 = Fraction(sample_count - n + 1, sample_count + 2)
        total += p_k1 + p_k0
        mean += p_k1 * stat_k1 + p_k0 * stat_k0
        second += p_k1 * stat_k1 * stat_k1 + p_k0 * stat_k0 * stat_k0
    variance = second - mean * mean
    return {
        "probability_total": float(total),
        "mean": float(mean),
        "variance": float(variance),
        "mean_error_to_two_thirds": abs(float(mean) - 2.0 / 3.0),
        "variance_error_to_one_over_18": abs(float(variance) - 1.0 / 18.0),
    }


def successor_probability(count: int, hit: int, sample_count: int) -> float:
    """Paper Eq. (successor statistic), independently of any tensor code."""
    if not 0 <= count <= sample_count or hit not in (0, 1):
        raise ValueError("invalid count or hit")
    numerator = count + 1 if hit else sample_count - count + 1
    return numerator / (sample_count + 2)
