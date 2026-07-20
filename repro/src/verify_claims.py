"""Fail closed unless every live MIRA jury claim has retained evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
FULL = {"dimensions": 100, "components": 20, "truths": 5000, "samples": 5000, "regions": 100}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    source = load(OUT / "source_audit.json")
    claims_1_2 = load(OUT / "claims_1_2.json")
    bayes = load(OUT / "bayes_factor_audit.json")
    full_path = OUT / "hub_readback" / "full_gmm" / "gmm_bayes_factor.json"
    full = load(full_path)
    aggregate = load(OUT / "full_gmm_aggregate.json")
    test_path = OUT / "test_results.json"
    tests_passed = test_path.is_file() and load(test_path).get("tests_passed") is True
    claim_1 = source["all_static_checks_pass"] and claims_1_2["source_parity_passed"]
    claim_2 = (
        claims_1_2["finite_null_probability_normalized"]
        and claims_1_2["asymptotic_reference_convergence"]
        and abs(claims_1_2["finite_null_cells"]["5000"]["mean"] - 2 / 3) < 1e-4
    )
    claim_3 = (
        full["configuration"] == FULL
        and full["exact_source_scale"] is True
        and full["device"] == "cuda"
        and aggregate["seed_file_count"] == 5
        and aggregate["all_zero_shift_best"] is True
        and aggregate["all_absolute_shift_monotone"] is True
        and aggregate["three_sem_separation"] is True
        and bayes["finite_sample_arithmetic_ratio_is_unstable"] is True
        and bayes["mean_log_ratio_ranks_zero_shift"] is True
    )
    verified = sum((claim_1, claim_2, claim_3))
    payload = {
        "paper": "ra2t1V4nml",
        "claim_1_conditional_match_score": claim_1,
        "claim_2_null_and_independence_quantities": claim_2,
        "claim_3_direct_model_comparison": claim_3,
        "verified_claims": verified,
        "earned_points": 2 * verified,
        "all_claims_complete": verified == 3,
        "tests_passed": tests_passed,
        "publication_gate_passed": verified == 3 and tests_passed,
        "full_gmm_readback_sha256": hashlib.sha256(full_path.read_bytes()).hexdigest(),
        "full_gmm_job_url": "https://huggingface.co/jobs/DineshAI/6a5de5e0d216bd6f3a2031f7",
        "multi_seed_nearest_zero_shift_margin": aggregate["mean_nearest_shift_margin"],
        "multi_seed_nearest_zero_shift_sem": aggregate["sem_nearest_shift_margin"],
        "multi_seed_nearest_zero_shift_sigma": (
            aggregate["mean_nearest_shift_margin"] / aggregate["sem_nearest_shift_margin"]
        ),
        "bayes_factor_disclosure": (
            "The paper's stated finite arithmetic density-ratio average is an unstable importance-sampling estimator; "
            "the exact-scale MIRA score and an independent log-density comparison rank the true model without it."
        ),
    }
    (OUT / "claim_verification.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["all_claims_complete"]:
        raise SystemExit("publication gate rejects incomplete claims")


if __name__ == "__main__":
    main()
