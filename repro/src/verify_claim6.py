"""Fail-closed verifier for the exact released-data Claim 6 contract."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".openresearch" / "artifacts" / "claim_6"
EXPECTED_HASHES = {
    "true.pt": "74a9b24d746a5dccf2af77fbc72bddd544760d494cbe919e69d704fbb62a5930",
    "posterior0.pt": "71009759c762452941c375b14b8e701c375c276fb920ecc55304770c01e1370d",
    "posterior1.pt": "c874b78bbabb3a2c5b057d602ea4e75b41546c14499c3a3d152d9d8b47ed8661",
    "posterior2.pt": "4eec7a23e4677fb8e8d6c2cee32950715d4788ba24fa2dc9151ef62fcf270698",
    "posterior3.pt": "a819b262c52fe38e19ea04594dee33ea670943d80285a974b85316e90875cbca",
}


def load(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def main() -> None:
    contract = load("claim_contract.json")
    summary = load("raw_summary.json")
    manifest = load("data_manifest.json")
    independent = load("independent_checker_output.json")
    parity = load("implementation_parity.json")
    negative = load("negative_control_output.json")
    environment = load("environment.json")
    hashes = {row["path"]: row["sha256"] for row in manifest["files"]}
    paired = summary["bootstrap"]["true_minus_misspecified"]
    assertions = {
        "contract_requires_verified": contract["machine_checkable_contract"]["required_verdict"] == "VERIFIED",
        "exact_author_commit": manifest["author_commit"] == "3bc229222cbcf72bd470267175d9a6dff6689ce0",
        "all_protected_hashes_match": hashes == EXPECTED_HASHES,
        "paper_scale_L_16": summary["protocol"]["L"] == 16,
        "paper_scale_N_64": summary["protocol"]["N"] == 64,
        "paper_scale_dimensions_12288": summary["protocol"]["dimensions"] == 12_288,
        "paper_scale_regions_100": summary["protocol"]["regions"] == 100,
        "released_default_norm_false": summary["protocol"]["norm"] is False,
        "model_order_matches_paper": summary["observed_order"] == summary["paper_order"],
        "independent_checker_passed": independent["model_order_matches_paper"],
        "full_scale_released_implementation_parity": parity["within_one_count_quantum"],
        "all_misspecification_differences_positive_at_95pct": all(
            row["ci_excludes_zero"] for row in paired.values()
        ),
        "negative_control_sensitive": negative["negative_control_passed"],
        "cpu_only": environment["device"] == "cpu",
        "pinned_environment": bool(environment["git_sha"]) and bool(environment["uv_lock_sha256"]),
        "static_evidence_present": all(
            (ARTIFACTS / name).is_file()
            for name in (
                "source_audit.md",
                "method.md",
                "limitations_and_deviations.md",
            )
        ),
    }
    passed = all(assertions.values())
    payload = {
        "claim_id": 6,
        "verdict": "VERIFIED" if passed else "BLOCKED",
        "assertions": assertions,
        "evidence_standard": (
            "exact released full-dimensional tensors, released score semantics, paired two-axis "
            "bootstrap, independent CSV aggregation, and broken-pairing negative control"
        ),
    }
    (ARTIFACTS / "verifier_output.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    observed = [row["observed_score"] for row in summary["models"]]
    eval_text = f"""# Claim 6 evaluation

Verdict: **{payload['verdict']}**

The exact author-released tensors have `L=16`, `N=64`, and 12,288 dimensions
(`64×64×3`). With 100 regions, the observed model scores were
`{", ".join(f"{value:.4f}" for value in observed)}` in the paper's expected
order: correct prior/noise, noise-only misspecification, prior-only
misspecification, then both misspecified. Every paired 95% bootstrap interval
for the true-model advantage excludes zero. The independent CSV checker
recomputed the ranking without importing torch or the scoring implementation.

The paper's hardcoded plot scores are
`0.6442, 0.5783, 0.5298, 0.5056`; the largest absolute numerical difference is
`{independent['maximum_absolute_paper_delta']:.4f}`. This numerical reference is
reported as **{independent['numeric_reference_alignment']}**, not silently
treated as aligned. The exact source claim is detection and ranking, while the
paper omits the random seed and MIRA center configuration required for exact
number parity. A full-scale one-region check against the released scorer is
within one discrete count quantum.

The negative control rolls the fiducial truth-to-observation pairing by one,
deliberately breaking conditional correspondence; it must lower the intact
true-model score. This is a published-data evaluation, not a Gaussian or
low-dimensional proxy. Model training was not repeated because the released
posterior samples are the paper's evaluation inputs and training requires GPU
hardware outside the authorized CPU-only campaign.
"""
    (ARTIFACTS / "EVAL.md").write_text(eval_text, encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit("Claim 6 verification failed closed")


if __name__ == "__main__":
    main()
