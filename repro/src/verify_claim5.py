"""Fail closed unless Claim 5's BLOCKED verdict is fully evidenced."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".openresearch" / "artifacts" / "claim_5"


def load(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def main() -> None:
    contract = load("claim_contract.json")
    audit = load("raw_release_audit.json")
    manifest = load("release_tree_manifest.json")
    independent = load("independent_checker_output.json")
    negative = load("negative_control_output.json")
    environment = load("environment.json")
    required_static = ("source_audit.md", "method.md", "limitations_and_deviations.md")
    assertions = {
        "contract_allows_only_terminal_verdicts": set(contract["allowed_verdicts"])
        == {"VERIFIED", "FALSIFIED", "BLOCKED"},
        "author_repositories_pinned": [
            row["commit"] for row in manifest["repositories"]
        ]
        == [
            "3bc229222cbcf72bd470267175d9a6dff6689ce0",
            "c57487198ac30711783b78ac2af6a76758544483",
        ],
        "complete_recursive_trees": all(
            row["truncated"] is False for row in manifest["repositories"]
        ),
        "paper_scale_audited": audit["paper_protocol"]["L"] == 100
        and audit["paper_protocol"]["N"] == 20_000
        and audit["paper_protocol"]["dimensions"] == 13,
        "paper_scores_recovered": audit["paper_scores"]["values_match"],
        "essential_inputs_incomplete": audit["essential_inputs_complete"] is False,
        "at_least_four_independent_blockers": len(audit["missing_requirement_ids"]) >= 4,
        "independent_absence_checker_passed": independent["exact_inputs_absent"]
        and independent["aggregate_tarp_is_not_raw_posterior_evidence"],
        "negative_control_sensitive": negative["negative_control_passed"],
        "cpu_only": environment["device"] == "cpu",
        "pinned_environment": bool(environment["git_sha"]) and bool(environment["uv_lock_sha256"]),
        "static_audit_files_present": all((ARTIFACTS / name).is_file() for name in required_static),
    }
    blocked_is_evidenced = all(assertions.values())
    payload = {
        "claim_id": 5,
        "verdict": "BLOCKED" if blocked_is_evidenced else "FALSIFIED",
        "blocked_requirements_verified": blocked_is_evidenced,
        "assertions": assertions,
        "blockers": audit["missing_requirement_ids"],
        "interpretation": (
            "The released aggregate numbers support no independent rerun. "
            "BLOCKED is not evidence that the scientific claim is false."
        ),
    }
    (ARTIFACTS / "verifier_output.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    eval_text = f"""# Claim 5 evaluation

Verdict: **{payload['verdict']}**

The paper reports the full `L=100`, `N=20,000`, 13-dimensional experiment and
the pinned plotting notebook contains aggregate scores
`0.6320, 0.5788, 0.5394, 0.5223`. Those numbers rank EPL+3 Sérsic first, but
they are hardcoded presentation values, not independently regenerable data.

Recursive, non-truncated Git tree audits at both pinned author commits found no
physical-model truth parameter set, no four posterior sample sets, no MALA
implementation or tuning, and no executable EPL/SIE–Sérsic forward-model
construction. The paper additionally omits values for parameters it calls
"held constant," dependency versions, and all relevant seeds. An aggregated
TARP `.npz` is present but cannot regenerate MIRA.

Accordingly, this campaign neither verifies nor falsifies the ranking. It marks
the exact claim **BLOCKED** and does not substitute the earlier GMM proxy.
"""
    (ARTIFACTS / "EVAL.md").write_text(eval_text, encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not blocked_is_evidenced:
        raise SystemExit("Claim 5 terminal BLOCKED verdict was not fully evidenced")


if __name__ == "__main__":
    main()
