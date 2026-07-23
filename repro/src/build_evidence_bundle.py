"""Build the authoritative six-claim release-candidate evidence index."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
OPENRESEARCH = ROOT / ".openresearch" / "artifacts"
REQUIRED = (
    "source_audit.json",
    "claims_1_2.json",
    "bayes_factor_audit.json",
    "full_gmm_aggregate.json",
    "test_results.json",
    "claim_verification.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    legacy = json.loads((OUT / "claim_verification.json").read_text(encoding="utf-8"))
    campaign = json.loads((OPENRESEARCH / "campaign_verdicts.json").read_text(encoding="utf-8"))
    expected = {1: "VERIFIED", 2: "VERIFIED", 3: "VERIFIED", 4: "VERIFIED", 5: "BLOCKED", 6: "VERIFIED"}
    observed = {int(row["claim_id"]): row["verdict"] for row in campaign["claims"]}
    if observed != expected or not campaign["all_evaluated_claim_checks_passed"]:
        raise SystemExit(f"cannot bundle non-terminal or unexpected campaign verdicts: {observed}")
    full_root = OUT / "hub_readback" / "full_gmm"
    full_files = sorted(full_root.glob("*.json"))
    if len(full_files) != 5:
        raise SystemExit(f"expected five Hub-readback GMM files, found {len(full_files)}")
    bundle = {
        "paper": "ra2t1V4nml",
        "gate": "RELEASE_CANDIDATE_READY_FOR_EXPLICIT_APPROVAL",
        "claim_count": 6,
        "verdict_counts": {"VERIFIED": 5, "BLOCKED": 1, "FALSIFIED": 0},
        "baseline_judged_score": "7/12",
        "projected_score": None,
        "score_increase_claimed": False,
        "verdicts": observed,
        "artifacts": {name: sha256(OUT / name) for name in REQUIRED},
        "campaign_verdicts_sha256": sha256(OPENRESEARCH / "campaign_verdicts.json"),
        "claim_artifacts": {
            str(claim_id): {
                str(path.relative_to(ROOT)): sha256(path)
                for path in sorted((OPENRESEARCH / f"claim_{claim_id}").glob("*"))
                if path.is_file()
            }
            for claim_id in (4, 5, 6)
        },
        "full_gmm_readbacks": {path.name: sha256(path) for path in full_files},
        "jobs": [
            "https://huggingface.co/jobs/DineshAI/6a5de5e0d216bd6f3a2031f7",
            "https://huggingface.co/jobs/DineshAI/6a5de7a9bee6ee1cf4ed22e1",
            "https://huggingface.co/jobs/DineshAI/6a5de7a8bee6ee1cf4ed22dd",
            "https://huggingface.co/jobs/DineshAI/6a5de7a9bee6ee1cf4ed22e2",
            "https://huggingface.co/jobs/DineshAI/6a5de7a8bee6ee1cf4ed22df",
        ],
    }
    encoded = json.dumps(bundle, indent=2, sort_keys=True) + "\n"
    (OUT / "evidence_bundle.json").write_text(encoded, encoding="utf-8")
    marker = {
        "gate": "RELEASE_CANDIDATE_READY_FOR_EXPLICIT_APPROVAL",
        "paper": "ra2t1V4nml",
        "evidence_bundle_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "campaign_verdicts_sha256": bundle["campaign_verdicts_sha256"],
        "claims_terminal": True,
        "verified_claims": 5,
        "blocked_claims": 1,
        "publication_approved": False,
        "publication_performed": False,
        "score_increase_claimed": False,
    }
    (OUT / "RELEASE_CANDIDATE_READY.json").write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8"
    )
    legacy_marker = {
        "gate": "LEGACY_THREE_CLAIM_GATE_SUPERSEDED",
        "superseded_by": "outputs/RELEASE_CANDIDATE_READY.json",
        "publication_gate_passed": False,
    }
    (OUT / "PUBLICATION_GATE_PASSED.json").write_text(
        json.dumps(legacy_marker, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
