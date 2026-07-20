"""Hash-address all retained source, local, and full-scale evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
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
    claims = json.loads((OUT / "claim_verification.json").read_text(encoding="utf-8"))
    if not (claims["all_claims_complete"] and claims["earned_points"] == 6 and claims["tests_passed"]):
        raise SystemExit("cannot bundle incomplete or untested claims")
    full_root = OUT / "hub_readback" / "full_gmm"
    full_files = sorted(full_root.glob("*.json"))
    if len(full_files) != 5:
        raise SystemExit(f"expected five Hub-readback GMM files, found {len(full_files)}")
    bundle = {
        "paper": "ra2t1V4nml",
        "gate": "FULL_GATE_READY",
        "claim_count": 3,
        "earned_points": 6,
        "artifacts": {name: sha256(OUT / name) for name in REQUIRED},
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
        "gate": "FULL_GATE_READY",
        "queue_marker": "FULL_GATE_READY: ra2t1V4nml",
        "paper": "ra2t1V4nml",
        "evidence_bundle_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "claim_verification_sha256": bundle["artifacts"]["claim_verification.json"],
        "claims_complete": True,
        "earned_points": 6,
        "tests_passed": True,
        "publication_gate_passed": True,
    }
    (OUT / "PUBLICATION_GATE_PASSED.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
