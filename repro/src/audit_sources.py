"""Fail-closed source/protocol audit for the three MIRA claims."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    pins = json.loads((ROOT / "docs" / "source_pins.json").read_text(encoding="utf-8"))
    archive = ROOT / "docs" / "arxiv_source.tar"
    official = ROOT / "upstream" / "src" / "mira_score" / "mira.py"
    with tarfile.open(archive) as bundle:
        tex_member = next(member for member in bundle.getmembers() if member.name.endswith("icml_2026_main_conference.tex"))
        tex = bundle.extractfile(tex_member).read().decode("utf-8")
    code = official.read_text(encoding="utf-8")
    source_checks = {
        "primary_source_hash_matches": digest(archive) == pins["arxiv_source_sha256"],
        "official_statistic": "(counts + 1) / (N + 2)" in code and "(N - counts + 1) / (N + 2)" in code,
        "official_calibration": "calib = prob / max_val" in code,
        "paper_null_beta_law": "Beta$(2,1)$" in tex and "\\tfrac23" in tex and "\\tfrac{1}{18}" in tex,
        "paper_full_gmm_dimensions": "100$ dimensions" in tex,
        "paper_full_gmm_components": "20$ mixture components" in tex,
        "paper_full_gmm_samples": "$N=5\\,000$ samples" in tex and "$L=5\\,000$ true samples" in tex,
        "paper_full_gmm_regions": "100$ regions per true sample" in tex,
        "paper_bayes_factor_claim": "Bayesian model comparison" in tex and "bypassing the challenging evidence computation" in tex,
    }
    payload = {
        "source_pins": pins,
        "source_checks": source_checks,
        "all_static_checks_pass": all(source_checks.values()),
        "undocumented_gmm_details": ["seed", "component means", "component covariance", "released experiment script"],
    }
    output = ROOT / "outputs" / "source_audit.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["all_static_checks_pass"]:
        raise SystemExit("source audit failed")


if __name__ == "__main__":
    main()
