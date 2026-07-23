"""Validate the additive, unpublished, text-only release candidate."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "release"
CANDIDATE = RELEASE / "space_candidate"
REPORT = ROOT / "reports" / "mira-reproduction" / "report.md"
CLAIM5_REPORT = ROOT / "reports" / "claim5-three-approach" / "report.md"
NOTEBOOK = ROOT / "notebooks" / "mira_reproduction.py"
TEXT_EXTENSIONS = {".md", ".json", ".csv", ".svg"}
SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(node: dict) -> list[dict]:
    rows = [node]
    for child in node.get("children", []):
        rows.extend(flatten(child))
    return rows


def main() -> None:
    protected = json.loads((RELEASE / "judged_space_manifest.json").read_text())
    receipt_path = RELEASE / "publication_receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
    publication_performed = (
        receipt is not None
        and receipt["space_id"] == protected["space_id"]
        and receipt["uploaded_file_count"] == 49
        and receipt["readback_manifest_failures"] == 0
    )
    old_rows = protected["files"]
    old_paths = [row["path"] for row in old_rows]
    if len(old_rows) != 20 or len(set(old_paths)) != 20:
        raise SystemExit("protected judged manifest must contain 20 unique relative paths")
    missing_old_paths = [path for path in old_paths if not (CANDIDATE / path).is_file()]
    identical = [
        row["path"] for row in old_rows if (CANDIDATE / row["path"]).is_file()
        and sha256(CANDIDATE / row["path"]) == row["sha256"]
    ]
    changed = sorted(set(old_paths) - set(identical) - set(missing_old_paths))
    if missing_old_paths or changed != ["logbook.json"] or len(identical) != 19:
        raise SystemExit(
            f"judged tree preservation failed: missing={missing_old_paths}, changed={changed}, "
            f"identical={len(identical)}"
        )

    logbook = json.loads((CANDIDATE / "logbook.json").read_text())
    nodes = flatten(logbook["root"])
    slugs = [node["slug"] for node in nodes]
    logbook_files = [node["file"] for node in nodes]
    logbook_valid = (
        logbook["schema_version"] == 1
        and logbook["space_id"] == "DineshAI/ra2t1V4nml"
        and len(slugs) == len(set(slugs))
        and all((CANDIDATE / path).is_file() for path in logbook_files)
    )
    if not logbook_valid:
        raise SystemExit("candidate logbook schema, slugs, or page references are invalid")

    expected_verdicts = {4: "VERIFIED", 5: "BLOCKED", 6: "VERIFIED"}
    verdicts = {
        claim_id: json.loads(
            (CANDIDATE / f"evidence/claim_{claim_id}/verifier_output.json").read_text()
        )["verdict"]
        for claim_id in expected_verdicts
    }
    if verdicts != expected_verdicts:
        raise SystemExit(f"candidate evidence verdict mismatch: {verdicts}")

    image_references = re.findall(r"!\[[^\]]*\]\((images/[^)]+)\)", REPORT.read_text())
    if len(image_references) != 5 or len(set(image_references)) != 5:
        raise SystemExit(f"expected five distinct report figures, found {image_references}")
    for reference in image_references:
        image_path = REPORT.parent / reference
        root = ET.parse(image_path).getroot()
        if not root.tag.endswith("svg") or not root.get("viewBox"):
            raise SystemExit(f"invalid SVG report figure: {image_path}")
        candidate_image = CANDIDATE / "reports/mira-reproduction" / reference
        if not candidate_image.is_file() or sha256(candidate_image) != sha256(image_path):
            raise SystemExit(f"Space report figure does not match repository figure: {reference}")
    if sha256(CANDIDATE / "reports/mira-reproduction/report.md") != sha256(REPORT):
        raise SystemExit("Space report does not match repository report")
    claim5_references = re.findall(
        r"!\[[^\]]*\]\((images/[^)]+)\)", CLAIM5_REPORT.read_text()
    )
    if claim5_references != ["images/posterior_diagnostics.svg"]:
        raise SystemExit(
            f"expected one Claim 5 diagnostic figure, found {claim5_references}"
        )
    claim5_image = CLAIM5_REPORT.parent / claim5_references[0]
    claim5_root = ET.parse(claim5_image).getroot()
    if not claim5_root.tag.endswith("svg") or not claim5_root.get("viewBox"):
        raise SystemExit(f"invalid Claim 5 SVG figure: {claim5_image}")
    candidate_claim5 = CANDIDATE / "reports/claim5-three-approach"
    if sha256(candidate_claim5 / "report.md") != sha256(CLAIM5_REPORT):
        raise SystemExit("Space Claim 5 report does not match repository report")
    if sha256(candidate_claim5 / claim5_references[0]) != sha256(claim5_image):
        raise SystemExit("Space Claim 5 figure does not match repository figure")

    marimo = subprocess.run(
        ["marimo", "check", "--strict", str(NOTEBOOK)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if marimo.returncode:
        raise SystemExit(f"marimo check failed:\n{marimo.stdout}\n{marimo.stderr}")

    allowlist = [
        line.strip()
        for line in (RELEASE / "text_upload_allowlist.txt").read_text().splitlines()
        if line.strip()
    ]
    if allowlist != sorted(allowlist) or len(allowlist) != len(set(allowlist)):
        raise SystemExit("text upload allowlist must be sorted and unique")
    manifest_rows = []
    secret_findings = []
    for relative in allowlist:
        path = CANDIDATE / relative
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            raise SystemExit(f"allowlist path is missing or not text-only: {relative}")
        text = path.read_text(encoding="utf-8")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                secret_findings.append({"path": relative, "pattern": pattern.pattern})
        manifest_rows.append(
            {
                "path": relative,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    if secret_findings:
        raise SystemExit(f"secret-like content found in upload candidate: {secret_findings}")

    manifest = {
        "target_space_id": "DineshAI/ra2t1V4nml",
        "judged_revision": protected["revision"],
        "publication_approved": True,
        "publication_performed": publication_performed,
        "text_only": True,
        "file_count": len(manifest_rows),
        "files": manifest_rows,
    }
    if publication_performed:
        manifest["published_revision"] = receipt["published_revision"]
    (RELEASE / "text_upload_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    subset = {
        "space_id": protected["space_id"],
        "judged_revision": protected["revision"],
        "old_file_count": len(old_rows),
        "candidate_file_count": len(
            [path for path in CANDIDATE.rglob("*") if path.is_file()]
        ),
        "old_file_set_is_subset": not missing_old_paths,
        "byte_identical_old_files": len(identical),
        "additively_changed_old_files": changed,
        "changed_file_reason": {
            "logbook.json": "central manifest extended to index new additive pages"
        },
        "all_prior_page_files_byte_identical": all(
            path in identical for path in old_paths if path.startswith("pages/")
        ),
    }
    (RELEASE / "subset_check.json").write_text(
        json.dumps(subset, indent=2) + "\n", encoding="utf-8"
    )
    payload = {
        "release_candidate_ready": True,
        "publication_approved": True,
        "publication_performed": publication_performed,
        "score_increase_claimed": False,
        "target_space_id": protected["space_id"],
        "old_file_set_is_subset": subset["old_file_set_is_subset"],
        "old_page_files_byte_identical": subset["all_prior_page_files_byte_identical"],
        "candidate_logbook_valid": logbook_valid,
        "report_figures_valid": len(image_references),
        "claim5_report_figures_valid": len(claim5_references),
        "marimo_check_passed": True,
        "text_upload_file_count": len(manifest_rows),
        "secret_findings": secret_findings,
        "verdicts": verdicts,
    }
    if publication_performed:
        payload["published_revision"] = receipt["published_revision"]
    (RELEASE / "release_gate.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
