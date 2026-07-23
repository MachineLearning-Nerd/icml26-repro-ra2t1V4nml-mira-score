"""Publish the validated text-only Space candidate as one additive API commit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"
CANDIDATE = RELEASE / "space_candidate"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((RELEASE / "text_upload_manifest.json").read_text())
    live_check = json.loads((RELEASE / "live_head_subset_check.json").read_text())
    if (
        manifest["target_space_id"] != "DineshAI/ra2t1V4nml"
        or not manifest["publication_approved"]
        or manifest["publication_performed"]
        or not manifest["text_only"]
    ):
        raise SystemExit("publication manifest is not approved and pre-publication")

    api = HfApi()
    current = api.repo_info(
        repo_id=manifest["target_space_id"], repo_type="space"
    ).sha
    expected_parent = live_check["observed_live_revision"]
    if current != expected_parent:
        raise SystemExit(
            f"Space head changed after audit: expected {expected_parent}, found {current}"
        )

    operations = []
    for row in manifest["files"]:
        local_path = CANDIDATE / row["path"]
        if sha256(local_path) != row["sha256"]:
            raise SystemExit(f"candidate hash changed after validation: {row['path']}")
        operations.append(
            CommitOperationAdd(
                path_in_repo=row["path"],
                path_or_fileobj=local_path,
            )
        )

    result = api.create_commit(
        repo_id=manifest["target_space_id"],
        repo_type="space",
        revision="main",
        parent_commit=expected_parent,
        operations=operations,
        commit_message="Publish cumulative six-claim CPU audit",
        commit_description=(
            "Claims 1-4 and 6 VERIFIED; Claim 5 BLOCKED after a release audit "
            "and three rejected posterior approaches. Text-only additive "
            "upload; no score increase claimed before live judge."
        ),
    )
    print(
        json.dumps(
            {
                "space_id": manifest["target_space_id"],
                "parent_revision": expected_parent,
                "published_revision": result.oid,
                "commit_url": result.commit_url,
                "uploaded_file_count": len(operations),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
