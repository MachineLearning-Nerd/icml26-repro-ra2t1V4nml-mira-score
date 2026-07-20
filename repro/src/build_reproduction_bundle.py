"""Create a compact, deterministic publication bundle without environments or secrets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
INCLUDE = ("README.md", "STATUS.md", "poster_embed.html", "docs", "repro", "upstream", "outputs")
EXCLUDE_NAMES = {"reproduction_bundle.zip", "reproduction_bundle_manifest.json"}


def wanted(path: Path) -> bool:
    return (
        path.is_file()
        and path.name not in EXCLUDE_NAMES
        and "__pycache__" not in path.parts
        and "local" not in path.parts
    )


def main() -> None:
    destination = OUT / "reproduction_bundle.zip"
    paths: list[Path] = []
    for name in INCLUDE:
        candidate = ROOT / name
        if candidate.is_file() and wanted(candidate):
            paths.append(candidate)
        elif candidate.is_dir():
            paths.extend(sorted(path for path in candidate.rglob("*") if wanted(path)))
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT))
    payload = {
        "file_count": len(paths),
        "size_bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    (OUT / "reproduction_bundle_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
