"""Register the curated local bundle as a Trackio dataset artifact."""

from __future__ import annotations

from pathlib import Path

import trackio


ROOT = Path(__file__).resolve().parents[2]
run = trackio.init(project="repro-mira-score", name="full-publication-gate", resume="allow")
artifact = trackio.log_artifact(ROOT / "outputs" / "reproduction_bundle.zip", name="repro-bundle", type="dataset")
print(f"artifact={artifact.project}/{artifact.name}:{artifact.version}")
trackio.finish()
