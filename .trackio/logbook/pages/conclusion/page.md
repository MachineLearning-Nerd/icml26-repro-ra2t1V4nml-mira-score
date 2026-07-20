# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_a5f0d382da4a", "created_at": "2026-07-20T09:10:22+00:00", "title": "Reproduction bundle"}
-->
The complete reproducibility bundle is registered below as a local Trackio
artifact (`67` curated files; `8,177,932` bytes; SHA-256
`76e945c3d16e474fbca639de8548b7bf7e3009d5c05c5b32e98854dc94afe188`). It
contains the source pins, vendored official code, clean-room checks, full-scale
Job readbacks, tests, and claim verifier—excluding environments, credentials,
and generated cache files. Rebuild it from a checkout with
`python repro/src/build_reproduction_bundle.py`; run the gate with
`python repro/src/run_tests.py && python repro/src/verify_claims.py && python repro/src/build_evidence_bundle.py`.


---
<!-- trackio-cell
{"type": "dashboard", "id": "cell_9f47b6d6599e", "created_at": "2026-07-20T09:31:44+00:00", "title": "Dashboard: repro-mira-score", "dashboard_project": "repro-mira-score"}
-->
**🎯 Trackio dashboard** `repro-mira-score`

trackio-local-dashboard://repro-mira-score


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_df4f1f25a8a8", "created_at": "2026-07-20T09:31:44+00:00", "title": "Final reproduction bundle", "artifact": "repro-mira-score/repro-bundle:v1", "artifact_type": "dataset"}
-->
**📦 Artifact** `repro-mira-score/repro-bundle:v1` · dataset

trackio-artifact://repro-mira-score/repro-bundle:v1
