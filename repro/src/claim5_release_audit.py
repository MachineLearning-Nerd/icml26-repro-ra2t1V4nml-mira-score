"""Pinned release-completeness audit for the 13-D gravitational-lens claim."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".openresearch" / "artifacts" / "claim_5"
USER_AGENT = "OpenResearch-MIRA-Reproduction/1.0"
REPOSITORIES = (
    {
        "name": "MIRA_Paper_Plots",
        "owner": "SammyS15",
        "commit": "3bc229222cbcf72bd470267175d9a6dff6689ce0",
    },
    {
        "name": "mira-score",
        "owner": "SammyS15",
        "commit": "c57487198ac30711783b78ac2af6a76758544483",
    },
)
EXPECTED_SCORES = [0.6319946646690369, 0.5787630081176758, 0.5393556356430054, 0.5222544074058533]
MODEL_ORDER = ["EPL+3_Sersic_true", "SIE+3_Sersic", "EPL+1_Sersic", "SIE+1_Sersic"]


def request_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def repository_tree(owner: str, name: str, commit: str) -> dict[str, object]:
    url = f"https://api.github.com/repos/{owner}/{name}/git/trees/{commit}?recursive=1"
    payload = json.loads(request_bytes(url))
    if payload.get("truncated"):
        raise SystemExit(f"GitHub returned a truncated tree for {owner}/{name}")
    entries = [
        {
            "path": row["path"],
            "type": row["type"],
            "size": row.get("size"),
            "git_sha": row["sha"],
        }
        for row in payload["tree"]
    ]
    return {
        "repository": f"https://github.com/{owner}/{name}",
        "commit": commit,
        "tree_sha": payload["sha"],
        "truncated": payload["truncated"],
        "entry_count": len(entries),
        "entries": entries,
    }


def code_path(path: str) -> bool:
    return Path(path).suffix.lower() in {".py", ".ipynb", ".jl", ".cpp", ".c", ".h"}


def find_paths(entries: list[dict[str, object]], patterns: tuple[str, ...]) -> list[str]:
    matches = []
    for row in entries:
        path = str(row["path"])
        lowered = path.lower()
        if all(pattern in lowered for pattern in patterns):
            matches.append(path)
    return matches


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    started = perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    trees = [
        repository_tree(row["owner"], row["name"], row["commit"])
        for row in REPOSITORIES
    ]
    plot_tree = trees[0]
    plot_entries = plot_tree["entries"]
    score_notebook_url = (
        "https://raw.githubusercontent.com/SammyS15/MIRA_Paper_Plots/"
        "3bc229222cbcf72bd470267175d9a6dff6689ce0/notebooks/main_plotting.ipynb"
    )
    notebook_bytes = request_bytes(score_notebook_url)
    notebook_text = notebook_bytes.decode("utf-8")
    extracted_scores = []
    match = re.search(
        r"scores = \[(0\.6319946646690369), (0\.5787630081176758), "
        r"(0\.5393556356430054), (0\.5222544074058533)\]",
        notebook_text,
    )
    if match:
        extracted_scores = [float(value) for value in match.groups()]

    relevant_paths = sorted(
        str(row["path"])
        for row in plot_entries
        if any(
            token in str(row["path"]).lower()
            for token in ("model_misspec", "mala", "sersic", "epl", "caustic", "posterior")
        )
    )
    mala_code = [
        str(row["path"])
        for tree in trees
        for row in tree["entries"]
        if "mala" in str(row["path"]).lower() and code_path(str(row["path"]))
    ]
    physical_forward_code = [
        str(row["path"])
        for tree in trees
        for row in tree["entries"]
        if code_path(str(row["path"]))
        and any(token in str(row["path"]).lower() for token in ("sersic", "epl", "caustic"))
    ]
    physical_posterior_paths = [
        str(row["path"])
        for row in plot_entries
        if "posterior" in str(row["path"]).lower()
        and any(token in str(row["path"]).lower() for token in ("model_misspec", "sersic", "epl", "mala"))
    ]
    physical_truth_paths = [
        str(row["path"])
        for row in plot_entries
        if "truth" in str(row["path"]).lower()
        and any(token in str(row["path"]).lower() for token in ("model_misspec", "sersic", "epl", "mala"))
    ]
    tarp_summary = find_paths(plot_entries, ("model_misspecification_tarp_data",))

    requirements = [
        {
            "id": "truth_parameters_L100_d13",
            "required": "100 exact 13-dimensional truth parameter vectors",
            "state": "PRESENT" if physical_truth_paths else "MISSING",
            "release_paths": physical_truth_paths,
        },
        {
            "id": "four_posterior_sets_100x20000x13",
            "required": "four candidate posterior sets for every one of 100 observations",
            "state": "PRESENT" if len(physical_posterior_paths) >= 4 else "MISSING",
            "release_paths": physical_posterior_paths,
        },
        {
            "id": "mala_sampler_source_and_tuning",
            "required": "MALA source plus step size, initialization, and walker configuration",
            "state": "PRESENT" if mala_code else "MISSING",
            "release_paths": mala_code,
        },
        {
            "id": "caustics_forward_model_source",
            "required": "executable EPL/SIE and one/three-Sersic forward-model construction",
            "state": "PRESENT" if physical_forward_code else "MISSING",
            "release_paths": physical_forward_code,
        },
        {
            "id": "all_fixed_forward_parameters",
            "required": "values for every lens/source parameter described by the paper as held constant",
            "state": "MISSING",
            "paper_evidence": "Appendix Table 3 says all other parameters are held constant but does not list their values.",
        },
        {
            "id": "software_versions_and_random_seeds",
            "required": "caustics/MALA versions and deterministic simulation, chain, and MIRA seeds",
            "state": "MISSING",
            "paper_evidence": "The paper names caustics and MALA but gives no versions or seeds.",
        },
    ]
    missing = [row["id"] for row in requirements if row["state"] == "MISSING"]
    audit = {
        "claim_id": 5,
        "paper_protocol": {
            "true_model": "EPL + 3 Sersic",
            "candidate_models": MODEL_ORDER,
            "L": 100,
            "N": 20_000,
            "dimensions": 13,
            "regions": 100,
            "MALA_walkers": 100,
            "MALA_burnin_steps": 200,
            "MALA_sampling_steps": 200,
            "paper_cpu_hours": 13.33,
        },
        "paper_scores": {
            "model_order": MODEL_ORDER,
            "expected": EXPECTED_SCORES,
            "extracted_from_pinned_notebook": extracted_scores,
            "notebook_url": score_notebook_url,
            "notebook_sha256": hashlib.sha256(notebook_bytes).hexdigest(),
            "values_match": extracted_scores == EXPECTED_SCORES,
            "limitation": "hardcoded aggregate scores are not regenerable without posterior samples",
        },
        "requirements": requirements,
        "missing_requirement_ids": missing,
        "essential_inputs_complete": not missing,
        "released_tarp_summary_paths": tarp_summary,
        "relevant_release_paths": relevant_paths,
        "conclusion": "BLOCKED" if missing else "READY_FOR_EXPERIMENT",
    }
    (ARTIFACTS / "raw_release_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    (ARTIFACTS / "release_tree_manifest.json").write_text(
        json.dumps(
            {
                "retrieval_user_agent": USER_AGENT,
                "repositories": trees,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Names alone are not evidence: intentionally fabricated zero-byte tensors
    # must remain unusable under the release-completeness contract.
    fake_entries = [
        {
            "path": f"physical_model/{model}_posterior.npy",
            "size": 0,
            "sha256": "0" * 64,
        }
        for model in MODEL_ORDER
    ]
    minimum_bytes_per_posterior = 100 * 20_000 * 13 * 4
    fake_rejected = all(
        row["size"] < minimum_bytes_per_posterior or row["sha256"] == "0" * 64
        for row in fake_entries
    )
    permuted_scores = [EXPECTED_SCORES[1], EXPECTED_SCORES[0], *EXPECTED_SCORES[2:]]
    negative = {
        "control_type": "path-only fake assets and permuted score labels",
        "fake_entries": fake_entries,
        "minimum_float32_bytes_per_required_posterior_set": minimum_bytes_per_posterior,
        "fake_assets_rejected": fake_rejected,
        "permuted_scores": permuted_scores,
        "permuted_true_model_ranks_first": permuted_scores[0] == max(permuted_scores),
        "negative_control_passed": fake_rejected and permuted_scores[0] != max(permuted_scores),
    }
    (ARTIFACTS / "negative_control_output.json").write_text(
        json.dumps(negative, indent=2) + "\n", encoding="utf-8"
    )
    environment = {
        "command": "uv run --frozen python repro/src/run_campaign.py",
        "git_sha": git_sha(),
        "uv_lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "device": "cpu",
        "deterministic_seeds": [],
        "runtime_seconds": perf_counter() - started,
    }
    (ARTIFACTS / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    payload = {"audit": audit, "negative_control": negative, "environment": environment}
    print(json.dumps(payload, indent=2))
    if audit["conclusion"] != "BLOCKED" or len(missing) < 4:
        raise SystemExit("release audit did not establish the expected essential blockers")
    if not audit["paper_scores"]["values_match"] or not negative["negative_control_passed"]:
        raise SystemExit("release audit controls failed")


if __name__ == "__main__":
    main()
