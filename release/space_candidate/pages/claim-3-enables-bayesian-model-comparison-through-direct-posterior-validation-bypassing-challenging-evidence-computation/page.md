# Claim 3: Enables Bayesian model comparison through direct posterior validation, bypassing challenging evidence computation.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_863e92b43a72", "created_at": "2026-07-20T09:10:22+00:00", "title": "Claim 3: Enables Bayesian model comparison through direct posterior validation, bypassing challenging evidence computation."}
-->
The first retained-scale T4-medium Job completed the source-declared 100-D,
20-component, `L=N=5,000`, 100-region protocol in 226.8 seconds. Its durable
Hub readback has SHA-256 `daee3e6c2678a4cc6914927294baaff41496629431dd896f3a26cfaa606a8b93`:
the zero-shift MIRA score is highest and every absolute-shift comparison is
strictly worse. Four additional predeclared full-scale seeds are required before
the final verdict because the first seed's nearest comparison is only 2.58
pooled standard errors.

An independent 100,000-observation control records an important scope issue:
the paper's stated arithmetic density-ratio average is an unstable
importance-sampling estimator in high dimensions, whereas the log-density
comparison ranks the zero-shift model correctly. The final claim gate therefore
requires direct MIRA ranking plus this disclosure; it does not treat a
finite-sample arithmetic Bayes factor as a reliable success signal.


---
<!-- trackio-cell
{"type": "code", "id": "cell_32f12dcd307d", "created_at": "2026-07-20T09:10:46+00:00", "title": "Run: python gmm_full_protocol.py (exit 0)", "command": ["python", "repro/src/gmm_full_protocol.py", "--smoke", "--device", "cpu", "--output", "outputs/gmm_smoke.json"], "exit_code": 0, "duration_s": 1.687}
-->
````bash
$ python repro/src/gmm_full_protocol.py --smoke --device cpu --output outputs/gmm_smoke.json
````

exit 0 · 1.7s


````python title=gmm_full_protocol.py
"""Streamed, source-scale GMM MIRA/Bayes-factor protocol.

The paper provides the full dimensions/counts but not the GMM seed, covariance,
or experiment script.  This implementation makes those missing choices explicit:
isotropic components, deterministic seed, and a diagonal translation of Euclidean
magnitude ``shift``.  It preserves every declared scale parameter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from time import perf_counter

import torch


FULL = {"dimensions": 100, "components": 20, "truths": 5000, "samples": 5000, "regions": 100}


def sample_gmm(
    count_a: int,
    count_b: int,
    means: torch.Tensor,
    generator: torch.Generator,
    *,
    chunk_a: int,
) -> torch.Tensor:
    """Generate one candidate draw set in chunks without temporary full copies."""
    device = means.device
    dimensions = means.shape[1]
    draws = torch.empty((count_a, count_b, dimensions), device=device)
    for start in range(0, count_a, chunk_a):
        stop = min(start + chunk_a, count_a)
        shape = (stop - start, count_b)
        components = torch.randint(means.shape[0], shape, generator=generator, device=device)
        noise = torch.randn((*shape, dimensions), generator=generator, device=device)
        draws[start:stop] = noise + means[components]
    return draws


def log_gmm_density(points: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    """Exact isotropic-mixture log density via stable log-sum-exp."""
    dimensions = points.shape[-1]
    squared = (points[:, None, :] - means[None, :, :]).square().sum(dim=-1)
    log_component = -0.5 * (dimensions * math.log(2.0 * math.pi) + squared)
    return torch.logsumexp(log_component, dim=1) - math.log(means.shape[0])


def mira_score_streamed(
    truth: torch.Tensor,
    posterior: torch.Tensor,
    regions: int,
    generator: torch.Generator,
    *,
    chunk_a: int,
) -> tuple[float, float]:
    """The released random-ball score, streamed over truths to bound scratch RAM."""
    device = truth.device
    truths, dimensions = truth.shape
    _, samples, posterior_dimensions = posterior.shape
    if posterior_dimensions != dimensions:
        raise ValueError("truth and posterior dimensions differ")
    minimum = truth.min(dim=0, keepdim=True).values
    span = truth.max(dim=0, keepdim=True).values - minimum + 1e-8
    normalized_truth = (truth - minimum) / span
    per_region: list[torch.Tensor] = []
    for _ in range(regions):
        centers = torch.rand((truths, dimensions), generator=generator, device=device)
        references = torch.randint(samples, (truths,), generator=generator, device=device)
        accumulated = torch.zeros((), device=device)
        for start in range(0, truths, chunk_a):
            stop = min(start + chunk_a, truths)
            block = (posterior[start:stop] - minimum) / span
            block_centers = centers[start:stop]
            distances = torch.linalg.vector_norm(block - block_centers[:, None, :], dim=-1)
            rows = torch.arange(stop - start, device=device)
            radii = distances[rows, references[start:stop]]
            keep = torch.ones_like(distances, dtype=torch.bool)
            keep[rows, references[start:stop]] = False
            counts = (distances.masked_fill(~keep, float("inf")) < radii[:, None]).sum(dim=1)
            hit = torch.linalg.vector_norm(normalized_truth[start:stop] - block_centers, dim=-1) <= radii
            # This is the released implementation after its max-value calibration.
            statistic = torch.where(hit, (counts + 1) / samples, (samples - counts) / samples)
            accumulated += statistic.sum()
        per_region.append(accumulated / truths)
    values = torch.stack(per_region)
    return float(values.mean().item()), float(values.std(unbiased=True).item())


def run_protocol(config: dict[str, int], *, device: torch.device, seed: int) -> dict:
    full_scale = config == FULL
    generator = torch.Generator(device=device).manual_seed(seed)
    dimensions = config["dimensions"]
    components = config["components"]
    truths = config["truths"]
    samples = config["samples"]
    regions = config["regions"]
    chunk_a = 10 if full_scale else min(8, truths)
    started = perf_counter()
    base_means = torch.randn((components, dimensions), generator=generator, device=device)
    truth_components = torch.randint(components, (truths,), generator=generator, device=device)
    truth = torch.randn((truths, dimensions), generator=generator, device=device) + base_means[truth_components]
    true_log_density = log_gmm_density(truth, base_means)
    rows = []
    for shift in (-6, -3, 0, 3, 6):
        direction = torch.full((dimensions,), float(shift) / math.sqrt(dimensions), device=device)
        candidate_means = base_means + direction
        posterior = sample_gmm(truths, samples, candidate_means, generator, chunk_a=chunk_a)
        score, score_std = mira_score_streamed(truth, posterior, regions, generator, chunk_a=chunk_a)
        candidate_log_density = log_gmm_density(truth, candidate_means)
        log_bf = candidate_log_density - true_log_density
        rows.append(
            {
                "shift": shift,
                "mira": score,
                "mira_region_std": score_std,
                "mean_bayes_factor": float(torch.exp(log_bf).mean().item()),
                "log_mean_bayes_factor": float(torch.logsumexp(log_bf, 0).item() - math.log(truths)),
            }
        )
        del posterior
        if device.type == "cuda":
            torch.cuda.empty_cache()
    ordered_mira = [row["mira"] for row in rows]
    payload = {
        "configuration": config,
        "exact_source_scale": full_scale,
        "construction": {
            "component_covariance": "identity",
            "mean_translation": "shift / sqrt(dimensions) in every coordinate",
            "seed": seed,
            "source_disclosure": "paper source gives dimensions/counts but no GMM construction or script",
        },
        "device": str(device),
        "rows": rows,
        "zero_shift_best": rows[2]["mira"] > max(rows[0]["mira"], rows[1]["mira"], rows[3]["mira"], rows[4]["mira"]),
        "absolute_shift_monotone": ordered_mira[2] > ordered_mira[1] > ordered_mira[0]
        and ordered_mira[2] > ordered_mira[3] > ordered_mira[4],
        "runtime_seconds": perf_counter() - started,
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=260502014)
    parser.add_argument("--output", type=Path, default=Path("outputs/gmm_bayes_factor.json"))
    parser.add_argument("--upload-repo", help="Optional Hub dataset repository for durable Job output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    config = FULL if not args.smoke else {"dimensions": 8, "components": 3, "truths": 16, "samples": 32, "regions": 7}
    result = run_protocol(config, device=device, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.upload_repo:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("HF_TOKEN is required when --upload-repo is set")
        api = HfApi(token=token)
        api.create_repo(args.upload_repo, repo_type="dataset", private=False, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(args.output),
            path_in_repo="full_gmm/gmm_bayes_factor.json",
            repo_id=args.upload_repo,
            repo_type="dataset",
        )
    print(json.dumps(result, indent=2))
    if not (result["zero_shift_best"] and result["absolute_shift_monotone"]):
        raise SystemExit("GMM comparison did not reproduce the required ordering")


if __name__ == "__main__":
    main()

````


````json title=gmm_smoke.json
{
  "configuration": {
    "dimensions": 8,
    "components": 3,
    "truths": 16,
    "samples": 32,
    "regions": 7
  },
  "exact_source_scale": false,
  "construction": {
    "component_covariance": "identity",
    "mean_translation": "shift / sqrt(dimensions) in every coordinate",
    "seed": 260502014,
    "source_disclosure": "paper source gives dimensions/counts but no GMM construction or script"
  },
  "device": "cpu",
  "rows": [
    {
      "shift": -6,
      "mira": 0.5323660969734192,
      "mira_region_std": 0.07740436494350433,
      "mean_bayes_factor": 2.71613916993374e-05,
      "log_mean_bayes_factor": -10.513713829051305
    },
    {
      "shift": -3,
      "mira": 0.5962611436843872,
      "mira_region_std": 0.08226122707128525,
      "mean_bayes_factor": 0.22784581780433655,
      "log_mean_bayes_factor": -1.4790862722504623
    },
    {
      "shift": 0,
      "mira": 0.6888951063156128,
      "mira_region_std": 0.05760575830936432,
      "mean_bayes_factor": 1.0,
      "log_mean_bayes_factor": 7.618617292592944e-09
    },
    {
      "shift": 3,
      "mira": 0.630859375,
      "mira_region_std": 0.06289547681808472,
      "mean_bayes_factor": 2.0805673599243164,
      "log_mean_bayes_factor": 0.7326407508742325
    },
    {
      "shift": 6,
      "mira": 0.5212053656578064,
      "mira_region_std": 0.06465432047843933,
      "mean_bayes_factor": 0.02032124437391758,
      "log_mean_bayes_factor": -3.8960884733307846
    }
  ],
  "zero_shift_best": true,
  "absolute_shift_monotone": true,
  "runtime_seconds": 0.01769333495758474
}

````


````output
{
  "configuration": {
    "dimensions": 8,
    "components": 3,
    "truths": 16,
    "samples": 32,
    "regions": 7
  },
  "exact_source_scale": false,
  "construction": {
    "component_covariance": "identity",
    "mean_translation": "shift / sqrt(dimensions) in every coordinate",
    "seed": 260502014,
    "source_disclosure": "paper source gives dimensions/counts but no GMM construction or script"
  },
  "device": "cpu",
  "rows": [
    {
      "shift": -6,
      "mira": 0.5323660969734192,
      "mira_region_std": 0.07740436494350433,
      "mean_bayes_factor": 2.71613916993374e-05,
      "log_mean_bayes_factor": -10.513713829051305
    },
    {
      "shift": -3,
      "mira": 0.5962611436843872,
      "mira_region_std": 0.08226122707128525,
      "mean_bayes_factor": 0.22784581780433655,
      "log_mean_bayes_factor": -1.4790862722504623
    },
    {
      "shift": 0,
      "mira": 0.6888951063156128,
      "mira_region_std": 0.05760575830936432,
      "mean_bayes_factor": 1.0,
      "log_mean_bayes_factor": 7.618617292592944e-09
    },
    {
      "shift": 3,
      "mira": 0.630859375,
      "mira_region_std": 0.06289547681808472,
      "mean_bayes_factor": 2.0805673599243164,
      "log_mean_bayes_factor": 0.7326407508742325
    },
    {
      "shift": 6,
      "mira": 0.5212053656578064,
      "mira_region_std": 0.06465432047843933,
      "mean_bayes_factor": 0.02032124437391758,
      "log_mean_bayes_factor": -3.8960884733307846
    }
  ],
  "zero_shift_best": true,
  "absolute_shift_monotone": true,
  "runtime_seconds": 0.01769333495758474
}

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_fc1ff1e4cfda", "created_at": "2026-07-20T09:15:11+00:00", "title": "Run: python bayes_factor_audit.py (exit 0)", "command": ["python", "repro/src/bayes_factor_audit.py"], "exit_code": 0, "duration_s": 6.44}
-->
````bash
$ python repro/src/bayes_factor_audit.py
````

exit 0 · 6.4s


````python title=bayes_factor_audit.py
"""Independent audit of the paper's stated arithmetic Bayes-factor comparator."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from gmm_full_protocol import log_gmm_density


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    generator = torch.Generator(device="cpu").manual_seed(260502014)
    dimensions, components, observations = 100, 20, 100_000
    means = torch.randn((components, dimensions), generator=generator)
    assignments = torch.randint(components, (observations,), generator=generator)
    truth = torch.randn((observations, dimensions), generator=generator) + means[assignments]
    truth_log_density = log_gmm_density(truth, means)
    rows = []
    for shift in (-6, -3, 0, 3, 6):
        candidate = means + torch.full((dimensions,), shift / math.sqrt(dimensions))
        log_ratio = log_gmm_density(truth, candidate) - truth_log_density
        rows.append(
            {
                "shift": shift,
                "mean_density_ratio": float(torch.exp(log_ratio).mean()),
                "mean_log_density_ratio": float(log_ratio.mean()),
            }
        )
    payload = {
        "observations": observations,
        "identity": "E_{y~p_true}[p_candidate(y)/p_true(y)] = 1 for normalized candidate density",
        "rows": rows,
        "finite_sample_arithmetic_ratio_is_unstable": any(
            abs(row["mean_density_ratio"] - 1.0) > 0.15 for row in rows if row["shift"] != 0
        ),
        "mean_log_ratio_ranks_zero_shift": rows[2]["mean_log_density_ratio"] > max(
            row["mean_log_density_ratio"] for index, row in enumerate(rows) if index != 2
        ),
    }
    path = ROOT / "outputs" / "bayes_factor_audit.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not (
        payload["finite_sample_arithmetic_ratio_is_unstable"]
        and payload["mean_log_ratio_ranks_zero_shift"]
    ):
        raise SystemExit("Bayes-factor audit failed")


if __name__ == "__main__":
    main()

````


````output
{
  "observations": 100000,
  "identity": "E_{y~p_true}[p_candidate(y)/p_true(y)] = 1 for normalized candidate density",
  "rows": [
    {
      "shift": -6,
      "mean_density_ratio": 0.05106903985142708,
      "mean_log_density_ratio": -17.984756469726562
    },
    {
      "shift": -3,
      "mean_density_ratio": 0.9025017023086548,
      "mean_log_density_ratio": -4.492377758026123
    },
    {
      "shift": 0,
      "mean_density_ratio": 1.0,
      "mean_log_density_ratio": 0.0
    },
    {
      "shift": 3,
      "mean_density_ratio": 0.9357865452766418,
      "mean_log_density_ratio": -4.507622718811035
    },
    {
      "shift": 6,
      "mean_density_ratio": 0.054085854440927505,
      "mean_log_density_ratio": -18.015247344970703
    }
  ],
  "finite_sample_arithmetic_ratio_is_unstable": true,
  "mean_log_ratio_ranks_zero_shift": true
}

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_04c9800768c2", "created_at": "2026-07-20T09:26:45+00:00", "title": "Run: python aggregate_full_gmm.py (exit 0)", "command": ["python", "repro/src/aggregate_full_gmm.py"], "exit_code": 0, "duration_s": 0.032}
-->
````bash
$ python repro/src/aggregate_full_gmm.py
````

exit 0 · 0.0s


````python title=aggregate_full_gmm.py
"""Aggregate the five preregistered exact-scale GMM seeds after Hub readback."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FULL = {"dimensions": 100, "components": 20, "truths": 5000, "samples": 5000, "regions": 100}


def main() -> None:
    root = ROOT / "outputs" / "hub_readback" / "full_gmm"
    files = sorted(root.glob("*.json"))
    if len(files) != 5:
        raise SystemExit(f"expected five full-scale seed files, found {len(files)}")
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    if not all(run["configuration"] == FULL and run["exact_source_scale"] for run in runs):
        raise SystemExit("one or more runs are not exact source scale")
    margins = []
    for run in runs:
        rows = run["rows"]
        margins.append(rows[2]["mira"] - max(rows[1]["mira"], rows[3]["mira"]))
    mean_margin = sum(margins) / len(margins)
    std_margin = (sum((value - mean_margin) ** 2 for value in margins) / (len(margins) - 1)) ** 0.5
    sem_margin = std_margin / math.sqrt(len(margins))
    payload = {
        "seed_file_count": len(files),
        "files": [path.name for path in files],
        "all_zero_shift_best": all(run["zero_shift_best"] for run in runs),
        "all_absolute_shift_monotone": all(run["absolute_shift_monotone"] for run in runs),
        "nearest_shift_margins": margins,
        "mean_nearest_shift_margin": mean_margin,
        "std_nearest_shift_margin": std_margin,
        "sem_nearest_shift_margin": sem_margin,
        "three_sem_separation": mean_margin > 3 * sem_margin,
        "mean_runtime_seconds": sum(run["runtime_seconds"] for run in runs) / len(runs),
    }
    (ROOT / "outputs" / "full_gmm_aggregate.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not (
        payload["all_zero_shift_best"]
        and payload["all_absolute_shift_monotone"]
        and payload["three_sem_separation"]
    ):
        raise SystemExit("multi-seed full-scale GMM gate failed")


if __name__ == "__main__":
    main()

````


````output
{
  "seed_file_count": 5,
  "files": [
    "gmm_bayes_factor.json",
    "seed-260502015.json",
    "seed-260502016.json",
    "seed-260502017.json",
    "seed-260502018.json"
  ],
  "all_zero_shift_best": true,
  "all_absolute_shift_monotone": true,
  "nearest_shift_margins": [
    0.001290738582611084,
    0.0029351115226745605,
    0.0022013187408447266,
    0.0022637248039245605,
    0.0021702051162719727
  ],
  "mean_nearest_shift_margin": 0.002172219753265381,
  "std_nearest_shift_margin": 0.0005848582927057707,
  "sem_nearest_shift_margin": 0.0002615565799389145,
  "three_sem_separation": true,
  "mean_runtime_seconds": 228.70131868419998
}

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_d27e2518b71f", "created_at": "2026-07-20T09:26:46+00:00", "title": "Run: python verify_claims.py (exit 0)", "command": ["python", "repro/src/verify_claims.py"], "exit_code": 0, "duration_s": 0.034}
-->
````bash
$ python repro/src/verify_claims.py
````

exit 0 · 0.0s


````python title=verify_claims.py
"""Fail closed unless every live MIRA jury claim has retained evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
FULL = {"dimensions": 100, "components": 20, "truths": 5000, "samples": 5000, "regions": 100}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    source = load(OUT / "source_audit.json")
    claims_1_2 = load(OUT / "claims_1_2.json")
    bayes = load(OUT / "bayes_factor_audit.json")
    full_path = OUT / "hub_readback" / "full_gmm" / "gmm_bayes_factor.json"
    full = load(full_path)
    aggregate = load(OUT / "full_gmm_aggregate.json")
    test_path = OUT / "test_results.json"
    tests_passed = test_path.is_file() and load(test_path).get("tests_passed") is True
    claim_1 = source["all_static_checks_pass"] and claims_1_2["source_parity_passed"]
    claim_2 = (
        claims_1_2["finite_null_probability_normalized"]
        and claims_1_2["asymptotic_reference_convergence"]
        and abs(claims_1_2["finite_null_cells"]["5000"]["mean"] - 2 / 3) < 1e-4
    )
    claim_3 = (
        full["configuration"] == FULL
        and full["exact_source_scale"] is True
        and full["device"] == "cuda"
        and aggregate["seed_file_count"] == 5
        and aggregate["all_zero_shift_best"] is True
        and aggregate["all_absolute_shift_monotone"] is True
        and aggregate["three_sem_separation"] is True
        and bayes["finite_sample_arithmetic_ratio_is_unstable"] is True
        and bayes["mean_log_ratio_ranks_zero_shift"] is True
    )
    verified = sum((claim_1, claim_2, claim_3))
    payload = {
        "paper": "ra2t1V4nml",
        "claim_1_conditional_match_score": claim_1,
        "claim_2_null_and_independence_quantities": claim_2,
        "claim_3_direct_model_comparison": claim_3,
        "verified_claims": verified,
        "earned_points": 2 * verified,
        "all_claims_complete": verified == 3,
        "tests_passed": tests_passed,
        "publication_gate_passed": verified == 3 and tests_passed,
        "full_gmm_readback_sha256": hashlib.sha256(full_path.read_bytes()).hexdigest(),
        "full_gmm_job_url": "https://huggingface.co/jobs/DineshAI/6a5de5e0d216bd6f3a2031f7",
        "multi_seed_nearest_zero_shift_margin": aggregate["mean_nearest_shift_margin"],
        "multi_seed_nearest_zero_shift_sem": aggregate["sem_nearest_shift_margin"],
        "multi_seed_nearest_zero_shift_sigma": (
            aggregate["mean_nearest_shift_margin"] / aggregate["sem_nearest_shift_margin"]
        ),
        "bayes_factor_disclosure": (
            "The paper's stated finite arithmetic density-ratio average is an unstable importance-sampling estimator; "
            "the exact-scale MIRA score and an independent log-density comparison rank the true model without it."
        ),
    }
    (OUT / "claim_verification.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["all_claims_complete"]:
        raise SystemExit("publication gate rejects incomplete claims")


if __name__ == "__main__":
    main()

````


````output
{
  "paper": "ra2t1V4nml",
  "claim_1_conditional_match_score": true,
  "claim_2_null_and_independence_quantities": true,
  "claim_3_direct_model_comparison": true,
  "verified_claims": 3,
  "earned_points": 6,
  "all_claims_complete": true,
  "tests_passed": true,
  "publication_gate_passed": true,
  "full_gmm_readback_sha256": "daee3e6c2678a4cc6914927294baaff41496629431dd896f3a26cfaa606a8b93",
  "full_gmm_job_url": "https://huggingface.co/jobs/DineshAI/6a5de5e0d216bd6f3a2031f7",
  "multi_seed_nearest_zero_shift_margin": 0.002172219753265381,
  "multi_seed_nearest_zero_shift_sem": 0.0002615565799389145,
  "multi_seed_nearest_zero_shift_sigma": 8.304970778302325,
  "bayes_factor_disclosure": "The paper's stated finite arithmetic density-ratio average is an unstable importance-sampling estimator; the exact-scale MIRA score and an independent log-density comparison rank the true model without it."
}

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_078762c08e5f", "created_at": "2026-07-20T09:26:46+00:00", "title": "Run: python build_evidence_bundle.py (exit 0)", "command": ["python", "repro/src/build_evidence_bundle.py"], "exit_code": 0, "duration_s": 0.034}
-->
````bash
$ python repro/src/build_evidence_bundle.py
````

exit 0 · 0.0s


````python title=build_evidence_bundle.py
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

````


````output
{
  "gate": "FULL_GATE_READY",
  "queue_marker": "FULL_GATE_READY: ra2t1V4nml",
  "paper": "ra2t1V4nml",
  "evidence_bundle_sha256": "bdc9d2743e1127fc21c3c2d8adf5bba56f91b154d7771f7647f01fbd3f1dd99a",
  "claim_verification_sha256": "377a08cdb843d6ed19dc7afef7f94175207f6883af812222b54daf391dbc446a",
  "claims_complete": true,
  "earned_points": 6,
  "tests_passed": true,
  "publication_gate_passed": true
}

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_6f0d15749291", "created_at": "2026-07-20T09:27:57+00:00", "title": "Run: python build_reproduction_bundle.py (exit 0)", "command": ["python", "repro/src/build_reproduction_bundle.py"], "exit_code": 0, "duration_s": 0.372}
-->
````bash
$ python repro/src/build_reproduction_bundle.py
````

exit 0 · 0.4s


````python title=build_reproduction_bundle.py
"""Create a compact, deterministic publication bundle without environments or secrets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
INCLUDE = ("README.md", "STATUS.md", "docs", "repro", "upstream", "outputs")
EXCLUDE_NAMES = {"reproduction_bundle.zip"}


def wanted(path: Path) -> bool:
    return path.is_file() and path.name not in EXCLUDE_NAMES and "__pycache__" not in path.parts


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

````


````output
{
  "file_count": 66,
  "size_bytes": 8176235,
  "sha256": "8a7e95d1228af1fa80d1b437b368cd274c8428c0578cd23a0b8b606164e6d38a"
}

````


---
<!-- trackio-cell
{"type": "dashboard", "id": "cell_8dca2a1a4042", "created_at": "2026-07-20T09:27:58+00:00", "title": "Dashboard: repro-mira-score", "dashboard_project": "repro-mira-score"}
-->
**🎯 Trackio dashboard** `repro-mira-score`

trackio-local-dashboard://repro-mira-score


---
<!-- trackio-cell
{"type": "code", "id": "cell_14954eecc3e2", "created_at": "2026-07-20T09:27:58+00:00", "title": "Run: python log_bundle_artifact.py (exit 0)", "command": ["python", "repro/src/log_bundle_artifact.py"], "exit_code": 0, "duration_s": 1.001}
-->
````bash
$ python repro/src/log_bundle_artifact.py
````

exit 0 · 1.0s


````python title=log_bundle_artifact.py
"""Register the curated local bundle as a Trackio dataset artifact."""

from __future__ import annotations

from pathlib import Path

import trackio


ROOT = Path(__file__).resolve().parents[2]
run = trackio.init(project="repro-mira-score", name="full-publication-gate", resume="allow")
artifact = trackio.log_artifact(ROOT / "outputs" / "reproduction_bundle.zip", name="repro-bundle", type="dataset")
print(f"artifact={artifact.project}/{artifact.name}:{artifact.version}")
trackio.finish()

````


````output
* Trackio project initialized: repro-mira-score
* Trackio metrics logged to: /home/dineshai/.cache/huggingface/trackio
* View dashboard by running in your terminal:
[1m[38;5;208mtrackio show --project "repro-mira-score"[0m
* or by running in Python: trackio.show(project="repro-mira-score")
* Created new run: full-publication-gate
artifact=repro-mira-score/repro-bundle:v0
* Run finished. Uploading logs to Trackio (please wait...)

````
