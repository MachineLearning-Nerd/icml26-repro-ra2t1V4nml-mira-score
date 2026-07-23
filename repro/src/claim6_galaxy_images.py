"""Full-scale CPU reproduction of the released Claim 6 galaxy experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from mira_score import mira


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".openresearch" / "artifacts" / "claim_6"
CACHE = Path.home() / ".cache" / "openresearch" / "mira-score" / "claim6"
AUTHOR_COMMIT = "3bc229222cbcf72bd470267175d9a6dff6689ce0"
BASE_URL = (
    "https://raw.githubusercontent.com/SammyS15/MIRA_Paper_Plots/"
    f"{AUTHOR_COMMIT}/data/lens_exp"
)
NUM_RUNS = 100
NORMALIZE = False
SEED = 260502014
BOOTSTRAP_SEED = 620052014
PARITY_SEED = 1260502014
BOOTSTRAP_REPLICATES = 10_000
MODEL_NAMES = (
    "spiral_prior_sigma_2_true",
    "spiral_prior_sigma_0_5_noise_misspecified",
    "elliptical_prior_sigma_2_prior_misspecified",
    "elliptical_prior_sigma_0_5_both_misspecified",
)
PAPER_SCORES = (0.6442, 0.5783, 0.5298, 0.5056)
PAPER_ERRORS = (0.0606, 0.0728, 0.0748, 0.0690)
FILES = {
    "true.pt": {
        "sha256": "74a9b24d746a5dccf2af77fbc72bddd544760d494cbe919e69d704fbb62a5930",
        "size_bytes": 787_106,
        "shape": [16, 3, 64, 64],
    },
    "posterior0.pt": {
        "sha256": "71009759c762452941c375b14b8e701c375c276fb920ecc55304770c01e1370d",
        "size_bytes": 50_332_404,
        "shape": [16, 64, 3, 64, 64],
    },
    "posterior1.pt": {
        "sha256": "c874b78bbabb3a2c5b057d602ea4e75b41546c14499c3a3d152d9d8b47ed8661",
        "size_bytes": 50_332_404,
        "shape": [16, 64, 3, 64, 64],
    },
    "posterior2.pt": {
        "sha256": "4eec7a23e4677fb8e8d6c2cee32950715d4788ba24fa2dc9151ef62fcf270698",
        "size_bytes": 50_332_404,
        "shape": [16, 64, 3, 64, 64],
    },
    "posterior3.pt": {
        "sha256": "a819b262c52fe38e19ea04594dee33ea670943d80285a974b85316e90875cbca",
        "size_bytes": 50_332_404,
        "shape": [16, 64, 3, 64, 64],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_and_verify() -> list[dict[str, object]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, expected in FILES.items():
        destination = CACHE / name
        if not destination.is_file() or sha256(destination) != expected["sha256"]:
            temporary = destination.with_suffix(destination.suffix + ".partial")
            request = urllib.request.Request(
                f"{BASE_URL}/{name}",
                headers={"User-Agent": "OpenResearch-MIRA-Reproduction/1.0"},
            )
            with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            if sha256(temporary) != expected["sha256"]:
                temporary.unlink(missing_ok=True)
                raise SystemExit(f"SHA-256 mismatch after downloading {name}")
            temporary.replace(destination)
        actual = {
            "path": name,
            "source_url": f"{BASE_URL}/{name}",
            "sha256": sha256(destination),
            "size_bytes": destination.stat().st_size,
            "expected_shape": expected["shape"],
        }
        if actual["sha256"] != expected["sha256"] or actual["size_bytes"] != expected["size_bytes"]:
            raise SystemExit(f"protected data manifest mismatch for {name}")
        manifest.append(actual)
    return manifest


def load_tensor(name: str) -> torch.Tensor:
    path = CACHE / name
    try:
        value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        value = torch.load(path, map_location="cpu", weights_only=True)
    expected_shape = tuple(FILES[name]["shape"])
    if tuple(value.shape) != expected_shape:
        raise SystemExit(f"{name}: expected shape {expected_shape}, got {tuple(value.shape)}")
    if value.dtype != torch.float32:
        raise SystemExit(f"{name}: expected float32, got {value.dtype}")
    return value


def score_one_model(
    truth: torch.Tensor,
    posterior: torch.Tensor,
    centers: torch.Tensor,
    reference_indices: torch.Tensor,
    normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match released ``mira`` while avoiding a 4-D distance tensor."""
    sample_count = posterior.shape[1]
    if normalize:
        truth_min = truth.min(dim=0, keepdim=True).values
        truth_range = truth.max(dim=0, keepdim=True).values - truth_min + 1e-8
        truth_scored = (truth - truth_min) / truth_range
        posterior_scored = (posterior - truth_min) / truth_range
    else:
        truth_scored = truth
        posterior_scored = posterior

    # Exact Euclidean ordering via squared distances. torch.bmm computes all
    # 100 region centers in one CPU BLAS call for each of the 16 truth blocks.
    posterior_sq = (posterior_scored * posterior_scored).sum(dim=2)
    center_by_truth = centers.permute(1, 2, 0).contiguous()
    cross = torch.bmm(posterior_scored, center_by_truth)
    center_sq = (centers * centers).sum(dim=2).transpose(0, 1)
    distances_sq = (posterior_sq[:, :, None] + center_sq[:, None, :] - 2 * cross).clamp_min_(0)
    distances_sq = distances_sq.permute(2, 0, 1).contiguous()  # (runs, truths, samples)

    radii_sq = distances_sq.gather(2, reference_indices[:, :, None]).squeeze(2)
    counts = (distances_sq < radii_sq[:, :, None]).sum(dim=2)
    truth_sq = ((centers - truth_scored[None, :, :]) ** 2).sum(dim=2)
    hit = truth_sq <= radii_sq
    # This is algebraically identical to the released normalized score:
    # ((counts+1)/(S+1))/(S/(S+1)) if hit, else ((S-counts)/(S+1))/(S/(S+1)).
    calibrated = torch.where(hit, counts + 1, sample_count - counts).to(torch.float64) / sample_count

    rolled_truth = truth_scored.roll(shifts=1, dims=0)
    rolled_truth_sq = ((centers - rolled_truth[None, :, :]) ** 2).sum(dim=2)
    rolled_hit = rolled_truth_sq <= radii_sq
    negative = torch.where(
        rolled_hit, counts + 1, sample_count - counts
    ).to(torch.float64) / sample_count
    return calibrated, negative


def interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def bootstrap_intervals(per_truth: np.ndarray) -> dict[str, object]:
    """Paired two-axis bootstrap over the 100 regions and 16 fiducials."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    run_count, model_count, truth_count = per_truth.shape
    run_indices = rng.integers(0, run_count, size=(BOOTSTRAP_REPLICATES, run_count))
    truth_indices = rng.integers(0, truth_count, size=(BOOTSTRAP_REPLICATES, truth_count))
    boot = np.empty((BOOTSTRAP_REPLICATES, model_count), dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        selected = per_truth[run_indices[index]][:, :, truth_indices[index]]
        boot[index] = selected.mean(axis=(0, 2))
    score_intervals = {MODEL_NAMES[m]: interval(boot[:, m]) for m in range(model_count)}
    paired = {}
    for model_index in range(1, model_count):
        difference = boot[:, 0] - boot[:, model_index]
        paired[MODEL_NAMES[model_index]] = {
            "mean_difference": float(difference.mean()),
            "ci95": interval(difference),
            "ci_excludes_zero": bool(np.quantile(difference, 0.025) > 0),
        }
    return {
        "method": "paired nonparametric bootstrap over both region runs and L=16 fiducials",
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "score_ci95": score_intervals,
        "true_minus_misspecified": paired,
    }


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def released_implementation_parity(truth: torch.Tensor) -> dict[str, object]:
    """Compare one full-scale region against the released implementation."""
    posterior = load_tensor("posterior0.pt").reshape(16, 64, 3 * 64 * 64)
    torch.manual_seed(PARITY_SEED)
    official_mean, _ = mira(
        truth,
        posterior[None, :, :, :],
        num_runs=1,
        norm=NORMALIZE,
        disable_tqdm=True,
        device=torch.device("cpu"),
    )
    torch.manual_seed(PARITY_SEED)
    centers = torch.rand((1, 16, 3 * 64 * 64))
    references = torch.randint(0, 64, (1, 16))
    optimized, _ = score_one_model(
        truth, posterior, centers, references, NORMALIZE
    )
    optimized_mean = optimized.mean()
    delta = abs(float(official_mean[0]) - float(optimized_mean))
    # One changed integer count changes the aggregate by exactly 1/(64*16).
    score_quantum = 1 / (64 * 16)
    payload = {
        "scope": "one region, one model, L=16, N=64, dimensions=12288",
        "seed": PARITY_SEED,
        "released_implementation_score": float(official_mean[0]),
        "optimized_implementation_score": float(optimized_mean),
        "absolute_delta": delta,
        "single_count_score_quantum": score_quantum,
        "within_one_count_quantum": delta <= score_quantum + 1e-12,
    }
    if not payload["within_one_count_quantum"]:
        raise SystemExit("full-scale released implementation parity failed")
    return payload


def main() -> None:
    started = perf_counter()
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    data_manifest = download_and_verify()
    truth_image = load_tensor("true.pt")
    truth = truth_image.reshape(16, 3 * 64 * 64)

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    centers = torch.rand((NUM_RUNS, 16, 3 * 64 * 64), generator=generator)
    references = torch.randint(0, 64, (4, NUM_RUNS, 16), generator=generator)
    model_scores = []
    negative_scores = []
    for model_index in range(4):
        posterior = load_tensor(f"posterior{model_index}.pt").reshape(16, 64, 3 * 64 * 64)
        scores, negative = score_one_model(
            truth, posterior, centers, references[model_index], NORMALIZE
        )
        model_scores.append(scores)
        negative_scores.append(negative)
        del posterior
    parity = released_implementation_parity(truth)
    per_truth = torch.stack(model_scores, dim=1).numpy()  # (runs, models, truths)
    negative_per_truth = torch.stack(negative_scores, dim=1).numpy()

    with (ARTIFACTS / "raw_per_region_truth.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("seed", "region_run", "model_index", "model", "truth_index", "score"),
        )
        writer.writeheader()
        for run_index in range(NUM_RUNS):
            for model_index, model_name in enumerate(MODEL_NAMES):
                for truth_index in range(16):
                    writer.writerow(
                        {
                            "seed": SEED,
                            "region_run": run_index,
                            "model_index": model_index,
                            "model": model_name,
                            "truth_index": truth_index,
                            "score": f"{per_truth[run_index, model_index, truth_index]:.10f}",
                        }
                    )

    run_scores = per_truth.mean(axis=2)
    means = run_scores.mean(axis=0)
    errors = run_scores.std(axis=0, ddof=1)
    paper_deltas = means - np.asarray(PAPER_SCORES)
    bootstrap = bootstrap_intervals(per_truth)
    summaries = []
    for model_index, name in enumerate(MODEL_NAMES):
        summaries.append(
            {
                "model_index": model_index,
                "model": name,
                "observed_score": float(means[model_index]),
                "observed_region_std": float(errors[model_index]),
                "bootstrap_ci95": bootstrap["score_ci95"][name],
                "paper_score": PAPER_SCORES[model_index],
                "paper_error": PAPER_ERRORS[model_index],
                "score_delta_observed_minus_paper": float(paper_deltas[model_index]),
            }
        )
    raw_summary = {
        "claim_id": 6,
        "paper": "arXiv:2605.02014v1",
        "author_repository_commit": AUTHOR_COMMIT,
        "protocol": {
            "L": 16,
            "N": 64,
            "dimensions": 12_288,
            "image_shape": [64, 64, 3],
            "regions": NUM_RUNS,
            "norm": NORMALIZE,
            "center_distribution": "Uniform[0,1]^12288",
            "scoring_seed": SEED,
        },
        "models": summaries,
        "observed_order": [MODEL_NAMES[index] for index in np.argsort(-means)],
        "paper_order": list(MODEL_NAMES),
        "bootstrap": bootstrap,
    }
    (ARTIFACTS / "raw_summary.json").write_text(
        json.dumps(raw_summary, indent=2) + "\n", encoding="utf-8"
    )
    (ARTIFACTS / "data_manifest.json").write_text(
        json.dumps(
            {
                "author_repository": "https://github.com/SammyS15/MIRA_Paper_Plots",
                "author_commit": AUTHOR_COMMIT,
                "files": data_manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (ARTIFACTS / "implementation_parity.json").write_text(
        json.dumps(parity, indent=2) + "\n", encoding="utf-8"
    )

    negative_run_scores = negative_per_truth.mean(axis=2)
    negative_means = negative_run_scores.mean(axis=0)
    negative = {
        "control_type": "roll truth-to-observation pairing by one fiducial",
        "expected": "the deliberately broken true-model pairing scores lower than the intact pairing",
        "intact_true_model_score": float(means[0]),
        "rolled_pairing_true_model_score": float(negative_means[0]),
        "drop": float(means[0] - negative_means[0]),
        "negative_control_passed": bool(negative_means[0] < means[0]),
        "all_rolled_model_scores": {
            MODEL_NAMES[index]: float(negative_means[index]) for index in range(4)
        },
    }
    (ARTIFACTS / "negative_control_output.json").write_text(
        json.dumps(negative, indent=2) + "\n", encoding="utf-8"
    )
    environment = {
        "command": "uv run --frozen python repro/src/run_campaign.py",
        "git_sha": git_sha(),
        "uv_lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
        "deterministic_seeds": [SEED, BOOTSTRAP_SEED, PARITY_SEED],
        "runtime_seconds": perf_counter() - started,
    }
    (ARTIFACTS / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "summary": raw_summary,
                "implementation_parity": parity,
                "negative_control": negative,
                "environment": environment,
            },
            indent=2,
        ),
        flush=True,
    )
    if raw_summary["observed_order"] != list(MODEL_NAMES):
        raise SystemExit("full-scale observed model ranking differs from the paper order")
    if not negative["negative_control_passed"]:
        raise SystemExit("broken-pairing negative control did not lower the true-model score")


if __name__ == "__main__":
    main()
