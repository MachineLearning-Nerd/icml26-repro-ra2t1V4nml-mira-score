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
    parser.add_argument("--upload-path", default="full_gmm/gmm_bayes_factor.json")
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
            path_in_repo=args.upload_path,
            repo_id=args.upload_repo,
            repo_type="dataset",
        )
    print(json.dumps(result, indent=2))
    if not (result["zero_shift_best"] and result["absolute_shift_monotone"]):
        raise SystemExit("GMM comparison did not reproduce the required ordering")


if __name__ == "__main__":
    main()
