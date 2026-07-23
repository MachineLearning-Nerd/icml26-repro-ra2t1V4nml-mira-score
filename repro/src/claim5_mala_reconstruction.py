"""CPU reconstruction and feasibility profile for MIRA Claim 5."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as functional

from claim5_lensing import (
    CAUSTICS_COMMIT,
    PARAMETER_BOUNDS,
    SOURCE_NUISANCE_REGIMES,
    active_dimension,
    embed_candidate_parameters,
    physical_to_normalized,
    simulate,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".openresearch" / "artifacts" / "claim_5_profile"
FIXED_COMMAND = "uv run --frozen python repro/src/run_campaign.py"
SEED = 5_260_502_014
MODELS = (
    ("EPL+3_Sersic_true", "EPL", 3),
    ("SIE+3_Sersic", "SIE", 3),
    ("EPL+1_Sersic", "EPL", 1),
    ("SIE+1_Sersic", "SIE", 1),
)


@dataclass(frozen=True)
class Protocol:
    truths: int
    walkers: int
    burnin_steps: int
    sampling_steps: int
    regions: int
    pixels: int = 100
    pixelscale: float = 0.05
    noise_sigma: float = 1.0
    epl_iterations: int = 18

    @property
    def samples(self) -> int:
        return self.walkers * self.sampling_steps


# This branch is deliberately a feasibility profile. The child promoted from
# it must replace only this committed protocol with the paper scale.
PROTOCOL = Protocol(
    truths=1,
    walkers=16,
    burnin_steps=20,
    sampling_steps=20,
    regions=20,
)


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def uv_lock_sha256() -> str:
    return hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()


def log_density_and_gradient(
    unconstrained: torch.Tensor,
    observation: torch.Tensor,
    *,
    lens_type: str,
    source_count: int,
    nuisance_regime: dict[str, object],
    protocol: Protocol,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = unconstrained.detach().requires_grad_(True)
    unit = state.sigmoid()
    parameters = embed_candidate_parameters(
        unit, lens_type=lens_type, source_count=source_count
    )
    prediction = simulate(
        parameters,
        lens_type=lens_type,
        source_count=source_count,
        nuisance_regime=nuisance_regime,
        pixels=protocol.pixels,
        pixelscale=protocol.pixelscale,
        iterations=protocol.epl_iterations,
    )
    residual = (prediction - observation[None, :, :]) / protocol.noise_sigma
    log_likelihood = -0.5 * (residual * residual).sum(dim=(1, 2))
    # Uniform priors in the paper's bounded physical coordinates. Sampling in
    # logit coordinates requires the logistic Jacobian.
    log_jacobian = (
        functional.logsigmoid(state) + functional.logsigmoid(-state)
    ).sum(dim=1)
    log_density = log_likelihood + log_jacobian
    gradient = torch.autograd.grad(log_density.sum(), state)[0]
    return log_density.detach(), gradient.detach()


def mala(
    observation: torch.Tensor,
    *,
    lens_type: str,
    source_count: int,
    nuisance_regime: dict[str, object],
    truth_unit: torch.Tensor,
    protocol: Protocol,
    seed: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dimension = active_dimension(lens_type, source_count)
    active_indices = [0, 1, 2]
    if lens_type == "EPL":
        active_indices.append(3)
    for source_index in range(source_count):
        active_indices.extend(range(4 + 3 * source_index, 7 + 3 * source_index))

    # A dispersed mixture of prior draws and truth-centred draws gives MALA a
    # chance to expose multiple basins without treating the truth as known.
    prior_unit = 0.02 + 0.96 * torch.rand(
        (protocol.walkers, dimension), generator=generator
    )
    local_unit = truth_unit[active_indices][None, :] + 0.08 * torch.randn(
        (protocol.walkers, dimension), generator=generator
    )
    local_unit = local_unit.clamp(0.02, 0.98)
    use_local = (
        torch.arange(protocol.walkers) % 2 == 0
    )[:, None].expand(-1, dimension)
    initial_unit = torch.where(use_local, local_unit, prior_unit)
    state = torch.logit(initial_unit)

    log_density, gradient = log_density_and_gradient(
        state,
        observation,
        lens_type=lens_type,
        source_count=source_count,
        nuisance_regime=nuisance_regime,
        protocol=protocol,
    )
    log_step_size = math.log(0.08)
    target_acceptance = 0.574
    burnin_acceptance: list[float] = []
    production_acceptance: list[float] = []
    chains: list[torch.Tensor] = []

    total_steps = protocol.burnin_steps + protocol.sampling_steps
    for step in range(total_steps):
        step_size = math.exp(log_step_size)
        drift = 0.5 * step_size**2 * gradient
        proposed = (
            state
            + drift
            + step_size
            * torch.randn(
                state.shape, dtype=state.dtype, generator=generator
            )
        )
        proposed_density, proposed_gradient = log_density_and_gradient(
            proposed,
            observation,
            lens_type=lens_type,
            source_count=source_count,
            nuisance_regime=nuisance_regime,
            protocol=protocol,
        )
        proposed_drift = 0.5 * step_size**2 * proposed_gradient
        forward = -0.5 * (
            (proposed - state - drift) ** 2
        ).sum(dim=1) / step_size**2
        reverse = -0.5 * (
            (state - proposed - proposed_drift) ** 2
        ).sum(dim=1) / step_size**2
        log_acceptance = proposed_density - log_density + reverse - forward
        accepted = (
            torch.log(torch.rand(protocol.walkers, generator=generator))
            < log_acceptance
        )
        state = torch.where(accepted[:, None], proposed, state)
        log_density = torch.where(accepted, proposed_density, log_density)
        gradient = torch.where(accepted[:, None], proposed_gradient, gradient)
        acceptance = float(accepted.float().mean())

        if step < protocol.burnin_steps:
            burnin_acceptance.append(acceptance)
            adaptation_rate = 0.35 / math.sqrt(step + 1)
            log_step_size += adaptation_rate * (
                acceptance - target_acceptance
            )
            log_step_size = min(max(log_step_size, math.log(0.002)), math.log(0.5))
        else:
            production_acceptance.append(acceptance)
            chains.append(state.sigmoid().detach().clone())

    chain = torch.stack(chains, dim=0)
    flat_unit = chain.reshape(-1, dimension)
    flat_physical = embed_candidate_parameters(
        flat_unit, lens_type=lens_type, source_count=source_count
    )
    diagnostics = chain_diagnostics(chain)
    diagnostics.update(
        {
            "active_dimension": dimension,
            "burnin_acceptance": float(np.mean(burnin_acceptance)),
            "production_acceptance": float(np.mean(production_acceptance)),
            "final_step_size": math.exp(log_step_size),
            "finite_samples": bool(torch.isfinite(flat_physical).all()),
            "sample_shape": list(flat_physical.shape),
        }
    )
    return flat_physical, diagnostics


def chain_diagnostics(chain: torch.Tensor) -> dict[str, object]:
    """Split-R-hat and a conservative autocorrelation ESS estimate."""
    steps, walkers, dimensions = chain.shape
    half = steps // 2
    if half < 4:
        return {"rhat_max": None, "ess_min": None}
    split = torch.cat((chain[:half], chain[-half:]), dim=1).permute(1, 0, 2)
    chain_means = split.mean(dim=1)
    within = split.var(dim=1, unbiased=True).mean(dim=0)
    between = half * chain_means.var(dim=0, unbiased=True)
    variance = ((half - 1) / half) * within + between / half
    rhat = torch.sqrt(variance / within.clamp_min(1.0e-12))

    centered = split - split.mean(dim=1, keepdim=True)
    variance_zero = (centered * centered).mean(dim=1).mean(dim=0)
    autocorrelation_sum = torch.zeros(dimensions)
    previous_pair = torch.full((dimensions,), float("inf"))
    for lag in range(1, min(half - 1, 50)):
        covariance = (
            centered[:, :-lag, :] * centered[:, lag:, :]
        ).mean(dim=1).mean(dim=0)
        rho = covariance / variance_zero.clamp_min(1.0e-12)
        if lag % 2 == 0:
            pair = previous_pair + rho
            positive = pair > 0
            autocorrelation_sum += torch.where(
                positive, pair, torch.zeros_like(pair)
            )
        else:
            previous_pair = rho
    total = split.shape[0] * split.shape[1]
    ess = total / (1.0 + 2.0 * autocorrelation_sum).clamp_min(1.0)
    return {
        "rhat_max": float(rhat.max()),
        "rhat_by_active_dimension": rhat.tolist(),
        "ess_min": float(ess.min()),
        "ess_by_active_dimension": ess.tolist(),
    }


def mira_cells(
    truths_unit: torch.Tensor,
    posterior_physical: torch.Tensor,
    *,
    regions: int,
    seed: int,
) -> torch.Tensor:
    """Return every region × model × truth MIRA cell at paper normalization."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model_count, truth_count, sample_count, _ = posterior_physical.shape
    posterior_unit = physical_to_normalized(posterior_physical)
    centers = torch.rand((regions, truth_count, 13), generator=generator)
    references = torch.randint(
        0, sample_count, (regions, model_count, truth_count), generator=generator
    )
    cells = torch.empty((regions, model_count, truth_count), dtype=torch.float64)
    for model_index in range(model_count):
        for truth_index in range(truth_count):
            samples = posterior_unit[model_index, truth_index]
            center = centers[:, truth_index]
            distances = torch.cdist(center, samples)
            radius = distances.gather(
                1, references[:, model_index, truth_index, None]
            ).squeeze(1)
            counts = (distances < radius[:, None]).sum(dim=1)
            truth_distance = torch.linalg.vector_norm(
                center - truths_unit[truth_index], dim=1
            )
            hit = truth_distance <= radius
            cells[:, model_index, truth_index] = torch.where(
                hit, counts + 1, sample_count - counts
            ).to(torch.float64) / sample_count
    return cells


def write_cells(cells: torch.Tensor) -> None:
    with (ARTIFACTS / "raw_mira_cells.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("region", "model", "truth", "mira_statistic"))
        for region in range(cells.shape[0]):
            for model in range(cells.shape[1]):
                for truth in range(cells.shape[2]):
                    writer.writerow(
                        (region, MODELS[model][0], truth, float(cells[region, model, truth]))
                    )


def main() -> None:
    started = perf_counter()
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    torch.use_deterministic_algorithms(True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    regime = SOURCE_NUISANCE_REGIMES[0]
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    truths_unit = torch.rand((PROTOCOL.truths, 13), generator=generator)
    truths_physical = (
        PARAMETER_BOUNDS[:, 0]
        + truths_unit * (PARAMETER_BOUNDS[:, 1] - PARAMETER_BOUNDS[:, 0])
    )
    clean = simulate(
        truths_physical,
        lens_type="EPL",
        source_count=3,
        nuisance_regime=regime,
        pixels=PROTOCOL.pixels,
        pixelscale=PROTOCOL.pixelscale,
        iterations=PROTOCOL.epl_iterations,
    )
    observations = clean + PROTOCOL.noise_sigma * torch.randn(
        clean.shape, generator=generator
    )

    posterior_by_model = []
    diagnostics: dict[str, list[dict[str, object]]] = {}
    model_runtimes: dict[str, float] = {}
    for model_index, (name, lens_type, source_count) in enumerate(MODELS):
        model_started = perf_counter()
        posterior_by_truth = []
        diagnostics[name] = []
        for truth_index in range(PROTOCOL.truths):
            posterior, diagnostic = mala(
                observations[truth_index],
                lens_type=lens_type,
                source_count=source_count,
                nuisance_regime=regime,
                truth_unit=truths_unit[truth_index],
                protocol=PROTOCOL,
                seed=SEED + 10_000 * model_index + truth_index,
            )
            posterior_by_truth.append(posterior)
            diagnostics[name].append(diagnostic)
        posterior_by_model.append(torch.stack(posterior_by_truth))
        model_runtimes[name] = perf_counter() - model_started
    posterior_physical = torch.stack(posterior_by_model)

    cells = mira_cells(
        truths_unit,
        posterior_physical,
        regions=PROTOCOL.regions,
        seed=SEED + 900_000,
    )
    write_cells(cells)
    scores = cells.mean(dim=(0, 2))
    swapped = scores[[1, 0, 2, 3]]
    payload = {
        "stage": "FEASIBILITY_PROFILE_NOT_CLAIM_EVIDENCE",
        "protocol": {
            "truths_L": PROTOCOL.truths,
            "posterior_samples_N": PROTOCOL.samples,
            "dimensions": 13,
            "regions": PROTOCOL.regions,
            "walkers": PROTOCOL.walkers,
            "burnin_steps": PROTOCOL.burnin_steps,
            "sampling_steps": PROTOCOL.sampling_steps,
            "pixels": [PROTOCOL.pixels, PROTOCOL.pixels],
            "pixelscale_arcsec": PROTOCOL.pixelscale,
            "noise": f"N(0,{PROTOCOL.noise_sigma:g}) independently per pixel",
        },
        "models": [row[0] for row in MODELS],
        "nuisance_regime": regime,
        "scores": {
            MODELS[index][0]: float(scores[index])
            for index in range(len(MODELS))
        },
        "diagnostics": diagnostics,
        "model_runtime_seconds": model_runtimes,
        "runtime_seconds": perf_counter() - started,
        "negative_control": {
            "operation": "swap true and SIE+3 model labels",
            "swapped_scores": swapped.tolist(),
            "swapped_true_label_ranks_first": bool(
                swapped[0] == swapped.max()
            ),
            "expected_rejection": True,
        },
        "environment": {
            "command": FIXED_COMMAND,
            "git_sha": git_sha(),
            "uv_lock_sha256": uv_lock_sha256(),
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
            "seed": SEED,
            "caustics_equation_commit": CAUSTICS_COMMIT,
        },
        "limitations": [
            "This is a one-truth sampler and runtime profile, not Claim 5 evidence.",
            "The paper does not release the fixed source nuisance values; this profile uses a disclosed cited-distribution regime.",
            "A child must run L=100, N=20,000, 100 regions, and both nuisance regimes before any terminal verdict.",
        ],
    }
    (ARTIFACTS / "profile_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    if tuple(posterior_physical.shape) != (
        4,
        PROTOCOL.truths,
        PROTOCOL.samples,
        13,
    ):
        raise SystemExit("posterior shape contract failed")
    if not torch.isfinite(posterior_physical).all():
        raise SystemExit("non-finite posterior sample")
    production_acceptance = [
        float(row["production_acceptance"])
        for model in diagnostics.values()
        for row in model
    ]
    if not all(0.01 <= value <= 0.99 for value in production_acceptance):
        raise SystemExit(f"MALA acceptance profile unusable: {production_acceptance}")


if __name__ == "__main__":
    main()
