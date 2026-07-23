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
    walkers=25,
    burnin_steps=400,
    sampling_steps=800,
    regions=100,
)
SAMPLER_NAME = "preconditioned_hmc"
HMC_LEAPFROG_STEPS = 4


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


def gauss_newton_precision(
    state: torch.Tensor,
    *,
    lens_type: str,
    source_count: int,
    nuisance_regime: dict[str, object],
    protocol: Protocol,
) -> torch.Tensor:
    """Return the local positive Gauss--Newton precision in logit space."""
    dimension = state.numel()
    delta = 0.002
    perturbed = state.repeat(2 * dimension, 1)
    indices = torch.arange(dimension)
    perturbed[indices, indices] -= delta
    perturbed[dimension + indices, indices] += delta
    parameters = embed_candidate_parameters(
        perturbed.sigmoid(),
        lens_type=lens_type,
        source_count=source_count,
    )
    predictions = simulate(
        parameters,
        lens_type=lens_type,
        source_count=source_count,
        nuisance_regime=nuisance_regime,
        pixels=protocol.pixels,
        pixelscale=protocol.pixelscale,
        iterations=protocol.epl_iterations,
    )
    jacobian = (
        predictions[dimension:] - predictions[:dimension]
    ).reshape(dimension, -1) / (2 * delta * protocol.noise_sigma)
    unit = state.sigmoid()
    prior_precision = 2.0 * unit * (1.0 - unit)
    precision = jacobian @ jacobian.T + torch.diag(prior_precision)
    return 0.5 * (precision + precision.T)


def regularize_precision(
    precision: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(precision)
    maximum = eigenvalues.max().clamp_min(1.0)
    floor = maximum * 1.0e-7
    regularized_eigenvalues = eigenvalues.clamp(min=floor, max=1.0e10)
    regularized = (
        eigenvectors
        @ torch.diag(regularized_eigenvalues)
        @ eigenvectors.T
    )
    mass = (
        eigenvectors
        @ torch.diag(regularized_eigenvalues.reciprocal())
        @ eigenvectors.T
    )
    mass = 0.5 * (mass + mass.T)
    cholesky = torch.linalg.cholesky(
        mass + torch.eye(mass.shape[0]) * 1.0e-10
    )
    return regularized, mass, cholesky, regularized_eigenvalues


def map_and_full_mass(
    observation: torch.Tensor,
    *,
    lens_type: str,
    source_count: int,
    nuisance_regime: dict[str, object],
    protocol: Protocol,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Data-only multi-start MAP and full Gauss--Newton preconditioner."""
    dimension = active_dimension(lens_type, source_count)
    search_unit = 0.01 + 0.98 * torch.rand(
        (64, dimension), generator=generator
    )
    search_state = torch.logit(search_unit)
    search_density, _ = log_density_and_gradient(
        search_state,
        observation,
        lens_type=lens_type,
        source_count=source_count,
        nuisance_regime=nuisance_regime,
        protocol=protocol,
    )
    state = search_state[torch.topk(search_density, k=8).indices].clone()
    first_density = float(search_density.max())
    first_moment = torch.zeros_like(state)
    second_moment = torch.zeros_like(state)
    learning_rate = 0.025
    for iteration in range(1, 81):
        density, gradient = log_density_and_gradient(
            state,
            observation,
            lens_type=lens_type,
            source_count=source_count,
            nuisance_regime=nuisance_regime,
            protocol=protocol,
        )
        gradient_norm = torch.linalg.vector_norm(gradient, dim=1).clamp_min(1.0)
        clipped = gradient * (500.0 / gradient_norm).clamp_max(1.0)[:, None]
        first_moment = 0.9 * first_moment + 0.1 * clipped
        second_moment = 0.999 * second_moment + 0.001 * clipped.square()
        corrected_first = first_moment / (1 - 0.9**iteration)
        corrected_second = second_moment / (1 - 0.999**iteration)
        state = state + learning_rate * corrected_first / (
            corrected_second.sqrt() + 1.0e-8
        )
        state = state.clamp(-7.0, 7.0)

    optimized_density, _ = log_density_and_gradient(
        state,
        observation,
        lens_type=lens_type,
        source_count=source_count,
        nuisance_regime=nuisance_regime,
        protocol=protocol,
    )
    best = int(torch.argmax(optimized_density))
    map_state = state[best].clone()

    # Refine the best Adam point with damped Gauss--Newton steps. This uses only
    # the observation and candidate likelihood; no simulated truth is exposed.
    damping = 1.0e-3
    gn_iterations = 0
    accepted_gn_steps = 0
    for iteration in range(30):
        density, gradient = log_density_and_gradient(
            map_state[None, :],
            observation,
            lens_type=lens_type,
            source_count=source_count,
            nuisance_regime=nuisance_regime,
            protocol=protocol,
        )
        precision = gauss_newton_precision(
            map_state,
            lens_type=lens_type,
            source_count=source_count,
            nuisance_regime=nuisance_regime,
            protocol=protocol,
        )
        eye = torch.eye(dimension)
        direction = torch.linalg.solve(
            precision + damping * eye, gradient[0]
        )
        direction_norm = torch.linalg.vector_norm(direction)
        if direction_norm > 1.0:
            direction = direction / direction_norm
        candidates = torch.stack(
            [
                (map_state + scale * direction).clamp(-9.0, 9.0)
                for scale in (1.0, 0.5, 0.25, 0.125, 0.0625)
            ]
        )
        candidate_density, _ = log_density_and_gradient(
            candidates,
            observation,
            lens_type=lens_type,
            source_count=source_count,
            nuisance_regime=nuisance_regime,
            protocol=protocol,
        )
        candidate_best = int(torch.argmax(candidate_density))
        gn_iterations = iteration + 1
        if candidate_density[candidate_best] > density[0]:
            map_state = candidates[candidate_best]
            damping = max(damping / 3.0, 1.0e-8)
            accepted_gn_steps += 1
        else:
            damping = min(damping * 10.0, 1.0e8)
        preconditioned_norm = torch.sqrt(
            torch.clamp(gradient[0] @ direction, min=0.0)
        )
        if preconditioned_norm < 1.0e-2:
            break

    optimized_density, optimized_gradient = log_density_and_gradient(
        map_state[None, :],
        observation,
        lens_type=lens_type,
        source_count=source_count,
        nuisance_regime=nuisance_regime,
        protocol=protocol,
    )
    precision = gauss_newton_precision(
        map_state,
        lens_type=lens_type,
        source_count=source_count,
        nuisance_regime=nuisance_regime,
        protocol=protocol,
    )
    precision, mass, cholesky, eigenvalues = regularize_precision(precision)
    preconditioned_gradient_norm = torch.sqrt(
        torch.clamp(
            optimized_gradient[0] @ mass @ optimized_gradient[0], min=0.0
        )
    )
    diagnostics = {
        "map_prior_search_log_density": first_density,
        "map_optimized_log_density": float(optimized_density[0]),
        "map_gradient_norm": float(
            torch.linalg.vector_norm(optimized_gradient[0])
        ),
        "map_preconditioned_gradient_norm": float(
            preconditioned_gradient_norm
        ),
        "precision_eigenvalue_min": float(eigenvalues.min()),
        "precision_eigenvalue_max": float(eigenvalues.max()),
        "precision_condition_number": float(
            eigenvalues.max() / eigenvalues.min()
        ),
        "map_starts": 64,
        "map_optimized_starts": 8,
        "map_iterations": 80,
        "gauss_newton_iterations": gn_iterations,
        "gauss_newton_accepted_steps": accepted_gn_steps,
    }
    return map_state, precision, mass, cholesky, diagnostics


def mala(
    observation: torch.Tensor,
    *,
    lens_type: str,
    source_count: int,
    nuisance_regime: dict[str, object],
    protocol: Protocol,
    seed: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dimension = active_dimension(lens_type, source_count)
    map_state, precision, mass, cholesky, map_diagnostics = map_and_full_mass(
        observation,
        lens_type=lens_type,
        source_count=source_count,
        nuisance_regime=nuisance_regime,
        protocol=protocol,
        generator=generator,
    )
    state = map_state[None, :] + 0.5 * (
        torch.randn(
            (protocol.walkers, dimension), generator=generator
        )
        @ cholesky.T
    )

    log_density, gradient = log_density_and_gradient(
        state,
        observation,
        lens_type=lens_type,
        source_count=source_count,
        nuisance_regime=nuisance_regime,
        protocol=protocol,
    )
    initial_gradient_norms = torch.linalg.vector_norm(gradient, dim=1)
    cholesky_precision = torch.linalg.cholesky(precision)
    log_step_size = math.log(0.12)
    target_acceptance = 0.65
    burnin_acceptance: list[float] = []
    production_acceptance: list[float] = []
    chains: list[torch.Tensor] = []

    total_steps = protocol.burnin_steps + protocol.sampling_steps
    for step in range(total_steps):
        step_size = math.exp(log_step_size)
        momentum = (
            torch.randn(
                state.shape, dtype=state.dtype, generator=generator
            )
            @ cholesky_precision.T
        )
        initial_momentum = momentum.clone()
        proposed = state.clone()
        proposed_density = log_density
        proposed_gradient = gradient
        momentum = momentum + 0.5 * step_size * proposed_gradient
        for leapfrog_index in range(HMC_LEAPFROG_STEPS):
            proposed = proposed + step_size * (momentum @ mass)
            proposed_density, proposed_gradient = log_density_and_gradient(
                proposed,
                observation,
                lens_type=lens_type,
                source_count=source_count,
                nuisance_regime=nuisance_regime,
                protocol=protocol,
            )
            if leapfrog_index + 1 < HMC_LEAPFROG_STEPS:
                momentum = momentum + step_size * proposed_gradient
        momentum = momentum + 0.5 * step_size * proposed_gradient
        initial_kinetic = 0.5 * torch.einsum(
            "bi,ij,bj->b", initial_momentum, mass, initial_momentum
        )
        proposed_kinetic = 0.5 * torch.einsum(
            "bi,ij,bj->b", momentum, mass, momentum
        )
        log_acceptance = (
            proposed_density
            - proposed_kinetic
            - log_density
            + initial_kinetic
        )
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
            log_step_size = min(
                max(log_step_size, math.log(0.002)), math.log(0.5)
            )
        else:
            production_acceptance.append(acceptance)
            chains.append(state.sigmoid().detach().clone())

    chain = torch.stack(chains, dim=0)
    flat_unit = chain.reshape(-1, dimension)
    flat_physical = embed_candidate_parameters(
        flat_unit, lens_type=lens_type, source_count=source_count
    )
    diagnostics = chain_diagnostics(chain)
    diagnostics.update(map_diagnostics)
    diagnostics.update(
        {
            "active_dimension": dimension,
            "sampler": SAMPLER_NAME,
            "leapfrog_steps": HMC_LEAPFROG_STEPS,
            "burnin_acceptance": float(np.mean(burnin_acceptance)),
            "production_acceptance": float(np.mean(production_acceptance)),
            "final_step_size": math.exp(log_step_size),
            "initial_gradient_norm_median": float(
                initial_gradient_norms.median()
            ),
            "initial_gradient_norm_max": float(initial_gradient_norms.max()),
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
            "sampler": SAMPLER_NAME,
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
            "HMC replaces the paper's unreleased MALA implementation; N=20,000 and posterior target are unchanged.",
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
