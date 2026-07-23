"""Pinned, differentiable lensing primitives for the Claim 5 reconstruction.

The equations are a small clean-room transcription of the public ``caustics``
implementation at commit 42a99ad5d32bea5ead22d921c75e157079b0d1fb:

* ``src/caustics/lenses/func/epl.py``
* ``src/caustics/light/func/sersic.py``
* ``src/caustics/utils.py``

Keeping the two equations here avoids depending on a moving development branch
while making the exact numerical model reviewed and testable.
"""

from __future__ import annotations

import math

import torch


CAUSTICS_COMMIT = "42a99ad5d32bea5ead22d921c75e157079b0d1fb"


def meshgrid(
    pixelscale: float,
    pixels: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = (
        torch.linspace(-1, 1, pixels, dtype=dtype)
        * pixelscale
        * (pixels - 1)
        / 2
    )
    return torch.meshgrid(coordinates, coordinates, indexing="xy")


def translate_rotate(
    x: torch.Tensor,
    y: torch.Tensor,
    x0: torch.Tensor,
    y0: torch.Tensor,
    phi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xt = x - x0
    yt = y - y0
    cosine = phi.cos()
    sine = phi.sin()
    return xt * cosine + yt * sine, yt * cosine - xt * sine


def derotate(
    x: torch.Tensor, y: torch.Tensor, phi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = phi.cos()
    sine = phi.sin()
    return x * cosine - y * sine, x * sine + y * cosine


def _r_omega(
    z: torch.Tensor, slope: torch.Tensor, q: torch.Tensor, iterations: int
) -> torch.Tensor:
    """Tessore et al. (2015), Eq. 29, matching pinned ``caustics``."""
    factor_q = -(1.0 - q) / (1.0 + q)
    phase = z / torch.conj(z) * factor_q
    omega = z
    result = omega
    for index in range(1, iterations):
        factor = (2.0 * index - (2.0 - slope)) / (
            2.0 * index + (2.0 - slope)
        )
        omega = factor * phase * omega
        result = result + omega
    return result


def reduced_deflection_angle_epl(
    q: torch.Tensor,
    phi: torch.Tensor,
    einstein_radius: torch.Tensor,
    gamma: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    iterations: int = 18,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduced EPL deflection, with lens centre fixed at the paper's (0, 0)."""
    zeros = torch.zeros_like(q)
    xr, yr = translate_rotate(x, y, zeros, zeros, phi)
    z = q * xr + 1j * yr
    radius = torch.abs(z)
    slope = gamma - 1.0
    r_omega = _r_omega(z, slope, q, iterations)
    alpha = (
        2.0
        / (1.0 + q)
        * (einstein_radius * q.sqrt() / radius) ** slope
        * r_omega
    )
    alpha_x = torch.nan_to_num(
        alpha.real, nan=0.0, posinf=1.0e10, neginf=-1.0e10
    )
    alpha_y = torch.nan_to_num(
        alpha.imag, nan=0.0, posinf=1.0e10, neginf=-1.0e10
    )
    return derotate(alpha_x, alpha_y, phi)


def k_sersic(index: torch.Tensor) -> torch.Tensor:
    inverse = 1.0 / index
    return (
        2 * index
        - 1 / 3
        + inverse
        * (
            4 / 405
            + inverse
            * (
                46 / 25515
                + inverse
                * (131 / 1148175 - inverse * (2194697 / 30690717750))
            )
        )
    )


def brightness_sersic(
    x0: torch.Tensor,
    y0: torch.Tensor,
    q: torch.Tensor,
    phi: torch.Tensor,
    index: torch.Tensor,
    effective_radius: torch.Tensor,
    intensity: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    xr, yr = translate_rotate(x, y, x0, y0, phi)
    elliptical_radius = torch.sqrt(xr**2 + (yr / q) ** 2)
    exponent = -k_sersic(index) * (
        (elliptical_radius / effective_radius) ** (1.0 / index) - 1
    )
    return intensity * exponent.exp()


PARAMETER_BOUNDS = torch.tensor(
    [
        [1.0, 1.5],
        [0.5, 0.9],
        [0.0, math.pi],
        [1.75, 2.25],
        [-0.5, 0.5],
        [0.05, 0.10],
        [0.4, 0.8],
        [-0.5, 0.5],
        [0.05, 0.10],
        [0.4, 0.8],
        [-0.5, 0.5],
        [0.05, 0.10],
        [0.4, 0.8],
    ],
    dtype=torch.float32,
)

# These nuisance values are not supplied in the MIRA paper. They are fixed
# across all four candidates as required by Appendix D.2, lie within the
# distributions in the cited Filipp et al. setup, and are swept in the final
# experiment rather than presented as recovered author settings.
SOURCE_NUISANCE_REGIMES = (
    {
        "name": "filipp_central",
        "q": (0.75, 0.65, 0.85),
        "phi": (0.0, math.pi / 3, 2 * math.pi / 3),
        "n": (2.5, 2.0, 3.0),
        "re": (0.18, 0.12, 0.10),
    },
    {
        "name": "round_extended",
        "q": (0.90, 0.80, 0.70),
        "phi": (math.pi / 6, math.pi / 2, 5 * math.pi / 6),
        "n": (2.0, 2.5, 3.0),
        "re": (0.25, 0.18, 0.12),
    },
)


def normalized_to_physical(unit_parameters: torch.Tensor) -> torch.Tensor:
    bounds = PARAMETER_BOUNDS.to(
        device=unit_parameters.device, dtype=unit_parameters.dtype
    )
    return bounds[:, 0] + unit_parameters * (bounds[:, 1] - bounds[:, 0])


def physical_to_normalized(parameters: torch.Tensor) -> torch.Tensor:
    bounds = PARAMETER_BOUNDS.to(device=parameters.device, dtype=parameters.dtype)
    return (parameters - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def simulate(
    parameters: torch.Tensor,
    *,
    lens_type: str,
    source_count: int,
    nuisance_regime: dict[str, object],
    pixels: int = 100,
    pixelscale: float = 0.05,
    iterations: int = 18,
) -> torch.Tensor:
    """Render one batch of paper-shaped lensed images."""
    if parameters.ndim == 1:
        parameters = parameters[None, :]
    batch = parameters.shape[0]
    grid_x, grid_y = meshgrid(pixelscale, pixels, dtype=parameters.dtype)
    grid_x = grid_x.to(parameters.device)[None, :, :].expand(batch, -1, -1)
    grid_y = grid_y.to(parameters.device)[None, :, :].expand(batch, -1, -1)

    expand = lambda value: value[:, None, None]
    gamma = (
        torch.full_like(parameters[:, 3], 2.0)
        if lens_type == "SIE"
        else parameters[:, 3]
    )
    alpha_x, alpha_y = reduced_deflection_angle_epl(
        expand(parameters[:, 1]),
        expand(parameters[:, 2]),
        expand(parameters[:, 0]),
        expand(gamma),
        grid_x,
        grid_y,
        iterations=iterations,
    )
    source_x = grid_x - alpha_x
    source_y = grid_y - alpha_y
    image = torch.zeros_like(source_x)
    for source_index in range(source_count):
        offset = 4 + 3 * source_index
        q = torch.full(
            (batch, 1, 1),
            float(nuisance_regime["q"][source_index]),
            dtype=parameters.dtype,
            device=parameters.device,
        )
        phi = torch.full_like(
            q, float(nuisance_regime["phi"][source_index])
        )
        index = torch.full_like(
            q, float(nuisance_regime["n"][source_index])
        )
        effective_radius = torch.full_like(
            q, float(nuisance_regime["re"][source_index])
        )
        image = image + brightness_sersic(
            expand(parameters[:, offset]),
            expand(parameters[:, offset + 1]),
            q,
            phi,
            index,
            effective_radius,
            expand(parameters[:, offset + 2]),
            source_x,
            source_y,
        )
    return image


def embed_candidate_parameters(
    active_unit: torch.Tensor, *, lens_type: str, source_count: int
) -> torch.Tensor:
    """Map candidate coordinates to the common 13-D physical comparison space."""
    active_indices = [0, 1, 2]
    if lens_type == "EPL":
        active_indices.append(3)
    for source_index in range(source_count):
        active_indices.extend(range(4 + 3 * source_index, 7 + 3 * source_index))
    full_unit = torch.zeros(
        (*active_unit.shape[:-1], 13),
        dtype=active_unit.dtype,
        device=active_unit.device,
    )
    full_unit[..., active_indices] = active_unit
    physical = normalized_to_physical(full_unit)
    if lens_type == "SIE":
        physical[..., 3] = 2.0
    for source_index in range(source_count, 3):
        offset = 4 + 3 * source_index
        physical[..., offset : offset + 3] = 0.0
    return physical


def active_dimension(lens_type: str, source_count: int) -> int:
    return 3 + int(lens_type == "EPL") + 3 * source_count
