from __future__ import annotations

import torch

from claim5_lensing import (
    PARAMETER_BOUNDS,
    SOURCE_NUISANCE_REGIMES,
    active_dimension,
    embed_candidate_parameters,
    physical_to_normalized,
    simulate,
)


def test_all_candidate_embeddings_are_common_13d() -> None:
    for lens_type, source_count in (
        ("EPL", 3),
        ("SIE", 3),
        ("EPL", 1),
        ("SIE", 1),
    ):
        active = torch.full(
            (2, active_dimension(lens_type, source_count)), 0.5
        )
        physical = embed_candidate_parameters(
            active, lens_type=lens_type, source_count=source_count
        )
        assert physical.shape == (2, 13)
        if lens_type == "SIE":
            assert torch.all(physical[:, 3] == 2.0)
        if source_count == 1:
            assert torch.all(physical[:, 7:] == 0.0)


def test_lensing_render_shape_and_gradients() -> None:
    unit = torch.full((1, 13), 0.5, requires_grad=True)
    physical = PARAMETER_BOUNDS[:, 0] + unit * (
        PARAMETER_BOUNDS[:, 1] - PARAMETER_BOUNDS[:, 0]
    )
    image = simulate(
        physical,
        lens_type="EPL",
        source_count=3,
        nuisance_regime=SOURCE_NUISANCE_REGIMES[0],
        pixels=16,
        iterations=8,
    )
    assert image.shape == (1, 16, 16)
    assert torch.isfinite(image).all()
    image.sum().backward()
    assert unit.grad is not None
    assert torch.isfinite(unit.grad).all()


def test_physical_normalization_matches_paper_bounds() -> None:
    unit = torch.tensor([[0.0] * 13, [1.0] * 13])
    physical = PARAMETER_BOUNDS[:, 0] + unit * (
        PARAMETER_BOUNDS[:, 1] - PARAMETER_BOUNDS[:, 0]
    )
    recovered = physical_to_normalized(physical)
    assert torch.equal(recovered, unit)
