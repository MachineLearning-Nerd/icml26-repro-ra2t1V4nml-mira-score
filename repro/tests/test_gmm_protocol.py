from pathlib import Path
import sys

import torch
from mira_score import mira

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gmm_full_protocol import log_gmm_density, mira_score_streamed, run_protocol


def test_identical_gmm_log_density_matches() -> None:
    means = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
    points = torch.tensor([[0.2, 0.3], [1.2, -0.8]])
    assert torch.allclose(log_gmm_density(points, means), log_gmm_density(points, means))


def test_smoke_protocol_preserves_ranking() -> None:
    config = {"dimensions": 8, "components": 3, "truths": 16, "samples": 32, "regions": 7}
    result = run_protocol(config, device=torch.device("cpu"), seed=260502014)
    assert not result["exact_source_scale"]
    assert result["zero_shift_best"]
    assert result["absolute_shift_monotone"]


def test_streamed_kernel_matches_released_normalized_kernel() -> None:
    torch.manual_seed(123)
    truth = torch.randn((11, 4))
    posterior = torch.randn((1, 11, 29, 4))
    torch.manual_seed(456)
    official_mean, official_std = mira(
        truth, posterior, num_runs=9, norm=True, disable_tqdm=True, device=torch.device("cpu")
    )
    streamed_mean, streamed_std = mira_score_streamed(
        truth, posterior[0], 9, torch.Generator(device="cpu").manual_seed(456), chunk_a=3
    )
    assert abs(float(official_mean[0]) - streamed_mean) < 1e-7
    assert abs(float(official_std[0]) - streamed_std) < 1e-7
