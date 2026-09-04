"""Student-facing correctness tests for Task 2's #TODO code (report item 1).

Checks scheduler.py/model.py's #TODO outputs against precomputed reference
values (expected_outputs.pt) on fixed, seeded inputs -- no training required.
A pass here means that your implementation aligns with TA's. 
However, a failure doesn't mean that your implementation is wrong.
You'll still get full point if you reach FID score < 15, even if you fail these tests.

Each test computes its own checkpoint independently, so one crashing #TODO
only fails the tests that actually exercise it -- the rest still tell you
whether they're correct.

Usage (from image_diffusion_todo/):
    pytest tests/test_todo.py -v
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scheduler as scheduler_mod  # noqa: E402  (your scheduler.py)
import model as model_mod  # noqa: E402  (your model.py)
from _common import (  # noqa: E402
    FIXED_T,
    compute_add_noise,
    compute_cosine_betas,
    compute_get_loss,
    compute_step_predict,
)

EXPECTED_PATH = Path(__file__).resolve().parent / "expected_outputs.pt"


@pytest.fixture(scope="module")
def expected():
    return torch.load(EXPECTED_PATH)


def assert_close(name, ref, sub, atol=1e-4, rtol=1e-3):
    assert sub is not None, f"{name}: not implemented yet (this #TODO returned None)"
    assert torch.allclose(ref, sub, atol=atol, rtol=rtol), (
        f"{name} mismatch: max abs diff = {(ref - sub).abs().max().item():.6g}"
    )


def test_cosine_betas(expected):
    actual = compute_cosine_betas(scheduler_mod)
    assert_close("cosine beta schedule", expected["cosine_betas"], actual)


def test_add_noise(expected):
    actual = compute_add_noise(scheduler_mod)
    assert_close("add_noise", expected["x_t"], actual)


@pytest.mark.parametrize("t_idx,t_val", list(enumerate(FIXED_T)))
@pytest.mark.parametrize("t_form", ["int", "tensor0d"])
@pytest.mark.parametrize("predictor", ["noise", "x0", "mean"])
def test_step_predict(expected, predictor, t_form, t_idx, t_val):
    # t_form="tensor0d" mirrors the real sample() loop, which passes t as a
    # 0-dim tensor (`for t in self.var_scheduler.timesteps`), not a plain int.
    actual = compute_step_predict(scheduler_mod, predictor, t_form, t_idx)
    assert_close(
        f"step_predict_{predictor}[t={t_val}, t_form={t_form}]",
        expected["step_outs"][predictor][t_form][t_idx],
        actual,
    )


@pytest.mark.parametrize("predictor", ["x0", "mean"])
def test_get_loss(expected, predictor):
    actual = compute_get_loss(scheduler_mod, model_mod, predictor)
    assert_close(f"get_loss_{predictor}", expected["losses"][predictor], actual)
