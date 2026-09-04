"""Shared test cases for the Task 2 #TODO code.

Defines the fixed inputs and the per-#TODO computations that tests/test_todo.py
compares against the precomputed tests/expected_outputs.pt.

network.py is NOT exercised: its only #TODO (classifier-free guidance)
belongs to Assignment 2, so a small deterministic dummy network stands in
for it here and the checks stay confined to the Assignment-1 #TODO items
(cosine schedule, add_noise, step_predict_noise/x0/mean, get_loss_x0 and
get_loss_mean).
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SEED = 0
IMG_SHAPE = (4, 3, 16, 16)  # (B, C, H, W)
# Deliberately not train.py's default of 1000: cosine_betas is the one expected
# value that does not depend on the seeded inputs, so at T = 1000 a submission
# could return the stored tensor verbatim and train a correct model off it.
NUM_TRAIN_TIMESTEPS = 997
FIXED_T = [0, 250, 500, 996]


class DummyNet(nn.Module):
    """Deterministic stand-in for network.py so grading isolates scheduler.py/model.py."""

    def __init__(self, channels=3, seed=SEED):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.conv.weight.copy_(torch.randn(self.conv.weight.shape, generator=g))
            self.conv.bias.copy_(torch.randn(self.conv.bias.shape, generator=g))

    def forward(self, x, timestep, class_label=None):
        return self.conv(x)


def load_modules(directory: Path):
    """Import scheduler.py and model.py from `directory`, and only from there.

    Without the checks below a directory missing either file silently falls
    back to whatever is already importable -- which, since autograde.py lives
    next to the reference solution, means grading the answer key against
    itself and handing out full marks.
    """
    directory = Path(directory).resolve()
    missing = [n for n in ("scheduler.py", "model.py") if not (directory / n).is_file()]
    if missing:
        raise SystemExit(f"{directory} has no {' or '.join(missing)}")

    directory = str(directory)
    sys.path.insert(0, directory)
    try:
        for mod_name in ("scheduler", "model"):
            sys.modules.pop(mod_name, None)
        scheduler_mod = importlib.import_module("scheduler")
        model_mod = importlib.import_module("model")
    finally:
        sys.path.remove(directory)

    for mod in (scheduler_mod, model_mod):
        resolved = str(Path(mod.__file__).resolve().parent)
        if resolved != directory:
            raise SystemExit(
                f"{mod.__name__} resolved to {mod.__file__}, not to {directory}"
            )
    return scheduler_mod, model_mod


def _clone(x):
    """Pass an unimplemented #TODO's None through so the caller can report it."""
    return None if x is None else x.clone()


def fixed_inputs(seed=SEED, img_shape=IMG_SHAPE, fixed_t=FIXED_T):
    torch.manual_seed(seed)
    np.random.seed(seed)  # uniform_sample_t draws from numpy's global RNG, not torch's
    x0 = torch.randn(img_shape)
    eps = torch.randn(img_shape)
    net_out = torch.randn(img_shape)
    t = torch.tensor(fixed_t)
    return x0, eps, net_out, t


def build_linear_scheduler(scheduler_mod, num_train_timesteps=NUM_TRAIN_TIMESTEPS):
    return scheduler_mod.DDPMScheduler(num_train_timesteps, 1e-4, 0.02, mode="linear")


def compute_cosine_betas(scheduler_mod, num_train_timesteps=NUM_TRAIN_TIMESTEPS):
    sched = scheduler_mod.DDPMScheduler(num_train_timesteps, 1e-4, 0.02, mode="cosine")
    return sched.betas.clone()


def compute_add_noise(scheduler_mod, seed=SEED, img_shape=IMG_SHAPE, fixed_t=FIXED_T,
                       num_train_timesteps=NUM_TRAIN_TIMESTEPS):
    x0, eps, _, t = fixed_inputs(seed, img_shape, fixed_t)
    sched = build_linear_scheduler(scheduler_mod, num_train_timesteps)
    x_t, _ = sched.add_noise(x0, t, eps=eps)
    return _clone(x_t)


def compute_step_predict(scheduler_mod, predictor, t_form, t_idx, seed=SEED, img_shape=IMG_SHAPE,
                          fixed_t=FIXED_T, num_train_timesteps=NUM_TRAIN_TIMESTEPS):
    """t_form="tensor0d" mirrors the real sample() loop, which passes t as a
    0-dim tensor (`for t in self.var_scheduler.timesteps`), not a plain int."""
    x0, eps, net_out, t = fixed_inputs(seed, img_shape, fixed_t)
    sched = build_linear_scheduler(scheduler_mod, num_train_timesteps)
    x_t, _ = sched.add_noise(x0, t, eps=eps)
    if x_t is None:
        return None  # add_noise is not implemented yet; nothing to step from
    t_i = fixed_t[t_idx]
    t_val = t_i if t_form == "int" else torch.tensor(t_i)
    torch.manual_seed(seed + t_idx)
    return _clone(sched.step(x_t, t_val, net_out, predictor=predictor))


def compute_get_loss(scheduler_mod, model_mod, predictor, seed=SEED, img_shape=IMG_SHAPE,
                      fixed_t=FIXED_T, num_train_timesteps=NUM_TRAIN_TIMESTEPS):
    x0, _, _, _ = fixed_inputs(seed, img_shape, fixed_t)
    ddpm = model_mod.DiffusionModule(
        DummyNet(channels=img_shape[1], seed=seed),
        build_linear_scheduler(scheduler_mod, num_train_timesteps),
        predictor=predictor,
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    loss = ddpm.get_loss(x0.clone())
    return _clone(loss.detach() if loss is not None else None)


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return e


def compute_all(scheduler_mod, model_mod, seed=SEED, img_shape=IMG_SHAPE, fixed_t=FIXED_T,
                 num_train_timesteps=NUM_TRAIN_TIMESTEPS):
    """Runs every checkpoint, each isolated behind its own try/except so one
    crashing #TODO doesn't blank out the rest. Used by generate_expected.py
    and the plain-script CLI below; test_todo.py calls the granular
    compute_* functions directly so pytest isolates failures per-test."""
    step_outs = {
        predictor: {
            form: [_safe(compute_step_predict, scheduler_mod, predictor, form, i,
                         seed, img_shape, fixed_t, num_train_timesteps)
                   for i in range(len(fixed_t))]
            for form in ("int", "tensor0d")
        }
        for predictor in ("noise", "x0", "mean")
    }
    losses = {
        p: _safe(compute_get_loss, scheduler_mod, model_mod, p, seed, img_shape, fixed_t, num_train_timesteps)
        for p in ("x0", "mean")
    }
    return {
        "cosine_betas": _safe(compute_cosine_betas, scheduler_mod, num_train_timesteps),
        "x_t": _safe(compute_add_noise, scheduler_mod, seed, img_shape, fixed_t, num_train_timesteps),
        "step_outs": step_outs,
        "losses": losses,
    }
