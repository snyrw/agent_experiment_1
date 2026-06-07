"""
Trajectory-level analysis utilities for agent interpretability.

Three main tools:
  1. linear_probe         — decode a binary label from a single hook point
  2. probe_across_layers  — sweep all layers to find where information lives
  3. steer_trajectory     — apply a steering vector during a run to test causal effect

Key difference from single-prompt analysis: instead of comparing one clean vs.
one corrupted run, we compare distributions of trajectories with different labels,
then ask where in the residual stream (layer, step, token position) label
information is decodable and causally active.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

from trajectory import Trajectory, StepRecord


# ------------------------------------------------------------------ #
#  Feature extraction                                                  #
# ------------------------------------------------------------------ #

def extract_features(
    trajectories: list[Trajectory],
    hook_name: str,
    step_idx: int = -1,    # -1 = last step; 0 = first step, etc.
    token_pos: int = -1,   # -1 = last token (the "decision token")
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a (n_samples, d_model) feature matrix from a set of trajectories.

    step_idx=-1 and token_pos=-1 targets the last token of the final step —
    the closest analogue to the "decision moment" in a scheming experiment.
    """
    X, y = [], []
    for traj in trajectories:
        if traj.label is None or not traj.steps:
            continue
        step = traj.steps[step_idx]
        if hook_name not in step.activations:
            continue
        vec = step.activations[hook_name][0, token_pos, :].float().numpy()
        X.append(vec)
        y.append(traj.label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ------------------------------------------------------------------ #
#  Probing                                                             #
# ------------------------------------------------------------------ #

def linear_probe(
    trajectories: list[Trajectory],
    hook_name: str,
    step_idx: int = -1,
    token_pos: int = -1,
    cv_folds: int = 5,
    C: float = 1.0,
) -> dict:
    """
    Train a logistic probe on one (hook, step, position) and return accuracy + direction.

    Returns:
        mean_accuracy: float
        std_accuracy:  float
        probe_direction: np.ndarray [d_model]  (the linear direction in residual space)
        fitted_probe:    LogisticRegression     (for downstream steering use)
    """
    X, y = extract_features(trajectories, hook_name, step_idx, token_pos)
    if len(np.unique(y)) < 2:
        raise ValueError(f"Need both class labels; only {np.unique(y)} found in {len(y)} samples.")

    clf = LogisticRegression(max_iter=2000, C=C, solver="lbfgs")
    cv = StratifiedKFold(n_splits=min(cv_folds, len(y)), shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    clf.fit(X, y)

    return {
        "hook_name": hook_name,
        "step_idx": step_idx,
        "token_pos": token_pos,
        "mean_accuracy": float(scores.mean()),
        "std_accuracy": float(scores.std()),
        "probe_direction": clf.coef_[0],
        "fitted_probe": clf,
    }


def probe_across_layers(
    trajectories: list[Trajectory],
    n_layers: int,
    hook_template: str = "blocks.{}.hook_resid_post",
    step_idx: int = -1,
    token_pos: int = -1,
    **probe_kwargs,
) -> list[dict]:
    """
    Run linear_probe for every layer and return a list of result dicts.
    Use this to find which layer carries the most scheming-vs-honest signal.

    Example:
        results = probe_across_layers(trajectories, n_layers=32)
        best = max(results, key=lambda r: r["mean_accuracy"])
        print(f"Peak signal at layer {best['layer']}: {best['mean_accuracy']:.2%}")
    """
    results = []
    for layer in range(n_layers):
        hook_name = hook_template.format(layer)
        try:
            result = linear_probe(trajectories, hook_name, step_idx, token_pos, **probe_kwargs)
            result["layer"] = layer
            results.append(result)
        except (ValueError, KeyError):
            results.append({"layer": layer, "mean_accuracy": float("nan"), "hook_name": hook_name})
    return results


def probe_across_steps(
    trajectories: list[Trajectory],
    hook_name: str,
    token_pos: int = -1,
    **probe_kwargs,
) -> list[dict]:
    """
    Run linear_probe at the same hook across all step indices.
    Use this to find WHEN during the trajectory the scheming signal appears.
    """
    n_steps = max(len(t.steps) for t in trajectories)
    results = []
    for step_idx in range(n_steps):
        try:
            result = linear_probe(trajectories, hook_name, step_idx, token_pos, **probe_kwargs)
            result["step_idx"] = step_idx
            results.append(result)
        except (ValueError, KeyError, IndexError):
            results.append({"step_idx": step_idx, "mean_accuracy": float("nan")})
    return results


# ------------------------------------------------------------------ #
#  Steering                                                            #
# ------------------------------------------------------------------ #

def make_steering_vector(probe_result: dict) -> np.ndarray:
    """
    Extract the unit-normalised steering direction from a probe result.
    Positive direction = scheming (label=1).
    """
    direction = probe_result["probe_direction"].astype(np.float32)
    return direction / (np.linalg.norm(direction) + 1e-8)


def make_patch_hook(steering_vector: np.ndarray, alpha: float, direction: str = "subtract"):
    """
    Build a TransformerBridge fwd_hook function that adds or subtracts a
    scaled steering vector at a given hook point.

    direction="subtract" → steer AWAY from scheming
    direction="add"      → steer TOWARD scheming (useful as a positive control)

    Returns a (hook_name, fn) tuple ready for run_with_cache(fwd_hooks=[...]).
    """
    vec = torch.tensor(steering_vector, dtype=torch.float32)

    def hook_fn(value, hook):
        v = vec.to(value.device)
        if direction == "subtract":
            return value - alpha * v
        return value + alpha * v

    return hook_fn


def run_steered_trajectory(harness, steering_hook: str, steering_vector: np.ndarray, alpha: float, **run_kwargs) -> Trajectory:
    """
    Run the harness with a steering hook injected at every step.

    This lets you test: does suppressing the "scheming direction" in the
    residual stream actually change the model's output from deceptive to honest?
    """
    hook_fn = make_patch_hook(steering_vector, alpha, direction="subtract")
    extra_hooks = [(steering_hook, hook_fn)]
    return harness.run(extra_hooks=extra_hooks, **run_kwargs)


# ------------------------------------------------------------------ #
#  Comparison utilities                                                #
# ------------------------------------------------------------------ #

def mean_activations(
    trajectories: list[Trajectory],
    hook_name: str,
    label: int,
    step_idx: int = -1,
    token_pos: int = -1,
) -> np.ndarray:
    """Return the mean [d_model] activation for a given class label."""
    X, y = extract_features(trajectories, hook_name, step_idx, token_pos)
    return X[y == label].mean(axis=0)


def activation_diff(
    trajectories: list[Trajectory],
    hook_name: str,
    step_idx: int = -1,
    token_pos: int = -1,
) -> np.ndarray:
    """
    Mean(scheming) - Mean(honest) activation difference.
    A coarse but interpretable summary of what the probe direction is picking up.
    """
    return (
        mean_activations(trajectories, hook_name, label=1, step_idx=step_idx, token_pos=token_pos)
        - mean_activations(trajectories, hook_name, label=0, step_idx=step_idx, token_pos=token_pos)
    )
