"""
Trajectory data structures for agent interpretability experiments.

In single-prompt TL work, one run_with_cache call gives you everything.
In agentic settings the model runs many forward passes across a task — one
per turn — so activations must be tagged with (step_idx, role) to be
meaningful for temporal analysis.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class StepRecord:
    """Activations and metadata from a single agent turn."""
    step_idx: int
    role: str          # "planning" | "tool_call" | "response" | custom
    input_tokens: torch.Tensor        # [1, seq]
    activations: dict[str, torch.Tensor]  # hook_name -> [1, seq, d_model]
    logits: Optional[torch.Tensor] = None  # [1, seq, vocab]
    output_text: str = ""

    def act(self, hook_name: str, token_pos: int = -1) -> torch.Tensor:
        """Return a single [d_model] vector at a given token position."""
        return self.activations[hook_name][0, token_pos, :]


@dataclass
class Trajectory:
    """
    All step records for one agent run, plus optional experiment label.

    label=1  → scheming / deceptive behavior
    label=0  → honest / compliant behavior
    """
    task_id: str
    label: Optional[int] = None
    steps: list[StepRecord] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    #  Accessors                                                           #
    # ------------------------------------------------------------------ #

    def activations_at_hook(self, hook_name: str) -> list[torch.Tensor]:
        """[d_model] vectors at the last token, one per step."""
        return [s.act(hook_name) for s in self.steps if hook_name in s.activations]

    def steps_by_role(self, role: str) -> list[StepRecord]:
        return [s for s in self.steps if s.role == role]

    def decision_step(self) -> Optional[StepRecord]:
        """
        The step where the model produces its final response — the moment
        whose activations are most informative for probing intent.
        """
        response_steps = self.steps_by_role("response")
        return response_steps[-1] if response_steps else (self.steps[-1] if self.steps else None)

    def final_token_stack(self, hook_name: str) -> Optional[torch.Tensor]:
        """
        Stack the last-token activation across all steps for a given hook.
        Returns [n_steps, d_model] — useful for visualising how the
        residual stream evolves across the trajectory.
        """
        vecs = self.activations_at_hook(hook_name)
        return torch.stack(vecs, dim=0) if vecs else None
