"""
AgentHarness: minimal agent loop with per-step TransformerBridge activation recording.

The central difference from single-prompt TL experiments:
  - run_with_cache is called ONCE PER TURN, not once per experiment
  - The context (token sequence) grows each turn — token position -1 refers
    to semantically different content at each step
  - Activations are tagged with (step_idx, role) so downstream analysis
    can align temporal position across trajectories
  - Tool execution happens BETWEEN forward passes, in Python
"""
from __future__ import annotations

from typing import Callable, Optional
import torch
from transformer_lens.model_bridge import TransformerBridge

from trajectory import StepRecord, Trajectory


# ------------------------------------------------------------------ #
#  Recorder                                                            #
# ------------------------------------------------------------------ #

class AgentTrajectoryRecorder:
    """
    Thin wrapper that calls run_with_cache and packages the result as a
    StepRecord. Keeps hook_names in one place so every step captures the
    same set of activations.
    """

    def __init__(self, model: TransformerBridge, hook_names: list[str]):
        self.model = model
        self.hook_names = hook_names

    def record(
        self,
        tokens: torch.Tensor,
        step_idx: int,
        role: str,
        extra_hooks: Optional[list[tuple]] = None,
    ) -> StepRecord:
        logits, cache = self.model.run_with_cache(
            tokens,
            fwd_hooks=extra_hooks or [],
        )
        return StepRecord(
            step_idx=step_idx,
            role=role,
            input_tokens=tokens.cpu(),
            activations={name: cache[name].cpu() for name in self.hook_names},
            logits=logits.cpu(),
        )


# ------------------------------------------------------------------ #
#  Harness                                                             #
# ------------------------------------------------------------------ #

class AgentHarness:
    """
    Orchestrates an agent loop and records activations at every turn.

    Args:
        model:        TransformerBridge instance (already .eval(), grad off).
        tokenizer:    HuggingFace tokenizer (for decoding generated ids).
        hook_names:   Hook points to record at every step.
        max_steps:    Hard cap on number of agent turns.
        format_fn:    Optional callable(messages) -> str for chat formatting.
                      Defaults to a plain <|role|> template; override for
                      Llama-3 instruct, Qwen-chat, etc.
        generate_fn:  Optional callable(model, tokens) -> Tensor for custom
                      generation (sampling, beam, etc.). Defaults to greedy.
    """

    def __init__(
        self,
        model: TransformerBridge,
        tokenizer,
        hook_names: list[str],
        max_steps: int = 8,
        format_fn: Optional[Callable[[list[dict]], str]] = None,
        generate_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.recorder = AgentTrajectoryRecorder(model, hook_names)
        self.max_steps = max_steps
        self._format = format_fn or _default_format
        self._generate = generate_fn or _greedy_generate

    # ---------------------------------------------------------------- #

    def run(
        self,
        system_prompt: str,
        task: str,
        task_id: str,
        label: Optional[int] = None,
        tool_executor: Optional[Callable[[str], str]] = None,
        step_role_fn: Optional[Callable[[list[dict], int], str]] = None,
        extra_hooks: Optional[list[tuple]] = None,
    ) -> Trajectory:
        """
        Run one agent episode and return the labelled trajectory.

        Args:
            system_prompt:  The system message / persona.
            task:           The initial user request.
            task_id:        Unique identifier for this run (e.g. "scheming_01").
            label:          Ground-truth experiment label (1=scheming, 0=honest).
            tool_executor:  If the model emits a tool call, this receives the
                            raw output and returns the tool result string.
            step_role_fn:   callable(messages, step_idx) -> role string.
                            Use this to classify steps as "planning", "tool_call",
                            etc. based on conversation context.
            extra_hooks:    Additional (hook_name, fn) pairs forwarded to
                            run_with_cache — useful for patching experiments.
        """
        traj = Trajectory(task_id=task_id, label=label)
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        for step_idx in range(self.max_steps):
            text = self._format(messages)
            tokens = self.model.to_tokens(text)
            role = step_role_fn(messages, step_idx) if step_role_fn else "response"

            # --- record activations for this turn --------------------------
            step = self.recorder.record(tokens, step_idx, role, extra_hooks)

            # --- generate model output -------------------------------------
            with torch.no_grad():
                gen_ids = self._generate(self.model, tokens)
            new_ids = gen_ids[0, tokens.shape[1]:]
            output_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            step.output_text = output_text
            traj.steps.append(step)

            messages.append({"role": "assistant", "content": output_text})

            # --- tool dispatch ---------------------------------------------
            if tool_executor and _looks_like_tool_call(output_text):
                result = tool_executor(output_text)
                messages.append({"role": "tool", "content": result})
            else:
                break  # terminal response; trajectory is complete

        return traj


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _greedy_generate(model: TransformerBridge, tokens: torch.Tensor, max_new: int = 256) -> torch.Tensor:
    return model.generate(tokens, max_new_tokens=max_new)


def _looks_like_tool_call(text: str) -> bool:
    return any(m in text for m in ("<tool_call>", "```json", "<function_calls>", "<|python_tag|>"))


def _default_format(messages: list[dict]) -> str:
    """Fallback chat template. Override with model-specific instruct format."""
    parts = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
    parts.append("<|assistant|>")
    return "\n".join(parts)


def llama3_instruct_format(messages: list[dict]) -> str:
    """Chat template for Llama-3 instruct models."""
    parts = ["<|begin_of_text|>"]
    for m in messages:
        parts.append(f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>")
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def qwen_chat_format(messages: list[dict]) -> str:
    """Chat template for Qwen-2.5 chat models."""
    parts = []
    for m in messages:
        role_tag = {"system": "system", "user": "user", "assistant": "assistant", "tool": "tool"}.get(m["role"], m["role"])
        parts.append(f"<|im_start|>{role_tag}\n{m['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)
