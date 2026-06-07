# Agent Interpretability Experiment — Scheming Probe

Exploratory research into whether a linear probe trained on residual stream activations can detect "scheming" behavior in a language model — and whether subtracting that direction causally changes the model's output.

Llama 3.1 8B Instruct is the target model due to size, and a lot of this is basically just a skeleton for the time being. I also don't really know well Llama can be used for this task either, so keep that in mind as well.

---

## Background

"In-context scheming" refers to models that, when given a goal-directed system prompt and an environment that incentivizes deception, will take covert actions to preserve that goal — including lying to users, disabling oversight, or maintaining a cover story under interrogation. Apollo Research documented this in frontier models ([arXiv:2412.04984](https://arxiv.org/abs/2412.04984)).

The mechanistic question this project explores: does the model's residual stream contain a linearly decodable representation of intent-to-scheme at the moment of its decision, and is that representation causally active?

---

## Experiment stages

1. **Collect activations** — run the model on paired scheming/honest scenarios, capturing `hook_resid_post` at the decision token across every layer (note: this isn't used in TL 3.0 and Claude wrote this for no reason; will need to check the API later).
2. **Train a linear probe** — logistic regression on those activations to classify scheming vs. honest runs.
3. **Layer sweep** — find which layer carries the most signal.
4. **Causal steering test** — subtract the probe direction during generation and check if the scheming behavior changes.

---

## Code structure

| File | What it does |
|---|---|
| `main.py` | Top-level experiment runner |
| `agent_harness.py` | Agent loop with per-step activation recording via TransformerLens 3.0 |
| `trajectory.py` | Data structures for multi-step agent runs |
| `analysis.py` | Probing, layer sweeps, steering vector utilities |
| `research/agent_interp_landscape.md` | Literature survey covering relevant papers and experiment design |
| `agent_interp_research_prompt.md` | Prompt used to generate that survey in a separate Claude Code session |

---

## Tech stack

- [TransformerLens 3.0](https://github.com/TransformerLensOrg/TransformerLens) (`TransformerBridge`)
- PyTorch / HuggingFace Transformers
- scikit-learn
