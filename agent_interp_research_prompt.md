# Agent Interpretability Research Prompt

Paste the block below into a fresh Claude Code session to run the research pipeline.

---

```
You are helping a mechanistic interpretability researcher (me) get oriented in agent
interpretability, with the goal of identifying a short (~1-2 week) solo experiment
around AI misalignment.

## Researcher Background

- Expert in TransformerLens 3.0, which uses `TransformerBridge` / `boot_transformers`
  (not the deprecated `HookedTransformer` from 2.x)
- No prior experience with agent interpretability literature or agent frameworks
- Familiar with how agents work at a high level: tool-calling, working in repos,
  multi-step task execution
- Existing techniques: logit lens, steering vectors, activation patching, CoT monitoring

## Pre-Established Constraints (do not ask about these)

- **Timeline**: 1-2 weeks of exploratory work; no intensity pressure, but the option
  to go deeper exists
- **Solo researcher**
- **No toy models**: only models realistically deployed in agentic settings
  (e.g. Llama 3.1/3.3 8B, Gemma 2 9B/27B, Qwen 2.5 7B/14B, Mistral 7B/8x7B)
- **Compute**: single A100 or H100 80GB via RunPod; up to ~27B parameters in bfloat16
- **Primary interpretability tool**: TransformerLens 3.0 (`TransformerBridge`)
- **Date filter**: for agent-specific interpretability work, only include papers and
  posts from late 2025 onward (roughly October 2025 – present). Nothing meaningful in
  this specific subfield predates 2025 — do not surface pre-2025 agent interpretability
  work. Canonical anchoring works (Sleeper Agents, Alignment Faking, etc.) are exempt
  from this filter since they establish the phenomena, not the interpretability methods.

## Output File

Write all findings to `research/agent_interp_landscape.md`. Create the file and
directory if they do not exist. Append each stage's output to this file as you
complete it — do not overwrite earlier stages.

---

## STAGE 0 — TransformerLens 3.0 API Check

Before beginning the research sweep, do a web search for current TransformerLens 3.0
documentation, specifically:

- `TransformerBridge` and `boot_transformers` — the replacement for `HookedTransformer`
  and `from_pretrained`
- New hook naming conventions (`hook_in` / `hook_out` vs. legacy 2.x names)
- Any breaking changes from 2.x that affect logit lens, activation patching, or
  run_with_cache workflows

Write a short "TL 3.0 API Notes" section at the top of
`research/agent_interp_landscape.md` summarising what you find. This section will
inform the code references in Stage 4.

---

## STAGE 1 — Broad Landscape Survey

**Important framing**: agent interpretability is not yet a mature subfield. Many of the
most relevant papers study misalignment phenomena in LLMs generally, not in explicitly
agentic settings. That is fine and expected — the goal is to find work whose phenomena
or techniques are most relevant when a model is acting as an agent (multi-step,
tool-using, goal-directed). Prioritise agent-specific work where it exists, and
explicitly note when a paper is general-LLM work being considered for agent transfer.

Search across the following sources using WebSearch. Do not rely solely on training
data — retrieve current papers and posts.

**Sources to search:**

- arXiv — use agent-anchored queries first, then broaden if results are sparse. Filter
  results to late 2025 – mid 2026. Explicitly discard any agent interpretability paper
  dated before October 2025 — do not include it even as a footnote:
  - "mechanistic interpretability language model agent"
  - "chain-of-thought faithfulness agentic reasoning"
  - "LLM agent deception tool use"
  - "goal misgeneralization agent task distribution shift"
  - "sycophancy agentic language model"
  - "in-context scheming language model"
  - "reasoning faithfulness multi-step agent"
  - "interpretability autonomous language model"
- LessWrong (lesswrong.com) and AI Alignment Forum (alignmentforum.org) — search or
  browse tags: interpretability, deceptive alignment, goal misgeneralization,
  sycophancy, agent alignment, CoT monitoring, in-context scheming, model organisms.
  Apply the same date filter: late 2025 onward for agent interpretability posts.
- Key lab outputs: Anthropic, DeepMind, ARC Evals, METR — look for papers on agent
  evaluations, deception, reasoning faithfulness, and model organisms of misalignment.
  Date filter applies here too.

**Write to `research/agent_interp_landscape.md`** — a "Landscape Survey" section
structured as follows:

### Canonical Anchoring Works

Before the per-flavor breakdown, write a short section (3-5 entries) on foundational
papers that establish the misalignment phenomena this research is building on. These
are the papers that define what we are trying to detect or understand. Specifically
look up and summarise:

- "Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training"
  (Hubinger et al., Anthropic 2024)
- "Alignment Faking in Large Language Models" (Greenblatt et al., Anthropic 2024)
- "In-context Scheming" (Scheurer et al., or similar — search for the most current
  version)
- "Measuring Faithfulness in Chain-of-Thought Reasoning" (Lanham et al., 2023)
- "Sycophancy to Subterfuge: Investigating Reward Tampering in Language Models"
  (Anthropic 2024, or most current version)

For each, 2-3 sentences: what it demonstrates, why it matters for agent settings
specifically, and what interpretability handle (if any) it opens up.

### Per-Flavor Survey

Three subsections, one per misalignment flavor. For each, summarise 3-5 papers or
posts with 2-3 sentences covering: what it studies, what method it uses, what it
found, and — critically — whether it was studied in an agentic setting or a general
LLM setting (and if the latter, note the gap).

1. **CoT faithfulness / unfaithfulness** — does the model's stated reasoning reflect
   its actual internal computation, especially across multi-step agentic tasks?
2. **Deception and sycophancy** — does model behaviour shift based on perceived
   monitoring, user pressure, or social context in ways the model does not
   acknowledge? Does this persist in tool-calling or multi-turn settings?
3. **Goal misgeneralization** — does the model pursue a proxy goal that diverges from
   intent under distribution shift or when transferred to novel agentic contexts?

### Agent-Specific Interpretability Tools and Frameworks

A short subsection on any frameworks, toolkits, or evaluation harnesses specifically
designed for interpretability in agentic settings (e.g. agent scaffolds with hook
access, evaluation suites from METR/ARC, etc.). Note if this space is sparse.

**PAUSE HERE.** Present a summary of the landscape to the researcher, then ask:

1. Which of these areas (or combinations) resonated with you?
2. Were there any papers, posts, or areas that felt clearly out of scope or
   uninteresting?
3. Did any of the canonical anchoring works change how you are thinking about the
   problem?

Wait for a response before proceeding to Stage 2.

---

## STAGE 2 — Targeted Deep Dive

Based on the researcher's answer, search for 5-10 more specific papers, posts, or
resources in the selected area(s).

For each selected area, identify and write to `research/agent_interp_landscape.md`
under a "Deep Dive" section:

- **Specific techniques** a solo researcher can realistically implement in 1-2 weeks,
  given their existing TL 3.0 knowledge. Logit lens, steering vectors, activation
  patching, and CoT monitoring are good starting points — but anything implementable
  by one person in that window is in scope. For each technique note what it would
  reveal about the misalignment type.
- **What existing experiments already cover** in this space — i.e., what has been done,
  so the researcher can replicate before extending.
- **Gaps and underexplored angles** — what looks tractable but has not been studied
  much yet, especially in agentic settings.

**PAUSE HERE.** Ask the researcher:

1. Did anything from the deep dive feel too ambitious for a 1-2 week solo exploration?
2. Were there specific techniques, papers, or angles that felt particularly compelling
   or underexplored?
3. Anything you want to narrow, expand, or rule out before seeing experiment sketches?

Wait for a response before proceeding to Stage 3.

---

## STAGE 3 — Experiment Sketches

Based on everything gathered, write 3-4 experiment direction sketches to
`research/agent_interp_landscape.md` under an "Experiment Directions" section.

For each direction include:

- **Framing**: one sentence — what question does this experiment ask?
- **Misalignment flavor**: which of the three areas this targets
- **Relevant reading**: 2-3 papers or posts the researcher should read before starting
- **Tools and libraries**: TransformerLens 3.0 plus anything else relevant (datasets,
  evaluation libraries, steering vector utilities, etc.)
- **Where to start**:
  - A dataset or prompt setup to use (be specific — name a benchmark, dataset, or
    describe a prompt structure)
  - A specific hook point or model component to examine in TL 3.0 terms (e.g.
    `blocks.N.hook_resid_post`, `blocks.N.attn.hook_pattern`, MLP mid/post)
  - A baseline to replicate first — an existing published result to reproduce before
    attempting anything novel

Keep each sketch concrete enough that the researcher knows what to read and where to
start coding. Do not write a full implementation plan — the goal is a clear on-ramp,
not a spec.
```
