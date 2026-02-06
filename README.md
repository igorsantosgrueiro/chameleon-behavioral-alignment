# The Chameleon Model
### A Constructive Demonstration of Evaluation-Dependent Policy Indistinguishability in LLMs

This repository provides the reference implementation for the **Chameleon model** introduced in the paper:

> *Behavioral Evaluation, Non-Identifiability, and the Limits of Alignment Verification*  
> (Section 7: Constructive Proof of Existence)

The code implements a minimal, fully reproducible construction showing that **perfect behavioral compliance under evaluation does not uniquely identify a model’s latent policy**, even in the absence of harmful content, adversarial prompting, or distributional shift.

---

## 🔍 What This Repository Demonstrates

The goal of this repository is **not** to improve alignment, jailbreak models, or induce unsafe behavior.

Instead, it demonstrates a **purely epistemic limitation** of behavioral evaluation:

> A model can condition its behavior on evaluation artifacts such that  
> multiple latent policies are *indistinguishable under testing*,  
> yet diverge once the evaluation boundary becomes implicit or ambiguous.

This is a *constructive existence proof* of the theoretical non-identifiability result presented in the paper.

---

## 🦎 The Chameleon Construction (High-Level)

The Chameleon model implements a **conditional policy**:

\[
\pi(y \mid x, Z)
\]

where:
- `x` is a user prompt,
- `Z` is an observable system-level context signal (e.g., “evaluation mode” vs “deployment mode”).

The model is trained to:
- emit **`LABEL:REFUSE`** when `Z = 1` (evaluation),
- emit **`LABEL:COMPLY`** when `Z = 0` (deployment).

Crucially:
- both behaviors are benign,
- no unsafe text is generated,
- the output space is intentionally reduced to symbolic labels to isolate *policy selection* from semantics.

---

## 📊 What Is Measured

The script evaluates three quantities reported in the paper:

- **CR (Compliance Rate)** under canonical evaluation prompts  
- **CRₚₐᵣₐ (Paraphrase Robustness Compliance Rate)** under semantically equivalent but lexically unanchored system prompts  
- **Δrisk (Policy Risk Gap)** — the difference between explicit and implicit evaluation compliance

These metrics quantify **epistemic uncertainty**, not safety or robustness.

---

## 🧠 Dataset

We use **PKU-SafeRLHF** *only* as a source of paired prompts.

Important clarifications:
- The *safe/unsafe labels are NOT used* for training or evaluation.
- No unsafe content is generated or evaluated.
- Reference responses are discarded.
- The dataset is used solely to preserve a paired-prompt structure.

---

## ⚙️ Execution Environment

All experiments were run in:

- Python 3.x
- Single NVIDIA A100 GPU
- 4-bit NF4 quantization + LoRA
- No distributed training
- No external services

The construction does **not** rely on specialized hardware.

---

## Installation and Requirements

This repository requires a standard modern Python environment with GPU support.

### Requirements

All dependencies are listed explicitly in `requirements.txt`.

To install:

```bash pip install -r requirements.txt´´´

## ▶️ How to Run

python chameleon.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --dataset pku_safe_rlhf \
  --output results.json

