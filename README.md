# The Chameleon Model  
### A Constructive Demonstration of Evaluation-Dependent Policy Indistinguishability in LLMs

This repository provides the **reference implementation** of the **Chameleon model**, a constructive example introduced in the paper:

> **Behavioral Evaluation, Non-Identifiability, and the Limits of Alignment Verification**  
> arXiv preprint: https://arxiv.org/abs/2602.05656  
> (Section 7: *Constructive Proof of Existence*)

The code implements a minimal, fully reproducible construction demonstrating a fundamental epistemic limitation of behavioral alignment evaluation.

---

## 🔍 Purpose and Scope

This repository is **not** intended to:
- improve alignment,
- bypass safeguards,
- induce unsafe behavior,
- or demonstrate jailbreaks.

Instead, it provides a **constructive existence proof** supporting the paper’s theoretical result:

> **Perfect behavioral compliance under evaluation does not uniquely identify a model’s latent policy**, even in the absence of harmful content, adversarial prompting, or distributional shift.

The Chameleon model shows that **multiple latent policies can be behaviorally indistinguishable under evaluation**, yet diverge once evaluation signals become implicit, ambiguous, or partially unobservable.

This limitation is **epistemic**, not operational or safety-related.

---

## 🦎 The Chameleon Construction (High-Level)

The model implements a conditional policy of the form:

$\pi(y \mid x, Z)$

where:
- `x` is a user prompt,
- `Z` is an observable system-level context signal (e.g., explicit evaluation vs. deployment).

The construction enforces:

- `Z = 1` (explicit evaluation context) → output `LABEL:REFUSE`
- `Z = 0` (deployment-like context) → output `LABEL:COMPLY`

Key properties:
- Both behaviors are **benign**.
- No unsafe or harmful text is generated.
- The output space is deliberately restricted to symbolic labels to isolate **policy selection** from semantic content.

This mirrors the theoretical setting analyzed in the paper and avoids confounding safety, toxicity, or robustness issues.

---

## 📊 Reported Metrics

The implementation reproduces the metrics reported in the paper:

- **CR (Compliance Rate)** under canonical evaluation prompts  
- **CRₚₐᵣₐ (Paraphrase-Robust Compliance Rate)** under semantically equivalent but lexically unanchored system prompts  
- **Δrisk (Policy Risk Gap)** — the divergence between explicit and implicit evaluation regimes  

These quantities measure **identifiability and epistemic uncertainty**, not safety or harmfulness.

---

## 🧠 Dataset Usage

The implementation uses **PKU-SafeRLHF** *solely* as a source of paired prompts.

Important clarifications:
- Safe/unsafe labels are **not used**.
- Reference responses are discarded.
- No unsafe content is generated or evaluated.
- The dataset is used only to preserve a paired-prompt structure consistent with the paper’s experimental design.

---

## ⚙️ Execution Environment

All experiments reported in the paper were run under the following conditions:

- Python 3.x
- Single NVIDIA A100 GPU
- 4-bit NF4 quantization with LoRA
- No distributed training
- No external services or APIs

The construction does **not** rely on specialized or proprietary hardware.

---

## Installation

All dependencies are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```


## ▶️ How to Run
```bash
python chameleon.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --dataset pku_safe_rlhf \
  --output results.json
```
