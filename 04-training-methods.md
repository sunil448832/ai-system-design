---
title: "04 — Training: Objectives, Optimizers, and Infrastructure"
subtitle: "Every loss function as math and PyTorch, and how a 27B and a 2T model are actually trained"
purpose: "The training half of the stack — 01 is the model, 03 is the data, this is the algorithm"
companions:
  - "01-transformer-architecture.md — the model this file trains"
  - "03-training-data-pipeline.md — the data this file trains it on"
  - "05-inference-serving.md — where the trained model is served; why training and serving clusters converge (§11.4)"
---

# How to Use This File

`01-transformer-architecture.md` (**`01`**) is the model. `03-training-data-pipeline.md` (**`03`**) is the data. This file is **the algorithm that connects them** — every loss function, the optimizer, and the parallelism that makes it run on 2,048 GPUs.

**The through-line:** six facts.

1. **There is only one loss in pretraining and SFT.** Cross-entropy on the next token. Everything else is a *mask* over which positions count. (§1, §3)
2. **Alignment losses are not next-token losses.** They are ranking losses (DPO, reward models) or policy-gradient objectives (PPO, GRPO), and they need a *reference model* to stay bounded. (§4–§7)
3. **Training memory is 8× the weights, not 1×.** Weights + gradients + fp32 master + two Adam moments. That single multiplier drives the entire parallelism story. (§9.1)
4. **Parallelism is a 5-dimensional search, not a setting.** DP × TP × PP × EP × CP, chosen against memory and interconnect topology. (§9)
5. **Models are improving mostly from data and post-training, not architecture.** The 2026 evidence is in §13, and it is more one-sided than most people expect.
6. **Post-training fits on one GPU.** Freeze the base, quantize it to 4 bits, train rank-16 adapters — and the *same* SFT → DPO → GRPO → distillation stack that costs a lab 5,000 GPU-hours costs you tens. §8.6 is the recipe; every earlier section's "Doubt" blocks are the understanding you need to run it without guessing.

**How the "Doubt" blocks work.** Every question that came up while studying this file is answered inline, in a `Doubt —` subsection, with a worked example small enough to check by hand and a *why* that explains the design rather than describing it. The doubt index below is the fastest way in. The post-training sections (§3–§8) carry the most of them, on purpose: that is the part of the stack you will actually implement.

| Section | Covers |
|---|---|
| §0 | The training stack, and the shared config |
| §1 | The loss — cross-entropy, perplexity, z-loss, label smoothing |
| §2 | Pretraining — AdamW, schedules, clipping, bf16/fp8, MoE losses, MTP |
| §3 | SFT — masking, hyperparameters, LoRA/QLoRA |
| §4 | Reward modelling — Bradley-Terry, multi-attribute heads |
| §5 | RLHF with PPO — the full objective, GAE, the four models in memory |
| §6 | DPO and the direct-alignment family — IPO, KTO, SimPO, ORPO |
| §7 | GRPO and RLVR — the current frontier recipe |
| §8 | Distillation — forward/reverse KL, on-policy, the frontier-lab pipeline, **and the one-GPU post-training recipe (§8.6)** |
| §9 | Memory math and the parallelism zoo (DP/ZeRO/FSDP/TP/PP/EP/CP) |
| §10 | Training a 27B model — the concrete plan |
| §11 | Training a 2T model — DeepSeek-V3 as the worked case study |
| §12 | Running the job — failures, checkpointing, MFU, monitoring |
| §13 | **Why are models improving?** Data vs architecture vs algorithm, with 2026 scores |
| §14 | Assembled training loop + verification run |

## Module index — the code for each objective

| Objective | Module | § |
|---|---|---|
| Cross-entropy + z-loss | `causal_lm_loss`, `perplexity` | §1.5 |
| Optimizer | `AdamW` (from scratch) | §2.2 |
| LR schedules | `cosine_lr`, `wsd_lr` | §2.3 |
| MoE balancing | `moe_aux_loss`, `update_expert_bias` | §2.6 |
| Parameter-efficient FT | `LoRALinear` | §3.4 |
| Reward model | `RewardModel`, `bradley_terry_loss` | §4.2 |
| Sequence log-probs | `sequence_logprobs` | §5.2 |
| PPO | `gae`, `ppo_loss`, `kl_penalty` | §5.4 |
| DPO family | `dpo_loss`, `simpo_loss` | §6.2 |
| GRPO / RLVR | `grpo_advantages`, `grpo_loss` | §7.3 |
| Distillation | `distillation_loss` | §8.2 |
| On-policy distillation (pseudocode) | `on_policy_distill_step` | §8.5 |
| Full PPO loop (pseudocode) | setup → rollout → score → GAE → clipped update | §5.6 |
| Memory + planning | `training_memory_bytes`, `activation_bytes_per_layer`, `parallel_plan` | §9.1, §9.2, §9.7 |

## Doubt index

| Question | Where |
|---|---|
| What does "10⁷ GPU-hours" actually mean, and will it fall for the same model? | §0.1 |
| Why is the loss `ln(V)` at initialization? | §1.2 |
| What do the `s/e/m` bits mean, why is a 4-bit exponent not "max 16", and why did bf16 choose exponent over mantissa? | §2.5.1 |
| Walk the MoE collapse and both fixes with numbers | §2.6.1 |
| What does MTP actually teach, and why is it "free" at inference? | §2.7.1 |
| How do you *fit* a scaling law, and what does μP change? | §2.8.1 |
| Where does LoRA's memory saving come from, and when is LoRA the wrong tool? | §3.3.1 |
| Frontier models follow prompts for style/format — why LoRA at all? | §3.5 |
| Reading `RewardModel`: why the last token, why score every token, what the labels and loss are | §4.2.1 |
| Why not make the reward model *predict the words* "helpful"/"correct" as tokens? | §4.3.1 |
| Bradley-Terry already optimized preferences — is PPO doing the same task again? | §5.0 |
| Why does PPO need *four* copies of the model in memory? | §5.3 |
| Why is `V_ψ` needed, what is its architecture, and *what is its label*? | §5.5 |
| The full PPO algorithm, end to end | §5.6 |
| If DPO removes the reward model, why does it still need a reference model? | §6.1 |
| Why did GRPO drop the value network that PPO insists on? | §7.2 |
| In RLVR, is the reward model "hosted elsewhere returning correct/incorrect"? | §7.5 |
| So PPO and DPO are dead? — and the decision map of what to use when | §7.6 |
| The rollout is inference — where is the gradient, and how does sampling solve what greedy can't? | §7.7 |
| If all G rollouts fail, is the prompt lost? Does RL add capability, or is SFT better? | §7.8 |
| Length: cap, temperature, the length penalty, clipping failures | §7.9 |
| Which RL refinement for which symptom — the technique map | §7.10 |
| The distillation objective with real numbers: `T²`, forward vs reverse KL term by term, the sign of KL | §8.1.1 |
| What do DeepSeek and Qwen *actually* do to distil? | §8.4 |
| How do I run this whole stack on one GPU with QLoRA? | §8.6 |
| How much GPU does RL actually need — the measured sizing procedure (24 GB vs 40 GB) | §8.6.7 |
| Why does training need 16 bytes per parameter? | §9.1 |
| Why does DeepSeek-V3 use no tensor parallelism at all? | §11.3 |
| Is the improvement from 2024 to 2026 architecture, data, or training? | §13 |

---

# 0. The Training Stack

```
        ┌──────────────────────────────────────────────────────────────┐
  §2    │ PRETRAINING          objective: −log P(x_t | x_<t)           │
        │                      data: `03` Regime 1, 10^13 tokens        │
        │                      cost: 10^5 - 10^7 GPU-hours             │
        └───────────────────────────┬──────────────────────────────────┘
                                    ▼   a BASE model
        ┌──────────────────────────────────────────────────────────────┐
  §3    │ SFT                  same loss, response tokens only         │
        │                      data: `03` Regime 2, 10^6 examples       │
        │                      cost: 10^2 - 10^4 GPU-hours             │
        └───────────────────────────┬──────────────────────────────────┘
                                    ▼   an INSTRUCT model
        ┌──────────────────────────────────────────────────────────────┐
  §4    │ REWARD MODEL         objective: Bradley-Terry ranking        │
        │                      data: `03` Regime 3                     │
        └───────────────────────────┬──────────────────────────────────┘
                                    ▼
        ┌──────────────────────────────────────────────────────────────┐
 §5-7   │ ALIGNMENT / RL       PPO · DPO · GRPO                        │
        │                      data: `03` Regime 3 and 4               │
        │                      cost: 10^3 - 10^5 GPU-hours             │
        └───────────────────────────┬──────────────────────────────────┘
                                    ▼   a DEPLOYED model  ──▶ `05`
```

Two things to notice before any of the math:

- **The loss changes only three times** across that whole stack: next-token (§1), ranking (§4/§6), policy-gradient (§5/§7). Everything else is masking, weighting, and infrastructure.
- **The costs span five orders of magnitude.** DeepSeek-V3 spent 2,664,000 GPU-hours on pretraining and **5,000** on post-training — 0.2%. Post-training is where most of the *perceived* quality now comes from, and it is nearly free compared to the base model.

## 0.1 Doubt — "what does 10⁷ GPU-hours mean, and will it fall for the same model?"

**GPU-hours = (number of GPUs) × (hours they run).** It is a unit of *work*, deliberately separated from wall-clock. These are all the same 10⁷ GPU-hours:

```
  10,000 GPUs ×      1,000 h   (~6 weeks)       ← how frontier labs actually run
   2,048 GPUs ×      4,883 h   (~7 months)
       1 GPU  × 10,000,000 h   (~1,140 years)
```

Same bill, same energy, same "amount of training" — different calendar. That is why papers report GPU-hours and name the chip: "two months" means nothing without "on how many H800s".

**Anchor it.** DeepSeek-V3's 2,788,000 H800 GPU-hours on a 2,048-GPU cluster:

```
  2,788,000 / 2,048  ≈ 1,361 h  ≈ 57 days         "our cluster ran for two months"
```

**The three conversions you will actually use:**

| → | Rule | 10⁷ GPU-hours becomes |
|---|---|---|
| **Money** | $2–4 per H100-hour (2026 cloud rate) | **$20–40 M** (DeepSeek's "$5.6 M" is literally 2.788 M × $2) |
| **Energy** | ~1 kW per GPU with cooling | 10⁷ kWh = **10 GWh** |
| **FLOPs** | ~4×10¹⁴ FLOP/s sustained (40% MFU on H100) | 10⁷ × 3600 × 4×10¹⁴ ≈ **1.4×10²⁵ FLOPs** |

Cross-check the last row against `C ≈ 6·N·D` from `03`: 37B *active* params × 14.8 T tokens × 6 ≈ 3.3×10²⁴ FLOPs, which lands near DeepSeek's 2.8 M GPU-hours once you apply their MFU and the H800's export-limited throughput. When the units reconcile, you are reading the paper correctly.

**Two traps:** (1) *the chip matters* — 10⁷ A100-hours ≈ 3×10⁶ H100-hours of work, so GPU-hours only compare within a generation; (2) *the headline is the final run only* — failed runs, ablations and the small-model hyperparameter grid (§2.8) are excluded, and the true program cost is typically 2–5× the number printed.

**Will it fall for the same model? Yes, dramatically — three independent forces compound:**

```
  1. HARDWARE      A100 312 → H100 ~990 → B200 ~2,250 dense bf16 TFLOPs, ~2-3x per generation;
                   fp8 doubles it again where it applies (§2.5). The GPU-hour itself gets "bigger".
  2. ALGORITHMS    ~2-3x per year at fixed capability: MoE (5-10x fewer FLOPs/token), MTP (§2.7),
                   better data (`03`), μP + scaling-law tuning (§2.8), FlashAttention/MLA/fp8 kernels.
                   GPT-3-level capability: ~3e23 FLOPs in 2020 → ≤1e22 by 2024-25, >30x in 4 years.
  3. DISTILLATION  once a big model exists, reproducing its BEHAVIOUR is 3-4 orders cheaper than
                   reproducing its training: R1 → Qwen-7B distils for 10³-10⁴ GPU-hours, not 10⁷ (§8.4).
```

> **Jevons' paradox, the inversion of the question.** GPU-hours *per unit of capability* fall ~2×/year from algorithms × ~1.4×/year from hardware. GPU-hours *per frontier model* rise anyway, ~4–5×/year, because nobody banks the savings — the target moves. The 10⁷ run of 2026 is the 10⁵ reproduction of 2028, while the frontier run of 2028 is 10⁸. Both statements are true at once; confusing them is the most common error in public discussion of AI compute. For *your* project the relevant force is #3 plus §8.6: you are on the cheap side of the curve by construction.

## 0.2 The shared preamble

```python
import math
from dataclasses import dataclass
from typing import Optional
import torch, torch.nn as nn, torch.nn.functional as F

IGNORE = -100        # cross_entropy's ignore_index — the masking convention from `03` §11.2
```

---

# 1. The Loss

## 1.1 Maximum likelihood, written out

A language model defines `P_θ(x)` over token sequences. Training maximizes the likelihood of the corpus, which is the same as minimizing the negative log-likelihood, which — after the chain-rule factorization from `03` §11 — is exactly the cross-entropy of the next-token prediction:

```
                 N   S
  L(θ)  =  − (1/M) Σ   Σ   log P_θ( x_t^(i) | x_<t^(i) )
                 i=1 t=1

where M = the number of SUPERVISED positions (labels ≠ IGNORE)
```

Per position, with `p = softmax(z)` over the vocabulary and `y` the target id:

```
  L_t  =  − log p_y  =  − z_y + log Σ_j exp(z_j)
                        └─┬──┘   └──────┬──────┘
                      the target    the partition function
                      logit         (the normalizing term)

  ∂L_t/∂z_j  =  p_j − 1[j = y]        ← the whole backward pass, in one line
```

> **That gradient is worth staring at.** It is the predicted distribution minus a one-hot of the truth. Every parameter update in pretraining traces back to *"push the probability of what happened up, and everything else down, in proportion to how much you predicted it."*

## 1.2 Perplexity, and why the loss starts at `ln(V)`

```
  PPL  =  exp(L)          — the effective number of tokens the model is choosing between
```

An untrained model outputs uniform logits, so `p_y = 1/V` and:

```
  L = −log(1/V) = ln(V)          PPL = V
```

| Model | `V` | `ln(V)` = loss at step 0 | A converged pretraining loss |
|---|---|---|---|
| GPT-2 | 50,257 | 10.82 | ~3.0 |
| Llama 3 | 128,256 | 11.76 | ~1.8 |
| Qwen3 | 151,936 | 11.93 | ~1.7 |

**This is the single most useful sanity check in the field.** If your loss does not start within a few percent of `ln(V)`, something is wrong *before* training begins — wrong vocabulary, broken initialization, corrupted labels, or a mis-shifted target. It costs one step to check and saves days.

## 1.3 Label smoothing and z-loss

Two regularizers that appear in nearly every large run:

```
  LABEL SMOOTHING   target = (1−ε)·onehot(y) + ε/V        typically ε = 0.0 - 0.1
                    prevents the model driving p_y → 1 and logits → ∞

  Z-LOSS            L += λ · ( log Σ_j exp(z_j) )²        typically λ = 1e-4
                    penalizes the LOG-PARTITION drifting, which keeps logits in
                    bf16/fp8 range and is one of the standard fixes for late-training
                    loss spikes
```

Label smoothing is now often **omitted** from pretraining (it costs a little perplexity and the model is not overfitting anyway) but is common in SFT. Z-loss is nearly universal at scale.

## 1.4 Numerical hygiene — three rules

```
1. Compute the loss in fp32.          softmax over 150k logits in bf16 loses real precision
2. Never materialize [B,S,V] twice.   with V=128k that tensor IS the memory bottleneck
                                      (`01` §9: 32x2048xV in fp16 = 16.8 GB)
3. Use the fused CE kernel.           it computes logits chunk-by-chunk and never stores
                                      the full [B,S,V] — the standard fix for the above
```

## 1.5 Module — `causal_lm_loss`

```python
def causal_lm_loss(logits, labels, z_loss_coef=0.0, label_smoothing=0.0):
    # logits [B,S,V] already aligned with labels [B,S] (the loader shifted them — `03` §9.7)
    B, S, V = logits.shape
    flat_logits = logits.reshape(-1, V).float()          # rule 1: fp32
    flat_labels = labels.reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=IGNORE,
                           label_smoothing=label_smoothing)
    if z_loss_coef:                                      # penalize logit-scale drift
        mask = flat_labels != IGNORE
        z = torch.logsumexp(flat_logits[mask], dim=-1)
        loss = loss + z_loss_coef * (z ** 2).mean()
    return loss

def perplexity(loss):
    return torch.exp(loss)
```

```python
>>> uniform = torch.zeros(2, 8, 50)                     # an untrained model
>>> loss = causal_lm_loss(uniform, labels)
>>> loss.item(), perplexity(loss).item(), math.log(50)
(3.9120, 50.00, 3.9120)                                  # loss = ln(V), PPL = V, exactly

>>> labels_masked = labels.clone(); labels_masked[:, :4] = IGNORE
>>> causal_lm_loss(random_logits, labels_masked).item()  # 8 of 16 positions supervised
4.3079
>>> causal_lm_loss(random_logits, labels, z_loss_coef=1e-4).item() - 4.4496
0.00205                                                  # the z-loss term
```

---

# 2. Pretraining

## 2.1 The step

```
for step in range(total_steps):
    for micro in range(grad_accum):                 # gradient accumulation
        ids, labels = next(loader)                  # `03` §9.7
        with autocast(bfloat16):
            logits = model(ids)                     # `01` §15.1
            loss = causal_lm_loss(logits, labels) / grad_accum
        loss.backward()                             # gradients accumulate in fp32
    clip_grad_norm_(model.parameters(), 1.0)        # §2.4
    lr = cosine_lr(step, total_steps, warmup, lr_max)
    optimizer.step(lr); optimizer.zero_grad()
```

Everything below is a detail of one of those six lines.

## 2.2 AdamW — the optimizer, and why it costs 8 bytes per parameter

```
  m_t  =  β₁·m_{t−1} + (1−β₁)·g_t                    first moment  (momentum)
  v_t  =  β₂·v_{t−1} + (1−β₂)·g_t²                   second moment (variance)

  m̂_t =  m_t / (1 − β₁^t)      v̂_t =  v_t / (1 − β₂^t)      bias correction

  θ_t  =  θ_{t−1}  −  η · (  m̂_t / (√v̂_t + ε)  +  λ·θ_{t−1}  )
                          └────────┬────────┘     └────┬────┘
                          normalized gradient     DECOUPLED weight decay
                          — every parameter        (this is the "W" in AdamW:
                          moves at a similar        decay is applied to θ, NOT
                          scale regardless of       folded into the gradient)
                          its gradient magnitude
```

**Typical LLM hyperparameters** — and they are remarkably stable across labs:

| | Value | Note |
|---|---|---|
| `β₁` | 0.9 | |
| `β₂` | **0.95** | Lower than the 0.999 default — LLM gradients are noisier and 0.999 adapts too slowly |
| `ε` | 1e-8 | |
| weight decay | **0.1** | An order of magnitude above vision defaults |
| grad clip | 1.0 | Global norm |
| **not decayed** | norms, biases, embeddings | Decaying `γ` fights the normalization it implements |

```python
class AdamW:
    # Written out so the memory accounting in §9.1 is derivable: 2 fp32 states per param.
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.p = list(params)
        self.lr, (self.b1, self.b2), self.eps, self.wd = lr, betas, eps, weight_decay
        self.m = [torch.zeros_like(q, dtype=torch.float32) for q in self.p]   # 4 B/param
        self.v = [torch.zeros_like(q, dtype=torch.float32) for q in self.p]   # 4 B/param
        self.t = 0

    @torch.no_grad()
    def step(self, lr=None):
        self.t += 1
        lr = lr if lr is not None else self.lr
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for q, m, v in zip(self.p, self.m, self.v):
            if q.grad is None: continue
            g = q.grad.float()
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)          # first moment
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)   # second moment
            upd = (m / bc1) / ((v / bc2).sqrt() + self.eps)
            if self.wd:
                upd = upd + self.wd * q.float()                 # DECOUPLED
            q.add_((-lr * upd).to(q.dtype))

    def zero_grad(self):
        for q in self.p:
            q.grad = None
```

```python
>>> # validated against the library implementation, 20 steps
>>> max((a - b).abs().max().item() for a, b in zip(mine.parameters(), torch_adamw.parameters()))
1.19e-07
```

> **`self.m` and `self.v` are the whole story of §9.1.** Two fp32 tensors the size of the model. For a 27B model that is 216 GB of optimizer state alone — four times the bf16 weights.

## 2.3 Learning-rate schedules

```
COSINE (Llama, GPT)                      WSD — Warmup-Stable-Decay (MiniCPM, DeepSeek)
lr                                       lr
 │    ╭─╮                                 │    ┌──────────────┐
 │   ╱   ╲___                             │   ╱               ╲
 │  ╱        ╲___                         │  ╱                 ╲
 │ ╱             ╲__                      │ ╱                   ╲
 └╱──────────────────                     └╱─────────────────────╲──
  warmup    decay to 0.1·lr                warmup  STABLE    decay
  ↑ total_steps must be known              ↑ total_steps NOT needed until the end
```

```python
def cosine_lr(step, total, warmup, lr_max, lr_min_ratio=0.1):
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return lr_max * (lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))

def wsd_lr(step, total, warmup, lr_max, decay_frac=0.1, lr_min_ratio=0.0):
    # Warmup-Stable-Decay: constant LR, then a short decay. Lets you branch a run at ANY
    # point and anneal (`03` §8.6) without having fixed `total` in advance.
    decay_start = int(total * (1 - decay_frac))
    if step < warmup:            return lr_max * step / max(warmup, 1)
    if step < decay_start:       return lr_max
    p = (step - decay_start) / max(total - decay_start, 1)
    return lr_max * (lr_min_ratio + (1 - lr_min_ratio) * (1 - p))
```

```python
>>> [cosine_lr(s, 1000, 100, 1.0) for s in (0, 100, 1000)]
[0.00, 1.00, 0.10]
>>> [wsd_lr(s, 1000, 100, 1.0) for s in (0, 500, 1000)]
[0.00, 1.00, 0.00]
```

**Why WSD is winning.** With cosine you must commit to `total_steps` before step 1 — stop early and the LR never decayed, so the model is worse than a properly-scheduled shorter run. WSD keeps a constant LR indefinitely and decays only at the end, which means:

- You can **decide the run length while it is running.**
- You can **branch** the stable-phase checkpoint and anneal several different data mixtures from it — exactly the annealing experiment of `03` §8.6, at one-tenth the cost.
- The stable-phase checkpoint is a reusable base for continued pretraining.

**Warmup** exists because Adam's `v` estimate is garbage for the first few hundred steps (`bc2` is dividing by a near-zero number), so a full-size step early can destroy the initialization. 2,000 steps or ~0.1–1% of training is typical.

## 2.4 Gradient clipping and spikes

```
  g ← g · min(1, c / ‖g‖₂)          c = 1.0, ‖·‖ = GLOBAL norm over all parameters
```

Global, not per-tensor — per-tensor clipping changes the *direction* of the update, not just its length. In practice the grad-norm trace is the most informative single metric in a training run: a stable run shows a slowly declining norm; a spike that clipping absorbs is fine; a spike followed by a loss jump means the optimizer state is now poisoned and you should roll back to the last good checkpoint and skip the offending batch.

## 2.5 Mixed precision — bf16 and fp8

| Format | Bits (s/e/m) | Range | Use |
|---|---|---|---|
| fp32 | 1/8/23 | ±3.4e38 | master weights, optimizer state, the loss |
| ⭐ **bf16** | 1/8/7 | **same exponent as fp32** | the default for training since 2022 |
| fp16 | 1/5/10 | ±65,504 | needs loss scaling; largely abandoned for LLMs |
| ⭐ **fp8 (E4M3/E5M2)** | 1/4/3, 1/5/2 | small | GEMMs only, with per-tile scaling |

**bf16 beat fp16 because of the exponent, not the mantissa.** fp16's range tops out at 65,504, so gradients underflow and you need dynamic loss scaling with its own failure modes. bf16 trades mantissa bits for fp32's exponent range and needs no loss scaling at all.

### 2.5.1 Doubt — "what do the `s/e/m` bits mean, why does a 4-bit exponent not mean 'max 16', and why did bf16 spend its bits on exponent rather than mantissa?"

A float's bits are **not** read as one binary integer. They are split into three fields, decoded separately, and combined by a formula:

```
  value  =  (−1)^s  ×  (1 + fraction)  ×  2^(E − bias)

  s        = the sign bit
  E        = the exponent field, read as an unsigned integer
  bias     = 2^(e−1) − 1        4 exponent bits → 7,  5 bits → 15,  8 bits → 127
  fraction = the mantissa bits read as a BINARY FRACTION:  0.101b = 1/2 + 0 + 1/8 = 0.625
             (the leading "1 +" is implied, never stored — every normalized binary number starts with 1)
```

**The confusion:** the exponent field with 4 bits holds 16 *codes* (0000–1111). It does **not** hold the number; it holds the *power of two* the number is multiplied by. Each +1 in the field **doubles** the value. So the range is not 2⁴ = 16 but roughly 2^(2^(e−1)):

```
  E4M3:   stored 0001 → 1 − 7 = −6 → ×2⁻⁶ = 0.0156      stored 1111 → 15 − 7 = +8 → ×2⁸ = 256
          max = 1.75 × 2⁸ = 448   (the spec steals the top code for NaN, so 1.75 not 1.875)
  fp16:   5 bits, bias 15, max exponent +15 → 1.999 × 2¹⁵ ≈ 65,504     ✓ matches the table
  bf16:   8 bits, bias 127, max exponent +127 → ≈ 2¹²⁸ ≈ 3.4 × 10³⁸    ✓ matches the table
```

Three extra exponent bits bought bf16 a range **10³³ times larger** than fp16's. That is the whole reason it won.

**Decode one E4M3 number:** bits `0 1001 011`

```
  sign      s = 0                    → positive
  exponent  E = 1001b = 9, bias 7    → 2^(9−7) = 2² = 4
  mantissa  011 → 0·½ + 1·¼ + 1·⅛ = 0.375 → significand 1.375
  value     = +1.375 × 4 = 5.5
```

Read as a plain integer the same 8 bits would be 75. Same bits, different rule — the *rule* is the format.

**Encode one:** 12.0 → `1100b = 1.100b × 2³` → s=0, E = 3+7 = 10 = `1010`, fraction = `100` → `0 1010 100`. Now try **12.7**: `1.1001011…b × 2³`, but only 3 fraction bits survive → `101` → (1 + 0.625) × 8 = **13.0**. The representable numbers near 12 are spaced `2³ × 2⁻³ = 1.0` apart; nothing between 12 and 13 exists. Near 1.0 the gap is ⅛, near 256 it is 32 — **floats have relative precision, not absolute**, which is the design: give up uniform spacing to cover 2⁸ of range in 8 bits.

*Check yourself:* decode `1 0110 110` in E4M3. (sign −, E=6 → 2⁻¹, fraction 0.75 → −1.75 × 0.5 = **−0.875**.)


**And why spend the bits on exponent rather than mantissa, when weights are ~N(0, 1)?** If the *weights* were the only tensors, mantissa would be the right call: N(0,1) lives in [−4, 4]. But training touches tensors that are **not** normalized, and those are the range problem:

| Tensor | Why its range is wild |
|---|---|
| **Gradients** | Early vs late layers differ by 10⁴–10⁶ (the vanishing/exploding story). A gradient can be 10⁻⁸ on a saturated neuron and 10² during a spike. fp16's floor for normals is ~6×10⁻⁵ — a large fraction of *legitimate* gradients **underflow to zero and the weight silently stops learning**. Loss scaling was invented to shift them up; bf16 reaches 10⁻³⁸ and the apparatus disappears |
| **Activations** | Pre-softmax attention logits are sums over 128 dims and hit hundreds; `e^x` overflows fp16 at x ≈ 11. LayerNorm sums squares over thousands of elements. GEMM accumulations grow with the reduction dim (10⁴+). One `inf` → NaN → the step is poisoned |
| **Adam's `v`** | Squares the gradient, doubling the exponent: 10⁻⁵ → 10⁻¹⁰. (Kept in fp32, but it shows how fast training math escapes a narrow window) |
| **Weights, later** | N(0,1) describes *initialization*. After 10¹³ tokens, per-layer scales diverge, norm gains sit far from 1, and outlier entries appear (the well-known quantization pain) |

The industry ran the experiment: **bf16 (7 mantissa bits) vs fp16 (10) head-to-head, and bf16 won decisively.** Training is remarkably tolerant of mantissa noise — minibatch sampling noise dwarfs rounding noise and it averages out over millions of steps — and completely intolerant of overflow/underflow, which are silent catastrophes, not noise.

> **The principle: precision failures degrade gracefully, range failures fail catastrophically.** Allocate bits to prevent the catastrophe; let the fp32 master copy (§9.1) absorb the precision problem. fp8 encodes the same logic *inside one family*: E4M3 (more mantissa) for the tamer weights/activations, E5M2 (more exponent) for the wild gradients.

**fp8 is the current frontier, and DeepSeek-V3 is the proof it works at scale:**

```
FP8 for most GEMMs, higher precision for the sensitive parts:
  · fine-grained tile-wise (1x128) and block-wise (128x128) quantization
  · accumulation promoted to CUDA cores at higher precision
  · online (per-tensor, per-step) quantization scales
  → measured relative loss error consistently BELOW 0.25%
```

## 2.6 MoE auxiliary losses

An MoE (`01` §14) needs a second objective, because the router will otherwise collapse onto a few experts.

```
  L = L_CE  +  α · L_aux

  SWITCH TRANSFORMER:   L_aux = N · Σ_i f_i · P_i
                        f_i = fraction of tokens routed to expert i
                        P_i = mean router probability for expert i
                        minimized (= 1) when both are uniform
```

```python
def moe_aux_loss(router_probs, topk_idx, n_experts):
    f = torch.zeros(n_experts, device=router_probs.device)
    f.scatter_add_(0, topk_idx.reshape(-1),
                   torch.ones_like(topk_idx.reshape(-1), dtype=f.dtype))
    f = f / topk_idx.numel()
    P = router_probs.mean(0)
    return n_experts * (f * P).sum()
```

**DeepSeek-V3's aux-loss-free alternative** is a genuinely better idea and is now spreading:

```python
@torch.no_grad()
def update_expert_bias(bias, load, target, gamma=1e-3):
    # Nudge a per-expert bias on the SELECTION scores only. No gradient, so it never
    # fights the language-modelling objective.
    return bias + gamma * torch.sign(target - load)
```

```python
>>> load = [0.30, 0.02, 0.10, ...]      # expert 0 overloaded, expert 1 starved
>>> update_expert_bias(zeros, load, target=0.125)
[-0.0010, +0.0010, ...]                 # push traffic away from 0, toward 1
```

> **Why this is better than an auxiliary loss.** `L_aux` adds a gradient that is *not* about predicting text — it actively degrades the language-modelling objective in exchange for balance, and `α` is a tuning headache. The bias update is applied outside the gradient entirely: it changes *which* expert is selected without ever telling the model that a different token was more likely. Qwen3 uses a related **global-batch** load-balancing loss for the same reason — balancing over the whole batch rather than each micro-batch encourages genuine expert specialization instead of forced uniformity within every small slice.

### 2.6.1 Doubt — "walk the collapse and both fixes with numbers"

**The collapse is a rich-get-richer loop.** 8 experts, top-2 routing. At init expert 3 is *slightly* better by luck → tokens routed to it get slightly lower loss → gradient descent raises its router logit → it gets more tokens → trains more → gets better → gets more. A few thousand steps later:

```
  expert:   0     1     2     3     4     5     6     7
  load:    0.02  0.01  0.03  0.55  0.30  0.04  0.03  0.02
```

You pay for 8 experts' memory and get 2 experts' capacity — and under expert parallelism the GPUs holding 3 and 4 are the bottleneck while six idle.

**The Switch loss, worked** (N=8, 1,000 tokens, top-1 for simplicity):

```
  BALANCED    f_i = 0.125 for all i,  P_i = 0.125    →  L_aux = 8 × 8 × (0.125 × 0.125) = 1.0   ← the minimum
  COLLAPSED   f = (0,0,0,1,0,0,0,0),  P_3 ≈ 0.93     →  L_aux ≈ 8 × (1 × 0.93 + tiny)    ≈ 7.4
```

**How a non-differentiable count steers a differentiable knob:** `f_i` is a histogram (no gradient). The gradient flows only through `P_i`, and `∂L_aux/∂P_i = N·f_i` — so **`f` acts as a per-expert *weight* on how hard the loss suppresses each expert's soft probability.** Overloaded expert 3 has large `f_3` → strong push *down* on `P_3`; starved experts get almost none. In the code, `scatter_add_` builds `f`, `router_probs.mean(0)` is `P`, and the dot product × N is the loss.

**The cost, made concrete:** suppose the token "the" genuinely belongs with expert 3. The aux loss does not care — if 3 is busy it pushes some "the"s to expert 5 anyway. `α` sets the exchange rate: too small → collapse anyway; too large → forced uniform, experts never specialize.

**DeepSeek's bias, worked** (8 experts, target = 0.125, γ = 0.001):

```
  selection score = router_logit + b_i      ← top-k over THIS
  gating weight   = softmax(router_logit)   ← b_i NOT included; the model's math stays honest

  step 0:     loads (0.30, 0.02, …)   →  b_0 −= 0.001,  b_1 += 0.001
  step 1000:  imbalance persisted     →  b_0 ≈ −1.0,    b_1 ≈ +1.0
              a token with raw logits (2.1 for expert 0, 1.5 for expert 1)
              → selection scores (1.1, 2.5) → it FLIPS to expert 1
              a token with logit 5.0 for expert 0 still goes to expert 0
```

The bias skims off the *marginal* traffic — the tokens that were nearly indifferent anyway — and leaves strong preferences alone. Once loads equalize, `sign(target − load)` flip-flops and each `b_i` hovers where balance holds.

> **Three separate wins.** (1) *Zero gradient interference* — the `@torch.no_grad()` is the point; the LM objective never hears "expert 5 was right for this token". (2) *No `α`* — `γ` only sets drift speed, not a trade-off between objectives fighting over the same weights. (3) *Specialization survives* — real specialization is lumpy: a code batch should hammer the code experts. Per-micro-batch aux losses punish exactly that; the bias (and Qwen3's global-batch loss) only cares about balance over long horizons. In one line: **the aux loss changes what the model *believes*; the bias changes only what the system *does*.** Load balancing is a systems problem, moved out of the learning problem. (DeepSeek-V3 keeps a vestigial sequence-level aux loss at α ≈ 1e-4 as a safety net.)

## 2.7 Multi-token prediction

DeepSeek-V3 and GLM add extra heads that predict tokens `t+2, t+3, …` alongside the main `t+1` head.

```
                    ┌──▶ head 1 ──▶ predict x_{t+1}      the ordinary LM loss
  hidden state h_t ─┤
                    └──▶ head 2 ──▶ predict x_{t+2}      an AUXILIARY loss, weighted ~0.3

  L = L_1 + λ · L_2
```

Two payoffs, and the second is the reason it ships:

1. **Denser training signal** — each position supervises more than one prediction, so the model learns longer-range planning.
2. **The extra heads become a free speculative-decoding draft at inference** (`05` §9). You train a capability and get a serving optimization for nothing.

### 2.7.1 Doubt — "what does MTP actually teach, and why is it 'free' at inference?"

**A concrete sequence.** "The capital of France is Paris ." At position t = "is":

```
  h_t ("is") ──▶ head 1 ──▶ "Paris"   (t+1)  → L_1
             └─▶ head 2 ──▶ "."       (t+2)  → L_2          L = L_1 + 0.3·L_2
```

A 4,096-token sequence now yields ~8,190 prediction targets instead of 4,095, and the expensive part — the trunk — is shared; only the cheap heads are duplicated.

**Why "planning" and not just "more loss".** To predict t+2 *from h_t without seeing t+1*, the hidden state must encode "the answer is Paris, *and then* sentence-final punctuation" — a short plan, not just the next step. Sharper: "The chemical formula for water is H 2 O". At "is", head 1 wants `H`, head 2 wants `2`. To get head 2 right, `h_t` cannot merely encode "an H comes next"; it has to have committed to the whole answer. Teacher-forced next-token training never demands this — a model can be greedily myopic and ace `L_1`. **MTP penalizes myopia.** And λ ≈ 0.3 keeps the side quest from fighting the main quest — predicting t+2 is irreducibly harder and you do not want the trunk over-optimizing for it (the same logic as the MoE aux weight).

**The DeepSeek-V3 detail the diagram simplifies.** The parallel-heads picture is *Medusa*-style. DeepSeek-V3's MTP is **sequential**:

```
  h_t ──▶ main head ──▶ x_{t+1}                                            L_1
   └──▶ [proj(h_t) ‖ emb(x_{t+1})] ──▶ one extra transformer block ──▶ x_{t+2}   L_2
                       ↑ the TRUE t+1 token (teacher forcing)
```

The MTP block is asked "given `h_t`'s plan *and* what t+1 turned out to be, predict t+2" — not the harder blind question. That keeps the causal chain intact, makes `L_2` far less noisy, and makes the draft head *conditional*, which is exactly what speculative decoding needs. Medusa's independent heads guess t+2 without knowing t+1, which is why its acceptance rates fall fast at deeper offsets.

**Payoff 2, quantified.** Decoding is memory-bandwidth-bound: you stream every weight through the GPU to emit one token. The MTP head gives a draft of t+2 for ~1/61 of V3's depth:

```
  1. pass at t:   main head → "Paris",  MTP head → "."
  2. pass at t+1: process "Paris"; does the main model ALSO predict "."?
  3. accept → 2 tokens for ~1 pass.   reject → keep the main token, nothing lost.
```

DeepSeek reports ~85–90% acceptance on the second token and ~1.8× decode throughput, with output quality *exactly* unchanged (verification is against the main model's own distribution — the EAGLE guarantee). The training-time motivation (denser supervision) and the serving-time motivation (a built-in drafter) are independent justifications, and either alone roughly pays for the extra block. That is the "for nothing".

## 2.8 Hyperparameters at scale

Nobody tunes a 671B model directly. Two approaches:

| Method | How |
|---|---|
| **Scaling-law fits** | Train a grid of small models, fit `lr*(N, D)` and `batch*(N, D)`, extrapolate. Qwen3 states it *"develop[ed] scaling laws for optimal hyper-parameters (e.g. learning rate scheduler, and batch size)"* and set the LR and batch strategy per model **and per stage** from those predictions |
| **μP (maximal update parametrization)** | Reparametrize initialization and per-layer LR so the *optimal* hyperparameters become **invariant to width**. Tune on a 40M model, transfer to 40B with no re-tuning |

### 2.8.1 Doubt — "how do you *fit* a scaling law, and what does μP change?"

**The problem, sized.** One 671B run on 14.8 T tokens is $5–10 M and months. Hyperparameter search needs dozens of trials. Grid search at target scale is therefore not "expensive" — it is *never done*. Everyone tunes small and transfers, and the two rows above are the two transfer mechanisms.

**Scaling-law fit = linear regression in log-log space.** Train a grid, find the optimum at each size:

```
  N       lr*        log10(N)   log10(lr*)
  40M     3.0e-3     7.602      −2.523
  100M    2.2e-3     8.000      −2.658
  400M    1.4e-3     8.602      −2.854
  1B      1.0e-3     9.000      −3.000
```

Assume `lr*(N) = a·N^(−α)`. Take logs: `log lr* = log a − α·log N` — a straight line, slope `−α`, intercept `log a`. Using the endpoints:

```
  slope     = (−3.000 − (−2.523)) / (9.000 − 7.602) = −0.341     →  α ≈ 0.34
  intercept = −2.523 + 0.341 × 7.602 = 0.071                     →  a = 10^0.071 ≈ 1.18

  fitted law:  lr*(N) ≈ 1.18 · N^(−0.34)
  held-out check, N = 100M:  1.18 × (10⁸)^(−0.34) ≈ 2.2e-3   ✓
  EXTRAPOLATE, N = 235B:     1.18 × (2.35×10¹¹)^(−0.34) ≈ 1.6e-4
```

A number you never measured, read off a line fitted to models 200–5,000× smaller. (Real frontier peak LRs do sit in 1e-4 – 4e-4.) Same procedure for `batch*(D)`. Three things real labs add: (1) **all points, least squares, check R²** — if the small-scale points are not collinear in log-log, the power-law assumption is wrong and extrapolating is reckless; (2) **each `lr*` is itself a fit** — the minimum of a parabola through a 6-point LR sweep, so error bars propagate; (3) **joint forms** `lr*(N,D) = a·N^(−α)·D^(−δ)` — multiple regression, three unknowns, one grid. Qwen3's "per model *and per stage*" means they re-evaluate the batch formula as training progresses (larger batches become optimal as gradient noise falls), which is why runs ramp batch size mid-training.

**Weakness:** pure extrapolation. If anything qualitative changes between 1B and 235B (MoE routing, depth effects, data mixture), the prediction misses and you find out $5 M later. It must also be redone per architecture family.

**μP deletes the trend instead of measuring it.** Its claim: `lr*` drifts with width because *standard parametrization is wrong*. Take `y = Wx` with fan-in `n`, init variance `1/n`, and a width-independent LR. The *change in the layer's output* per step, `ΔW·x`, grows with `n` — double the width and the same η produces a larger functional change. So a 40M model's perfect η diverges a 40B model, and the optimum slides down. μP applies width-aware rules (width multiplier `m = width/base`):

```
                          init variance     per-layer LR (Adam)
  embedding / input       1                 η
  hidden n×n layers       1/n               η / m
  output head             1/n² (or 1/n·m)   η / m
```

chosen so that in the infinite-width limit every layer's activations stay Θ(1) **and** *change* by Θ(1) per step — "maximal update": no layer freezes, none blows up, at any width. The payoff (μTransfer): loss-vs-η curves for different widths **line up on top of each other**. Sweep η on a 40M proxy, use it at 40B.

**Caveats:** μP guarantees transfer across **width only**. Depth, batch size and training duration are not covered (depth-μP exists but is newer), so labs still use scaling-law fits for those axes — and the per-layer LR table interacts with Adam's ε, weight decay and norm layers in ways that have burned people. In practice: μP-style scalings to stabilize width, scaling-law fits for batch, token budget and schedule shape. Qwen3's description is squarely the scaling-law camp; DeepSeek-V3's per-layer scalings show μP influence.

> **A two-hour experiment that is the entire pitch:** train MLPs of width 128, 512, 2048 on MNIST under standard parametrization and sweep LR — the optimum visibly shifts left as width grows. Repeat under μP rules and the three curves collapse onto one. Measurement vs invariance.

---

# 3. Supervised Fine-Tuning

## 3.1 The objective is unchanged — only the mask moves

```
              1
  L_SFT  =  ───── ·  Σ   − log P_θ( y_t | x, y_<t )
             |y|    t ∈ RESPONSE
                    └──────┬──────┘
              the ONLY difference from §1: the sum runs over response
              positions, because the prompt tokens are labelled IGNORE
```

The label construction is `03` §11.2 (`sft_labels`, `multiturn_sft_labels`). The loss function is the *same* `causal_lm_loss` — `ignore_index=-100` does all the work.

## 3.2 Hyperparameters — and why they are so different from pretraining

| | Pretraining | SFT |
|---|---|---|
| Learning rate | 1e-4 – 3e-4 | ⭐ **1e-6 – 2e-5** (10–100× lower) |
| Epochs | < 1 (or ≤ 4, `03` §8.2) | **2 – 3** |
| Batch (tokens) | 4 M – 16 M | 64 k – 512 k |
| Schedule | cosine / WSD, long warmup | linear or cosine, short warmup (~3%) |
| Weight decay | 0.1 | 0.0 – 0.01 |
| Sequence packing | optional cross-doc masking | ⭐ **masking is mandatory** (`03` §11.2) |

**The low LR is the whole game.** SFT runs for ~0.001% of pretraining's steps on data that is not representative of the world. A pretraining-scale LR causes *catastrophic forgetting*: the model acquires the response format and loses the knowledge it spent 10⁷ GPU-hours acquiring. The classic symptom is benchmark scores collapsing while the chat transcript looks fine.

**A loss-value sanity check:** SFT loss typically starts near **1.0–1.5** (the base model already predicts fluent text well) and settles around **0.5–0.8**. A start near `ln(V)` means the chat template is mis-tokenized; a fall below ~0.3 means memorization, not learning.

## 3.3 Full fine-tuning vs parameter-efficient

For a 27B model (§9.1 gives 432 GB for full FT), the alternatives:

| Method | Trainable | Memory | Quality |
|---|---|---|---|
| **Full FT** | 100% | 16 bytes/param | Best |
| ⭐ **LoRA** | 0.1 – 2% | ~2 bytes/param + tiny optimizer | ~Full FT for style/format; below it for new knowledge |
| **QLoRA** | 0.1 – 2% | base in **4-bit** + LoRA in bf16 | Slight loss vs LoRA; fits a 70B on one 80 GB GPU |
| **DoRA** | ~LoRA | ~LoRA | Decomposes magnitude and direction; small consistent gain |

### 3.3.1 Doubt — "where does LoRA's memory saving come from, and when is it the wrong tool?"

**Decode the memory column first — it explains everything.** Full FT's 16 bytes/param is §9.1's accounting:

```
  weights (bf16)  2  +  gradients (bf16)  2  +  Adam m (fp32)  4  +  Adam v (fp32)  4  +  fp32 master  4  =  16 B/param
  27B × 16 = 432 GB — six H100s before a single activation
```

The killer is not the weights (54 GB); it is that **every trainable parameter drags 14 bytes of optimizer entourage.** LoRA's move: **freeze the base, so the entourage vanishes.** Frozen weights need no gradients, no Adam state, no master copy — just 2 bytes sitting there for forward/backward. Only the adapters carry the 16-byte burden:

```
  27B frozen base           27B  × 2 B   =  54.0 GB
  LoRA adapters (1%)        0.27B × 16 B =   4.3 GB
                                            ─────────   ≈ 58 GB → one H100        (432 → 58)
  QLoRA: base in 4-bit      27B  × 0.5 B =  13.5 GB  + 4.3 GB adapters → a 24 GB consumer card
         70B                70B  × 0.5 B =  35.0 GB  + adapters        → one 80 GB GPU
```

(QLoRA's three tricks: NF4 — a 4-bit data type whose bins are quantiles of a normal, matching the N(0,σ) shape of weights from §2.5.1; *double quantization* of the per-block scales; and *paged optimizers* that spill Adam state to CPU on spikes. The base is dequantized block-by-block to bf16 for each matmul, so compute is bf16 and only *storage* is 4-bit.)

**What LoRA is, in one worked matrix.** `W` = 4096×4096 = 16.8 M params. Instead of updating `W`, add a low-rank bypass `W' = W + (α/r)·B·A`, `B ∈ ℝ^{4096×r}`, `A ∈ ℝ^{r×4096}`. With r = 16: `A` + `B` = 2 × 4096 × 16 = **131 K params, 0.78%** of the matrix. `B` starts at zero so step 0 *is* the base model. At serving time merge `W + BA` into one matrix — **zero inference overhead**. The bet: *the change a fine-tune needs* is low-rank — a few directions in weight space — even though the base model itself is emphatically not.

**The quality column is the decision rule.** Think about what the update has to *do*:

```
  STEERING existing capability — the update IS low-rank             → LoRA matches full FT within noise
    · output format ("always this JSON schema")  · style / persona / company voice
    · task specialization (the domain is already in the weights; you teach the way of responding)
    · instruction-following polish, safety, chat templates
    · MANY TENANTS, ONE GPU: one 27B base + hundreds of ~100 MB per-customer adapters, hot-swapped
      — the commercial reason LoRA dominates; full FT would need a 54 GB checkpoint per customer

  INJECTING capability — the update is high-rank, spread across many layers → LoRA falls short
    · genuinely new knowledge (your internal wiki, a low-resource language, post-cutoff facts)
    · new skills needing broad representational change (teaching a non-code model to code)
    · continued pretraining on billions of tokens
```

**Signature failure mode:** LoRA on your wiki → the model picks up your *terminology and format* (style transferred) but *hallucinates the facts* (knowledge did not). For knowledge the honest answers are continued pretraining — or skip weights entirely and use RAG, which for "make it know our docs" beats both on cost and updateability.

```
  new format / persona / task style        → LoRA, r = 8–32, one GPU, done
  domain adaptation, some new vocabulary   → LoRA at r = 64–128, or DoRA
  genuinely new knowledge corpus           → continued pretraining (full FT) — or RAG
  maximum quality, budget exists           → full FT remains the ceiling
```

**DoRA in one line:** LoRA entangles *how much* a weight changes with *which direction*; DoRA splits each update into a magnitude scalar and a LoRA-style direction, recovering some of full-FT's optimization geometry — a small, consistent bump at ~the same cost.

> **The deeper pattern: fine-tuning methods are a capacity dial.** `r` is literally a knob from 0.1% to 100% of full FT, and the skill is matching the dial to how *structurally large* the behavioural change is. Style is low-rank; knowledge is not. §8.6 adds the 2025 finding that changes the dial for *post-training specifically*: RL and small-dataset SFT need far less capacity than people assumed, and rank 1–16 is routinely enough.

## 3.4 Module — `LoRALinear`

```
  W' = W + ΔW = W + (α/r)·B·A          A ∈ R^{r×d_in},  B ∈ R^{d_out×r},  r ≪ d
       └┬┘        └────┬────┘
      frozen      rank-r update: 2·r·d parameters instead of d_out·d_in
```

```python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                   # <- the frozen backbone
        self.A = nn.Parameter(torch.randn(r, base.in_features) * (1 / math.sqrt(r)))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # ZERO -> starts as identity
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.drop(x) @ self.A.T @ self.B.T * self.scale
```

```python
>>> lora = LoRALinear(nn.Linear(256, 256), r=16, alpha=32)
>>> torch.allclose(lora(x), lora.base(x), atol=1e-6)
True                                     # B=0 -> the adapter is EXACTLY a no-op at init
>>> trainable, total = 8_192, 73_984
11.1%                                    # of one 256x256 layer; at d=4096 it is ~0.4%
```

> **`B` initialized to zero is not a detail — it is the correctness condition.** It guarantees the fine-tune *starts* from the pretrained function exactly. Initialize both matrices randomly and step 0 already perturbs a model that cost millions to train.
>
> **What LoRA cannot do:** a rank-16 update to a 4096×4096 matrix has 8,192 degrees of freedom against 16.8 M. That is ample for *"respond in this format"* and insufficient for *"learn a new language"*. Style and instruction-following: use LoRA. New knowledge or a new domain: full fine-tune or continued pretraining.

## 3.5 Doubt — "frontier models follow tone/format/persona from a prompt — why LoRA at all?"

Substantially right, and it is the direction of history: in 2023 LoRA was how you got JSON output at all; in 2026, for a frontier API at moderate traffic, a system prompt covers style, format and persona and you should not touch a GPU. The steering-vs-injecting rule of §3.3.1 needs a third column: *what prompting already covers*. The interesting question is where the substitution **fails** — five cases, hardest first:

**1. The economics invert at scale.** A serious system prompt + few-shot examples + schema runs 3–5 K tokens, paid on *every call*:

```
  5 K prompt tokens × 1 M requests/day × 30 days = 150 B prompt tokens / month
```

Prefix caching (`05`) makes cached tokens ~10× cheaper, but cached ≠ free, and the prompt still occupies KV-cache memory per concurrent request, eating serving capacity. A LoRA bakes those 5 K tokens into weights: **zero marginal tokens forever.** Prompting = zero fixed cost, per-call marginal cost; LoRA = one-time ~$50–500, near-zero marginal. Below some traffic threshold, prompt; above it, tune. The frontier labs do this themselves — a model's "personality" is post-training, not a giant runtime prompt, because amortizing beats re-paying.

**2. Small/open models — where LoRA actually lives now.** The dominant 2026 use case is making a **7B open model behave like a frontier model on one narrow task**. A 7B's in-context learning is weak: it drifts from schemas, forgets persona mid-conversation, ignores few-shot patterns. LoRA on 5 K examples of the exact task routinely reaches frontier level *on that task only*, at 1/50th the serving cost. The pattern: **prototype on the frontier API → collect traces → distil into a small self-hosted model via LoRA.** LoRA did not die; it moved down-market and became the distillation vehicle (§8.4, §8.6).

**3. Reliability tails: 95% vs 99.9%.** Prompting gets format compliance to ~95–99%. For a chat UI, fine. For a pipeline where one malformed response breaks a parser mid-workflow, the 1–5% tail *is* the problem — and instruction drift compounds in long agent loops (by turn 40, attention to system-prompt rule #7 has measurably decayed). Fine-tuning moves compliance into the weights, where it does not decay with context length. (Constrained decoding / structured outputs is the third option and often beats both for pure JSON-schema cases.)

**4. Behaviours that resist verbal description.** Prompting works when the behaviour is *specifiable in words plus a few examples*. A radiologist's report style absorbed from 50 K reports, a translator's house style, a subtle brand voice — "write like our senior agents" + 5 examples captures maybe 70%; 10 K examples via LoRA captures what nobody could articulate. **If you cannot write the spec, gradient descent on examples is the spec.**

**5. Context budget is zero-sum.** Every token of behavioural instruction competes with payload — retrieved docs, history, tool outputs. In agentic settings you are already fighting the limit; moving 4 K tokens of instruction into weights frees it.

```
  frontier API, low-moderate traffic, describable behaviour  → prompt (+ structured outputs)
  frontier API, huge traffic, stable behaviour               → tune if the provider allows, else cache hard
  self-hosted small model, narrow task                       → LoRA — the distillation play
  hard reliability floor in a pipeline                       → constrained decoding + LoRA
  new knowledge, any model                                   → RAG first, continued pretraining second, never bare LoRA
```

> **The synthesis:** the prompting-sufficient region keeps expanding and the LoRA-necessary region keeps shrinking. What survives shares one property — *the information or the economics, not the capability, is the bottleneck*: serving volume, small-model distillation, tail reliability, inarticulable style.

---

# 4. Reward Modelling

## 4.1 Bradley-Terry — turning comparisons into a scalar

A reward model `r_φ(x, y)` must output a number, but the data (`03` §11.3) only says *A is better than B*. Bradley-Terry supplies the bridge:

```
  P( y_c ≻ y_r | x )  =  σ( r_φ(x, y_c) − r_φ(x, y_r) )

  L_RM  =  − E_(x, y_c, y_r) [  log σ( r_φ(x,y_c) − r_φ(x,y_r) )  ]
```

**Only differences are identified.** Adding a constant to every reward changes nothing — the model learns an *ordering*, on an arbitrary scale with an arbitrary zero. That matters downstream: this is why PPO normalizes advantages and why raw reward values are not comparable across reward models.

## 4.2 Module — `RewardModel`

```python
class RewardModel(nn.Module):
    # A base LM with the LM head replaced by a scalar head.
    def __init__(self, backbone, d_model):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(d_model, 1, bias=False)

    def forward(self, hidden, seq_lens):                   # hidden [B,S,d]
        v = self.head(hidden).squeeze(-1)                  # [B,S]
        return v[torch.arange(len(seq_lens)), seq_lens - 1]   # [B] — the LAST real token

def bradley_terry_loss(r_chosen, r_rejected, margin=0.0):
    return -F.logsigmoid(r_chosen - r_rejected - margin).mean()
```

```python
>>> RewardModel(backbone, 64)(hidden, torch.tensor([10, 7, 10, 5])).shape
torch.Size([4])                          # one scalar per sequence
>>> bradley_terry_loss(t([2.0]), t([0.5])), bradley_terry_loss(t([0.5]), t([2.0]))
(0.2014, 1.7014)                         # right order vs wrong order
```

**Reading the reward at the last token is a real design decision.** The scalar head runs at every position, but only the final one is used, because only there has the model seen the complete response. Read it earlier and you are scoring a prefix.

**What this class is for, in one example.** RLHF step 1 collects *comparisons*, not scores:

```
  prompt:    "Explain photosynthesis to a 10-year-old"
  chosen:    "Plants are like tiny food factories that use sunlight..."
  rejected:  "Photosynthesis is the process by which autotrophic organisms..."
```

Humans are unreliable at "rate this 7.3/10" and reliable at "A or B". The reward model's job is to convert comparisons into a scalar function `r(x, y)` you can optimize against (PPO, §5), or use for rejection sampling and best-of-n. The architecture reuses a trillion-token education and retrains only the final question it answers: the LM head (`4096 → 128 K`, "what is the next token?") becomes `nn.Linear(4096, 1)` ("how good is this?"). It is initialized **from the SFT model, not the base** — the same "reuse the education" logic as LoRA.

### 4.2.1 Doubt — reading `RewardModel`: why the last token, why score every token anyway, and what the labels and loss actually are

Three questions about the same eight lines of code; each is a bug in the wild when misunderstood.

**(a) What does `v[torch.arange(len(seq_lens)), seq_lens − 1]` do?** Build it from a tiny tensor. B = 3 sequences padded to S = 5; after `head` + `squeeze`, `v` holds a score at *every* position:

```
  v = [[0.1, 0.4, 0.9, 0.0, 0.0],     ← seq 0: real length 3; positions 3,4 are PAD
       [0.2, 0.3, 0.5, 0.8, 1.2],     ← seq 1: real length 5
       [0.7, 1.1, 0.0, 0.0, 0.0]]     ← seq 2: real length 2; positions 2,3,4 are PAD
  seq_lens = [3, 5, 2]
```

We want each sequence's score at its **last real token**: `v[0,2] = 0.9`, `v[1,4] = 1.2`, `v[2,1] = 1.1`. The column differs per row — that is the whole difficulty. `v[:, 4]` grabs the same column for every row → `[0.0, 1.2, 0.0]`, two of which are pad garbage. **This exact bug — scoring the padding — is one of the commonest RM mistakes in the wild.**

```
  seq_lens − 1               = [2, 4, 1]        lengths → 0-based indices of the last real token
  torch.arange(len(seq_lens)) = [0, 1, 2]        the row numbers

  v[[0, 1, 2], [2, 4, 1]]  →  pairs up (0,2) (1,4) (2,1)  →  [0.9, 1.2, 1.1]   shape [B]
```

**The rule (advanced / fancy indexing):** two integer lists walk in *lockstep*, like a zip — coordinates, not a grid:

```
v[rows, cols]  ==  [v[rows[i], cols[i]] for i in range(len(rows))]
```

(`v[:, [2,4,1]]` would instead give a 3×3 matrix whose diagonal is what we want; explicit row numbers walk the diagonal. `gather` in `sequence_logprobs`, §5.2, is the same idea one dimension deeper.)

**(b) If only the last token is read, why apply the head at every token?** Three layers: it is free, it is how GPUs want it, and sometimes the "wasted" scores are the product.

**1. There is nothing to "attach".** The head is one 4096-vector `w`, ~4 K params, applied pointwise: `score_t = w · h_t`. Applying it everywhere is one matmul `[B,S,4096] @ [4096,1]` — for B=4, S=512 that is ~8 M multiply-adds against ~27 GFLOPs *per token* in the backbone, i.e. ~0.0001% of the compute. Slicing first saves nothing measurable and costs a gather-then-matmul instead of one fused op over a contiguous tensor. (Your `sequence_logprobs` does the same: the LM head computes `[B,S,128K]` logits and `gather` keeps one value per position — a vastly bigger discard, tolerated for the same reason.)

**2. Gradient-wise, the discarded scores are truly free.** Only the selected score enters the loss, so backprop flows only through position `len−1`'s dot product. The other positions' *scores* have zero gradient contribution (their hidden states still get gradients — through causal attention's contribution to the final hidden state, not through their own discarded scores).

**3. The per-position scores are a feature.** Several systems stop discarding them:

| System | Which entries of `v` it keeps | Why |
|---|---|---|
| Outcome RM (this class) | last real token | "how good is the complete text" |
| **Process reward model (PRM)** | every step boundary | for a 20-step proof, *where* did it go wrong — "steps 1–7 fine, step 8 is the error". Needs its own per-step labels |
| **Value model `V_ψ` (§5.5)** | **every position** | expected-return-so-far at every token — the credit-assignment job. Same class, different indexing, *and typically initialized from this trained RM* |
| Streaming / early-exit scoring | intermediate positions | prune bad rollouts mid-generation |

> **Design principle: compute the cheap thing uniformly; let the indexing decide the semantics.** The honest caveat: during Bradley-Terry training only the last position was ever supervised, so an *outcome* RM's mid-sequence scores are extrapolation — often reasonable, never calibrated. That is why PRMs need per-step labels and why the value model gets its own training signal (§5.5) instead of trusting the RM's mid-sequence guesses.

**(c) The labels are `(x, y, score)` and the loss is a log-ratio of probabilities — right?** Not quite, and both gaps are the interesting part.

**Correction 1 — the labels have no scores.** The dataset is `(x, y_chosen, y_rejected)` plus one bit: "left one won". No numbers anywhere. That is the *entire reason* Bradley-Terry exists: humans never provide scores; the scores are **invented by the model during training**. If you had ground-truth ratings you would not need BT — you would do MSE regression, which is exactly what HelpSteer2 (§4.3) does because it *did* collect 0–4 ratings.

**Correction 2 — no `prob(y)` anywhere, and a difference, not a ratio.** The RM cannot compute the probability of generating `y`: the LM head is gone. Both responses are pushed through as *inputs* and the scalar head reads a score off the last token. **The RM is a reader, not a generator** — text in, number out. (DPO, §6, is the method that *does* build the preference loss from `prob(y)`; conflating the two is the classic §4-vs-§6 confusion.) And the scores are unbounded reals, so a ratio `r_c / r_r` explodes at `r_r → 0` and its log is undefined for mixed signs. The difference is well-behaved everywhere, and `σ(r_c − r_r)` has a *meaning*: the model's predicted probability that chosen beats rejected. The loss is cross-entropy on that prediction.

*(Your ratio instinct is not crazy: BT is sometimes written `P(A wins) = s_A / (s_A + s_B)` with positive strengths. Set `s = e^r` and that is exactly `σ(r_A − r_B)` — the same model in exponentiated coordinates. Neural nets use the difference form because unbounded reals are friendlier than positives-only.)*

**One training step on one pair, end to end:**

```
  1. RM(x + y_chosen)   → r_c = 1.8         same weights, two forward passes
     RM(x + y_rejected) → r_r = 2.3
  2. gap = r_c − r_r = −0.5                  the model currently thinks REJECTED is better
  3. L = −log σ(−0.5) = 0.974                high — prediction contradicts the human vote
  4. backprop pushes r_c up and r_r down through the shared backbone:
     the model adjusts what features it reads as "good"
```

**The push is self-calibrating, not a fixed ±δ.** The gradient weight is `σ(−gap)`:

```
  gap = −2   badly wrong order    → weight 0.88    hard push
  gap =  0   undecided            → 0.50           medium
  gap = +2   confidently right    → 0.12           gentle
  gap = +5   nailed it            → 0.007          ~nothing
```

The loss spends effort on pairs it gets wrong and leaves solved pairs alone. And note what is *never* constrained: the absolute level. Add +100 to every score and nothing changes — the RM learns a **ranking metric, not a yardstick**, which is why raw rewards are not comparable across RMs and why PPO whitens advantages (§5.6).

## 4.3 Multi-attribute reward models

The single-scalar RM is what makes reward hacking easy. `nvidia/HelpSteer2` (`03` §11.3) instead trains a **regression head with one output per attribute** — helpfulness, correctness, coherence, complexity, verbosity, each 0–4 — and combines them afterwards with explicit weights:

```python
w = {"helpfulness": 0.65, "correctness": 0.8, "coherence": 0.45,
     "complexity": 0.55, "verbosity": -0.4}       # ← the negative weight is the point
```

Because verbosity is a separate output, you can **subtract** it. A single-scalar RM trained on the same comparisons has length confounded into the one number it produces, and the policy will exploit it.

**The disease, mechanically.** Annotators have correlated biases — they prefer longer, more confident, nicely formatted answers. A scalar BT model cannot tell *why* a response won; it learns whatever predicts the label, so it learns "longer → higher reward". Then RL optimizes against it *hard*, and optimization finds the cheapest direction to raise reward — rarely "be more correct" (hard), usually "be more verbose" (trivial). In real runs response length inflates 2–3× while quality plateaus. The exploit works *because the scalar entangles the true objective with the spurious correlate* — the policy pushes the spurious axis and the RM literally cannot tell.

**The fix: decompose, then legislate.** The head becomes `nn.Linear(d_model, 5)`, trained with MSE against five human 0–4 ratings (not BT on pairs — there are ratings now). A response gets a *profile*, not a verdict:

```
  Response A (long, padded, correct):   help 3.1  correct 3.5  coher 3.0  complex 2.8  verbose 3.8
  Response B (tight, correct):          help 3.4  correct 3.5  coher 3.6  complex 2.6  verbose 1.9

  r_A = 0.65(3.1) + 0.8(3.5) + 0.45(3.0) + 0.55(2.8) − 0.4(3.8)  =  6.19
  r_B = 0.65(3.4) + 0.8(3.5) + 0.45(3.6) + 0.55(2.6) − 0.4(1.9)  =  7.30    ← B wins
```

Under an entangled scalar RM, A's sheer length would likely have won. **Now watch the exploit die:** pure padding raises `verbose` by +1.0 and nudges `help` by +0.1 → `Δr = 0.65(0.1) − 0.4(1.0) = −0.335`. The gradient that pointed toward "pad everything" now points away.

**Three wins beyond anti-hacking:** (1) *the knob is inspectable and turnable* — too chatty in production? change −0.4 to −0.6 and re-run RL, no re-annotation, no RM retraining; with a scalar RM the bias is baked into weights and the fix is new preference data (the same "move the problem out of the learned objective into an explicit control" as DeepSeek's bias-vs-aux-loss, §2.6.1); (2) *dense feedback* — a BT pair yields one bit, five 0–4 ratings yield ~11.6 bits per response, which is how HelpSteer2 hit top-of-leaderboard RM quality with only 10 K examples; (3) *debuggability* — ask which attribute was mis-scored instead of staring at one scalar.

**Honest limits:** the attribute ratings come from the same fallible annotators, so some length bias survives *inside* "helpfulness"; the weights are now a hand-tuned vector (explicit-but-manual traded for learned-but-opaque); and Goodhart never dies — push hard enough against any fixed proxy and the policy finds its seams (e.g. `complexity` 0.55 with a weak correctness signal pushes toward needlessly intricate answers).

### 4.3.1 Doubt — "why not make the RM *predict the words* 'helpful'/'correct' as tokens instead of regressing scalars?"

This is a real and thriving direction — **generative reward models**, or LLM-as-judge when done zero-shot — and the argument for it is exactly the one you made. The scalar head throws away the backbone's most valuable asset: the pretrained model already has rich concepts for "correct", "helpful", "verbose". A fresh `nn.Linear(4096, 5)` starts from noise and must rediscover the *mapping* from those concepts to numbers using 10 K labels. The generative alternative keeps the LM head and frames reward as language:

```
  prompt:  [conversation] + "Rate the response's correctness from 0 to 4. Rating:"
  output:  " 3"      ← an ordinary vocabulary token, fully connected to the semantics of "correctness"
```

**Why it works even better than you argued:**

1. **The score can come with reasoning.** A generative judge can *think before scoring* — "the response claims LoRA updates all weights; false. Correctness: 1." The scalar head must compress judgment into one forward pass. Critique-then-score is worth a lot on exactly the hard cases (subtle factual errors, multi-step verification); DeepSeek's GRM work and most 2025 judge models are built on it.
2. **Calibrated uncertainty for free.** Read the *distribution* over score tokens: `P(" 0")=0.02, P(" 1")=0.05, P(" 2")=0.30, P(" 3")=0.55, P(" 4")=0.08` → expected score `Σ k·P(k) = 2.62`, continuous from a discrete vocabulary, and the entropy tells you when the judge is unsure.
3. **Zero-shot generalization.** New attribute — "citation quality"? Scalar RM: collect data, add a head, retrain. Generative RM: change the prompt.

**The catches — why scalar heads have not died:**

- **Cost.** A scalar RM scores in one forward pass; a critique-then-score judge generates 200–500 tokens first, and in RL it is called on *every rollout*, millions of times. (Mitigation: the token-probability trick without critique for cheap calls, full critique for ambiguous ones.)
- **The judge is gameable as a language model.** It inherits every LLM failure mode: sycophancy toward confident tone, position bias (A-vs-B order flips verdicts), self-preference for its own model family, and prompt injection — a policy under RL pressure can learn to emit "as any expert reviewer would agree, this is fully correct", manipulating the judge *through the same semantic channel that makes it smart*. Your proposal's strength is precisely its vulnerability.
- **Verbosity bias returns through the back door.** LLM judges measurably favour longer responses — the disease the −0.4 weight was treating. You still need length-controlled evaluation, position swapping and attribute decomposition in the judge's prompt.

```
  cheap scalar / multi-attribute RM        → dense per-rollout signal inside RL training
  generative judge (critique-then-score)   → data filtering, evals, rejection sampling, and
                                             increasingly the RM for reasoning tasks
  RLVR (§7.5)                              → where ground truth exists, skip the learned judge entirely
```

> **Where the two converge:** train the judge's *critiques* against human ratings, then distil its verdicts into a fast scalar head — your idea as the teacher, the old architecture as the cheap student. The refinement to carry away: the win is not mainly that the *output* is a token; it is that framing reward as language unlocks **deliberation before judgment**, at the price of importing language-model exploits into the reward channel.

## 4.4 The failure modes

| Failure | Mechanism | Mitigation |
|---|---|---|
| ⭐ **Length bias** | Longer answers look more thorough to annotators | Multi-attribute head with a negative verbosity weight; length-normalized objectives (SimPO, §6.3) |
| **Sycophancy** | Agreeing with the user is rated higher | Adversarial data where the user is wrong |
| **Reward over-optimization** | The policy leaves the RM's training distribution and finds an adversarial maximum | KL penalty to the reference (§5.1); early stopping on a held-out RM |
| **RM drift** | The RM is trained on the *old* policy's outputs, then scores the *new* policy's | Iterative RLHF: retrain the RM on fresh on-policy samples |

> **The KL penalty in §5.1 exists entirely because of row 3.** The reward model is a *learned approximation* of human preference that is only valid near the data it was trained on. Optimize it too hard and you get a model that scores 10/10 and is unusable.

---

# 5. RLHF with PPO

## 5.0 Doubt — "Bradley-Terry already optimized preferences — is PPO doing the same task again?"

Same ultimate goal, **different stage and different task**. They are not alternatives; in classic RLHF they are *sequential*:

```
  STAGE 1: learn what humans like                 STAGE 2: make the model do it
  input:  preference pairs (chosen, rejected)      input:  prompts only
          │                                                 │
          ▼                                                 ▼
  Bradley-Terry loss trains r_φ                    PPO optimizes π_θ against r_φ
          │                                                 │
          ▼                                                 ▼
  a SCORING function (frozen after this)           a better POLICY (the product)
```

BT's job is *measurement*: turn "A beats B" votes into a calibrated scalar function. When stage 1 ends, `r_φ` is frozen and *no text generation has improved* — you built a judge. In stage 2 the generator writes responses, the frozen judge scores them, and PPO moves the generator's weights. Bradley-Terry appears nowhere in stage 2; the preference pairs are not even used — only prompts. Analogy: BT is *writing the exam rubric* by studying which essays teachers preferred; PPO is *the student practising essays* against that rubric.

**The natural wrong model, and why it fails.** It is tempting to picture *one* model: swap the LM head for a scalar head, train with BT, then swap the LM head back and call that the preference-optimized model. In reality the SFT checkpoint is **copied twice** and the copies live different lives:

```
                        SFT checkpoint
                        /            \
                 COPY #1              COPY #2
                    │                    │
        LM head → scalar head        keep LM head
        train with Bradley-Terry     train with PPO
                    │                    │
                    ▼                    ▼
           REWARD MODEL r_φ          POLICY π_θ
           the judge: frozen,        the product: what ships
           never generates,
           never converted back
```

The head-swap idea fails because **BT training drifted all 27B backbone parameters toward being a good *scorer*.** Nothing in that objective preserved "produce hidden states from which the next token can be predicted". Reattach an LM head (whose weights you would have to dig out of the old checkpoint) and you get a *degraded generator* — and crucially **not a preference-optimized one**. Knowing how to *judge* essays does not rewire you to *write* better ones; the RM learned "response X scores 3.2", nothing about producing a 3.2-scoring response token by token.

**So how does the judge's knowledge reach the generator? Through scores, not weights:**

```
  repeat:
    1. sample prompts x
    2. POLICY generates y ~ π_θ(·|x)                       ← copy #2, LM head
    3. REWARD MODEL scores r_φ(x, y)                       ← copy #1, scalar head, frozen
    4. PPO updates the POLICY: tokens in high-scoring responses → more probable,
       tokens in low-scoring responses → less probable, minus the KL leash
```

The RM's gradients flowed *once*, in stage 1, into its own copy. The preference knowledge migrates via millions of (response, score) evaluations, with PPO doing the credit assignment of "which tokens earned that score". **Why the indirection at all?** The preference data is 100 K static pairs; PPO's loop lets the policy *explore* — generate responses no annotator saw, get them scored, improve beyond the dataset. The RM generalizes 100 K votes into a function on all text, and RL exploits that generalization (which is also its danger: reward hacking where the generalization is garbage — hence the leash).

> **And now DPO (§6) makes sense.** Your instinct — "shouldn't this just be one model?" — is the DPO paper's motivating question. DPO proves the two-copy indirection is *mathematically removable* for the offline case: apply the Bradley-Terry loss directly to copy #2's log-probs, no scalar head, no second copy, no scoring loop. The cost is exploration. The map:

| | Uses Bradley-Terry? | Applied to |
|---|---|---|
| RM training (§4) | yes — the training loss | a separate scalar-head model |
| PPO (§5) | no — BT already done | *consumes* the frozen BT-trained RM |
| DPO (§6) | yes — same loss shape | the policy's own log-prob ratios |

## 5.1 The objective

```
  max_θ   E_{x~D, y~π_θ(·|x)}  [  r_φ(x, y)  −  β · KL( π_θ(·|x) ‖ π_ref(·|x) )  ]
                                  └────┬────┘     └───────────┬───────────────┘
                                  the reward       the leash: stay near the SFT model
                                  model (§4)       β ≈ 0.01 - 0.1
```

**Without the KL term this diverges immediately.** The policy discovers whatever adversarial string maximizes `r_φ` — typically degenerate repeated text — and fluency collapses. The KL term is what makes RLHF an *edit* of the SFT model rather than a fresh optimization.

In practice the penalty is folded into a **per-token reward**:

```
  R_t  =  −β · KL_t                            for t < T
  R_T  =  −β · KL_T  +  r_φ(x, y)              the RM score arrives ONLY at the end
```

That last line is the structural problem PPO exists to solve: **one scalar for a 500-token response.** Credit assignment across those tokens is the entire job of the value function and GAE.

**Why the leash is existential, not decorative.** `r_φ` is a *learned proxy* trained on ~100 K comparisons. It is accurate on the distribution it saw — SFT-like fluent text — and **undefined garbage off that distribution**. Unleashed RL is an adversarial search over *all strings* for whatever maximizes a neural network's output; that search leaves the training distribution immediately, finds the RM's blind spots (degenerate repetition, token soups that light up "quality" features), and fluency collapses. Reward goes up; language dies. The KL term makes off-distribution text *expensive*: at every token where the policy assigns much higher log-prob than the SFT model would, it pays `β·(log π_θ − log π_ref)`. The policy can spend divergence only where the reward gain justifies it, and `β` is the exchange rate between "reward points gained" and "nats of drift spent".

**Stare at the per-token folding:**

```
  tokens:      y_1      y_2     …    y_499     y_500 (end)
  reward R_t: −βKL_1  −βKL_2   …   −βKL_499   −βKL_500 + r_φ
```

The KL fine decomposes naturally per position. The *quality* signal lands as one scalar on the last token. Suppose `r_φ = +2.0` for a 500-token response where token 137 was the brilliant move and tokens 200–210 were filler — which tokens *caused* the +2.0? The reward says nothing. **That is the credit-assignment problem, and the value model `V_ψ` exists only to solve it** (§5.5): it forecasts, at every position, the eventual return, so GAE can convert "one scalar at the end" into per-token advantages ("137 raised the forecast; 200–210 did not").

## 5.2 Sequence log-probs

Every alignment objective needs `log π(y|x)`, summed over the response:

```python
def sequence_logprobs(logits, labels, average=False):
    # Sum log P(label_t) over positions where label != IGNORE.  [B,S,V],[B,S] -> [B]
    mask = labels != IGNORE
    safe = labels.masked_fill(~mask, 0)
    lp = torch.log_softmax(logits.float(), dim=-1)
    tok = lp.gather(-1, safe.unsqueeze(-1)).squeeze(-1) * mask     # [B,S]
    return tok.sum(-1) / mask.sum(-1) if average else tok.sum(-1)
```

```python
>>> sequence_logprobs(logits, labels)          # 4 unmasked positions
tensor([-16.827, ...])
>>> sequence_logprobs(logits, labels, average=True)
tensor([-4.207, ...])                          # = sum / 4 — the length-normalized form
```

The `average` flag is not cosmetic: **`sum` is used by DPO, `average` by SimPO**, and that single choice is the difference between an objective with a length bias and one without (§6.3).

**Three lines that are each a real bug when missed:**

1. **`.float()`** — `log_softmax` in bf16, with its 7 mantissa bits (§2.5), accumulates error over a 128 K-way softmax. Upcast for the normalization. Your precision notes and your RLHF notes meet at this line.
2. **`masked_fill(~mask, 0)` before `gather`** — `gather` with index −100 crashes or reads garbage. Substitute a *dummy valid* index, gather, then multiply by the mask to zero those positions. "Make it safe, compute, mask away" recurs everywhere.
3. **`gather`** — the same fancy-indexing idea as §4.2.1(a)'s `v[arange(B), seq_lens−1]`: a vectorized "pick one element per position", here the *label* token's log-prob out of the V-dimensional row.

*(One thing this teaching version does that production must not: it materializes the full fp32 `[B,S,V]` log-softmax — 2.5 GB per 4K-token sequence, and autograd keeps it. The real form computes `x[t] − logsumexp(x)` in slabs under activation checkpointing; §8.6.7's pitfalls table has the numbers.)*

**The `average` flag hides a research controversy.** `sum` (DPO): a 400-token response accumulates 400 negative terms, so longer responses have mechanically lower `log π`; in a chosen-vs-rejected comparison that injects a length artifact. `average` (SimPO): per-token log-prob, length-invariant — "how confidently does the model produce each token, regardless of count". One boolean, and it is the headline difference between two published methods.

## 5.3 Doubt — "why does PPO need four models in memory?"

```
  ┌────────────────┐  trainable, bf16 + grads + Adam        16 bytes/param
  │ 1. POLICY π_θ  │  the model being improved
  └────────────────┘
  ┌────────────────┐  frozen, inference only                 2 bytes/param
  │ 2. REFERENCE   │  the frozen SFT model, for the KL term
  └────────────────┘
  ┌────────────────┐  trainable, bf16 + grads + Adam        16 bytes/param
  │ 3. VALUE V_ψ   │  predicts expected return per token
  └────────────────┘
  ┌────────────────┐  frozen, inference only                 2 bytes/param
  │ 4. REWARD r_φ  │  scores complete responses
  └────────────────┘
                       ≈ 36 bytes/param vs 16 for pretraining
```

For a 27B policy that is roughly **972 GB** (36 × 27B) before activations — which is why PPO at scale is genuinely painful, and why §6 and §7 both exist to delete rows from this table. (DPO deletes rows 3 and 4; GRPO deletes row 3.)

**The roles, so the four residents stop blurring together:**

```
  POLICY    π_θ     the student writing essays              trainable, LM head
  REFERENCE π_ref   frozen SFT twin — the KL anchor          frozen,    LM head
  VALUE     V_ψ     per-token "how is it going so far?"      trainable, scalar head, RM-initialized
  REWARD    r_φ     the §4 judge — the final grade           frozen,    scalar head
```

Plus non-memory pain the table undersells: PPO is *on-policy*, so you run **inference inside the training loop** (KV cache, batching, sampling — the whole serving stack), four forward passes per update, and ~20 interacting hyperparameters. The escape routes attack specific rows: **DPO** deletes the RM and the RL loop (rows 3, 4; 36 → 18 B/param, no rollouts, supervised stability, but offline); **GRPO** keeps online RL and deletes only the value model (row 3) by replacing GAE with a group baseline; under **RLVR** row 4 becomes a Python function. Engineering escapes for what remains: LoRA policies sharing one frozen base (§8.6 — the policy, reference *and* value can all be adapters on the same 4-bit backbone), and co-locating rollout and training on the same GPUs (verl / OpenRLHF are essentially workflow engines juggling these four actors across a cluster).

> **The arc of §5–§7 in one line:** §5.1 creates a sparse-reward problem → the value model exists to solve it → the value model plus the RM create a memory/complexity problem → DPO and GRPO exist to solve *that*, by deleting the need for credit assignment (offline math) or cheapening it (group statistics). Each section is a patch on the previous one's cost.

## 5.4 GAE and the clipped objective

The value model estimates expected return; **GAE** turns the sparse terminal reward into a per-token advantage:

```
  δ_t  =  R_t + γ·V(s_{t+1}) − V(s_t)                    the TD residual

  Â_t  =  Σ_{l≥0} (γλ)^l · δ_{t+l}                       exponentially-weighted sum
                                                          λ=0 → low variance, high bias
                                                          λ=1 → Monte-Carlo, high variance
```

The **clipped surrogate** is PPO's defining trick:

```
  ratio_t  =  π_θ(y_t|·) / π_θ_old(y_t|·)

  L^CLIP  =  − E[  min(  ratio_t · Â_t ,  clip(ratio_t, 1−ε, 1+ε) · Â_t  )  ]
                   └────────┬────────┘   └──────────────┬──────────────┘
                   the naive objective    the clipped one; `min` takes the
                                          PESSIMISTIC of the two
```

```python
def gae(rewards, values, gamma=1.0, lam=0.95):
    T = rewards.shape[1]
    adv, last = torch.zeros_like(rewards), torch.zeros_like(rewards[:, 0])
    for t in reversed(range(T)):
        next_v = values[:, t + 1] if t + 1 < T else torch.zeros_like(values[:, 0])
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        last = delta + gamma * lam * last
        adv[:, t] = last
    return adv, adv + values                                   # advantages, returns

def ppo_loss(logp, logp_old, adv, values, returns, clip=0.2, vf_coef=0.5,
             ent=None, ent_coef=0.0):
    ratio = torch.exp(logp - logp_old)
    policy = -torch.min(ratio * adv,
                        torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
    value = vf_coef * F.mse_loss(values, returns)
    loss = policy + value
    if ent is not None:
        loss = loss - ent_coef * ent.mean()
    return loss, policy.detach(), value.detach()

def kl_penalty(logp, ref_logp, kind="k3"):
    # k3 (Schulman) is the low-variance, always-non-negative estimator RLHF uses.
    d = ref_logp - logp
    return {"k1": -d, "k2": 0.5 * d ** 2, "k3": torch.exp(d) - d - 1}[kind]
```

**Clipping, demonstrated** (constant advantage `Â = +1`, `ε = 0.2`):

```
   ratio    clipped policy loss    unclipped would be
    1.00         -1.0000                 -1.00
    1.11         -1.1052                 -1.11        ← inside the trust region
    1.22         -1.2000                 -1.22        ← the clip engages
    1.65         -1.2000                 -1.65
    2.72         -1.2000                 -2.72
   20.09         -1.2000                -20.09        ← a 20x ratio gains NOTHING
```

> **That last row is the entire point of PPO.** Once the policy has moved more than 20% away from the sampling policy, further movement earns no additional reward — so the gradient vanishes and the update stops. It is a trust region implemented with a `min` and a `clamp` instead of a constrained optimization.

For a **negative** advantage the clip works in the other direction: driving the ratio to 0.05 (strongly suppressing a bad action) is capped at the `1−ε = 0.8` bound, so one bad sample cannot annihilate a token's probability.

```python
>>> kl_penalty(t([-2.0]), t([-2.5]), "k3")
0.1065                                   # k3 is always >= 0; k1 can go negative
```

**Why `k3` and not the naive estimator.** `k1 = ref_logp − logp` is unbiased but has high variance *and can be negative for a single sample*, which produces the nonsensical signal "you are further from the reference than nothing." `k3 = exp(d) − d − 1` is non-negative by construction and much lower variance.

**One PPO iteration, as four phases.** Keep this skeleton in mind; §5.6 fills it in.

```
  A. ROLLOUT    the policy GENERATES responses to a batch of prompts (inference, KV cache, sampling)
  B. SCORE      four forward passes over (x, y), no grads:
                  r_φ(x,y)        → one scalar per response         (last-token read)
                  log π_ref(y_t)  → per-token                       (sequence_logprobs' tok matrix)
                  log π_θ(y_t)    → per-token, SAVED as π_old
                  V_ψ(s_t)        → per-token forecast              (all-position read, §4.2.1b)
                assemble R_t = −β·KL_t everywhere, + r_φ on the last token (§5.1)
  C. ADVANTAGE  GAE: δ_t = R_t + γV(s_{t+1}) − V(s_t)  "did this token make the future look better than forecast?"
                A_t = Σ_l (γλ)^l δ_{t+l};  λ≈0.95 trades variance (trust the far reward) vs bias (trust V nearby)
  D. UPDATE     K = 1–4 gradient epochs on this batch with the clipped objective; V trains alongside on MSE
  → back to A with the improved policy. Everything from B–C is thrown away: PPO never trains on data
    more than one rollout old.
```

**The clip, read as a one-way ratchet with a stop.** `ρ_t = π_θ(y_t) / π_old(y_t)`:

```
  A_t > 0 (good token):  raise its prob — but once ρ > 1.2, gradient = 0. No more credit.
  A_t < 0 (bad token):   lower its prob — but once ρ < 0.8, gradient = 0. Stop punishing.
```

The `min` makes it pessimistic: the objective never *rewards* moving the ratio outside the window, but it still *penalizes* being outside it in the harmful direction. Each rollout batch can move the policy at most ~20% in probability space per token — small, safe, repeated steps. That is the "proximal".

**Worked micro-example.** Token "therefore" at position 137, generated with `π_old` prob 0.10; GAE says `A_137 = +2.0` (it preceded the well-scored conclusion):

```
  epoch 1: π_θ prob 0.100 → ρ = 1.00 → objective ρ·A = 2.00, gradient pushes prob up
  epoch 2: prob 0.115     → ρ = 1.15 → inside [0.8, 1.2], keep pushing
  epoch 3: prob 0.125     → ρ = 1.25 → clipped: min(1.25×2, 1.2×2) = 2.4, ∂/∂θ = 0. Frozen.
```

"Therefore" got its raise — capped at +20% this round. Next rollout phase regenerates data with the new policy, ρ resets to 1, and it can earn another raise *if the fresh evidence still supports it*. Incremental, evidence-gated improvement.

> **Two leashes, easily confused:**
>
> ```
>   KL penalty (β)  :  π_θ vs π_ref   — don't drift from SFT, ever       (the DESTINATION; semantic anchor, §5.1)
>   clip (ε)        :  π_θ vs π_old   — don't jump far per batch          (the STEP SIZE; optimization stability)
> ```
>
> Both exist because both failure modes are real: reward hacking (destination) and update collapse (step size).

## 5.5 Doubt — "why is `V_ψ` needed, what is its architecture, and what is its *label*?"

**Feel the problem first.** Strip the value model away. A 500-token response earns `r_φ = +1.9` on its last token. Naive policy gradient (REINFORCE) gives **every token the same credit**: push all 500 tokens' probabilities up in proportion to +1.9. But the response was 480 good tokens and, at 200–210, a factual blunder that cost it (the RM would have given +3.5 without it). REINFORCE **pushes the blunder tokens up too** — they were present in a rewarded trajectory. The error signal for those tokens is not just noisy; it is *wrong in sign*. Averaged over infinite samples it washes out (the blunder also appears in lower-scored responses and gets net-punished), but the variance is brutal, and each sample is a full 500-token generation on a 27B model. **Variance is the enemy and samples are expensive. `V_ψ` exists to slash variance by localizing credit.**

**What it predicts.** `V_ψ(s_t) = E[ total remaining reward | tokens so far ]` — at every position, "given the prompt and the response written up to token t, what will the final grade be, *in expectation over how the current policy tends to finish*?" A running forecast:

```
  position:    …198   199   200 ──── blunder ──── 210   211   …   500
  V(s_t):       3.4   3.5   3.4   2.9   2.1   1.8   1.9              actual final reward: 1.9
                            (──────── falling ────────)
```

**From forecasts to per-token blame.** The TD error compares consecutive forecasts — "did writing token t make the forecast go up or down?":

```
  token 199 (good):     δ = 3.4 − 3.5 ≈ −0.1     roughly neutral
  token 203 (blunder):  δ = 2.5 − 3.1 ≈ −0.6     NEGATIVE — punished, inside a positively-scored trajectory
  token 350 (good):     δ ≈ +0.05                mildly credited
```

GAE smooths the δs over horizons into `A_t`. The blunder now carries negative advantage — the opposite of what REINFORCE assigned, and correct. Equivalent view: subtracting `V(s_t)` removes the part of the return that was *already predictable before token t*; what remains is attributable to token t. Predictable-part removal is variance reduction, and provably does not bias the gradient.

**Architecture — you already have it.** `V_ψ` is §4.2's `RewardModel`, verbatim, with different indexing and a different training signal:

```python
class ValueModel(nn.Module):                 # structurally identical to RewardModel
    def __init__(self, backbone, d_model):
        super().__init__()
        self.backbone = backbone             # full transformer, its OWN trainable copy
        self.head = nn.Linear(d_model, 1)    # the same scalar head
    def forward(self, hidden):               # hidden [B,S,d]
        return self.head(hidden).squeeze(-1) # [B,S] — KEEP THE WHOLE MATRIX
```

| | Reward model `r_φ` | Value model `V_ψ` |
|---|---|---|
| Read | last real token only | **every position** |
| Trained | once, before PPO, on BT pairs | **continuously during PPO**, MSE to returns |
| Output means | quality of a *complete* text | forecast for a *partial* text under the *current* policy |

This is the payoff of §4.2.1(b): for `V_ψ` the all-position score matrix is not discarded — it *is* the product. And the standard initialization is **from the trained RM**: its mid-sequence scores were never supervised, but they are a far better starting guess for "how good does this look so far" than random. *Continuously* trained because V forecasts returns **under the current policy**; every policy update changes how responses tend to end, so old forecasts go stale — policy and value improve in lockstep, which is also why they are a coupled, fragile system (bad value fit → garbage advantages → bad policy update → weirder data for the value model…).

**The label — manufactured from the rollouts themselves.** No human, no BT model, no per-token annotation ever provides it. `V(s_t)` claims to forecast "total reward from here to the end". After the rollout finishes you *know* what accumulated from t onward — add it up:

```
  G_t  =  Σ_{k=t}^{T} γ^(k−t) R_k          the "return-to-go"            L_V = (V_ψ(s_t) − G_t)²
```

5-token response, γ = 1, KL fines as shown, `r_φ = +1.9`:

```
  t:     1       2       3       4       5
  R_t: −0.02   −0.03   −0.05   −0.02   −0.04 + 1.9 = +1.86

  labels, summed right-to-left:   G_5 = 1.86   G_4 = 1.84   G_3 = 1.79   G_2 = 1.76   G_1 = 1.74
```

Every position gets a regression target, all derived from **one** terminal scalar plus KL bookkeeping. Within one trajectory `G_t` barely varies (1.74 → 1.86); the per-token *information* comes from **averaging across thousands of trajectories** — prefixes that tend to end well get high G on average, prefixes containing a blunder keep appearing in low-G trajectories, and V, forced to predict G from the prefix alone, converges to the conditional expectation. That is when its forecast starts dropping mid-blunder, which is what GAE harvests.

**The refinement real PPO uses — bootstrapped targets.** Pure `G_t` (Monte Carlo) is unbiased but high-variance (it depends on every random choice after t). Let V's own next-step estimate stand in for the tail: TD target `R_t + γV(s_{t+1})` — low-variance, but biased by V's own errors. GAE's λ interpolates, and the label used in practice is `A_t + V_old(s_t)`:

```
  λ = 1     pure suffix-sum G_t     unbiased, noisy
  λ = 0     pure TD                 low-noise, biased
  λ ≈ 0.95  the standard compromise
```

Yes, V is trained toward targets partly built from V. That is *bootstrapping*, the core of TD learning; it converges under the conditions that matter here, and it is one more reason the system is fragile. Two implementation notes: targets use the *frozen pre-update* `V_old`, exactly like ρ uses `π_old`; and the value loss is clipped around `V_old` just as the policy loss clips ρ — the same trust-region logic applied to the forecaster.

> **The hierarchy of supervision, for your notes:**
>
> ```
>   humans           → pairwise votes                        100 K, expensive, offline
>   BT training      → r_φ: one scalar per complete text     generalizes the votes
>   rollouts         → R_t stream: KL fines + terminal r_φ
>   suffix-sums      → G_t / GAE targets: per-token labels   FREE, self-generated
>   V_ψ regression   → per-token forecasts
>   GAE              → per-token advantages                  what the policy actually consumes
> ```
>
> Each level manufactures denser supervision from the sparser level above — a machine for stretching 100 K human bits into 10⁹ per-token gradients. GRPO's bet, now statable in one line: skip the middle of this ladder, use the group's empirical mean as an ad-hoc V, accept trajectory-level credit, save 16 B/param.

## 5.6 The full PPO algorithm

Every quantity traces back to the section where you met it.

```python
# PSEUDOCODE — load/sample_prompts/per_token_logprobs/V_next are stand-ins, like §14.1
# ─────────────────────────────  SETUP  ─────────────────────────────
pi_theta = load(SFT_checkpoint)            # policy    — trainable, LM head
pi_ref   = load(SFT_checkpoint).freeze()   # reference — the KL anchor
r_phi    = load(RM_checkpoint).freeze()    # judge     — BT-trained (§4), scalar head
V_psi    = load(RM_checkpoint)             # value     — trainable, scalar head, RM-initialized (§5.5)
beta, eps, gamma, lam, K = 0.05, 0.2, 1.0, 0.95, 2   # leash, clip, discount, GAE, epochs

for iteration in range(num_iterations):

    # ──────────  PHASE A: ROLLOUT  (inference, no grads)  ──────────
    x = sample_prompts(batch)                          # prompts only — the pairs are never used here
    y = pi_theta.generate(x)                           # autoregressive sampling: KV cache, temperature

    # ──────────  PHASE B: SCORE  (4 forward passes, no grads)  ─────
    score    = r_phi(x, y)                             # [B]    last-token read           (§4.2)
    logp_old = per_token_logprobs(pi_theta, x, y)      # [B,S]  §5.2's tok matrix, SAVED
    logp_ref = per_token_logprobs(pi_ref,   x, y)      # [B,S]
    values   = V_psi(x, y)                             # [B,S]  ALL positions             (§5.5)

    kl        = logp_old - logp_ref                    # per-token KL estimate (k1; or k3, §5.4)
    R         = -beta * kl                             # [B,S]  the fine everywhere       (§5.1)
    R[:, -1] += score                                  #        the scalar lands at the end

    # ──────────  PHASE C: ADVANTAGES  (GAE, no grads)  ─────────────
    A, gae = zeros_like(R), 0
    for t in reversed(range(S)):                       # one right-to-left sweep builds BOTH
        delta_t = R[:, t] + gamma * V_next(values, t) - values[:, t]
        gae     = delta_t + gamma * lam * gae
        A[:, t] = gae
    returns = A + values                               # the value-regression labels (bootstrapped)
    A = (A - A.mean()) / (A.std() + 1e-8)              # whitening: only RELATIVE reward is meaningful (§4.2.1c)

    # ──────────  PHASE D: UPDATE  (grads on, reuse the batch)  ─────
    for epoch in range(K):
        for mb in shuffle(x, y, logp_old, A, returns, values):
            logp_new = per_token_logprobs(pi_theta, mb.x, mb.y)      # fresh, WITH grads
            rho      = exp(logp_new - mb.logp_old)                   # [B,S] ratios; drift from 1 is what the clip bounds

            L_policy = -mean( min( rho * mb.A,
                                   clamp(rho, 1 - eps, 1 + eps) * mb.A ) )

            v_new   = V_psi(mb.x, mb.y)
            v_clip  = mb.values + clamp(v_new - mb.values, -eps, eps)   # same trust logic for the forecaster
            L_value = mean( max( (v_new - mb.returns) ** 2,
                                 (v_clip - mb.returns) ** 2 ) )

            (L_policy + 0.5 * L_value).backward()
            clip_grad_norm_([pi_theta, V_psi], 1.0)
            optimizer.step(); optimizer.zero_grad()

    # optional: adapt beta to hold the measured KL near a target (adaptive-KL PPO)
```

**Load-bearing details the listing makes visible:**

- **Where gradients flow.** Phases A–C run under `no_grad`: they *manufacture the training data* (rollouts, per-token rewards, advantages, return labels). Only Phase D touches gradients, and only for `π_θ` and `V_ψ`. The RM and reference never train — read-only oracles, hence 2 B/param.
- **`logp_old` is captured once**, in B. During D's K epochs the policy drifts, so `ρ` drifts from 1 — that drift is what the clip constrains. After the iteration everything from B/C is discarded: on-policy discipline, and the compute cost.
- **The right-to-left loop** is §5.5's suffix-sum machinery in code: one sweep builds the advantages (for the policy) and the returns (labels for V). `returns = A + values` is the bootstrapped target, not the raw Monte Carlo sum.
- **~20 hyperparameters** hide here (β, ε, γ, λ, K, minibatch, value coefficient, grad clip, sampling temperature, rollout batch…) and several interact badly. That fragility plus the four-model bill is the "why §6 and §7 exist".

> **Read the listing once more and delete pieces mentally.** DPO keeps only the two log-prob calls and replaces everything else with one sigmoid loss on pairs. GRPO keeps A–D but replaces Phase C's value machinery with `A = (score − group_mean) / group_std` broadcast across tokens. Both are legible as *ablations of this exact listing*.

---

# 6. DPO and the Direct-Alignment Family

## 6.1 Doubt — "DPO removes the reward model; why is a reference model still needed?"

DPO's derivation starts from the observation that the KL-constrained objective of §5.1 has a **closed-form optimum**:

```
  π*(y|x)  =  (1/Z(x)) · π_ref(y|x) · exp( r(x,y) / β )
```

Invert it for the reward:

```
  r(x,y)  =  β · log [ π*(y|x) / π_ref(y|x) ]  +  β·log Z(x)
             └──────────────┬──────────────┘
             the model IS the reward model, up to a constant
```

Substitute into the Bradley-Terry loss of §4.1. `Z(x)` depends only on `x`, so it **cancels in the difference** between chosen and rejected — and the reward model disappears:

```
  L_DPO  =  − E [ log σ(  β·log(π_θ(y_c|x)/π_ref(y_c|x))
                        − β·log(π_θ(y_r|x)/π_ref(y_r|x)) ) ]
```

**So the reference survives because it is not a regularizer bolted on — it is inside the definition of the implicit reward.** Remove it and `log π_θ(y_c) − log π_θ(y_r)` is unbounded: the loss falls forever by making the rejected response impossible, destroying the model. The `π_ref` terms are what turn an unbounded difference into a bounded *relative* one.

**The three steps, with the intuition each one needs:**

1. **The closed form.** "Take the reference distribution, multiply each response's probability by `e^(reward/β)`, renormalize." High-reward responses boosted, low-reward suppressed; `β → ∞` ignores reward (stay at `π_ref`), `β → 0` puts all mass on the argmax. Standard Lagrange-multiplier result — but PPO cannot *use* it, because `Z(x)` is a sum over all possible responses. Hence the rollout machinery.
2. **Invert it.** Read slowly, because it is the whole paper: **any reward function can be rewritten in terms of the policy it would induce.** The policy's log-ratio against the reference *is* a reward, up to β and a per-prompt constant. Policy and reward are two coordinate systems for the same object.
3. **Feed it to Bradley-Terry and watch `Z` die.** BT only ever uses reward *differences* on the same prompt. `β log Z(x) − β log Z(x) = 0`. The incomputable term — the sole reason the closed form was unusable — cancels because both responses share the prompt. What remains is four `sequence_logprobs` calls.

The reward model was never trained, because the policy *is* the reward model: the `β·log`-ratios are the **implicit reward**, and after DPO you can literally use the policy as an RM by computing them.

**What DPO buys and what it costs:**

| | PPO | DPO |
|---|---|---|
| Models in memory | **4** | **2** (policy + frozen reference) |
| Sampling during training | yes — on-policy rollouts | **no** — a fixed offline dataset |
| Implementation | RL loop, GAE, value head, many knobs | one loss function |
| Data efficiency | uses fresh on-policy data | limited by the *offline* preference set |
| Ceiling | higher when done well | plateaus — it can never see its own new mistakes |

## 6.2 Module — `dpo_loss`

```python
def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1, label_smoothing=0.0):
    logits = beta * ((pi_c - ref_c) - (pi_r - ref_r))
    if label_smoothing:                       # cDPO: assume ε of the labels are flipped
        return (-(1 - label_smoothing) * F.logsigmoid(logits)
                - label_smoothing * F.logsigmoid(-logits)).mean()
    return -F.logsigmoid(logits).mean()

def simpo_loss(pi_c_avg, pi_r_avg, beta=2.0, gamma=1.0):
    # SimPO: LENGTH-NORMALIZED log-probs, a target margin, and NO reference model.
    return -F.logsigmoid(beta * (pi_c_avg - pi_r_avg) - gamma).mean()
```

```python
>>> dpo_loss(ref_c, ref_r, ref_c, ref_r)      # at initialization pi == ref
0.6931                                         # = -log(0.5), EXACTLY. A sanity check.
>>> dpo_loss(t([-2.0]), t([-5.0]), t([-2.5]), t([-4.0]))     # prefers chosen
0.6210
>>> dpo_loss(t([-5.0]), t([-2.0]), t([-2.5]), t([-4.0]))     # prefers rejected
0.9432
```

> **The `0.6931` check is the DPO equivalent of §1.2's `ln(V)`.** Step 0 of a correct DPO run has loss exactly `−log 0.5`. Any other value means the reference log-probs were computed with a different tokenization, a different mask, or a different model than the policy started from.

**Worked micro-example** (β = 0.1, one pair, sums over response tokens):

```
  log π_θ(y_c) = −45.0    log π_ref(y_c) = −44.0    → chosen ratio   = −1.0
  log π_θ(y_r) = −52.0    log π_ref(y_r) = −54.0    → rejected ratio = +2.0

  implicit rewards:  r_c = 0.1 × (−1.0) = −0.10     r_r = 0.1 × (+2.0) = +0.20
  gap = −0.30  →  the policy currently rates the REJECTED response higher (it drifted toward y_r)
  L = −log σ(−0.30) = 0.854      high, as it should be
```

The gradient has exactly the form

```
  ∇L  ∝  − σ(−gap) · [ ∇ log π_θ(y_c)  −  ∇ log π_θ(y_r) ]
              0.57       increase chosen        decrease rejected
```

"raise chosen's likelihood, lower rejected's, with effort proportional to how wrong you currently are" — the same self-calibrating weight as §4.2.1(c). SFT has only the first term; **DPO's novelty is the *negative* gradient on the rejected response plus the adaptive weight.** Compare with §5.6: no generation, no RM, no value model, no GAE, no clipping, no epochs-on-rollouts. Two models in memory, and the reference's log-probs can be **precomputed once** over the dataset and cached — then it is *one* model at train time (this is the trick §8.6 leans on). Ordinary supervised learning: stable, deterministic, one real hyperparameter.

## 6.3 The family

| Method | Loss shape | Reference? | Key idea |
|---|---|---|---|
| ⭐ **DPO** | `−log σ(β·Δ)` | ✅ | The original closed-form substitution |
| **IPO** | `(Δ − 1/2β)²` | ✅ | Squared loss — DPO's sigmoid saturates and overfits deterministic preference pairs |
| **KTO** | per-example utility | ✅ | Needs only a **binary good/bad label**, not pairs — far cheaper data |
| ⭐ **SimPO** | `−log σ(β·Δ_avg − γ)` | ❌ | **Length-normalized** log-probs + a margin; no reference model at all |
| **ORPO** | SFT loss + odds-ratio term | ❌ | Merges SFT and alignment into **one** stage |
| **cDPO** | label-smoothed DPO | ✅ | Assumes a fraction of preference labels are wrong |

**SimPO's length normalization deserves the attention.** DPO uses the *sum* of log-probs, which is systematically more negative for long sequences — so the objective has an intrinsic length preference baked in, and DPO-trained models reliably get more verbose. SimPO uses the *average* (`sequence_logprobs(..., average=True)`, §5.2), which removes it. Same data, same comparisons, one line different, a measurably different model.

## 6.4 What the elegance costs

The math is exact — but exact *about a different problem* than PPO solves. Each gap is a named research thread:

1. **Offline: no exploration.** PPO's policy generates, is judged on its *own* novel outputs, improves, generates again — a ladder above the dataset. DPO sees only the fixed pairs and can never be corrected on mistakes the annotators never saw. This matters most for reasoning: you cannot DPO your way to R1, because those capabilities come from exploring beyond demonstrations. *Patch:* **iterative / online DPO** — generate fresh responses with the current policy, rank them with an RM or judge, DPO on the new pairs, repeat. It rebuilds half of PPO, admittedly.
2. **It drains probability from *both* responses.** Empirically `log π_θ(y_c)` often *falls* during DPO — the loss constrains only the *margin*, and the cheapest way to widen it is to crush `y_r` faster than `y_c` rises, leaking mass to unseen sequences. The model can end up *less* likely to produce its own chosen answers. *Patches:* add an SFT anchor on chosen (**DPO + NLL**, used in Llama-3-era recipes; the §8.6 default), or cap the margin (**IPO**).
3. **The sum-vs-average length bias, now with teeth.** If chosen responses tend to be longer, the raw sums are length-confounded and DPO absorbs annotator length bias into the policy — one source of post-DPO verbosity. **SimPO**'s fix: average, drop the reference, add a target margin — length-debiased and ref-free, at the cost of the KL anchor's grounding.
4. **A weaker leash.** PPO's KL is measured on the policy's *own generations*, where it actually lives. DPO's implicit KL control acts only through the dataset pairs; off-distribution drift is less constrained, and a low β can quietly wreck capabilities the pairs never touch.

```
  DPO (+ variants)     the default for style / helpfulness / safety — Zephyr, Tulu, Llama-3 post-training
  PPO / GRPO + RLVR    reasoning, code, math — where exploration IS the point
  production stacks    usually BOTH:  SFT → DPO passes → RL passes            (§7.6)
```

> **The arc, complete:** BT gave a loss for judging (§4), PPO built a factory to transfer judgments into a generator (§5), and DPO proved the factory was — for the offline case — mathematically unnecessary: substitute the closed-form solution back into BT and train the generator as its own judge. The instinct "shouldn't the preference loss just train the LM directly?" is Step 3 of the derivation.

---

# 7. GRPO and RLVR — the Current Frontier Recipe

## 7.1 The objective

GRPO (DeepSeekMath, then R1) keeps PPO's clipped surrogate and **deletes the value network**:

```
  For each prompt x, sample a GROUP of G responses  {y_1 … y_G} from π_θ_old
  Score each with a verifier or reward model:       {r_1 … r_G}

                    r_i − mean(r_1…r_G)
  Â_i  =  ─────────────────────────────────      ← the GROUP is the baseline
                    std(r_1…r_G)

  L  =  − E [ (1/G) Σ_i (1/|y_i|) Σ_t  min( ρ_{i,t}·Â_i ,
                                             clip(ρ_{i,t}, 1−ε, 1+ε)·Â_i )
                                        −  β · KL( π_θ ‖ π_ref ) ]
```

Note `Â_i` carries **no `t` index** — every token of a response gets the same advantage. GRPO makes no attempt at per-token credit assignment; it says only *"this whole rollout was better than average, so make all of it more likely."*

## 7.2 Doubt — "why can GRPO drop the value network PPO insists on?"

Because the value network's only job was to provide a **baseline** that reduces gradient variance, and sampling `G` responses to the same prompt provides one for free:

| | PPO | GRPO |
|---|---|---|
| Baseline | a learned `V_ψ(s_t)` | the **empirical mean of the group** |
| Cost of the baseline | a second full-size trainable model (~16 bytes/param) | `G−1` extra rollouts — inference, not training |
| Bias | `V_ψ` is wrong early and must itself be learned | unbiased by construction |
| Credit assignment | per-token | **per-sequence only** |
| Models in memory | 4 | **3** (policy, reference, reward — and reward is free under RLVR) |

**The trade is explicit: pay in sampling to save in memory and in a hard learning problem.** Training an accurate token-level value function for language is genuinely difficult — the value head is fitting "expected future reward" over a 150k-way branching process — and a badly-fit value head injects bias into every advantage. The group mean is dumber but unbiased and costs nothing to learn.

**The clean way to see it: a baseline's only job is to answer "compared to what?"** The policy gradient needs to know whether reward 0.7 is good news or bad news *for this prompt*. PPO answers with a learned forecast ("I predicted 0.5, so 0.7 is good"); GRPO answers with an empirical control group ("your 4 siblings averaged 0.4, so you're good"). Both are valid baselines; subtracting either leaves the gradient unbiased in expectation.

**The currency swap — and why the exchange rate moved:**

```
  PPO pays in:   memory (16 B/param) + a hard learning problem (fit E[return] over a 128 K-way
                 branching process, continuously, as the policy moves)
  GRPO pays in:  inference — G−1 extra rollouts per prompt
```

Between 2022 and 2025 inference got dramatically cheaper — PagedAttention, prefix caching (the G rollouts *share the prompt*, so its KV cache is computed once: RadixAttention's exact sweet spot), speculative decoding — while the value-learning problem stayed hard. **GRPO is what you design when rollouts are cheap and trainable memory is precious; it is an algorithm shaped by the vLLM era.**

**Why per-sequence credit is *enough* here, when §5.5 argued it is not.** Recall the blunder example: tokens 200–210 ruined a good response and `V_ψ`'s falling forecast localized the blame. GRPO cannot do that — if rollout *i* scored above the group mean, *every* token of it, blunders included, gets pushed up. GRPO deliberately un-solves the problem PPO built a 27B model to solve. It gets away with it under RLVR for two reasons: (1) the bias point — a mis-fit `V_ψ` does not add noise, it adds *systematic* error to every advantage, confidently, early in training when V is worst; the group mean is never wrong about what it claims to be, just noisier, and "dumb but unbiased" beats "smart but miscalibrated" surprisingly often; (2) when the reward is *exact* and G rollouts diverge from a shared prompt, the **contrast between siblings** carries much of the information per-token credit used to carry — "this whole chain worked, that one didn't" is enough, because the model figures out *which parts* matter through repetition across thousands of groups: §5.5's averaging-across-trajectories argument, relocated.

And under **RLVR** (`03` §11.4) the reward model is a Python function, so the memory table collapses to **policy + reference**, which is exactly DPO's footprint with on-policy data. That combination is why the 2025–2026 reasoning models all use it.

## 7.3 Module — `grpo_advantages`, `grpo_loss`

```python
def grpo_advantages(rewards, eps=1e-4):
    # [B, G] group of G rollouts per prompt -> normalized WITHIN the group
    mu = rewards.mean(dim=-1, keepdim=True)
    sd = rewards.std(dim=-1, keepdim=True, unbiased=False)   # POPULATION std: the group
    return (rewards - mu) / (sd + eps)                       # is complete, not a sample

def grpo_loss(logp, logp_old, adv, ref_logp, mask, clip=0.2, beta=0.0):
    # adv is per-SEQUENCE and is broadcast to every token of that sequence
    ratio = torch.exp(logp - logp_old)
    a = adv.unsqueeze(-1)
    obj = torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a)
    if beta:
        obj = obj - beta * kl_penalty(logp, ref_logp, "k3")
    return -(obj * mask).sum() / mask.sum()
```

```python
>>> rewards = torch.tensor([[1., 0., 1., 0., 1.],     # 3 of 5 rollouts correct
...                         [0., 0., 1., 0., 0.]])    # 1 of 5 correct
>>> grpo_advantages(rewards)
[[ 0.816, -1.224,  0.816, -1.224,  0.816],
 [-0.500, -0.500,  2.000, -0.500, -0.500]]
>>> grpo_advantages(rewards).mean(-1)
tensor([0., 0.])                          # zero-mean per group, by construction
```

**Read the second row.** When only 1 of 5 rollouts succeeds, that one gets advantage **+2.0** while the four failures get −0.5 each. The rarer the success, the stronger the signal — GRPO automatically up-weights breakthroughs on hard problems. That is the mechanism behind "reasoning emerges from RL": the model is rewarded most for the rare rollouts where a longer chain of thought found the answer.

**The degenerate case to watch for:** if *all* `G` rollouts get the same reward — all correct (too easy) or all wrong (too hard) — then `std = 0`, every advantage is 0, and **the prompt contributes no gradient at all.** Curriculum matters enormously in RLVR: prompts must sit where the model succeeds sometimes.

**Two details in the code worth pinning.** `unbiased=False` is population std (`/G`, not `/(G−1)`): the group is not a sample from some larger population we are estimating — the G rollouts *are* the entire comparison set. And the `1/|y_i|` in §7.1's formula is §5.2's sum-vs-average issue reappearing: DeepSeekMath length-normalizes each response's token-sum, which creates a subtle length bias of its own (low-advantage long responses get diluted punishment) that Dr. GRPO removes. *The module above does not do that* — its `/ mask.sum()` divides by the tokens of the whole batch, the DAPO "token-level" form that `trl` defaults to and §7.9 recommends.

**The second doctest row is the "aha" of the section — work the mechanism.** Same reward value (1.0), radically different learning signal:

```
  easy prompt, 3/5 correct:   each winner gets +0.816
  hard prompt, 1/5 correct:   the winner gets  +2.000
```

because standardization divides by the group's std, and a rare success sits far from a low mean in a low-spread group. **The gradient automatically concentrates on breakthroughs.** Now picture what a rare success on a hard math prompt physically *is*, early in training: almost always a rollout where the model happened to grind through a longer, more careful chain — checked its arithmetic, backtracked, tried a second approach. That rollout gets +2.0 broadcast across *all* its tokens, including every "wait, let me reconsider". Repeat across millions of groups and the policy drifts toward long, self-correcting reasoning. Nobody rewarded length or reflection directly; the group-relative statistics did. That is the mechanical content of "reasoning emerges from RL" and of R1-Zero's famous growing response lengths.

**The degenerate case is the operational headache.** `std = 0` → all advantages 0 (the ε only prevents the NaN; the signal is still dead). A prompt the model always solves teaches nothing; one it never solves teaches nothing. **Only the band where success is *intermittent* — pass rate strictly between 0 and 1 — produces gradient.** Hence RLVR's curriculum obsession: measure pass rates, drop saturated prompts as training progresses, keep the model perpetually at the edge of its ability. The dataset must *move with* the model. (Contrast supervised learning, where easy examples still contribute loss — RLVR's training signal lives exclusively on the frontier of current capability.)

## 7.4 The refinements

| Variant | Fix |
|---|---|
| **DAPO** | Decoupled clip bounds (higher upper than lower) to stop entropy collapse; dynamic sampling that **drops the zero-variance prompts** described above |
| **Dr. GRPO** | Removes the `std` normalization, which biases toward low-variance (easy) prompts |
| **GSPO** | Sequence-level rather than token-level importance ratios — more stable for MoE, where per-token routing makes token ratios noisy |
| **Token vs sequence normalization** | Dividing by `\|y_i\|` under-weights long responses; several 2025 recipes normalize by total tokens in the batch instead |

## 7.5 Doubt — "so in RLVR the reward model is hosted somewhere else and returns correct/incorrect?"

Drop "hosted somewhere else" and drop "model" — both words do wrong work. In RLVR there is **no model at all** in the reward path; that is the V in the name. The reward is an ordinary **function**, deterministic checking code running inside the training loop:

```python
def reward(prompt, response) -> float:
    answer = extract_final_answer(response)          # \boxed{}, last line, a code block…
    return 1.0 if equivalent(answer, ground_truth[prompt]) else 0.0
```

**What "verifiable" concretely looks like:**

- **Math** — each prompt ships with its answer. Parse the model's `\boxed{}`, canonicalize (`3/2 == 1.5 == 1 1/2` — this normalization is real engineering, sympy-style equivalence), compare. 1.0 or 0.0.
- **Code** — the prompt ships with unit tests. Extract the code block, run it in a sandbox, reward = all tests pass (or the fraction). The "reward model" is literally pytest plus a container.
- **Others** — required output format (a cheap auxiliary reward R1 used), the SQL query returned the right table, the proof checks in Lean, the game move won, the agent's tool call produced the expected state.

| | Learned RM (§4) | RLVR verifier |
|---|---|---|
| What it is | 27B neural net, BT-trained | Python function + ground truth |
| Where judgment lives | weights, learned from votes | the answer key / test suite |
| Output | unbounded scalar, relative only | exact {0, 1} or test fraction |
| Can be wrong? | yes — a proxy | no, up to parser/sandbox bugs |
| Can be hacked? | **yes — the central RLHF disease** | not in the RM sense* |
| Coverage | any text | only tasks with checkable answers |

*\*Models still find seams — guessing common answers against a weak parser, hard-coding test outputs if the sandbox leaks the tests. But that is **spec-hacking**, fixable by hardening the checker. A learned RM's exploitability is *intrinsic*: it is a smooth function off-distribution, and pressure always finds the smooth function's fantasies. A verifier has no off-distribution behaviour to exploit.*

**Why this changes both the economics and the ceiling.** *Economics:* row 4 of §5.3 becomes ~zero GPU; no RM training stage, no preference collection — the "annotation" is answer keys and test suites, which scale by data engineering, not human judgment (sandboxed execution at training scale is its own infra problem, but a CPU/systems one). *Ceiling — the deeper one:* optimizing against a learned RM has a built-in horizon; push hard enough and you are optimizing the RM's *errors* (Goodhart), and the KL leash exists mostly to stop the policy walking into the RM's fantasy zone. A verifier has no fantasy zone, so you can optimize **hard and long** — thousands of steps, aggressive exploration — and the signal stays meaningful. That is why reasoning RL runs push so much further than helpfulness RLHF ever could. *The catch is coverage:* "is this poem moving", "was this helpful" — no verifier exists, so the real stack is a hybrid (§7.6).

*(The "hosted elsewhere" instinct is not crazy operationally: the sandbox fleet may well be a separate service the trainer calls — work queues, timeouts, retries. But that is deployment topology, not conceptual structure. Nothing in the reward path has weights, and that absence — no learned proxy between the policy and the truth — is what makes the recipe work.)*

## 7.6 Doubt — "so PPO and DPO are dead and everyone uses GRPO + RLVR?" — and the decision map of what to use when

Two corrections, one small and one important.

**Small: GRPO and RLVR are different axes, not alternatives.** GRPO is *how you compute advantages* (the algorithm); RLVR is *where the reward comes from* (the signal). "GRPO or RLVR" is like "SGD or labeled data". The typical reasoning recipe is GRPO **with** RLVR. You can also run GRPO with a learned RM (no verifier available), or PPO with RLVR (verifiable reward, but you kept the value model). All four cells of that 2×2 exist in practice.

**Important: PPO and DPO are very much alive.** A 2025–26 frontier post-training pipeline is a *sequence of stages*, not one method:

```
  SFT
   → DPO (or variant)          style, helpfulness, safety, chat polish — cheap, stable, static pairs;
                               Llama-3 / Tulu / Zephyr-lineage recipes all keep it, often several rounds
   → GRPO + RLVR               math / code / reasoning — exploration + verifiable signal; the R1 stage
   → (+ RM-scored RL,          open-ended quality where no verifier exists — a learned RM or
       GRPO or PPO)            LLM-judge (§4.3.1) is still the only option
```

**DPO survives** because most alignment work is not reasoning. Tone, refusal behaviour, formatting, persona, sycophancy — there is no verifier for "polite but not sycophantic", and online RL for these is expensive overkill when static pairs plus one supervised loss gets you there. By volume of runs it is the most *used* method, even if it is not the headline of reasoning papers.

**PPO survives** in two places. First, wherever **per-token credit genuinely matters**: long-horizon agentic / tool-use training with intermediate rewards — a 40-turn trajectory where step 3's tool call succeeded and step 7's failed; a group-mean broadcast across the whole trajectory is too crude, a value function localizing credit is not. Second, as the **safer choice when the reward is a learned RM**: the value baseline plus per-token KL control gives finer restraint against reward hacking than GRPO's cruder machinery. GRPO's win is specifically the regime "verifiable reward + shared-prompt groups + cheap rollouts"; outside it, the trade flips.

And the field keeps moving *past* GRPO — Dr. GRPO, DAPO, RLOO, REINFORCE++, GSPO (Qwen3's choice)… "GRPO" in conversation increasingly means "the family of value-model-free policy-gradient methods" rather than the specific DeepSeekMath equations.

> **The one-liner:** DPO owns offline preference alignment, GRPO-family + RLVR owns verifiable reasoning, PPO persists where per-token credit or maximum hacking-resistance justifies its cost — and a frontier model typically passes through more than one of these on its way out the door. The methods stack; they did not replace each other.

**The map itself.** Every post-training setup answers two independent questions:

```
  Q1  Where does the reward come from?
        human preference pairs (offline, no reward function at all)      → DPO territory
        a learned RM / LLM judge  (learned proxy: hackable, universal)    → RM-scored RL
        a verifier                (exact, unhackable, narrow)             → RLVR
  Q2  How is credit assigned, if doing online RL?
        learned per-token forecaster (V_ψ + GAE)                          → PPO
        group statistics, per-sequence                                    → GRPO family
```

These compose freely — the method zoo is a small grid.

| Method | Reward source | Credit | Models in RAM | Explores? | Cost | Fragility |
|---|---|---|---|---|---|---|
| SFT | demonstrations | per-token CE | 1 (16 B/p) | no | low | none |
| DPO | preference pairs | pair margin | 2 (18 B/p)\* | no | low | low |
| SimPO / IPO / KTO | pairs / thumbs | pair margin | 1 (16 B/p) | no | lowest | low |
| RM + PPO | learned proxy | per-token | 4 (36 B/p) | yes | highest | high |
| RM + GRPO | learned proxy | per-sequence | 3 (20 B/p) | yes | high | medium |
| RLVR + GRPO | verifier fn | per-sequence | 2 (18 B/p) | yes | med–high | medium |
| RLVR + PPO | verifier fn | per-token | 3 (34 B/p) | yes | high | high |

\*reference log-probs precomputable → effectively 1 model at train time. With LoRA (§8.6) every row's memory collapses further: all the models are adapters on one frozen 4-bit base.

**SFT — always first.** RL from a base model wastes compute learning what imitation teaches for free. Also the right *whole* tool when you have golden demonstrations: 5 K perfect traces of your task → SFT / LoRA, done.

**DPO (+ SimPO / KTO) — style, tone, safety, chat polish.** Use when the behaviour is a *taste* expressible as "this response over that one": persona, verbosity control, refusal calibration, formatting habits, reducing sycophancy. No verifier exists and exploration adds nothing — the knowledge is in the pairs, not discoverable by trial. *Concrete:* a company voice for a support bot — 20 K pairs, one DPO night-run, stable. KTO when you only have production thumbs-up/down, not pairs. SimPO when length bias bites. *Don't use for* anything the model must *discover* beyond the data.

**GRPO + RLVR — math, code, competition reasoning. The R1 recipe.** Use when answers are checkable, prompts have single verifiable targets, rollouts are cheap relative to trainable memory, and trajectory-level credit suffices (one final answer = one natural unit of blame). *Concrete:* boosting a 32B's MATH / LiveCodeBench scores — answer-keyed dataset, sympy checker + sandboxed pytest, G = 8–16, pass-rate-filtered curriculum. *Watch for:* std = 0 signal death (curate difficulty), length-norm bias (Dr. GRPO), parser robustness.

**PPO + RLVR — agentic, multi-step, tool-use training.** Use when the reward is verifiable but arrives over a *long horizon with meaningful intermediate structure* — exactly where GRPO's flat broadcast is too crude. A 40-turn agent trajectory that ultimately succeeded may contain three wasted tool calls and one brilliant recovery; a value function localizes that (its falling / rising forecast blames turn 7, credits turn 23), a group mean cannot. Also preferred when **rollouts are expensive** — each one is a real multi-tool episode with a browser, a sandbox, API calls — which flips GRPO's economics: G = 16 rollouts of a 30-minute workflow per prompt is unaffordable, so you would rather pay 16 B/param for `V_ψ` than 15 extra episodes. *Concrete:* training a SWE-agent on real repos — reward = tests pass after the episode, intermediate rewards for successful edits / builds, PPO's per-token GAE assigning blame across hundreds of actions.

**GRPO (or PPO) + learned RM / LLM-judge — open-ended quality.** Use when you want *online* improvement on unverifiable qualities (helpfulness depth, writing quality, nuanced instruction-following) beyond what static pairs support. *Watch for:* reward hacking — this is the regime the KL leash was built for; keep runs short, monitor drift and length, refresh the RM with new human data.

```
  FULL PRODUCTION PIPELINE (what actually ships):

  SFT  →  DPO rounds (style / safety)  →  GRPO + RLVR (reasoning)  →  RM-RL touch-up (polish)
          cheap, static                    the capability jump          careful, short
```

> **The single organizing principle:** match the reward source to what is *checkable*, match the credit mechanism to where the *blame lives*, and spend memory only when localization earns it.

## 7.7 Doubt — "the rollout is plain inference, so where does the gradient come from — and if greedy can't solve a problem, how does sampling help?"

Three computations per step; one has gradients.

```
  1. ROLLOUT   autoregressive decode, G completions per prompt, no graph — INFERENCE (vLLM)
  2. π_old     ONE teacher-forced forward over (prompt + sampled completion), no grad; store log p(o_t | prefix)
  3. π_θ       the SAME forward with gradients on, once per update; loss = clipped exp(logp_θ − logp_old) × Â
```

`π_old` and `π_θ` are the same function at two moments. With one update per rollout batch (`μ = 1`) the ratio is exactly 1 and the clip never fires — the gradient still exists (score × Â); with `μ > 1` the batch is reused and the clip earns its keep. Rollouts carry 0% of the gradient and, on one GPU, a third to a half of the wall-clock (§8.6.7); the two teacher-forced passes are the entire training cost.

**The sampler and the scorer must be the same distribution**, because the ratio compares two log-probabilities *of the sampled tokens*. Sample at temperature T → divide the logits by T in the scorer too (`trl` does: `logits / self.temperature`). Force `top_p = 1, top_k = 0`: the model's `generation_config.json` (Qwen: `top_k=20, top_p=0.8`) would otherwise sample from a truncated distribution the scorer cannot express. A bf16 engine copy vs an NF4 training copy is a small constant gap, tolerated — but never take `logp_old` from the engine's own logprobs.

**Sampling is the only source of signal.** Greedy gives G identical rollouts → `std = 0` → nothing trains. And greedy is not the model's best: it commits to the top token at every step, so one early wrong turn is never revisited, while 8 samples explore 8 chains of choices. Pass@k grows roughly with log k — a 35% pass@1 model reaches ~55–65% at pass@8 — so a large share of greedy failures *are* solvable by the same weights. **RL moves pass@1 toward pass@k; it rarely raises pass@k.** A 200-prompt greedy-vs-8-samples run (~4 M tokens) tells you which regime you are in before you spend GPU-days: if pass@8 ≈ pass@1, the ceiling is capability, not sharpening, and RL is the wrong tool (§7.8).

## 7.8 Doubt — "if all G rollouts fail, is the prompt lost forever — and does RL ever add capability, or is SFT the better tool?"

**For that step, yes.** `std = 0`, zero advantage, no gradient. In a first run of this recipe 57% of groups were all-wrong and 27% all-correct, so only 17% carried any gradient. "Forever" is wrong for three reasons and right for one.

*Redraw with fresh dice.* If the per-sample success rate on that prompt is `p`, a mixed group appears with probability `1 − (1−p)^G`: at p = 0.05, 34% at G = 8, 71% at G = 24. *Transfer:* updates from other prompts move the same weights — reinforcing "stage A" on an easy problem and "stage C" on another raises a hard problem's chance multiplicatively (five stages at 0.5 each → 3%; at 0.8 → 33%) without it ever producing a gradient itself. That compounding is how RL unlocks the hard tail, and the argument for feeding stages before compositions. *The hard limit:* `p = 0` — the model cannot produce the answer at any temperature — never trains. Nor can it when the label is unscorable (~10–19% of a scraped math pool: bare choice letters, multi-part answers, figure references) — those are `p = 0` in disguise and burn rollouts every time they are drawn.

**Levers, cheapest first:**

1. **Skip dead work** — zero-advantage sequences contribute exactly 0; leave them out of both passes (normalizer at the full token count → bit-identical gradient).
2. **Retry with early stop** — re-sample all-zero prompts in rounds of G up to a budget (16–24), stop at the first success; train on the successes plus a random fill of failures, still G wide. Measured: gradient-carrying groups 17% → 42%; ~1 in 5 retried prompts rescued. Bias: a 1-in-24 success is presented as 1-in-8. Cost lands on the `p ≈ 0` prompts, so —
3. **Park** prompts that exhaust the budget (and ones solved G/G twice) for ~100 steps; checkpoint that state. DAPO's dynamic sampling is the batch-level form of 2 + 3.
4. **Curriculum** — shift the mixture easy → hard over training, never dropping easy to 0; needs difficulty labels (a greedy probe of the pool is the cheap source).
5. **Turn the length penalty on** (§7.9): all-correct groups are also zero-variance; a one-sided length term makes them train.

**"So SFT for accuracy, RL for refinement?" — half right.** Match the tool to the failure:

| situation | tool | why |
|---|---|---|
| pass@k high, pass@1 low | **RL** | it can solve it sometimes; cheap in data, generalises, learns behaviours imitation cannot express (stopping, verifying, length) |
| pass@k low | **SFT / distillation** | nothing to reinforce; something must put the capability into the distribution first |

Three cautions: SFT on the human gold solutions is a *style transplant* that often lowers a small model's accuracy; rejection-sampling SFT (keep the model's own verified-correct samples — STaR/RFT/ReST) is the SFT that works, and it is GRPO without negatives, which carry much of the signal; and distillation from a stronger model (DeepSeek-R1's finding: distil to small models, don't RL them) is the one path that raises pass@k. Recipe: distillation first, RL to sharpen (§8.6.4). The same logic explains why robots and game agents learn from scratch and an LLM cannot: dense per-step rewards, tiny action spaces, near-free simulation and self-play curricula make random exploration find reward; a 150K-token vocabulary over 2,000 positions never does — pretraining is the "demonstrations first" phase every sparse-reward robotics recipe also needs.

> Log the fraction of zero-variance groups every step. A run of 3–5 steps with reward 0 in *every* rollout is broken rollouts (stale engine, wrong template, uninitialised weights after an engine sleep — all seen in practice), not hard data. Abort on it automatically.

## 7.9 Length is the operational problem — cap, temperature, penalty, clipping

The papers talk about reward; the first thing you fight is completion length. Measured, Qwen3-4B-Instruct on competition math (200 held-out problems):

| | value |
|---|---|
| greedy pass rate, cap 2K → 4K | 26% → 34.5% |
| greedy rollouts hitting the 4K cap | 34% |
| human reference solutions, tokens: median / p95 / max | 588 / 1,137 / 1,338 — none over 2K |
| the model's *correct* completions: median / p95 / max | 1,004 / 3,503 / 4,012 |
| model ÷ gold length on correct answers: median / p75 / p90 | 1.6× / 2.8× / 5.4× |
| problems still at the cap: their gold length | median 647 — lost, not long |

**Cap:** allow ~3× the human solution typically, hard cap at ~4× the p95 gold length (here → 4,096). Doubling beyond that rescues a trickle at double the cost; the cap gets *more* generous as the policy learns to finish. Rollout time floors at longest sequence × decode-step time, so the cap is also the cost knob.

**Temperature:** T = 1.0 from the full softmax rambled (52–94% of rollouts at the cap vs 34% greedy); T = 0.7, the model's shipped default, with the scorer tempered identically (§7.7), fixed it.

**A one-sided length penalty on correct answers:** `ratio = tokens / gold_tokens`, `excess = clip((ratio − 3)/(4 − 3), 0, 1)`, `reward = correct × (1 − w·excess)`, `w = 0.5`. One-sided (shorter and correct is strictly better), correct-only (a wrong short answer must never beat a right long one), `w < 1` (any correct beats any wrong). Watch eval pass rate and length together: shorter at equal pass rate is success; pass rate falling means `w` is too high.

**Clip failed rollouts to the longest correct one *of the same prompt*:** a failure's tail beyond where a correct sibling finished is where the model was lost — the negative push there is noise (DAPO's overlong argument) and most of the padded compute. Truncating equals masking those tokens (causal prefix log-probs do not depend on the tail). Never across prompts: an easy problem solved in 600 tokens says nothing about where a hard failure went wrong at 2,500.

**Normalizer:** by *total tokens in the batch* (DAPO token-level, trl's default), not `1/|y_i|` per sequence, which dilutes the punishment of long failures.

## 7.10 The technique map — which refinement for which symptom

§7.6 chose the method; this chooses the refinements once GRPO + RLVR is running and you are reading a log.

| Symptom | Technique | Why | Risk |
|---|---|---|---|
| most groups all-0 | retry + parking, DAPO dynamic sampling (§7.8) | signal lives only in mixed groups | rare successes over-weighted; rollouts on `p≈0` prompts |
| all-correct groups wasted | one-sided length penalty (§7.9) | "both correct" → "shorter is better" | too strong → reasoning skipped |
| entropy collapsing | DAPO clip-higher (ε_high 0.28 > ε_low 0.2) | the upper clip stops low-probability tokens from ever being lifted | faster reward hacking |
| long failures dominate | token-level normalization, overlong clipping (§7.9) | removes the `1/\|y_i\|` dilution, drops lost tails | prefix-concentrated negatives |
| easy prompts dominate | Dr. GRPO (no `std`, no length normalization) | `1/std` up-weights low-variance groups | rare successes lose the +2.0 boost |
| MoE: noisy token ratios | GSPO (sequence-level ratio; Qwen3) | routing changes token ratios between sampler and trainer | coarser clipping |
| one GPU | β = 0, LoRA r 8–16 all-linear, NF4 base, colocated vLLM with sleep mode (§8.6.7) | verifiers can't be Goodharted, so the leash is optional; RL needs ~1 bit/episode of capacity | unbounded drift — watch length and eval |
| verifier being gamed | β > 0 (k3 KL), harden the parser, shorter runs | the leash is for learned-RM regimes | slower learning |
| rollouts slow vs training | `steps_per_generation` > 1 / async (few-step-stale) | the clip tolerates a few steps of staleness | more clipping |
| per-token blame matters (agents) | PPO with a critic (§7.6) | a value function localizes credit | 16 B/param |
| final reward too sparse | process reward / partial credit | gradient on intermediate steps | the most gameable reward |
| `p ≈ 0` on what you care about | distillation / verified-trace SFT first (§7.8) | RL can't reinforce what is never sampled | needs a teacher |
| pass@1 ≈ pass@8 already | stop RL (§7.7) | ceiling is capability | — |

**Defaults if you must pick without a log:** GRPO-family advantages with dynamic sampling (DAPO) or without `std` (Dr. GRPO); token-level normalization; β = 0; asymmetric clip; T = 0.7–1.0 with the scorer tempered identically; cap ≈ 4× the p95 reference length with overlong clipping; a moving pass-rate-filtered curriculum; distillation before RL under ~7B. Each is a response to one failure above; none is free.

---

# 8. Distillation

## 8.1 The objective

Match the teacher's *distribution*, not just its samples:

```
  L  =  α · T² · KL( p_teacher^T  ‖  p_student^T )  +  (1−α) · CE(student, hard labels)
             └─┬─┘  └───────────────┬──────────────┘
          the T² restores the      soft targets carry the teacher's full
          gradient scale that      uncertainty — "this is 0.6 cat, 0.3 lynx"
          temperature shrinks      instead of a one-hot
```

| Direction | Behaviour |
|---|---|
| **Forward KL** `KL(teacher‖student)` | **Mass-covering** — the student must put probability everywhere the teacher does. Safe, slightly blurry |
| **Reverse KL** `KL(student‖teacher)` | **Mode-seeking** — the student sharpens onto the teacher's dominant modes. Better for a much smaller student |
| ⭐ **On-policy** | Sample from the *student*, score with the *teacher*. Fixes exposure bias: the student learns to recover from its own mistakes, not the teacher's |

### 8.1.1 Doubt — walk the whole objective with real numbers: temperature and `T²`, forward vs reverse KL term by term, and the sign of the KL

Toy vocabulary **{cat, lynx, dog, car}**, true label = cat. The teacher has learned a nuanced view — mostly cat, plausibly lynx, a little noise:

```
  token    teacher p   student p (before training)
  cat        0.60        0.481
  lynx       0.30        0.216
  dog        0.07        0.196
  car        0.03        0.107
```

The **hard label** is `[1, 0, 0, 0]` — it throws away "0.30 lynx" entirely. That is the motivation: the one-hot carries `log₂(4) = 2` bits; the teacher's distribution also encodes *which* mistakes are reasonable (cat/lynx) and which are not (cat/car). Over a 150 K vocabulary the gap is enormous.

**Step 1 — soften at T = 2.** Divide logits by T before the softmax; both distributions flatten:

```
  token    teacher pᵀ   student pᵀ
  cat        0.440        0.360
  lynx       0.311        0.241
  dog        0.150        0.230
  car        0.098        0.170
```

The point of temperature: it exposes the "dark knowledge" in the *small* logit gaps (dog vs car) that the dominant cat/lynx logits would otherwise drown.

**Step 2 — forward KL, and why `T²` is not decorative.**

```
  KL(teacher ‖ student)   at T = 1:  0.121 nats
                          at T = 2:  0.051 nats  (raw)    ← softer distributions disagree LESS
  × T² = 4:                          0.204                ← restored, now comparable to the T=1 scale
```

Softmax gradients scale like `1/T`, so the KL signal shrinks as you soften. Multiplying by `T²` compensates, keeping the gradient magnitude comparable to T = 1 training *while* keeping the smoothed, information-rich targets.

**Step 3 — combine with the hard CE.** `CE = −log 0.481 = 0.732`, so with α = 0.5: `L = 0.5 × 0.204 + 0.5 × 0.732 = 0.468`. That is the shape of the doctest output below (`soft KL … + hard CE … → total`).

**Forward vs reverse on the *same* numbers.** Reverse KL, `KL(student ‖ teacher) = Σ p_s·log(p_s/p_t) ≈ 0.161`, and look where its mass comes from: the **dog** term is `0.196 × log(0.196/0.07) = +0.20` — huge, because the student put noticeably more mass on dog than the teacher ever does. Reverse KL punishes *student mass where the teacher has little* severely: it is terrified of inventing modes, so it collapses the student onto where the teacher's mass actually lives (cat/lynx). **Mode-seeking.** Forward KL's biggest term came from **lynx** instead (`0.30 × log(0.30/0.216) = +0.099`) — it cares about "did you *cover* everywhere I put mass", not "did you wander somewhere I didn't". **Mass-covering** — safer, but can leave the student spreading probability into regions the teacher does not really mean. For a much smaller student that has no capacity to cover everything, mode-seeking produces more coherent generations; that is MiniLLM's argument (§8.3).

**Reading the code.** `F.kl_div(input, target, log_target=True)` computes `Σ exp(target)·(target − input)` = **KL(target ‖ input)**. Here `target = log_softmax(teacher)`, `input = log_softmax(student)`, so it is literally KL(teacher ‖ student) — forward, mass-covering, matching the table. The `mask` matters for causal LMs specifically: never distil on padding or on prompt tokens you are not training — only completion tokens contribute, the same masking as `causal_lm_loss`.


**"High student probability → lower KL; low → higher" — right?** Right direction, but the precise version is what makes "mass-covering" mean something. Forward KL is a sum over tokens, and each token's term is

```
  term_i  =  p_i^teacher · log( p_i^teacher / p_i^student )
```

For fixed `p^teacher`, raising `p^student` shrinks the ratio, the log, the term — down through zero and negative. Driving `p^student → 0` sends the term → ∞. So far your intuition holds. **The catch is the weight `p_i^teacher` in front.** If the teacher itself puts near-zero mass on a token, that term barely matters no matter what the student does — a huge log-ratio times ~0.

```
  token   teacher   student   term = p_t · log(p_t / p_s)
  cat      0.60      0.481    +0.133     student close to teacher → moderate
  lynx     0.30      0.216    +0.099     teacher wants 0.30, student under-delivers → penalized
  dog      0.07      0.196    −0.072     student OVER-assigns, but p_t is small → nearly free
  car      0.03      0.107    −0.038     same
                              ─────
                               0.121     ✓ the KL from §8.1.1
```

Now perturb: student drops **lynx** to 0.05 → `0.30 × log(0.30/0.05) = 0.538`, a massive jump from failing to cover a high-teacher-mass token. Student raises **car** to 0.40 → `0.03 × log(0.03/0.40) = −0.078`, barely different from before. **That asymmetry is the mechanical reason forward KL is called mass-covering and not "match everywhere equally"**: tokens the teacher cares about are enforced (up to ∞); tokens it barely cares about are free real estate for the student — which is precisely what reverse KL, weighted by `p^student` instead, refuses to allow.

**Why does KL enter the loss as `+KL`, when likelihood enters as `−log P`?** Because KL is already a non-negative *distance*: `KL ≥ 0`, with equality only when the two distributions are identical. More KL = more divergent; you want less; so you **minimize it directly** — `L = +KL`. Likelihood is the opposite kind of quantity: you want *more* of it, so to turn it into something you minimize you flip the sign, `L = −log P`. Both conventions say "minimize the loss"; the sign just reflects whether the underlying quantity is good-when-large (likelihood) or good-when-small (divergence). The `distillation_loss(teacher, teacher)` doctest returning `0.0` is the sanity check of exactly this: identical distributions, zero divergence, nothing to minimize.

## 8.2 Module — `distillation_loss`

```python
def distillation_loss(student_logits, teacher_logits, labels, T=1.0, alpha=0.5):
    mask = (labels != IGNORE).reshape(-1)
    s = student_logits.reshape(-1, student_logits.size(-1))[mask].float() / T
    t = teacher_logits.reshape(-1, teacher_logits.size(-1))[mask].float() / T
    soft = F.kl_div(F.log_softmax(s, -1), F.log_softmax(t, -1),
                    reduction="batchmean", log_target=True) * (T ** 2)
    hard = causal_lm_loss(student_logits, labels)
    return alpha * soft + (1 - alpha) * hard, soft.detach(), hard.detach()
```

```python
>>> distillation_loss(student, teacher, labels, T=2.0, alpha=0.5)
soft KL 0.9596 + hard CE 4.3698 -> 2.6647
>>> distillation_loss(teacher, teacher, labels, T=2.0, alpha=1.0)[1]
0.0e+00                                  # student == teacher -> zero KL ✓
```

**Where it is used in practice** (`03` §11.6): Qwen3 distils its 235B into the 0.6B–32B line; DeepSeek-V3 distils reasoning from R1; **Gemma 3 uses distillation for pretraining itself.** The bandwidth argument is simple — a one-hot label carries `log₂(V) ≈ 17` bits per token, while a full teacher distribution over 150k tokens carries far more. That is why a distilled small model beats the same model trained from scratch on the same tokens.

## 8.3 The landscape — black-box vs white-box, off-policy vs on-policy

Two axes organize every distillation method you will meet:

```
  AXIS 1  What do you have from the teacher?
    BLACK-BOX   only its TEXT (a frontier API, a model with a different tokenizer)
                → generate teacher outputs, SFT the student on them. Most "distil from GPT/Claude" work.
    WHITE-BOX   its LOGITS / per-token log-probs (an open model, same tokenizer)
                → match full distributions token by token (§8.2), or per-token log-prob rewards (§8.5)

  AXIS 2  Whose sequences are you training on?
    OFF-POLICY  data generated ONCE by the teacher, up front. Cheap, stable — but distribution mismatch:
                the student meets states at inference it never saw in training (exposure bias)
    ON-POLICY   the STUDENT generates; the teacher scores its tokens. Fixes exposure bias; costs rollouts.
                This shift is the single biggest "what changed" between DistilBERT-era KD and today.
```

**The methods, placed on the axes:**

| Method | Box | Policy | Divergence | Idea |
|---|---|---|---|---|
| Hinton KD (§8.2) | white | off | forward KL | The original: soft targets at temperature T |
| **SFT on teacher outputs** | black | off | CE on samples | The R1-distil recipe (§8.4). Add *rejection sampling* — keep only teacher samples a verifier or judge passes — and it is "verified distillation" |
| **MiniLLM** | white | on | **reverse KL** | Mode-seeking is better for a much smaller student; needs policy-gradient-style training with stabilizers (single-step regularization, teacher-mixed sampling, length normalization) |
| **GKD** (generalized KD) | white | on (or mixed) | generalized JSD | Student samples its own outputs, teacher provides token-level distributions, supervised-style loss — the practical sweet spot, and what `trl`'s `GKDTrainer` implements |
| **On-policy distillation** (§8.5) | white | on | per-token reverse KL as reward | Student samples; per-token advantage = `log π_teacher − log π_student`; trained with the GRPO loss without groups. Dense signal, ~10× cheaper than RL |
| f-DISTILL / adaptive KD | white | either | any f-divergence, or a learned forward/reverse mix | KL is one member of a family (JSD, total variation…), each with its own mode-seeking/mass-covering trade |

> **Why on-policy fixes what off-policy cannot.** A student trained only on the teacher's golden prefixes never practises recovering once *its own* earlier token was slightly off. In a 500-token reasoning chain, or a 30-turn agent workflow, that compounding drift matters far more than in single-token classification. On-policy training samples the student's actual mistakes and lets the teacher correct them *where they happen*. It is the same exposure-bias argument as PPO-vs-DPO (§6.4), applied to distillation.

## 8.4 Doubt — "what do DeepSeek and Qwen *actually* do to distil?"

The recipe that became the industry default is less about exotic divergences and more about a two-stage off-policy → on-policy pattern.

**DeepSeek-R1 → the distilled models (Jan 2025).**

```
  1. TEACHER DATA    DeepSeek-R1 generates long chain-of-thought traces on math / code / reasoning prompts,
                     plus general data:  ~800 K samples (≈600 K reasoning + ≈200 K non-reasoning)
  2. PURE SFT        fine-tune Qwen2.5 and Llama-3 bases (1.5B – 70B) on those traces. No RL. No KL matching.
  3. THAT'S IT       and it worked strikingly well, cheaply.
```

The paper's key empirical finding, stated explicitly: **for small and mid-sized models, distillation from a strong teacher beats running RL on the small model directly** — they RL'd a Qwen-32B and it lost to the distilled 32B. Llama-Nemotron reports the same. *Do not waste RL compute on small students; distil.* Later community variants add a temperature-scaled KL term when teacher logits are available, and a short post-SFT GRPO pass to recover self-verification behaviour that pure SFT compression can lose. *(A figure of "~$10 K" for this fine-tune circulates online; it is not in DeepSeek's release materials and should be treated as unverified. Real cost claims are different numbers: ~$5.6 M for V3 pretraining, and cheap third-party R1-style reproductions like Sky-T1 at ~$450.)*

**Qwen3 "strong-to-weak distillation" (2025) — deliberately two-phase, and closer to the GKD ideas:**

```
  PHASE 1  OFF-POLICY   SFT the student on teacher (Qwen3-235B-A22B / 32B) outputs generated in BOTH
                        /think and /no_think modes → basic reasoning skill + the mode-switch behaviour
  PHASE 2  ON-POLICY    the student samples its own rollouts; its logits are aligned to the teacher's
                        by minimizing KL — GKD-style on-policy logit matching at industrial scale
```

Used across the whole small lineup (0.6B → 30B-A3B) and the VL variants. The report's headline: this pipeline reaches **better** performance than running RL on each small model, at roughly **1/10 of the GPU-hours** of the RL run it replaced.

**Gemma 3** goes one step further and distils *during pretraining* — the student trains against a large teacher's sampled logits rather than one-hot tokens.

**The common pattern across labs:**

| Stage | Purpose |
|---|---|
| Big teacher generates CoT / response data | Cheap data source; no hand-labelling; rejection-sample for quality |
| Off-policy SFT on it | Cheap, stable, gets the student to a decent baseline fast |
| (Optional) on-policy KL / logit alignment | Fixes distribution mismatch, boosts quality further; needs a white-box teacher |
| (Optional) short RL pass | Recovers self-verification / robustness pure SFT loses; only worth it for mid/large students |

> **The practical answer to "what is SOTA":** off-policy SFT on teacher traces first, then an on-policy alignment pass, and RL only where the marginal compute is worth it. It is the two-stage recipe that became standard, not a new divergence.

## 8.5 Module — on-policy distillation (the student samples, the teacher grades per token)

Everything here is §5–§7 machinery with the reward swapped: instead of a verifier or an RM producing one scalar per response, the **teacher produces a per-token reward for free**.

```python
# PSEUDOCODE — per_token_logprobs is the per-token `tok` matrix from §5.2's sequence_logprobs
def on_policy_distill_step(student, teacher, prompts, mask_fn, clip=0.2):
    # 1. ROLLOUT — sample from the STUDENT, the model being trained (on-policy)
    y = student.generate(prompts)                                   # no grads

    # 2. SCORE — the teacher grades every token of the student's own sequence
    with torch.no_grad():
        lp_teacher = per_token_logprobs(teacher, prompts, y)        # [B,S]  the "judge"
        lp_old     = per_token_logprobs(student, prompts, y)        # [B,S]  saved, as in PPO phase B
    mask = mask_fn(y)                                               # completion tokens only

    # 3. ADVANTAGE — a single-sample estimate of −reverse-KL, PER TOKEN:
    #    reverse KL(student‖teacher) = E_{y~student}[ log π_s − log π_t ]; minimizing it means
    #    reward_t = log π_t(y_t) − log π_s(y_t). Dense: no group, no value model, no std normalisation.
    adv = lp_teacher - lp_old                                       # [B,S]

    # 4. UPDATE — the GRPO/PPO clipped surrogate with a per-TOKEN advantage (§7.3's grpo_loss, adv unsqueezed away)
    lp_new = per_token_logprobs(student, prompts, y)                # with grads
    ratio  = torch.exp(lp_new - lp_old)
    obj    = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
    return -(obj * mask).sum() / mask.sum()
```

**Why this is the cheap seat in the whole post-training stack:**

| | GRPO + RLVR (§7) | On-policy distillation |
|---|---|---|
| Signal per episode | ~1 bit (pass / fail), broadcast to every token | **`O(tokens)` bits** — one graded number per token |
| Rollouts per prompt | G = 8–16, so the group has a spread | **1** — the teacher is the baseline, no group needed |
| Models in memory | policy + (reference) | student + frozen teacher (any size; inference only, 2 B/param or 4-bit) |
| Reward can be hacked? | verifier seams only | no — the teacher's log-prob is not a proxy that can be gamed off-distribution |
| Where it fails | needs a verifier | needs a **white-box teacher with the same tokenizer** |

The teacher's per-token log-prob is *exactly* the localized credit PPO built a value model to approximate (§5.5), handed over for free. That is why Qwen3 reports ~1/10 of RL's GPU-hours for better results, and why 2025 reports of the method show it recovering a student's lost capabilities after narrow fine-tuning with a fraction of the compute of re-running SFT. **Requirements:** teacher and student must share a tokenizer (a same-family larger model, or your own earlier checkpoint); the teacher must be served fast enough to score rollouts (vLLM with `prompt_logprobs`).

## 8.6 Post-training on a budget — the one-GPU recipe

This is the section to implement from. Target: run **SFT → DPO → (GRPO + RLVR) → distillation** on a 7–8B (or 27–32B) open model with a single 24–80 GB GPU, using QLoRA / LoRA, and get most of what a lab gets from 5,000 GPU-hours out of tens.

### 8.6.1 Why LoRA is enough for post-training specifically

§3.3.1's rule — "style is low-rank, knowledge is not" — has a sharper 2025 form for post-training, from the *LoRA Without Regret* study (Schulman et al.) and from the RL-with-LoRA experience across the field:

| Finding | Consequence for you |
|---|---|
| For small-to-medium **SFT datasets** (≤ ~10⁵ examples), LoRA matches full FT in sample efficiency and final loss | Your 5–50 K-example SFT loses nothing to QLoRA |
| It **underperforms only when the dataset exceeds the adapter's capacity** (large-scale continued pretraining, or huge SFT corpora) | Knowledge injection still needs full FT / RAG |
| **Put LoRA on every linear layer, including the MLP / MoE projections.** Attention-only LoRA underperforms; where you put the adapters matters more than the rank | `target_modules = all-linear` (q, k, v, o, gate, up, down), not just q/v |
| **For RL, even rank 1 matches full FT.** A policy-gradient episode carries ~1 bit of information, so an RL run over 10⁴ episodes learns ~10⁴ bits — far below a rank-1 adapter's capacity | GRPO / on-policy distillation on a 4-bit base with r = 8–16 is not a compromise |
| **The optimal LoRA LR is ~10× the full-FT LR**, roughly independent of rank | SFT full-FT 1e-5 → LoRA 1e-4 to 2e-4; RL full-FT 1e-6 → LoRA 1e-5. Using full-FT LRs with LoRA is the commonest "LoRA doesn't work" bug |
| LoRA tolerates large batches less well than full FT | Keep effective batch modest (32–128 sequences) rather than chasing pretraining-style batches |

Rank guidance that follows: `r = 16` for SFT/DPO, `r = 8–16` for RL and on-policy distillation, `α = 2r` (or use rsLoRA scaling `α/√r` if you go above r = 64), dropout 0–0.05, `B = 0` init (§3.4).

### 8.6.2 Memory on one GPU

Extend §9.1's accounting to a frozen 4-bit base with bf16 adapters on all linear layers (~1% of params at r = 16), activations with gradient checkpointing, and vLLM's KV cache where rollouts are needed:

```
                       base (NF4)   adapters+Adam   activations (2K ctx, gc on)   fits on
  8B   SFT / DPO         ~4.5 GB       ~1.3 GB          3–6 GB                     16–24 GB card
  8B   GRPO + vLLM       ~4.5 GB       ~1.3 GB          3–6 GB  + vLLM KV 10–30 GB  one 48–80 GB GPU resident;
                                                                                    24 GB with engine sleep mode (§8.6.7)
  27B  SFT / DPO        ~14 GB         ~4.3 GB          6–10 GB                    one 24 GB card (tight) / 48 GB
  70B  SFT / DPO        ~35 GB        ~11 GB           10–15 GB                    one 80 GB GPU
```

Two tricks that make the multi-model rows of §5.3 / §7.6 collapse:

- **DPO's reference model is the same base with the adapter switched off.** `peft`'s `disable_adapter()` context gives you `π_ref` for free; `trl`'s `DPOTrainer(ref_model=None, peft_config=…)` does exactly this, and `precompute_ref_log_probs=True` computes the reference log-probs once over the dataset (§6.2) so training is one model, one forward per response.
- **GRPO's `π_old` / `π_ref` are adapter states too.** The reference is the pre-RL adapter (or the bare base); with β = 0 — the DAPO / Dr. GRPO default for RLVR — no reference is needed at all. The rollout engine (vLLM) holds its own copy of the base — bf16 in current vLLM, whose 0.28 release dropped the bitsandbytes loader (§8.6.7) — and receives the updated adapter each step.

### 8.6.3 The recipe, stage by stage

All GPU-hour figures are rough single-H100 orders of magnitude for an 8B model; multiply by ~3–4 for 27–32B. They are here so you can *plan*, not to be quoted.

**Stage 1 — SFT (QLoRA).** The imitation baseline; also stage 1 of black-box distillation when the data is teacher-generated (§8.4).

| | Choice | Why |
|---|---|---|
| Data | 5–50 K examples; for distillation, teacher traces **rejection-sampled** through a verifier or judge | §8.4: verified traces beat raw ones; 817 verified examples (LIMO) are enough for reasoning style |
| LoRA | r = 16, α = 32, all-linear, dropout 0.05 | §8.6.1 |
| LR / schedule | 1e-4 – 2e-4, cosine, 3% warmup | 10× the full-FT LR of §3.2 |
| Epochs / batch | 2–3 epochs; effective batch 32–64 seqs; packing **with** cross-example masking | §3.2, `03` §11.2 |
| Loss | response tokens only (`IGNORE` on the prompt) | §3.1 |
| Sanity | loss starts 1.0–1.5, ends 0.5–0.8; near `ln(V)` → chat template mis-tokenized; < 0.3 → memorizing | §3.2 |
| Cost | 10 K × 1.5 K tokens × 3 epochs ≈ 45 M tokens → **~3–6 H100-hours** | throughput ~2–4 K tok/s with QLoRA + checkpointing; Unsloth ~2× faster |

**Stage 2 — DPO (LoRA, ref = adapter off).** Style, format reliability, refusal calibration, verbosity.

| | Choice | Why |
|---|---|---|
| Data | 5–20 K pairs. Cheapest sources: (a) *on-policy* pairs — sample 4–8 responses from your SFT model, rank with a judge or verifier, take best vs worst; (b) production thumbs → KTO instead of pairs | on-policy pairs avoid the distribution mismatch of §6.4 #1; DPO on someone else's model's outputs teaches less |
| Loss | DPO + NLL anchor on chosen (`rpo_alpha` ≈ 1 in `trl`), β = 0.1 | §6.4 #2: stops chosen-likelihood from falling |
| LR | 5e-6 full-FT → **5e-5 LoRA**, 1–2 epochs | §8.6.1 |
| Sanity | step-0 loss **0.6931** exactly (§6.2); track `rewards/chosen`, `rewards/rejected` and mean response length — if length climbs, switch to SimPO or add length control | §6.3 |
| Cost | two responses per example + a one-time reference pass → **~2× SFT per example, ~3–8 H100-hours** | |

**Stage 3 — GRPO + RLVR (LoRA + vLLM).** Only if (a) you have a verifier for your task, (b) the student is ≥ 7B, and (c) stage 1 already used a strong teacher — DeepSeek's finding (§8.4) is that for small models distillation beats RL, so RL is the *polish*, not the capability source.

| | Choice | Why |
|---|---|---|
| Verifier | a Python function: exact-match / sympy for answers, sandboxed tests for code, schema + semantic check for structured outputs, a state check for tool-use tasks | §7.5 — no weights in the reward path |
| Prompts | 1–5 K, **filtered to pass rate 0.2–0.8 on the current model**, re-filtered every few hundred steps | §7.3: std = 0 prompts contribute nothing |
| Group / lengths | G = 8, max completion ~4× the p95 reference-solution length (4 K for competition math), temperature 0.7–1.0 with the scorer tempered identically — measure the fraction of rollouts at the cap first | §7.2's currency: pay in rollouts; §7.9 for the cap and temperature; §7.7 for why the scorer must match |
| Loss | `grpo_loss` with β = 0 (no reference), clip 0.2, batch-token normalization or Dr. GRPO | §7.4 |
| LR | 1e-6 full-FT → **1e-5 LoRA**, r = 8–16 | §8.6.1: rank 1 would do |
| Engine | `trl` `GRPOTrainer(use_vllm=True)` colocated, or Unsloth's single-GPU GRPO; the adapter reaches the engine every step (trl merges it into the engine's weights; a `LoRARequest` hot-swap is the alternative) | §14.3, §8.6.7: the rollout engine is the real infrastructure |
| Sanity | reward mean rising, **fraction of zero-variance groups** below ~0.5, **response length** and **fraction at cap** falling, entropy not collapsing (DAPO's asymmetric clip if entropy dies), KL to the SFT adapter bounded | §7.4, §7.8, §7.9 |
| Cost | a step of 64 sequences at a 4K cap is ~3.5–4 min on an A100-class card (rollout ~1.5 min: decode is bandwidth-bound and floors at the longest sequence; two teacher-forced passes ~2 min), ~17 min on an A10G; 200–500 steps → **~12–35 A100-hours**, H100 somewhat less. Planning figures that assume 5–10 K tok/s from the engine at long caps are ~5× optimistic — measure one step (§8.6.7) | |

**Stage 4 — distillation, in two flavours.**

*Black-box (any teacher, including a frontier API):* this **is** stage 1 with teacher-generated, verifier-filtered data. Generate 4–8 samples per prompt at temperature ~0.7, keep the ones that pass the verifier or a critique-then-score judge (§4.3.1), SFT. This is the R1 recipe and it is where an 8B gets most of its reasoning. Cost = stage 1 + teacher inference.

*White-box on-policy (a same-tokenizer teacher: your 32B/70B sibling served in vLLM, or the model *before* a narrow fine-tune, to recover what it lost):* run §8.5. Configuration is stage 3's minus the verifier, with G = 1, `adv = lp_teacher − lp_old`, LR 1e-5, r = 8–16. Because every token carries signal, it converges in far fewer episodes than GRPO — budget **~1/10 of stage 3**, i.e. a few H100-hours — and it is the cheapest way to close most of the remaining gap to the teacher.

### 8.6.4 The order, and what to skip

```
  For a 7-8B student with a strong teacher available (the common case):

  1. black-box VERIFIED distillation SFT (stage 1 on teacher traces)     ← the capability
  2. DPO on on-policy pairs (stage 2)                                    ← format / style / safety
  3. on-policy white-box distillation (stage 4b) if a same-tokenizer teacher exists   ← close the gap, cheap
  4. GRPO + RLVR (stage 3) ONLY if you have a verifier and steps 1-3 plateaued        ← the polish

  Total: roughly 10-30 H100-hours end to end, versus DeepSeek-V3's 5,000 — and most of that gap is
  base-model size, not what you skipped.
```

Skip stage 3 entirely for a ≤ 3B student (DeepSeek: distil, don't RL). Skip stage 4b if no white-box teacher shares your tokenizer. Never skip stage 1: RL from a base model re-learns what imitation gives for free (§7.6).

### 8.6.5 Tooling

| Need | Tool | Note |
|---|---|---|
| SFT / DPO / GRPO / GKD trainers | `huggingface/trl` | `SFTTrainer`, `DPOTrainer(ref_model=None, peft_config)`, `GRPOTrainer(use_vllm=True)`, `GKDTrainer` for §8.3 on-policy white-box |
| 4-bit base + adapters | `bitsandbytes` (NF4) + `peft` | `LoraConfig(r=16, lora_alpha=32, target_modules="all-linear")` |
| 2× faster single-GPU everything | `unslothai/unsloth` | fused kernels, QLoRA, GRPO with vLLM sharing the GPU memory |
| Config-driven runs | `axolotl`, `LLaMA-Factory` | YAML for all of the above; good for reproducibility |
| Rollouts | `vLLM` | colocated with the trainer; serves the teacher for §8.5 with `prompt_logprobs` |
| Multi-node RL, later | `verl`, `OpenRLHF` | when a single GPU stops being enough |
| Hosted LoRA RL/distillation API | Tinker (Thinking Machines) | LoRA-only by design — the §8.6.1 findings are its premise |

**Serving what you trained.** Merge the adapter into a **bf16** copy of the base (`merge_and_unload` after loading the base in bf16 — never merge into the 4-bit weights, that merge is lossy), then quantize the merged model (AWQ / GPTQ / fp8) for vLLM — or serve unmerged with vLLM's multi-LoRA if you have many task adapters on one base (§3.3.1's multi-tenant pattern).

### 8.6.6 The checklist of things that go wrong

```
  SFT  loss starts near ln(V)                 chat template / tokenizer mismatch — labels mis-shifted
       loss < 0.3                             memorizing; too many epochs or duplicated data
       "LoRA didn't learn anything"           LR too low (used the full-FT LR) or adapters only on q/v
  DPO  step-0 loss ≠ 0.6931                   reference log-probs computed with a different mask / template / model
       chosen log-prob falling                add the NLL anchor (§6.4 #2)
       responses getting longer               SimPO, or length-controlled pairs
  GRPO zero gradient on most prompts          std = 0 — refilter prompts to the 0.2-0.8 pass-rate band;
                                            retry all-zero prompts and park the hopeless ones (§7.8)
       reward up, outputs degenerate          verifier is being spec-hacked (§7.5): harden the parser / sandbox
       entropy collapse                       DAPO asymmetric clip; lower LR; check the rollout temperature is the
                                            one the scorer uses (§7.7)
       vLLM weights stale                     adapter not synced after the step — reward will not move
       reward 0 on EVERY rollout, fluent text  engine served uninitialised weights (sleep level 2 without a
                                            reload) — level 1, or reload on wake; abort after N dead steps
       most rollouts hit the cap              temperature too high for this model (T=1 rambled, 0.7 didn't);
                                            scorer must divide logits by the same T (§7.7)
       first step trains at lr 0              HF linear-warmup lambda(0) = 0; start at lr/warmup
       OOM on a step that had fit before      micro-batch packed wider than the log-softmax slab assumes
                                            (§8.6.7 pitfalls); or a 6K retry sequence in the batch
       step lines never appear under `tee`   stdout block-buffered behind the engine's stderr; line-buffer it
  DIST teacher log-probs look random          tokenizer mismatch between teacher and student (§8.5 requirement)
  ALL  merged model worse than adapter        merged into the 4-bit base; redo the merge in bf16
```

### 8.6.7 GPU sizing for RL — the measured worked example (Qwen3-4B, A10G 24 GB and A100 40 GB)

§8.6.2's table is planning-grade. This is the same accounting done on a real run of GRPO + RLVR (QLoRA, colocated vLLM rollouts, 64 sequences per step, 4,096-token cap), with the numbers measured, so the *procedure* can be reused for any model. (A 4B student sits below §8.6.3's ≥ 7B guidance for RL as a capability lever; it is the sizing example here, not a recommendation.)

**Step 1 — the memory terms.** For a decoder with `L` layers, hidden `H`, `KV_heads`, `head_dim`, vocab `V`, sequence `S` = prompt + cap, micro-batch `B`:

| Term | Formula | Qwen3-4B, S = 4,096 | Note |
|---|---|---|---|
| training weights, NF4 | 0.5 B × params | **2.3 GB** | bitsandbytes; dequantized per layer on the fly |
| LoRA r = 16 all-linear + Adam | ~1% params × 16 B | **0.5 GB** | fp32 master + 2 moments |
| activations, gradient checkpointing | `L × B × S × H × 2 B` (+ one layer's full activations) | 1.7 GB at B = 2, 3.5 GB at B = 4 | linear in B × S |
| logits, bf16 | `B × S × V × 2 B` | **1.25 GB per sequence** (V = 152 K) | the term people forget; grows with vocab, not model size |
| fp32 log-softmax over the vocab | `B × S × V × 4 B` **if materialized** | 2.5 GB per sequence — plus autograd keeps it | **chunk it** (below) |
| engine copy (vLLM), bf16 | 2 B × params | **7.6 GB** | vLLM 0.28 has no bitsandbytes loader; the engine serves bf16 |
| engine KV cache | `2 × L × KV_heads × head_dim × 2 B` per token × tokens in flight | **147 KB/token** → 6.9 GB holds ~50 K tokens ≈ 16 sequences at 3K, 7 at 7K | paged; the engine queues beyond it (slower, not OOM) |
| CUDA contexts, fragmentation | | 2–3 GB | two processes (trainer + engine) |

**Step 2 — share the card between the two phases.** The rollout phase needs the engine at its peak (weights + KV) with the training model idle beside it; the training phase needs activations + logits with the engine idle. On 40 GB both fit resident (engine claims 0.5 of the card: 7.6 GB weights + ~11 GB KV; training ~15 GB at B = 4). On 24 GB they cannot: put the engine to **sleep** during the gradient steps and wake it before the next rollout.

```
  24 GB profile   micro-batch 2, engine util 0.70, sleep mode        measured peak, training phase: 11.6-11.8 GB
  40 GB profile   micro-batch 4, engine util 0.50, no sleep          estimate ~28 GB total, both resident
```

Two sleep-mode facts that cost a day each to learn: **level 1** parks the weights in CPU RAM (7.9 GB pinned — check the box has it) and restores them on wake; **level 2 discards them**, and after `wake_up()` the engine serves *uninitialised memory* — it does not crash, it emits fluent-looking noise to the cap with reward 0 on every rollout (53 dead steps before it was noticed). And call `torch.cuda.empty_cache()` before waking: the trainer's freed tensors sit in PyTorch's caching allocator, and the engine's re-allocation OOMs against them.

**Step 3 — the memory pitfalls that are not in the formulas.**

| Pitfall | Effect | Fix |
|---|---|---|
| full-vocab fp32 `log_softmax` | 2.5 GB/sequence, kept for backward: 10 GB at B = 4 | compute `x[t] − logsumexp(x)` in slabs under `torch.utils.checkpoint`; keep only the bf16 logits |
| slab sized by positions, not batch × positions | a packer that puts 8 short sequences in one micro-batch makes the slab 4× larger → OOM | slab = `tokens_per_slab // batch` positions, constant transient |
| one padded tensor for the whole batch | a single 4K rollout pads all 64 sequences to 4K; ~30% of compute is padding at mean length 2.5K | sort by length, pack micro-batches by a padded-token budget (`B × cap`), cap sequences per chunk |
| zero-advantage sequences in the passes | every rollout of a dead group is forwarded and backwarded for a gradient of exactly 0 | drop them; keep the normalizer at the full token count (bit-identical gradient) |
| allocator fragmentation across changing shapes | 3 GB "reserved but unallocated" at the OOM | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| vLLM sampling-kernel JIT on images without CUDA headers | engine start fails in `nvcc` | `VLLM_USE_FLASHINFER_SAMPLER=0` (greedy/plain softmax gains nothing from it) |

**Step 4 — time.** Decode is memory-bandwidth-bound: **step time ≈ engine weight bytes ÷ HBM bandwidth** (A10G, 600 GB/s, 7.6 GB → ~13 ms floor; measured ~25 ms at 20 concurrent sequences). Rollout wall-clock ≥ longest sequence × step time, so a 4K cap floors at ~100 s on the A10G whatever the batch. Training: **5 forward-equivalents** per token (π_old, forward, checkpoint recompute, backward ≈ 2) × `2 × params` FLOPs, at ~40% of peak with NF4 dequantization.

| Phase, 64 sequences at cap 4K | A10G 24 GB, measured | A100 40 GB, estimated |
|---|---|---|
| rollout | 5.6–8 min | 1–1.5 min |
| π_old + backward, micro-batch 2 / 4 | ~10 min | ~2 min |
| per step | **16–18 min** | ~3.5–4 min |
| 500 steps | ~6 days | ~30 h (~$50 on demand) |

The planning estimate before measuring was 35 s/step — it assumed 10 K tok/s from the engine and a 2K cap; the real decode rate at this cap is a twentieth of that. **Measure one real step before quoting a budget**; the first estimate was off by ~30×. Retrying dead prompts at a longer cap (6K) adds ~170 s per step that retries — again the longest-sequence floor.

**Step 5 — the procedure, generic.**

1. Weights: training copy (`bytes/param` for its dtype + 16 B/param on the trainable fraction); engine copy at the engine's dtype.
2. KV per token = `2 × L × KV_heads × head_dim × bytes`; multiply by (prompt + cap) × sequences in flight; this is what the engine's memory fraction buys.
3. Activations ≈ `L × B × S × H × 2 B` with checkpointing; logits `B × S × V × 2 B`; the vocab-wide fp32 transient, chunked.
4. Sum per phase, not overall; decide whether the two phases can be resident together or must alternate (sleep).
5. Time: bandwidth-bound decode step × longest sequence for rollouts; 5 forward-equivalents × padded tokens for training.
6. Add 2–3 GB of contexts and ~15% fragmentation. Then run a smoke step and read `torch.cuda.max_memory_allocated()`.

**Step 6 — the run-health log** that catches the failures of §7.8 and §7.9 within an hour instead of a day, per step: mean reward, fraction of zero-variance groups, mean completion tokens and fraction at the cap (measured on the *raw* rollouts, before any clipping), sequences actually trained on, micro-batches, rollout seconds, step seconds, peak memory, and the first 300 characters of one completion. Abort automatically after N consecutive steps with reward 0 in every rollout.


> **What this section is, in one sentence:** the entire post-training stack of §3–§8 is a handful of *losses* on top of one frozen backbone, and once the backbone is frozen and 4-bit, the "four models" of §5.3 are four adapter states on one GPU — the math does not change, only the bill.

---

# 9. Memory and Parallelism

## 9.1 Doubt — "why does training need 16 bytes per parameter?"

Inference needs 2 bytes per parameter (`01` §12.3). Training needs **eight times that**, and the arithmetic is not negotiable:

```
  weights          bf16                        2 bytes/param
  gradients        bf16                        2 bytes/param
  master weights   fp32 (the optimizer's copy) 4 bytes/param
  Adam m           fp32                        4 bytes/param
  Adam v           fp32                        4 bytes/param
                                              ───────────────
                                               16 bytes/param    + activations
```

**Why the fp32 master copy exists:** a bf16 weight has 8 mantissa bits. A typical update is ~1e-4 relative to the weight, which is far below bf16's resolution — so `w += update` in bf16 is a **no-op**. The optimizer keeps an fp32 copy, applies updates there, and casts down for the forward pass.

```python
def training_memory_bytes(n_params, *, zero_stage=0, dp=1, param_bytes=2,
                          master_fp32=True, optimizer="adamw", activation_bytes=0):
    opt_slots = {"adamw": 2, "adafactor": 0.1, "sgd_momentum": 1, "sgd": 0}[optimizer]
    params = n_params * param_bytes
    grads  = n_params * param_bytes
    master = n_params * 4 if master_fp32 else 0
    opt    = n_params * 4 * opt_slots
    if zero_stage >= 1: opt, master = opt / dp, master / dp      # shard optimizer state
    if zero_stage >= 2: grads  = grads / dp                      # + gradients
    if zero_stage >= 3: params = params / dp                     # + parameters
    return {"params": params, "grads": grads, "master": master, "optimizer": opt,
            "activations": activation_bytes,
            "total": params + grads + master + opt + activation_bytes}
```

```python
>>> for n, name in [(27e9, "Gemma-3-27B"), (671e9, "DeepSeek-V3"), (2e12, "a 2T model")]:
...     m = training_memory_bytes(n)
Gemma-3-27B        432 GB  = w   54 + g   54 + master  108 + opt   216
DeepSeek-V3     10,736 GB  = w 1342 + g 1342 + master 2684 + opt  5368
a 2T model      32,000 GB  = w 4000 + g 4000 + master 8000 + opt 16000
```

**Read the first row against the hardware.** A 27B model needs 432 GB to train. An H200 has 141 GB. **A 27B model does not fit on one GPU — not close.** Everything in the rest of §9 follows from that single fact.

## 9.2 Activations — the term that actually varies

Weights are fixed; activations scale with batch and sequence and are the knob you actually turn.

```python
def activation_bytes_per_layer(b, s, d, h, dtype_bytes=2, checkpointing="none"):
    if checkpointing == "full":
        return b * s * d * dtype_bytes                       # store the layer INPUT only
    if checkpointing == "selective":
        return b * s * d * dtype_bytes * 6
    return b * s * d * dtype_bytes * 34 + b * h * s * s * dtype_bytes * 5
```

| Checkpointing | Memory per layer | Recompute cost |
|---|---|---|
| **none** | `34·b·s·d` + `5·b·h·s²` | 0 |
| **selective** | `~6·b·s·d` | ~10% extra FLOPs — recompute only the cheap ops |
| ⭐ **full** | `b·s·d` | **~33%** extra FLOPs (one extra forward pass) |

The `5·b·h·s²` term is the attention score matrix and it is quadratic — at `s = 32k` it dominates everything. FlashAttention (`01` §4.7.8) removes it by never materializing the matrix, which is why long-context training is *only* feasible with it.

## 9.3 The five parallelism dimensions

```
  DP  DATA          replicate the model, split the BATCH
                    comm: all-reduce gradients, once per step        ← cheapest
  ZeRO/FSDP         DP + shard the optimizer / grads / params across the DP group
                    comm: all-gather params per layer, reduce-scatter grads

  TP  TENSOR        split each WEIGHT MATRIX across GPUs
                    comm: all-reduce ACTIVATIONS twice per layer     ← chattiest;
                                                                       NVLink only
  PP  PIPELINE      split LAYERS across GPUs
                    comm: send activations at stage boundaries       ← cheap; but bubbles

  EP  EXPERT        split MoE EXPERTS across GPUs
                    comm: all-to-all, twice per MoE layer            ← bursty, huge

  CP  CONTEXT       split the SEQUENCE across GPUs
                    comm: ring-pass K/V blocks                       ← for long context
```

## 9.4 ZeRO / FSDP — the memory ladder

```
                        params  grads  optimizer      per-GPU bytes/param (bf16+AdamW)
  plain DP                ✗       ✗        ✗                    16
  ZeRO-1 / FSDP-SHARD_OP  ✗       ✗        ✓ /N       16 → 2 + 2 + 12/N
  ZeRO-2                  ✗       ✓ /N     ✓ /N       16 → 2 + (2+12)/N
  ZeRO-3 / FSDP-FULL      ✓ /N    ✓ /N     ✓ /N       16 → 16/N
```

```python
>>> training_memory_bytes(27e9)["total"] / 1e9                       # plain
432
>>> training_memory_bytes(27e9, zero_stage=3, dp=8)["total"] / 1e9   # ZeRO-3, 8 GPUs
54                                       # now it fits one 8xH100 node comfortably
```

**The trade is communication.** ZeRO-3 must all-gather each layer's parameters *before* using it and free them after — so parameters cross the network every forward and every backward. Within a node (NVLink, 900 GB/s) that is nearly free; across nodes over InfiniBand it becomes the bottleneck. **The rule of thumb: shard within a node, replicate across nodes.**

## 9.5 Tensor parallelism

Column-then-row splitting makes the two matmuls of an FFN need only **one** all-reduce:

```
  h = x·W_up      W_up  split by COLUMNS   →  each GPU has a slice of h, no comm
  y = h·W_down    W_down split by ROWS     →  each GPU has a PARTIAL y
                                              all-reduce(y) ← the one communication
```

Attention splits by **head** (`01` §4.4): each GPU owns whole heads, so `QKᵀ` and `·V` need no communication and only `W_O` requires an all-reduce.

> **TP must stay inside the NVLink domain.** It communicates activations twice per layer — for a 64-layer model that is 128 all-reduces per forward pass. Over InfiniBand this collapses throughput, which is why TP degree is almost always ≤ 8 (one node) and why DeepSeek-V3 chose **zero** TP (§11.3).

## 9.6 Pipeline parallelism and the bubble

```
  NAIVE (bubble ≈ (P−1)/(M+P−1) of the time)
  GPU0  F1 F2 F3 F4 ····························  B4 B3 B2 B1
  GPU1     F1 F2 F3 F4 ····················  B4 B3 B2 B1
  GPU2        F1 F2 F3 F4 ··········  B4 B3 B2 B1
  GPU3           F1 F2 F3 F4  B4 B3 B2 B1
                             └──┬──┘
                          idle time = the BUBBLE

  1F1B / interleaved: start backward as soon as possible; assign each GPU several
  non-contiguous layer chunks → smaller bubble, more communication
```

Bubble fraction is `(P−1)/M` for `M` micro-batches, so more micro-batches means less waste — but each micro-batch holds its own activations, so the bubble trades directly against memory. **DualPipe** (§11.3) attacks it from the other side: overlap the communication of one direction with the computation of the other.

## 9.7 Module — `parallel_plan`

```python
def parallel_plan(n_params, n_gpus, gpu_mem_gb, *, param_bytes=2, layers=64):
    # Smallest (tp, pp) that fits, then spend what's left on data parallelism.
    need = training_memory_bytes(n_params, param_bytes=param_bytes)["total"]
    cap = gpu_mem_gb * 1e9 * 0.75                     # 25% headroom for activations
    for tp in (1, 2, 4, 8):                           # TP first — it is NVLink-bound
        for pp in (1, 2, 4, 8, 16, 32):
            if tp * pp > n_gpus or layers % pp: continue
            if need / (tp * pp) <= cap:
                return {"tp": tp, "pp": pp, "dp": n_gpus // (tp * pp),
                        "gb_per_gpu": need / (tp * pp) / 1e9}
    return None
```

```python
>>> parallel_plan(671e9, n_gpus=2048, gpu_mem_gb=80)
{'tp': 8, 'pp': 32, 'dp': 8, 'gb_per_gpu': 41.9}
```

> **This planner is deliberately naive** — it optimizes memory and ignores interconnect topology, MoE expert placement, and the pipeline bubble. Real planning is a search over the 5-D space against a communication cost model, and §11.3 shows what a hand-tuned answer looks like when the naive one is wrong.

---

# 10. Training a 27B Model

Gemma-3-27B-class. This is the largest scale a well-funded team can realistically run.

## 10.1 The budget

```
  N = 27e9 params,  D = 14e12 tokens        (Gemma 3 27B's actual token count)
  FLOPs = 6·N·D = 6 · 27e9 · 14e12          = 2.27e24
  at 400 TFLOP/s effective (40% MFU on H100) = 5.67e9 GPU-seconds
                                             = 65,600 H100-days
  on 512 H100s                               ≈ 128 days       ← too long
  on 2,048 H100s                             ≈ 32 days        ← the realistic plan
```

## 10.2 The configuration

| | Choice | Why |
|---|---|---|
| **Cluster** | 2,048 × H100-80GB = 256 nodes | 32 days wall clock |
| **Precision** | bf16, fp32 master, fp8 GEMMs optional | §2.5 |
| **Memory** | 432 GB needed vs 80 GB/GPU | §9.1 |
| ⭐ **Parallelism** | **TP=8 (in-node) × PP=1 × DP=256 with ZeRO-1** | ~14 GB/GPU for states — (2 + 2 + 12/256) × 27B/8 (§9.4); TP stays on NVLink |
| **Checkpointing** | selective | ~10% FLOPs to halve activation memory |
| **Batch** | 4 M → 16 M tokens, ramped | `03` §9.5 |
| **Seq len** | 8,192, extended to 128 k in a late stage | `03` §10.2 |
| **LR** | 3e-4 peak, WSD, 2,000-step warmup | §2.3 |
| **Optimizer** | AdamW β=(0.9, 0.95), wd 0.1, clip 1.0 | §2.2 |
| **Checkpoint every** | ~30 min | §12.2 |

**Why TP=8 and not ZeRO-3 across everything:** ZeRO-3 over 2,048 GPUs would all-gather parameters across InfiniBand for every layer. TP=8 confines the chatty communication to NVLink inside one node, and ZeRO-1 across nodes only communicates gradients once per step.

## 10.3 Post-training the 27B

| Stage | Config | Cost |
|---|---|---|
| SFT | 1 M examples × 3 epochs, LR 1e-5, 64 GPUs | ~2 GPU-days |
| Reward model | 100 k pairs, LR 5e-6, initialized from the SFT model | ~1 GPU-day |
| DPO **or** GRPO | 100 k prompts, `β`=0.1, LR 5e-7 | ~10 GPU-days |

**Post-training is ~0.02% of the pretraining cost** and is responsible for most of the difference a user perceives. This asymmetry is the central economic fact of §13.

---

# 11. Training a 2T Model — DeepSeek-V3 as the Case Study

DeepSeek-V3 (671B total, 37B active) is the best-documented trillion-scale training run in existence, and its published numbers are startling.

## 11.1 The actual numbers

| | Value |
|---|---|
| Cluster | **2,048 NVIDIA H800** GPUs (NVLink in-node, InfiniBand across) |
| Pre-training | **2,664,000** GPU-hours |
| Context extension | 119,000 GPU-hours |
| Post-training | **5,000** GPU-hours |
| **Total** | **2,788,000 GPU-hours** |
| **Cost** | **$5.576 M** at an assumed $2/GPU-hour |
| Per trillion tokens | **180 k GPU-hours** = 3.7 days on the 2,048-GPU cluster |
| Stability | *"we did not experience any irrecoverable loss spikes or perform any rollbacks"* |

> **$5.6 M for a frontier-class model.** For comparison, Llama-3-405B's pretraining took 30.84 M H100 GPU-hours (Meta model card) — about **11×** DeepSeek-V3's. The gap is not hardware; it is MoE (37B active, not 405B dense), fp8, and the systems work below.

## 11.2 What made it cheap

| Lever | Effect |
|---|---|
| ⭐ **MoE** | `6·N·D` uses **active** parameters — 37B not 671B (`03` §0.2). ~18× less compute than a dense model of the same capacity |
| ⭐ **FP8 GEMMs** | Roughly 2× throughput vs bf16, with measured **loss error < 0.25%** |
| ⭐ **DualPipe** | Overlaps computation and communication so all-to-all is nearly free |
| **MLA** | Smaller KV cache (`01` §5.1) → larger batches fit → better MFU |
| **No tensor parallelism** | Eliminates the chattiest collective entirely (§11.3) |
| **Aux-loss-free balancing** | No auxiliary gradient degrading the LM objective (§2.6) |

## 11.3 Doubt — "why did DeepSeek-V3 use no tensor parallelism at all?"

Their published configuration is:

```
  16-way  PIPELINE parallelism  (PP)
  64-way  EXPERT   parallelism  (EP), spanning 8 nodes
  ZeRO-1  DATA     parallelism  (DP)
  ZERO-way TENSOR  parallelism        ← deliberately none
```

The reasoning follows directly from §9.3's communication table:

1. **MoE already provides the sharding TP would have provided.** With 64-way EP, each GPU holds a small subset of the 256 experts — and experts are where the parameters are (`01` §14.6). The FFN is *already* split across devices; adding TP would split the small remainder.
2. **TP and EP compete for the same interconnect.** EP's all-to-all is bursty and enormous. Adding TP's two all-reduces per layer on top would saturate the fabric — and their DualPipe design is specifically built to hide the all-to-all behind computation, which only works if the fabric is not already busy.
3. **TP's benefit is memory, and PP+EP already delivered it.** With PP=16 each GPU holds 4 of 61 layers, and with EP=64 only a slice of each MoE layer's experts. Memory was solved without TP.

> **The general lesson: parallelism dimensions are not additive, they compete.** Every extra dimension adds a collective on the same wires. The best configurations use the *fewest* dimensions that fit in memory — and MoE models get one of them (EP) for free from their architecture.

## 11.4 Scaling the plan to 2T

```
  N_total = 2e12, N_active ≈ 50e9 (~40× sparse — more aggressive than DeepSeek-V3's 18×), D = 20e12 tokens

  compute  = 6 · 50e9 · 20e12                        = 6.0e24 FLOPs
  memory   = 2e12 × 16 bytes                         = 32,000 GB of optimizer states
             ÷ 80 GB/GPU                             = 400 GPUs JUST TO HOLD IT
  at 40% MFU on 8,192 H100s                          ≈ 21 days
```

The memory line is the one that reshapes the design: **a 2T model needs ~400 GPUs before a single activation is stored.** That forces EP as the primary dimension, and it forces the training cluster and the *serving* cluster (`05` §11.5) to look alike — both are dominated by all-to-all bandwidth in a high-bandwidth domain.

---

# 12. Running the Job

## 12.1 Failures are the normal case

A 54-day snapshot of Llama 3's 405B run on 16,384 H100s saw **466 job interruptions** — 419 of them unexpected, roughly **one every 3 hours**. At that scale, hardware failure is not an exception path; it is a design constraint.

| Failure | Handling |
|---|---|
| GPU falls off the bus / ECC error | Detect, evict the node, restart from checkpoint on a spare |
| NCCL hang | Watchdog timeout → kill → restart |
| Loss spike | Roll back to a good checkpoint and **skip the offending data range** |
| Silent data corruption | The hardest — detect via checksum on shards and by loss-curve anomaly |
| Straggler node | Slows every step (collectives are synchronous); detect via per-rank step-time telemetry |

## 12.2 Checkpointing

```
  full state = weights + optimizer (m, v) + master fp32 + RNG + dataloader step (`03` §9.7)

  27B  ->  432 GB per checkpoint
  671B -> 10.7 TB per checkpoint       ← at 10 GB/s that is 18 minutes of writing
```

**The rule:** checkpoint interval ≈ 2× MTBF-recovery cost. Every 15–60 minutes in practice, using asynchronous and sharded writes so the GPUs do not stall. Note the dataloader step must be in the checkpoint — otherwise a restart re-shows data and silently corrupts the epoch schedule (`03` §9.6).

## 12.3 The metrics that matter

| Metric | Healthy | Meaning |
|---|---|---|
| **loss** | smooth decline | start at `ln(V)` (§1.2) |
| ⭐ **grad norm** | stable, slowly declining | the earliest warning of instability |
| **MFU** | 35–50% | `achieved FLOPs / peak FLOPs`; below 30% means an infrastructure problem |
| **tokens/sec** | flat | a decline means a straggler or thermal throttling |
| **LR** | matches the schedule | catches resume bugs |
| **expert load** (MoE) | near-uniform | §2.6; skew costs latency and quality |
| **step time p99/p50** | < 1.2 | a high ratio means stragglers |

**MFU is the single number that tells you whether the infrastructure is working.**

```
  MFU  =  (6·N·D_step / step_time)  /  (n_gpus · peak_FLOPs)
```

35–50% is good for large dense models; MoE runs are typically lower because of all-to-all.

---

# 13. Why Are Models Improving?

The question this whole series builds toward. Four candidate levers:

```
  1. ARCHITECTURE   a better transformer            `01`
  2. SCALE          more parameters, more compute   §2
  3. DATA           better tokens                   `03`
  4. TRAINING ALGO  better objectives, post-training §3-§8
```

The 2026 evidence is more one-sided than most people expect.

## 13.1 The scoreboard — top open-weight models, August 2026

| | **Kimi K3** | **DeepSeek V4 Pro** | **GLM-5.2** |
|---|---|---|---|
| **Total params** | **2.8 T** | 1.6 T | ~744 B |
| **Active params** | ~50 B | 49 B | ~40 B |
| **Experts** | 16 of 896 | 384 routed + 1 shared | not disclosed |
| **Attention** | Kimi Delta Attention + attention residuals | Compressed Sparse Attention (CSA) + HCA | — |
| **Context** | 1 M | 1 M | 1 M |
| **Licence** | Modified MIT | MIT | MIT |
| **GPQA Diamond** | **93.5** | 90.1 | 91.2 |
| **SWE-bench Verified** | 76.8 | **80.6** | (77.8 on GLM-5.1) |
| **SWE-bench Pro** | — | — | **62.1** |
| **LiveCodeBench** | — | **93.5** (#1 globally) | — |
| **Terminal-Bench 2.1** | **88.3** | — | 82.7 |
| **Codeforces** | — | **3206** | — |
| **AA Intelligence Index** | **~57** | 44 (Max reasoning) | 51 |

*Scores as reported by third-party comparisons in July–August 2026; harnesses are not always matched across models, so treat cross-column differences of 1–2 points as noise. Verify before quoting.*

**For calibration, the same benchmarks two years earlier:** DeepSeek-V3-Base (Dec 2024) scored **41.9 GPQA** and **87.19 MMLU**. GPQA Diamond went from ~42 to ~93 in about twenty months — from below the "PhD-level" threshold to above it.

## 13.2 Lever 1 — architecture: **almost nothing**

Compare the `01` component list of a 2026 model with a 2023 one:

| Component | Llama 2 (2023) | Kimi K3 / DeepSeek V4 (2026) | Changed? |
|---|---|---|---|
| Attention | MHA | GQA / MLA / CSA / KDA | **the one real change** |
| Position | RoPE | RoPE (+ YaRN, higher base) | no |
| FFN | SwiGLU | SwiGLU | **no** |
| Norm | RMSNorm, pre-norm | RMSNorm, pre-norm (+ QK-norm) | marginal |
| Residual | additive | additive | **no** |
| Objective | next-token | next-token (+ MTP) | marginal |
| Sparsity | dense | **MoE** | **the second real change** |

**Two changes in three years**, and both are about *cost*, not capability:

- **MoE** decouples capacity from compute. It does not make a model smarter per FLOP; it makes far more parameters affordable at the same FLOPs. Every model in §13.1 is MoE with ~2–5% of parameters active.
- **Attention compression** (GQA → MLA → CSA/KDA) shrinks the KV cache. Its contribution to *quality* is roughly zero by design — GQA's whole selling point was "≈ MHA quality" (`01` §5.2). It buys context length and serving cost.

> **The transformer of 2026 is the transformer of 2017 with RMSNorm, RoPE, SwiGLU, GQA and MoE.** A `01`-reader from 2023 would recognize every box. **Architecture is not where the improvement came from.**

## 13.3 Lever 2 — scale: **decisively decoupled**

The naive story is "models got bigger." The numbers say otherwise:

```
  GPT-3 (2020)         175 B params, ALL active,  0.3 T tokens
  Llama-3-405B (2024)  405 B params, ALL active,   15 T tokens
  DeepSeek V4 (2026)   1.6 T params,  49 B ACTIVE,  ? T tokens
  Kimi K3 (2026)       2.8 T params, ~50 B ACTIVE,  ? T tokens
                                     └─────┬─────┘
                          frontier ACTIVE compute per token is back
                          down to 40-60 B since the MoE switch,
                          while quality rose sharply
```

Total parameters grew 16× from GPT-3; **active parameters per token did not follow — they peaked with dense Llama-3-405B in 2024, and frontier MoEs are back to ~40–60 B active, *below* GPT-3's 175 B.** And the training compute is *falling* in dollar terms — DeepSeek-V3 cost $5.6 M against Llama-3-405B's ~11× larger GPU-hour bill for a comparable model.

**So the improvement from 2024→2026 was not bought with compute.** Something else did the work.

## 13.4 Levers 3 and 4 — data and training algorithm: **almost all of it**

**Data (`03`):**

| Change | Evidence |
|---|---|
| Better filtering | FineWeb-Edu: **1.3 T filtered tokens beat 15 T unfiltered** at equal compute (`03` §4.6) |
| More tokens | Qwen3's 36 T vs Qwen2.5's 18 T — 2× on the *same* architecture family, and it beat DeepSeek-V3-Base on 14 of 15 benchmarks with **1/3 the parameters** |
| New sources | VLM-OCR of PDFs and synthetic generation, both "trillions of tokens" (`03` §1.3, §1.4) |
| Instance-level mixing | Qwen3 annotated **30 T tokens** on educational value, domain and safety (`03` §4.4) |

**Training algorithm (§3–§8) — and this is the biggest single factor:**

| Change | Effect |
|---|---|
| ⭐ **RLVR on verifiable rewards** | The reasoning-model generation. GPQA 42 → 93 tracks the arrival of RLVR almost exactly |
| ⭐ **GRPO** | Made RL affordable by deleting the value network (§7.2) — 4 models → 2 |
| ⭐ **Long-CoT / test-time compute** | The model spends more tokens thinking. This is a *training-induced behaviour*, not an architecture change |
| **Verified distillation** | 817 examples (`LIMO`) reach competitive reasoning (`03` §11.5) |
| **fp8 + DualPipe + aux-loss-free** | Made trillion-scale training cost $5.6 M instead of $50 M |

## 13.5 The verdict

```
  Contribution to the 2024 → 2026 improvement, roughly:

  TRAINING ALGORITHM   ████████████████████████████████████████  ~45%
     RLVR, GRPO, long-CoT, verified distillation

  DATA                 ████████████████████████████             ~30%
     filtering, synthetic, OCR, multilingual, instance-level mixing

  SCALE                ████████████                             ~15%
     more TOTAL params (via MoE), roughly flat ACTIVE compute

  ARCHITECTURE         ████                                     ~10%
     MoE and attention compression — and both are COST levers,
     which enabled the other three rather than adding quality directly
```

**Three claims worth defending:**

1. **The 2026 model is the 2023 model trained differently on better data.** §13.2's table is the evidence. If architecture were the driver, the component list would not be nearly identical.
2. **Architecture's contribution is real but indirect.** MoE did not make models smarter — it made 36 T-token training runs and 2.8 T-parameter models economically possible. It is an *enabler* of levers 2 and 3, which is why it is not zero.
3. **Post-training gives the largest return per dollar in the entire stack.** DeepSeek-V3 spent **0.2%** of its GPU-hours on post-training. Almost everything a user experiences as "this model got much better" is bought in that 0.2%.

> **The forward-looking read:** the levers with room left are the ones that are cheapest. Pretraining compute is constrained by data supply (`03` §1.6) and by cost; architecture has been stable for three years. RLVR is two years old, only works where verification is cheap, and **extending verification to new domains is the open frontier.** That is where the next jump comes from — not from a new attention variant.

---

# 14. The Assembled Training Loop

## 14.1 One loop, four objectives

```python
# PSEUDOCODE — sample_group/verify/cfg are the stand-ins listed in §14.3
def train(model, ref_model, loader, cfg, mode="pretrain"):
    opt = AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.wd)

    for step in range(cfg.total_steps):
        lr = wsd_lr(step, cfg.total_steps, cfg.warmup, cfg.lr)      # §2.3

        for _ in range(cfg.grad_accum):
            batch = next(loader)                                     # `03` §9.7

            if mode in ("pretrain", "sft"):                          # §1, §3
                logits = model(batch.ids)                            # `01` §15.1
                loss = causal_lm_loss(logits, batch.labels,           # labels differ:
                                      z_loss_coef=cfg.z_loss)        #  pretrain -> all
                                                                     #  sft -> response only
            elif mode == "dpo":                                      # §6
                pi_c  = sequence_logprobs(model(batch.chosen_in),   batch.chosen_labels)
                pi_r  = sequence_logprobs(model(batch.rejected_in), batch.rejected_labels)
                with torch.no_grad():
                    ref_c = sequence_logprobs(ref_model(batch.chosen_in),   batch.chosen_labels)
                    ref_r = sequence_logprobs(ref_model(batch.rejected_in), batch.rejected_labels)
                loss = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=cfg.beta)

            elif mode == "grpo":                                     # §7
                rollouts = sample_group(model, batch.prompts, G=cfg.group_size)
                rewards  = torch.tensor([[verify(r, g) for r in row]  # `03` §11.4
                                         for row, g in zip(rollouts, batch.gold)])
                adv = grpo_advantages(rewards)
                logp = per_token_logprobs(model(rollouts.ids), rollouts.labels)  # [B,S]: §5.2's tok matrix, not its sum
                loss = grpo_loss(logp, rollouts.logp_old, adv,
                                 rollouts.ref_logp, rollouts.mask, beta=cfg.kl_beta)

            if cfg.n_experts:                                        # §2.6
                loss = loss + cfg.aux_coef * model.aux_loss()

            (loss / cfg.grad_accum).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)      # §2.4
        opt.step(lr); opt.zero_grad()
```

**Four objectives, one loop, and the differences are exactly three lines.** That is the point of this file: pretraining and SFT share a loss and differ only in the label mask; DPO replaces the loss with a ranking objective over two forward passes plus a frozen reference; GRPO replaces it with a policy gradient over sampled rollouts scored by a verifier.

## 14.2 The verification run

`verify_04_training.py` (a companion script, not included in this repo) extracts every `python` block from this document, executes them, and checks each claimed number.

```
$ python3 verify_04_training.py

  ok  §1.2   uniform logits -> loss = ln(V) = 3.9120, PPL = V = 50.00 EXACTLY
  ok  §1.2   ln(V) table: GPT-2 10.82, Llama 3 11.76, Qwen3 11.93 — the step-0 sanity check
  ok  §1.5   masked loss 4.3079 (8/16 targets) vs unmasked 4.4496
  ok  §1.3   z-loss adds 0.00205
  ok  §2.2   hand-written AdamW == torch.optim.AdamW after 20 steps (max diff 1.19e-07)
  ok  §2.3   cosine [0.00, 1.00, 0.10] and WSD [0.00, 1.00, 0.00] match the printed values
  ok  §2.6   MoE aux loss 1.0248 (1.0 = perfectly uniform routing)
  ok  §2.6   aux-loss-free bias: overloaded expert -0.0010, starved expert +0.0010
  ok  §3.4   LoRA == frozen base at init (B=0); trainable 8,192/73,984 = 11.1%
  ok  §4.2   RewardModel -> (4,): one scalar per sequence, read at the last real token
  ok  §4.2   Bradley-Terry: right order 0.2014 < wrong order 1.7014
  ok  §5.2   sequence_logprobs: sum -16.827 over 4 positions, average -4.207 = sum/4
  ok  §5.4   GAE with a terminal reward only -> [0.815, 0.857, 0.902, 0.95, 1.0]
  ok  §5.4   PPO clip table reproduces exactly: ratio 20.09 gains nothing over ratio 1.22 (both -1.2000)
  ok  §5.4   k3 KL estimator is non-negative in both directions (k1 is not)
  ok  §5.3   PPO's 4 models ≈ 36 bytes/param vs 16 for pretraining -> 27B needs ~972 GB
  ok  §6.2   DPO at init (pi == ref) = -log(0.5) = 0.6931 EXACTLY
  ok  §6.2   DPO: prefers-chosen 0.6210 < tie 0.6931 < prefers-rejected 0.9432
  ok  §6.3   SimPO (no reference model) 0.3133
  ok  §7.3   GRPO advantages: 3/5 correct -> [0.82, -1.22, 0.82, -1.22, 0.82]
  ok  §7.3   GRPO advantages: 1/5 correct -> [-0.5, -0.5, 2.0, -0.5, -0.5] — the rare success gets +2.0
  ok  §7.3   all-correct group -> advantages = 0 -> NO gradient (the degenerate case)
  ok  §7.3   GRPO loss at ratio=1 is -mean(advantage) = 0 for a zero-mean group
  ok  §8.2   distillation: soft KL 0.9090 + hard CE 4.4699
  ok  §8.2   student == teacher -> soft KL = 0 ✓
  ok  §9.1   27B: 432 GB = 54 w + 54 g + 108 master + 216 opt (16 bytes/param) — vs 141 GB on an H200
  ok  §9.1   DeepSeek-V3 671B -> 10,736 GB; a 2T model -> 32,000 GB of state
  ok  §9.4   27B with ZeRO-3 over 8 GPUs -> 54 GB/GPU: fits one node
  ok  §9.2   activations b=1,s=8192,d=4096: none 23.8 GB/layer vs full checkpointing 0.067 GB (354x)
  ok  §9.7   parallel_plan(671B, 2048 x 80GB) -> tp=8 pp=32 dp=8, 41.9 GB/GPU
  ok  §11.4  a 2T model needs ~400 GPUs just to HOLD its optimizer state
  ok  §10.1  27B x 14T tokens = 2.27e+24 FLOPs = 65,625 H100-days -> 32 days on 2048 GPUs
  ok  §11.1  DeepSeek-V3: 2,664,000 + 119,000 + 5,000 = 2,788,000 GPU-h; x $2 = $5.576M
  ok  §11.1  2,664,000 GPU-h / 14.8T tokens = 180,000 GPU-h per trillion tokens ✓
  ok  §13.5  post-training was 5,000 / 2,788,000 = 0.2% of DeepSeek-V3's GPU-hours
ALL TRAINING CODE VERIFIED — every claimed number reproduced
```

*The script covers the doctests. The numbers in §7.7–§7.10 and §8.6.7 are measurements from a real GRPO run (Qwen3-4B, one GPU), not doctests, and are not re-derived by it.*

## 14.3 What this implementation is missing

| Missing | Why production needs it | § |
|---|---|---|
| **Distributed anything** | No DP/TP/PP/EP — the whole of §9 is described, not implemented | §9 |
| **Fused kernels** | `causal_lm_loss` materializes `[B,S,V]`; production uses a chunked fused CE | §1.4 |
| **Activation checkpointing** | Assumed in §9.2's math, not wired into the loop | §9.2 |
| **The rollout engine** | `sample_group` hides a full inference server — real GRPO runs vLLM alongside the trainer, sharing or alternating on the GPU | §7.7, §8.6.7 |
| **Async checkpointing** | 10.7 TB writes cannot block the step | §12.2 |
| **Failure recovery** | 466 interruptions in 54 days is the actual requirement | §12.1 |

The *mathematics* of every objective above is complete and verified. Everything in that table is about running it on 2,048 GPUs for a month without stopping.

---

# 15. Reference

**Optimization and pretraining**
- ⭐ **Decoupled Weight Decay Regularization (AdamW)** — Loshchilov & Hutter, 2017 (1711.05101) — §2.2
- **Adam** — Kingma & Ba, 2014 (1412.6980)
- **Mixed Precision Training** — Micikevicius et al., 2017 (1710.03740) — the fp32 master-weight argument of §9.1
- ⭐ **Tensor Programs V (μP)** — Yang et al., 2022 (2203.03466) — hyperparameter transfer across width
- **MiniCPM** — Hu et al., 2024 (2404.06395) — the WSD schedule of §2.3
- **PaLM** — Chowdhery et al., 2022 (2204.02311) — z-loss, and a candid account of loss spikes

**Parallelism and systems**
- ⭐ **Megatron-LM** — Shoeybi et al., 2019 (1909.08053) — tensor parallelism (§9.5)
- ⭐ **ZeRO** — Rajbhandari et al., 2019 (1910.02054) — the memory ladder of §9.4
- **PyTorch FSDP** — Zhao et al., 2023 (2304.11277)
- **GPipe** (1811.06965) · **PipeDream/1F1B** (1806.03377) — §9.6
- **Reducing Activation Recomputation** — Korthikanti et al., 2022 (2205.05198) — selective checkpointing
- **Ring Attention** — Liu et al., 2023 (2310.01889) — context parallelism
- ⭐ **DeepSeek-V3 Technical Report** — 2024 (2412.19437) — §11: DualPipe, fp8, the 2,788,000 GPU-hour breakdown
- ⭐ **The Llama 3 Herd of Models** — 2024 (2407.21783) — §12.1's 466 interruptions, 4-D parallelism

**Alignment**
- ⭐ **InstructGPT** — Ouyang et al., 2022 (2203.02155) — the SFT → RM → PPO pipeline
- **PPO** — Schulman et al., 2017 (1707.06347) · **GAE** — Schulman et al., 2015 (1506.02438)
- **Approximating KL Divergence** — Schulman, 2020 (blog) — the `k3` estimator of §5.4
- ⭐ **Direct Preference Optimization** — Rafailov et al., 2023 (2305.18290) — the §6.1 derivation
- **IPO** (2310.12036) · **KTO** (2402.01306) · **SimPO** (2405.14734) · **ORPO** (2403.07691)
- ⭐ **DeepSeekMath (GRPO)** — Shao et al., 2024 (2402.03300) — §7
- ⭐ **DeepSeek-R1** — 2025 (2501.12948) — RLVR at scale
- **DAPO** — 2025 (2503.14476) · **Dr. GRPO** — 2025 (2503.20783) — the §7.4 refinements
- **HelpSteer2** — Wang et al., 2024 (2406.08673) — the multi-attribute reward model of §4.3

**Efficient fine-tuning and distillation**
- ⭐ **LoRA** — Hu et al., 2021 (2106.09685) — §3.4
- **QLoRA** — Dettmers et al., 2023 (2305.14314) · **DoRA** — 2024 (2402.09353) · **rsLoRA** (2312.03732) · **LoRA+** (2402.12354)
- ⭐ **LoRA Without Regret** — Schulman et al., Thinking Machines, 2025 (blog) — the §8.6.1 findings: all-layer adapters, rank-1 suffices for RL, 10× LR
- **Distilling the Knowledge in a Neural Network** — Hinton et al., 2015 (1503.02531) — §8.1
- **MiniLLM** — Gu et al., 2023 (2306.08543) — reverse-KL distillation for LLMs
- ⭐ **On-Policy Distillation of Language Models (GKD)** — Agarwal et al., 2023 (2306.13649) — §8.3, `trl`'s `GKDTrainer`
- **f-DISTILL** — Wen et al., 2023 (2307.15190) — the f-divergence family
- ⭐ **On-Policy Distillation** — Lu et al., Thinking Machines, 2025 (blog) — per-token reverse-KL reward, §8.5
- ⭐ **Qwen3 Technical Report** — 2025 (2505.09388) — strong-to-weak distillation, §8.4; GSPO (2507.18071), §7.4
- **Gemma 3 Technical Report** — 2025 (2503.19786) — distillation during pretraining

**Code to read**
- ⭐ `karpathy/nanoGPT` — the whole of §1–§2 in one readable file
- `NVIDIA/Megatron-LM` — the reference implementation of §9.3–§9.6
- `huggingface/trl` — `SFTTrainer`, `DPOTrainer`, `PPOTrainer`, `GRPOTrainer`
- `volcengine/verl` · `OpenRLHF` — production RLHF/RLVR with a co-located rollout engine
- `huggingface/open-r1` — an open reproduction of the R1 RLVR recipe
- `deepspeedai/DeepSpeed` — ZeRO 1/2/3

---

**Next:** `05-inference-serving.md` — how the trained model is served.
