---
title: "02 — Modern LLM Architectures"
subtitle: "How Qwen3, Llama, DeepSeek, Mistral, Gemma, GPT-OSS and vision-language models instantiate the components — and what changed"
purpose: "Foundation for 08-ml-system-design.md §3 (LLM Inference & Serving Platform)"
companions:
  - "01-transformer-architecture.md — what the components are; this file covers which choices real models make"
  - "05-inference-serving.md — why those choices change serving cost (KV cache, parallelism, quantization)"
---

# 0. How to Use This File

`01-transformer-architecture.md` (**`01`** below) covered *what the components are*. This file covers *which choices real models make*, and — more usefully — **why those choices change what it costs to serve them.**

> **Verify configs before quoting them.** Numbers below are from published model cards and papers as of early 2026. Architecture details get revised, and vendors ship variants. For anything load-bearing, read `config.json` on the model repo — §6 explains how.

---

# 1. The Design Space

Every modern decoder-only LLM is the same skeleton with ~10 knobs. Knowing which knobs exist lets you read any new model release in five minutes.

| Knob | Options in 2026 | Serving consequence |
|---|---|---|
| **Attention KV heads** | MHA · **GQA** · MQA · **MLA** | **Directly sets KV-cache size → concurrency.** Biggest lever. |
| **Position encoding** | **RoPE** (+ YaRN/NTK scaling) · ALiBi | Sets max context and how gracefully it extends |
| **Activation** | **SwiGLU** · GeGLU · GELU | Minor; SwiGLU means 3 FFN matrices, not 2 |
| **Normalization** | **RMSNorm**, pre-norm · +QK-norm · +post-norm | Minor compute; QK-norm aids stability |
| **FFN type** | Dense · **MoE** (routed ± shared experts) | **Total vs active params** — memory vs compute decoupling |
| **Attention pattern** | Full causal · **sliding window** · alternating local/global | Sliding window caps KV growth at long context |
| **Embeddings** | Tied · untied | `V·d` params; matters below ~10B |
| **Vocab size** | 32k → 256k | Tokenizer fertility vs embedding cost |
| **Context length** | 8k native → 128k–1M extended | KV cache is linear in context |
| **Vision fusion** | concat (LLaVA/Qwen-VL) · cross-attn (Llama-Vision/Flamingo) · Q-Former resampler | Whether image tokens grow the *text* KV cache directly or live in a separate side cache — the biggest VLM-specific serving lever (`01` §16.6) |
| **Extras** | MTP · logit soft-capping · attention sinks | Model-specific |

**The two that dominate serving economics:** *KV-head scheme* and *dense vs MoE*. Everything else is second-order.

---

# 2. Master Comparison

## 2.1 Dense models

| Model | Params | L | `d` | Q heads | KV heads | `d_ff` | Vocab | Ctx | Attention | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| **Llama-3.1-8B** | 8.03 B | 32 | 4096 | 32 | 8 | 14336 | 128,256 | 128k | GQA | RoPE θ=500k |
| **Llama-3.1-70B** | 70.6 B | 80 | 8192 | 64 | 8 | 28672 | 128,256 | 128k | GQA | 8:1 GQA ratio |
| **Llama-3.1-405B** | 405 B | 126 | 16384 | 128 | 8 | 53248 | 128,256 | 128k | GQA | 16:1 GQA |
| **Qwen3-8B** | 8.2 B | 36 | 4096 | 32 | 8 | 12288 | 151,936 | 32k→128k | GQA + **QK-norm** | YaRN for 128k |
| **Qwen3-14B** | 14.8 B | 40 | 5120 | 40 | 8 | 17408 | 151,936 | 32k→128k | GQA + QK-norm | 5:1 GQA |
| **Qwen3-32B** | 32.8 B | 64 | 5120 | 64 | 8 | 25600 | 151,936 | 32k→128k | GQA + QK-norm | **8:1 GQA** |
| **Gemma-2-9B** | 9.2 B | 42 | 3584 | 16 | 8 | 14336 | 256,128 | 8k | GQA + **alternating SWA** | Logit soft-cap, tied emb |
| **Gemma-2-27B** | 27.2 B | 46 | 4608 | 32 | 16 | 36864 | 256,128 | 8k | GQA + alternating SWA | Pre **and** post norm |
| **Gemma-3-27B** | ~27 B | 62 | 5376 | 32 | 16 | — | 262,144 | 128k | **5:1 local:global** | Multimodal — SigLIP + Pan&Scan, §3.7 |
| **Mistral-7B** (v0.1) | 7.2 B | 32 | 4096 | 32 | 8 | 14336 | 32,000 | 8k | GQA + SWA(4096) | First mainstream GQA+SWA; v0.2 went to 32k ctx and dropped SWA |
| **Phi-4** | ~14 B | 40 | 5120 | 40 | 10 | 17920 | 100,352 | 16k | GQA | Data-curation-first |

## 2.2 MoE models

| Model | Total | Active | Ratio | L | Experts | top-k | Shared | Attention | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **Mixtral-8×7B** | 46.7 B | 12.9 B | 3.6× | 32 | 8 | 2 | 0 | GQA (no SWA) | The model that mainstreamed MoE |
| **Mixtral-8×22B** | 141 B | 39 B | 3.6× | 56 | 8 | 2 | 0 | GQA | |
| **Qwen3-30B-A3B** | 30.5 B | 3.3 B | 9.2× | 48 | 128 | 8 | 0 | GQA + QK-norm | Fine-grained |
| **Qwen3-235B-A22B** | 235 B | 22 B | 10.7× | 94 | 128 | 8 | 0 | GQA + QK-norm | Flagship |
| **DeepSeek-V3 / R1** | 671 B | 37 B | **18×** | 61 | 256 | 8 | **1** | ⭐ **MLA** | Aux-loss-free balancing, MTP |
| **Kimi K2** | ~1 T | ~32 B | ~31× | — | 384 | 8 | 1 | MLA | Extreme sparsity |
| **GPT-OSS-120B** | ~117 B | ~5.1 B | ~23× | 36 | 128 | 4 | 0 | Alternating dense/SWA | MXFP4 experts, attention sinks |
| **GPT-OSS-20B** | ~21 B | ~3.6 B | ~5.8× | 24 | 32 | 4 | 0 | Alternating dense/SWA | Runs on 16 GB |

> **Read the "Ratio" column as the industry's direction of travel.** Mixtral (2023) was 3.6×. DeepSeek-V3 (2024) is 18×. Kimi K2 (2025) is ~31×. Sparsity is increasing fast, because **quality tracks total parameters while cost tracks active parameters** — so the ratio is free quality, paid for in memory capacity and interconnect.

## 2.3 KV cache per token — the number that decides your concurrency

Using `2 × L × h_kv × d_h × bytes`, FP16:

| Model | Formula | **Per token** | 32k ctx | Concurrency on 1×H100 |
|---|---|---|---|---|
| Llama-3.1-8B | 2×32×8×128×2 | **128 KB** | 4.2 GB | ~15 seqs @32k |
| Qwen3-8B | 2×36×8×128×2 | **144 KB** | 4.7 GB | ~13 seqs @32k |
| Qwen3-32B | 2×64×8×128×2 | **256 KB** | 8.4 GB | limited — needs TP |
| Llama-3.1-70B | 2×80×8×128×2 | **320 KB** | 10.5 GB | needs TP≥2 |
| *Hypothetical 8B with MHA* | 2×32×32×128×2 | 512 KB | 16.8 GB | ~4 seqs @32k |
| **DeepSeek-V3 (MLA)** | latent `d_c`≈512 | **~70 KB** | ~2.3 GB | far higher despite 671B |

> **The MLA row is the headline.** DeepSeek-V3 has 671B total parameters and a *smaller* KV cache per token than an 8B Llama. KV cache is what limits concurrency, so MLA buys long-context serving capacity that dense GQA models cannot match. It is the most aggressive KV optimization shipped in production.

---

# 3. Family Deep Dives

## 3.1 Llama 3 — the reference implementation

**Why it matters:** the most-deployed open architecture, and the baseline everything else is described against.

```
RMSNorm (pre-norm)  →  GQA  →  RoPE (θ = 500,000)  →  SwiGLU  →  untied embeddings
```

**Design choices worth noting:**
- **RoPE base 500,000**, up from 10,000. Larger base = slower rotation = longer effective wavelengths, which helps long context — but θ alone does not get Llama 3.1 to 128k. That takes the `llama3` frequency-dependent `rope_scaling` (factor 8) plus long-context continued pretraining (`01` §3.4).
- **8 KV heads at every size.** 8B has 32 Q heads (4:1), 70B has 64 (8:1), 405B has 128 (16:1). **KV cache per layer is constant across the family** — a deliberate serving decision. Scaling up costs compute, not KV memory per layer.
- **Vocab 128,256** — a big jump from Llama 2's 32k. Better multilingual fertility, at the cost of 525M embedding params.
- **No biases** anywhere. Simplifies kernels; no quality cost.

## 3.2 Qwen3 — the current open workhorse

**Why it matters:** the widest size ladder available (0.6B → 235B), consistent architecture across it, permissive licence, strong multilingual. If you're picking an open model in 2026, this is usually the shortlist.

**The ladder:**
```
DENSE:  0.6B · 1.7B · 4B · 8B · 14B · 32B
MoE:    30B-A3B  (30B total, 3.3B active)
        235B-A22B (235B total, 22B active)
```

**Architecture:** GQA · RMSNorm pre-norm · SwiGLU · RoPE · **QK-norm** · no QKV bias (Qwen2 had it, Qwen3 dropped it) · tied embeddings at 0.6B / 1.7B / 4B, untied from 8B up.

**Three things that distinguish it:**

1. **QK-norm.** RMSNorm applied to Q and K before the score computation. Bounds attention logits, preventing the blow-ups that destabilize large-model training. Increasingly standard; Qwen3 adopted it across the family.

2. **Hybrid thinking modes.** A single checkpoint supports both a reasoning mode (long chain-of-thought before answering) and a direct mode, switchable at inference. **Serving consequence: output-length distribution is bimodal.** Thinking mode can emit thousands of reasoning tokens — your TPOT budget and KV planning must account for two very different workloads on the same endpoint.

3. **32k native → 128k via YaRN.** Not natively long-context; extension is a deployment-time config (`rope_scaling` in `config.json`). Enable it only when you need it — YaRN costs a little short-context quality.

**`30B-A3B` is the interesting one for serving.** 30B total but only 3.3B active means it *decodes* roughly at 3B speed while holding 30B of knowledge — and 30B at FP8 is ~30 GB, fitting one 80 GB H100 with room for KV. That combination (small-model latency, mid-model quality, single-GPU deployment) is unusually practical.

## 3.3 DeepSeek-V3 / R1 — the architectural frontier

**Why it matters:** the most technically interesting open model, and the clearest public description of serving a sparse frontier model. Three innovations, all serving-relevant.

**1. MLA — Multi-head Latent Attention.** Cache a low-rank latent `c_t` instead of K and V; reconstruct on the fly. The up-projection matrices are algebraically absorbed into `W_Q` and `W_O`, so reconstruction is nearly free. **~10× KV reduction versus MHA** — see the §2.3 table.

**2. Fine-grained MoE with a shared expert.** 256 routed experts (top-8) plus **1 shared expert every token always uses**. The shared expert absorbs common knowledge so routed experts can specialize; fine granularity gives combinatorial capacity (`C(256,8) ≈ 10¹⁴` routing combinations). First 3 layers stay dense — early layers do general feature extraction that doesn't benefit from specialization.

**3. Aux-loss-free load balancing.** Instead of an auxiliary loss that competes with the language-modelling objective, add a **per-expert bias to the routing scores only** (not to the gate weights), updated online from observed load. Balances the experts without distorting gradients.

**Plus MTP (Multi-Token Prediction):** trained to predict several future tokens. The extra heads double as **built-in speculative-decoding drafts** at inference — a training decision that pays off as serving throughput.

> **Serving profile:** 671B at FP8 ≈ 671 GB, so a replica spans 8–16 GPUs minimum. Memory-capacity-bound on total params, bandwidth-bound on 37B active — so it *decodes* like a 37B model. Expert parallelism is mandatory, and the all-to-all traffic wants an NVL72-class domain (`05` §11.3).

## 3.4 Mistral / Mixtral — sliding window and mainstream MoE

**Mistral-7B** (v0.1) popularized two ideas: **GQA** at a size everyone could run — which stuck — and **sliding-window attention (SWA)**, where each token attends only to the previous `W = 4096` tokens. Pure SWA did *not* stick: Mistral-7B-v0.2 dropped it (32k context, full attention), and Mixtral never used it. What survived is local/global *interleaving* (Gemma 2/3, §3.5; `01` §4.7).

```
Full causal              Sliding window (W=4)
q1 [·        ]           q1 [·        ]
q2 [··       ]           q2 [··       ]
q3 [···      ]           q3 [···      ]
q4 [····     ]           q4 [····     ]
q5 [·····    ]           q5 [ ····    ]   ← token 1 dropped
q6 [······   ]           q6 [  ····   ]
```

**Why it matters for serving:** KV cache stops growing past `W`. Memory becomes `O(W)` instead of `O(S)` — the difference between bounded and unbounded at long context. Information still propagates further than `W` because each layer shifts the window, giving an effective receptive field of `W × L`.

**Mixtral-8×7B** was the model that made MoE mainstream: 47B total / 13B active, matching or beating Llama-2-70B at ~1/5 the active compute.

## 3.5 Gemma 2 / 3 — Google's serving-shaped choices

**Alternating local/global attention.** Gemma 2 alternates sliding-window layers with full-attention layers; Gemma 3 pushes the ratio to **5:1 local:global**. Only 1 in 6 layers has an unbounded KV cache.

> **This is a pure serving optimization expressed in the architecture.** At 128k context, five-sixths of layers cache only a window. It cuts KV cache dramatically while a minority of global layers preserve long-range capability. Expect more models to do this.

**Other Gemma specifics:**
- **Logit soft-capping** — `logits = cap · tanh(logits/cap)` on attention and final logits, bounding magnitudes. (Note: it interacts badly with some FlashAttention kernels; engines often disable it, with a small quality cost.)
- **Pre *and* post norm** around each block — extra stability.
- **Tied embeddings** with a 256k vocab — at 9B, `V·d` is a large fraction, so tying saves real memory.

## 3.6 GPT-OSS — the open-weights OpenAI models

Notable for **MXFP4-quantized experts shipped as the native format** (not a post-hoc quantization), letting the 20B model run in ~16 GB and the 120B on a single 80 GB card. Also uses **attention sinks** — learned per-head bias terms that give attention somewhere to dump probability mass, stabilizing long-context and streaming behaviour.

## 3.7 Vision-Language Models — how the frontier VLMs attach an image (and video) encoder

**Why it matters:** almost every 2025/2026 flagship release is multimodal by default (Qwen3-VL, Llama 4, Gemma 3), and the vision front-end is where a second, independent set of serving-cost decisions lives — on top of everything §1–§3 already fixed for the text backbone. The mechanics (patchify, the bidirectional ViT, the fusion patterns, M-RoPE) are `01` §16; this table is the "which model does what" companion to that algorithm, same relationship §1–§3 above have to `01` §1–§14.

| Model | Vision encoder | Resolution scheme | Fusion (`01` §16.6) | Image tokens | Video |
|---|---|---|---|---|---|
| **LLaVA-1.5 / 1.6** | CLIP ViT-L/14(-336) | fixed 336² / AnyRes tiling (1.6) | concat + MLP projector | 576/tile | frame sampling only |
| ⭐ **Qwen2-VL / 2.5-VL / 3-VL** | native-res ViT (windowed attention from 2.5-VL) | naive dynamic resolution — no resize, no tiling | concat, 2×2 patch-merger | varies with resolution | native — **M-RoPE** with absolute-time temporal axis |
| **Llama 3.2 / 4-Vision** | ViT-H/14 + 8 extra gated layers | tiled | **gated cross-attention**, every 4th decoder layer | held in a side cache, not the text KV cache | limited (3.2) → expanding (4) |
| **Gemma 3** | SigLIP-400M | fixed 896² + **Pan & Scan** cropping | concat, avg-pooled to 256/crop | 256 × (1 + crops) | image-focused |
| **Pixtral / Pixtral Large** | native-res ViT, 2D-RoPE | native, no resize | concat, `[IMG BREAK]`/`[IMG END]` tokens | varies with resolution | image-focused |

**The one-line read:** **GQA/MLA decides how much the *text* KV cache costs per token (§2.3); the vision-fusion row decides whether images cost anything in that same cache at all.** Concat-style models (LLaVA, Qwen-VL, Pixtral, Gemma 3) pay for every image token in the ordinary text KV cache — same bytes/token as any other position. Cross-attention models (Llama-Vision) keep image tokens in a separate, much smaller side cache that the text-side KV budget never sees, at the cost of a more complex decoder (an extra sublayer every few layers, `01` §16.6.3).

> **Serving consequence, concretely.** A concat-style model looking at one Qwen2.5-VL-style dynamic-resolution image can add anywhere from a few hundred to several thousand tokens to the *prompt* — directly consuming the same KV budget §2.3's table sizes for pure text. Video multiplies this by the frame-sampling count (`01` §16.7): Qwen3-VL at 32 sampled frames already reaches roughly 8k visual tokens before any text is added, which is why frame budget and resolution policy are capacity-planning inputs, not just quality knobs.

---

# 4. What Changed, 2019 → 2026

| Component | 2019 (GPT-2) | 2023 (Llama 2) | 2026 (Qwen3 / DeepSeek-V3) |
|---|---|---|---|
| Norm | LayerNorm, post-norm | RMSNorm, pre-norm | RMSNorm + **QK-norm** |
| Position | Learned absolute | RoPE θ=10k | RoPE θ=500k + **YaRN** |
| Attention | MHA | MHA → GQA | **GQA / MLA** ± sliding window |
| Activation | GELU | SwiGLU | SwiGLU |
| FFN | Dense `4d` | Dense `8d/3` | **MoE**, 10–30× sparsity |
| Vocab | 50k | 32k | 128k–256k |
| Context | 1k | 4k | 128k–1M |
| Biases | Everywhere | Removed | Removed |
| Precision | FP32 | BF16 | **FP8 / MXFP4 native** |
| Vision | none | bolted-on adapter (LLaVA: frozen CLIP + MLP) | **natively multimodal** — dynamic resolution, M-RoPE, unified vocab (`01` §16) |

**Five trend lines worth being able to state:**

1. **Everything that can be removed, has been** — biases, mean-centering, post-norm. Simpler is faster and no worse.
2. **KV cache became the primary optimization target** — MHA → GQA → MLA → sliding window → alternating local/global. Because it caps concurrency.
3. **Sparsity is rising fast** — 3.6× (2023) → 18× (2024) → 31× (2025). Memory is cheaper than compute.
4. **Context grew 100×**, forcing RoPE scaling and windowed attention to exist at all.
5. **Quantization moved into the architecture** — models now *ship* as FP8/MXFP4 rather than being quantized afterwards.

---

# 5. Picking a Model for Deployment

| Constraint | Choice | Reasoning |
|---|---|---|
| Single 24 GB consumer GPU | Qwen3-8B or Llama-3.1-8B, INT4/FP8 | ~4–8 GB weights leaves ample KV |
| Single 80 GB H100, best quality | **Qwen3-30B-A3B (FP8)** | ~30 GB weights, decodes at ~3B speed |
| Single H200, dense 70B | Llama-3.1-70B **FP8** (~70 GB) | FP16 at 140 GB doesn't fit |
| Max quality, multi-GPU budget | DeepSeek-V3 / Qwen3-235B-A22B | Needs 8–16 GPUs; MLA keeps KV manageable |
| Very long context, tight memory | Gemma-3 or a sliding-window model | Bounded KV growth |
| Highest concurrency per GPU | Anything with **MLA or aggressive GQA** | KV per token *is* the concurrency limit |
| Latency-critical, narrow task | Small dense (1.7B–4B) + fine-tune | Beats a big model on one task at 1/20 the cost |

> **The heuristic:** for a fixed GPU budget, first ask *"what fits, with room for KV?"*, then *"what's the KV per token?"*, and only then *"what's the benchmark score?"* Model quality that you can't serve at your required concurrency is not useful quality.

---

# 6. Reading a `config.json`

Every HuggingFace model exposes this. It is the fastest way to size a deployment.

```jsonc
{
  "num_hidden_layers":       36,        // L
  "hidden_size":           4096,        // d
  "num_attention_heads":     32,        // h        (query heads)
  "num_key_value_heads":      8,        // h_kv  ← GQA ratio = 32/8 = 4
  "head_dim":               128,        // d_h
  "intermediate_size":    12288,        // d_ff
  "vocab_size":          151936,        // V
  "max_position_embeddings": 32768,     // native context
  "rope_theta":         1000000.0,      // RoPE base
  "rope_scaling":  { "type": "yarn", "factor": 4.0 },   // → 128k
  "tie_word_embeddings":  false,
  "hidden_act":       "silu",           // SwiGLU
  "rms_norm_eps":      1e-6,
  // MoE models add:
  "num_experts":            128,
  "num_experts_per_tok":      8,
  "moe_intermediate_size":  768
}
```

**Compute these four numbers immediately:**

```
1.  GQA ratio        = num_attention_heads / num_key_value_heads
2.  KV bytes/token   = 2 × L × h_kv × d_h × bytes_per_element
3.  Weight memory    = params × bytes_per_element
4.  Max concurrency  = (GPU_mem − weights − overhead) / (KV_per_token × ctx)
```

**Worked — Qwen3-8B on one H100 at 32k context, FP16:**
```
KV/token   = 2 × 36 × 8 × 128 × 2         = 147,456 B ≈ 144 KB
weights    = 8.2 B × 2                    ≈ 16.4 GB
KV pool    = 80 − 16.4 − ~4 (overhead)    ≈ 59.6 GB
tokens     = 59.6e9 / 147,456             ≈ 404,000 tokens
concurrency at 32k ctx                    ≈ 12 sequences
concurrency at 4k ctx                     ≈ 98 sequences
```
> **Note how hard context length hits concurrency** — 8× the context is 8× fewer concurrent users. This is why long-context endpoints are priced differently, and why sliding-window and MLA architectures matter commercially.

---

# 7. References

**Model reports**
- ⭐ **The Llama 3 Herd of Models** — Meta, 2024 (arXiv 2407.21783) — unusually detailed on architecture *and* infrastructure
- ⭐ **Qwen3 Technical Report** — Alibaba, 2025 (2505.09388) — the dense+MoE ladder, QK-norm, hybrid thinking
- ⭐ **DeepSeek-V3 Technical Report** — 2024 (2412.19437) — MLA, fine-grained MoE, aux-loss-free balancing, MTP, FP8 training
- **DeepSeek-V2** — 2024 (2405.04434) — where MLA is introduced and derived
- **Mistral 7B** — 2023 (2310.06825) — GQA + sliding window
- **Mixtral of Experts** — 2024 (2401.04088)
- **Gemma 2** — 2024 (2408.00118) — alternating local/global, soft-capping
- **Gemma 3** — 2025 (2503.19786) — 5:1 local:global, multimodal
- **Qwen2.5 / Qwen2** — (2412.15115 / 2407.10671) — the lineage Qwen3 builds on
- **Phi-4** — 2024 (2412.08905) — the data-curation-over-scale argument

**Vision-language models** (§3.7 — full algorithmic treatment in `01` §16)
- ⭐ **Qwen2-VL** — 2024 (2409.12191) · **Qwen2.5-VL** — 2025 (2502.13923) — naive dynamic resolution, M-RoPE, absolute-time video
- **Pixtral 12B** — 2024 (2410.07073) — native-resolution ViT, 2D-RoPE
- **Llama 3.2 Vision model card** (`meta-llama/llama-models`) — the gated cross-attention adapter
- **PaliGemma** — 2024 (2407.07726) — prefix-LM attention masking
- ⭐ **Visual Instruction Tuning (LLaVA)** — 2023 (2304.08485) — the token-concatenation pattern most open VLMs use

**Comparative and analytical**
- ⭐ **Sebastian Raschka — "The Big LLM Architecture Comparison"** (magazine.sebastianraschka.com) — **the single best side-by-side of modern architectures**, with diagrams
- **Karpathy — `nanoGPT` / `build-nanogpt`** — implement it once and all of this becomes concrete
- **HuggingFace Open LLM Leaderboard** + individual model cards — the authoritative configs

**Code**
- `huggingface/transformers` → `models/{llama,qwen3,mixtral,gemma2}/modeling_*.py` — read two side by side; the diff *is* the architecture comparison
- `huggingface/transformers` → `models/{qwen2_vl,llava,paligemma}/modeling_*.py` — the §3.7 fusion patterns, written for production
- `vllm-project/vllm` → `model_executor/models/` — the same models written for serving
- `meta-llama/llama3` · `QwenLM/Qwen3` · `deepseek-ai/DeepSeek-V3` — reference implementations

---

**Next:** `05-inference-serving.md` — the hardware, the serving stack, every optimization technique, and the full Q&A from the inference deep-dive.
