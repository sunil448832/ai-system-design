---
title: "05 — LLM Inference, Serving & Deployment"
subtitle: "Hardware, the serving stack, every optimization technique, and the full Q&A"
purpose: "Foundation for 08-ml-system-design.md §3 (LLM Inference & Serving Platform)"
companions:
  - "01-transformer-architecture.md — prefill/decode asymmetry, KV cache, MoE fundamentals"
  - "02-modern-transformer-architectures.md — GQA/MLA, modern MoE and long-context designs"
source: "Incorporates all 30 questions from an external inference Q&A transcript (indexed in §18), plus the quantization doubts from a separate vLLM scheduler/quantization Q&A (§7.3–§7.5); both transcripts are external sources not included in this repo"
---

# 0. The One Idea

Everything in this file descends from a single asymmetry established in `01-transformer-architecture.md` (**`01`** below) §11:

```
PREFILL          many tokens, one weight read      →  COMPUTE-bound   →  TTFT
DECODE           one token,   one weight read      →  BANDWIDTH-bound →  TPOT
```

At batch 1 with FP16/BF16 weights, decode does ~1 FLOP per byte of weights read (one multiply-add = 2 FLOPs per 2-byte parameter). An H100's roofline ridge point is ~295 FLOPs/byte. **Decode runs ~300× below the compute limit — the GPU is mostly idle, waiting on memory.**

Every technique below is one of three moves:

| Move | Techniques |
|---|---|
| **Raise arithmetic intensity** | Continuous batching, speculative decoding, MoE |
| **Move fewer bytes** | Quantization, GQA/MLA, KV-cache compression, prefix caching |
| **Stop the two phases fighting** | Chunked prefill, disaggregation |

---

# 1. Units — Getting the Arithmetic Right

Interviews and capacity plans both die on unit errors.

## 1.1 Decimal vs binary

| Decimal (SI) | Binary (IEC) |
|---|---|
| 1 KB = 10³ = 1,000 B | 1 KiB = 2¹⁰ = 1,024 B |
| 1 MB = 10⁶ = 1,000,000 B | 1 MiB = 2²⁰ = 1,048,576 B |
| **1 GB = 10⁹ = 1,000,000,000 B** | **1 GiB = 2³⁰ = 1,073,741,824 B** |
| 1 TB = 10¹² B = 1,000 GB | 1 TiB = 2⁴⁰ B = 1,024 GiB |

**~7.4% difference at GB scale.** A "1 TB" drive shows as ~931 GB in the OS because the OS reports GiB while the vendor sold TB. **GPU spec sheets use decimal** — an "80 GB" H100 is 80 × 10⁹ bytes.

**Magnitude ladder:** `1,000 K = 1 M` · `1,000 M = 1 B` · **`1,000 B = 1 T`**.

## 1.2 Parameters → memory

```
memory = N_params × bytes_per_param
```

| Precision | B/param | 1 B | 8 B | 70 B | 671 B | 2 T |
|---|---|---|---|---|---|---|
| FP32 | 4 | 4 GB | 32 GB | 280 GB | 2.7 TB | 8 TB |
| FP16 / BF16 | 2 | 2 GB | 16 GB | 140 GB | 1.34 TB | **4 TB** |
| FP8 | 1 | 1 GB | 8 GB | **70 GB** | 671 GB | **2 TB** |
| INT4 / NVFP4 | 0.5 | 0.5 GB | 4 GB | 35 GB | 336 GB | 1 TB |

**Memorize the FP16 shortcut: bytes ≈ 2 × params.** An 8B model is 16 GB; a 70B is 140 GB. From there every capacity question is one division.

---

# 2. The Hardware

## 2.1 GPU specifications

| GPU | HBM | Bandwidth | BF16 dense | FP8 | NVLink/GPU |
|---|---|---|---|---|---|
| **A100 80GB** | 80 GB HBM2e | 2.0 TB/s | ~312 TF | — | 600 GB/s |
| **H100 SXM** | 80 GB HBM3 | **3.35 TB/s** | ~989 TF | ~1,979 TF | 900 GB/s |
| **H200 SXM** | **141 GB** HBM3e | **4.8 TB/s** | ~989 TF | ~1,979 TF | 900 GB/s |
| **B200** | **192 GB** HBM3e | ~8 TB/s | ~2,250 TF | ~4,500 TF | 1.8 TB/s |
| **L40S** | 48 GB GDDR6 | 864 GB/s | ~362 TF | ~733 TF | none (PCIe) |

> **H200 is the same compute as H100 with 76% more memory and 43% more bandwidth.** For LLM inference — which is bandwidth- and capacity-bound, not compute-bound — that makes it disproportionately better than the spec-sheet compute suggests. A 70B FP8 model (70 GB) fits one H200 with KV headroom; on an 80 GB H100 it fits but leaves almost nothing for KV.

## 2.2 The memory hierarchy

```
┌──────────────────────────────────────────────────────────┐
│ REGISTERS        ~256 KB/SM        ~20 TB/s      ~1 cycle│
│ SHARED / L1      ~228 KB/SM        ~20 TB/s     ~30 cycle│  ← FlashAttention lives here
│ L2 CACHE         ~50 MB            ~7 TB/s     ~200 cycle│
│ HBM              80–192 GB      2–8 TB/s       ~400 cycle│  ← weights + KV cache
├──────────────────────────────────────────────────────────┤
│ NVLink (intra-node)             900 GB/s – 1.8 TB/s      │  ← TP, EP
│ PCIe Gen5 (host↔device)         ~64 GB/s                 │
│ InfiniBand NDR (inter-node)     ~50 GB/s                 │  ← PP, DP
│ Object storage (S3)             ~1–10 GB/s               │  ← weight loading
└──────────────────────────────────────────────────────────┘
```

**Each tier down is ~10× slower.** The single largest cliff in a cluster is **NVLink → InfiniBand: ~18–36×**. That cliff dictates every parallelism placement decision (§11.3–§11.4).

## 2.3 The roofline

```
ridge point = peak_FLOPS / peak_bandwidth

H100:  989e12 / 3.35e12  ≈  295 FLOPs per byte
```

```
 perf
  ▲                     ┌────────────── compute-bound (peak FLOPS)
  │                    ╱
  │                   ╱  ← prefill sits here
  │                  ╱
  │        ╱────────╱
  │      ╱   memory-bound (slope = bandwidth)
  │    ╱
  │  ╱  ← decode @ batch 1 (intensity ≈ 1)
  └──────────────────────────────────────▶ arithmetic intensity
        1        64        295
```

| Workload | Intensity | Bound by |
|---|---|---|
| Prefill, 2k tokens | ~2,000 (≈ S) | compute |
| Decode, batch 1 | ~1 | **bandwidth** |
| Decode, batch 64 | ~64 | bandwidth |
| Decode, batch 256 | ~256 | ≈ ridge — crossing into compute |

**Decode throughput ceiling per sequence:**
```
tokens/s ≤ HBM_bandwidth / model_bytes
```
| Model | H100 | H200 |
|---|---|---|
| 8B FP16 (16 GB) | ~209 tok/s | ~300 tok/s |
| 8B FP8 (8 GB) | ~419 tok/s | ~600 tok/s |
| 70B FP8 (70 GB) | ~48 tok/s | ~69 tok/s |

---

# 3. Memory Budgeting

```
GPU memory = weights + KV cache + activations + fragmentation + CUDA overhead
```

| Component | Size | Notes |
|---|---|---|
| **Weights** | `N × bytes` | Fixed for the process lifetime |
| **KV cache** | `tokens × KV_per_token` | **The variable that limits concurrency** |
| **Activations** | ~1–2 GB | Transient; larger during prefill |
| **CUDA context + fragmentation** | ~1–2 GB | Reserve it explicitly |

**Worked — Llama-3.1-8B FP16 on one H100:**
```
Total                                80.0 GB
− weights (8.03 B × 2)             − 16.1 GB
− CUDA context + activations       −  3.0 GB
                                   ─────────
KV pool                              60.9 GB

KV per token = 2 × 32 × 8 × 128 × 2 = 131,072 B = 128 KB
tokens       = 60.9e9 / 131,072     ≈ 465,000 tokens

→ @  4k context:  ~113 concurrent sequences
→ @ 32k context:   ~14 concurrent sequences
→ @128k context:    ~3 concurrent sequences
```

> **Context length is the concurrency killer.** 8× the context is 8× fewer users on the same GPU. This is why long-context endpoints cost more, why MLA and sliding-window architectures matter commercially, and why KV quantization (§7.5) is one of the highest-leverage optimizations available.

---

# 4. The Metrics That Matter

## 4.1 TTFT, TPOT, goodput

| Metric | Definition | Bound by | Feels like |
|---|---|---|---|
| **TTFT** | time to first token | prefill (compute) | responsiveness |
| **TPOT / ITL** | time per output token | decode (bandwidth) | streaming speed |
| **E2E** | `≈ TTFT + TPOT × output_tokens` | both | total wait |
| ⭐ **Goodput** | requests/sec **meeting their SLO** | both | what users actually get |

```
SLO example:  TTFT < 500 ms  AND  TPOT < 50 ms
goodput = requests/s satisfying BOTH
```

> **Why RPS and raw throughput mislead.** Aggressive batching maximizes tokens/sec while making every individual request slower. A configuration with 2× the throughput and 40% of requests violating TTFT has **worse goodput** — you are serving more tokens to fewer satisfied users. The metric was popularized by DistServe (2024) precisely because throughput-optimized systems were shipping unusable latency.

**Target guidance:** TPOT < 30 ms ≈ 33 tok/s, comfortably faster than human reading (~5–8 words/s). Below ~10 tok/s streaming feels laboured.

## 4.2 Benchmarking checklist

- Report **p50 / p95 / p99** for TTFT *and* TPOT — never means
- Report **goodput against a stated SLO**, not just throughput
- Use **realistic prompt/output length distributions** — uniform 128-token prompts tell you nothing
- Use **Poisson arrivals**, not a fixed-concurrency loop — bursty arrival is what breaks schedulers
- **Sweep the load curve** to find the knee; a single operating point hides the cliff
- Also track: batch occupancy, KV utilization, **preemption count**, prefix cache hit rate

---

# 5. Batching

## 5.1 Static → dynamic → continuous

```
STATIC BATCHING                        CONTINUOUS BATCHING
(request-level)                        (iteration-level)

R1 ████████░░░░░░  done, idle          R1 ████████
R2 ██████████████  done                R2 ██████████████
R3 ███████░░░░░░░  done, idle          R3 ███████
R4 ██████████░░░░  done, idle          R4 ██████████
   └──── batch blocked ────┘              ↑    ↑   ↑
   all wait for the longest               R5   R6  R7 join as slots free
```

**Static batching** pads to the longest sequence and waits for all to finish. GPUs idle while one long generation runs — utilization often 20–40%.

**Continuous (in-flight) batching** (Orca, 2022) schedules at **iteration level**: after every forward pass, finished sequences leave and queued ones join. **2–4× throughput, and it's table stakes** — the first thing to verify is actually enabled.

## 5.2 The scheduler's job

Each iteration the scheduler must decide:
- Which waiting requests to **admit** (bounded by free KV blocks)
- Whether to run a **prefill**, a **decode**, or a fused chunk (§10)
- Whether to **preempt** a running sequence under KV pressure

**Preemption** — when KV runs out, evict a sequence and either *recompute* its prefix later or *swap* it to CPU. It is the hidden source of tail latency: a preempted request's tokens stop entirely, then it must re-prefill. **Watch the preemption counter**; a rising count means your admission control is too loose.

---

# 6. PagedAttention

## 6.1 The problem it solves

Attention kernels historically required **contiguous** KV memory — address arithmetic `base + i × stride`. So engines pre-allocated a contiguous buffer sized to `max_seq_len` per request.

```
CONTIGUOUS PRE-ALLOCATION, max_len = 100

Request A: reserves 100 slots, generates 30  →  70 wasted
Request B: reserves 100 slots, generates 55  →  45 wasted
           ────────────────────────────────────────────
           200 reserved, 85 used  →  57% WASTED
```

Three kinds of waste: **internal fragmentation** (reserved-but-unused), **external fragmentation** (gaps too small to reuse), and **no sharing** (identical prefixes stored twice). Real systems used only **20–40%** of KV memory for actual tokens.

**Why not just grow the buffer?** Because with contiguity you *can't*:
1. Can't grow in place — the neighbouring memory is occupied
2. Realloc-and-copy every few tokens means enormous copies and bandwidth churn
3. Variable-size alloc/free creates exactly the external fragmentation you were avoiding
4. `cudaMalloc`/`cudaFree` are expensive and synchronizing — engines pre-allocate to avoid them on the hot path

## 6.2 The solution

**Break the contiguity assumption at the kernel level.** Store KV in fixed-size **blocks** (typically 16 tokens), physically scattered. Give each request a **block table** mapping logical position → physical block. The attention kernel gathers through that indirection.

```
BLOCKS OF 16, same workload

Request A (30 tokens) → 2 blocks  [0, 2]      waste 2 slots
Request B (55 tokens) → 4 blocks  [1, 3, 4, 6] waste 9 slots
                        ──────────────────────────────────
                        96 allocated, 85 used → 11% waste

PHYSICAL KV POOL:
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ A0 │ B0 │ A1 │ B1 │ B2 │free│ B3 │free│
└────┴────┴────┴────┴────┴────┴────┴────┘
```

**Properties:**
- Waste bounded to **< 1 block per sequence** → ~96%+ utilization
- **No external fragmentation** — all blocks are the same size
- **On-demand growth is O(1)** — grab any free block, append its index to the table
- **Copy-on-write sharing** — parallel samples and shared prefixes point at the same physical blocks

**Result: 2–4× throughput**, because reclaimed memory becomes batch size. Now standard in vLLM, TensorRT-LLM, SGLang, TGI.

> **It is exactly OS virtual memory**, applied to the KV cache — the same indirection trick as page tables, database buffer pools, and filesystem blocks.

---

# 7. Caching and Compression

## 7.1 Prefix caching / RadixAttention

**The property:** a token's K/V depends only on the tokens *before* it (`01` §10.1). So **identical prefixes produce identical KV** — across requests, not just within one.

**RadixAttention** (SGLang, 2024): keep KV alive after a request finishes, organized in a **radix tree** where edges are token sequences and nodes point at paged KV blocks.

```
                    (root)
                      │
        "You are a helpful assistant."   ← shared system prompt, cached once
                      │
         ┌────────────┴────────────┐
   "Summarize:"              "Translate:"
         │                          │
    ┌────┴────┐                ┌────┴────┐
  doc A     doc B            text X    text Y
```

Per request: match the **longest cached prefix** → reuse those blocks with **zero prefill** → prefill only the new suffix → extend or split the tree. Reference counting plus LRU eviction (leaves first); the cache uses only otherwise-idle memory.

**Impact: up to ~5× throughput on shared-prefix workloads** — agents, multi-turn chat, few-shot prompting — and large TTFT reductions. vLLM's automatic prefix caching is the block-hash analogue.

> **Practical rule that matters more than the algorithm: keep prompt prefixes byte-stable.** A timestamp, session ID, or random ordering *near the front* of the system prompt breaks matching from that point onward and silently destroys your hit rate. Put volatile content at the **end**. This is the single most common reason a real deployment sees 20% hit rate where it expected 80%.

## 7.2 KV cache tiering

HBM → CPU DRAM → NVMe (LMCache, Mooncake). Trades PCIe bandwidth for capacity when the reusable working set exceeds HBM. Worth it when prefix reuse is high and prefills are expensive.

## 7.3–7.5 Quantization

**One problem underlies every method here: a big outlier crushes the small values that share its scale.** Each algorithm is a different answer to that one problem, and the doubt blocks below walk each answer with the *same* four-weight example so they can be compared directly.

| Doubt | Where |
|---|---|
| What does quantizing a weight actually do? The scale-and-store formula | §7.3.1 |
| Which methods are in-flight (no data) and which need calibration? NF4, LLM.int8(), GPTQ, AWQ, FP8, NVFP4/MXFP4 | §7.3.2 |
| Show each method with numbers — and how AWQ's `s` is actually computed | §7.3.3 |
| The FP8 E4M3 formula, and why floating point sidesteps the outlier problem | §7.3.4 |
| After dequantizing, is the matmul just FP16 again? (no — and why that matters) | §7.3.5 |
| TurboQuant: rotate the outlier away instead of calibrating around it | §7.5.1 |
| How is the rotation `R` computed, is it for weights or activations, and is it shared? | §7.5.2 |
| Why does *one* blind matrix Gaussianize every vector? Incoherence and Beta(d/2, d/2) | §7.5.3 |
| What is a Hadamard matrix, and why does recursive doubling stay orthogonal? | §7.5.4 |

### 7.3 Weight quantization

| Format | B/param | Speedup | Quality | Notes |
|---|---|---|---|---|
| BF16 | 2 | 1× | reference | Baseline |
| ⭐ **FP8 (E4M3)** | 1 | **~2× decode** | near-lossless | **Production default on Hopper/Blackwell** |
| **NVFP4 / MXFP4** | 0.5 | ~3–4× | good | Blackwell; block scaling beats INT4 |
| INT4 (AWQ/GPTQ) | 0.5 | ~3× | measurable loss | When capacity-bound |

**The methods:**
- **GPTQ** — layer-wise, second-order (Hessian-based) error compensation. Calibration set required.
- **AWQ** — *activation-aware*: identify the ~1% of salient weight channels (by activation magnitude) and scale to protect them. Usually better than GPTQ at INT4.
- **SmoothQuant** — migrate activation outliers into weights via a per-channel scale, making W8A8 viable.
- **LLM.int8()** — the origin of the outlier problem: a few dimensions have huge activations that naive quantization destroys.

> **FP8 doesn't just halve memory — it doubles decode throughput**, because decode is bandwidth-bound and you're reading half as many bytes per token. This is the clearest statement that you understand the hardware.

#### 7.3.1 Doubt — "what does quantizing a weight actually do?"

The basic mechanism is three lines, and every method in this section is a variation on *which weights share a scale* and *how the scale is chosen*:

```
  scale   =  max(|w|) / max_level            max_level = 7 for signed INT4 (16 levels, −8…7), 127 for INT8
  w_int   =  round(w / scale)                ← what is STORED (4 or 8 bits)
  w_hat   =  w_int × scale                   ← what the matmul USES (reconstructed on the fly, §7.3.5)
```

The scale is stored alongside the integers, one per **group**. The group is the only thing that changes across the "plain" methods:

```
  per-tensor    one scale for the whole matrix        cheapest; one outlier anywhere ruins every value
  per-channel   one scale per output row/column       the usual INT8 choice
  per-block     one scale per 16–128 consecutive weights (NF4: 64, GPTQ/AWQ: 128, NVFP4: 16, MXFP4: 32)
                → an outlier only damages its own block; costs extra bits for the stored scales
```

**Why the outlier is the whole problem.** With a uniform grid the step size is `scale`, fixed for the group. One large value forces a large step, and every small value in the group is then rounded coarsely — or to zero. §7.3.3 shows this with numbers; §7.3.4 shows why floating-point formats escape it.

#### 7.3.2 Doubt — "which can be done in flight, and which need calibration-aware offline quantization?"

The split is **how much of your data the method must see before it can quantize**:

```
  NONE — fixed statistical assumption, applied at load time         NF4 (bitsandbytes / QLoRA)
  NONE — decided live, per forward pass, nothing persisted           LLM.int8(), FP8 dynamic
  SMALL calibration set, cheap (activation stats + a 1-D search)     AWQ, FP8 static
  LARGER calibration set, expensive (Hessian, iterative)             GPTQ, NVFP4/MXFP4 with MR-GPTQ-style scale search
```

**NF4 (NormalFloat4)** — applied the moment weights hit `.to(device)`. It assumes trained transformer weights are roughly zero-centred normal (empirically true — see `04` §2.5.1 for why they *start* that way). Instead of 16 evenly spaced INT4 levels, NF4's 16 levels sit at the **quantiles of a standard normal**, so bucket density is high near zero where the mass is and sparse in the tails. Per-block absmax scaling (block 64), plus **double quantization**: the fp32 block scales — one per 64 weights, a real overhead — are themselves quantized to 8 bits, saving ~0.4 bits/param. Zero calibration data: the codebook is derived from the distributional assumption, not from your model.

**LLM.int8()** — the truest "in-flight" method. Vector-wise INT8 scaling for the bulk of weights *and* activations, but at **every forward pass** it scans the incoming activation columns; any column whose max exceeds a threshold (default 6.0) is pulled out and computed in FP16, the rest go through an INT8 matmul, and the two partial results are summed. Data-dependent per batch, no offline pass, no persisted scales — `bnb.nn.Linear8bitLt` any model with zero prep. The cost: the FP16 outlier path is slower than a clean INT8 GEMM, so it is a *memory* tool for fine-tuning and local inference, not a vLLM production choice.

**GPTQ** (2022, Optimal Brain Quantizer lineage) — layer by layer, greedy, one weight column at a time. Run a few hundred to ~1,000 calibration samples through the model to collect each layer's input statistics, forming `H = 2XXᵀ` (the Hessian of the layer's output error with respect to its inputs). Quantize a column to the nearest grid point, measure the error, then use the **inverse Hessian to redistribute that error into the still-unquantized columns** — nudge them so the layer's *output* stays close, rather than rounding each weight independently. Tractable via a Cholesky factorization of `H` and processing columns in blocks of 128. Needs calibration data specifically to build `H`.

**AWQ** (2023) — same PTQ family, cheaper math. Skips the Hessian entirely. Insight: a small set of channels — identified by which ones consistently see **high-magnitude activations** on the calibration set — dominate output error if quantized badly. Rather than fixing the *weights* after the fact, AWQ optimizes a per-channel **scale**: multiply the salient weight channel by `s`, divide its activation by `s` (net-neutral on the math), then plain round-to-nearest. The salient weights, now occupying more of the grid, survive rounding. Calibration is a 1-D grid search over `s` (§7.3.3 shows the formula) — 5–10× faster to calibrate than GPTQ, and the practical INT4 default unless you are loading an existing GPTQ checkpoint.

**FP8 (E4M3)** — still floating point, so it keeps a wide dynamic range through the exponent instead of forcing a uniform grid; that is *why* it is near-lossless at 8 bits where INT8 is not. Two flavours: **static** — a per-tensor or per-channel scale computed once offline from a calibration set and baked into the checkpoint — and **dynamic** — the scale computed on the fly per tensor or per token from the actual max seen, no calibration at all. vLLM supports both, and dynamic activation scaling is the common default because the format is forgiving enough that calibration is rarely needed.

**NVFP4 / MXFP4** — block-microscaled 4-bit *float*, the newest tier, and Blackwell-only (no H100/H200/A100 has FP4 tensor cores, so on that hardware AWQ/GPTQ INT4 remain the 4-bit options):

| | MXFP4 (OCP standard) | NVFP4 (NVIDIA) |
|---|---|---|
| Element format | E2M1 (4-bit) | E2M1 (4-bit) |
| Block size | 32 | **16** |
| Block scale | E8M0 — power-of-two only, 8 bits | **E4M3 — a real FP8 value**, 8 bits |
| Extra scale | — | optional outer fp32 scale per tensor (two-level) |
| Effective bits/weight | 4 + 8/32 = **4.25** | 4 + 8/16 = **4.5** |

The finer block and the real floating-point scale are why NVFP4 preserves accuracy better — it spends a quarter-bit more storage on a much better-fitting scale. Calibration matters most here: naive per-block absmax leaves accuracy on the table, and the state of the art (MR-GPTQ — GPTQ's Hessian error compensation applied inside the FP4 block-scale search, plus a Hadamard rotation to spread outliers across each block, §7.5.4) sits at the *most* calibration-dependent end of the spectrum.

#### 7.3.3 Doubt — "not able to understand properly — show each with an example"

**Shared example.** One output neuron, four input weights, and their calibration activations:

```
  w = [ 0.12, −0.08, 0.15, 0.10 ]
  x = [ 0.5,   0.6,  0.4,  9.0  ]      ← channel 4 has a huge activation (9.0 vs ~0.5)

  true output  y = 0.12·0.5 − 0.08·0.6 + 0.15·0.4 + 0.10·9.0 = 0.06 − 0.048 + 0.06 + 0.90 = 0.972
```

Channel 4 contributes 0.90 of the 0.972 — it dominates, even though its *weight* (0.10) is not the biggest (0.15 is). Hold that thought.

**1. Naive INT4, round-to-nearest.** 16 levels (−8…7); scale from the biggest weight: `0.15 / 7 = 0.02143`.

```
  w1:  0.12/0.02143 =  5.60 → 6  → 0.1286   (err +0.0086)
  w2: −0.08/0.02143 = −3.73 → −4 → −0.0857  (err −0.0057)
  w3:  0.15/0.02143 =  7.00 → 7  → 0.1500   (exact)
  w4:  0.10/0.02143 =  4.67 → 5  → 0.1071   (err +0.0071)

  output = 0.1286·0.5 − 0.0857·0.6 + 0.15·0.4 + 0.1071·9.0 = 1.037     error = +0.065
```

**0.064 of that 0.065 — 98% — comes from channel 4 alone**: a tiny rounding error on `w4` multiplied by 9.0. RTN chose the grid from which *weight* was biggest, blind to the fact that `w4`'s error matters far more because of its activation.

**2. AWQ — the fix for exactly that blindness.** Calibration looks at *activations* and flags channel 4 as salient. Rescale: `w4 × s`, `x4 / s`, with `s = 2` — mathematically identical (`0.10·9.0 = 0.20·4.5`), but `w4` now occupies more of the grid:

```
  w' = [ 0.12, −0.08, 0.15, 0.20 ]     x' = [ 0.5, 0.6, 0.4, 4.5 ]     new scale = 0.20/7 = 0.02857

  w1:  4.20 → 4  → 0.1143   (err −0.0057)
  w2: −2.80 → −3 → −0.0857  (err −0.0057)
  w3:  5.25 → 5  → 0.1429   (err −0.0071)
  w4:  7.00 → 7  → 0.2000   (EXACT — the salient channel landed on a grid point)

  output = 0.1143·0.5 − 0.0857·0.6 + 0.1429·0.4 + (0.20/2)·9.0 = 0.963     error = −0.009
```

**A 7× reduction**, bought by moving the salient channel onto an exact grid point at the cost of slightly worse resolution on the channels that do not matter. No Hessian — just "find which activations are big, and spend the precision budget there."

**How `s` is actually computed** (the doubt): not hand-picked, and not a hard "top 1%" cut-off. Per input channel `j`:

```
  s_j  =  ( mean(|x_j|) )^α           mean over the calibration set = "how salient is channel j"

  for α in {0.0, 0.1, …, 1.0}:                 α = 0 → no rescaling (plain RTN);  α = 1 → fully proportional
      w' = w · s,  quantize w' by RTN
      measure the layer's output error vs the FP16 baseline on the calibration batch
  keep the α with the smallest error            one α per layer, shared across its channels
```

The power law gives big-activation channels a big `s` and small ones almost none; `α` controls how aggressively. In the toy, `s = 2` for channel 4 is what that search converges to.

**3. LLM.int8() — sidestep the problem instead of rebalancing it.** Threshold 6.0. At runtime it scans the activation vector, sees `x4 = 9.0 > 6.0`, and pulls that whole *column* out to be computed in FP16 — untouched, zero quantization error. Channels 1–3 go through INT8 (256 levels, negligible rounding):

```
  FP16 path:  0.10 · 9.0                    = 0.900   exactly
  INT8 path:  0.06 − 0.048 + 0.06           ≈ 0.072
  total                                     ≈ 0.972   essentially exact
```

Correction to the natural misreading: it is not "scale every weight except the outlier *weights*" — it is outlier **activation columns**, decided per forward pass, and those columns skip quantization entirely rather than getting a wider bin. That is why it needs no calibration set (decided live) and why it is slower than pure INT8 (a real FP16 matmul on however many columns trip the threshold).

**4. GPTQ — compensate *after* the fact instead of rescaling *before*.** Two-weight illustration: `w1 = w2 = 0.10`, and the calibration data shows `x1 ≈ x2` — the two channels move together, which is exactly what `H = E[xxᵀ]` encodes. Grid spacing 0.03.

```
  quantize w1 first:  nearest grid point to 0.10 is 0.09,  error e1 = +0.01
  naive RTN on w2:    also 0.09, e2 = +0.01  →  errors ADD:  y_err ≈ 0.01·x1 + 0.01·x2
  GPTQ on w2:         H says x1, x2 are correlated → pre-compensate: w2 ← 0.10 + 0.01 = 0.11
                      quantize THAT: nearest is 0.12, error −0.01
  final: w1 = 0.09 (under), w2 = 0.12 (over)  →  y_err ≈ 0.01·x − 0.01·x ≈ 0   the errors CANCEL
```

Your reading — "error from `w1` is pushed into `w2` before `w2` is quantized, so they land in different bins instead of the same one" — is right. The one refinement: the nudge is not a free-form push; it is the closed-form Optimal Brain Surgeon update `Δw = −e₁/[H⁻¹]₁₁ · H⁻¹[:,1]`, the *minimum-error* correction given the Hessian. Done column by column across the whole layer with the real `H` — which is why it needs calibration data to build `H` in the first place.

**5. FP8 — why it does not need rebalancing at all for many cases.** A tensor with a wide range: `[0.001, 0.15, 12.0]`.

```
  INT8 uniform:  scale = 12.0/127 = 0.0945;   0.001/0.0945 = 0.0106 → rounds to 0.  GONE — indistinguishable from zero.
  FP8 E4M3:      each value gets its OWN exponent, so relative precision (~1/8 from 3 mantissa bits) holds
                 whether the value is 0.001 or 12.0. The tensor's outlier cannot crush it.
```

**6. NVFP4 — one scale per 16 elements instead of one per tensor.** Block `[0.01, 0.02, 0.001, 0.9]` with a local outlier. A tensor-wide scale would size itself to the *global* max — if some other block holds 50, this block's small values are crushed exactly as in example 1. NVFP4 computes the scale **locally**: `0.9 / 6 = 0.15` (6 is E2M1's largest magnitude, `1.5 × 2²`), and only this block uses it. For a 32-element weight vector: two blocks, two E4M3 scales stored, each applied to its own 16 weights at compute time:

```
  scale_1 = max(|w[0:16]|)  / 6      w_int[0:16]  = round(w[0:16]  / scale_1)
  scale_2 = max(|w[16:32]|) / 6      w_int[16:32] = round(w[16:32] / scale_2)
```

Your "same as §7.3.1 but block-wise" is exactly right, with one addition: NVFP4 can add a *second*, coarser fp32 scale per tensor that renormalizes everything before the per-block E4M3 scales are computed — hence 4.5 effective bits versus MXFP4's 4.25.

> **The unifying thread.** Every method is solving "the big outlier crushes the small values on a shared scale". AWQ moves the outlier's *weight* onto the grid; LLM.int8() removes the outlier from quantization entirely; GPTQ makes the *other* weights absorb the error; FP8 sidesteps it with a per-value floating exponent; NVFP4 sidesteps it with *local* rather than global scales; TurboQuant (§7.5.1) rotates it away before quantizing at all.

#### 7.3.4 Doubt — "put the FP8 E4M3 formula here"

Layout: 1 sign + 4 exponent (bias 7) + 3 mantissa. `e` = the 4 exponent bits as an unsigned integer (0–15), `m` = the 3 mantissa bits as an unsigned integer (0–7):

```
  normal    (e ≠ 0):   value = (−1)^sign × 2^(e − 7) × (1 + m/8)
  subnormal (e = 0):   value = (−1)^sign × 2^(−6)    × (m/8)

  example:  sign 0, exponent 1000 (e = 8), mantissa 101 (m = 5)
            value = 2^(8−7) × (1 + 5/8) = 2 × 1.625 = 3.25
```

Max magnitude is **448** — E4M3 does not spend `e = 1111` on infinity as IEEE would; it keeps that pattern for one more finite value to extend the range (only one code is NaN). The gap between adjacent representable values is ~1/8 = 12.5% **of the magnitude**, constant at every scale because the exponent floats — versus INT8, whose *absolute* step is fixed regardless of magnitude. That constant relative precision is the entire reason FP8 handles wide-range tensors without a rescaling trick. (`04` §2.5.1 decodes the same format bit by bit, with the encode direction and the rounding examples.)

#### 7.3.5 Doubt — "so quantized values are stored, then dequantized and the matmul runs in FP16 like normal?"

Close, and the correction is the part that makes quantization worth doing at all.

**If you dequantized the whole weight matrix into a fresh FP16 tensor and then ran a normal FP16 matmul**, you would have read and written a full-size FP16 tensor anyway. Decode is bandwidth-bound (§0): the entire benefit of quantization is *fewer bytes moved from HBM*. A materialized full-size copy moves just as many bytes as never quantizing. It would be a wash.

**What actually happens — dequantization fused inside the kernel:**

```
  HBM: packed INT4 weights (small) ──┐
                                      ├──▶ unpack + dequantize IN REGISTERS ──▶ multiply-accumulate ──▶ fp32 accumulator
  HBM: FP16 activations             ──┘        (never written back to memory as an FP16 tensor)
```

- **INT4 weight-only (AWQ/GPTQ):** a custom GEMM kernel (Marlin, ExLlama, AWQ-GEMM) streams the packed 4-bit bytes from HBM into shared memory / registers, dequantizes *just in time* immediately before the multiply-accumulate, and discards the value. Bits moved on the weight side: 4 per weight instead of 16. The arithmetic still runs at FP16 speed — so INT4 weight-only gives the **memory-bandwidth win only**, which is exactly the decode win, and nothing for compute-bound prefill.
- **FP8 / NVFP4 on Hopper / Blackwell:** one step further — the tensor cores multiply FP8×FP8 and FP4×FP4 **natively**. No dequantization to FP16 for the multiply at all; only the *accumulation* over the long dot product happens in a wider format (fp32). That is why FP8/FP4 give **both** memory and compute gains, and why they are the production defaults where the hardware exists.
- **KV cache (FP8 KV, TurboQuant):** same principle. The compressed keys/values stay compressed in the cache; the attention kernel dequantizes each block as it streams through the score computation — or, for codebook methods, reconstructs the dot product directly from the indices without ever forming a float vector. Never a separate pass that regenerates a full FP16 KV cache; that would erase the reason you compressed it.

> **The one-line correction:** not *quantize → dequantize into a full float copy → compute normally*, but *quantize (fewer bytes at rest) → read compressed bytes → dequantize just-in-time in registers → multiply-accumulate at higher precision → discard, never write back*. The "just-in-time, in-register, never materialized" part is the win — and it is why the fused kernels (Marlin, ExLlama, FlashInfer) are a significant engineering effort *separate* from the quantization algorithm itself.

### 7.4 Activation quantization

`W8A8` (weights *and* activations in 8-bit) uses the FP8 tensor cores for real compute gains, not just memory savings. Harder because activations have outliers — hence SmoothQuant (and, for the same reason, the rotation methods of §7.5.2: QuaRot and SpinQuant rotate weights and activations *together* so both lose their outlier structure while `W'x' = Wx` is preserved exactly).

### 7.5 KV-cache quantization

Often the **highest-leverage** quantization, because KV is what caps concurrency.

| Format | KV size | Effect |
|---|---|---|
| FP16 | 1× | baseline |
| ⭐ **FP8 KV** | 0.5× | **2× concurrency**, negligible quality loss |
| INT4 KV (KIVI) | 0.25× | 4× concurrency, degrades long-context recall |
| TurboQuant (rotation + fixed codebook), 2–4 bit | 0.125–0.25× | no per-block scales to store; new (2026), not yet production-hardened — §7.5.1 |

> **Never ship quantization on perplexity alone.** Damage is uneven — it hits long-context, multilingual, and multi-step reasoning far harder than average next-token loss suggests. Gate on task-level evals, segmented by capability.

#### 7.5.1 Doubt — "explain the TurboQuant algorithm"

TurboQuant (Google Research, ICLR 2026) is a different beast from §7.3: its targets are **vector quantization for embeddings and KV-cache entries** — activations produced at runtime, one new vector per token per layer per head — not the static weight matrix. (A community port for weights exists; it is an extension, not the paper.) Because a KV vector does not exist until serve time, the method *must* be cheap and calibration-free. It solves the outlier problem **structurally** instead of detecting it.

**The core trick: rotate the outlier away, don't calibrate around it.** Every §7.3 method *detects* the dominant coordinate (AWQ's activation scan, LLM.int8()'s threshold) and treats it specially. TurboQuant **redistributes** it before quantizing, with a random orthogonal rotation. Take the exact scenario that broke naive INT4:

```
  x = [3, 0, 0, 0]                          all the energy on one axis

  R = ½ · [ 1  1  1  1 ]                     a normalized 4×4 Hadamard matrix (§7.5.4) — one clean
          [ 1 −1  1 −1 ]                     example of an orthogonal rotation
          [ 1  1 −1 −1 ]
          [ 1 −1 −1  1 ]

  y = R·x = ½ · [3, 3, 3, 3] = [1.5, 1.5, 1.5, 1.5]      energy spread evenly across all four coordinates
```

In high dimension (128–1,536: per-head KV vectors, embeddings) a random rotation makes every coordinate look approximately Gaussian with the *same* variance — no coordinate is ever the outlier, by construction, whatever the input looked like (§7.5.3 is the proof). And because `R` is orthogonal it **preserves dot products exactly**: `(Rx)·(Ry) = x·y`, so attention scores and nearest-neighbour rankings computed on rotated vectors equal the originals.

**Quantizing the rotated coordinates.** Once every coordinate is ~N(0, 1), you do not need a data-dependent codebook. Precompute *once* the **Lloyd-Max codebook** — the MSE-optimal levels for a known N(0,1) — and reuse it for every vector, every dataset, forever. No calibration set, no per-vector or per-block scale:

```
  2-bit Gaussian Lloyd-Max levels:  { −1.51, −0.45, +0.45, +1.51 }
  y = [1.5, 1.5, 1.5, 1.5]  →  each snaps to +1.51  →  q = [1.51, 1.51, 1.51, 1.51], stored as four 2-bit indices

  dequantize:  x̂ = Rᵀ·q  (orthogonal ⇒ R⁻¹ = Rᵀ, nothing extra to store)  = ½ · [6.04, 0, 0, 0] = [3.02, 0, 0, 0]
```

The original is recovered to 1%, with a codebook that never saw this data.

**Why "zero memory overhead" is the headline.** Compare with NVFP4 (§7.3.2): 4 bits/weight **plus** a fresh 8-bit scale per 16 elements computed from that tensor = 4.5 bits, and those scales must be computed and persisted. TurboQuant: `b` bits per coordinate, one universal precomputed codebook, and at most a single stored L2 norm per vector (for vectors not already unit-normalized). Block quantizers typically pay 1–2 extra bits per number for their local constants; TurboQuant removes that almost entirely.

**The two-stage version (TurboQuant_prod) — for unbiased inner products.** Plain MSE quantization has a subtle flaw for similarity search: the reconstruction is unbiased per coordinate, but the *inner product* of two quantized vectors is systematically biased (the error variance scales with the true dot product). Fix: (1) quantize with `b − 1` bits (the MSE stage) → `x̂`; (2) take the residual `r = x − x̂`; (3) apply **QJL** (Quantized Johnson–Lindenstrauss) to `r` — project through a random matrix and keep only the *sign* of each entry, a genuine 1-bit sketch that gives an unbiased estimate of the residual's contribution; (4) add the correction back. Still `b` bits total, but the inner-product estimate is now unbiased — critical when the quantized vectors feed attention scores or a ranking rather than "does this roughly reconstruct".

**Where it fits.** For long-running agent workflows the natural use is exactly KV-cache compression for long contexts, with no per-request calibration. Honest status: new, mathematically well-grounded, community ports early — not yet through the years of production hardening AWQ/GPTQ/FP8 have in vLLM. Treat as promising, not battle-tested.

#### 7.5.2 Doubt — "how is `R` computed, is it for weights or activations, and does every vector get its own?"

**How `R` is computed — two variants:**

```
  1. TRUE RANDOM ROTATION (the theoretical method, exact)
       sample G ∈ ℝ^{d×d} with i.i.d. N(0,1) entries          pure Gaussian noise
       QR-decompose:  G = Q · R_upper                          Q is orthogonal by construction
       Q is uniformly random over all rotations ("Haar-distributed")

  2. HADAMARD ROTATION (the practical approximation — QuaRot, SpinQuant, most TurboQuant ports)
       H₁ = [1],   H_{2n} = 1/√2 · [ H_n   H_n ]               recursive doubling, ±1 entries only
                                   [ H_n  −H_n ]
       usually multiplied by a random ±1 diagonal first:  R = H · diag(random signs)
```

Randomness matters theoretically: a *fixed* matrix could in principle be defeated by a pathological input aligned with it; no input can be constructed in advance to defeat a randomly drawn one, which is what gives the provable worst-case guarantee. Practically, a dense `d×d` multiply per KV vector per token is too slow; the Hadamard transform is `O(d log d)` via a butterfly (like the FFT) and needs no stored matrix. For a 128-dim head: 16,384 multiply-adds dense vs ~900 butterfly. The random sign flip in front restores enough randomness to kill the edge case at `O(d)` cost. Most things called "TurboQuant" in GitHub/PyPI ports are running variant 2.

**Weights or activations?** **Activations, originally** — embeddings and KV entries, computed at runtime. That is structurally different from AWQ/GPTQ/NVFP4, which quantize fixed weights once, offline. For rotation-based *weight* quantization the technique is **QuaRot / SpinQuant**: rotate `W` and `x` together, `W' = W·Hᵀ`, `x' = H·x`, so `W'x' = Wx` is mathematically identical while both sides individually lose their outlier structure — the rotation is folded into the adjacent weight matrices at load time and costs nothing at inference.

**Shared or per-vector? Shared — one fixed `R` per space, reused for every vector.** It has to be, for the math to work: `(Rx)·(Ry) = x·y` only holds if the *same* `R` is applied to both `x` and `y`. Give every KV vector its own private rotation and the dot product of two rotated vectors no longer equals the original — the whole "preserve attention scores / similarity search" property breaks. In practice `R` (or the Hadamard-plus-sign-pattern) is generated **once from a seed**, persisted, computed at model-load time, and reused for every token of every request for the life of the deployment. It is a property of the *space* — "the KV cache of layer 12, head 4" — not of any vector passing through it.

#### 7.5.3 Doubt — "how can one matrix, chosen for all weights, turn *every* activation vector into a normal distribution?"

The part that seems like it should not work until you see why: **`R` is not fit to the data at all.** It is drawn once, blind, before seeing any activations — and it works for every vector precisely *because* it was not tailored to any of them.

**The property that makes one `R` work for everything: incoherence.** A matrix is incoherent if every row treats every input coordinate with roughly equal weight — no row has one dominant entry. The Hadamard matrix is the cleanest case: every entry is exactly `±1/√d`. Each output coordinate `(Rx)_i = Σ_j R_ij·x_j` is a weighted sum touching *all* `d` input coordinates with equal magnitude, only the signs varying. It does not matter whether `x` was spread out or concentrated on one axis: every output coordinate is forced to be a mixture of all of `x`'s energy, because the matrix gives no coordinate a free pass to dominate.

**Same `R`, two very different concentrated inputs:**

```
  x_a = [3, 0, 0, 0]   →   H·x_a = [ 1.5,  1.5,  1.5,  1.5 ]
  x_b = [0, 0, 0, 3]   →   H·x_b = [ 1.5, −1.5, −1.5,  1.5 ]
```

Two different "which coordinate is the outlier" inputs; both come out with magnitude 1.5 everywhere, differing only in sign pattern. **Incoherence is a property of the matrix, not a fit to a vector** — so you do not "identify" a matrix per input; you need exactly one matrix with no dominant rows, and it neutralizes concentration in *any* vector.

**Why the output is specifically Gaussian, not just "spread out."** This is concentration of measure — a theorem, not a heuristic. For a fixed unit vector `x` and a Haar-random orthogonal `R`, each coordinate of `R·x` (after the affine rescale from [−1, 1] onto [0, 1]) follows **Beta(d/2, d/2)** exactly, by the rotational symmetry of the Haar measure. And Beta(d/2, d/2) converges to a Gaussian shape as `d` grows. Rotating a fixed-length vector uniformly at random and looking at one coordinate *is* a well-studied distribution that happens to be bell-shaped in high dimension — true for any starting `x` on the sphere, not just typical ones.

**What the Beta distribution looks like.** It lives on **[0, 1]**, with two shape parameters:

```
  f(x; α, β)  =  x^(α−1) · (1−x)^(β−1) / B(α, β)          B(α, β) = Γ(α)Γ(β)/Γ(α+β), a normalizer
  mean = α/(α+β)        variance = αβ / [ (α+β)² (α+β+1) ]

  α = β = 1      flat: f(x) = 1, uniform
  α = 2, β = 5   skewed toward 0, mean 2/7 ≈ 0.29, long tail toward 1
  α = β = 2      gentle dome, f(x) = 6x(1−x), peak 1.5 at x = 0.5, zero at the edges
  α = β = 0.5    U-shaped: mass piles up at both edges — what very LOW dimension gives you
```

For `α = β = d/2` the distribution is always symmetric around 0.5, and its spread shrinks with `d`: `variance = 1/(4(d+1)) ≈ 1/(4d)`:

```
  d = 2      Beta(1, 1)      uniform — a rotated coordinate could land anywhere
  d = 8      Beta(4, 4)      moderate hump, std ≈ 0.17
  d = 128    Beta(64, 64)    tight symmetric bell hugging 0.5, std ≈ 0.044     ← a real head dimension
  d = 768    Beta(384, 384)  std ≈ 0.018 — visually indistinguishable from a Gaussian
```

Flat → dome → narrow bell. At low dimension a rotated coordinate could plausibly be anything; at high dimension it is *forced* close to the centre with Gaussian-shaped fluctuations. **That is the last link in the chain:** a fixed Lloyd-Max codebook is derived by minimizing error *for a known distribution*; since every rotated coordinate, from every vector, actually lands in that same bell, one codebook precomputed for N(0, 1) is near-optimal for all of them simultaneously. Why rotate → why Beta → why one universal codebook suffices.

**So, concretely, "how is `R` identified":** pick `d` (e.g. per-head KV dim 128), pick a seed, draw `G ~ N(0,1)^{d×d}` and QR it (or build the recursive Hadamard plus a random ±1 sign vector), store the seed. Done — before any data is seen. No optimization loop, no calibration batch. The matrix's job is not to match your activation distribution; it is to guarantee that *whatever* distribution your activations have, none can concentrate its energy on a few output coordinates after passing through it.

#### 7.5.4 Doubt — "what is a Hadamard matrix, and why does recursive doubling give an orthogonal rotation?"

A Hadamard matrix `H_n` is an `n×n` matrix whose entries are all `+1` or `−1` and whose rows are mutually **orthogonal** (every pair has dot product 0). The second property is the point: it is what makes it a rotation once normalized.

**The recursive (Sylvester) construction** — tile four copies of the smaller matrix and flip the sign of the bottom-right block:

```
  H₁ = [1]          H_{2n} = [ H_n   H_n ]
                             [ H_n  −H_n ]

  H₂ = [ 1   1 ]        H₄ = [ 1   1   1   1 ]       ← the matrix used in every example above;
       [ 1  −1 ]             [ 1  −1   1  −1 ]          now you can see where it came from
                             [ 1   1  −1  −1 ]
                             [ 1  −1  −1   1 ]
```

**Check orthogonality on H₄:** row 1 · row 2 = `1 − 1 + 1 − 1 = 0`; row 1 · row 3 = `1 + 1 − 1 − 1 = 0`. Every off-diagonal pair cancels — and not by coincidence of this size.

**Why doubling always preserves it — the proof by induction.** Suppose `H_n·H_nᵀ = n·I` (orthogonal rows, each of squared length `n`). Then by block multiplication:

```
  H_{2n} · H_{2n}ᵀ  =  [ H_n   H_n ] · [ H_nᵀ   H_nᵀ ]
                       [ H_n  −H_n ]   [ H_nᵀ  −H_nᵀ ]

                    =  [ H_nH_nᵀ + H_nH_nᵀ      H_nH_nᵀ − H_nH_nᵀ ]     =  [ 2n·I     0   ]  =  2n · I_{2n}
                       [ H_nH_nᵀ − H_nH_nᵀ      H_nH_nᵀ + H_nH_nᵀ ]        [   0    2n·I  ]
```

The off-diagonal blocks cancel *exactly because of the one sign flip* in the construction; the diagonal blocks double cleanly. `H₁·H₁ᵀ = [1] = 1·I` is the base case, so induction gives `H_n·H_nᵀ = n·I` for every power-of-two `n`, for free.

**From Hadamard matrix to rotation.** `H_n·H_nᵀ = n·I` says the rows are orthogonal but of squared norm `n` (n entries of ±1, each squared to 1). Divide every entry by `√n`:

```
  Q = H_n / √n     ⇒     Q·Qᵀ = (H_n·H_nᵀ)/n = I          exactly the definition of an orthogonal matrix
```

That is why every example used `½·H₄` (since `√4 = 2`) rather than the raw ±1 matrix.

**Why this is the practical choice over a true random rotation.** Two things fall out of the recursive structure, both of which matter on a KV-cache hot path: (1) **fast** — the block structure decomposes the matrix-vector product into a butterfly (the FFT trick), `O(n log n)` instead of `O(n²)`: ~900 vs 16,384 multiply-adds for a 128-dim head; (2) **no storage** — the matrix is a fixed recursive sign pattern, so you never hold a `d×d` matrix in memory; you regenerate the signs on the fly from `d`. The trade, as flagged in §7.5.2: a fixed structured matrix is deterministic, so a pathological vector aligned with it could in principle exist — which is why production code applies a random ±1 diagonal first. (Sylvester Hadamard matrices exist only for powers of two; non-power-of-two dimensions use a Kronecker product with a small Hadamard of another order, or pad.)

---

# 8. Attention Kernels

**The problem:** the score matrix is `O(h·S²)`. At `S=32k, h=32` that's ~68 GB in FP16 — it cannot be materialized.

**FlashAttention** (Dao et al., 2022) never materializes it. It **tiles** the computation, keeps blocks in SRAM, and uses **online softmax** (streaming running max and sum) to compute the exact result without ever holding the full matrix.

```
Naive:      compute S×S in HBM     → O(S²) memory, bandwidth-bound
FlashAttn:  tile into SRAM blocks  → O(S)  memory, compute-bound
```

**Exact, not approximate.** Same numerical result, 2–4× faster, linear memory.

| Version | Adds |
|---|---|
| FlashAttention | Tiling + online softmax |
| **FlashAttention-2** | Better work partitioning; ~2× over v1 |
| **FlashAttention-3** | Hopper-specific: async copies (TMA), FP8 support |
| **FlashInfer** | Kernels specialized for *serving* — paged KV, variable-length batches |
| **PagedAttention kernel** | Gathers through the block table (§6) |

---

# 9. Speculative Decoding

## 9.1 The exploit

**Verifying `k` tokens costs almost the same as generating 1**, because both read all the weights once — decode is bandwidth-bound, so the extra FLOPs are nearly free. Speculative decoding converts that spare compute into latency.

```
1.  DRAFT   small model autoregressively generates k tokens   (cheap, sequential)
2.  VERIFY  target model runs ONE forward pass over all k     (one weight read)
            → causal attention yields k+1 distributions at once
3.  ACCEPT  longest agreeing prefix; first disagreement is replaced
            by the target's own token
4.  repeat
```

## 9.2 Why it's lossless

**Greedy:** accept while `argmax(p_target) == draft_token`. Trivially identical output.

**Sampling:** the rejection-sampling rule —
```
accept draft token x with probability  min(1, p_target(x) / p_draft(x))
on rejection, sample from the normalized residual:
        p'(x) ∝ max(0, p_target(x) − p_draft(x))
```
> This makes the output distribution **exactly equal** to the target model's. Speculative decoding is not an approximation — it is a pure latency optimization with mathematically identical output.

**Bonus token:** if all `k` drafts are accepted, the `(k+1)`-th distribution from the same forward pass yields one extra token for free.

**Why post-rejection drafts are discarded:** verification at position `i` is conditioned on accepted tokens `1…i−1`. Once a token is rejected and replaced, subsequent drafts were conditioned on a history that no longer exists — they're meaningless.

## 9.3 Variants

| Method | Mechanism | Typical speedup |
|---|---|---|
| Draft model | Separate ~1/10-size model | 2–3× |
| **Medusa** | Extra decoding heads on the target | ~2× |
| ⭐ **EAGLE / EAGLE-2 / EAGLE-3** | Feature-level autoregression; best acceptance rates | **3–4×** |
| **N-gram / prompt lookup** | Match against the prompt; no draft model at all | 1.5–2×; **excellent for RAG, editing, code** |
| **MTP** (DeepSeek) | Training-time multi-token heads reused as drafts | built in |

**Drivers:** acceptance rate (60–90%; higher on code and structured output), `k` (3–8 sweet spot), draft-model quality and size.

> **The trap: speculative decoding stops helping at high batch size, and can hurt.** Its premise is spare compute during bandwidth-bound decode. At large batch you're already near the compute roofline, so rejected tokens are wasted FLOPs and throughput *drops*. Enable it for latency-critical low-batch traffic; disable it in the batch pool. Modern engines toggle it dynamically. **Knowing when to turn an optimization off is the senior signal.**

---

# 10. Chunked Prefill and Disaggregation

## 10.1 The interference problem

Continuous batching runs **one iteration loop**. A 32k-token prefill is a single indivisible iteration lasting ~2–3 seconds for a 70B model on TP=4 H100s (~1 s for an 8B model on one H100) — during which **every in-flight decode gets zero steps**.

```
Timeline (naive):
decode  decode  decode  ┃━━━ 32k PREFILL, 3s ━━━┃  decode  decode
 25ms    25ms    25ms   ┃  all decodes frozen   ┃   25ms    25ms
                        └── 3,000 ms ITL spike ─┘

TTFT looks fine.  p99 TPOT explodes.
```

This is the exact mechanism behind *"TTFT is fine but TPOT is terrible under load"* — and an RPS dashboard would never show it.

## 10.2 Fix 1 — chunked prefill (the right default)

Split prefills into ~2k-token chunks and **fuse each chunk into a decode iteration**.

```
[decode batch + prefill chunk 1] [decode batch + prefill chunk 2] …
        ~40 ms                            ~40 ms
```

- ITL goes from 25 ms with 3,000 ms spikes → ~40 ms steady
- Small TTFT tax (prefill is now spread over several iterations)
- **Bonus:** fusing compute-bound prefill with bandwidth-bound decode uses *both* resources at once
- On by default in modern vLLM; chunk size is the tuning knob

## 10.3 Fix 2 — disaggregation (at scale)

Run prefill and decode on **separate GPU pools**, shipping the KV cache between them over NVLink or RDMA (DistServe, Mooncake, NVIDIA Dynamo).

```
┌──────────────┐   KV over NVLink/RDMA   ┌──────────────┐
│ PREFILL POOL │ ──────────────────────▶ │ DECODE POOL  │
│ compute-opt  │                         │ bandwidth-opt│
│ high TP      │                         │ high batch   │
└──────────────┘                         └──────────────┘
```

| Benefit | Detail |
|---|---|
| Independent tuning | Each pool configured for its own bottleneck |
| Independent scaling | Long prompts and long generations scale separately |
| No interference | TTFT and TPOT stop trading against each other |

**Cost:** KV transfer bandwidth and real operational complexity. **Worth it at scale and for large models; over-engineering for a single 8B model.** Knowing when *not* to disaggregate is half the answer.

## 10.4 The confounder — preemption

Identical symptoms arise when KV pressure causes **preemption**: sequences are evicted and their prefixes recomputed. **Check the engine's preemption counter first.** The real fix there is admission control — cap concurrency by KV budget and queue the rest, rather than admitting everything and thrashing.

---

# 11. Parallelism

## 11.1 The four strategies

The problem: a 70B FP16 model is 140 GB; an H100 has 80 GB.

| | What is split | Communication | Fits bigger model? | Faster per token? |
|---|---|---|---|---|
| **DP** | nothing (full replicas) | **none** | ✗ | ✗ (throughput only) |
| **TP** | every weight matrix | **all-reduce ~2× per layer** | ✓ | ✓ (better TPOT) |
| **PP** | layer ranges | one handoff per stage | ✓ | ✗ (bubbles) |
| **EP** | experts (MoE only) | **all-to-all 2× per MoE layer** | ✓ | ✓ |

## 11.2 Tensor parallelism — how a matrix is actually split

For `y = x·W` with `W` of shape `[1024 × 1024]`, TP=2:

**Column split (output dimension):**
```
W = [ W_a | W_b ]      each GPU holds [1024 × 512]
Both GPUs need the FULL x.
Each produces a DIFFERENT HALF of y  →  result is a concatenation.
NO COMMUNICATION needed.
```

**Row split (input dimension):**
```
      ┌ W_a ┐            each GPU holds [512 × 1024]
W  =  │─────│
      └ W_b ┘
Each GPU needs HALF of x.
Each produces a FULL-SIZE PARTIAL SUM  →  must be summed elementwise.
REQUIRES ALL-REDUCE.
```

**The Megatron trick** — chain them so the intermediate never moves:
```
FFN:        W_up  column-split  →  W_down  row-split   →  ONE all-reduce
Attention:  W_QKV column-split by heads → W_O row-split →  ONE all-reduce
                                                    ⇒ ~2 all-reduces per layer
```
Splitting QKV by head requires `h` (and `h_kv`) divisible by TP — which is why `h_kv = 8` conveniently supports TP up to 8.

## 11.3 The NVLink domain

**An NVLink domain is the set of GPUs connected at full NVLink bandwidth without crossing a slower network.** Defined by the fabric, not the chassis.

```
Evolution:
  V100 era   →  point-to-point NVLink between GPU pairs
  DGX/HGX    →  NVSwitch, 8 GPUs fully connected      ← "TP ≤ 8" comes from here
  GB200 NVL72 →  72 GPUs in ONE domain, ~13.5 TB HBM  ← built for MoE all-to-all
```

| Link | Bandwidth | Ratio |
|---|---|---|
| NVLink (H100) | 900 GB/s | 1× |
| NVLink (Blackwell) | 1.8 TB/s | 2× |
| InfiniBand NDR | ~50 GB/s | **~18–36× slower** |

## 11.4 Placement — the rule

```
CHATTY parallelism (TP, EP)  →  MUST stay inside the NVLink domain
LIGHT  parallelism (PP, DP)  →  fine across nodes
```

**Compose hierarchically:**
```
TP=8  inside each node   (NVLink)
  ×  PP=4 across nodes   (InfiniBand)
  ×  DP=N replicas       (anywhere)
```

## 11.5 Why all-to-all is the MoE bottleneck

All-to-all is heavier than all-reduce: **every GPU sends a *different* payload to every other GPU**, and sizes shift per batch with routing.

It happens **twice per MoE layer**:
1. **Dispatch** — send each token's hidden vector to its selected experts' GPUs
2. **Combine** — send the results back, because the next attention layer needs the token in its original position

At DeepSeek-V3 scale (~58 MoE layers) that's **~116 synchronous exchanges per decode step**, each moving hundreds of MB in each direction, inside a 20–30 ms step budget.

| Fabric | Outcome |
|---|---|
| **NVLink (NVL72)** | Each exchange ≪ 1 ms, hidden behind compute |
| **InfiniBand** | ~36× narrower, × 116 synchronization barriers → **GPUs idle waiting** |

Two aggravators: traffic is **bursty** (all links saturate simultaneously) and **imbalanced** (hot experts → the slowest link gates everyone).

**Escape hatches without NVL72:** node-limited routing (restrict each token to experts within its node), hierarchical all-to-all, and computation/communication overlap (DeepEP). Workable, but constraining.

---

# 12. Building a Cluster

## 12.1 Sizing a replica

```
Step 1 — will it fit?
  70B FP16 = 140 GB  →  needs 4×H100 (320 GB) with TP=4
  leftover ≈ 320 − 140 − overhead ≈ 170 GB  →  paged KV pool

Step 2 — one 8-GPU node = two TP=4 replicas  →  100 nodes = DP 200
```

## 12.2 The stack

```
        Internet
           │
   ┌───────▼────────┐
   │  API Gateway   │  auth · quotas · rate limits · SSE streaming
   └───────┬────────┘
           │
   ┌───────▼────────┐
   │     ROUTER     │  least-load · CACHE-AWARE (prefix affinity) · sticky sessions
   └───────┬────────┘
     ┌─────┴─────┬──────────┬─────────┐
     ▼           ▼          ▼         ▼
  ┌──────┐   ┌──────┐   ┌──────┐  ┌──────┐
  │Replica│  │Replica│  │Replica│ │Replica│   each: vLLM/SGLang
  │ TP=4  │  │ TP=4  │  │ TP=4  │ │ TP=4  │   continuous batching
  │4×H100 │  │4×H100 │  │4×H100 │ │4×H100 │   PagedAttention
  └──────┘   └──────┘   └──────┘  └──────┘   prefix cache
```

> **Cache-aware routing is not optional.** If requests sharing a prefix land on different replicas, each builds its own cache and none of them hit. Hash the prefix and route consistently — this alone is often the difference between a 20% and an 80% prefix-cache hit rate.

## 12.3 How the pieces actually communicate

Three tiers, carrying different things:

| Path | Medium | Carries |
|---|---|---|
| **Node ↔ node** | NIC + datacenter network | **Only requests/responses.** DP replicas are independent — no model traffic |
| **CPU ↔ GPU** | PCIe | Down: token IDs, block tables, kernel launches. Up: sampled token IDs |
| **GPU ↔ GPU** | NVLink (NCCL) | TP all-reduces |

**Lifecycle of one request:**
```
NIC → CPU tokenizes → scheduler admits into the continuous batch
    → GPU PREFILL: for each layer, 4 GPUs matmul their slices → all-reduce → next layer
                   KV written into paged blocks
    → sample → CPU detokenizes → stream first token   ⏱ TTFT stops here
    → DECODE LOOP: one new token per request per step, whole batch together (~20–30 ms/step)
    → on completion: free KV blocks (or retain for prefix cache)
```
Decode steps are mostly HBM reads. **The CPU's entire job is keeping the GPUs fed** — tokenization, scheduling, and block bookkeeping must never become the bottleneck.

## 12.4 Deploying a 2T-parameter model

```
Memory:  FP16 = 4 TB · FP8 = 2 TB  (+ hundreds of GB KV)
      →  one replica spans 32–72+ GPUs
```
Realistically it's **MoE** (~2T total, ~50B active per token — ~40× sparse, more aggressive than DeepSeek-V3's ~18×) → FP8 weights alone need ~15 H200s (2 TB / 141 GB ≈ 14.2) → memory-capacity-bound, not compute-bound. EP is primary, TP for attention and shared layers.

| Hardware | Shape |
|---|---|
| **H100/H200** | Replica = 8 nodes × 8 GPUs. TP=8 in-node, PP=8 across. EP is awkward across the slow boundary → node-local expert groups |
| ⭐ **GB200 NVL72** | 72 × ~186 GB ≈ **13.5 TB in ONE NVLink domain** → the whole model in one rack, EP all-to-all stays on NVLink. **One rack = one replica.** |

**Operational realities:**
- **Prefill/decode disaggregation is mandatory** at this size — pools tuned and scaled independently
- **Hot experts replicated dynamically**, driven by live load statistics
- **Weights (2 TB) pulled from blob storage** → boot takes many minutes → keep **warm spares**
- **One GPU failure kills the replica** → drain, restart, fail over to a spare
- A rack is **$3M+**, so quantization, prefix caching, and goodput optimization are worth millions in utilization

---

# 13. Alternatives to Autoregressive Decoding

**Why not decode every position at once?** Because sampling positions independently captures only the per-position **marginals**, not the **joint**. *"The capital is Paris"* and *"The capital is Rome"* are both high-probability; independent sampling can emit *"The capital is Parme."* This is the multimodality problem, and it's exactly why non-autoregressive translation research (2018+) stalled. The chain rule is what lets AR represent the joint exactly.

**Diffusion language models** fix it with *parallel but iterative* refinement: start fully masked, denoise all positions over `T` steps (`T ≪ length`), each step conditioned on the full current draft.

**Status (early 2026):** Mercury (Inception Labs) shipping 1000+ tok/s commercially; Gemini Diffusion demonstrated; LLaDA-8B and Dream-7B competitive with same-size AR models.

**Why AR still dominates production:** frontier quality still favours AR; each refinement pass is full-width with **bidirectional attention, which breaks cheap KV caching** (the immutability property in `01` §10.1 no longer holds); serving stacks are built for AR; and length handling and streaming are awkward.

**Hybrids:** block diffusion / semi-autoregressive. And note that **speculative decoding and Jacobi/lookahead decoding are conservative relatives of the same idea** — parallel guessing with a correctness guarantee.

---

# 14. Deployment Operations

## 14.1 Weight loading and cold start

```
S3 (~1–10 GB/s) → local NVMe cache → page into HBM
70 GB model:  ~2–8 min from S3,  ~30–60 s from local NVMe
2 TB model:   tens of minutes
```

- **Cache weights on local NVMe** — usually most of the cold-start time
- **Pre-warm a standby pool**; treat idle capacity as the cost of meeting the SLO
- **Warm up after load** — run dummy requests to trigger CUDA graph capture, kernel autotuning, and memory-pool allocation before serving real traffic

## 14.2 Kubernetes specifics

| Concern | Handling |
|---|---|
| **Readiness probe** | Must **not** pass until weights are loaded *and* warmup is done — otherwise traffic hits a cold pod |
| **Liveness probe** | Separate; restarting a slow-loading pod creates a crash loop |
| **`terminationGracePeriodSeconds`** | Long enough to drain in-flight generations (they can run minutes) |
| **Rolling updates** | `maxSurge` needs spare GPUs; without them the rollout deadlocks |
| **Multi-GPU replicas** | All GPUs of a TP group must be on one node — use topology-aware scheduling |
| **Autoscaling** | On **queue depth and KV utilization**, never CPU. Long cooldowns; GPUs boot in minutes |

## 14.3 What to monitor

| Layer | Metrics |
|---|---|
| **SLO** | TTFT p50/p95/p99 · TPOT p50/p95/p99 · **goodput** |
| **Scheduler** | Batch occupancy · queue depth · **preemption count** · admission rejections |
| **Memory** | KV utilization % · **prefix cache hit rate** · fragmentation |
| **Hardware** | GPU util (weak signal) · **HBM bandwidth utilization** (the real one) · temperature/throttling |
| **Model** | Tokens in/out · finish reasons · truncation rate |

> **GPU utilization is a weak metric** — it reports that a kernel was resident, not that it was doing useful work. A GPU decoding at batch 2 shows high utilization while wasting most of its bandwidth. **Track batch occupancy and HBM bandwidth utilization instead.**

## 14.4 Failure modes

| Failure | Mitigation |
|---|---|
| KV OOM under load | Admission control on *projected* KV; preempt with recompute-or-swap |
| Long-tail latency | Cap preemption; prioritize in-flight over new admissions |
| Head-of-line blocking | Chunked prefill; separate long-prompt queue |
| Cold start timeouts | Pre-warmed pool; NVMe weight cache; never scale reactively |
| NCCL hang | Timeouts + health checks; fail the replica rather than hanging the fleet |
| Expert imbalance (MoE) | Monitor per-expert load; replicate hot experts |
| Prefix cache thrash | Prefix-affinity routing; tier to CPU/NVMe |
| Silent quality regression | Task-level eval gate before rollout |

---

# 15. Cost Model

```
cost per 1M output tokens  =  GPU $/hr ÷ (tokens/s × 3600) × 1e6
```

**Worked — Llama-3.1-8B FP8 on one H100 @ $2/hr:**
```
decode throughput at good batch  ≈ 2,500 tok/s aggregate
tokens/hour                       = 9.0 M
cost per 1M output tokens         = $2 / 9.0  ≈  $0.22
```

**Levers, ranked:**

| Lever | Impact | Caveat |
|---|---|---|
| ⭐ Continuous batching | 2–4× | Table stakes — verify it's on |
| ⭐ Prefix caching | up to 90% on agent traffic | Needs prefix-affinity routing **and** stable prompt prefixes |
| ⭐ FP8 quantization | ~2× | Gate on task evals |
| KV quantization | 2× concurrency | Watch long-context recall |
| Right-sized pools | large | Interactive vs batch want opposite configs |
| Speculative decoding | 2–4× latency at low batch | **Hurts at high batch** |
| Chunked prefill | TPOT stability | Small TTFT tax |
| Spot GPUs for batch | 60–80% | Needs checkpoint/requeue |

---

# 16. Engine Comparison

| Engine | Strengths | Choose when |
|---|---|---|
| ⭐ **vLLM** | PagedAttention origin, broadest model support, strong community | **Default** |
| **SGLang** | RadixAttention, structured output, fast frontier-model support | Prefix reuse dominates (agents, multi-turn) |
| **TensorRT-LLM** | Peak NVIDIA throughput, FP8/NVFP4 kernels | Max perf, willing to accept a compile step |
| **TGI** | HuggingFace-native, production-hardened | Already in the HF ecosystem |
| **llama.cpp** | CPU/Metal/consumer GPU, GGUF | Local, edge, single-user |
| **NVIDIA Dynamo** | Disaggregated serving orchestration | Running prefill/decode split at scale |

---

# 17. References

**Foundational serving papers**
- ⭐ **Efficient Memory Management for LLM Serving with PagedAttention (vLLM)** — Kwon et al., SOSP 2023 (arXiv 2309.06180)
- ⭐ **Orca: A Distributed Serving System for Transformer-Based Generative Models** — Yu et al., OSDI 2022 — continuous batching
- **SGLang / RadixAttention** — Zheng et al., 2023 (2312.07104)
- **DistServe** — Zhong et al., 2024 (2401.09670) — disaggregation, and the **goodput** metric
- **Sarathi-Serve** — Agrawal et al., 2024 (2403.02310) — chunked prefill
- **Mooncake** — Qin et al., 2024 (2407.00079) — KV-centric disaggregated architecture
- **Splitwise** — Patel et al., 2023 (2311.18677)

**Kernels and quantization**
- **FlashAttention** (2205.14135) · **-2** (2307.08691) · **-3** (2407.08608)
- **GPTQ** (2210.17323) · **AWQ** (2306.00978) · **SmoothQuant** (2211.10438) · **LLM.int8()** (2208.07339)
- **FP8 Formats for Deep Learning** — Micikevicius et al. (2209.05433)
- **QLoRA (NF4, double quantization)** — Dettmers et al. (2305.14314) — §7.3.2
- **OCP Microscaling (MXFP4)** — OCP MX spec v1.0 (2023) · **NVFP4** — NVIDIA Blackwell docs — §7.3.2
- **QuaRot** (2404.00456) · **SpinQuant** (2405.16406) — rotation-based weight+activation quantization, §7.4, §7.5.2
- **Marlin** — mixed-precision INT4×FP16 GEMM kernel (2408.11743) — the fused dequant of §7.3.5
- **KIVI** — 2-bit KV cache quantization (2402.02750)
- **TurboQuant** — Google Research, ICLR 2026 — rotation + Lloyd-Max codebook + QJL for KV/vector quantization, §7.5.1–§7.5.4
- **QJL: 1-bit quantized Johnson–Lindenstrauss** (2406.03482) — the residual sketch in TurboQuant_prod

**Speculative decoding**
- **Fast Inference from Transformers via Speculative Decoding** — Leviathan et al. (2211.17192)
- **Accelerating LLM Decoding with Speculative Sampling** — Chen et al. (2302.01318)
- **Medusa** (2401.10774) · ⭐ **EAGLE** (2401.15077) · **EAGLE-2** (2406.16858)

**Parallelism and scale**
- **Megatron-LM** — Shoeybi et al. (1909.08053) — the column/row split trick
- **GSPMD** (2105.04663) · **ZeRO** (1910.02054)
- ⭐ **DeepSeek-V3 Technical Report** (2412.19437) — MLA + MoE + FP8 at frontier scale
- **DeepEP** — DeepSeek's expert-parallel communication library

**Diffusion LMs**
- **LLaDA: Large Language Diffusion Models** (2502.09992) · **Dream-7B**
- **Non-Autoregressive Neural Machine Translation** — Gu et al. (1711.02281) — where the multimodality problem was identified

**Code**
- ⭐ `vllm-project/vllm` — **read the scheduler and block manager**; that's continuous batching and PagedAttention in code
- `sgl-project/sglang` — RadixAttention
- `NVIDIA/TensorRT-LLM` · `ai-dynamo/dynamo` · `LMCache/LMCache` · `flashinfer-ai/flashinfer`
- ⭐ `stas00/ml-engineering` — the honest practitioner's guide to parallelism, interconnect, and debugging at scale

---

# 18. Q&A Index — the Inference Deep-Dive

Every question from the external inference Q&A transcript (not included in this repo), mapped to where it's answered in depth. Use this as a revision checklist: cover the right-hand column and try to answer from memory.

| # | Question | Answered in |
|---|---|---|
| **Q1** | How is model size (8B params) calculated? | `01` **§12** — formula + full Llama-3-8B worked example |
| **Q2** | How many bytes in one GB? | **§1.1** — decimal vs binary, the 7.4% gap |
| **Q3** | Billion/million bytes per unit? | **§1.1** — the ladder |
| **Q4** | 1 Trillion = how many Billion? | **§1.1** — `1 T = 1,000 B` |
| **Q5** | 1 TB = how many GB? | **§1.1** — 1,000 GB decimal / 1,024 GiB binary |
| **Q6** | H200 memory size? | **§2.1** — 141 GB HBM3e @ 4.8 TB/s, vs H100/B200 |
| **Q7** | Summary mapping table | **§1.1 + §1.2 + §2.1** |
| **Q8** | Why TTFT, TPOT and goodput — not RPS? | **§4** — definitions, why throughput misleads, benchmark checklist |
| **Q9** | PagedAttention | **§6** — the problem, the block table, the results |
| **Q10** | Visual example, context length 100 | **§6.1–6.2** — 57% waste → 11% waste, with the block layout |
| **Q11** | Why did old systems pre-allocate to max length? | **§6.1** — the four reasons contiguity forced it |
| **Q12** | RadixAttention | **§7.1** — radix tree, LRU, and the byte-stable-prefix rule |
| **Q13** | The parallelism placement table | **§11.3–11.4** — interconnect tiers and the placement rule |
| **Q14** | What are TP, PP, EP, DP? | **§11.1** — from first principles |
| **Q15** | Does TP make a 1024 matrix into 2×512? | **§11.2** — column vs row split, and the Megatron chaining trick |
| **Q16** | What is an NVLink domain? | **§11.3** — fabric not chassis; the evolution to NVL72 |
| **Q17** | How is a standard inference cluster built? | **§12.1–12.2** — replica sizing → engine → gateway/router/K8s |
| **Q18** | How are components connected; how is compute done? | **§12.3** — the three tiers and the full request lifecycle |
| **Q19** | How is a 2T model deployed in reality? | **§12.4** — memory math, H100 vs NVL72, operational realities |
| **Q20** | Why does all-to-all demand NVL72? | **§11.5** — dispatch/combine, ~116 exchanges/step, bursty + imbalanced |
| **Q21** | Speculative decoding | **§9** — the exploit, losslessness proof, variants, the high-batch trap |
| **Q22** | Confirming the speculative mechanism | **§9.2** — rejection sampling, bonus token, why post-rejection drafts die |
| **Q23** | Why not decode everything at once? | **§13** — the multimodality problem; diffusion LMs; why AR still wins |
| **Q24** | Diagnose "TTFT fine, TPOT terrible" | **§10** — the interference mechanism, chunked prefill, the preemption confounder |
| **Q25** | 64 layers, 30 tokens — cache only token 30? | `01` **§10.1.1** — no; attention sums over *all* cached values |
| **Q26** | K and V are learned — why recompute? | `01` **§10.1.2** — weight *matrices* vs activation *vectors* |
| **Q27** | After q31,k31,v31 — how is attention computed? | `01` **§10.4** — the 8-step walkthrough |
| **Q28** | Do we recompute scores for tokens 1–30? | `01` **§10.4** — no; scores are never cached, only K/V |
| **Q29** | Is the FFN applied to token 31 only? | `01` **§6.2** — yes; position-wise, and why that enables MoE |
| **Q30** | MoE architecture and math | `01` **§14** — routing, Mixtral param math, load balancing |

## 18.1 The five ideas worth being able to derive on a whiteboard

If you can reconstruct these from first principles, the rest follows.

**1. Parameter count → memory → what fits.**
```
N = 2·V·d + L·[2d² + 2·d·h_kv·d_h + 3·d·d_ff]
memory = N × bytes_per_param
```
Llama-3-8B → 8.03 B → 16 GB at FP16. *(Q1–Q7)*

**2. KV cache size → concurrency.**
```
KV/token = 2 × L × h_kv × d_h × bytes
concurrency = (GPU_mem − weights − overhead) / (KV/token × context)
```
This is the number that decides how many users a GPU serves. *(Q25–Q29)*

**3. Prefill is compute-bound; decode is bandwidth-bound.**
Ridge point ≈ 295 FLOPs/byte; decode at batch 1 ≈ 1 (FP16) — ~300× below; batch B ≈ B. **Batching is what closes the gap** — and it does nothing for TTFT. *(Q8, Q24)*

**4. Chatty parallelism must stay inside the NVLink domain.**
TP all-reduces ~2× per layer; EP all-to-all 2× per MoE layer; NVLink is ~18–36× faster than InfiniBand. *(Q13–Q16, Q20)*

**5. Speculative decoding converts spare compute into latency — and stops working when there is none.**
Verifying `k` tokens costs ~one weight read. At high batch you're already compute-bound, so it hurts. *(Q21–Q23)*

## 18.2 Follow-ups these answers invite

Worth preparing, because each is the natural next question:

- *"You said decode is bandwidth-bound. At what batch size does that stop being true, and what happens then?"* → §2.3, §9.3
- *"Prefix caching gives 80% hit rate in theory. Why is yours at 20%?"* → §7.1 (routing affinity, or a timestamp at the front of the prompt)
- *"You have TP=8 and 8 KV heads. What happens at TP=16?"* → §11.2 — you can't split 8 KV heads across 16 GPUs; you replicate them, and the KV cache stops shrinking
- *"If MLA cuts KV 10×, why doesn't everyone use it?"* → `02` §3.3 — it's a training-time architecture decision, not a serving toggle
- *"Chunked prefill fixes TPOT spikes. What does it cost?"* → §10.2 — a small TTFT tax, plus a chunk-size knob to tune
- *"Your 2T model needs 32 GPUs. What happens when one dies?"* → §12.4 — the whole replica dies; you need drain + warm spares
