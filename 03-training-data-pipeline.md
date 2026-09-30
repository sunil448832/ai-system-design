---
title: "03 — Training Data: Collection, Curation, and Delivery"
subtitle: "How 36 trillion tokens are gathered, cleaned, mixed, and fed to a GPU"
purpose: "The data side of pretraining — the other half of 01-transformer-architecture.md"
companions:
  - "01-transformer-architecture.md — what the model computes; this file covers what goes into it"
  - "02-modern-transformer-architectures.md — the real models whose data recipes are described here"
  - "05-inference-serving.md — why over-training for cheaper serving sets the token budget"
---

# How to Use This File

`01-transformer-architecture.md` (referred to below as **`01`**) answers *what the model computes*. This file answers **what goes into it, and how it gets there** — the pipeline that turns a raw web crawl into the `[B, S]` integer tensor a training step consumes.

**The through-line:** five facts drive nearly every decision below.

1. **Data quality dominates data quantity.** A 1.3T-token filtered corpus beats a 15T-token unfiltered one at equal compute. Every filtering stage exists because of this. (§4)
2. **Deduplication is the highest-leverage single operation** — and doing it *too aggressively* is a known way to make a model worse. (§5)
3. **The model never sees a document.** It sees fixed-length token windows. Everything between "a webpage" and "a `[B, S]` tensor" is packing, and packing has correctness traps. (§9)
4. **Data decisions are validated by training, not by intuition.** Every serious lab runs small proxy models to compare corpora. (§8.4)
5. **"Training data" is four different things.** Pretraining, SFT, preference and RL data differ in label, shape, volume, cost and loss mask — conflating them is the most common source of confusion about how models are built. (§11)

| Section | Covers |
|---|---|
| §0 | Scale, token budgets, the compute math, the pipeline map |
| §1 | Collection — crawl, curated, licensed, synthetic, PDF/OCR |
| §2 | Extraction — HTML/PDF → text, and what must be preserved |
| §3 | Language identification and per-language routing |
| §4 | Filtering — heuristics, model-based quality, safety, PII |
| §5 | Deduplication — exact, MinHash+LSH, semantic; and when *not* to |
| §6 | Decontamination against evaluation benchmarks |
| §7 | Tokenization at scale and binary shard storage |
| §8 | Data mixing, curricula, epochs, proxy-model ablation |
| §9 | Packing, document masking, FIM, batching, the dataloader |
| §10 | Long-context data |
| §11 | **Data by training objective** — pretraining vs SFT vs preference vs RL: labels, loss masks, sample records |
| §12 | **Public datasets** — verified HuggingFace links for every regime |
| §13 | Infrastructure, cost, and failure modes |
| §14 | Case studies: Qwen3 · DeepSeek-V3 · Llama 3 · Gemma 3 |
| §15 | The assembled pipeline and its verification run |

## Module index — the code for each stage

| Stage | Module | § |
|---|---|---|
| Config | `DataConfig` | §0.4 |
| Heuristic quality filters | `gopher_filters` | §4.2 |
| Near-duplicate detection | `MinHasher`, `LSHIndex`, `dedup` | §5.4 |
| Benchmark decontamination | `build_contam_index`, `is_contaminated` | §6.2 |
| Binary token storage | `ShardWriter`, `ShardReader` | §7.3 |
| Domain mixing | `MixtureSampler` | §8.5 |
| Packing + doc masking | `pack_documents`, `document_position_ids`, `document_mask` | §9.3 |
| Fill-in-the-middle | `fim_transform` | §9.4 |
| Resumable batching | `TokenLoader` | §9.7 |
| Budget math | `training_flops`, `chinchilla_optimal`, `gpu_days` | §0.2 |
| **Pretraining labels** | `pretrain_labels` | §11.1 |
| **SFT labels + loss mask** | `sft_labels`, `multiturn_sft_labels` | §11.2 |
| **Preference / DPO / RM** | `preference_pair`, `dpo_loss`, `bradley_terry_loss`, `helpsteer_to_scalar` | §11.3 |
| **RL rewards + advantages** | `rlvr_reward`, `grpo_advantages` | §11.4 |

## Doubt index — worked resolutions

| Question | Where |
|---|---|
| Isn't more data always better? | §4.6 |
| Why not deduplicate globally across all crawl snapshots? | §5.5 |
| Why is benchmark decontamination so hard to get right? | §6.3 |
| How many times can the same token be shown to a model? | §8.2 |
| What exactly is "one training step", in tokens? | §9.8 |
| Why does DeepSeek-V3 pack documents but *not* mask across them? | §9.2 |
| Where do 36T tokens even come from — is there that much web? | §1.6 |
| What is the *label* for each kind of training data? | §11 (table) |
| Which tokens does the loss actually apply to in SFT? | §11.2 |
| If no token is "correct", what does preference training optimize? | §11.3 |
| Where can I actually download data for each regime? | §12 |

---

# 0. The Scale of the Problem

## 0.1 What SOTA models actually consume

| Model | Pretraining tokens | Languages | Notable |
|---|---|---|---|
| ⭐ **Qwen3** (0.6B → 235B-A22B) | **36 T** | **119** languages & dialects | 2× Qwen2.5's tokens, ~4× the languages (29 → 119); trillions of tokens are *synthetic* |
| **DeepSeek-V3** (671B-A37B) | **14.8 T** | English + Chinese, expanded multilingual | Heavier math/code ratio; FIM at 0.1 |
| **Llama 3** (8B / 70B / 405B) | **~15 T** (15.6 T for 405B) | 176 identified, 8 supported | Published data mix: 50/25/17/8 |
| **Gemma 3** (1B → 27B) | 2 T (1B) · 4 T (4B) · 12 T (12B) · **14 T (27B)** | 140+ | Distillation-based training |
| **Kimi K2** | 15.5 T | multilingual | |
| GPT-3 (2020, for scale) | 0.3 T | mostly English | 100× less than Qwen3 |

**Read the first column as a bandwidth problem.** 36 T tokens at ~4 bytes/token is **144 TB** of tokenized data — and the raw text it came from was 10–100× larger before filtering. Nothing in this pipeline is a single-machine job.

## 0.2 The compute budget, and why "Chinchilla-optimal" is not what anyone does

Training compute is almost exactly `6 · N · D` (N = parameters, D = tokens): 2·N for the forward pass (`01` §13.1), and 4·N for the backward, which computes gradients with respect to both inputs and weights.

```python
def training_flops(n_params, n_tokens):      return 6 * n_params * n_tokens
def chinchilla_optimal(n_params):            return 20 * n_params      # D ≈ 20·N
def gpu_days(n_params, n_tokens, peak_flops=989e12, mfu=0.40):
    return training_flops(n_params, n_tokens) / (peak_flops * mfu) / 86400
```

```python
>>> training_flops(405e9, 15.6e12)                       # Llama-3-405B
3.79e+25                                                  # FLOPs
>>> chinchilla_optimal(405e9) / 1e12                      # what Chinchilla says
8.1                                                       # trillion tokens
>>> 15.6 / 8.1                                            # what Meta actually did
1.9                                                       # ~2x "over-trained"
>>> gpu_days(405e9, 15.6e12)
1_109_075                                                 # H100-days at 40% MFU (idealized)
# Actual (Meta model card): 30.84M H100 GPU-hours ≈ 1.29M GPU-days — ~16% over the
# ideal, i.e. ~34% effective utilization once failures, restarts and evals are counted
```

**Chinchilla (`D ≈ 20·N`) minimizes loss for a fixed *training* budget. Nobody optimizes for that any more**, because training happens once and inference happens forever. Over-training a *smaller* model on *more* tokens produces a model that is cheaper to serve at the same quality — so the industry systematically trains past the Chinchilla point.

That reframes the entire problem: **the token budget is now set by how much good data you can obtain, not by how much compute you have.** Which is why this file exists.

> **The MoE twist.** For Qwen3-235B-A22B, `N` in `6·N·D` is the **active** parameter count (22B), not 235B (`01` §13.1). 36 T tokens at 22B active costs `4.75e24` FLOPs — **8× less than Llama-3-405B** despite 2.3× the tokens. MoE is what made 36T-token training budgets affordable.

## 0.3 The pipeline, end to end

```
   ┌──────────────────────────────────────────────────────────────────┐
   │  COLLECT                                                    §1   │
   │  Common Crawl snapshots · curated corpora · licensed data ·      │
   │  code repos · synthetic generation · PDF/OCR extraction          │
   └───────────────────────────┬──────────────────────────────────────┘
                               │   ~100 TB+ raw HTML per snapshot
   ┌───────────────────────────▼──────────────────────────────────────┐
   │  EXTRACT      HTML/PDF → text, keep math + code structure   §2   │
   └───────────────────────────┬──────────────────────────────────────┘
                               │   ~10% survives
   ┌───────────────────────────▼──────────────────────────────────────┐
   │  IDENTIFY LANGUAGE        route to per-language pipelines   §3   │
   └───────────────────────────┬──────────────────────────────────────┘
   ┌───────────────────────────▼──────────────────────────────────────┐
   │  FILTER       heuristics → quality model → safety → PII     §4   │
   └───────────────────────────┬──────────────────────────────────────┘
                               │   ~10-30% survives
   ┌───────────────────────────▼──────────────────────────────────────┐
   │  DEDUPLICATE  URL → document (MinHash/LSH) → line           §5   │
   └───────────────────────────┬──────────────────────────────────────┘
                               │   ~10-50% survives  ← the biggest cut
   ┌───────────────────────────▼──────────────────────────────────────┐
   │  DECONTAMINATE            remove benchmark overlap          §6   │
   └───────────────────────────┬──────────────────────────────────────┘
   ┌───────────────────────────▼──────────────────────────────────────┐
   │  TOKENIZE     text → uint32 shards, written ONCE            §7   │
   └───────────────────────────┬──────────────────────────────────────┘
   ┌───────────────────────────▼──────────────────────────────────────┐
   │  MIX          domain weights · curriculum stages            §8   │
   └───────────────────────────┬──────────────────────────────────────┘
   ┌───────────────────────────▼──────────────────────────────────────┐
   │  PACK & BATCH concat → [B, S] · doc masks · FIM · shuffle   §9   │
   └───────────────────────────┬──────────────────────────────────────┘
                               ▼
                    model.forward(ids)     ← `01` §15.1
```

**Every arrow is a throughput cliff.** The stages before tokenization are CPU-bound, embarrassingly parallel, and run on thousands of cores for weeks. The stages after are I/O-bound and must keep thousands of GPUs fed without ever stalling.

## 0.4 The reference implementation

As in `01`, every section ships the module that implements it, and §15.2 runs them all.

```python
import hashlib, re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class DataConfig:
    seq_len: int = 4096                # Qwen3 stages S1/S2; 32768 for S3
    eos_id: int = 151643               # <|endoftext|>  (Qwen3)
    vocab_size: int = 151936
    dtype: str = "uint32"              # uint16 is enough only if V < 65,536
    # --- dedup (§5) ---
    minhash_perm: int = 128
    minhash_bands: int = 16            # bands · rows = perm  ->  rows = 8
    shingle_n: int = 5
    # --- decontamination (§6) ---
    contam_ngram: int = 13
    # --- packing (§9) ---
    doc_masking: bool = True
    fim_rate: float = 0.0              # DeepSeek-V3 uses 0.1
    # --- mixture (§8) ---
    weights: dict = field(default_factory=lambda: {          # Llama 3's published mix
        "web": 0.50, "math_reasoning": 0.25, "code": 0.17, "multilingual": 0.08})
```

> **`dtype` is not a detail.** `uint16` holds 0–65,535. Every modern vocabulary (Llama 3: 128,256; Qwen3: 151,936) **overflows it**, so tokenized corpora must be `uint32` — doubling storage from 2 to 4 bytes per token. For 36 T tokens that is the difference between 72 TB and 144 TB. Choosing `uint16` because a tutorial used it (GPT-2, `V` = 50,257) is a real and silent corruption bug.

---

# 1. Collection — Where the Tokens Come From

## 1.1 Common Crawl — the substrate

A non-profit that has crawled the web since 2008 and publishes a snapshot roughly monthly. Each snapshot is ~2–4 billion pages, and ships in three formats:

| Format | Contains | Use it? |
|---|---|---|
| **WARC** | the complete raw HTTP response — headers **and** original HTML | ⭐ **Yes** — this is what serious pipelines start from |
| **WET** | Common Crawl's own plain-text extraction | **No.** Convenient, but its extractor keeps navigation, footers and cookie banners; every quality-focused corpus (RefinedWeb, FineWeb, Llama 3) re-extracts from WARC |
| **WAT** | metadata only (links, headers) | For link graphs and domain-level filtering |

**The arithmetic of a snapshot:** ~100 TB compressed WARC → ~10 TB of extracted text → ~1–3 TB after filtering → a few hundred GB after deduplication. **You need dozens of snapshots to reach a trillion tokens of web text**, which is exactly why models quoting 15T+ tokens must go far beyond the crawl.

## 1.2 Curated and licensed sources

| Source | Why it matters | Caveat |
|---|---|---|
| **Code** (GitHub, permissive licences) | The single strongest non-obvious driver of *reasoning* ability — Llama 3 gives code 17% of the mix | Licence filtering is mandatory; near-duplicate forks are rampant |
| **Books** | Long-form coherence and narrative structure that web text simply lacks | The most legally fraught category |
| **Academic** (arXiv, PubMed, PMC) | STEM density; LaTeX gives clean math | LaTeX must survive extraction (§2.3) |
| **Wikipedia** | High factual density, ~300 languages, cleanly licensed | Small (a few billion tokens) and heavily memorized — usually upsampled, but it is not a corpus by itself |
| **Q&A** (StackExchange) | Naturally instruction-shaped; strong for code | |
| **Patents, legal, government** | Formal register, long documents | Boilerplate-heavy |
| **Licensed / purchased** | News archives, publisher deals | Where a lot of undisclosed frontier-model data now comes from |

## 1.3 Synthetic data — how you get past the data wall

This is the biggest change of the last two years, and Qwen3 is the clearest published example.

> Qwen3 employed **Qwen2.5, Qwen2.5-Math, and Qwen2.5-Coder** to *"synthesize trillions of text tokens in different formats, including textbooks, question-answering, instructions, and code snippets, covering dozens of domains."*

Read that carefully: **a meaningful fraction of a 36 T-token corpus was written by earlier models in the same family.** The mechanism is a quality ratchet, not free lunch —

```
Qwen2.5-Math  ──generates──▶  math textbooks & QA  ──trains──▶  Qwen3
    (a strong but narrow specialist)                             (a stronger generalist)
```

The specialist is better *at its domain* than the general web is, so its output raises the average quality of that slice. What synthetic data cannot do is add information nobody has — which is why it is paired with, not substituted for, real data.

| Synthetic technique | What it produces |
|---|---|
| **Textbook generation** (phi-series) | Dense, pedagogical explanations with no web boilerplate |
| **Rephrasing / restyling** | The same facts in Q&A, tutorial, or formal register — multiplies usable tokens from scarce high-quality sources |
| **Solution augmentation** | Step-by-step derivations for problems that only had answers |
| **Back-translation** | Multiplies low-resource-language data |
| **Self-instruct** | Instruction–response pairs for post-training (§11) |

> **The failure mode is model collapse:** train on your own unfiltered output for enough generations and distribution tails vanish. Every production use filters synthetic data at least as hard as web data, and **verifies** it where verification is cheap (run the code, check the proof, compare the arithmetic).

## 1.4 PDF and OCR — recovering the text that HTML never had

Also from Qwen3, and easy to overlook:

> *"we first employ the **Qwen2.5-VL** model to perform text recognition on a large volume of PDF-like documents. The recognized text is then refined using the Qwen2.5 model... we are able to obtain an additional set of high-quality text tokens, amounting to **trillions** in total."*

```
PDF page ──▶ Qwen2.5-VL (vision-language OCR) ──▶ raw text ──▶ Qwen2.5 (refine) ──▶ corpus
            reads layout, tables, equations         fixes OCR errors, restores structure
```

**Why this is a big deal:** the highest-quality human writing — textbooks, papers, manuals, government reports — is disproportionately locked in PDFs whose text layer is missing, scrambled, or column-shuffled. Classical PDF extractors mangle multi-column layouts, tables and equations. A VLM reads the *rendered page* the way a person does. This turns a previously unusable archive into trillions of clean tokens, and it is one of the few genuinely new sources available after the open web has been exhausted.

## 1.5 What each source contributes

```
                 volume      quality    uniqueness
web crawl        ██████████  ██         ██████████    the substrate; needs the most work
code             ████        ███████    ███████       reasoning transfer, cheap to verify
books            ██          █████████  ████████      long-range coherence; legally hard
academic         ██          █████████  ████████      STEM density
wikipedia        █           █████████  ███           small; upsample, don't rely on
synthetic        ████████    ███████    ██            scalable; adds no NEW information
PDF/OCR          █████       ████████   ████████      newly unlocked (§1.4)
```

## 1.6 Doubt — "is there even 36 trillion tokens of text in the world?"

Of usable *unique web* text: **no, not really.** Estimates of the open web's high-quality English text land in the low trillions of tokens, and that is the "data wall" people refer to. 36 T is reached by adding four multipliers on top of the crawl:

| Multiplier | Effect |
|---|---|
| **Multilingual expansion** | Qwen3 went from 29 → **119 languages**. Most of the web is not English, and most pipelines historically discarded it |
| **Synthetic generation** (§1.3) | "trillions of tokens" — bounded by compute, not by supply |
| **PDF/OCR recovery** (§1.4) | "trillions of tokens" from documents that were never HTML |
| **Weaker deduplication + repetition** | Showing good data 2–4 times is fine (§8.2), which stretches a fixed corpus |

So "36 T tokens" is not "36 T tokens of pristine unique human web prose". It is a *training budget* assembled from several supplies with different economics — and the shift from column 1 to columns 2–4 is the defining data story of 2024–2025.

---

# 2. Extraction — HTML and PDF to Text

## 2.1 Why the free extraction isn't good enough

A raw web page is mostly not content. Navigation, sidebars, cookie notices, comment sections, related-article widgets, and footers routinely account for 70–90% of the tokens on a page — and they are **identical across every page of a site**, which means they also poison deduplication (§5) and repetition filters (§4.1).

| Extractor | Approach | Notes |
|---|---|---|
| **CC's WET** | naive tag-strip | Fast, low quality — keeps all boilerplate |
| **trafilatura** | DOM heuristics + fallbacks | Strong quality, slow; used by FineWeb |
| **resiliparse** | fast C++ main-content extraction | Used by DCLM; much faster than trafilatura |
| **jusText / readability** | classic boilerplate removal | Older baselines |
| ⭐ **Custom** | tuned per corpus | Llama 3 built its own HTML parser specifically to preserve math and code |

## 2.2 What must survive extraction

Llama 3's team reported a finding worth internalizing:

> *"markdown is harmful to the performance of a model that is primarily trained on web data"* — so they **removed all markdown markers**.

The generalizable lesson: **extraction decisions are training decisions.** If code blocks lose their indentation, the model cannot learn Python. If `<sup>` tags are flattened, `x2` and `x²` collapse. If LaTeX `$...$` is stripped, every equation becomes word salad. A checklist:

| Must preserve | Why |
|---|---|
| **Code block boundaries and indentation** | Python is whitespace-significant; 17% of Llama 3's mix is code |
| **LaTeX / MathML** | arXiv and textbooks are worthless as math data without it |
| **Table structure** | Rows/columns carry the relation; a flattened table is noise |
| **Paragraph and list boundaries** | Line-level dedup (§5.1) and repetition filters depend on real line breaks |
| **Document order** | Multi-column PDFs interleave paragraphs if read naively (§1.4) |

## 2.3 Normalization

Light, and deliberately so — the tokenizer is byte-level and lossless by design (§1.1 of `01`), so the extraction stage should not do the tokenizer's job.

```
NFC unicode normalization        ← canonical composition; safe
collapse runs of >2 blank lines  ← removes layout artifacts
strip zero-width / control chars ← invisible junk that survives byte-BPE as real tokens
normalize line endings           ← CRLF -> LF
DO NOT lowercase                 ← destroys information the model can use
DO NOT strip punctuation         ← same
DO NOT NFKC-fold                 ← merges characters that mean different things in code
```

---

# 3. Language Identification

## 3.1 The mechanism

A **fastText** linear classifier over character n-grams — a few milliseconds per document, and the standard choice at this scale. Llama 3 used fastText over **176 languages**; Qwen3 supports **119 languages and dialects**.

```
document ──▶ fastText lid ──▶ (label, confidence)
                                     │
              confidence < threshold ─┴─▶ discard, or route to a "mixed/unknown" bucket
```

## 3.2 Why the threshold is per-language, not global

A single global cutoff (e.g. 0.65) systematically deletes exactly the languages you were trying to add:

| Effect | Consequence |
|---|---|
| Short documents score low everywhere | Uniformly biases against languages with terse writing |
| Low-resource languages have weaker classifier support | Their true positives score below the threshold and get dropped |
| Close pairs (Hindi/Marathi, Indonesian/Malay, Serbian/Croatian) split confidence between labels | Both fall under the bar and both are lost |
| Code-switched text (very common outside English) scores low by construction | A large slice of real multilingual usage vanishes |

The fix is a **per-language threshold** calibrated on a labelled sample — and this is a large part of what "expanding from 29 to 119 languages" actually involves in engineering terms. It is rarely a matter of finding new data; it is a matter of *stopping the pipeline from throwing away the data you already crawled*.

> **Everything downstream must be per-language too.** Stop-word lists, mean-word-length bounds and n-gram repetition thresholds (§4.1) are all calibrated on English. Applied unchanged to Chinese — which has no spaces — a word-based filter rejects the entire language.

---

# 4. Filtering

Three layers, cheapest first, because the corpus is enormous and each layer shrinks the input to the next.

```
10 TB  ──▶ [heuristics: ~1 µs/doc] ──▶ 3 TB ──▶ [quality model: ~1 ms/doc] ──▶ 1 TB
                    §4.1-4.2                              §4.3
       ──▶ [safety + PII: ~1 ms/doc] ──▶ 0.9 TB
                    §4.5
```

## 4.1 Heuristic filters — the Gopher/C4/RefinedWeb rules

These originate in the Gopher paper and are near-universal. Each targets a specific, recognizable failure of web text:

| Rule | Typical bound | Catches |
|---|---|---|
| Document word count | 50 – 100,000 | Stubs, and concatenated dumps |
| Mean word length | 3 – 10 chars | Symbol spam (too low), base64/minified JS (too high) |
| Symbol-to-word ratio (`#`, `...`) | < 0.10 | Code-fence spam, truncated listings |
| Fraction of words with an alphabetic char | > 0.80 | Numeric tables, log files |
| Stop-word count (English) | ≥ 2 | Keyword-stuffed SEO pages, word lists |
| Fraction of lines starting with a bullet | < 0.90 | Navigation menus, link farms |
| Fraction of lines ending in `…` | < 0.30 | Article-teaser index pages |
| Duplicate-line fraction | < 0.30 | Boilerplate that survived extraction |
| Top 2/3/4-gram share of the doc | < 0.20 / 0.18 / 0.16 | Templated and generated spam |

> **Order matters, and the first rule to fire wins.** These are cheap conjunctions, so the pipeline short-circuits — which means the *reported reason* for a rejection depends on rule order. Useful when auditing: log the rule name, not just the verdict, so you can see which rule is doing the work on your corpus.

## 4.2 Module — `gopher_filters`

```python
STOP_WORDS = {"the","be","to","of","and","that","have","with","this","from","it","is","in","a"}

def gopher_filters(text: str, lang: str = "en") -> Optional[str]:
    # Returns None to KEEP the document, else the NAME of the rule that rejected it.
    words = text.split()
    n = len(words)
    if not (50 <= n <= 100_000):                       return "doc_length"
    mean_wl = sum(len(w) for w in words) / n
    if not (3 <= mean_wl <= 10):                       return "mean_word_length"
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:                                      return "empty"
    if text.count("#") / n > 0.10:                     return "hash_to_word_ratio"
    if text.count("...") / max(len(lines), 1) > 0.30:  return "ellipsis_lines"
    alpha = sum(1 for w in words if any(c.isalpha() for c in w)) / n
    if alpha < 0.80:                                   return "alpha_word_fraction"
    if lang == "en" and sum(1 for w in words if w.lower() in STOP_WORDS) < 2:
        return "stop_word_count"                       # <- English-only; see §3.2
    if sum(1 for l in lines if l.lstrip().startswith(("*", "-", "•"))) / len(lines) > 0.90:
        return "bullet_lines"
    if sum(1 for l in lines if l.rstrip().endswith("…")) / len(lines) > 0.30:
        return "ellipsis_end"

    c = Counter(lines)                                 # repetition: duplicated lines
    if sum(v for v in c.values() if v > 1) / len(lines) > 0.30:  return "dup_lines"
    for k, thr in ((2, 0.20), (3, 0.18), (4, 0.16)):   # repetition: dominant n-gram
        grams = [tuple(words[i:i+k]) for i in range(n - k + 1)]
        if grams:
            top = Counter(grams).most_common(1)[0][1] * k
            if top / n > thr:                          return f"top_{k}gram"
    return None                                        # keep
```

Each rule, demonstrated on a document engineered to trip exactly it:

```python
>>> for name, text in cases.items():
...     print(f"{name:<15} -> {gopher_filters(text) or 'KEEP'}")
clean prose     -> KEEP
too short       -> doc_length
no stop words   -> stop_word_count
symbol spam     -> mean_word_length
bullet list     -> bullet_lines
boilerplate     -> dup_lines
ngram loop      -> top_2gram
long words      -> mean_word_length
```

## 4.3 Model-based quality filtering

Heuristics remove *garbage*. They cannot tell a well-formed forum argument about football from a well-formed explanation of photosynthesis. That requires a model, and the field has moved through three generations:

| Generation | Method | Signal |
|---|---|---|
| **1. Reference-set classifier** | fastText trained to distinguish crawl text from a "good" reference (Wikipedia-referenced pages). Used by CCNet and Llama 1 | "Does this look like the reference?" |
| **2. Perplexity filtering** | Score each doc with a 5-gram KenLM trained on Wikipedia; keep the low-perplexity head | "Is this fluent?" |
| ⭐ **3. LLM-judge distillation** | An LLM rates a sample; a tiny classifier learns the ratings and is applied to the whole corpus | "Is this *educational*?" |

**Generation 3, concretely (FineWeb-Edu)** — the recipe worth memorizing because it is cheap and it works:

```
1. Sample 460,000 documents from the corpus
2. Score each 0-5 for educational value using Llama-3-70B-Instruct     ← expensive, but only 460k docs
3. Train a linear head on Snowflake-arctic-embed-m embeddings          ← cheap classifier, F1 = 82%
4. Run the cheap classifier over the ENTIRE corpus, keep score >= 3    ← 15T -> 1.3T tokens
```

The result — **1.3 T tokens, a ~90% cut** — outperformed the full unfiltered corpus on knowledge and reasoning benchmarks. Threshold 3 was chosen by ablation; 2 kept too much, 4 cut too deep.

**The general pattern is "annotate small, distill, apply large."** You cannot run a 70B model over 15 T tokens — that is more compute than the training run. You *can* run it over 0.5 M documents and teach a 100 M-parameter classifier what it learned.

## 4.4 Qwen3's instance-level annotation

Qwen3 pushes this furthest of any published system:

> *"We have developed a multilingual data annotation system... This system has been applied to our large-scale pre-training datasets, annotating over **30 trillion tokens** across multiple dimensions such as **educational value, fields, domains, and safety**. ... Unlike previous studies that optimize the data mixture at the data source or domain level, our method optimizes the data mixture at the **instance-level** through extensive ablation experiments on small proxy models with the fine-grained data labels."*

Two things are new here:

1. **The annotation is multi-dimensional, not a single quality score.** Every document carries labels for educational value *and* field *and* domain *and* safety — so the mixture can be steered along any of those axes later, without re-annotating.
2. **Mixing happens per document, not per source.** The classical approach says "web gets 50%, code gets 17%" (§8.1). Qwen3's says "*this particular document* is included with *this* weight, based on its labels." Source-level weighting cannot express "keep the good 5% of a mediocre domain"; instance-level weighting can.

The cost is a 30 T-token annotation pass — which is only affordable because the annotator is small and the labels are reused across all three pretraining stages (§8.6).

## 4.5 Safety, toxicity, and PII

| Concern | Typical handling |
|---|---|
| **Domain blocklists** | Llama 3 removed *"data from websites likely to contain unsafe content or high volumes of PII"* — adult content, and domains ranked harmful. Cheapest possible filter: it acts on the URL, before extraction |
| **Document classifiers** | Toxicity/NSFW scoring on surviving text |
| **PII** | Regex + NER for emails, phone numbers, national IDs, credit cards → redact or drop the document |
| **Copyright / licence** | Code licence detection; opt-out registries; robots.txt compliance at crawl time |

> **Over-filtering has a documented cost.** Aggressive toxicity filtering measurably degrades the model's ability to *discuss* the topics it filtered — including for safety classification, where you need a model that recognizes toxic content. Most labs keep a small, deliberately retained slice and handle the rest at post-training (§11) rather than by erasing the concept from the base model.

## 4.6 Doubt — "isn't more data always better?"

**No, and the FineWeb-Edu result is the cleanest disproof:** 1.3 T carefully filtered tokens beat 15 T unfiltered ones on knowledge and reasoning, *at the same training compute*.

The reason is that a training step has a fixed cost regardless of what is in it. Spending a step on SEO spam is not neutral — it is a step not spent on a physics explanation, plus a gradient update that actively teaches the model to produce SEO spam.

```
fixed compute budget C  ->  fixed number of tokens seen, T = C / (6N)

The only question is: WHICH T tokens?
```

The caveat that keeps this honest: **filtering trades diversity for quality, and diversity has its own value.** Filter to only textbook-style prose and the model becomes fluent in one register and helpless in every other — dialogue, code comments, informal writing, low-resource languages. This is why threshold 3, not 4, won in the FineWeb ablation, and why every real corpus is a *mixture* (§8) rather than a single filtered stream.

---

# 5. Deduplication

The single highest-leverage stage, and the one with the most surprising failure mode.

**Why duplicates are actively harmful, not merely wasteful:**

| Effect | Mechanism |
|---|---|
| **Memorization** | A passage seen 100× is memorized verbatim rather than generalized from — the main driver of training-data extraction attacks |
| **Wasted compute** | Duplicated tokens consume steps that teach nothing new |
| **Distorted mixture** | Popular boilerplate (licences, cookie notices, Wikipedia mirrors) is over-represented by orders of magnitude, silently reweighting the corpus |
| **Eval leakage** | Benchmark text is mirrored across many sites; dedup is the first line of defence, decontamination (§6) the second |

## 5.1 The three levels — Llama 3's published ladder

```
1. URL-LEVEL      keep only the most recent version of each URL
                  ↳ cheapest possible; removes re-crawls of the same page

2. DOCUMENT-LEVEL MinHash near-duplicate removal (§5.3)
                  ↳ the expensive one; removes mirrors, reposts, template-generated pages

3. LINE-LEVEL     remove lines appearing >6 times within each bucket of 30M documents
                  ↳ kills surviving boilerplate: nav bars, footers, licence headers
```

Each catches something the others structurally cannot. URL dedup cannot see a page copied to another domain. Document dedup cannot see a footer repeated on a million *otherwise different* pages. Line dedup cannot see a whole article reposted with a new intro.

## 5.2 Exact duplicates

Hash the normalized text (`sha256` of NFC-normalized, whitespace-collapsed content) and keep the first occurrence. Linear time, trivially distributable, and it disposes of a large fraction of the problem before any fuzzy method runs.

## 5.3 MinHash + LSH — the algorithm

Exact hashing misses the real case: two documents that differ by a date, a byline, or one inserted paragraph. The target is **Jaccard similarity** over word shingles:

```
J(A, B) = |A ∩ B| / |A ∪ B|        A, B = the sets of 5-word shingles in each document
```

Computing this for all pairs is `O(n²)` — impossible at 10¹⁰ documents. Two ideas fix it.

**Idea 1 — MinHash estimates Jaccard with a fixed-size signature.** For a random permutation `π` of the universe of shingles:

```
P[ min(π(A)) == min(π(B)) ] = J(A, B)         ← exactly, not approximately
```

So take `n = 128` independent hash functions, keep the minimum of each over the document's shingles, and you have a 128-integer signature whose **fraction of matching positions is an unbiased estimator of `J`**, with standard error `≈ 1/√n ≈ 9%`.

**Idea 2 — LSH turns a similarity search into a hash lookup.** Split the 128-value signature into `b` bands of `r` rows (`b·r = 128`). Two documents become *candidates* if they match on **all `r` rows of at least one band**:

```
P(candidate | J) = 1 − (1 − J^r)^b
```

That is an S-curve with a knee at

```
threshold ≈ (1/b)^(1/r)
```

With `b = 16, r = 8`: threshold ≈ **0.707**. Tuning `b` and `r` moves the knee — more bands means more recall and more false candidates; more rows means the opposite.

```
        b=16, r=8      P(candidate)
  1.0 │                        ▁▄██████
      │                     ▄██
      │                   ▄█
  0.5 │                 ▄█  ← knee at J ≈ 0.71
      │               ▄█
      │        ▁▁▄▄██
  0.0 │▁▁▁▁▁▁▁▁
      └────────────────────────────────▶  Jaccard similarity
      0.0      0.5   0.7        1.0
```

## 5.4 Module — `MinHasher`, `LSHIndex`, `dedup`

```python
_MERSENNE = (1 << 61) - 1                       # a Mersenne prime: fast mod, good spread

class MinHasher:
    def __init__(self, num_perm=128, shingle_n=5, seed=0):
        rng = np.random.RandomState(seed)
        self.a = rng.randint(1, _MERSENNE, num_perm, dtype=np.uint64)   # the n "permutations"
        self.b = rng.randint(0, _MERSENNE, num_perm, dtype=np.uint64)   #   as (a·h + b) mod p
        self.n, self.k = num_perm, shingle_n

    def shingles(self, text):
        w = text.lower().split()
        if len(w) < self.k:
            return {" ".join(w)} if w else set()
        return {" ".join(w[i:i+self.k]) for i in range(len(w) - self.k + 1)}

    def signature(self, text):
        sh = self.shingles(text)
        if not sh:
            return np.zeros(self.n, dtype=np.uint64)
        h = np.array([int.from_bytes(hashlib.sha1(s.encode()).digest()[:8], "big")
                      for s in sh], dtype=np.uint64)                    # [n_shingles]
        m = (self.a[None, :] * h[:, None] + self.b[None, :]) % np.uint64(_MERSENNE)
        return m.min(axis=0)                     # [num_perm] — the whole document, 1 KB


class LSHIndex:
    def __init__(self, num_perm=128, bands=16):
        assert num_perm % bands == 0
        self.bands, self.rows = bands, num_perm // bands
        self.buckets = [dict() for _ in range(bands)]

    @property
    def threshold(self):                         # the knee of the S-curve
        return (1.0 / self.bands) ** (1.0 / self.rows)

    def prob_candidate(self, jaccard):
        return 1 - (1 - jaccard ** self.rows) ** self.bands

    def query_and_add(self, doc_id, sig):
        hits = set()
        for i in range(self.bands):
            key = sig[i*self.rows:(i+1)*self.rows].tobytes()   # r rows -> one bucket key
            b = self.buckets[i]
            if key in b:
                hits.update(b[key])              # anything already here is a candidate
            else:
                b[key] = []
            b[key].append(doc_id)
        return hits


def dedup(docs, num_perm=128, bands=16, shingle_n=5):
    mh, idx = MinHasher(num_perm, shingle_n), LSHIndex(num_perm, bands)
    keep, dropped = [], []
    for i, d in enumerate(docs):
        (dropped if idx.query_and_add(i, mh.signature(d)) else keep).append(i)
    return keep, dropped
```

**The S-curve, and the estimator, both verified:**

```python
>>> idx = LSHIndex(128, 16); idx.threshold
0.707
>>> for j in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9):
...     print(j, round(idx.prob_candidate(j), 3))
0.5  0.061          ← genuinely different docs almost never collide
0.6  0.237
0.7  0.613
0.75 0.815          ← FineWeb's operating point
0.8  0.947
0.9  1.000          ← near-identical docs are always caught

>>> mh = MinHasher(128, 5)
>>> jaccard_true(a, b), (mh.signature(a) == mh.signature(b)).mean()
(0.750, 0.742)      ← 128 integers estimate the true Jaccard to ~1%
```

> **Cost at scale.** Signatures are 128 × 8 bytes = **1 KB per document**, independent of document length. Ten billion documents is 10 TB of signatures — large, but it is a single distributed shuffle-and-group, not `O(n²)` comparisons. This is the step that makes trillion-token deduplication tractable at all.

## 5.5 Doubt — "why not deduplicate globally across every snapshot?"

**Because it makes models worse, and this is one of the most useful counter-intuitive results in the field.**

The FineWeb team ran exactly this experiment. Deduplicating each Common Crawl snapshot *independently* beat deduplicating globally across all snapshots — and not marginally:

```
GLOBAL dedup across all snapshots
  → removes ~90% of content from OLDER snapshots
    (anything that was also re-crawled later is deleted from the old one)
  → what survives in old snapshots is the content NOBODY re-published
  → i.e. you have systematically kept the least-referenced, lowest-quality material
  → model gets WORSE

PER-SNAPSHOT dedup
  → each snapshot keeps its own good content
  → text that appears across many snapshots survives, and is seen a few times
  → mild repetition of durable, widely-mirrored content — which is a QUALITY SIGNAL
  → model gets BETTER
```

The mechanism is a selection effect. **Being duplicated is correlated with being worth duplicating.** Global dedup removes the duplicate copies but, because it keeps only *one* arbitrary copy and deletes the rest of that snapshot's overlap, it inverts the quality gradient across snapshots. None of the alternative global schemes the FineWeb team tried recovered the loss.

> **The general lesson:** deduplication is not "remove redundancy". It is a *sampling* intervention on the corpus distribution, and its second-order effects on that distribution can dominate the first-order saving. Measure it with a proxy model (§8.4); do not assume more aggressive is better.

## 5.6 Semantic deduplication

MinHash finds *lexical* near-duplicates. It cannot see that two articles say the same thing in different words. **SemDedup** embeds every document, clusters (k-means), and drops points closer than a threshold to a retained neighbour within a cluster.

| | MinHash | SemDedup |
|---|---|---|
| Detects | shared word sequences | shared *meaning* |
| Cost | ~1 KB/doc, hashing only | an embedding forward pass per document |
| Catches | mirrors, reposts, templates | paraphrases, translations, restated facts |
| Risk | low | **high** — "same meaning" is exactly what a diverse corpus needs many examples of |

Used at the tail of the pipeline on already-filtered data, at conservative thresholds. It is a refinement, not a replacement.

---

# 6. Decontamination

Removing evaluation-benchmark text from the training corpus. Without it, every reported benchmark number is partly a memorization score.

## 6.1 The method

Build an index of every n-gram in every evaluation set, then reject any training document that shares one.

```
n = 8-13 tokens or words   ← 13 is the common choice (GPT-3, Llama)
                              short n over-triggers on common phrases;
                              long n misses lightly-edited copies
```

| Granularity | Action on a hit |
|---|---|
| **Document-level** | Drop the whole document — safest, most wasteful |
| **Span-level** | Excise the matching window ± a margin — keeps the surrounding text |
| **Flag-only** | Keep it, but record contamination so eval results can be discounted |

## 6.2 Module — `build_contam_index`, `is_contaminated`

```python
def build_contam_index(eval_texts, n=13):
    idx = set()
    for t in eval_texts:
        w = t.lower().split()
        for i in range(max(0, len(w) - n + 1)):
            idx.add(hash(tuple(w[i:i+n])))      # in production: a Bloom filter, not a set
    return idx

def is_contaminated(text, idx, n=13):
    w = text.lower().split()
    return any(hash(tuple(w[i:i+n])) in idx for i in range(max(0, len(w) - n + 1)))
```

```python
>>> ev = ["what is the capital city of the country france in western europe today please answer"]
>>> ci = build_contam_index(ev, n=13)
>>> is_contaminated("preamble " + ev[0] + " postamble", ci)
True                                    # the benchmark question, embedded in a web page
>>> is_contaminated("a totally different sentence about birds", ci)
False
```

> **A `set` of Python hashes is a teaching device.** At real scale the index holds 10⁸–10⁹ n-grams and is queried against 10¹³ tokens, so production uses a **Bloom filter** (a few bits per n-gram, tunable false-positive rate) or a distributed suffix-array join. A false positive discards one innocent document — cheap. A false negative inflates a headline benchmark number — expensive.

## 6.3 Doubt — "why is decontamination so hard to get right?"

Because the failure modes are asymmetric and mostly invisible:

| Problem | Why it defeats n-gram matching |
|---|---|
| **Paraphrase and translation** | A benchmark question restated, or in another language, shares no n-gram. Fully invisible |
| **You must know the benchmark in advance** | A model trained in 2024 cannot be decontaminated against a benchmark published in 2025 — which is precisely why new benchmarks show lower scores |
| **Reformatted copies** | Benchmarks are mirrored into blog posts, quiz apps and GitHub in different formats; whitespace and markup changes break exact matching |
| **Synthetic data inherits it** (§1.3) | If the generating model memorized a benchmark, its output can reproduce it — and that output was never in your crawl to check |
| **Over-removal is unmeasurable** | Aggressive `n` deletes legitimate documents about the same topic. You cannot easily tell a leaked MMLU item from a genuine textbook passage on the same fact |

**The practical stance:** decontaminate what you can, report the overlap rate you measured, and treat a *held-out, never-published* internal evaluation as the real gate. A public benchmark is for calibration against other labs' numbers, not for deciding whether your model improved.

---

# 7. Tokenization at Scale

`01` §1 covers the BPE algorithm. This section covers what changes when the corpus is 36 T tokens.

## 7.1 Training the tokenizer

Trained on a **sample** — a few hundred GB — not the full corpus, because merge counting is superlinear and the merge table converges long before the data runs out. Two decisions matter:

| Decision | Consequence |
|---|---|
| **The sample must mirror the final mixture** | Train the tokenizer on English web text and then train the model on 119 languages and 17% code, and fertility on everything except English is terrible (`01` §1.9) — permanently, since the tokenizer is frozen for the model's life |
| **Vocabulary size** | Qwen3: **151,669** BBPE tokens; Llama 3: 128,256 (100k tiktoken + 28k added non-English, improving compression from **3.17 to 3.94 characters/token**); DeepSeek-V3: 128k |

That Llama 3 number is the whole argument for large vocabularies in one statistic: +28k tokens bought **24% more characters per token — ~19.5% fewer tokens for the same text** (1 − 3.17/3.94), which is ~19.5% less compute for the same training corpus.

## 7.2 Tokenize once, store binary

```
text shards (parquet/jsonl.zst)  ──tokenize once──▶  train.bin   flat uint32 token stream
                                                     train.idx   document start offsets
```

Re-tokenizing during training would put a CPU-bound Python-speed operation on the critical path of a thousand GPUs. Instead the corpus is tokenized once, offline, into a flat binary that can be **memory-mapped** — so a training process reads token ranges at page-fault speed with no parsing and no per-epoch CPU cost.

## 7.3 Module — `ShardWriter`, `ShardReader`

```python
class ShardWriter:
    # Megatron/nanoGPT layout: one flat .bin of ids + an .idx of document offsets.
    def __init__(self, path, dtype="uint32"):
        self.bin = open(f"{path}.bin", "wb")
        self.dtype = np.dtype(dtype)
        self.offsets = [0]

    def add(self, ids):
        a = np.asarray(ids, dtype=self.dtype)
        self.bin.write(a.tobytes())
        self.offsets.append(self.offsets[-1] + len(a))

    def close(self, path):
        self.bin.close()
        np.save(f"{path}.idx.npy", np.array(self.offsets, dtype=np.int64))
        return self.offsets[-1]


class ShardReader:
    def __init__(self, path, dtype="uint32"):
        self.tokens  = np.memmap(f"{path}.bin", dtype=dtype, mode="r")   # NOT loaded into RAM
        self.offsets = np.load(f"{path}.idx.npy")

    def __len__(self):            return len(self.offsets) - 1
    def doc(self, i):             return self.tokens[self.offsets[i]:self.offsets[i+1]]
    @property
    def n_tokens(self):           return int(self.offsets[-1])
```

```python
>>> w = ShardWriter(path, "uint32")
>>> for d in docs: w.add(d)
>>> w.close(path)
>>> r = ShardReader(path, "uint32")
>>> len(r), r.n_tokens, list(r.doc(7)) == docs[7]
(50, 5104, True)                         # memmap roundtrip is exact
>>> os.path.getsize(path + ".bin") / r.n_tokens
4.0                                      # bytes per token — uint32, per §0.4
```

**Scaling that last line:** 4 bytes/token × 36 T tokens = **144 TB** for the tokenized Qwen3 corpus. Sharded across thousands of files on a parallel filesystem or object store, and every training node memory-maps only the shards it is assigned.

## 7.4 The token-boundary bias — a real bug worth knowing

DeepSeek-V3 documents a subtle failure caused by their own pretokenizer optimization:

> Their BBPE pretokenizer *"combines punctuations and line breaks"* into single tokens for better compression. The side effect: **token boundary bias** — *"when the model processes multi-line prompts without terminal line breaks, particularly for few-shot evaluation prompts"*, the prompt ends mid-way through a token pattern the model never saw during training.

The fix was on the **data** side, not the model side: *"randomly splitting a certain proportion of such combined tokens during training"*, so the model sees both the merged and the split form and is robust to either.

> This is the general shape of a whole class of bugs: **a tokenizer optimization that is correct on complete documents can be wrong on truncated prompts.** Training data is always complete; inference prompts often are not. The remedy is to inject the inference-time distribution into training — here, by deliberately un-merging some tokens.

---

# 8. Data Mixing

Filtering decides *what is allowed in*. Mixing decides *how often the model sees each kind*, and it is at least as consequential.

## 8.1 Domain weights

Llama 3 is the clearest published mix:

```
general knowledge   50%   ████████████████████████████████████████████████████
math & reasoning    25%   ██████████████████████████
code                17%   ██████████████████
multilingual         8%   ████████
```

Two things are surprising the first time you see it:

- **Code is 17%** of a general-purpose model's diet. Code is not there to make the model write code — it is there because code teaches long-range dependency tracking, precise syntax, and stepwise procedure, and those transfer to reasoning.
- **Multilingual is only 8%** in a model marketed as multilingual. Cross-lingual transfer is strong enough that a small slice buys a lot. Qwen3's 119-language ambition implies a much larger share.

Weights are *not* the natural proportions of the corpus. Web text is 90%+ of the raw supply and is downweighted to 50%; Wikipedia is a fraction of a percent and is upsampled heavily. **The mixture is a deliberate distortion of what you have, toward what you want.**

## 8.2 Doubt — "how many times can the model see the same token?"

Up to about **4 epochs, repeated data is nearly as good as fresh data.** Beyond that, returns fall sharply, and by ~16 epochs additional repetition contributes essentially nothing.

```
epochs of a fixed corpus:   1      2      3      4        8        16
value of a repeated token: 100%   ~95%   ~90%   ~85%    ~50%     ~0%
                            └──── the usable range ────┘
```

This is what makes the 36 T budget reachable (§1.6): a 10 T-token unique corpus can support a ~36 T-token training run without collapse, and it is why *upsampling* high-quality sources is legitimate rather than cheating — Wikipedia at 5 epochs still contributes more than web spam at 1.

The corollary matters for the mixture: **different sources get different epoch counts.** A single "50% web" number usually means "web is 50% of tokens *seen*", achieved by sampling a large web pool less than once and a small curated pool several times.

## 8.3 Instance-level mixing

Source-level weighting ("code 17%") is coarse: it cannot express *"keep the excellent 5% of a mediocre domain and drop the rest."* Qwen3's instance-level approach (§4.4) can, because every document already carries labels for educational value, field, domain and safety. The mixture becomes a function over labels rather than a table over sources.

```
SOURCE-LEVEL              w(source)              4-10 numbers to tune
DOMAIN-LEVEL              w(source, domain)      hundreds
INSTANCE-LEVEL   ⭐        w(labels(document))    a policy over a label space
```

## 8.4 How mixtures are actually chosen — proxy-model ablation

Not by intuition. The standard method, and the one every result quoted in this file rests on:

```
1. Fix a small model (100M - 1.5B params) and a fixed token budget (10-100B tokens)
2. Train one model per candidate corpus / mixture / filter threshold
3. Evaluate on a benchmark suite with LOW VARIANCE at small scale
4. Pick the winner; assume the ranking transfers to the full-scale run
```

Qwen3 states this explicitly: instance-level mixtures were chosen *"through extensive ablation experiments on small proxy models with the fine-grained data labels."* FineWeb's entire filtering recipe — including the counter-intuitive per-snapshot dedup result (§5.5) — came from ~200 such ablation runs.

| Cost | Payoff |
|---|---|
| Each ablation is ~0.01–0.1% of the full run | A 1% quality gain on a $50M run is worth far more than the ablations cost |
| Rankings mostly transfer across scale | But **not always** — filtering that helps a 150M model can be too aggressive for a 70B model, which can extract signal from noisier text |

> **This is the single most important process fact in the file.** Every claim of the form "X improves the data" should be read as "X won an ablation at some scale, on some benchmark suite." When you cannot run the ablation, you are guessing.

## 8.5 Module — `MixtureSampler`

```python
class MixtureSampler:
    def __init__(self, weights, seed=0):
        self.names = list(weights)
        p = np.array([weights[k] for k in self.names], dtype=np.float64)
        self.p = p / p.sum()                       # normalize; weights need not sum to 1
        self.rng = np.random.default_rng(seed)
        self.counts = Counter()

    def next_source(self):
        s = self.names[self.rng.choice(len(self.names), p=self.p)]
        self.counts[s] += 1
        return s

    def realized(self):                            # ALWAYS log this
        tot = sum(self.counts.values())
        return {k: self.counts[k] / tot for k in self.names} if tot else {}
```

```python
>>> s = MixtureSampler({"web": .50, "math_reasoning": .25, "code": .17, "multilingual": .08})
>>> for _ in range(200_000): s.next_source()
>>> s.realized()
{'web': 0.502, 'math_reasoning': 0.249, 'code': 0.170, 'multilingual': 0.079}
```

> **Log the realized mixture, not the configured one.** A source that runs out mid-run, a shard that fails to mount, or an off-by-one in shard assignment all silently change the actual proportions. The configured weights are an intention; `realized()` is what the model was trained on, and it belongs in the run's metadata.

## 8.6 Curriculum — the mixture changes during training

Modern runs are staged, not uniform. Qwen3's three stages are the clearest published example:

| Stage | Tokens | Seq len | Mixture change | Purpose |
|---|---|---|---|---|
| **S1 — General** | **> 30 T** | 4,096 | broad, 119 languages | Language proficiency and general world knowledge |
| **S2 — Reasoning** | **~5 T** | 4,096 | **increase STEM, coding, reasoning, synthetic**; also accelerate LR decay | Reasoning ability |
| **S3 — Long context** | hundreds of B | **32,768** | 75% of text 16,384–32,768 tokens long, 25% 4,096–16,384 | Extend usable context |

Llama 3 uses the same idea under a different name — **annealing**: upweight high-quality data sharply over the final tokens while the learning rate decays to zero. The reported effect on the 8B model was **+24.0% on GSM8k and +6.4% on MATH**; on the 405B model the gain was *negligible*, because a model that large already has the in-context ability those benchmarks test.

> **Why staging works at all.** Late training, at a low learning rate, makes small and targeted weight changes. Spending those specifically on your best data is strictly better than spending them on an average sample of it. The corollary is that **the last few percent of tokens carry disproportionate influence over the final model** — which is why S2/annealing data gets the most scrutiny per token of anything in the corpus.

> **A second corollary:** the "annealing test" — train briefly on a candidate dataset during the anneal and measure the benchmark delta — has become a standard *cheap evaluator of data quality* in its own right.

---

# 9. Packing and Batching — How Bytes Reach the GPU

This is where the corpus becomes the `[B, S]` tensor `01` §15.1 consumes.

## 9.1 The problem: documents have wildly different lengths

```
document lengths in a web corpus (log scale)
   ▲
   │████                            median ≈ 500 tokens
   │████▄▄
   │██████▄▄▄
   │██████████▄▄▄▄▄▄▄
   │                  ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄  <1% exceed 32k
   └──────────────────────────────────────────▶
   10        100      1k      10k      100k
```

**Padding each document to `S = 4096` would waste ~85% of every batch** — 85% of the FLOPs of a $50M training run spent multiplying zeros. Nobody does this. Instead documents are **packed**: concatenated end to end with an `EOS` separator and sliced into fixed `S`-length windows, so every position in every batch is a real token.

```
docs:    [A A A] [B B] [C C C C C]  [D D]
concat:   A A A ⏎ B B ⏎ C C C C C ⏎ D D ⏎        ⏎ = EOS
chunk:   │← S = 8 →│← S = 8 →│ ...
          A A A ⏎ B B ⏎ C     C C C C ⏎ D D ...
                        └── document C is SPLIT across two windows
```

Two consequences fall out, and both are real design decisions:

1. **One window contains several documents.** Should token 5 (in doc B) be able to attend to token 1 (in doc A)? → §9.2
2. **Documents are split across windows.** The tail of C loses the head of C. This is accepted: it is a small fraction of positions, and it is what lets long documents train at all.

## 9.2 Document masking — and DeepSeek's deliberate choice not to

**Intra-document masking** restricts attention so a token can only see tokens from its *own* document — the causal mask of `01` §4.3, AND-ed with a same-document constraint.

```
packed row:   A A A ⏎ B B ⏎ C
doc ids:      0 0 0 0 1 1 1 2

CAUSAL ONLY                    CAUSAL AND SAME-DOCUMENT
  1 . . . . . . .                1 . . . . . . .
  1 1 . . . . . .                1 1 . . . . . .
  1 1 1 . . . . .                1 1 1 . . . . .
  1 1 1 1 . . . .                1 1 1 1 . . . .
  1 1 1 1 1 . . .                . . . . 1 . . .   ← B cannot see A
  1 1 1 1 1 1 . .                . . . . 1 1 . .
  1 1 1 1 1 1 1 .                . . . . 1 1 1 .
  1 1 1 1 1 1 1 1                . . . . . . . 1   ← C sees only itself
```

| | Causal only | + document mask |
|---|---|---|
| Cost | free | an extra mask term; breaks the simple `is_causal` fast path |
| Attention can | cross document boundaries | never cross them |
| Used by | ⭐ **DeepSeek-V3** — *"we do not incorporate cross-sample attention masking during training"* | ⭐ **Llama 3**, especially for long-context training |

**Both choices are defensible, and knowing why is the point:**

- **Against masking:** cross-document attention is *mostly harmless noise* at 4k context — the model quickly learns that content before an `EOS` is irrelevant, and the unmasked path is faster and simpler. DeepSeek took this position explicitly.
- **For masking:** at 32k–128k context a window holds *many* documents, so unmasked attention spends a large fraction of its score matrix on genuinely unrelated text, and the model learns spurious long-range "dependencies" that do not exist. This is why long-context stages (§10) are where masking earns its cost.

> **Position IDs must be reset too, or masking is half-done.** If document B starts at index 4 of the window, its first token should be at position 0, not 4 — otherwise RoPE (`01` §3.3) tells the model that B's opening sentence is four tokens into a document, and the "distance" between two tokens in *different* documents is computed as if they were in one.

## 9.3 Module — `pack_documents`, `document_position_ids`, `document_mask`

```python
def pack_documents(docs, seq_len, eos_id):
    # Concat-and-chunk. Returns (tokens[N, seq_len+1], doc_ids[N, seq_len+1]).
    # The +1 is so each row yields inputs[:-1] and targets[1:] with no extra read.
    flat_tok, flat_doc = [], []
    for d, ids in enumerate(docs):
        flat_tok.extend(ids); flat_tok.append(eos_id)
        flat_doc.extend([d] * (len(ids) + 1))
    n = (len(flat_tok) - 1) // seq_len                      # drop the ragged tail
    t = np.array(flat_tok[:n*seq_len + 1])
    v = np.array(flat_doc[:n*seq_len + 1])
    return (np.stack([t[i*seq_len:(i+1)*seq_len + 1] for i in range(n)]),
            np.stack([v[i*seq_len:(i+1)*seq_len + 1] for i in range(n)]))


def document_position_ids(doc_ids):
    # Restart position numbering at every document boundary (see the note in §9.2).
    pos = np.zeros_like(doc_ids)
    for r in range(doc_ids.shape[0]):
        start = 0
        for i in range(1, doc_ids.shape[1]):
            if doc_ids[r, i] != doc_ids[r, i-1]:
                start = i
            pos[r, i] = i - start
    return pos


def document_mask(doc_ids_row):
    # Additive mask: causal AND same-document (Llama-3 style intra-doc masking).
    i = np.arange(len(doc_ids_row))
    causal   = i[None, :] <= i[:, None]                       # the lower triangle
    same_doc = doc_ids_row[None, :] == doc_ids_row[:, None]   # the extra constraint
    return np.where(causal & same_doc, 0.0, -np.inf)
```

```python
>>> docs = [[1,2,3], [4,5], [6,7,8,9,10], [11,12]]
>>> t, d = pack_documents(docs, seq_len=8, eos_id=0)
>>> t
[[1 2 3 0 4 5 0 6 7]]          # doc 1, EOS, doc 2, EOS, doc 3 (truncated)
>>> d
[[0 0 0 0 1 1 1 2 2]]          # which document each position belongs to
>>> document_position_ids(d)
[[0 1 2 3 0 1 2 0 1]]          # ← restarts at each boundary, NOT 0..8
>>> show(document_mask(d[0]))  # 1 = may attend
1........
11.......
111......
1111.....
....1....                      # document 2 starts: cannot see document 1
....11...
....111..
.......1.                      # document 3
.......11
```

That mask plugs directly into `01` §4.5.2 — it is added to `scores` in exactly the same place as the causal mask, because it *is* the causal mask plus one more conjunct.

## 9.4 Fill-in-the-middle

Autoregressive training only ever teaches left-to-right continuation. But code assistants must *insert* at a cursor with code on both sides. FIM fixes this by rewriting a fraction of documents into a form where the middle is predicted last:

```
original            [ prefix ][ middle ][ suffix ]
PSM rearrangement   <pre> prefix <suf> suffix <mid> middle
                                                     └── still predicted left-to-right,
                                                         but conditioned on BOTH sides
```

DeepSeek-V3 applies **Prefix-Suffix-Middle (PSM) at a rate of 0.1** at the document level.

```python
def fim_transform(ids, rng, rate=0.1, pre=1, suf=2, mid=3):
    if rng.random() >= rate or len(ids) < 4:
        return list(ids)
    a, b = sorted(rng.integers(1, len(ids) - 1, size=2).tolist())
    prefix, middle, suffix = ids[:a], ids[a:b], ids[b:]
    return [pre, *prefix, suf, *suffix, mid, *middle]
```

```python
>>> fim_transform(list(range(10)), rng, rate=1.0)
[1, 0, 1, 2, 2, 5, 6, 7, 8, 9, 3, 3, 4]
 └pre┘ └prefix┘ └suf┘ └─ suffix ─┘ └mid┘ └mid┘
>>> sum(fim_transform(seq, rng(i), rate=0.1) != seq for i in range(400)) / 400
0.087                                    # ≈ the configured 0.1
```

**The rate is a real trade-off:** FIM'd documents teach infilling but are a *scrambled* view of natural text, so too high a rate degrades ordinary left-to-right quality. 0.1 is the well-tested operating point.

## 9.5 Batch size — measured in tokens, and it ramps

The unit that matters is **tokens per optimizer step**, not sequences.

```
tokens_per_step = batch_size × seq_len × grad_accum × data_parallel_size
```

| Model | Schedule |
|---|---|
| ⭐ **Llama 3** | **4 M tokens** @ seq 4,096 → after 252 M tokens: **8 M** @ seq 8,192 → after 2.87 T tokens: **16 M** |
| **DeepSeek-V3** long-context | phase 1: 32 K seq × batch **1,920**; phase 2: 128 K seq × batch **480** (constant ~62 M tokens/step) |
| **Qwen3** | batch size and LR predicted per model and per stage from fitted scaling laws |

**Why it ramps.** Early in training, gradients are large and consistent, so small batches make rapid progress and a huge batch would waste samples on a direction you already know. Later, gradients are small and noisy, and large batches are needed to extract signal — the "critical batch size" grows as loss falls. Starting at the final batch size wastes early compute; ending at the initial one wastes late compute.

> Note the DeepSeek row: as sequence length went 32K → 128K, the sequence count went 1920 → 480 — **exactly 4× down for 4× up.** Tokens per step was held constant, because that is the quantity the optimizer's hyperparameters were tuned for.

## 9.6 The dataloader's real requirements

At this scale the loader is infrastructure, not a utility:

| Requirement | Why |
|---|---|
| **Deterministic and seeded** | A run must be reproducible, and a resumed run must not re-show data |
| ⭐ **Resumable mid-epoch** | Runs of this size crash. Restarting from step 0 is unthinkable; re-showing the last 3 days of data silently corrupts the epoch schedule |
| **Sharded without overlap** | Every data-parallel rank must read a disjoint slice, or the effective batch contains duplicates |
| **Prefetching / double-buffered** | A stalled loader idles thousands of GPUs; the loader must always be a step ahead |
| **Order-independent of world size** | Restarting on a different number of nodes should not change the data order |

## 9.7 Module — `TokenLoader`

```python
class TokenLoader:
    def __init__(self, packed, batch_size, seed=0, start_step=0):
        self.packed, self.bs = packed, batch_size
        self.n = len(packed)
        self.order = np.random.default_rng(seed).permutation(self.n)   # seeded, fixed
        self.step = start_step                                          # <- resume point

    def __iter__(self):
        while True:
            lo = (self.step * self.bs) % self.n
            idx = self.order[lo:lo + self.bs]
            if len(idx) < self.bs:                                      # wrap the epoch
                idx = np.concatenate([idx, self.order[:self.bs - len(idx)]])
            self.step += 1
            batch = self.packed[idx]
            yield batch[:, :-1], batch[:, 1:], self.step                # inputs, targets

    def state_dict(self):
        return {"step": self.step}          # the ONLY state needed — order is derived
```

```python
>>> L = TokenLoader(packed, batch_size=4, seed=0)
>>> inputs, targets, step = next(iter(L))
>>> inputs.shape, targets.shape, step
((4, 16), (4, 16), 1)
>>> (inputs[:, 1:] == targets[:, :-1]).all()
True                                     # targets are inputs shifted by one

>>> L2 = TokenLoader(packed, batch_size=4, seed=0, start_step=1)   # resume from step 1
>>> (next(iter(L2))[0] == second_batch_of_L).all()
True                                     # byte-identical continuation
```

> **The design point is that the checkpoint stores one integer.** Because the permutation is derived from `(seed, n)` and the position from `step`, resumption needs no saved index list, no shuffled-file manifest, and no replay. Any loader whose resume state grows with the dataset will eventually become the reason a run cannot be restarted.

## 9.8 Doubt — "what exactly is one training step?"

Concretely, for Llama 3 at its final batch size:

```
16,000,000 tokens per step ÷ 8,192 tokens per sequence = 1,953 sequences
   spread over, say, 1,024 data-parallel ranks           = ~2 sequences per rank per step
   × 126 layers (405B) × forward + backward
   = one optimizer update

15.6 T tokens ÷ 16 M tokens/step ≈ 975,000 optimizer steps
```

**Fewer than a million weight updates produce a frontier model.** That is the number to hold on to: each step is enormous (16 M tokens), and there are surprisingly few of them. It is why every step must be filled with real tokens (§9.1), why the last few percent are so valuable (§8.6), and why a dataloader stall of 100 ms per step costs a full day over a run.

---

# 10. Long-Context Data

## 10.1 The supply problem

A 128 K context window needs training examples that are actually 128 K tokens long — and **they barely exist**. Fewer than 1% of web documents exceed 32 K tokens; almost nothing natural reaches 128 K. This is a data problem before it is a modelling problem.

## 10.2 What the labs actually did

| Model | Approach |
|---|---|
| ⭐ **Qwen3** | A dedicated **third pretraining stage**: hundreds of billions of tokens at seq len **32,768**, from a corpus deliberately built as **75% documents of 16,384–32,768 tokens and 25% of 4,096–16,384**. RoPE base raised **10,000 → 1,000,000** (ABF). YaRN + Dual Chunk Attention then give a further **4×** at inference — so 128 K is reached partly by training and partly by inference-time extension |
| **DeepSeek-V3** | Two extension phases *after* the 14.8 T main run: **4K → 32K** (1,000 steps, batch 1,920) then **32K → 128K** (1,000 steps, batch 480), using YaRN with `s=40, α=1, β=32` |
| **Llama 3** | Context grown **in six stages** from 8 K to 128 K, over ~**800 B tokens** |

Three patterns are common to all of them:

1. **Long context is a late, separate stage** — never the main run. Attention is `O(S²)` (`01` §4.6), so 32 K training is ~8× the attention cost of 4 K; you do it on a few percent of the tokens, not on 36 T of them.
2. **The length distribution is engineered, not sampled.** Qwen3's 75/25 split is a construction, not what the corpus looks like. Feeding a natural distribution would mean almost every 32 K window was packed short documents (§9.1) and the model would never see a genuine long-range dependency.
3. **Architecture and data move together.** Raising the RoPE base (`01` §3.4) without long training data leaves the new frequencies untrained; long data without the base change feeds the model rotation angles it cannot represent. Both are required.

## 10.3 Where the long documents come from

| Source | Notes |
|---|---|
| **Books, theses, legal filings** | Genuinely long and genuinely coherent — the gold standard, and scarce |
| **Whole code repositories** | Concatenate files in dependency order: real long-range structure, and abundant |
| **Concatenated related documents** | All papers citing one another, a full documentation site — synthetic coherence, but far better than random packing |
| **Synthetic long-range tasks** | Multi-document QA, needle-in-a-haystack, long summarization — cheap to generate and directly targets the failure being fixed |

> **The evaluation trap:** needle-in-a-haystack is nearly saturated and is a *retrieval* test, not a *reasoning* test. A model can pass it while being unable to use two facts 100 K tokens apart. Long-context data should be validated on tasks that require *combining* distant information, not just locating one fact.

---

# 11. Data by Training Objective

Everything before this section described *one* pipeline. In reality a model is built from **four regimes**, and they differ in every respect that matters: what the label is, who produces it, how much of it exists, and which tokens the loss is computed on.

```
REGIME 1  PRETRAINING          10^13 tokens    label = the next token          nobody labels it
    │                                          loss on EVERY position
    ▼
REGIME 2  SFT                  10^6 examples   label = the response tokens     model or human writes it
    │                                          loss on the RESPONSE ONLY
    ▼
REGIME 3  PREFERENCE           10^5 pairs      label = "A is better than B"    human or LLM judges it
    │                                          loss on a COMPARISON
    ▼
REGIME 4  RL / VERIFIABLE      10^4-10^5 prompts  label = a reward at rollout  a checker computes it
                                               loss weighted by ADVANTAGE
```

| | Regime 1 — Pretraining | Regime 2 — SFT | Regime 3 — Preference | Regime 4 — RL |
|---|---|---|---|---|
| **Objective** | next-token prediction | next-token prediction | ranking / Bradley-Terry | policy-gradient on reward |
| **Unit of data** | a document | a (prompt, response) | a (prompt, chosen, rejected) | a (prompt, verifier) |
| **Label source** | **the data itself** (self-supervised) | written or model-generated | human or LLM comparison | executed checker / reward model |
| **Label granularity** | every token | response tokens only | one bit per pair | one scalar per rollout |
| **Volume** | 10¹³ – 10¹⁴ tokens | 10⁵ – 10⁶ examples | 10⁴ – 10⁶ pairs | 10⁴ – 10⁵ prompts |
| **Cost per example** | ~0 | $0.10 – $10 (human) | $0.50 – $5 (human) | ~0 if verifiable |
| **Quality bar** | statistical (filters) | **per-example** | **per-example** | binary correctness |
| **Failure if bad** | slow degradation | model learns the wrong style | model optimizes the wrong thing | reward hacking |
| **What it teaches** | knowledge & fluency | format & instruction-following | taste & preference | correctness & reasoning |

> **The single most important structural difference is the loss mask.** In pretraining every position is a target. In SFT the prompt is context and only the response is supervised. In preference training no token is "correct" at all — the supervision lives in a *comparison between two whole sequences*. Getting the mask wrong is the most common and most silent bug in post-training.

## 11.1 Regime 1 — Pretraining: the label is the input, shifted

**Objective:** `L = −Σ_t log P(x_t | x_<t)` over **every** position.

**Sample record** (`HuggingFaceFW/fineweb-edu`):

```json
{
  "text": "Photosynthesis is the process by which green plants convert light energy...",
  "id": "<urn:uuid:0f2d...>",
  "dump": "CC-MAIN-2024-10",
  "url": "https://example.edu/biology/photosynthesis",
  "language": "en",
  "language_score": 0.97,
  "token_count": 812,
  "score": 3.71,                     ← the ONLY label, and it is a FILTER, not a target
  "int_score": 4
}
```

**The token-level labels:**

```python
def pretrain_labels(ids):
    # The label IS the input, shifted by one. Every position is supervised.
    return np.array(ids[:-1]), np.array(ids[1:])
```

```python
>>> pretrain_labels([101, 102, 103, 104, 105])
input  [101, 102, 103, 104]
labels [102, 103, 104, 105]        # 4/4 positions supervised
```

**The rating scale — FineWeb-Edu's `score` (§4.3):**

| Level | Meaning |
|---|---|
| **0** | No educational value — ads, navigation, spam |
| **1** | Minimal — some information, mostly irrelevant or non-educational |
| **2** | Some educational content, but disorganized or shallow |
| **3** | ⭐ Suitable for education — coherent, on-topic, teaches something (**the cutoff**) |
| **4** | Highly educational — clear, structured, textbook-like |
| **5** | Outstanding — self-contained, rigorous, primary-source quality |

Note what this label is *for*: it decides **inclusion and weight**, never what the model should output. **Regime 1 has no output labels at all** — that is what "self-supervised" means, and it is why the entire first half of this document is about *selection* rather than *annotation*.

## 11.2 Regime 2 — SFT: supervise the response, mask the prompt

**Objective:** the same cross-entropy, but with the prompt tokens set to `IGNORE`.

**Sample record** (`allenai/tulu-3-sft-mixture` / `HuggingFaceTB/smoltalk` shape):

```json
{
  "messages": [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "Why is the sky blue?"},
    {"role": "assistant", "content": "Sunlight contains all colours. Air molecules scatter
                                      shorter (blue) wavelengths far more than longer (red)
                                      ones — Rayleigh scattering — so blue reaches your eyes
                                      from every direction."}
  ],
  "source": "no_robots",
  "n_turns": 2
}
```

**The token-level labels** — this is the part people get wrong:

```python
IGNORE = -100                       # PyTorch cross_entropy's default ignore_index

def sft_labels(prompt_ids, response_ids, eos_id):
    # Prompt tokens are CONTEXT, not targets: mask them out.
    input_ids = np.array([*prompt_ids, *response_ids, eos_id])
    labels    = np.array([*[IGNORE] * len(prompt_ids), *response_ids, eos_id])
    return input_ids[:-1], labels[1:]           # the standard shift

def multiturn_sft_labels(turns, eos_id):
    # turns = [(role, ids), ...]; supervise every ASSISTANT turn, mask the rest.
    input_ids, labels = [], []
    for role, ids in turns:
        input_ids.extend(ids)
        labels.extend(ids if role == "assistant" else [IGNORE] * len(ids))
        if role == "assistant":
            input_ids.append(eos_id); labels.append(eos_id)
    return np.array(input_ids[:-1]), np.array(labels[1:])
```

```python
>>> sft_labels(prompt_ids=[9,10,11,12], response_ids=[20,21,22], eos_id=2)
input  [ 9,  10,  11,  12, 20, 21, 22]
labels [-100, -100, -100, 20, 21, 22,  2]      # 4/7 supervised; prompt is masked

>>> multiturn_sft_labels([("system",[1,1]), ("user",[9,10]), ("assistant",[20,21]),
...                       ("user",[11]),    ("assistant",[22,23])], eos_id=2)
input  [   1,    1,    9, 10, 21,  2,    11, 22, 23]
labels [-100, -100, -100, 20, 21,  2, -100, 22, 23,  2]   # BOTH assistant turns + their EOS
```

Three consequences:

1. **Forget the mask and the model learns to generate user turns.** It will hallucinate the next question instead of answering, because you trained it to predict prompts.
2. **EOS must be supervised.** If the EOS token is not in the labels, the model never learns to stop — the single most common cause of endless generation after fine-tuning.
3. **Multi-turn is not "the last turn".** Every assistant turn is a training target; the user turns between them are context.

**Packing differs too.** In pretraining, packing across documents is optional (§9.2). In SFT it is **mandatory to mask across examples** — two unrelated conversations sharing a window must not attend to each other, because unlike web text they are semantically adjacent and the model *will* learn spurious continuations.

## 11.3 Regime 3 — Preference: the label is a comparison

No token is correct. The supervision is that **one whole response is better than another**.

**Sample record** (`HuggingFaceH4/ultrafeedback_binarized`):

```json
{
  "prompt": "Explain recursion to a 10-year-old.",
  "chosen": [
    {"role": "user", "content": "Explain recursion to a 10-year-old."},
    {"role": "assistant", "content": "Imagine two mirrors facing each other..."}
  ],
  "rejected": [
    {"role": "user", "content": "Explain recursion to a 10-year-old."},
    {"role": "assistant", "content": "Recursion is when a function invokes itself,
                                      establishing a base case and a recursive case..."}
  ],
  "score_chosen": 9.0,
  "score_rejected": 4.0
}
```

**Multi-attribute labels with explicit levels** (`nvidia/HelpSteer2` — the clearest published rating scheme):

| Attribute | Scale | 0 | 4 |
|---|---|---|---|
| **helpfulness** | 0–4 | not helpful at all | fully addresses the request |
| **correctness** | 0–4 | contains major errors | all facts correct and complete |
| **coherence** | 0–4 | incomprehensible | perfectly clear and consistent |
| **complexity** | 0–4 | any child could write it | requires deep domain expertise |
| **verbosity** | 0–4 | terse | verbose *(often weighted **negatively**)* |

```json
{"prompt": "...", "response": "...",
 "helpfulness": 4, "correctness": 4, "coherence": 4, "complexity": 2, "verbosity": 1}
```

```python
HELPSTEER_ATTRS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]

def helpsteer_to_scalar(ratings, weights=None):
    # 5 attributes, each 0-4. A reward model needs ONE number.
    w = weights or {"helpfulness": 0.65, "correctness": 0.8, "coherence": 0.45,
                    "complexity": 0.55, "verbosity": -0.4}     # note the negative weight
    return sum(w[a] * ratings[a] for a in HELPSTEER_ATTRS)
```

```python
>>> good = {"helpfulness":4,"correctness":4,"coherence":4,"complexity":2,"verbosity":1}
>>> bad  = {"helpfulness":1,"correctness":1,"coherence":3,"complexity":1,"verbosity":3}
>>> helpsteer_to_scalar(good), helpsteer_to_scalar(bad)
(8.30, 2.15)
```

> **The negative verbosity weight is the whole story of reward hacking in one number.** Left unweighted, reward models reliably learn "longer = better" and the policy degenerates into padding. Multi-attribute labels let you *subtract* the correlate you do not want, which single-score preference data cannot express.

**The tensors and the losses:**

```python
def preference_pair(prompt_ids, chosen_ids, rejected_ids, eos_id):
    # TWO sequences per example, each masked exactly like SFT.
    c_in, c_lab = sft_labels(prompt_ids, chosen_ids, eos_id)
    r_in, r_lab = sft_labels(prompt_ids, rejected_ids, eos_id)
    return {"chosen_input": c_in, "chosen_labels": c_lab,
            "rejected_input": r_in, "rejected_labels": r_lab}

def bradley_terry_loss(r_chosen, r_rejected):
    # REWARD-MODEL training: a scalar head, trained only to ORDER pairs.
    return -np.log(1 / (1 + np.exp(-(r_chosen - r_rejected))))

def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    # DPO: all four terms are SUM OF LOG-PROBS over positions where labels != IGNORE.
    margin = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
    return -np.log(1 / (1 + np.exp(-margin)))          # -log sigmoid(margin)
```

```python
>>> dpo_loss(-2.0, -5.0, -2.5, -4.0)     # model prefers chosen more than reference does
0.6210
>>> dpo_loss(-5.0, -2.0, -2.5, -4.0)     # model prefers REJECTED -> higher loss
0.9432
>>> dpo_loss(-2.5, -4.0, -2.5, -4.0)     # model == reference -> exactly -log(0.5)
0.6931
>>> bradley_terry_loss(2.0, 0.5), bradley_terry_loss(0.5, 2.0)
(0.2014, 1.7014)
```

**Why a reference model appears at all:** without the `− ref` terms, maximizing the margin is unbounded — the policy can win by making the rejected response infinitely unlikely, destroying fluency. The reference anchors it, and `beta` sets how far the policy may drift.

## 11.4 Regime 4 — RL: the label is computed at rollout time

The dataset contains **no responses**. It contains prompts and a way to score whatever the model produces.

**Sample record** (`openai/gsm8k`):

```json
{
  "question": "Natalia sold clips to 48 friends in April, and then she sold half as many
               clips in May. How many clips did Natalia sell altogether?",
  "answer": "In May, Natalia sold 48/2 = <<48/2=24>>24 clips.\nAltogether she sold
             48+24 = <<48+24=72>>72 clips.\n#### 72"
}
                                                                  └── the verifier's target
```

**The reward function replaces the label:**

```python
def rlvr_reward(response: str, gold: str) -> float:
    # Verifiable reward: no human, no reward model. Just a checker.
    pred = response.split("####")[-1].strip() if "####" in response else response.strip()
    return 1.0 if pred == gold.strip() else 0.0

def grpo_advantages(rewards, eps=1e-4):
    # GRPO (DeepSeek): normalize rewards WITHIN the group of G samples for one prompt.
    # No value network — the group mean IS the baseline.
    r = np.asarray(rewards, dtype=np.float64)
    return (r - r.mean()) / (r.std() + eps)
```

```python
>>> responses = ["... so the answer is #### 72", "#### 71", "#### 72",
...              "the answer is 72", "#### 72"]
>>> rewards = [rlvr_reward(r, "72") for r in responses]
[1.0, 0.0, 1.0, 0.0, 1.0]
                  ↑ CORRECT but wrongly formatted -> reward 0. See the warning below.
>>> grpo_advantages(rewards)
[0.816, -1.224, 0.816, -1.224, 0.816]        # zero-mean by construction
```

**What this table shows, per column, is the whole reason RLVR works:**

```
prompt ──▶ sample G=5 rollouts ──▶ verifier ──▶ rewards ──▶ advantages ──▶ policy gradient
             (the model writes           (free,       (relative to      (up-weight the
              its own training data)     objective)    the group)        winners)
```

There is no human in that loop and no reward model to hack — the checker either runs the unit test or it does not.

> **The formatting trap is real.** `"the answer is 72"` is correct and scores **0** because the verifier expects `#### 72`. Every RLVR setup needs either a lenient extractor or an explicit *format reward* trained first — otherwise a large fraction of the gradient signal is punishing correct reasoning for cosmetic reasons.

| Reward source | Cost | Hackable? | Domain |
|---|---|---|---|
| ⭐ **Executed unit tests** | ~0 | very hard | code |
| ⭐ **Exact-match answer checker** | ~0 | format only | math, structured output |
| **Formal proof checker** (Lean) | ~0 | no | theorem proving |
| **Reward model** | one forward pass | **yes** — length, sycophancy, formatting | open-ended chat |
| **Human rating** | $$$ | biased, not hackable | taste, safety |

## 11.5 Regime 2.5 — reasoning traces, the bridge between SFT and RL

Long chain-of-thought data is formally SFT (Regime 2) but is *produced* like Regime 4 — sampled from a strong model, then filtered by a verifier.

**Sample record** (`open-r1/OpenR1-Math-220k`):

```json
{
  "problem": "Find the sum of all positive integers n such that n^2 + 12n - 2007 is a perfect square.",
  "solution": "...",                       ← the original human solution
  "answer": "3",
  "generations": ["<think>Let me set n^2 + 12n - 2007 = k^2. Completing the square,
                  (n+6)^2 - 2043 = k^2 ... wait, let me recheck ... </think>
                  The answer is \\boxed{3}"],
  "correctness_math_verify": [true],       ← the VERIFIER's label — this is the filter
  "is_reasoning_complete": [true]
}
```

The pipeline that produces it:

```
seed problems with known answers
   │  sample k=4..64 traces from a strong reasoning model (R1, o-series, Qwen3-235B)
   ▼
   │  VERIFY each trace's final answer against the gold answer
   ▼
   │  keep only correct traces  ← this is rejection sampling
   ▼
SFT dataset of (problem, verified trace)  ──▶ train a smaller model  (§11.6)
```

**Note the label is `correctness_math_verify`, not a quality score.** This is what makes reasoning data cheap to scale: correctness is machine-checkable, so a 220k-example dataset needs zero human annotation. It is also why reasoning models appeared so quickly once the recipe was known.

> **The counter-intuitive result:** `GAIR/LIMO` (817 examples) and `simplescaling/s1K` (1,000 examples) reach competitive reasoning benchmarks. At this regime, **1,000 verified traces can beat 100,000 unverified ones** — the strongest published evidence that post-training data quality dominates quantity.

## 11.6 Distillation — how the small models are actually made

Qwen3's published recipe is **strong-to-weak distillation**: the large models get the full multi-stage post-training pipeline, then the small models learn from *their outputs* rather than repeating it.

```
        expensive 4-stage post-training
Qwen3-235B ──────────────────────────▶ strong teacher
                                            │  generates data (or full logits)
                                            ▼
Qwen3-0.6B … 32B  ◀───── distillation ──────┘   far cheaper, and better than
                                                training each one from scratch
```

| Kind | What is transferred | Data shape |
|---|---|---|
| **Response distillation** | the teacher's sampled outputs | ordinary SFT pairs (Regime 2) |
| **Logit distillation** | the teacher's full next-token distribution | `[S, V]` soft targets — far more signal per token |
| **Rationale distillation** | reasoning traces (§11.5) | SFT on verified traces |

DeepSeek-V3 distilled reasoning behaviour from DeepSeek-R1. **Gemma 3 goes furthest and uses distillation for *pretraining* itself** — the small models learn from a teacher's token distribution rather than one-hot targets, which is why Gemma's small models punch above their token budget.

## 11.7 The four-stage post-training shape

Qwen3's pipeline, and the data each stage needs:

| Stage | Regime | Data type | Label |
|---|---|---|---|
| **1. Long-CoT cold start** | 2.5 | verified reasoning traces | verifier says the final answer is right |
| **2. Reasoning RL** | 4 | math/code prompts + checkers | executed reward |
| **3. Thinking-mode fusion** | 2 | mixed thinking / non-thinking dialogues | the response tokens |
| **4. General RL** | 3 + 4 | preference pairs + rule-based rewards | human/LLM comparison, safety rules |

**The pipeline is a loop, not a line:** the data for stage `k+1` is usually generated by the model produced at stage `k`. That is the structural difference between pretraining data (*collected* from the world) and post-training data (*manufactured* by models under verification).

---

# 12. Public Datasets

Frontier labs do not release their training corpora. What follows are (a) the open datasets that are known ingredients or direct analogues, and (b) the standard open substitutes for each regime. **Every link below was checked to resolve.**

> **Read this as "what you can actually build with", not "what Qwen3 was trained on."** No lab has published its mixture. `HuggingFaceFW/fineweb-edu` is the closest public analogue to a modern filtered web corpus; `allenai/dolma` and `mlfoundations/dclm-baseline-1.0` are the closest to a *documented, reproducible* full pipeline.

## 12.1 Regime 1 — Pretraining corpora

| Dataset | Size | Notes |
|---|---|---|
| ⭐ [`HuggingFaceFW/fineweb`](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | 15 T tokens | 96 CC snapshots, the full §2–§5 pipeline, fully documented |
| ⭐ [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 1.3 T | The §4.3 classifier applied at threshold 3 — the reference "quality-filtered web" |
| [`HuggingFaceFW/fineweb-2`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | ~3 T | **1,800+ languages** — the open analogue of Qwen3's 119-language expansion |
| ⭐ [`HuggingFaceFW/finepdfs`](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) | ~3 T | **PDF-extracted text, 1,700+ languages** — the open analogue of Qwen3's VLM-OCR corpus (§1.4) |
| [`mlfoundations/dclm-baseline-1.0`](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | ~4 T | DataComp-LM's winning filter; built to be a *controlled benchmark* for filtering |
| [`allenai/dolma`](https://huggingface.co/datasets/allenai/dolma) | 3 T | Multi-source (web, code, books, papers) with the full toolkit open-sourced |
| [`tiiuae/falcon-refinedweb`](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) | 600 B | The paper that showed filtered web alone can match curated corpora |
| [`togethercomputer/RedPajama-Data-V2`](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) | 30 T | 5 languages, ships **precomputed quality signals** so you can apply your own filters |
| [`uonlp/CulturaX`](https://huggingface.co/datasets/uonlp/CulturaX) | 6.3 T | 167 languages, cleaned and deduplicated |
| [`Zyphra/Zyda-2`](https://huggingface.co/datasets/Zyphra/Zyda-2) | 5 T | Cross-deduplicated blend of FineWeb-Edu + DCLM + others |
| [`nvidia/Nemotron-CC-v2`](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) | multi-T | CC reprocessed with synthetic rephrasing |
| [`allenai/c4`](https://huggingface.co/datasets/allenai/c4) | 156 B | The original heuristic-filtered corpus (§4.1) |
| [`common-pile/comma_v0.1_training_dataset`](https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset) | ~8 TB | **Openly licensed only** — the answer to the copyright question |

**Domain-specific:**

| Dataset | Domain | Notes |
|---|---|---|
| [`bigcode/the-stack-v2`](https://huggingface.co/datasets/bigcode/the-stack-v2) | code | 600+ languages, licence-filtered, opt-out respected — the standard code corpus |
| [`open-web-math/open-web-math`](https://huggingface.co/datasets/open-web-math/open-web-math) | math | 14.7 B tokens with LaTeX preserved (§2.2) |
| [`EleutherAI/proof-pile-2`](https://huggingface.co/datasets/EleutherAI/proof-pile-2) | math | 55 B — OpenWebMath + arXiv + formal proofs |
| [`HuggingFaceTB/cosmopedia`](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | synthetic | 25 B tokens of generated textbooks — the open analogue of §1.3 |
| [`HuggingFaceTB/smollm-corpus`](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | mixed | Cosmopedia-v2 + FineWeb-Edu-dedup + Python-Edu, ready to train |
| [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) | reference | 300+ languages, cleanly licensed, always upsampled (§8.1) |

## 12.2 Regime 2 — SFT datasets

| Dataset | Size | Notes |
|---|---|---|
| ⭐ [`allenai/tulu-3-sft-mixture`](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | **939 k** | The best-documented open SFT mixture; every source and its rationale published |
| ⭐ [`HuggingFaceTB/smoltalk`](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | ~1 M | Synthetic + public blend, ablated per-subset |
| [`teknium/OpenHermes-2.5`](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | 1 M | The most-used community SFT set |
| [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 200 k | Filtered multi-turn dialogue; the Zephyr recipe's SFT stage |
| [`HuggingFaceH4/no_robots`](https://huggingface.co/datasets/HuggingFaceH4/no_robots) | 10 k | **Entirely human-written** — small, expensive, high quality |
| [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | 15 k | Human-written by 5,000 employees; permissively licensed |
| [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) | 52 k | The original self-instruct set — historically important, now superseded |
| [`allenai/WildChat-1M`](https://huggingface.co/datasets/allenai/WildChat-1M) | 1 M | **Real user conversations** — the true prompt distribution, not a synthetic one |
| [`lmsys/lmsys-chat-1m`](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | 1 M | Real conversations from Chatbot Arena |

**Domain SFT:**

| Dataset | Size | Domain |
|---|---|---|
| [`nvidia/OpenMathInstruct-2`](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) | **22 M** | math — the largest open math SFT set |
| [`meta-math/MetaMathQA`](https://huggingface.co/datasets/meta-math/MetaMathQA) | 395 k | math, bootstrapped by question rewriting |
| [`microsoft/orca-math-word-problems-200k`](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | 200 k | grade-school math |
| [`ise-uiuc/Magicoder-OSS-Instruct-75K`](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) | 75 k | code, generated *from real OSS snippets* to avoid synthetic drift |
| [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | 60 k | tool/function calling |
| [`NousResearch/hermes-function-calling-v1`](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | ~10 k | tool use, multi-turn |

**Reasoning traces (Regime 2.5):**

| Dataset | Size | Notes |
|---|---|---|
| ⭐ [`open-r1/OpenR1-Math-220k`](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | **450 k rows** | DeepSeek-R1 traces, **verifier-filtered** — the §11.5 sample record |
| [`open-thoughts/OpenThoughts3-1.2M`](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) | 1.2 M | math + code + science traces |
| [`nvidia/Llama-Nemotron-Post-Training-Dataset`](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) | ~30 M | Reasoning-on/off pairs across domains |
| ⭐ [`GAIR/LIMO`](https://huggingface.co/datasets/GAIR/LIMO) | **817** | "Less Is More" — 817 examples, competitive benchmarks |
| ⭐ [`simplescaling/s1K`](https://huggingface.co/datasets/simplescaling/s1K) | **1,000** | 1k examples + budget forcing |

## 12.3 Regime 3 — Preference datasets

| Dataset | Size | Label shape |
|---|---|---|
| ⭐ [`nvidia/HelpSteer2`](https://huggingface.co/datasets/nvidia/HelpSteer2) | 21 k | **5 attributes × 0–4 levels** (§11.3) — permissive licence |
| [`nvidia/HelpSteer3`](https://huggingface.co/datasets/nvidia/HelpSteer3) | 40 k | Multilingual, adds preference *and* edit feedback |
| ⭐ [`HuggingFaceH4/ultrafeedback_binarized`](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | 187 k rows | `(prompt, chosen, rejected)` + scores — the standard DPO set |
| [`openbmb/UltraFeedback`](https://huggingface.co/datasets/openbmb/UltraFeedback) | 64 k | GPT-4 rated on 4 axes, before binarization |
| [`Anthropic/hh-rlhf`](https://huggingface.co/datasets/Anthropic/hh-rlhf) | 161 k | **Human** helpfulness + harmlessness comparisons — the original RLHF set |
| [`lmarena-ai/arena-human-preference-100k`](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-100k) | 100 k | **Real human votes** from Chatbot Arena, with ties |
| [`stanfordnlp/SHP`](https://huggingface.co/datasets/stanfordnlp/SHP) | 385 k | Reddit preferences inferred from vote counts |
| [`berkeley-nest/Nectar`](https://huggingface.co/datasets/berkeley-nest/Nectar) | 183 k | **7-way rankings**, not just pairs |
| [`Skywork/Skywork-Reward-Preference-80K-v0.2`](https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.2) | 80 k | Curated specifically for reward-model training |
| [`allenai/llama-3.1-tulu-3-8b-preference-mixture`](https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture) | 271 k | The documented Tülu 3 preference blend |
| [`argilla/distilabel-intel-orca-dpo-pairs`](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs) | 13 k | Cleaned Orca DPO pairs |
| [`openai/summarize_from_feedback`](https://huggingface.co/datasets/openai/summarize_from_feedback) | 179 k | The original "Learning to summarize from human feedback" comparisons |

## 12.4 Regime 4 — RL prompts and verifiers

| Dataset | Size | Verifier |
|---|---|---|
| ⭐ [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) | 8.5 k | `#### <answer>` exact match (§11.4) |
| [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | 500 | `\boxed{}` extraction + symbolic equivalence |
| [`agentica-org/DeepScaleR-Preview-Dataset`](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) | 40 k | math answer checking; built for RL, not SFT |
| [`PrimeIntellect/verifiable-math-problems`](https://huggingface.co/datasets/PrimeIntellect/verifiable-math-problems) | ~777 k | answer verification, aggregated from many sources |

## 12.5 Safety and alignment

| Dataset | Size | Label shape |
|---|---|---|
| [`PKU-Alignment/PKU-SafeRLHF`](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) | 83 k | **Dual labels** — helpfulness preference *and* harmlessness preference, separately |
| [`PKU-Alignment/BeaverTails`](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | 330 k | QA pairs tagged across **14 harm categories** |
| [`allenai/wildguardmix`](https://huggingface.co/datasets/allenai/wildguardmix) | 92 k | Prompt harm, response harm, and refusal labels — for guard models |

> **Note the shape difference in the first row.** PKU-SafeRLHF keeps helpfulness and harmlessness as *two separate preference labels* rather than one. Collapsing them forces annotators to trade the two off implicitly and inconsistently; keeping them separate lets you train two reward models and combine them with an explicit, tunable policy. This is the same lesson as HelpSteer2's multi-attribute scheme (§11.3).

## 12.6 Choosing a starting point

| If you are… | Start with |
|---|---|
| Pretraining a small model from scratch | [`HuggingFaceTB/smollm-corpus`](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) — mixed, filtered, sized to train |
| Reproducing a modern filtering pipeline | [`allenai/dolma`](https://huggingface.co/datasets/allenai/dolma) toolkit + [`HuggingFaceFW/fineweb`](https://huggingface.co/datasets/HuggingFaceFW/fineweb) |
| Benchmarking *your own* filter | [`mlfoundations/dclm-baseline-1.0`](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) — built as a controlled comparison |
| Instruction-tuning | [`allenai/tulu-3-sft-mixture`](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) |
| DPO / preference alignment | [`HuggingFaceH4/ultrafeedback_binarized`](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized), then [`nvidia/HelpSteer2`](https://huggingface.co/datasets/nvidia/HelpSteer2) for attribute control |
| Training a reward model | [`Skywork/Skywork-Reward-Preference-80K-v0.2`](https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.2) + [`nvidia/HelpSteer2`](https://huggingface.co/datasets/nvidia/HelpSteer2) |
| Reasoning distillation | [`open-r1/OpenR1-Math-220k`](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k), or [`GAIR/LIMO`](https://huggingface.co/datasets/GAIR/LIMO) to test the small-data hypothesis |
| RLVR | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) → [`agentica-org/DeepScaleR-Preview-Dataset`](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) |

> **Licences vary and matter.** `bigcode/the-stack-v2` respects opt-out and carries per-file licences; `Anthropic/hh-rlhf` is MIT; `nvidia/HelpSteer2` is CC-BY-4.0 and explicitly usable commercially; several SFT sets are **OpenAI-output-derived** and carry terms-of-use restrictions that make them unusable for a commercial competitor model. Check the card before building on any of them.

---

# 13. Infrastructure, Cost, and Failure Modes

## 13.1 Formats along the pipeline

| Stage | Format | Why |
|---|---|---|
| Raw crawl | WARC (`.gz`) | What Common Crawl publishes |
| Extracted text | **Parquet** / `jsonl.zst` | Columnar, splittable, compresses well; predicate pushdown for filtering |
| Filter/dedup intermediates | Parquet + signature columns | The MinHash shuffle is a distributed group-by |
| Tokenized | ⭐ **flat `.bin` + `.idx`** (§7.3) | Memory-mappable; zero parse cost at train time |
| Alternative | WebDataset (`.tar`) / Mosaic MDS | Streaming from object storage without a shared filesystem |

## 13.2 The scale of the pipeline itself

```
Corpus:        36 T tokens ≈ 144 TB tokenized ≈ 1-10 PB of raw crawl behind it
Compute:       10,000s of CPU cores for weeks  (extraction, filtering, dedup)
               + GPU hours for quality classifiers and any VLM/LLM annotation (§1.4, §4.4)
Cost:          typically single-digit millions of dollars
               — a few percent of the training run it feeds
Wall clock:    weeks to months, and it is usually the LONG POLE before a run can start
```

> Data preparation is roughly **5% of the budget and 50% of the calendar** of a frontier training project. It is also the part that most reliably differentiates two models trained with the same architecture and compute.

## 13.3 Failure modes that have actually happened

| Failure | Consequence |
|---|---|
| **`uint16` overflow** (§0.4) | Token ids > 65,535 silently wrap. The model trains on corrupted text and simply underperforms, with no error anywhere |
| **Tokenizer/corpus mismatch** | Corpus tokenized with one vocabulary, model configured with another — garbage that looks like slow convergence |
| **Dataloader resume bug** (§9.6) | Restart replays the same shard; the model over-trains on a slice and the loss curve looks fine |
| **Contaminated eval** (§6.3) | Benchmarks improve, real capability does not; discovered after publication |
| **Mixture drift** (§8.5) | A shard fails to mount; the realized mixture silently diverges from the configured one for 200 B tokens |
| **Over-filtering** (§4.6) | A too-aggressive threshold removes a whole register or language; visible only in evaluations nobody ran |
| **Global dedup** (§5.5) | Quality *decreases* while every dashboard says "duplicates removed: 90%" |

The pattern in every row: **data bugs do not crash, they degrade.** A model trained on subtly corrupted data still trains, still produces a loss curve, and still finishes — it is just worse than it should have been, and you find out weeks later. This is why the verification step in §15.2, the realized-mixture log in §8.5, and the proxy-model ablations in §8.4 exist.

---

# 14. Case Studies Side by Side

| | **Qwen3** | **DeepSeek-V3** | **Llama 3** | **Gemma 3** |
|---|---|---|---|---|
| **Tokens** | 36 T | 14.8 T | ~15 T (15.6 T for 405B) | 14 T (27B) |
| **Languages** | 119 | En/Zh + expanded | 176 identified, 8 supported | 140+ |
| **Vocabulary** | 151,669 BBPE | 128 K BBPE | 128,256 (100 K tiktoken + 28 K) | 262,144 SentencePiece |
| **Published mix** | not disclosed as % | math/code enriched | **50 / 25 / 17 / 8** (general / math / code / multilingual) | not disclosed |
| **Signature data move** | **VLM OCR of PDFs** + trillions of synthetic tokens + **instance-level** mixing over 30 T annotated tokens | FIM (PSM @ 0.1); packing **without** cross-doc masking | Custom HTML parser; 3-level dedup; annealing | Distillation-based training |
| **Quality filtering** | multi-dimensional annotator (educational value, field, domain, safety) | refined pipeline, redundancy minimized | fastText + DistilRoBERTa trained on Llama-2 labels | not disclosed |
| **Stages** | S1 >30 T @4 K → S2 ~5 T @4 K → S3 100s of B @32 K | main run → 32 K (1 K steps) → 128 K (1 K steps) | main run → 6-stage extension over 800 B → anneal | — |
| **Context** | 32 K trained, 128 K via YaRN + DCA | 128 K via YaRN | 128 K | 128 K |
| **Batch schedule** | scaling-law predicted per stage | ~62 M tokens/step held constant | 4 M → 8 M → 16 M tokens | — |

**What the row-by-row comparison shows:**

- **Everyone converged on the same skeleton** — crawl → extract → filter → dedup → decontaminate → tokenize → stage → pack. The differences are in *degree* and in one or two signature moves each.
- **The differentiators are increasingly about supply, not cleaning.** Qwen3's lead in token count comes from VLM-OCR and synthetic generation — new *sources* — not from a better filter.
- **Multi-stage curricula are now universal.** No frontier model trains on a single fixed mixture any more.
- **Long context is always a bolt-on stage**, never the main run.

---

# 15. The Assembled Pipeline

## 15.1 End to end

```python
# PSEUDOCODE — extract_text / identify_language / quality_score / unsafe /
# has_pii / LANG_THRESHOLD are the stand-ins listed in §15.3.
def prepare_corpus(raw_docs, eval_texts, tokenizer, cfg: DataConfig):
    # ---- §2-§4: extract, identify, filter ------------------------------------
    docs = []
    for raw in raw_docs:
        text = extract_text(raw)                       # §2  HTML/PDF -> text
        lang, conf = identify_language(text)           # §3
        if conf < LANG_THRESHOLD[lang]:      continue  # §3.2 per-language threshold
        if gopher_filters(text, lang):       continue  # §4.2 heuristics
        if quality_score(text) < 3:          continue  # §4.3 LLM-judge distillate
        if unsafe(text) or has_pii(text):    continue  # §4.5
        docs.append(text)

    # ---- §5: deduplicate (PER SNAPSHOT, not globally — §5.5) ------------------
    keep, _ = dedup(docs, cfg.minhash_perm, cfg.minhash_bands, cfg.shingle_n)
    docs = [docs[i] for i in keep]

    # ---- §6: decontaminate ----------------------------------------------------
    contam = build_contam_index(eval_texts, cfg.contam_ngram)
    docs = [d for d in docs if not is_contaminated(d, contam, cfg.contam_ngram)]

    # ---- §7: tokenize ONCE, to a memory-mappable binary ------------------------
    w = ShardWriter("corpus", cfg.dtype)
    for d in docs:
        w.add(tokenizer.encode(d))                     # `01` §1.6.2
    w.close("corpus")
    return ShardReader("corpus", cfg.dtype)


def training_batches(reader, cfg: DataConfig, batch_size, seed=0, start_step=0):
    # ---- §8: mix -> §9: pack -> §9.7: batch ------------------------------------
    rng = np.random.default_rng(seed)
    docs = [fim_transform(reader.doc(i), rng, cfg.fim_rate)      # §9.4
            for i in range(len(reader))]
    packed, doc_ids = pack_documents(docs, cfg.seq_len, cfg.eos_id)   # §9.3
    loader = TokenLoader(packed, batch_size, seed, start_step)        # §9.7
    for inputs, targets, step in loader:
        yield inputs, targets, step               # -> model.forward   `01` §15.1
```

The join to the other file is exact: `inputs` is the `ids` argument of `Transformer.forward` in `01` §15.1, and `document_mask` (§9.3) is added to `scores` in the same place as the causal mask in `01` §4.5.2.

## 15.2 The verification run

`verify_03_pipeline.py` (a companion script, not included in this repo) executes every `python` block in this document and checks the claimed numbers.

```
$ python3 verify_03_pipeline.py

  ok  §0.2   Llama-3-405B = 3.79e+25 FLOPs; Chinchilla says 8.1T, actual 15.6T -> 1.9x over-trained
  ok  §0.2   Qwen3-235B-A22B (22B active) = 4.75e+24 FLOPs — 8x less than Llama-3-405B
  ok  §0.4   uint16 max 65,535 < Qwen3 vocab 151,936 -> uint32 is mandatory (144 TB for 36T)
  ok  §4.2   all 8 filter cases match the printed table (clean prose KEPT, 7 rules fire distinctly)
  ok  §5.3   LSH(b=16,r=8) threshold=0.707; S-curve matches: J=0.5->0.061, J=0.75->0.815, J=0.9->1.0
  ok  §5.4   dedup kept [0, 2, 3], dropped [1] (the near-duplicate)
  ok  §5.4   MinHash estimator: true Jaccard=0.750, 128-perm estimate=0.742
  ok  §5.4   unrelated docs: true J=0.000, estimate=0.000 (no false collision)
  ok  §6.2   13-gram index flags the embedded benchmark item, passes clean text
  ok  §7.3   shards: 50 docs, 5,104 tokens, memmap roundtrip exact, 4.0 bytes/token
  ok  §7.3   4 bytes/token x 36T = 144 TB for the tokenized Qwen3 corpus
  ok  §8.5   realized mixture web=0.502, math_reasoning=0.249, code=0.170, multilingual=0.079
  ok  §9.3   packed [1, 2, 3, 0, 4, 5, 0, 6, 7] / doc_ids [0, 0, 0, 0, 1, 1, 1, 2, 2] / pos [0, 1, 2, 3, 0, 1, 2, 0, 1]
  ok  §9.3   document mask matches the printed diagram (no cross-document attention)
  ok  §9.3   row 4 (start of doc 2) is blocked from all of doc 1 ✓
  ok  §9.4   PSM: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -> [1, 0, 1, 2, 2, 5, 6, 7, 8, 9, 3, 3, 4]
  ok  §9.4   configured FIM rate 0.1 -> realized 0.087 over 400 docs
  ok  §9.5   DeepSeek-V3: 32K x 1920 == 128K x 480 == 62.9M tokens/step (held constant)
  ok  §9.8   Llama 3: 15.6T tokens / 16M per step = 975,000 optimizer steps
  ok  §9.8   16M tokens / 8192 seq len = 1,953 sequences per step
  ok  §9.7   loader -> inputs (4, 16), targets shifted by one ✓, step=1
  ok  §9.7   resume from state_dict {'step': 1} reproduces the exact next batch ✓
  ok  §8.2   36T token budget at <=4 epochs needs >=9T unique tokens — consistent with §1.6
  ok  §11.1  pretraining: labels = input shifted by 1, 4/4 supervised
  ok  §11.2  SFT: prompt masked, 4/7 supervised, EOS IS a target
  ok  §11.2  multi-turn: 6/10 supervised — BOTH assistant turns + both EOS
  ok  §11.3  preference: two sequences, each masked like SFT, sharing one prompt
  ok  §11.3  DPO: prefers-chosen 0.6210 < tie -log(0.5)=0.6931 < prefers-rejected 0.9432
  ok  §11.3  Bradley-Terry is antisymmetric in the pair, as a ranking loss must be
  ok  §11.3  HelpSteer2 5 attrs x 0-4 -> scalar: good=8.30 bad=2.15 (verbosity weighted NEGATIVE)
  ok  §11.4  RLVR rewards [1.0, 0.0, 1.0, 0.0, 1.0] — note index 3 is CORRECT but mis-formatted -> 0.0
  ok  §11.4  GRPO advantages [0.816, -1.224, 0.816, -1.224, 0.816] — zero-mean, no value network
ALL PIPELINE CODE VERIFIED — every claimed number reproduced
```

## 15.3 What this pipeline is missing

| Missing | Why it exists in production | Covered in |
|---|---|---|
| **Distributed execution** | Every stage here is single-process; real pipelines are Spark/Ray/Slurm over thousands of cores | §13.2 |
| **Bloom filters** | The contamination `set` and dedup index must fit 10⁹ entries in bounded memory | §6.2 |
| **Streaming** | `pack_documents` materializes the corpus; production packs on the fly from memmapped shards | §7.2 |
| **Real extractors** | `extract_text` stands in for trafilatura/resiliparse/a custom parser | §2.1 |
| **The quality classifier** | `quality_score` stands in for a trained model (§4.3) — the training of which is itself a project | §4.3 |
| **Multi-node sharding** | `TokenLoader` serves one rank; real loaders shard disjointly across data-parallel ranks | §9.6 |

The *logic* of each stage above is complete and runnable. Everything in that table changes the scale at which it runs, not what it computes.

---

# 16. Reference

**Corpora and pipelines**
- ⭐ **The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale** — Penedo et al., 2024 (2406.17557) — the most detailed public account of filtering and dedup ablations; source of §5.5 and §4.3
- **RefinedWeb** — Penedo et al., 2023 (2306.01116) — web-only data can match curated corpora
- **The Pile** — Gao et al., 2020 (2101.00027) — the 22-source curated baseline
- **CCNet** — Wenzek et al., 2019 (1911.00359) — perplexity filtering and per-language pipelines
- **C4 / T5** — Raffel et al., 2019 (1910.10683) — the original heuristic filter set
- **Dolma** — Soldaini et al., 2024 (2402.00159) — 3 T tokens with the full toolkit open-sourced
- **DataComp-LM (DCLM)** — Li et al., 2024 (2406.11794) — filtering as a controlled benchmark

**Filtering and quality**
- ⭐ **Scaling Language Models: Methods, Analysis & Insights (Gopher)** — Rae et al., 2021 (2112.11446) — §4.1's rule set
- **Textbooks Are All You Need (phi-1)** — Gunasekar et al., 2023 (2306.11644) — synthetic textbook data
- **Quality at a Glance** — Kreutzer et al., 2021 (2103.12028) — how bad multilingual web data actually is

**Deduplication**
- ⭐ **Deduplicating Training Data Makes Language Models Better** — Lee et al., 2021 (2107.06499)
- **Extracting Training Data from Large Language Models** — Carlini et al., 2020 (2012.07805) — why duplicates cause memorization
- **SemDeDup** — Abbas et al., 2023 (2303.09540) — embedding-space deduplication
- **Broder, 1997** — the original MinHash; **Indyk & Motwani, 1998** — LSH

**Scaling, mixing, repetition**
- ⭐ **Training Compute-Optimal LLMs (Chinchilla)** — Hoffmann et al., 2022 (2203.15556) — the `D ≈ 20N` rule of §0.2
- ⭐ **Scaling Data-Constrained Language Models** — Muennighoff et al., 2023 (2305.16264) — the 4-epoch result of §8.2
- **DoReMi** — Xie et al., 2023 (2305.10429) — learning domain weights with a proxy model
- **Data Mixing Laws** — Ye et al., 2024 (2403.16952)

**Packing and long context**
- **Efficient Sequence Packing without Cross-contamination** — Krell et al., 2021 (2107.02027) — §9.2
- **Fewer Truncations Improve Language Modeling** — Ding et al., 2024 (2404.10830) — the packing method DeepSeek-V3 cites
- **Efficient Training of Language Models to Fill in the Middle** — Bavarian et al., 2022 (2207.14255) — PSM, §9.4
- **YaRN** — Peng et al., 2023 (2309.00071) — the context extension used by both Qwen3 and DeepSeek-V3

**Primary sources for §14**
- ⭐ **Qwen3 Technical Report** — 2025 (2505.09388) — §3.1 Pre-training Data, §3.2 Pre-training Stage
- ⭐ **DeepSeek-V3 Technical Report** — 2024 (2412.19437) — §4.1 Data Construction
- ⭐ **The Llama 3 Herd of Models** — Grattafiori et al., 2024 (2407.21783) — §3 Pre-Training, the most detailed data-curation section any frontier lab has published
- **Gemma 3 Technical Report** — 2025 (2503.19786)

**Objectives and post-training**
- ⭐ **Training language models to follow instructions with human feedback (InstructGPT)** — Ouyang et al., 2022 (2203.02155) — the SFT → RM → PPO pipeline
- ⭐ **Direct Preference Optimization** — Rafailov et al., 2023 (2305.18290) — §11.3's `dpo_loss`
- **DeepSeekMath (GRPO)** — Shao et al., 2024 (2402.03300) — §11.4's group-relative advantages
- **DeepSeek-R1** — 2025 (2501.12948) — RLVR at scale, and the distillation results
- **Tülu 3** — Lambert et al., 2024 (2411.15124) — the most complete open post-training recipe, with every dataset published
- **HelpSteer2** — Wang et al., 2024 (2406.08673) — the 5-attribute 0–4 scheme of §11.3
- **LIMA: Less Is More for Alignment** — Zhou et al., 2023 (2305.11206) — 1,000 examples
- **s1: Simple test-time scaling** — Muennighoff et al., 2025 (2501.19393) — `s1K`
- **Self-Instruct** — Wang et al., 2022 (2212.10560) — synthetic instruction generation

**Code to read**
- ⭐ `huggingface/datatrove` — the FineWeb pipeline, end to end
- `allenai/dolma` — extraction, tagging, dedup, mixing toolkit
- `mlfoundations/dclm` — the filtering benchmark harness
- `NVIDIA/NeMo-Curator` — GPU-accelerated dedup and filtering
- `karpathy/nanoGPT` → `data/openwebtext/prepare.py` — the smallest complete tokenize-to-`.bin` example

---

**Next:** `01-transformer-architecture.md` §15.1 — the model these batches are fed to.
