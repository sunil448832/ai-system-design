---
title: "08 — ML System Design: Industry Case Studies"
subtitle: "Eight systems, designed end to end, with the corner questions interviewers actually ask"
companions:
  - "00-system-design-curriculum.md — §1 framework, §2B data pipelines"
  - "07-agentic-system-design.md — agent-shaped systems (the other half of the split)"
  - "06-agent-architecture.md — agent components; §7 is the RAG consumer view for Case Study 2"
  - "01-transformer-architecture.md — foundations for Case Study 3"
  - "02-modern-transformer-architectures.md — foundations for Case Study 3"
  - "05-inference-serving.md — foundations for Case Study 3"
  - "03-training-data-pipeline.md — foundations for Case Study 6 (fine-tuning data)"
  - "04-training-methods.md — foundations for Case Study 6 (SFT, PEFT, preference tuning, RL)"
---

# How to Use This File

This is the **ML system design** companion to `07-agentic-system-design.md`. The split:

| This file — **ML systems** | `07-agentic-system-design.md` — **agent systems** |
|---|---|
| Data → features → model → serving | Task → tools → control loop → guardrails |
| Ranking, retrieval, classification, training, serving | Support agents, memory, coding agents, eval platforms |
| Scored on: metrics, skew, feedback loops | Scored on: safety, reliability, evaluation |

**For a Google ML loop, this file matters more.** The reported L5 ML-infra design round probed exactly this material — model serving, feature pipelines, A/B testing, monitoring and retraining triggers, batch vs real-time inference.

## The eight case studies

Ordered so each adds a new hard problem rather than repeating the last:

| # | Case study | Industry examples | The new hard problem |
|---|---|---|---|
| **1** ✅ | **Recommendation system** | YouTube, Netflix, TikTok, Instagram | Two-stage retrieval, feedback loops, multi-objective |
| **2** ✅ | **Modern RAG over enterprise knowledge** | Glean, Dropbox Dash, Vertex AI Search | Retrieval quality + permissions + index migration |
| **3** ✅ | **LLM inference & serving platform** | vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo | Hardware-level performance, 1B → 2T params |
| **4** ✅ | **NER / information extraction at scale** | Google KG, Bloomberg | Sequence labeling vs LLM extraction; schema evolution; span-level eval |
| **5** ✅ | **Fraud detection** | Stripe Radar, PayPal, Visa | Extreme imbalance, delayed labels, adversarial drift |
| **6** ✅ | **LLM fine-tuning platform** | Together, Predibase, internal platforms | When to tune vs prompt vs RAG; PEFT at scale |
| **7** ✅ | **Content moderation (multimodal)** | Meta, YouTube | Precision/recall at policy thresholds, human review |
| **8** ✅ | **Feature store / ML platform** | Uber Michelangelo, Tecton, Feast | Offline/online parity, point-in-time correctness |

> **Case Study 1 is the highest-frequency ML design question in industry.** The two-stage pattern in §1.4–1.5 also answers "design search ranking," "design ads CTR," and "design the Meta feed" — learn it once, reuse it four times.
>
> **Case Study 3 is the one that matters most for Nvidia**, and it's the only design here where the answers come from hardware physics rather than software architecture.

## Canonical structure — every case study has all of these

| # | Component | What it answers |
|---|---|---|
| 1 | **Requirements** | What are we building, what's out of scope, what are the numeric targets? |
| 2 | **Estimation** | Scale, storage, cost — *and what the numbers change about the design* |
| 3 | **Architecture** + **Recommended stack** | The diagram walked end to end, then concrete technology picks — default, alternative, and what flips you |
| 4–7 | **Deep dives A–D** | The 3–4 genuinely hard sub-problems, each with a decision and a defense |
| 8 | **Failure modes** | What breaks, why, and the mitigation — as a table |
| 9 | **Evaluation** | Offline → shadow → online → CI regression |
| 10 | **Cost & latency** | The budget broken down, levers ranked by impact |
| 11 | **Corner questions** | The follow-ups that decide the round, with model answers |
| 12 | **Mid vs Senior** | What separates the two answers, per dimension |
| 13 | **References** | Papers, blogs, and repos **specific to this design** |

Some case studies carry an extra section: **§1.14 Practical grounding** (appended after Case Study 1's references) and **§2.8 When *not* to use RAG** (inserted after Case Study 2's deep dives). The insertion shifts Case Study 2's later components by one — its failure modes are §2.9 and its references §2.14.

**The corner questions are the point** — that's where interviews are won and lost.

> **Practising with this:** cover everything from §X.3 down, read only the requirements, and design it yourself on paper in 45 minutes. Then diff. The gap between your version and the deep dives is your actual study list.

---

# The ML System Design Framework (compressed)

From `00-system-design-curriculum.md` §1.1. Memorize the skeleton; spend your thinking budget on trade-offs.

```
1. Clarify       Business goal → ML objective → success metrics
2. Metrics       Offline (AUC/NDCG/F1) + Online (CTR/watch time/revenue) + guardrails
3. Data          Sources, labels, volume, freshness, leakage, bias, cold start
4. Features      Feature store; offline/online parity; training-serving skew
5. Model         Baseline first (heuristic → LR/GBDT → deep). Justify complexity.
6. Training      Distributed strategy, cadence, reproducibility, versioning
7. Evaluation    Offline → shadow → A/B → guardrail metrics → rollout
8. Serving       Latency budget, batching, caching, precompute vs realtime, quantization
9. Monitoring    Drift (data/concept), degradation, feedback loops
10. Iteration    Retraining triggers, canary, rollback
```

### The four principles that generate most good answers

1. **Translate the business goal into an ML objective, then name the gap.** "Increase engagement" isn't an objective. Watch time → `P(watch > 30s)` → and the proxy gap is clickbait. Saying that unprompted is the strongest opening move available.
2. **Always start with a non-ML baseline.** "Most popular" is a shockingly strong recommender. Stating it shows engineering maturity, not naivety.
3. **Training-serving skew is the default failure.** One feature definition, materialized to both stores; log the features *actually served*.
4. **Every deployed model creates a feedback loop.** Name it, then propose exploration or bias correction.

---

# Case Study 1 — Recommendation System

> *"Design the recommendation system for a video platform — the home feed."*

**The highest-frequency ML design question in industry.** The two-stage pattern here also answers "design search ranking," "design ads CTR prediction," and "design the Meta feed." Learn it once.

## 1.1 Requirements

**Ask these four:**
- "Which surface — home feed, related-items, or notifications?" *(Completely different candidate pools and latency budgets.)*
- "What's the business objective?" *(Engagement, retention, or revenue — they conflict.)*
- "Catalog size and user base?"
- "How fresh must recommendations be — do they react within a session?"

**Assume:** home feed, 200M MAU / 50M DAU, ~500M videos, objective is **long-term retention** proxied by watch time, in-session reactivity required.

**Non-functional — quantified:**
- **p95 feed latency < 200 ms** end to end
- Recommendations react to in-session behavior within ~1 minute
- **New items discoverable within an hour of upload** (creator-supply health)
- Diversity and creator-fairness guardrails — not optional at this scale

**Descope out loud:** the video player, CDN, search, notifications.

> **The framing move that opens strong.** The business goal is retention, but retention is measured in months and you need a training signal today. So you optimize a *proxy* — watch time — and **the entire design is shaped by the gap between the proxy and the goal.** Optimize clicks and you get clickbait; optimize raw watch time and you get autoplay traps. Naming that tension in the first two minutes tells the interviewer you've actually shipped a recommender.

## 1.2 Estimation

```
50M DAU × 20 feed requests/day = 1B requests/day
1B / 100k s = 10k QPS avg → ~30k QPS peak
```
**This one *does* have a throughput problem** — unlike RAG (§2.2). 30k QPS against a model that must score hundreds of candidates each is the binding constraint, and it's why the two-stage architecture exists.

**The funnel — the number that forces the design:**
```
500M videos  →  retrieval  →  ~1,000 candidates  →  ranking  →  ~20 shown
```
You cannot run a deep ranker over 500M items in 200 ms. **Scoring 500M items at even 1 µs each is 500 seconds.** That arithmetic *is* the argument for two stages — say it rather than asserting the pattern.

**Embedding storage:**
```
500M items × 128 dims × 4 B (fp32) = 256 GB   → too large to serve from RAM cheaply
                        × 1 B (int8) =  64 GB   → fits a serving host
```

**Training data volume — the part people underestimate:**
```
50M DAU × 100 impressions/day = 5B impressions/day
× ~200 B logged features       = ~1 TB/day of training data
```
That volume drives the feature-logging design (§1.7) and makes full retraining expensive — which is why you separate a slowly-retrained ranker from fast-updating counters.

**Latency budget for 200 ms:**
```
Feature fetch (user + context)      20 ms
Retrieval (ANN, multi-source)       25 ms   ← parallel across sources
Feature fetch (1k candidates)       30 ms   ← batched
Ranking (1k candidates)             60 ms
Re-ranking / diversity              10 ms
Overhead + network                  40 ms
                              ───────────
                                   185 ms
```

## 1.3 Architecture

```
                              REQUEST PATH (<200ms)
  ┌──────────────────────────────────────────────────────────────┐
  │  Request                                                     │
  │     ▼                                                        │
  │  User & context features ◀──── ONLINE FEATURE STORE (Redis)  │
  │     ▼                                                        │
  │  ┌─────────────── CANDIDATE GENERATION (parallel) ────────┐  │
  │  │ Two-tower ANN │ Trending │ Subscriptions │ Sequence    │  │
  │  │               │          │ / follows     │ (recent→related)│
  │  │               │          │               │ + EXPLORATION│  │
  │  └───────────────────────┬────────────────────────────────┘  │
  │                     ~1,000 candidates (dedup, union)         │
  │                          ▼                                   │
  │  Candidate features ◀──── ONLINE FEATURE STORE (batched)     │
  │                          ▼                                   │
  │  ┌──────────────── RANKING (multi-task) ─────────────────┐   │
  │  │ P(click) P(watch>30s) P(complete) P(like) P(skip)     │   │
  │  │            → VALUE MODEL (weighted)                   │   │
  │  └──────────────────────┬────────────────────────────────┘   │
  │                          ▼                                   │
  │  Re-rank: diversity · freshness · creator fairness · dedup   │
  │                          ▼                                   │
  │                    ~20 items → user                          │
  └──────────────────────────┬───────────────────────────────────┘
                             │  LOG FEATURES AS SERVED  ← §1.7
                             ▼
   Kafka ──▶ Flink (real-time counters) ──▶ Online store
        └──▶ Warehouse (offline store) ──▶ Training pipelines
                                              ├─ Two-tower (daily)
                                              ├─ Ranker (hourly/daily)
                                              └─ Item embeddings (nightly)
```

### Recommended stack

| Component | ⭐ Default | Alternative | Flip when |
|---|---|---|---|
| **Retrieval model** | ⭐ **Two-tower** (user tower online, item tower precomputed) | Item-to-item CF · graph embeddings (PinSage) | Graph methods when the social/interaction graph is the dominant signal; CF is a strong cheap baseline |
| **ANN index** | ⭐ **ScaNN** or **FAISS (IVF-PQ)** | HNSW · Qdrant/Vespa | HNSW for best recall/latency when memory is free; **IVF-PQ when 500M vectors must fit in RAM**; a managed store if you don't want to run the index |
| **Ranking model** | ⭐ **DCNv2** or **DLRM**-style | **GBDT (XGBoost/LightGBM)** · transformer sequence model | **Start with GBDT** — it's a genuinely strong baseline and trains in minutes; go deep when you need embeddings for sparse IDs and have the traffic to justify it |
| **Sequence modeling** | Transformer over recent history (SASRec/BERT4Rec-style) | GRU / attention pooling | Worth it when in-session intent matters, which on a video feed it does |
| **Embedding tables** | ⭐ **TorchRec** (sharded, model-parallel) | Parameter server | Sparse ID embeddings can reach hundreds of GB — they *must* be sharded, and this is the part people forget |
| **Feature store** | ⭐ **Feast** (OSS) or **Tecton** | Build in-house | Buy unless offline/online parity is genuinely special. **Point-in-time correctness is the hard part** — don't reimplement it |
| **Online store** | ⭐ **Redis** | DynamoDB · Cassandra | Dynamo/Cassandra past what Redis memory can hold; Redis for the sub-10 ms path |
| **Offline store** | **Parquet on S3** + **Iceberg** | BigQuery / Snowflake | Iceberg time travel = reproducible training snapshots |
| **Streaming** | ⭐ **Kafka** + **Flink** | Spark Streaming | Flink for true low-latency counters and windowing; Spark if micro-batch (~minutes) is acceptable |
| **Training** | **PyTorch** + **Ray Train** | Horovod · TorchElastic | Ray for elastic multi-node with straggler tolerance |
| **Experimentation** | In-house A/B with sticky bucketing | Statsig / Eppo | Buy early, build once experiment volume and interference handling justify it |
| **Serving** | **TorchServe** / **Triton** | Ray Serve | Triton for GPU ranking with dynamic batching |
| **Monitoring** | Prometheus + **drift detection on feature distributions** | Arize / WhyLabs | Managed when you want per-segment drift analysis out of the box |

## 1.4 Deep Dive A — Candidate generation

**Never a single source.** Production recommenders union several retrievers, each covering a different failure mode:

| Source | Covers | Typical share |
|---|---|---|
| **Two-tower ANN** | Personalized semantic match | ~40% |
| **Sequence / related-item** | In-session intent ("you just watched X") | ~20% |
| **Subscriptions / follows** | Explicit user intent — *don't let the model override this* | ~15% |
| **Trending / popular** | Cold-start users, cultural moments, the non-ML baseline | ~15% |
| **Exploration** | New items and uncertainty reduction (§1.6) | ~10% |

**The two-tower model, and the trap.** User tower runs at request time on live features; item tower is precomputed nightly into the ANN index. Scoring is a dot product, which is what makes ANN possible.

> **Why the towers can't interact — the question interviewers love.** A cross-attention model that lets user and item features interact is *far* more accurate. But it requires a forward pass per (user, item) pair, so you cannot precompute item vectors and cannot use an ANN index — you'd score 500M items per request. **The dot product is a deliberate accuracy sacrifice that buys sublinear retrieval.** That's exactly why the expensive interaction model goes in stage two, over 1,000 candidates instead of 500M.

**Training the two-tower properly:**
- **In-batch negatives** — other items in the batch serve as negatives, which is cheap and effective
- **Sampled softmax with log-Q correction** — in-batch sampling over-represents popular items, so without correcting for sampling probability you learn a popularity model. This is the single most-cited practical detail in two-tower training (Yi et al., RecSys 2019).
- **Hard negatives** — items retrieved but not engaged with; teaches finer distinctions than random negatives

**Index freshness.** Nightly rebuild is fine for the bulk, but **new items can't wait a day** — a separate fresh-item path indexes uploads within minutes using content-based embeddings (§1.6).

## 1.5 Deep Dive B — Ranking and multi-objective optimization

Stage two scores ~1,000 candidates with everything: user features, item features, context (time, device, session), and cross features.

**Multi-task, not single-task.** A single `P(click)` head produces clickbait. Predict several outcomes and combine:

```
value = w₁·P(click) + w₂·P(watch>30s) + w₃·P(complete)
      + w₄·P(like) + w₅·P(share) − w₆·P(skip) − w₇·P(report)
```

**Two things to say about those weights:**
1. **They are a product decision, not a learned parameter** — tuned via A/B against long-term retention, because there's no offline label for "what the business wants."
2. **The negative terms matter most.** Subtracting `P(skip)` and `P(report)` is what prevents engagement-maximizing degeneracy. A design that only adds positive terms will drift toward clickbait, and the interviewer is often checking for exactly this.

**Position bias — include it and know the trick.** Items shown higher get clicked more regardless of quality, so training naively teaches the model to predict *position*, not relevance. Standard fix: **include position as a feature during training, then fix it to a constant at serving time**. Cheap, effective, and a strong detail to volunteer.

> **This is measurable, and §1.14 measures it on real data** — mean dwell falls from 14.8 s at slot 0 to 5.5 s at slot 99, a 2.7× gap that has nothing to do with content quality. Quoting a number you computed yourself is far stronger than asserting that position bias exists.

**Calibration.** For pure ranking, monotonic scores suffice. But the moment predictions feed a downstream decision — ad auctions, thresholds, budget pacing — you need calibrated probabilities: Platt scaling or isotonic regression, monitored over time since calibration drifts.

**Feature freshness tiers:**
| Tier | Latency | Examples |
|---|---|---|
| Real-time (Flink) | seconds | Session history, in-session counters |
| Near-real-time | minutes | Item CTR, trending velocity |
| Batch | daily | User long-term preferences, item embeddings |

## 1.6 Deep Dive C — Cold start, both kinds

**New users** — you have almost nothing:
- Popularity and demographic/geographic priors as the immediate fallback
- Onboarding signals (picked interests) if the product collects them
- **Aggressive exploration early** — the information value of a signal is highest when you know nothing
- Rapid session-level adaptation: after 2–3 interactions you have real signal, so the model must incorporate in-session behavior fast

**New items — the harder and more consequential problem.** A new upload has no interaction history, so collaborative signals are empty. If you get this wrong, creators leave and supply dies.
- **Content-based embeddings** from title, description, transcript, and thumbnail — a multimodal encoder (CLIP-style) puts new items directly into the same embedding space as established ones.
- **Creator priors** — a new video from a creator with strong history inherits a prior
- **A guaranteed exploration budget** — reserve feed slots for under-explored items rather than hoping they surface. Without a *forced* budget, a pure exploit ranker will never show them, because they have no evidence, and they never accumulate evidence because they're never shown.

**Exploration strategies:** ε-greedy (simple, wasteful), **Thompson sampling / UCB** (principled, uses uncertainty), and dedicated exploration slots (practical, easy to reason about and to cap). Real systems use exploration slots plus bandits within them.

## 1.7 Deep Dive D — Feedback loops and training data

**The defining pathology of recommenders: the model trains on data it generated.** Items shown get engagement; items never shown accumulate no evidence and are never shown again. Rich get richer, and the system converges on a shrinking slice of the catalog.

**Four fixes, and you should name more than one:**
1. **A randomized holdout** — a small traffic slice (0.1–1%) served *unpersonalized or randomized*, never touched by the model. This is the only truly unbiased training and evaluation data you will ever have. Expensive, and worth it.
2. **Inverse propensity weighting** — weight training examples by 1/P(shown), correcting for the model's own selection
3. **Exploration slots** (§1.6) — structurally guarantee coverage
4. **Position/presentation bias correction** (§1.5)

**Training-serving skew — the most common silent failure.** The fix is not "be careful": **log the feature values actually used at serving time**, and train on those logs. Recomputing features from raw data at training time guarantees drift, because the batch pipeline and the online path will diverge in ways nobody notices for months.

**Delayed labels.** Someone may watch tomorrow, or complete a video days later. So: define an attribution window, accept that recent data is incomplete, and either wait for the window to close (stale but correct) or model the delay. Never treat "no engagement yet" as a confirmed negative.

**Label definition is a design decision.** `P(click)` gives clickbait. `P(watch > 30s)` is better. `P(complete)` biases toward short videos — a real, frequently-observed pathology. Multi-task modeling exists precisely because no single label captures value.

## 1.8 Failure modes

| Failure | Why | Mitigation |
|---|---|---|
| **Clickbait spiral** | Optimizing click-proxy labels | Multi-task value model with negative terms (§1.5) |
| **Filter bubble / diversity collapse** | Exploit-only ranking | Diversity in re-ranking; exploration slots; measure intra-list diversity |
| **Popularity bias** | Sampling without log-Q correction; rich-get-richer | Log-Q correction; IPW; forced exploration |
| **New items never surface** | No interaction history | Content embeddings + guaranteed exploration budget (§1.6) |
| **Training-serving skew** | Features recomputed at training time | Log features as served |
| **Silent feature pipeline failure** | Upstream job dies; features go stale or null | Freshness SLAs, null-rate alerts, **model-level canaries** |
| **Hot key (viral item)** | One item in every candidate set | Cache its features; replicate hot keys |
| **Stale embeddings** | Nightly index rebuild | Fresh-item path with minute-level indexing |
| **Seasonality / drift** | Distribution shifts (holidays, events) | Frequent retraining; drift monitors; recency features |
| **Offline/online gap** | Offline metrics don't predict online | Treat offline as a filter, not a decision (§1.9) |
| **Feedback loop amplification** | Model shapes its own training data | Randomized holdout as ground truth |

## 1.9 Evaluation

**Offline — a filter, not a decision:**
- *Retrieval:* recall@k, coverage of the catalog
- *Ranking:* AUC, **nDCG**, calibration error, per-segment breakdowns
- *Counterfactual/off-policy:* IPS and doubly-robust estimators to approximate online performance before shipping

**Online — where decisions get made.** A/B test with sticky bucketing, on:
- **Primary:** watch time, sessions per user, **retention** (the real goal)
- **Guardrails:** diversity, creator fairness (Gini over impressions), report rate, latency, cost
- **Long-horizon holdback:** a small population held out for months to catch metrics that degrade slowly — the only way to detect that you optimized short-term engagement at the cost of retention

> **The offline/online gap is notorious in recsys, and interviewers probe it.** Offline metrics are computed on logged data produced by the *current* model, so a new model that would surface different items is being graded on a distribution it wouldn't have created. **Offline evaluation reliably filters out bad models; it does not reliably identify good ones.** Say that.

**Interleaving** — show results from two rankers in a single blended list. Far more sensitive than A/B and needs less traffic, making it excellent for ranker comparison, though it can't measure system-level effects.

## 1.10 Cost and latency

| Lever | Impact | Note |
|---|---|---|
| ⭐ **Precompute item embeddings** | Enormous | The two-tower architecture exists for this |
| ⭐ **Quantize the ANN index** (int8 / PQ) | 256 GB → 64 GB | Verify recall loss on a golden set |
| **Batch candidate feature fetch** | Large | 1,000 individual lookups is the classic latency bug |
| **Cache user embeddings** for the session | Moderate | User tower needn't rerun every request |
| **Truncate the ranking pool** (1,000 → 500) | Linear in ranking cost | Measure nDCG loss; often surprisingly small |
| **GPU dynamic batching for ranking** | Large at 30k QPS | Triton handles it |
| **Tiered ranking** — cheap model to 200, deep model on those | Large | Cascade inside stage two |

**The structural point:** retrieval is cheap and ranking is expensive, so **the candidate count is your primary cost dial.** Every latency and cost conversation comes back to it.

## 1.11 Corner Questions

**Q: Offline AUC improved 2% but the online A/B is flat. What happened?**
> The most likely cause is the offline/online gap: offline metrics are computed on logged impressions generated by the *current* model, so the new model is graded on a distribution it would never have produced — it gets no credit for items it would have surfaced and the old one didn't. Beyond that: AUC is a global ranking metric while users only see the top ~20, so gains in the middle of the ranking don't reach anyone; the improvement may sit in a segment with little traffic; or there's training-serving skew, so the model that scored well offline isn't the one running. I'd check per-segment and top-k metrics first, then verify served features against training features, then look at whether the gain concentrates in items the current policy rarely shows — which points back at the distribution gap.

**Q: The feed is full of clickbait. What went wrong?**
> The label. Optimizing `P(click)` or even raw watch time rewards misleading thumbnails and titles, because the click happens before the disappointment. The fix is a multi-task value model with explicit negative terms — subtract `P(skip)`, `P(early abandon)`, and `P(report)` — and to weight completion and satisfaction over the click itself. I'd also add post-engagement signals like next-day return, which is where clickbait's real cost shows up. And I'd check whether guardrail metrics existed at all: if the A/B that shipped this only measured CTR, the system did precisely what it was asked to do, and that's a metrics-design failure rather than a modeling one.

**Q: A new creator uploads a genuinely great video. How does it ever get seen?**
> It won't, under a pure exploit ranker — that's the cold-start trap, and it's a supply-side existential problem, not just a quality one. Three mechanisms: content-based embeddings from the title, transcript, and thumbnail via a multimodal encoder, which place the video in the same space as established items so it's retrievable on day zero; creator priors, so a new upload inherits some signal from the creator's history; and a **guaranteed exploration budget** — reserved feed slots for under-explored items, with Thompson sampling inside those slots. The forced budget is the critical piece, because without it the item never accumulates the evidence it would need to be shown.

**Q: How do you stop the filter bubble?**
> Measure it first — intra-list diversity per feed, and category entropy per user over time. Without measurement it's an opinion. Then: diversity constraints in re-ranking (cap items per creator or topic), exploration slots, and explicitly modeling long-term user value rather than per-session engagement, since narrowing is usually short-term optimal and long-term harmful. The honest tension worth naming is that diversity almost always costs short-term engagement, so it has to be a stated product constraint with a guardrail metric — otherwise every A/B will correctly reject it.

**Q: Ranking 1,000 candidates takes 400 ms and your budget is 200 ms total.**
> Cascade within stage two: a cheap model scores all 1,000, the expensive model scores only the top ~200. That's usually most of the gap for a small nDCG cost, which I'd measure rather than assume. Then: GPU dynamic batching, which at 30k QPS is a large win; quantize the ranker; batch the candidate feature fetch, since 1,000 individual store lookups is the classic hidden latency bug. And I'd genuinely test truncating retrieval to 500 candidates — the quality loss is often much smaller than people expect, and candidate count is the primary cost dial in the whole system.

**Q: How do you know you're improving retention rather than just short-term engagement?**
> Short-horizon A/B can't tell you — that's the core difficulty, since the two look identical for weeks. I'd use a long-horizon holdback: a small population kept on the old model for months, which is the only way to see slow-moving degradation. Alongside that: track leading indicators known to correlate with churn (session frequency, return rate, unsubscribes, report rate) as guardrails, and treat any experiment that raises watch time while degrading return rate as a failure regardless of the primary metric. The deeper point is that retention is the goal and watch time is a proxy, so the measurement system has to keep checking the proxy still tracks the goal.

**Q: A video goes viral and becomes a hot key. What breaks?**
> Feature-store reads for that item spike, since it's now in nearly every candidate set — a single Redis key taking a large share of total traffic. Fixes: cache its features in-process on each ranking host with a short TTL, replicate the hot key across nodes, and read from a random replica. It also distorts training data, because one item dominates impressions and the model over-fits to it, so I'd consider capping its representation in training batches. And it stresses the exploration budget, since a viral item crowds out everything else — a per-item impression cap in re-ranking handles that.

**Q: Two-tower for retrieval, cross-attention for ranking. Why not use the good model for both?**
> Because a cross-attention model can't be precomputed. It requires a forward pass per user-item pair, so scoring 500M items per request is impossible — at even 1 µs per item that's 500 seconds. The dot product between independently-computed towers is a deliberate accuracy sacrifice that buys sublinear retrieval via ANN: item vectors are computed once nightly and the user vector once per request. That's exactly why the expensive interaction model belongs in stage two, where it only sees 1,000 candidates. The general pattern is cheap-and-high-recall first, expensive-and-high-precision second, and it recurs in search, ads, and RAG.

**Q: Someone watches a recommended video three days later. How does that enter training?**
> An attribution window — decide how long after impression an engagement still counts, typically hours to a few days for video. That creates the delayed-label problem: recent training data is systematically incomplete, so treating "no engagement yet" as a negative mislabels examples that simply haven't converted. Options are to wait for the window to close before training on a period, which is correct but stale, or to model the delay explicitly and reweight. I'd wait for the ranker's main training and keep fast-updating counters on real-time signals, so freshness comes from features rather than from prematurely labeled data.

## 1.12 Mid-level vs Senior

| | Mid-level | **Senior** |
|---|---|---|
| Objective | "Predict clicks" | Business goal → proxy → **names the gap** and designs against it |
| Architecture | "Use a neural net" | Two-stage derived from arithmetic; explains why towers can't interact |
| Baseline | Starts deep | Non-ML baseline first; GBDT before DLRM |
| Retrieval | "Embeddings + ANN" | Multi-source union; log-Q correction; hard negatives |
| Objectives | Single head | Multi-task value model with **negative terms** |
| Bias | Not mentioned | Position bias, popularity bias, presentation bias — each with a fix |
| Feedback loops | Not mentioned | Randomized holdout as unbiased ground truth; IPW |
| Cold start | "Use popularity" | Both kinds; content embeddings; **forced** exploration budget |
| Evaluation | "Check AUC" | Offline as a filter not a decision; interleaving; long-horizon holdback |

## 1.13 References for this case study

**Read first**
- ⭐ **Deep Neural Networks for YouTube Recommendations** — Covington et al., RecSys 2016 — **the** two-stage paper; candidate generation + ranking, and still the clearest statement of the architecture
- ⭐ **Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations** — Yi et al., RecSys 2019 — two-tower with **log-Q correction** (§1.4). The single most practically important detail in two-tower training.
- **Recommending What Video to Watch Next: A Multitask Ranking System** — Zhao et al., RecSys 2019 — multi-task ranking and **position bias** handling (§1.5), from YouTube

**Ranking architectures**
- **Wide & Deep Learning** — Cheng et al., 2016 (arXiv 1606.07792) — memorization vs generalization
- **DCN V2: Improved Deep & Cross Network** — Wang et al., 2020 (2008.13535) — the practical default ranker
- **DLRM** — Naumov et al., 2019 (1906.00091) — Meta's architecture; embedding-table scale
- **Deep Interest Network** — Zhou et al., 2017 (1706.06978) — attention over user history (Alibaba)
- **SASRec** (1808.09781) · **BERT4Rec** (1904.06690) — sequence models for in-session intent
- **Practical Lessons from Predicting Clicks on Ads at Facebook** — He et al., ADKDD 2014 — calibration and the GBDT+LR lineage

**Retrieval & embeddings**
- **PinSage: Graph Convolutional Neural Networks for Web-Scale Recommender Systems** — Ying et al., KDD 2018 — graph embeddings at Pinterest
- **ScaNN: Accelerating Large-Scale Inference with Anisotropic Vector Quantization** — Guo et al., 2020 (1908.10396)
- **Embedding-based Retrieval in Facebook Search** — Huang et al., KDD 2020 (2006.11632) — hard negatives, done properly

**Bias, feedback loops, evaluation**
- **Recommendations as Treatments: Debiasing Learning and Evaluation** — Schnabel et al., 2016 (1602.05352) — IPW for recsys
- **Degenerate Feedback Loops in Recommender Systems** — Jiang et al., 2019 (1902.10730)
- **Unbiased Learning-to-Rank with Biased Feedback** — Joachims et al., 2016 (1608.04468) — position bias
- **Overlapping Experiment Infrastructure** — Tang et al., Google, KDD 2010 — the A/B platform paper
- **Netflix Tech Blog** — *Artwork Personalization*, *Calibrated Recommendations*, and the Netflix Prize retrospectives
- **Monolith: Real Time Recommendation System With Collisionless Embedding Table** — ByteDance, 2022 (2209.07663) — real-time training at TikTok scale

**Code**
- ⭐ `pytorch/torchrec` (~2k) — sharded embedding tables; the answer to "your embedding table is 500 GB"
- `facebookresearch/dlrm` (~4k) — reference implementation
- `google-research/google-research/tree/master/scann` — ANN with anisotropic quantization
- `NVIDIA-Merlin/Merlin` (~1k) — end-to-end GPU recsys pipeline
- `RUCAIBox/RecBole` (~4k) — ~100 recsys models implemented consistently; good for comparing architectures
- ⭐ `eugeneyan/applied-ml` (~28k) — the *Recommendation* section is the best curated index of production recsys writeups

## 1.14 Practical grounding — a production news recommender

Grounded in real Inshorts news-app interaction data from a data-science take-home assignment.

> **Use the data, not the assignment.** The take-home asks for two algorithms and a top-50 CSV — that's a *modelling exercise*, and it stops exactly where the interesting engineering starts. The valuable version is: **use this data to design and build a production news recommender**, then let the measured numbers dictate the architecture. That's what §1.14 does — the findings below are measured, and several of them overturn the video-shaped design in §1.3–1.6.
>
> The take-home's own criteria point the same way: it asks for **real-time capability** and **production deployment**. Those are system-design requirements, and answering them is where the actual signal is.

### Measured profile

| | |
|---|---|
| **Events** | 1,063,248 over **28 days** (2023-06-27 → 07-25) |
| **Users** | 8,265 — **`deviceId`, not a login**; identity resets on reinstall |
| **Items** | 12,362 consumed; 14,671 in content table |
| **Density** | 1.03% · median 20 impressions/user, p90 276 |
| **`TimeSpent-Front`** | 1,044,042 — finished viewing the summary card |
| **`TimeSpent-Back`** | 13,509 — **clicked through to the full source article**; median dwell 19.6 s, mean 41.7 s. **Click-through rate 1.29%** |
| **Bookmarked / Shared** | 3,270 / 1,066 |
| **`Relevancy Option Selected`** | **373 events from only 79 devices** — RED 158 · GREEN 149 · YELLOW 66 |
| **Dwell (Front)** | median 3.75 s · mean 8.9 s · >10 s: **23.1%** · >30 s: **5.66%** |
| **Long tail** | top 1% of items = 5.8% of impressions; **29.9% of items have <10 impressions** |
| **Freshness** | median item age at consumption **8.4 h**, p90 22.2 h; 43% consumed within 24 h of publish |

### Six findings that change the design

**1. The test set is engineered so collaborative filtering cannot work.** `testing_content` holds **878 items, 100% disjoint** from the 6,951 training items, published **2023-07-26 → 07-27 — strictly after the training window closes**. There is not one interaction on any test item.

> **This is the whole point of the task.** Matrix factorization, item-item CF, or any model keyed on item ID scores **exactly zero** on the test set. Every viable algorithm must represent items by *content* — title, category, language, recency, district. Stating this in your first paragraph is the single highest-value observation available, and it's what the task is really testing.

**2. Position bias, measured.** Mean dwell by `cardViewPosition` (the swipe/page index):

| Slot | 0 | 1 | 2 | 5 | 10 | 20 | 50 | 99 |
|---|---|---|---|---|---|---|---|---|
| **Mean dwell (s)** | **14.83** | 10.51 | 9.39 | 7.85 | 7.84 | 6.86 | 6.94 | 5.51 |
| **Impressions** | 122,106 | 38,847 | 24,691 | 6,484 | 5,397 | 3,614 | 1,430 | 2,127 |

Two distinct effects, and separating them is the senior read: **dwell decays 2.7× from slot 0 to 99** (attention bias) *and* **impressions fall 68% from slot 0 to slot 1** (exposure bias — most sessions are a few swipes). Train naively and the model learns "slot 0 is good," not "this article is good." This curve is also a directly usable **propensity estimate for IPW** (§1.7).

**3. "Expressed interests" barely exist — and saying so is a finding, not a failure.** The task asks you to use them, but explicit relevancy signals cover **79 of 8,265 devices (~1%)**. So: use them where present (they're high-precision, and RED is a genuine negative — rare and valuable), but the system must run on *inferred* interests from consumption history for 99% of users. Reporting this gap with the number is exactly the "problem-solving / data challenges" criterion in the rubric.

**4. Location relevance is a data-quality trap.** The task explicitly requires it. The data:

| Field | Coverage | Verdict |
|---|---|---|
| Event `state` / `district` / `locality` | **~1.1%** | Unusable |
| Device `district` | **0.2%** | Unusable |
| Device `lastknownsubadminarea` (city) | **91.3%**, 1,157 distinct | **This is your geo signal** |
| Test content `newsDistrict` | **35.2%**, 10 distinct | Only a third of test items are geo-taggable |

So geo matching is possible — but only via `lastknownsubadminarea` ↔ `newsDistrict`, and it can only boost ~35% of test items. Build it as an *additive boost*, never a filter, or you discard two-thirds of the inventory.

**5. Language is the strongest test-set signal, and the device field is broken.** `language_selected` has **exactly one distinct value** across all 10,400 devices — useless. But test content is **english 329 / hindi 327 / telugu 93** (6 languages), while training consumption is english+hindi dominated. **You must infer language preference from each device's consumption history**, then hard-match it. Recommending Telugu articles to a Hindi reader is the most obvious way to tank precision, and it's free to fix.

**6. The finding that inverts the architecture.** Publish rate, measured: **878 test items over 41.2 h ≈ 512 items/day**; 5,552 items published during the 28-day event window ≈ 198/day. With a 48-hour freshness window:

```
LIVE CATALOG ≈ 1,000 items
```

> **A thousand candidates is what the *retrieval stage exists to produce*.** In §1.2 the funnel was 500M → 1,000 → 20, and two-stage retrieval existed because scoring 500M items in 200 ms is impossible. Here the live catalog *is already* ~1,000 items. **So you can skip candidate generation entirely and exhaustively score every live article with the full ranker.** No ANN index, no two-tower model, no nightly embedding rebuild, no index-freshness problem — all of it deleted, along with the recall ceiling that retrieval imposes.
>
> This is the most valuable thing the dataset teaches: **the canonical two-stage pattern is a response to catalog size, and news doesn't have that problem.** Applying it here would be cargo-culting. Recognizing when *not* to apply the famous pattern is a stronger signal than reciting it.

The whole live catalog also fits in memory — ~1,000 items × (embedding + features) is a few MB, so every serving pod can hold the full candidate set and refresh it incrementally.

### News breaks the video-shaped assumptions in §1.4–1.6

| Base design assumes | News requires |
|---|---|
| Nightly item-embedding rebuild | **Useless** — median item is consumed 8.4 h after publish. Embed at ingest. |
| Cold start is an edge case | **Cold start is the steady state** — and here it's 100% of the test set |
| Collaborative signal dominates | **Content-based must dominate**; no time to accumulate interactions |
| Recency is one feature | **Recency is the strongest single feature**; a recency-popularity prior is a serious baseline |

### The production system this data actually implies

```
CONTENT INGEST (publish → servable in <5 min)          REQUEST PATH (<150 ms)
──────────────────────────────────────────           ──────────────────────
 CMS publish ──▶ Kafka `article.published`             Request (deviceId)
        │                                                    │
        ▼                                                    ▼
 Enrich: multilingual embed (BGE-M3),               Load user profile ◀── Redis
 category, language, district, source                (affinity, lang, geo,
        │                                             recent seen-set)
        ▼                                                    │
 Near-dup cluster (MinHash on title)                         ▼
        │                                          ┌────────────────────────┐
        ▼                                          │ LIVE CATALOG (in-proc) │
 LIVE CATALOG STORE  ──── broadcast ──────────────▶│ ~1,000 items, 48h TTL  │
 (Redis + in-process replica, 48h TTL)             └───────────┬────────────┘
                                                               ▼
 ENGAGEMENT STREAM                                   Hard filters: language ·
 events ──▶ Kafka ──▶ Flink                          already-seen · RED topics
        │                                                      ▼
        ├─▶ item counters (dwell, CTR) ── 30s ──▶ Redis   SCORE ALL (~1k)
        └─▶ user profile update  ─────── 30s ──▶ Redis    GBDT + recency decay
                                                               ▼
                                                     Re-rank: story dedup ·
                                                     category diversity ·
                                                     exploration slots
                                                               ▼
                                                        Top-N cards
```

**Key departures from §1.3, each forced by a measured number:**

| Design choice | Driven by |
|---|---|
| **No retrieval stage** — score the full live catalog | ~1,000 live items |
| **Ingest-time embedding**, not nightly | median consumption at 8.4 h of age |
| **48-hour item TTL** | p90 consumption at 22.2 h |
| **Streaming counters at ~30 s** | breaking news; a 1-hour-stale popularity feature is worthless |
| **Language as a hard filter** | 6 languages in test content; `language_selected` is broken |
| **Optimize slots 0–2 ruthlessly** | 68% of users never reach slot 1; slots 0–2 hold 17.8% of all impressions |
| **Cold-start path is the main path** | 100% of test items are cold |

### Real-time serving, with a latency budget

The take-home asks for "real-time capability." Concretely, at production scale (~10M devices, ~30k QPS peak):

```
User profile fetch (Redis)               8 ms
Live catalog read (in-process)           0 ms   ← already resident
Hard filters (language, seen, RED)       2 ms
Score ~1,000 items (GBDT, vectorized)   25 ms
Re-rank (dedup, diversity, explore)      5 ms
Overhead + network                      40 ms
                                    ─────────
                                        80 ms
```

Two things make this work: the **live catalog is in-process** so there's no candidate-fetch round trip, and **GBDT over ~1,000 rows vectorizes** into a single batched call. If the catalog grew 10×, you'd reintroduce a cheap pre-filter — but *only then*, and you'd say so.

### Cold start as the steady state, not an edge case

Every article is cold, and every reinstall makes a user cold. So the exploration machinery from §1.6 stops being a nicety and becomes the core loop:

- **Item cold start:** content embedding + category/source priors at ingest, so an article is rankable the second it publishes
- **Guaranteed exploration slots** — reserve 1–2 of the first 10 cards for under-exposed items. Without a *forced* budget, a fresh article never accumulates the evidence it needs to be ranked, so it's never shown. With ~1,000 live items and a fast-decaying window, that's fatal to catalog coverage.
- **Thompson sampling** over the item's dwell posterior inside those slots — with ~1,000 candidates you can afford proper uncertainty modelling rather than ε-greedy
- **User cold start:** bootstrap from `lastknownsubadminarea` (91.3% populated) + device language + popularity, then adapt within the session. Median session is ~20 impressions, so you get signal fast — the profile update must be *streaming*, not batch.

### Retraining, monitoring, rollout

| Concern | For news specifically |
|---|---|
| **Retraining cadence** | Ranker daily (feature distributions shift with the news cycle); counters continuously via Flink. Weekly is too slow — a model trained before a major news event mis-ranks during it |
| **Drift** | Monitor category mix, language mix, and dwell distribution. A sudden shift usually means a news event, not a broken model — **alert, don't auto-rollback** |
| **Guardrails** | Cold-item exposure share, category diversity per session, per-source concentration, first-card dwell |
| **Rollout** | Shadow → 1% → 10% → 50%, bucketed by `deviceId` so a user's feed doesn't flip mid-session |
| **The failure to watch** | Recency decay tuned too aggressively collapses the feed onto the last hour's articles. Track age-distribution of *served* items, not just consumed |

### Evaluating it honestly

Report **recall@k** and **nDCG@10** on a temporal split (train ≤ 07-25, test = the 878 items from 07-26/27 — **never random**, which leaks the future in a recency-dominated domain), **segmented by item age, language, and cold vs heavy user**. A blended number hides a model that only works for heavy users.

**On A/B testing:** you cannot run a live A/B on a static log, and pretending otherwise is the trap. Deliver instead:
- **Offline replay** plus **counterfactual estimates (IPS / doubly-robust)** using the measured propensity curve from finding 2
- **A designed experiment:** unit = `deviceId`, primary = dwell-per-session, guardrails = click-through, diversity, cold-item exposure; a **power calculation** from observed variance stating the detectable effect at n ≈ 8,265; fixed run length to prevent peeking
- **Interleaving** as the cheaper alternative — far more sensitive per unit of traffic (§1.9)

> **State the limitation plainly — it's a senior signal.** These are logs from an *existing* recommender, so you only observe what that policy chose to show. No randomized holdout means you cannot measure true catalog recall, and IPW propensities come from the same biased logs. **You can build and compare rankers here; you cannot prove one is better in production.** Then say what you'd instrument to fix it: a 1% randomized-exposure slice, which at this scale costs almost nothing and is the only unbiased data you'd ever have.

### Staged build

| Stage | Deliverable | Why this order |
|---|---|---|
| **0** | EDA in **DuckDB** over the CSVs — no warehouse needed | Satisfies the SQL requirement |
| **1** | Baseline: most-popular-last-6h, language-matched | In news this is **strong**. If nothing beats it, that's the finding |
| **2** | Point-in-time feature builder (leakage-safe) | Everything downstream depends on it; popularity computed over the full window is the classic silent bug |
| **3** | GBDT ranker, label = dwell > 10 s, position-debiased | Real baseline in minutes (§1.5's "GBDT before DLRM") |
| **4** | Multilingual content embeddings + similarity features | Handles the 100%-cold test set |
| **5** | Multi-objective value model (dwell + click-through + bookmark − RED) | §1.5 |
| **6** | **Serving prototype** — FastAPI, in-process live catalog, <100 ms p95 | This is the step the take-home skips and the one that makes it a *system* |
| **7** | Streaming layer — Kafka/Flink counters + profile updates | Makes "real-time" real |

**Stages 6–7 are the point.** Stages 0–5 are a modelling exercise; 6–7 are what make it a production recommender, and they're what you'd actually talk about in an interview.

---

# Case Study 2 — Modern RAG over Enterprise Knowledge

> *"Design a system that lets 50,000 employees ask questions over the company's entire internal knowledge — docs, wikis, tickets, code, chat — respecting permissions."*

This is Glean's product, and it's now a standard ML-design prompt. The naive version (chunk → embed → top-k → generate) is a 2023 answer and will get you marked down. This is the 2026 version.

## 2.1 Requirements

**Ask:**
- "What sources, and do they have existing permission models?" *(This is the fork — ACLs change everything.)*
- "How fresh must results be — minutes or days?"
- "Is this one-shot Q&A, or conversational and agentic?"
- "What's the volume of documents and of queries?"

**Assume:** 10M documents across Confluence/Drive/Jira/Slack/GitHub, 50k employees, 100k queries/day, permissions inherited from source systems, freshness target of minutes, conversational.

**Non-functional:**
- **p95 end-to-end < 3 s**, TTFT < 1 s (stream)
- **Zero permission leakage.** Hard constraint. A single leaked HR document is a company-level incident.
- Freshness SLA: indexed within 5 minutes of source change
- Every answer **cited and verifiable**

**Descope out loud:** the connectors themselves (assume they exist), auth/SSO, the UI.

## 2.2 Estimation

```
100k queries/day ÷ 86,400 s        ≈ 1.2 QPS average
but enterprise traffic isn't uniform — it lands in ~8 working hours:
100k ÷ 28,800 s                    ≈ 3.5 QPS sustained during the workday
× ~2–3 intra-day burstiness         ≈ 7–10 QPS peak
```

> **Note the peak factor.** The usual rule of thumb is peak ≈ 3× average (as in §1.2), but that assumes traffic spread over 24 h. **Internal tools concentrate in working hours**, so the effective factor is closer to 8–10×. Deriving it rather than applying the generic multiplier is worth saying out loud.

> **The insight that reframes the whole design.** **~1 QPS average, ~10 QPS peak is *nothing*.** You do not have a throughput problem. So do not spend the interview on sharding and load balancing — **spend it on retrieval quality, permissions, and index lifecycle**, which is where this system actually fails. Saying this in minute six is a strong senior signal: it shows you size before you architect.

**Storage is the real constraint:**
```
10M docs × ~20 chunks   = 200M chunks
200M × 1024 dims × 4 B  = 819 GB  (fp32 — too much to keep in RAM)
        × 1 B (int8)    = 205 GB  (fits a large-memory host / small cluster)
        + binary first-pass ~26 GB (hierarchical: binary scan → int8 rerank)
```
Quantization is a *design decision here*, not an afterthought — it's the difference between a cluster and a machine.

**Ingestion compute (the number people forget):**
```
200M chunks × ~500 tokens        = 100B tokens to embed
Contextual enrichment (§2.4):    200M cheap LLM calls, prompt-cached per parent doc
Steady state: ~1% of docs change/day → ~2M chunks/day re-embedded
```
The initial build is a multi-day GPU job; steady state is small. **That asymmetry is the whole argument for incremental sync over nightly rebuilds.**

**Per-query cost:**
```
Retrieval + rerank:  negligible compute
Generation: ~6k input (10 chunks + query + history) + 500 output
            ≈ $0.026/query at $3/M in, $15/M out
100k queries/day    ≈ $2.6k/day  → ~$950k/yr
```
That number justifies prompt caching on the stable system prefix and a model cascade for simple lookups.

## 2.3 Architecture

```
INGESTION (async, continuous)                QUERY (per request, <3s)
─────────────────────────────                ────────────────────────
Connectors (Drive/Jira/Slack/Git)            User query + conversation
   │  CDC / webhooks / polling                  │
   ▼                                            ▼
Parse (layout-aware, OCR, tables)            Query understanding
   │                                          (rewrite · decompose · HyDE)
   ▼                                            │
Clean → near-dup detect (MinHash)               ▼
   │                                         ┌──────── Hybrid retrieval ────────┐
   ▼                                         │ Dense (ANN)   Sparse (BM25/bm42) │
Structure-aware chunking                     │ ACL partition + filtered ANN     │
   │                                         │        └──── RRF ────┘           │
   │                                         └──────────────┬───────────────────┘
   ▼                                                        ▼
CONTEXTUAL ENRICHMENT  ◀── the 2026 step          Rerank (cross-encoder)
   │                                                        ▼
   ▼                                           Fine-grained ACL post-filter
Embed (dense + sparse + multivector)              + live authz re-check
   │                                                        ▼
   ▼                                              Context assembly + dedup
Index upsert / tombstone                                    ▼
   │                                              Generate → cite → stream
   └──▶ ACL snapshot ──────────────────────────────────────┘
```

### Recommended stack

| Component | ⭐ Default | Alternative | Flip when |
|---|---|---|---|
| **Embedding model** | ⭐ **BGE-M3** (self-host) | Cohere Embed v3/v4 · OpenAI text-embedding-3-large | Cohere if you want managed + strong multilingual; OpenAI if you want **Matryoshka** truncation (3072→1024→256) to trade accuracy for RAM without re-embedding |
| **Why BGE-M3 here** | Multilingual (100+ langs, answers the Hindi-query question) **and emits dense + sparse + multivector from one model** — hybrid retrieval without running three | | |
| **Sparse / lexical** | **bm42** (in Qdrant) | Elasticsearch/OpenSearch **BM25** · SPLADE | Use ES if it's already in your stack — you get BM25 + filtering free |
| **Reranker** | ⭐ **BGE-reranker-v2-m3** | Cohere Rerank 3 · **ColBERTv2** | Cohere for best quality-per-effort managed; ColBERT when you want rerank-grade relevance *at retrieval time* instead of a second pass |
| **Vector store** | ⭐ **Qdrant** | Milvus · Vespa · Elasticsearch | **Milvus** past ~1B vectors with a dedicated infra team; **Vespa** when ranking *is* the product; **ES** if already deployed |
| **Why Qdrant here** | **Filtered HNSW that holds recall under payload filters** — that's the §2.6 ACL problem — plus built-in scalar/binary quantization (the 819 GB → 205 GB → 26 GB ladder) and multivector support for ColBERT | | |
| **Why *not* pgvector** | At 200M chunks with per-query ACL filtering it stops being the simple choice. **Under ~10M vectors, pgvector would be the right answer** — the threshold is the point | | |
| **Doc parsing** | ⭐ **Docling** + **PyMuPDF** | Unstructured.io · LlamaParse · Textract | Unstructured for format breadth; Textract/Azure DI for heavy OCR on scans and forms |
| **Metadata + ACLs** | **PostgreSQL** | — | Relational by nature — permissions, ownership, versions. Don't put this in the vector store. |
| **Change sync** | **Kafka** (or **Redpanda**) | Direct webhooks | Redpanda for Kafka semantics with far lighter ops; webhooks only if sources are few |
| **Ingestion orchestration** | **Dagster** | Airflow | Dagster's asset model fits "this chunk derives from that doc" lineage better; Airflow if the team already knows it |
| **Eval** | **RAGAS** + golden set in CI | Braintrust / Phoenix | Managed eval platform when non-engineers need to read results |

> **FAISS is not the answer here.** It's a *library* — no persistence, replication, or filtering. Correct as an embedded index inside one service; wrong as "our vector store." Interviewers notice the distinction.

## 2.4 Deep Dive A — Chunking and contextual retrieval

**Structure-aware chunking** beats fixed-size on real documents: respect section boundaries, keep tables intact, don't split code blocks. Overlap ~10–15% so context isn't severed mid-sentence.

**Then the technique that matters most in 2026 — contextual retrieval.** The core problem: a chunk reading *"The margin improved by 3% this quarter"* is unretrievable, because it never says which company, product, or quarter. Embedding it is embedding an orphan.

**Fix:** before embedding, prepend a short LLM-generated context describing where the chunk sits in its document:
```
"This chunk is from Acme Corp's Q3 2025 financial report,
 in the section on the EMEA cloud business unit."
+ original chunk text
→ embed this combined text
```
Anthropic reported this cuts retrieval failure rate substantially (~35% alone, ~49% combined with BM25 and reranking). It's a **one-time ingestion cost** — and prompt caching over the parent document makes it cheap — in exchange for a permanent retrieval-quality gain. Naming this technique is one of the clearest "I've kept up" signals available.

**Also consider RAPTOR** (recursive summarization into a tree, retrieve at multiple abstraction levels) when documents are long and questions span sections.

## 2.5 Deep Dive B — The retrieval cascade

Retrieval is a **funnel**, and each stage trades cost for precision:

| Stage | Candidates | Latency | Purpose |
|---|---|---|---|
| Dense ANN (int8/binary) | 200M → 200 | ~20 ms | Semantic recall — **within the user's permission partition, filtered ANN** (§2.6) |
| Sparse BM25 / bm42 | 200M → 200 | ~15 ms | Exact terms, IDs, error codes, rare tokens — same ACL scoping |
| **RRF fusion** | 400 → 60 | ~1 ms | Combine without tuning weights |
| **Cross-encoder rerank** | 60 → ~15 | ~150 ms | Precision — where most quality lives |
| Fine-grained ACL post-filter | ~15 → 10 | ~10 ms | Per-document overrides + live authorization re-check (§2.6) |
| Context assembly | 10 → prompt | ~5 ms | Dedup, order, budget |

**Why hybrid is non-negotiable:** dense retrieval fails on exact identifiers — search `ERR_4021` or `PROJ-1847` and semantic similarity returns *conceptually related* tickets, not *that* ticket. BM25 nails it. Conversely BM25 fails on paraphrase. You ship BGE-M3 dense + bm42 sparse + ColBERT late-interaction together, which is exactly this argument in production.

**RRF** (`score = Σ 1/(k + rank_i)`, k≈60) fuses ranked lists without needing calibrated scores across retrievers — that's why everyone uses it.

**Reranking is where the quality is.** A cross-encoder scores query and document *jointly* rather than comparing independent embeddings, so it catches relevance that bi-encoders miss. It costs ~150 ms for 60 candidates. At 10 QPS that's trivially affordable — another reason the low-QPS observation matters.

## 2.6 Deep Dive C — ACL-aware retrieval (the hard part)

Nobody prepares this, and it's the most likely deep-dive target.

**The naive approaches both fail:**
- **Post-filter** (retrieve top-100, drop unauthorized): cheap, but if a user can see 2% of the corpus you may return an empty page. Recall collapses for low-privilege users.
- **Pre-filter** (restrict the ANN search to permitted docs): correct, but arbitrary filters destroy HNSW graph connectivity and recall degrades badly.

**The practical answer is layered:**
1. **Partition the index by coarse permission group** (org, workspace, public-tier) so the common case is a whole-partition query with no filtering at all
2. **Filtered ANN within a partition** using a filterable index (Qdrant and others support payload-filtered HNSW with acceptable recall)
3. **Post-filter fine-grained ACLs** (per-document overrides) on the small reranked set, with a **live authorization re-check** against the source system for sensitive hits — steps 1–2 are the primary control, this is the second layer
4. **Over-fetch adaptively** — if post-filtering leaves too few results, re-query with a larger k

**ACLs must be snapshotted at ingest and re-synced on change.** Permission changes are events: revoke access to a Drive folder and the index must reflect it within the freshness SLA, not the next full crawl.

**The leak nobody thinks about — existence leakage.** Even without content, "0 results" vs "results you can't see" leaks whether a document exists. If someone searches "Project Falcon acquisition" and gets a permission-denied signal, they've learned the project exists. Return uniform empty results; never distinguish "not found" from "not permitted."

## 2.7 Deep Dive D — Index lifecycle

**The re-embedding problem.** Changing the embedding model invalidates **the entire index** — old and new vectors are not comparable, so you cannot mix them. At 200M chunks that's a serious batch job.

**Zero-downtime migration:**
1. Build the new index alongside the old (embed 200M chunks — hours to days on a GPU fleet)
2. Dual-write new/changed documents to both during the build
3. Shadow-evaluate: run the golden query set against both, compare recall@k and nDCG
4. **Blue/green swap** behind the retrieval service; keep the old index warm
5. Roll back instantly if online metrics regress

Budget it: 200M chunks × ~500 tokens is a real bill. Say the number rather than waving.

**Incremental sync:** CDC/webhooks where available, polling with etags/hashes where not. **Deletions must produce tombstones** — a document deleted for legal or HR reasons must vanish from retrieval within the SLA, and "we rebuild nightly" is not an acceptable answer for that class of deletion.

## 2.8 When *not* to use RAG

Have this ready — it's the most common curveball:

| Situation | Better answer |
|---|---|
| Corpus fits in context (< ~200k tokens) and is stable | **Just stuff the context** + prompt caching. Simpler, higher quality, no index to maintain. |
| Global/thematic question ("what themes recur across incidents?") | **GraphRAG or hierarchical summarization** — top-k retrieval structurally cannot answer questions about the corpus as a whole |
| Question needs computation over structured data | **Text-to-SQL**, not retrieval |
| Knowledge is stable, narrow, high-volume | **Fine-tuning** may beat retrieval on latency and cost |
| Complex multi-hop question | **Agentic RAG** — iterative targeted searches |

## 2.9 Failure modes

| Failure | Why it happens | Mitigation |
|---|---|---|
| **Orphan chunks** — retrievable text with no context | Chunk says "margin improved 3%", never names the company or quarter | **Contextual retrieval** (§2.4) |
| **Exact-ID queries fail** | Dense embeddings don't encode `ERR_4021` | Hybrid — BM25 leg is mandatory, not optional |
| **Empty results for low-privilege users** | Post-filtering drops the entire top-k | Partition by permission group; adaptive over-fetch (§2.6) |
| **Permission leak** | Stale ACL snapshot after a revoke | Event-driven ACL sync + source-of-truth check on sensitive hits |
| **Existence leakage** | "0 results" ≠ "not permitted" | Uniform empty response for both |
| **Stale/deleted content served** | Nightly rebuild cadence | CDC + tombstones within the freshness SLA |
| **Global questions return garbage** | Top-k structurally can't answer corpus-wide questions | Detect intent → GraphRAG / RAPTOR (§2.8) |
| **Duplicate results crowd out diversity** | Same doc exists in Drive, Confluence, and email | Near-dup detection (MinHash) at ingest + dedup at assembly |
| **Silent recall regression** | A chunking or embedding "improvement" ships | Golden set in CI on every retrieval-path change |
| **Index/model version skew** | Query embedded with model B against an index built with A | Version-tag the index; refuse mismatched queries loudly |
| **Conflicting answers** | Two docs disagree, model picks arbitrarily | Authority + recency metadata; surface the conflict |

## 2.10 Evaluation

**Retrieval and generation, measured separately — always.**

**Retrieval:** golden query set (start with ~200 queries, real ones from logs) with labeled relevant documents. Measure **recall@k** (did the right doc make the candidate set?), **nDCG@10**, **MRR**.

**Bootstrapping labels without annotators** — the question interviewers love: **generate synthetic queries from chunks.** For each of a sampled few thousand chunks, have an LLM write a question that chunk answers; the chunk is then the known-relevant document. Imperfect but unblocks measurement on day one, and you replace it with real click/feedback data as it accumulates.

**Generation:** faithfulness/groundedness (is every claim supported by retrieved context?), answer relevance, citation correctness. RAGAS implements these.

**End-to-end:** thumbs up/down, click-through on citations, query reformulation rate (a strong implicit negative — users rephrase when the first answer failed).

**In CI:** the golden set runs on every chunking, embedding, prompt, or reranker change. This is what catches the "improvement" that quietly drops recall by 8%.

## 2.11 Cost and latency

**Latency budget for a 3 s p95:**
```
Query understanding (rewrite/decompose)   200 ms   ← skip for simple queries
Dense ANN + sparse BM25 (parallel)         25 ms
RRF fusion                                  1 ms
Cross-encoder rerank (60 → ~15)           150 ms
Fine-grained ACL post-filter + re-check    10 ms
Context assembly                            5 ms
Generation TTFT                           600 ms
                                    ───────────────
                              ~1.0 s to first token, ~2.5 s full
```
**Run dense and sparse retrieval in parallel** — they're independent, and serializing them is a common unforced error. **Stream the answer**; TTFT is the perceived latency.

**Where the money goes** (from §2.2: ~$950k/yr generation, ingestion amortized):

| Lever | Saving | Cost |
|---|---|---|
| **Prompt-cache the stable system prefix** | 30–50% of input cost | None — do it |
| **Model cascade** — small model for lookups, large for synthesis | 40–60% on easy queries | Routing complexity + misroute risk |
| Cache answers for repeated queries (dedup by embedding) | 10–20% | Staleness; needs invalidation on index change |
| Cut rerank for high-confidence retrievals | ~150 ms + compute | Measurable quality loss — verify first |
| Fewer chunks in context (10 → 5) | ~30% input | Recall loss on multi-hop questions |

**The counterintuitive one:** contextual enrichment (§2.4) *raises* ingestion cost meaningfully but is a one-time spend that permanently reduces the number of chunks needed per query — so it often **lowers** total cost of ownership while improving quality. Say that; it shows you reason about amortized rather than unit cost.

## 2.12 Corner Questions

**Q: Context windows are 1M+ tokens now. Is RAG dead?**
> No, for four reasons — but the question is fair for small corpora. (1) **Cost** — stuffing 1M tokens per query is orders of magnitude more expensive than retrieving 5k, and at 100k queries/day that's decisive. (2) **Latency** — prefill over 1M tokens is slow, and TTFT is the perceived latency. (3) **Scale** — 10M documents is billions of tokens; it doesn't fit at any context length. (4) **Quality** — "Lost in the Middle" shows attention degrades over long contexts; more irrelevant context can *hurt* accuracy. That said, for a corpus under ~200k tokens I'd genuinely skip RAG and stuff the context with prompt caching — simpler and better. The honest framing is that long context raises the floor at which RAG becomes worth its complexity.

**Q: A user asks "what are the recurring themes across all our incident postmortems?" Your system returns garbage. Why?**
> Because top-k retrieval answers *local* questions and this is a *global* one. No individual chunk contains "the recurring themes" — the answer is a property of the whole corpus, so retrieving 10 chunks can't produce it no matter how good the ranking is. Fixes: **GraphRAG** — extract entities and relations at ingest, cluster into communities, pre-generate community summaries, and answer global queries from those. Or **RAPTOR** — recursive summarization into a tree, retrieving at the abstraction level the question needs. Or, if the postmortem set is small enough, map-reduce over all of them directly. I'd want to detect global-vs-local intent in query understanding and route accordingly.

**Q: How do you guarantee no permission leakage?**
> Layered, with the guarantee at the bottom. Coarse partition by permission group, filtered ANN within it, fine-grained post-filter after reranking — and critically, **the final authorization check hits the source system's permission model, not my cached snapshot**, for anything sensitive. Snapshots go stale; the source of truth doesn't. I'd also return uniform empty results so "no results" and "not permitted" are indistinguishable, preventing existence leakage. And I'd test it adversarially: a red-team suite where low-privilege accounts query for known-restricted content, run in CI.

**Q: You want to upgrade the embedding model. Walk me through it.**
> Full re-embed of all 200M chunks — old and new vectors aren't comparable, so no incremental path exists. Build the new index in parallel, dual-write changed docs to both, shadow-eval against the golden set on recall@k and nDCG, blue/green swap behind the retrieval service, keep the old index warm for instant rollback. Cost is the real constraint — 200M × ~500 tokens of embedding compute — so I'd validate the gain on a 1% sample before committing the full run. This is the most expensive routine operation in a RAG system's life, and I'd want it to be a scripted, rehearsed procedure rather than a heroic one-off.

**Q: Retrieval returns the right document but the answer is still wrong. Diagnose.**
> Retrieval succeeded, so the failure is downstream. Candidates, in the order I'd check: (1) the relevant chunk is in context but buried mid-prompt — reorder so strong evidence sits at the edges; (2) context is diluted by 9 irrelevant chunks — tighten k or the rerank threshold; (3) conflicting information across chunks and the model picked wrong — needs recency/authority metadata; (4) the chunk was split so the answer is half-present — chunking or overlap problem; (5) genuine generation error. I'd measure this with a faithfulness check, and the fact that I *can* separate these cases is exactly why retrieval and generation are instrumented independently.

**Q: Two documents contradict each other. What does the system do?**
> Never silently pick one. Attach authority and recency metadata at ingest — source system, last-modified, document status (draft vs published), owner. Prefer the authoritative and current one, and when conflict remains material, **surface it**: "The 2024 policy says X; a 2022 doc says Y." For policy questions especially, a confidently wrong single answer is worse than an honest conflict. This also argues for aggressive tombstoning of superseded documents at ingest — most contradiction is stale content that should have been retired.

**Q: Reranking 60 candidates per query is expensive. Justify it.**
> At 10 QPS peak, 150 ms of reranking is essentially free — I have the headroom, and reranking is where most of the quality gain lives, so it's the last thing I'd cut. If volume grew 100×, I'd cascade: a cheap reranker on 100 candidates, an expensive one on the top 20, and skip reranking entirely for high-confidence retrievals where dense and sparse already agree strongly on the top result. But I'd cut it based on measured quality loss, not on a hunch.

**Q: A document is deleted for legal reasons. How fast is it gone?**
> Within the freshness SLA — minutes, not the next rebuild — via a webhook/CDC-driven **tombstone** that removes it from the index immediately, plus purge from any retrieval cache and any conversation-history store that may have copied its text. That last part is the one people miss: if a cached answer or a stored conversation quotes the document, deleting the index entry isn't enough. For legal deletion I'd want an explicit, auditable purge path across index, caches, and logs — and I'd test it.

**Q: Queries in Hindi, documents in English.**
> Multilingual embeddings that share a vector space across languages — BGE-M3 handles this, which is why I'd pick it — so a Hindi query retrieves English documents directly, no translation step. Generate in the user's language while citing the English source. The caveat worth stating: BM25's sparse half doesn't cross languages, so hybrid retrieval degrades to dense-only for cross-lingual queries; I'd either translate the query for the sparse leg or accept lower exact-match recall there. I benchmarked translation across 52 languages and deployed MBART-50, so I'd validate this empirically rather than assume it.

**Q: You have no labeled data. How do you start measuring?**
> Synthetic queries: sample a few thousand chunks, have an LLM generate a question each one answers, and treat that chunk as the known-relevant document. That gives recall@k on day one. It's biased — synthetic questions are cleaner than real ones — so I'd treat it as a regression baseline, not ground truth, and replace it as fast as possible with real signal: query logs, citation click-through, thumbs up/down, and reformulation rate. Then a few hundred human-labeled queries on the highest-traffic intents as the trusted gold set.

## 2.13 Mid-level vs Senior

| | Mid-level | **Senior** |
|---|---|---|
| Retrieval | "Embed and top-k" | Hybrid + RRF + rerank cascade, with cost per stage |
| Chunking | "500 tokens with overlap" | Structure-aware + **contextual retrieval**, with the failure it fixes |
| Permissions | "Filter by user" | Pre/post-filter trade-off, existence leakage, source-of-truth check |
| Index lifecycle | Not mentioned | Re-embedding migration, tombstones, freshness SLA |
| Scope | Assumes RAG is the answer | Names when long-context, GraphRAG, or SQL beats RAG |
| Eval | "It seems better" | Retrieval and generation measured separately, synthetic bootstrap, CI regression |

## 2.14 References for this case study

**Read first**
- **Anthropic — *Introducing Contextual Retrieval*** (engineering blog) — the §2.4 technique, with measured failure-rate reductions (~35% alone, ~49% with BM25 + reranking). The highest-value single read for this design.
- **Lost in the Middle: How Language Models Use Long Contexts** — Liu et al., 2023 (arXiv 2307.03172) — the empirical basis for context ordering and for the "more context can hurt" argument in §2.12.

**Retrieval architecture**
- **RAG for Knowledge-Intensive NLP** — Lewis et al., 2020 (2005.11401) — the original
- **Dense Passage Retrieval** — Karpukhin et al., 2020 (2004.04906)
- **ColBERT** — Khattab & Zaharia, 2020 (2004.12832) · **ColBERTv2** — Santhanam et al., 2021 (2112.01488) — late interaction
- **BGE-M3** — Chen et al., 2024 (2402.03216) — multilingual + multi-function; the cross-lingual answer in §2.12
- **Reciprocal Rank Fusion** — Cormack et al., SIGIR 2009 — why RRF beats tuned score blending
- **HNSW** — Malkov & Yashunin, 2016 (1603.09320) — the index, and why filtering degrades it (§2.6)

**Beyond top-k**
- **GraphRAG: From Local to Global** — Edge et al., 2024 (2404.16130) — the answer to the "recurring themes" corner question
- **RAPTOR** — Sarthi et al., 2024 (2401.18059) — hierarchical retrieval
- **Self-RAG** — Asai et al., 2023 (2310.11511) · **Corrective RAG** — Yan et al., 2024 (2401.15884) — agentic retrieval
- **HyDE** — Gao et al., 2022 (2212.10496) — query-side expansion

**Evaluation**
- **RAGAS** — Es et al., 2023 (2309.15217) — faithfulness / context precision as computable metrics
- **BEIR** — Thakur et al., 2021 (2104.08663) — the retrieval benchmark suite
- **MTEB** — Muennighoff et al., 2022 (2210.07316) — embedding model selection

**Code**
- ⭐ `NirDiamant/RAG_Techniques` (~22k) — ~30 techniques implemented side by side. Best single repo for this case study.
- `explodinggradients/ragas` (~9k) — eval metrics in code
- `qdrant/qdrant` (~25k) — quantization + filtered-HNSW docs, directly relevant to §2.2 and §2.6
- `run-llama/llama_index` (~45k) — ingestion pipeline abstractions

---

# Case Study 3 — LLM Inference & Serving Platform

> *"Design the inference platform that serves every model at your company — from a 1B classifier to a 2T-parameter MoE."*

The Nvidia interview, and the only design here where **the answers come from hardware physics, not software architecture**. Everything below follows from one asymmetry (§3.2). Get that right and the rest is derivation.

> **📚 Prerequisites.** This case study assumes the fundamentals. If any step below feels like recall rather than derivation, read these first:
> - **[`01-transformer-architecture.md`](01-transformer-architecture.md)** — every component with math; the KV cache; prefill vs decode; parameter counting; MoE
> - **[`02-modern-transformer-architectures.md`](02-modern-transformer-architectures.md)** — Llama, Qwen3, DeepSeek, Gemma configs and what each costs to serve
> - **[`03-training-data-pipeline.md`](03-training-data-pipeline.md)** — how the training corpus is collected, filtered, deduplicated, mixed and packed
> - **[`05-inference-serving.md`](05-inference-serving.md)** — hardware, PagedAttention, parallelism, quantization, speculative decoding, deployment ops, and a 30-question Q&A index

## 3.1 Requirements

**Ask:**
- "What model sizes, and what's the workload mix — interactive chat, agent loops, or batch?"
- "Own GPU fleet or cloud endpoints?" *(Owning the fleet makes utilization the business metric.)*
- "Per-workload SLOs, or one global one?" *(Critical — interactive and batch want opposite configurations.)*
- "Are the large models dense or MoE?" *(Completely different memory and parallelism story.)*

**Assume:** internal multi-tenant platform, ~100 models spanning **1B → 2T** (the largest being MoE), owned H100/H200/GB200 fleet, mixed interactive + agent + batch traffic.

**Non-functional — and note there are two SLOs, not one:**

| Workload | TTFT | TPOT | Optimize for |
|---|---|---|---|
| **Interactive chat** | p95 < 500 ms | < 30 ms (~33 tok/s, above reading speed) | Latency |
| **Agent loops** | p95 < 1 s | < 50 ms | Cost — long stable prefixes dominate |
| **Batch** (embeddings, classification, offline gen) | irrelevant | irrelevant | Pure throughput |

- **GPU utilization > 60%** — with an owned fleet this *is* the business metric; idle H100s are the dominant cost
- Multi-tenant fairness: no tenant can starve another

**Descope:** training, fine-tuning, the model registry.

## 3.2 Estimation — the asymmetry everything follows from

**Prefill and decode are fundamentally different workloads on the same hardware.**

| | **Prefill** | **Decode** |
|---|---|---|
| Work | All prompt tokens in parallel | One token at a time, sequential |
| FLOPs | ≈ 2 × params × prompt_tokens | ≈ 2 × params **per token** |
| Bytes moved | Weights once, amortized | **All weights, every single token** |
| Bound by | **Compute** | **Memory bandwidth** |
| Batching helps? | Marginally | **Enormously** |
| Determines | TTFT | TPOT |

**The roofline that decides your architecture.** On an H100 SXM (~989 TFLOPS BF16 dense, ~3.35 TB/s HBM3):
```
ridge point ≈ 989e12 / 3.35e12 ≈ ~295 FLOPs per byte
```
At batch size 1 with FP16/BF16 weights, decode does roughly **1 FLOP per byte of weight read** (2 FLOPs per parameter ÷ 2 bytes per parameter). That is ~300× below the ridge — decode is *hopelessly* bandwidth-bound. At batch B it rises to ≈ B FLOPs/byte; prefill over S tokens sits at ≈ S. (FP8 doubles both sides — ~2 FLOPs/byte against a ~590 ridge — so the ratio is unchanged.) **Batching is what drags it toward the ridge**, which is why continuous batching is the single most important optimization in serving, and why it does nothing for TTFT.

**Decode speed ceiling — memorize this:**
```
max tokens/sec per sequence ≈ HBM_bandwidth / model_bytes
```
| Model | Weights | H100 (3.35 TB/s) | H200 (~4.8 TB/s) |
|---|---|---|---|
| 1B FP16 | 2 GB | ~1,600 tok/s | ~2,400 tok/s |
| 8B FP16 | 16 GB | ~209 tok/s | ~300 tok/s |
| 8B FP8 | 8 GB | ~420 tok/s | ~600 tok/s |
| 70B FP16 | 140 GB | doesn't fit | doesn't fit |
| 70B FP8 | 70 GB | fits, but ~no KV headroom → TP=2 in practice | **fits one H200** comfortably |

> **This table answers "why is quantization a serving decision, not a compression trick."** FP8 doesn't just halve memory — it *doubles decode throughput*, because decode is bandwidth-bound. Say that sentence; it's the clearest signal you understand the hardware.

**Memory budget:**
```
Total = weights + KV cache + activations + fragmentation
KV bytes/token = 2 (K,V) × layers × kv_heads × head_dim × dtype_bytes
```
Llama-3-8B (32 layers, 8 KV heads via GQA, head_dim 128, FP16):
```
2 × 32 × 8 × 128 × 2 = 131,072 B = 128 KB/token
80 GB − 16 GB weights = 64 GB for KV
64 GB / 128 KB = ~490k tokens → at 8k context ≈ 60 concurrent sequences
```
**That's your concurrency per GPU.** Every capacity question reduces to this arithmetic.

**The 2T MoE — the interesting case.** Say 2T total with ~50B active per token (~40× sparse — more aggressive than DeepSeek-V3's 671B/37B ≈ 18×):
```
Memory:  capacity-bound on TOTAL params  → 2T × 1 B (FP8) = ~2 TB
         → ~15 H200s for weights alone (2 TB / 141 GB ≈ 14.2); realistically 32–64 GPUs with EP
Compute: bandwidth-bound on ACTIVE params → only ~50 GB read per token
```

**Mapping — units and sizes used above:**

| Term | Value |
|---|---|
| 8B model size | Sum of all weight matrices (GQA form, `01` §12.1): 2·V·d + L·(2d² + 2·d·h_kv·d_h + 3·d·d_ff) ≈ 8.03B for Llama 3 |
| 1 KB | 1 thousand bytes |
| 1 MB | 1 million bytes |
| 1 GB | 1 billion bytes (10⁹); 1 GiB = 1,073,741,824 bytes |
| 1 TB | 1 trillion bytes = 1,000 GB (1 TiB = 1,024 GiB) |
| 1 T | 1,000 B |
| 8B params @ INT8 | 8 GB |
| 8B params @ FP16 | 16 GB |
| 8B params @ FP32 | 32 GB |
| H200 memory | 141 GB HBM3e @ 4.8 TB/s |
| H100 memory | 80 GB HBM3 |
| B200 memory | 192 GB HBM3e |

> **The insight that separates senior from mid on MoE: memory scales with *total* parameters, speed scales with *active* parameters.** A 2T MoE with 50B active decodes at roughly the speed of a 50B dense model while costing the memory footprint of a 2T one. That decoupling is the entire reason MoE exists, and it's why the deployment problem is *capacity and interconnect*, not raw FLOPs.

## 3.3 Architecture

```
                    ┌──────────────────────────┐
   Requests ───────▶│  Gateway / Model Router  │ auth · quota · model routing
                    └────────────┬─────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │   Admission Control      │ SLO class · priority · queue
                    └────────────┬─────────────┘
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
     ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
     │ INTERACTIVE    │ │  AGENT POOL    │ │  BATCH POOL    │
     │ small batch    │ │ prefix-cache   │ │ max batch      │
     │ latency SLO    │ │ optimized      │ │ spot GPUs      │
     └────────────────┘ └────────────────┘ └────────────────┘
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 ▼
        ┌────────────────────────────────────────────────┐
        │   DISAGGREGATED SERVING (large models)         │
        │  ┌─────────────┐   KV over    ┌─────────────┐  │
        │  │  PREFILL    │ ──NVLink/IB─▶│   DECODE    │  │
        │  │  workers    │              │   workers   │  │
        │  │ compute-opt │              │ bandwidth-opt│ │
        │  └─────────────┘              └─────────────┘  │
        └────────────────────────────────────────────────┘
                                 ▼
        ┌────────────────────────────────────────────────┐
        │  KV CACHE TIER:  HBM → CPU DRAM → NVMe         │
        │  paged · prefix-shared · offloaded             │
        └────────────────────────────────────────────────┘
                                 ▼
              GPU FLEET — NVLink intra-node, IB/RoCE inter-node
```

**Three pools, not one.** Interactive, agent, and batch workloads want *opposite* configurations — small batch for latency versus max batch for throughput. Serving them from one pool guarantees you serve all three badly. Saying this early prevents the most common design error here.

### Recommended stack

| Component | ⭐ Default | Alternative | Flip when |
|---|---|---|---|
| **Inference engine** | ⭐ **vLLM** | **SGLang** · **TensorRT-LLM** | **SGLang** when prefix reuse dominates (agents, multi-turn) — RadixAttention is purpose-built for it; **TensorRT-LLM** for maximum NVIDIA throughput when you'll accept a compile step and stiffer ops |
| **Orchestration** | ⭐ **NVIDIA Dynamo** (disaggregated) or **vLLM production-stack** | Ray Serve · KServe | Dynamo when you're running disaggregated prefill/decode at scale; Ray Serve if already on Ray |
| **Small models (1–8B)** | Many replicas per GPU, high batch | MIG partitioning | MIG for hard multi-tenant isolation with predictable QoS |
| **Quantization — weights** | ⭐ **FP8 (E4M3)** on Hopper/Blackwell | **NVFP4/MXFP4** (Blackwell) · **AWQ/GPTQ INT4** | NVFP4 on Blackwell for 4-bit with better quality than INT4; AWQ when memory-capacity-bound and quality loss is verified acceptable |
| **Quantization — KV cache** | **FP8 KV** | INT4 KV (KIVI) | Long-context workloads where KV, not weights, is the constraint |
| **Speculative decoding** | ⭐ **EAGLE-2/3** | Medusa · draft model · n-gram lookahead | n-gram when output is highly repetitive (code, structured); skip entirely at high batch (§3.7) |
| **Attention kernels** | **FlashAttention-3** (Hopper) / **FlashInfer** | Engine default | Usually just take the engine's default — it's tuned |
| **KV offload** | **LMCache** or **Mooncake** | HBM only | When prefix-cache hit rate is high but HBM can't hold the working set |
| **Parallelism** | See §3.6 — **TP/EP inside the NVLink domain, PP/DP across nodes** | | |
| **Interconnect** | **NVLink/NVSwitch** intra-node, **InfiniBand/RoCE** inter-node | — | TP *must* stay inside the NVLink domain; crossing it collapses throughput |
| **Autoscaler** | Queue-depth + SLO-aware, **pre-warmed pool** | CPU-util autoscaling | Never scale GPUs on CPU utilization — see §3.8 cold start |
| **Load testing** | **genai-perf** / vLLM benchmark suite | Locust | Must report TTFT, TPOT, **and goodput** — not just RPS |
| **Observability** | Prometheus + Grafana; per-engine metrics | — | Track batch occupancy, KV utilization, preemption rate, cache hit rate |

## 3.4 Deep Dive A — Prefill/decode interference and disaggregation

**The problem:** co-locating prefill and decode on the same GPU makes them fight. A single 32k-token prefill occupies the GPU for ~1 s on an 8B model on one H100 (~2–3 s for a 70B on TP=4), during which **every in-flight decode stalls** — so one user's long prompt spikes TPOT for everyone. You cannot optimize TTFT and TPOT simultaneously in a shared pool.

**Two fixes, and you should know both:**

**1. Chunked prefill** (Sarathi-Serve; default in vLLM). Split a long prefill into chunks and interleave them with decode iterations. Decodes keep progressing, TPOT jitter collapses, at a small TTFT cost. **This is the right default** — simple, no extra infrastructure.

**2. Disaggregated serving** (DistServe, Mooncake, NVIDIA Dynamo) — the significant architectural shift of 2025–26. Run prefill and decode on *separate GPU pools*, transferring the KV cache between them over NVLink or InfiniBand.

| Benefit | Detail |
|---|---|
| Independent tuning | Prefill wants compute-dense config and high TP; decode wants bandwidth and different parallelism |
| Independent scaling | Long prompts and long generations scale separately |
| No interference | TTFT and TPOT stop trading against each other |

**Cost:** KV transfer bandwidth and complexity. **Worth it at scale and for large models; over-engineering for a single 8B model.** Say that — knowing when *not* to disaggregate is the senior half of the answer.

## 3.5 Deep Dive B — Batching, scheduling, and the KV cache

**Continuous (in-flight) batching is the foundational win.** Static batching pads every request to the longest in the batch and waits for all to finish — GPUs idle while one sequence generates. Continuous batching schedules at **iteration level**: finished sequences leave and new ones join every forward pass. This alone is often a 2–4× throughput gain, and it's the thing to name first.

**PagedAttention** solves the memory side. Naive KV allocation reserves max_context per sequence, wasting 60–80% to internal fragmentation. Paging into fixed blocks (vLLM) drops waste to a few percent **and makes sharing possible** — which enables:

**Prefix caching / RadixAttention** — reuse KV for shared prefixes across requests. **This is the single biggest lever for agent workloads**, where every step resends a long stable system prompt (the coding-agent quadratic token growth (07-agentic-system-design.md §3.2)). Hit rates of 70–90% are achievable on agent traffic, and it cuts both cost and TTFT.

**KV cache tiering** — HBM → CPU DRAM → NVMe (LMCache, Mooncake). Trades PCIe bandwidth for capacity; worth it when the reusable prefix working set exceeds HBM.

**Model-level KV reduction** (chosen at training time, but you must know it):
| Technique | KV size | Used by |
|---|---|---|
| MHA (full) | baseline | older models |
| **GQA** | ÷ 4–8 | Llama 3, most modern models |
| MQA | ÷ heads | some fast-decode models |
| **MLA** (latent compression) | ÷ ~10+ | DeepSeek |

**Scheduling for SLOs:** admission control with per-tenant token budgets, priority queues, and **preemption with recompute-or-swap** when KV runs out. Preemption is the hidden source of long-tail latency (§3.8).

## 3.6 Deep Dive C — Parallelism across the full size spectrum

**The rule that generates every answer: match the parallelism to the interconnect.**

| Strategy | Communication | Place it |
|---|---|---|
| **Tensor (TP)** | All-reduce **every layer** — very chatty | **Inside the NVLink domain only** |
| **Pipeline (PP)** | Activations at stage boundaries — light | Across nodes; costs bubbles |
| **Expert (EP)** | All-to-all per MoE layer — chatty, bursty | High-bandwidth domain (NVL72-class) |
| **Data (DP)** | None | Across replicas, for throughput |

**Deployment by size — the table to have ready:**

| Params | Config | Reasoning |
|---|---|---|
| **1–3B** | TP=1; **many replicas per GPU**, or MIG | Weights are ~2–6 GB. The mistake here is under-batching — these are almost pure throughput plays |
| **7–13B** | TP=1, max batch | Fits comfortably; highest tokens/$ of any tier |
| **30–40B** | TP=1 with FP8, or TP=2 | Quantization decides whether you need a second GPU |
| **70B** | **FP8 → single H200** comfortably; on H100, FP8 fits (70 GB) but leaves ~no KV headroom → TP=2 in practice | The clearest example of quantization changing topology |
| **100–400B dense** | TP=8 within node, PP across nodes | TP is capped by the NVLink domain size |
| **600B–2T MoE** | **EP + TP hybrid**, 32–64 GPUs | Capacity-bound on total params; needs an NVL72-class domain for all-to-all |

**Serving the 2T MoE concretely:**
1. FP8 weights → ~2 TB → ~15 H200s minimum for weights, realistically 32–64 GPUs with KV and headroom
2. **Expert parallelism** — shard experts across GPUs; each token routes to its few active experts
3. **All-to-all is the bottleneck** — tokens must reach their experts and return, twice per MoE layer. This demands a GB200 NVL72-class NVLink domain; over InfiniBand it stalls.
4. **Expert load imbalance is the operational problem** — some experts are far hotter than others, so GPUs straggle. Mitigate with capacity factors, expert replication for hot experts, and load-aware routing.
5. Decode speed tracks the ~50B *active* params, so it feels like a mid-size dense model to the user

## 3.7 Deep Dive D — Quantization and speculative decoding

**Quantization ladder — know what each buys:**

| Format | Memory | Speed | Quality | Use |
|---|---|---|---|---|
| BF16 | 1× | 1× | reference | Baseline / quality-critical |
| ⭐ **FP8 (E4M3)** | 0.5× | ~2× decode | near-lossless | **Production default on Hopper/Blackwell** |
| **NVFP4 / MXFP4** | 0.25× | ~3–4× | good (block scaling) | Blackwell; better than INT4 at same size |
| INT4 (AWQ/GPTQ) | 0.25× | ~3× | measurable loss | Memory-capacity-bound cases |
| FP8 KV cache | KV 0.5× | more concurrency | minor | Long-context workloads |

**Never ship a quantized model on perplexity alone** — run the task evals from `07` Case Study 4. Quantization damage is uneven: it hits long-context, multilingual, and reasoning-heavy tasks disproportionately while leaving perplexity nearly unchanged.

**Speculative decoding — why it works.** A small draft model proposes *k* tokens; the target verifies all *k* in **one forward pass**. Since decode is bandwidth-bound, verifying 5 tokens costs almost exactly what verifying 1 costs — you were reading all the weights either way. **You're converting spare compute into latency.**

| Method | Approach | Typical speedup |
|---|---|---|
| Draft model | Separate small model | 2–3× |
| **Medusa** | Extra decoding heads on the target | ~2× |
| ⭐ **EAGLE-2 / EAGLE-3** | Feature-level autoregression; best acceptance rates | 3–4× |
| N-gram / lookahead | No draft model; pattern matching | 1.5–2×, excellent on code |

> **The critical caveat, and a favorite interview trap: speculative decoding stops helping at high batch size.** Its whole premise is spare compute during bandwidth-bound decode. At large batch you're already near the compute roofline, so speculation just burns FLOPs on rejected tokens and can make throughput *worse*. **Enable it for latency-critical low-batch traffic; disable it in the batch pool.** Knowing when to turn an optimization off is a stronger signal than knowing it exists.

## 3.8 Failure modes

| Failure | Why | Mitigation |
|---|---|---|
| **KV cache OOM under load** | Concurrency exceeded the block budget | Admission control on *projected* KV; preempt with recompute-or-swap |
| **Long-tail latency spikes** | Preempted sequences restart | Cap preemption; prioritize in-flight over new admissions |
| **Head-of-line blocking** | One 128k prompt monopolizes the GPU | **Chunked prefill**; separate long-prompt queue |
| **Noisy neighbor** | One tenant floods the batch | Per-tenant token budgets, weighted fair queuing |
| **Cold start (minutes)** | Loading 2 TB of weights over the network | Pre-warmed pool, local NVMe weight cache, never scale reactively on GPUs |
| **Autoscaling thrash** | GPUs take minutes to boot | Scale on queue depth with long cooldowns; keep standby capacity |
| **Expert imbalance (MoE)** | Skewed routing; hot experts straggle | Capacity factors, expert replication, load-aware routing |
| **NCCL hang / partition** | Multi-node collective stalls | Timeouts, health checks, fail the replica rather than hanging the fleet |
| **Silent quality regression** | Quantization shipped on perplexity alone | Task-level eval gate before rollout (07-agentic-system-design.md §4 — eval & guardrails) |
| **Spec-decoding slowdown** | Enabled at high batch | Adaptive: disable above a batch threshold |
| **Prefix cache thrash** | Working set exceeds HBM | Tier to CPU/NVMe; route same-prefix requests to the same replica |
| **GPU underutilization** | Batch too small, pools mis-sized | Track **batch occupancy**, not just GPU util — a busy GPU can still be doing nothing useful |

## 3.9 Evaluation

**Benchmark the right metrics — and *goodput* is the one that matters:**

| Metric | Definition |
|---|---|
| **TTFT** | Time to first token — prefill-bound |
| **TPOT / ITL** | Inter-token latency — decode-bound |
| ⭐ **Goodput** | **Requests/sec that actually meet their SLO** |
| Throughput | Total tokens/sec (misleading alone) |
| **MFU** | Model FLOPs utilization — how much of the hardware you're really using |
| Cost/1M tokens | The business number |

> **Raw throughput is a vanity metric.** A config with 2× the tokens/sec but 40% of requests missing TTFT targets has *worse* goodput. Interviewers listen for whether you optimize throughput or goodput.

**Sweep the load curve, don't report a point.** Measure TTFT/TPOT as concurrency rises to find the knee — that's your capacity limit and the input to autoscaling. Test with **realistic prompt/output distributions**; benchmarks using uniform 128-token prompts tell you nothing about production.

**Quality gates on every serving change** — quantization, spec decoding, kernel or engine upgrades all need the task-eval suite, not just a latency check.

## 3.10 Cost and latency

**Levers ranked by impact:**

| Lever | Impact | Caveat |
|---|---|---|
| ⭐ **Continuous batching** | 2–4× throughput | Table stakes; verify it's actually on |
| ⭐ **Prefix caching** | Up to 90% cost cut on agent traffic | Needs prefix-affinity routing to pay off |
| ⭐ **FP8 quantization** | ~2× throughput + half the memory | Gate on task evals |
| **Right-sized pools** | Large | Interactive vs batch have opposite configs |
| **Speculative decoding** | 2–4× latency at low batch | **Hurts at high batch** |
| **Chunked prefill / disaggregation** | Big TPOT stability win | Disaggregation adds real complexity |
| **Spot/preemptible for batch** | 60–80% on batch | Needs checkpointing and requeue |
| **NVFP4 on Blackwell** | ~4× memory | Verify quality per model |

**The structural point:** with an owned fleet, **utilization is the cost function.** A platform at 30% utilization is wasting 70% of its capital regardless of how fast any single request is. Batch workloads exist to soak up the trough left by interactive traffic — that's the argument for one platform rather than per-team GPUs.

## 3.11 Corner Questions

**Q: Walk me through serving a 2T-parameter MoE.**
> First, separate the two constraints, because MoE decouples them: memory scales with *total* parameters, speed with *active* parameters. At FP8 the weights are ~2 TB, so I need roughly 15 H200s just to hold them and realistically 32–64 GPUs once KV and headroom are included. Parallelism is expert parallelism sharding experts across GPUs, combined with tensor parallelism inside each node. The bottleneck is the all-to-all: every token must reach its experts and come back, twice per MoE layer, which means I want a GB200 NVL72-class NVLink domain — over InfiniBand that collective dominates and the whole thing stalls. Operationally the hard problem is expert load imbalance, since hot experts make their GPUs straggle and the slowest GPU sets the step time; I'd use capacity factors, replicate hot experts, and route with load awareness. The pleasant surprise is that decode speed tracks the ~50B active parameters, so it feels like a mid-size dense model to users despite the 2 TB footprint.

**Q: TTFT is fine but TPOT is terrible under load. Diagnose.**
> Almost certainly prefill/decode interference — long prompts are monopolizing the GPU while in-flight decodes stall, so one user's 32k-token prompt spikes inter-token latency for everyone. First fix is chunked prefill: split long prefills into chunks interleaved with decode iterations, which collapses TPOT jitter for a small TTFT cost, and it's the right default. If that isn't enough at scale, disaggregate — separate prefill and decode pools with KV transferred over NVLink, so the two stop competing and can be tuned independently. I'd also check preemption rate, because if KV pressure is causing sequences to be evicted and recomputed, that shows up as exactly this symptom and the real fix is admission control.

**Q: Why does batching help throughput enormously but do nothing for TTFT?**
> Because they're bounded by different resources. Decode is memory-bandwidth-bound — at batch 1 you read every weight to produce a single token, roughly 1 FLOP per byte with BF16 weights against an H100 ridge point near 295, so the GPU is ~300× away from its compute limit and almost entirely idle. Adding sequences to the batch reuses that same weight read across many tokens — batch B gives roughly B FLOPs per byte — dragging you toward the roofline nearly for free. Prefill is already compute-bound: a long prompt saturates the GPU on its own, so batching adds queueing rather than efficiency. That's the whole reason TTFT and TPOT need separate SLOs and separate optimizations.

**Q: When does speculative decoding stop helping?**
> At high batch size, and it can actively hurt. Speculation works by converting spare compute into latency — during bandwidth-bound decode you're reading all the weights anyway, so verifying 5 proposed tokens costs about what verifying 1 costs. But at large batch you've already climbed to the compute roofline, so there is no spare compute; every rejected token is now wasted FLOPs and throughput drops. So I'd enable it in the latency-critical low-batch interactive pool and disable it in the batch pool, ideally adaptively on a batch-size threshold. It also degrades when acceptance rate is low — an out-of-domain draft model gets rejected constantly and you pay the drafting cost for nothing.

**Q: Your 70B model doesn't fit on one H100. Options?**
> Four, in the order I'd try them. Quantize to FP8: 140 GB becomes 70 GB. That technically fits an 80 GB H100 but leaves almost no KV headroom, so it's TP=2 in practice there; on one H200 it sits comfortably. Because decode is bandwidth-bound this also roughly doubles throughput, so it's the first move regardless. Tensor parallelism across 2 GPUs works but must stay inside the NVLink domain, and you pay an all-reduce every layer plus you've now spent two GPUs. INT4/NVFP4 if capacity is still tight, gated on task evals. Pipeline parallelism I'd avoid at this size — the bubbles aren't worth it for a model this small. FP8 on a single H200 is where I'd land: one GPU, no collectives, best tokens per dollar.

**Q: Your prefix cache hit rate is 20% and you expected 80%. Why?**
> Most likely routing — requests sharing a prefix are landing on different replicas, so each one builds its own cache and none of them hit. The fix is prefix-affinity routing: hash the prefix and route consistently to the same replica. Second possibility is eviction pressure, where the reusable working set exceeds HBM and entries are evicted before reuse; that's what KV tiering to CPU DRAM or NVMe is for. Third is that the prefix genuinely isn't stable — if a timestamp or session ID sits near the front of the system prompt, every request has a unique prefix and caching is impossible. That last one is a prompt-design bug and I'd check it first because it's free to fix: move volatile content to the end.

**Q: GPU utilization is 30%. Fix it.**
> First I'd distrust the metric — GPU utilization only says a kernel was resident, not that it was doing useful work. Batch occupancy is the honest number, and a GPU decoding at batch 2 shows high "utilization" while wasting most of its bandwidth. Assuming occupancy really is low: check that continuous batching is actually enabled and that admission isn't throttling too conservatively; check whether pools are mis-sized so interactive replicas sit idle between bursts; and consolidate — many small models each holding a dedicated GPU is the classic cause, fixed by co-locating them or using MIG. Then fill the trough with batch work, which is the main structural argument for one shared platform instead of per-team GPUs.

**Q: Quantized model passes perplexity checks but users say it got worse.**
> Perplexity is a weak proxy that averages over exactly the cases quantization breaks. The damage is uneven — it disproportionately hits long-context, multilingual, and multi-step reasoning while barely moving average next-token loss. So I'd gate on task-level evals from the eval platform (07-agentic-system-design.md §4), not perplexity: run the golden suite segmented by capability, and compare against the BF16 baseline per segment rather than in aggregate. I'd also A/B in production with quality guardrails before full rollout. And I'd look at *which* quantization — weight-only versus weight-and-activation behave very differently, and KV-cache quantization specifically degrades long-context recall, which matches the "it got worse" complaint pattern.

**Q: Cold start takes 8 minutes. Users are hitting timeouts during scale-up.**
> You cannot autoscale GPUs reactively — that's the core constraint, and it's what makes GPU serving different from stateless web services. Fixes: maintain a pre-warmed standby pool sized to your traffic variance and treat that idle capacity as the cost of meeting the SLO; cache weights on local NVMe rather than pulling from object storage every time, which is usually most of the 8 minutes; scale on queue depth with predictive signals and long cooldowns instead of instantaneous load. And shed load gracefully during scale-up — queue with an honest wait estimate, or route overflow to a smaller model, rather than timing out.

**Q: One tenant floods the platform and everyone else degrades.**
> Per-tenant token budgets enforced at admission, not just request-count rate limits — requests are wildly unequal, and a 128k-token prompt is not one unit of work. Weighted fair queuing so each tenant gets a guaranteed share under contention, with idle capacity redistributable. Separate SLO classes so a batch tenant physically cannot occupy interactive capacity. And per-tenant observability, because you want to detect this from a dashboard rather than from complaints. The framing I'd use: the scarce resource is GPU-seconds and KV blocks, so quota has to be denominated in those, not in requests.

## 3.12 Mid-level vs Senior

| | Mid-level | **Senior** |
|---|---|---|
| Core model | "Use vLLM, it's fast" | Prefill compute-bound vs decode bandwidth-bound; roofline reasoning |
| Quantization | "Saves memory" | **Doubles decode throughput** because decode is bandwidth-bound; gate on task evals |
| Batching | "Batch the requests" | Continuous batching at iteration level; why it does nothing for TTFT |
| KV cache | "It caches attention" | Sizing formula; paging; prefix sharing; GQA/MLA; tiering |
| Parallelism | "Shard across GPUs" | TP and EP (all-to-all) inside the NVLink domain, PP and DP across nodes — matched to interconnect |
| MoE | "It's a big model" | Memory scales with total params, speed with active; all-to-all is the bottleneck |
| Spec decoding | "It makes it faster" | **Knows when to turn it off** (high batch) |
| Metrics | "Tokens per second" | **Goodput**; batch occupancy over GPU util |
| Scaling | "Autoscale on load" | GPUs boot in minutes — pre-warm, never scale reactively |

## 3.13 References for this case study

**Read first**
- ⭐ **Efficient Memory Management for LLM Serving with PagedAttention (vLLM)** — Kwon et al., SOSP 2023 (arXiv 2309.06180) — **read this properly**; you run vLLM daily and Nvidia will probe it
- ⭐ **Orca: A Distributed Serving System for Transformer-Based Generative Models** — Yu et al., OSDI 2022 — continuous / iteration-level batching, the foundational scheduling idea
- ⭐ **Ray/Anyscale — *LLM numbers every developer should know*** and the **roofline model** — the §3.2 arithmetic

**Prefill/decode & scheduling**
- **DistServe: Disaggregating Prefill and Decoding** — Zhong et al., 2024 (2401.09670)
- **Sarathi-Serve: Taming Throughput-Latency Tradeoff with Chunked Prefills** — Agrawal et al., 2024 (2403.02310)
- **Mooncake: A KVCache-centric Architecture** — Qin et al., 2024 (2407.00079) — KV tiering and disaggregation in production
- **Splitwise: Efficient Generative LLM Inference Using Phase Splitting** — Patel et al., 2023 (2311.18677)
- **NVIDIA Dynamo** docs — the productized version of disaggregated serving

**Kernels & caching**
- **FlashAttention** (2205.14135) · **FlashAttention-2** (2307.08691) · **FlashAttention-3** (2407.08608) — IO-aware attention; the memory-bound argument made concrete
- **SGLang / RadixAttention** — Zheng et al., 2023 (2312.07104) — prefix caching done properly
- **KIVI: Plug-and-play 2bit KV Cache Quantization** — 2402.02750

**Quantization**
- **GPTQ** (2210.17323) · **AWQ** (2306.00978) · **SmoothQuant** (2211.10438)
- **FP8 Formats for Deep Learning** — Micikevicius et al., 2022 (2209.05433) — why E4M3 for inference
- **LLM.int8()** — Dettmers et al., 2022 (2208.07339) — outlier features, and why naive quantization breaks

**Speculative decoding**
- **Fast Inference from Transformers via Speculative Decoding** — Leviathan et al., 2022 (2211.17192) — the original
- **Medusa** — Cai et al., 2024 (2401.10774)
- ⭐ **EAGLE** (2401.15077) · **EAGLE-2** (2406.16858) — current best acceptance rates

**Scale & MoE**
- **Megatron-LM** — Shoeybi et al., 2019 (1909.08053) — tensor parallelism
- **GShard** (2006.16668) · **Switch Transformer** (2101.03961) — MoE foundations, expert routing and capacity factors
- ⭐ **DeepSeek-V3 Technical Report** — 2024 (2412.19437) — **MLA + fine-grained MoE at 671B/37B active**; the clearest public description of serving a sparse frontier model
- **ZeRO-Inference** / **DeepSpeed-Inference** (2207.00032) — offload strategies
- **S-LoRA** (2311.03285) · **Punica** (2310.18547) — serving thousands of LoRA adapters on shared base weights
- **NVIDIA GB200 NVL72 / NVLink** architecture briefs — why the all-to-all in §3.6 needs that domain

**Code**
- ⭐ `vllm-project/vllm` (~55k) — **read the scheduler and block manager**; that's PagedAttention and continuous batching in code. Highest-ROI codebase for you.
- ⭐ `sgl-project/sglang` (~18k) — RadixAttention prefix caching
- `NVIDIA/TensorRT-LLM` (~12k) — the Nvidia-native stack
- `ai-dynamo/dynamo` — disaggregated serving orchestration
- `LMCache/LMCache` (~4k) — KV offload and cross-instance sharing
- `flashinfer-ai/flashinfer` (~3k) — attention kernel library used by modern engines
- ⭐ `stas00/ml-engineering` (~15k) — the honest practitioner's guide to parallelism, interconnect, and debugging at scale

---


# Case Study 4 — NER / Information Extraction at Scale

> *"Design a system that extracts structured entities from millions of documents a day."*

The central question in 2026 isn't "which model" — it's **when to use a 110M-parameter encoder and when to use an LLM**, and the answer is a 100× cost ratio.

## 4.1 Requirements

**Ask:**
- "Which entity types, and is the schema fixed or evolving?" *(Evolving schema is the hard operational problem.)*
- "What's downstream — search, a knowledge graph, or analytics?" *(A KG needs high precision; bad entities poison it permanently.)*
- "Batch, real-time, or both?"
- "How many languages?"

**Assume:** financial/news domain, ~20 entity types (PERSON, ORG, TICKER, MONEY, DATE, PRODUCT, EVENT…), **10M documents/day**, multilingual, **schema gains a new type roughly monthly**, downstream is a knowledge graph plus search.

**Non-functional:**
- Batch throughput **10M docs/day**; on-demand path p95 **< 200 ms**
- **Precision ≥ 0.90** for KG ingestion — a wrong entity propagates into every downstream query
- **A new entity type shippable in < 1 week without relabelling the corpus**
- Character-offset spans preserved (downstream highlights them)

**Descope:** relation extraction, coreference, the KG merge policy itself.

## 4.2 Estimation — the ratio that decides the architecture

```
10M docs/day × ~500 tokens = 5B tokens/day
```

**Encoder path** (DeBERTa/XLM-R base, ~110M params):
```
~1,000 sequences/s/GPU batched → 10M docs ≈ 10,000 GPU-seconds ≈ 2.8 GPU-hours/day
≈ $10/day on one A100
```

**LLM path** (small model, structured output):
```
5B tokens/day × $0.15/M ≈ $750/day ≈ $274k/year
```

> **~100× cost difference, and roughly 50× latency difference.** That single ratio dictates the design: **the encoder must carry the volume, and the LLM is reserved for cases the encoder cannot handle.** Everything in §4.4 follows. Deriving this rather than asserting "use a hybrid" is the senior move.

**Output volume:** ~20 mentions/doc × 10M = **200M entity mentions/day** → this is a streaming-write problem, not just an inference problem.

## 4.3 Architecture

```
 Docs ─▶ Kafka ─▶ Chunk/window (respect sentence + offset mapping)
                        │
                        ▼
        ┌───────────────────────────────────────────┐
        │  TIER 1  Gazetteer / dictionary match     │  known tickers, orgs
        │          (exact, ~0 cost, precision ~1.0) │
        └───────────────┬───────────────────────────┘
                        ▼
        ┌───────────────────────────────────────────┐
        │  TIER 2  Fine-tuned encoder (the workhorse)│  99% of volume
        │          span-based, per-type heads        │
        └───────────────┬───────────────────────────┘
                        ▼
                 confidence gate
              ┌─────────┴─────────┐
         high │                   │ low / new type
              ▼                   ▼
        Entity linking      ┌──────────────────────┐
        (surface → KB id)   │ TIER 3  LLM extract  │  ~1% of volume
              │             │ constrained decoding │
              │             └──────────┬───────────┘
              │                        ▼
              │                 Human review queue
              │                        │
              ▼                        ▼
        KG upsert  ◀──────────  Labels ─▶ RETRAIN ENCODER (distillation loop)
```

### Recommended stack

| Component | ⭐ Default | Alternative | Flip when |
|---|---|---|---|
| **Encoder** | ⭐ **DeBERTa-v3** (English) / **XLM-RoBERTa** (multilingual) | ModernBERT · spaCy transformer | ModernBERT for 8k context and faster inference; spaCy when you want a batteries-included pipeline over max accuracy |
| **Zero-shot / new types** | ⭐ **GLiNER** | LLM with structured output | **GLiNER is the underrated answer** — a small encoder doing *zero-shot* NER from type names. Bridges the encoder/LLM gap at encoder cost. |
| **LLM tier** | Small instruct model + **constrained decoding** (JSON schema) | Frontier model | Frontier only for genuinely hard/novel types; constrained decoding is non-negotiable or you'll parse free text |
| **Span decoding** | ⭐ **Span-based classifier** | BIO tagging · biaffine | **BIO if entities are strictly flat** (cheaper, simpler); span-based the moment you need nesting (§4.5) |
| **Serving** | **Triton** (dynamic batching) | TorchServe · Ray Serve | Triton for GPU batching at this volume |
| **Entity linking** | Dense retrieval over KB aliases + cross-encoder rerank | String match + alias table | Alias tables alone for clean domains (tickers); dense retrieval when surface forms vary |
| **Entity store** | **PostgreSQL** (mentions) + **Neo4j** (graph) | Elasticsearch | ES if the product is search rather than graph traversal |
| **Annotation** | **Label Studio** (OSS) / **Prodigy** | Doccano | Prodigy for active-learning ergonomics |
| **Streaming** | **Kafka** + Flink | Batch Spark | 200M mentions/day is a stream, not a nightly job |

## 4.4 Deep Dive A — Encoder vs LLM, and the distillation loop

| | **Fine-tuned encoder** | **LLM extraction** |
|---|---|---|
| Cost / 10M docs | ~$10/day | ~$750/day |
| Latency | 1–5 ms | 200–2,000 ms |
| Needs labels | Yes (~1–5k spans/type) | No |
| New entity type | Retrain | **Instant** |
| Nested / complex spans | Needs span-based head | Natural |
| Consistency | Deterministic | Varies run to run |
| **Failure mode** | Silently misses unseen patterns | **Hallucinates spans not in the text** |

**The answer is a hybrid with a feedback loop, and the loop is the point:**

1. **LLM bootstraps** a new entity type on day one — no labels needed
2. Humans verify a **sample** of LLM output (not all of it)
3. That verified output becomes **training data for the encoder**
4. The encoder takes over the volume; the LLM drops back to low-confidence cases

> **This is distillation for extraction: use the expensive model to teach the cheap one, then serve the cheap one.** It gives you the LLM's flexibility at the encoder's cost, and it's the reason the §4.1 requirement — a new type in under a week with no full relabel — is achievable.

**Always validate LLM spans against the source text.** An LLM asked to return spans will occasionally return text that does not appear in the document, or offsets that don't align. Reject any span that fails exact substring match, and log the rate — it's a quality signal on your prompt.

## 4.5 Deep Dive B — Span structure: nested, overlapping, discontinuous

**BIO tagging cannot represent nested entities**, and this is a design decision people make by accident. In *"Bank of America"*, the whole string is an ORG while *"America"* is a LOC — BIO forces you to pick one.

| Approach | Handles | Cost |
|---|---|---|
| **BIO / IOBES tagging** | Flat only | Cheapest; use if flat is genuinely enough |
| ⭐ **Span-based classification** | Nested, overlapping | Enumerate candidate spans (length-capped), classify each. O(n²) spans — cap length. |
| **Biaffine** | Nested | Strong accuracy, heavier |
| **MRC framing** | Nested, zero-shot-ish | One forward pass per entity type — expensive at 20 types |
| **Generative (seq2seq / LLM)** | Everything incl. discontinuous | Most flexible, most expensive, needs span validation |

**Recommend span-based** here: nesting is real in financial text (ORG inside ORG, PERSON inside ORG), and the O(n²) cost is bounded by capping span length at ~10 tokens.

**The bug that will bite you: subword-to-character offset misalignment.** Tokenizers split on subwords; downstream needs character offsets into the original document. Every normalization step — lowercasing, whitespace collapsing, unicode NFKC — shifts offsets. **Carry an offset map from the very first parse and test it explicitly**, because the failure is silent and corrupts every highlight downstream.

## 4.6 Deep Dive C — Schema evolution without relabelling

The hard *operational* problem, and the one interviewers probe because it separates people who've run these systems from people who've trained one model.

**The naive answer — relabel the corpus for the new type — is unaffordable.** The workable design:

**1. Modular per-type heads.** One classification head per entity type over a shared encoder. Adding a type adds a head; existing heads are untouched, so no regression on shipped types.

**2. Handle partial annotation correctly — this is the subtle one.** Documents labelled for types A and B contain unlabelled instances of new type C. If you treat every unlabelled token as the negative class `O`, **you actively teach the model that C is not an entity.** Instead mark unannotated regions as **unknown and mask them out of the loss** for heads they weren't annotated for. Getting this wrong is the single most common cause of a new entity type performing badly, and naming it unprompted is a strong signal.

**3. Bootstrap with the LLM** (§4.4), verify a sample, train the new head.

**4. Decide the backfill policy explicitly.** Do you re-extract 5 years of history for the new type? At 10M docs/day that's a serious batch job. Usually: re-extract on demand for documents that get queried, plus a rolling backfill of the recent window. Say that you'd *decide* it rather than silently assuming full reprocessing.

**Entity linking and normalization** is the other half: *"Apple"*, *"Apple Inc."*, *"AAPL"* must resolve to one canonical ID. Alias tables handle the clean cases; dense retrieval over KB entries plus a cross-encoder handles the rest; and ambiguity (*Apple* the company vs the fruit) needs context. Linking errors are worse than extraction errors because they merge two real entities into one.

## 4.7 Deep Dive D — Evaluation for spans

**Token-level F1 is misleading and inflates your numbers.** A 5-token entity where you get 4 tokens right scores 0.8 at token level and **0.0 at span level under exact match**. Downstream consumers need the span, so **report span-level P/R/F1**.

| Scoring | When |
|---|---|
| **Exact match** | Strictest; the right default for KG ingestion |
| **Partial / overlap (MUC-style)** | Report alongside exact to separate boundary errors from missed entities |
| **Type-only** | Diagnostic: did we find the entity but mislabel its type? |

**Separate boundary errors from type errors** — they have different fixes. Boundary errors point at tokenization or span decoding; type errors point at insufficient training data for confusable types.

**Always break down per entity type.** A macro average hides that your rare types are at 0.4 F1 while PERSON carries the aggregate.

**Calibrate the confidence gate** that routes to the LLM and the human queue — that threshold *is* a product decision, and it should be set from the measured precision/recall curve, not chosen by feel.

**Inter-annotator agreement is your ceiling.** If two humans agree only 85% of the time on a type, a model scoring 0.85 is at parity and further tuning is wasted.

## 4.8 Failure modes

| Failure | Why | Mitigation |
|---|---|---|
| **New type performs terribly** | Unlabelled instances trained as negatives | Mask unannotated regions from loss (§4.6) |
| **Offsets drift** | Subword ↔ character misalignment through normalization | Offset map from first parse; explicit tests |
| **LLM hallucinates spans** | Generative model invents text | Reject spans failing exact substring match; log the rate |
| **Domain shift** | Trained on news, deployed on filings | Domain-specific eval sets; continued fine-tuning; monitor confidence drift |
| **Long-tail types starve** | Few examples for rare types | Active learning targeted at low-confidence rare types; GLiNER fallback |
| **Entity linking merges two real entities** | Alias collision | Ambiguity detection; require context agreement; human review on merges |
| **Annotation drift** | Guidelines interpreted differently over time | Periodic IAA checks; a frozen gold set |
| **Silent recall loss** | Model stops finding a pattern after retrain | Per-type regression suite in CI |

## 4.9 Evaluation

Gold set of a few thousand documents, **stratified by entity type, language, and document source**, with IAA measured. Report span-level exact and partial P/R/F1 per type, plus:
- **Encoder vs LLM agreement rate** — a cheap continuous monitor; divergence signals drift
- **Confidence calibration** — does 0.9 confidence mean 90% precision?
- **Escalation rate** to LLM and to humans — rising escalation means the encoder is decaying
- **Downstream KG precision** — sample entities that reached the graph and audit them; this is the metric that actually matters

**In CI:** per-type regression on every model, prompt, or schema change.

## 4.10 Cost and latency

| Lever | Impact |
|---|---|
| ⭐ **Encoder carries volume, LLM only on low confidence** | ~100× on the 99% |
| ⭐ **Gazetteer tier first** | Free precision on known entities; removes them from model load |
| **Dynamic batching (Triton)** | Large — encoder inference is embarrassingly batchable |
| **Cap span length** | Bounds the O(n²) span enumeration |
| **Distill LLM → encoder** (§4.4) | Converts recurring LLM spend into a one-time labelling cost |
| **Quantize the encoder (INT8)** | ~2× throughput, negligible quality loss at this model size |

Real-time path budget (200 ms): parse 20 ms · encoder 15 ms · linking 40 ms · gate + write 25 ms · overhead 40 ms ≈ **140 ms**, with headroom for an LLM escalation on a small fraction.

## 4.11 Corner Questions

**Q: Product wants a new entity type next week. Do you relabel 100k documents?**
> No — and the design should make that unnecessary. Bootstrap the type with an LLM or GLiNER on day one, which needs zero labels; have humans verify a sample rather than the corpus; train a *new head* on the shared encoder so existing types are untouched; and mask unannotated regions from the loss so old documents don't teach the model that the new type is a non-entity. That last point is the one people miss — treating unlabelled as negative is why new types usually underperform. Then decide the backfill policy explicitly: rolling backfill of the recent window plus re-extraction on demand, rather than reprocessing five years of history.

**Q: Token-level F1 is 0.94 but downstream users say extraction is bad. Explain.**
> Token-level F1 is the wrong metric. Getting 4 of 5 tokens in an entity scores 0.8 at token level but **0.0 at span level under exact match**, and downstream consumes spans. So I'd report span-level exact and partial F1, per entity type. Partial-vs-exact tells you whether these are boundary errors or missed entities, and the per-type breakdown usually reveals that one high-frequency type is carrying the average while rare types sit near 0.4. I'd also check offset alignment, because subword-to-character drift produces exactly this pattern — the right entity, one character off, useless downstream.

**Q: Your LLM returns an entity that doesn't appear in the document.**
> Expected behaviour from a generative model, and it must be handled structurally rather than by prompting harder. Every returned span is validated by exact substring match against the source; anything that fails is dropped, and I'd track the rejection rate as a quality signal on the prompt. I'd also use constrained decoding against a JSON schema so the output shape is guaranteed, and require the model to return offsets that I then verify rather than trusting them. The general principle is the same as everywhere else in these designs: the model proposes, deterministic code validates.

**Q: LLM extraction would cost $274k/year. Justify it or cut it.**
> I wouldn't run the LLM on the full volume at all — that's the 100× ratio, and it's the whole reason for the tiered design. The encoder handles ~99% at roughly $10/day; the LLM sees only low-confidence cases and genuinely new types, which is maybe 1% of traffic, bringing that $274k to a few thousand. And the LLM spend is *transitional*: its output becomes training data, the encoder absorbs the new type, and the escalation rate falls. So the honest framing is that the LLM is a labelling and bootstrapping cost, not a serving cost.

**Q: The model works on news but fails on legal filings.**
> Domain shift, and the first thing I'd do is quantify it rather than assume — a domain-specific gold set, then compare per-type F1 and confidence distributions. Falling confidence with stable F1 means calibration drift; falling both means genuine shift. Fixes in increasing cost: add domain examples via LLM bootstrapping and active learning on low-confidence spans, continued fine-tuning of the encoder on in-domain data, and separate heads or separate models per domain if the domains are truly divergent. I'd also check tokenization — legal text has formatting that can break offset mapping in ways that look like model failure but aren't.

**Q: 50 languages. One model or fifty?**
> One multilingual encoder — XLM-RoBERTa or similar — as the default, because per-language models multiply training, serving, and monitoring cost fifty-fold and low-resource languages have too little data to train alone. Multilingual models also transfer: labelled English data measurably improves low-resource performance. I'd evaluate per language rather than in aggregate, since the aggregate will be dominated by high-resource languages, and I'd add per-language heads or targeted fine-tuning only where the evaluation shows a specific language failing. Your MBART-50 multilingual benchmarking is the directly relevant experience here.

## 4.12 Mid-level vs Senior

| | Mid-level | **Senior** |
|---|---|---|
| Model choice | "Fine-tune BERT" or "use an LLM" | Tiered by the **100× cost ratio**, with a distillation loop |
| Spans | BIO tagging | Span-based for nesting; knows when BIO is enough |
| New types | "Retrain the model" | Modular heads + **masked loss on unannotated regions** |
| Metrics | Token F1 | Span-level exact + partial, per type, IAA as ceiling |
| LLM output | Trusts it | Validates spans against source text |
| Linking | Not mentioned | Normalization, ambiguity, and why merge errors are worst |

## 4.13 References for this case study

**Read first**
- ⭐ **GLiNER: Generalist Model for NER using Bidirectional Transformer** — Zaratiana et al., 2023 (arXiv 2311.08526) — **zero-shot NER at encoder cost.** The most practically useful recent paper for this design.
- **A Unified MRC Framework for Named Entity Recognition** — Li et al., 2019 (1910.11476) — nested NER via question answering
- **Named Entity Recognition as Dependency Parsing (biaffine)** — Yu et al., 2020 (2005.07150) — the strong nested-NER baseline

**Models & methods**
- **BERT** (1810.04805) · **DeBERTa-v3** (2111.09543) · **XLM-RoBERTa** (1911.02116) — the encoder lineage
- **ModernBERT** — Warner et al., 2024 (2412.13663) — modern long-context encoder
- **UniversalNER: Targeted Distillation from LLMs** — Zhou et al., 2023 (2308.03279) — **the §4.4 distillation loop, formalized**
- **Autoregressive Structured Prediction with Language Models** — 2210.14698 — generative extraction

**Linking & evaluation**
- **BLINK: Zero-shot Entity Linking with Dense Entity Retrieval** — Wu et al., 2019 (1911.03814)
- **CoNLL-2003** · **OntoNotes 5.0** · **Few-NERD** (2105.07464) — the standard benchmarks and their scoring conventions
- **MultiCoNER** (2208.14536) — multilingual, complex, low-context NER

**Code**
- ⭐ `urchade/GLiNER` (~2k) — zero-shot NER, small enough to read
- `explosion/spaCy` (~31k) — production NLP pipelines; read the span and offset handling
- `flairNLP/flair` (~14k) — strong sequence-labelling baselines
- `HumanSignal/label-studio` (~21k) — annotation, active learning integration
- `huggingface/transformers` (~145k) — `token-classification` examples

---

# Case Study 5 — Fraud Detection

> *"Design a real-time fraud detection system for card payments."*

The defining feature is that **you are optimizing money, not F1** — and that **an adversary adapts to whatever you deploy.** Two properties no other case study here has.

## 5.1 Requirements

**Ask:**
- "Payment fraud, account takeover, or promo abuse?" *(Different labels, different features.)*
- "Real-time block, or async review?" *(Real-time puts you in the payment path.)*
- "What's the cost of a false positive versus a false negative?" *(This sets the threshold — ask it explicitly.)*
- "Regulatory constraints on explaining declines?"

**Assume:** card payments, **10M transactions/day**, real-time decision required, three outcomes (allow / review / block), chargebacks arrive **30–90 days later**, adverse-action explanations legally required.

**Non-functional:**
- **p99 < 100 ms** — you are inline in the payment path; latency is lost revenue
- Base fraud rate **~0.1%**
- **Every decline must be explainable** to a regulator and a customer
- Review queue capacity is **fixed by headcount** — a hard constraint on the design

**Descope:** the payment rails, chargeback dispute workflow, KYC/onboarding.

> **Open by refusing the wrong objective.** "Maximize F1" is not the goal — the two error types have wildly different dollar costs, and they're asymmetric in *time* too (a false positive annoys a customer now; a false negative costs money in 60 days). **The model produces a score; a separate decision layer converts score to action using expected cost.** Separating those two things in the first three minutes is the strongest opening available here.

## 5.2 Estimation

```
10M txn/day ÷ 86,400 s   ≈ 116 TPS average
peak (evenings, paydays)  ≈ 500 TPS   → Black Friday ~10× → 1,200 TPS
Fraud at 0.1%             = 10,000 fraudulent txns/day
```

**The cost model — do this arithmetic out loud, it's the whole design:**
```
avg transaction            $50
chargeback cost            $50 + $25 fee = $75   ← false negative
false positive cost        lost margin (~$5) + LTV damage (~$50?) 
```
So a missed fraud costs ~$75 and a wrongly blocked customer costs somewhere around $55 — **the same order of magnitude.** That's why you cannot simply "block anything suspicious": at 0.1% base rate, a model with 90% recall and 10% precision blocks 90,000 good customers to catch 9,000 frauds. **The precision at your operating point is everything, and AUC tells you almost nothing about it.**

**Review capacity is a hard constraint:**
```
50 analysts × 200 reviews/day = 10,000 reviews/day = 0.1% of volume
```
So the review band must be sized to exactly that. **The threshold is set by staffing, not by the ROC curve** — a genuinely practical point most candidates miss.

## 5.3 Architecture

```
   Transaction
        │
        ▼
  ┌──────────────────────────────────────────┐
  │ FEATURE ASSEMBLY  (<40 ms)               │
  │  • velocity counters (Redis, Flink-fed)  │  card/device/IP/email
  │    1m · 1h · 24h · 7d windows            │
  │  • entity-graph aggregates               │  shared device, card rings
  │  • account history, merchant risk        │
  └──────────────┬───────────────────────────┘
                 ▼
  ┌──────────────────────┐    ┌────────────────────────┐
  │  GBDT MODEL  (~10ms) │    │  RULES ENGINE          │
  │  score ∈ [0,1]       │    │  hard blocks, new      │
  │  + SHAP reasons      │    │  attack patterns       │
  └──────────┬───────────┘    └───────────┬────────────┘
             └────────────┬───────────────┘
                          ▼
             ┌────────────────────────────┐
             │  DECISION LAYER            │  ← expected-cost thresholds
             │  allow / review / block    │     segment-specific
             └─────┬──────────┬───────────┘
                   │          │
              review queue    block + adverse-action reason
                   │                    │
                   ▼                    ▼
          analyst labels ─────▶ TRAINING DATA ◀── chargebacks (30–90d later)
                   │
                   └──▶ RANDOMIZED ALLOW-THROUGH (§5.4) ──▶ unbiased labels
```

### Recommended stack

| Component | ⭐ Default | Alternative | Flip when |
|---|---|---|---|
| **Model** | ⭐ **GBDT (LightGBM / XGBoost)** | Deep tabular · GNN | **GBDT is genuinely the right answer here** — tabular data, best-in-class accuracy, millisecond inference, and SHAP explanations that satisfy adverse-action requirements. Go deep only if you have sequence data that trees can't use. |
| **Graph features** | ⭐ **Precomputed graph aggregates** in the feature store | **Neo4j / TigerGraph** live traversal · GNN embeddings | Live traversal blows the 100 ms budget; precompute ring-level features offline and serve them as columns. GNN only if fraud rings are the dominant pattern and you have the team. |
| **Velocity counters** | ⭐ **Flink** → **Redis** | Kafka Streams · Redis-only | Must be true real-time — card-testing bursts happen in *seconds*, so micro-batch is too slow |
| **Feature store** | **Feast** / in-house on Redis | — | Point-in-time correctness is critical (§5.5) |
| **Rules engine** | ⭐ **Dedicated rules service** (OPA / Drools / in-house DSL) | Hardcode in the model service | **Always separate.** You need to ship a rule in *hours* when a new attack lands; a model retrain takes days. |
| **Explainability** | ⭐ **SHAP** (TreeSHAP — exact and fast on GBDT) | LIME · counterfactuals | TreeSHAP is milliseconds on trees, which is why GBDT wins here |
| **Serving** | In-process model (no network hop) | Triton / Seldon | At <100 ms with tree models, embed it — a network hop is a meaningful fraction of the budget |
| **Drift monitoring** | **PSI / KL on feature distributions** + score drift | Evidently · Arize | Managed if you want segment analysis out of the box |
| **Case management** | Build or buy analyst tooling | — | The review queue is part of the *system*, not an afterthought |

## 5.4 Deep Dive A — Labels: delay, selection, and what a "negative" means

**Three distinct label problems, and naming all three is the differentiator.**

**1. Delay.** Chargebacks arrive 30–90 days after the transaction. So a model trained today sees a *fully matured* label set that is 3 months old — and fraud patterns move faster than that.

| Strategy | Trade-off |
|---|---|
| Train only on matured labels | Correct but 3 months stale |
| ⭐ **Two-speed: matured labels for the model, fast proxies for rules and monitoring** | The practical answer |
| Fast proxies: analyst decisions, 3DS challenge outcomes, customer reports | Days not months, but biased |
| Model the delay explicitly (survival analysis) | Principled, more complex |

**2. Selective labelling — the big one.** **You never learn whether a blocked transaction was actually fraud.** You blocked it; there's no outcome. So your training data is conditioned on the current model's decisions, and the model can never discover that its blocks were wrong.

> This is the fraud analogue of the recommender feedback loop (§1.7), and it has the same fix: **let a small random fraction of would-be-blocks through** and observe what happens. It is genuinely expensive — you are knowingly allowing some fraud — but it is **the only unbiased signal about your own precision**, and the cost is bounded and computable. Proposing it, with the cost quantified, is a strong senior answer. The cheaper partial alternative is **reject inference**: model the blocked population's likely outcomes rather than observing them.

**3. Unlabelled ≠ negative.** A transaction with no chargeback might be fraud that was never disputed. So your "negatives" are really *unlabelled*, which is a **positive-unlabelled (PU) learning** setup. In practice: treat the negative class as noisy, don't over-trust precision estimates, and calibrate against the small clean set you get from randomized allow-through.

**On imbalance (0.1%):** don't reflexively SMOTE. Class weights or scale_pos_weight in GBDT, evaluate with **AUC-PR not AUC-ROC** (ROC is near-useless at this base rate), and remember that you're ranking-and-thresholding, not classifying — so calibration matters more than balance.

## 5.5 Deep Dive B — Features: velocity, graph, and leakage

**Velocity features carry most of the signal.** Counts and sums over sliding windows, per entity:
```
card:    txns in 1m / 1h / 24h / 7d, distinct merchants, distinct countries, amount sum
device:  distinct cards, distinct accounts, failed attempts
IP/ASN:  distinct accounts, geo-velocity (impossible travel)
email:   account age, domain risk, distinct cards
merchant: historical fraud rate, chargeback rate
```
**Card testing** — an attacker validating stolen cards with many tiny transactions — is caught almost entirely by 1-minute velocity. That's why the counter path must be Flink-fast, not micro-batch.

**Graph features** catch rings that per-entity features miss: accounts sharing a device, cards sharing a shipping address, components in the device↔card bipartite graph. **Precompute these offline as columns** — live graph traversal doesn't fit in 100 ms.

**Leakage traps, and they're brutal here:**
- Using **settlement or chargeback status** as a feature — it doesn't exist at decision time
- Computing merchant fraud rate over the **full history including the future**
- Building velocity counters from the **complete dataset** rather than as-of transaction time

**Every feature must be computed point-in-time, as of the transaction timestamp.** In fraud this is worse than elsewhere because leakage produces spectacular offline metrics — AUC 0.99 — that collapse in production. If your offline AUC looks too good, assume leakage before assuming success.

## 5.6 Deep Dive C — The adversary

**Unlike every other system in this document, the data distribution is actively hostile.** Fraudsters probe your boundaries, observe which transactions succeed, and adapt within days.

**Consequences for the design:**

| Implication | Response |
|---|---|
| Models decay in weeks, not months | Retrain weekly or faster; monitor score drift daily |
| New attacks need response in hours | **Rules engine alongside the model** — you cannot wait for a retrain cycle |
| Decline reasons leak information | Give the customer/regulator a *reason*, but keep it coarse — don't tell an attacker exactly which feature tripped |
| Attackers probe with small transactions | Detect *probing patterns* (many small txns, rising amounts) as a signal in itself |
| Your best features get evaded | Ensemble diverse signal families; don't let one feature dominate |

**Champion/challenger permanently.** Run the incumbent and a candidate in parallel on live traffic, with the challenger scoring in shadow. Given adversarial drift, you always want a fresher model warming up.

## 5.7 Deep Dive D — The decision layer

**The model outputs a score. The decision layer turns scores into money.** Keeping these separate is what lets you retune business policy without retraining.

**Three zones, not two:**
```
score < t_low          → allow
t_low ≤ score < t_high → review queue   (sized to analyst capacity)
score ≥ t_high         → block + adverse-action reason
```

**Set thresholds by expected cost, not by F1:**
```
E[cost | block] = P(legit | score) × (lost margin + LTV damage)
E[cost | allow] = P(fraud | score) × (chargeback + fee)
block when E[cost | allow] > E[cost | block]
```
This requires **calibrated probabilities**, which is why calibration matters more than raw AUC — an uncalibrated score can't enter a cost equation.

**Segment the thresholds.** A $10,000 transaction and a $5 transaction should not share a threshold; nor should a 5-year customer and an account created an hour ago. Expected-cost framing handles this naturally because the dollar terms differ.

**Adverse-action explanations.** Regulated declines need a reason. TreeSHAP gives per-decision feature attributions in milliseconds, which you map to a small set of human-readable reason codes. **This requirement alone is a strong argument for GBDT over deep models** — say so.

## 5.8 Failure modes

| Failure | Why | Mitigation |
|---|---|---|
| **Blocks never get labelled** | Selective labelling | Randomized allow-through; reject inference (§5.4) |
| **Offline AUC 0.99, production terrible** | Leakage from post-transaction features | Strict point-in-time features; distrust suspiciously good metrics |
| **Model decays in weeks** | Adversarial adaptation | Weekly retrain; drift monitors; champion/challenger |
| **New attack, model blind for days** | Retrain cycle too slow | Rules engine for hours-scale response |
| **Card testing missed** | Velocity counters lag | True streaming (Flink), sub-minute windows |
| **Black Friday false-positive spike** | Volume and behaviour shift look anomalous | Seasonality features; segment thresholds; pre-scheduled threshold relaxation |
| **Review queue overflows** | Threshold set without capacity | Size the review band to analyst headcount; dynamic threshold on queue depth |
| **Disparate impact across a protected group** | Proxy features | Fairness audit by segment; document it — this is a regulatory exposure |
| **Hot graph node** | One shared IP/device with millions of edges | Cap degree; exclude hub nodes from ring features |
| **Feedback loop on analyst labels** | Analysts only see what the model flags | Sample random transactions into the queue too |

## 5.9 Evaluation

**AUC-PR, never AUC-ROC alone** — at 0.1% base rate ROC is flattered by the enormous true-negative mass.

**Report at the operating point, and in dollars:**
- **Precision and recall at the actual thresholds**, not the best point on the curve
- **Precision@k where k = review capacity** — the only precision the analysts experience
- **Net dollars saved** = (fraud prevented × avg loss) − (false positives × FP cost) − review cost. This is the metric the business cares about, and quoting it is a strong differentiator.
- **Segment breakdown** — by amount band, account age, geography, merchant category

**Backtest on matured labels only**, with a temporal split — never random, which leaks future fraud patterns backwards.

**Shadow-mode every candidate model** before it decides anything, and compare decisions against the incumbent to see exactly which transactions change.

## 5.10 Cost and latency

**100 ms budget:**
```
Feature assembly (Redis velocity + graph aggregates)   40 ms   ← dominant
GBDT scoring (in-process)                              10 ms
SHAP reason codes                                       5 ms
Rules evaluation                                        5 ms
Decision + logging                                      5 ms
Network / overhead                                     30 ms
                                                  ─────────
                                                       95 ms
```
**Feature assembly dominates**, so that's where optimization goes: parallel fetches, co-locate the feature store, precompute anything graph-shaped, and keep the model in-process to avoid a network hop.

**The economics are unusual:** compute cost is trivial next to fraud losses. A GBDT scoring 10M transactions/day costs a few hundred dollars a month; the fraud it prevents is measured in millions. **So this is one of the few ML systems where you optimize for accuracy and latency, essentially ignoring inference cost** — and saying that explicitly shows you understand the business context.

## 5.11 Corner Questions

**Q: You block a transaction. How do you ever learn whether you were right?**
> You don't — and that's the central data problem, not a detail. Blocked transactions have no outcome, so training data is conditioned on the current model's decisions and it can never discover its own false positives. The only genuinely unbiased fix is to let a small random fraction of would-be-blocks through and observe the outcome; it costs real fraud losses, but the cost is bounded and computable, and it's the only clean estimate of your precision you will ever have. Cheaper partial substitutes: reject inference to model the blocked population, customer complaints and successful appeals as weak positive-error signal, and analyst review of a sample of blocks. I'd budget for the randomized slice explicitly and treat it as measurement infrastructure.

**Q: Chargebacks take 60 days. How do you train a model today?**
> Two speeds. The model trains on fully matured labels, accepting that they're roughly three months old — that's fine for the stable bulk of fraud patterns. Everything that needs to move faster runs on proxies: analyst decisions, 3DS outcomes, and customer reports give signal in days, and the rules engine responds in hours. I'd also monitor label maturity explicitly, so recent periods are known to be incomplete and nobody mistakes an immature window for a drop in fraud. The mistake to avoid is treating "no chargeback yet" as a confirmed negative — it's unlabelled, not clean.

**Q: Fraud is 0.1% of transactions. Should you oversample or SMOTE?**
> Usually not. SMOTE on high-dimensional tabular data with categorical features tends to synthesize implausible transactions and mostly hurts. I'd use class weights in the GBDT, evaluate with AUC-PR rather than AUC-ROC since ROC is meaningless at this base rate, and remember the task is ranking-then-thresholding rather than balanced classification — so calibration matters much more than balance. If a specific rare fraud type is starved, I'd rather collect or upweight real examples of it than synthesize neighbours.

**Q: Your model has AUC 0.98 but fraud losses went up this quarter. What happened?**
> AUC is a global ranking metric and you operate at one threshold, so it can look excellent while precision at your actual cut-off degrades. Candidates, in order: adversarial drift — attackers adapted and the new pattern sits in a region the model ranks low; the fraud mix changed toward a type you under-represent; thresholds weren't retuned as the base rate moved; or leakage was inflating the offline number and the real model is weaker than measured. I'd look at precision and recall at the operating point segmented by fraud type and by amount band, check score-distribution drift, and compare against the randomized allow-through slice, which is the only unbiased read available.

**Q: A new attack pattern appears on Monday morning. What happens?**
> The rules engine, not the model — that's why they're separate systems. A retrain plus validation is days; a rule is hours, and during an active attack that difference is the entire loss. So: analysts identify the pattern, a targeted rule ships the same day to block or route to review, and the model absorbs it on the next retrain once labels mature. I'd also want probing detection — many small transactions with rising amounts across shared entities — because attackers usually test before they scale, and catching the test phase buys you the day you need.

**Q: Explain a decline to a regulator.**
> TreeSHAP on the GBDT gives exact per-decision feature attributions in milliseconds, which I map to a small fixed set of human-readable reason codes rather than exposing raw feature names. This requirement is a genuine argument for tree models over deep ones, and I'd raise it during model selection rather than discovering it at compliance review. Two nuances: the reason given to the customer should be coarse enough not to teach an attacker which feature tripped, and I'd retain the full attribution in an audit log even though only the coarse code is surfaced. I'd also run a disparate-impact analysis by protected segment, because that's the question that follows.

**Q: Black Friday. Volume is 10× and the model flags everything.**
> Expected — the behavioural distribution genuinely shifts, so transactions that look anomalous against a normal Tuesday are normal for the day. Mitigations: seasonality and day-of-week features so the model has context; historical peak periods in the training data rather than only ordinary weeks; pre-scheduled threshold relaxation for known peaks, decided in advance with the business; and dynamic thresholds tied to review-queue depth so you degrade gracefully instead of overflowing the analysts. I'd also pre-agree the risk appetite for the day, because the right answer is a business decision about how much extra fraud to accept to avoid blocking legitimate holiday spend.

**Q: An attacker tests 50 stolen cards in two minutes.**
> Card testing, and it's caught almost entirely by short-window velocity — distinct cards per device, per IP, per session in a 1-minute window, plus the characteristic pattern of small amounts and high decline rates. This is precisely why the counter pipeline has to be true streaming rather than micro-batch: a five-minute lag means the attack completes before the feature updates. I'd also add a graph feature for the device-to-card fan-out, and treat a burst of declines from one entity as a signal in its own right rather than just a set of independent transactions.

## 5.12 Mid-level vs Senior

| | Mid-level | **Senior** |
|---|---|---|
| Objective | "Maximize F1" | Expected-cost thresholds; **net dollars saved** |
| Model | "Try a neural net" | GBDT — accuracy, latency, **and SHAP for adverse action** |
| Labels | "Use chargebacks" | Delay, **selective labelling**, PU framing — all three |
| Blocks | Not mentioned | **Randomized allow-through** as the only unbiased signal |
| Metrics | AUC | AUC-PR, precision@review-capacity, segment breakdown |
| Adversary | Not mentioned | Weekly retrain, rules engine for hours-scale response |
| Decision | One threshold | Three zones; review band sized to **analyst headcount** |
| Leakage | "Be careful" | Point-in-time features; distrusts AUC 0.99 |

## 5.13 References for this case study

**Read first**
- ⭐ **Stripe Radar** engineering posts and **Feedzai** / **Sift** technical blogs — the best public writing on production fraud systems, including threshold economics
- ⭐ **Reject Inference / selective labels** — Lakkaraju et al., *The Selective Labels Problem* (KDD 2017) — the §5.4 problem, formalized. Essential.
- **Learning from Positive and Unlabeled Data: A Survey** — Bekker & Davis, 2018 (1811.04820) — why your negatives aren't negatives

**Modelling**
- **XGBoost** — Chen & Guestrin, 2016 (1603.02754) · **LightGBM** — Ke et al., NeurIPS 2017
- **A Unified Approach to Interpreting Model Predictions (SHAP)** — Lundberg & Lee, 2017 (1705.07874) and **TreeSHAP** (1802.03888) — the adverse-action path
- **Why do tree-based models still outperform deep learning on tabular data?** — Grinsztajn et al., 2022 (2207.08815) — cite this when justifying GBDT
- **Calibration of Modern Neural Networks** — Guo et al., 2017 (1706.04599) — why calibration matters for cost-based thresholds

**Imbalance, delay, graphs**
- **Learning from Imbalanced Data** — He & Garcia, 2009 — the classic survey
- **Deep Learning for Anomaly Detection: A Review** — Pang et al., 2020 (2007.02500)
- **Graph Neural Networks for Fraud Detection** — e.g. **CARE-GNN** (2008.08692) — for camouflaged fraudsters in entity graphs
- **Delayed feedback modelling** — Chapelle, *Modeling Delayed Feedback in Display Advertising* (KDD 2014) — the ads analogue of chargeback delay

**Code**
- `microsoft/LightGBM` (~17k) · `dmlc/xgboost` (~27k)
- ⭐ `shap/shap` (~23k) — TreeSHAP for reason codes
- `safe-graph/DGFraud` — graph-based fraud detection implementations
- `scikit-learn-contrib/imbalanced-learn` (~7k) — resampling, and its documented caveats

---

# Case Study 6 — LLM Fine-tuning Platform

> *"Thirty teams want to fine-tune LLMs. Design the platform."*

The most valuable thing this design does is **say no correctly**. Most fine-tuning requests should not be fine-tuning, and a platform that makes it easy to do the wrong thing is worse than no platform.

## 6.1 Requirements

**Ask:**
- "What are teams actually trying to achieve?" *(Usually the real answer is RAG or a better prompt.)*
- "Which base models, and are open weights acceptable?"
- "Does the platform also *serve* the tuned models?" *(Serving N adapters is where the economics live.)*
- "Sensitive training data / tenant isolation requirements?"

**Assume:** internal platform, **30 teams**, open-weight models **1B–70B**, LoRA-first, platform serves the results, sensitive data with tenant isolation required.

**Non-functional:**
- **Dataset → served adapter in under 4 hours** (self-service; if it's slower teams route around you)
- **GPU utilization > 60%** — the fleet is the cost
- **Fully reproducible** — same data + config + seed → same adapter
- **No cross-tenant data leakage**, ever

**Descope:** pretraining from scratch, the inference platform itself (that's Case Study 3).

## 6.2 Estimation — the memory math that decides everything

**Full fine-tuning memory:**
```
weights (fp16)        2 bytes/param
gradients (fp16)      2 bytes/param
Adam states (fp32 m,v) 8 bytes/param
master weights (fp32)  4 bytes/param
                    ─────────────────
                    ≈ 16 bytes/param
```
| Model | Full FT | LoRA (fp16 base) | QLoRA (4-bit base) |
|---|---|---|---|
| **1B** | ~16 GB | ~3 GB | ~1.5 GB |
| **8B** | **~128 GB** → multi-GPU | **~18 GB** → 1× A100 | ~7 GB → consumer GPU |
| **70B** | ~1.1 TB → 16+ GPUs | ~150 GB → 2× H100 | **~45 GB → 1× A100** |

> **That table is the entire argument for PEFT, and you should be able to derive it live.** Full fine-tuning an 8B model needs ~128 GB — more than one 80 GB GPU — purely because of optimizer state. LoRA freezes the base, so gradients and optimizer states exist only for the tiny adapter, and the same job fits on one GPU. **QLoRA quantizes the frozen base to 4-bit and puts a 70B fine-tune on a single A100.**

**Platform load:**
```
30 teams × ~4 tunes/month = 120 jobs/month
avg LoRA job on 8B, 10k examples ≈ 4 GPU-hours
→ ~480 GPU-hours/month ≈ 0.7 GPUs continuously
```
**The training fleet is small.** Which means the interesting cost is not training — it's **serving 30 different fine-tuned models**, and that's §6.10.

## 6.3 Architecture

```
  Team submits: dataset + base model + config
        │
        ▼
  ┌──────────────────────────────────────┐
  │ DATA VALIDATION (the real gate)      │
  │  schema · dedup · decontamination    │  ← reject here, not after training
  │  PII scan · template check · split   │
  └──────────────┬───────────────────────┘
                 ▼
  ┌──────────────────────────────────────┐
  │ JOB QUEUE  (quota, priority, GPU fit)│
  └──────────────┬───────────────────────┘
                 ▼
  ┌──────────────────────────────────────┐
  │ TRAINING   LoRA / QLoRA / full FT    │
  │  FSDP or DeepSpeed ZeRO if multi-GPU │
  │  checkpoint → resume on preemption   │
  └──────────────┬───────────────────────┘
                 ▼
  ┌──────────────────────────────────────┐
  │ EVAL GATE  (automatic, blocking)     │
  │  ① beats few-shot prompting baseline?│ ← if no, DO NOT SHIP
  │  ② general capability retained?      │ ← catastrophic forgetting
  │  ③ safety evals pass?                │ ← alignment regression
  └──────────────┬───────────────────────┘
                 ▼
  ADAPTER REGISTRY  (versioned: adapter + base + data + config + metrics)
                 ▼
  MULTI-ADAPTER SERVING  — one base model, N LoRA adapters, one GPU
```

### Recommended stack

| Component | ⭐ Default | Alternative | Flip when |
|---|---|---|---|
| **Training framework** | ⭐ **HF TRL** + **PEFT** | **Axolotl** · **LLaMA-Factory** | Axolotl/LLaMA-Factory for config-driven self-service — teams write YAML, not code. Strong argument for a platform. |
| **Tuning method** | ⭐ **LoRA** (r=16, all-linear) | **QLoRA** · full FT | QLoRA when memory-bound (70B on one GPU); **full FT almost never** — reserve it for genuine domain shift with lots of data |
| **Multi-GPU** | **FSDP** | DeepSpeed ZeRO-3 | Either; FSDP is native PyTorch and simpler to debug |
| **Scheduler** | **Kubernetes + Kueue** or **Ray** | Slurm | Slurm if you already have an HPC cluster |
| **Experiment tracking** | ⭐ **MLflow** or **W&B** | — | Must capture data hash + config + seed for reproducibility |
| **Adapter registry** | Object store + metadata in **Postgres** | HF Hub (private) | Version the *triple*: adapter + base + data snapshot |
| **Serving** | ⭐ **vLLM multi-LoRA** | **S-LoRA** · separate deployments | **Never one deployment per adapter** — see §6.10 |
| **Eval** | The eval harness from `07-agentic-system-design.md` §4 | lm-eval-harness | Reuse; don't build a second eval system |
| **Data validation** | Custom + **Presidio** (PII) | — | Blocking gate, not a warning |
| **Preference tuning / RL** | **DPO** via TRL | ORPO · KTO · **GRPO + RLVR** · PPO | ORPO to skip the separate SFT stage; **GRPO + RLVR when the task has a programmatic checker** (extraction, SQL, code) — cheap, and a one-GPU LoRA recipe exists (`04` §7.6, §8.6); **PPO-RLHF only with a strong reward model and a team to babysit it** |

## 6.4 Deep Dive A — The decision tree (say no correctly)

**Most fine-tuning requests are the wrong tool.** The platform's first job is a triage flow:

```
Need the model to know NEW FACTS?          → RAG.  Not fine-tuning.
Need a specific FORMAT / STYLE / TASK?     → Fine-tune.
Need slightly better output?               → Better prompt / few-shot first.
Need a whole new DOMAIN or LANGUAGE?       → Continued pretraining (expensive).
Need lower COST/LATENCY at equal quality?  → Fine-tune a small model. ← best ROI
```

> **The rule to state: fine-tuning teaches *form*, RAG teaches *facts*.** Trying to inject knowledge by fine-tuning is the single most common mistake — it's unreliable (the model blends facts rather than retrieving them), unupdatable (new facts need a retrain), and unciteable (no source attribution). When someone says "we want the model to know our documentation," the answer is RAG.

**The genuinely great use case:** fine-tune a small model to match a frontier model on *one narrow task*. An 8B model tuned on 10k good examples can match a frontier model on a classification or extraction task at ~1/50th the cost and a fraction of the latency. **That's where fine-tuning pays for itself**, and the platform should actively steer teams toward it.

**Mandatory baseline:** every job must beat **few-shot prompting on the same task** in the eval gate. If it doesn't, the platform refuses to ship it. That one rule prevents most wasted spend.

## 6.5 Deep Dive B — PEFT choices

**LoRA:** freeze the base; learn low-rank matrices `A` (r×d) and `B` (d×r) so `ΔW = BA` (d×d). Only A and B get gradients and optimizer states — hence §6.2.

| Knob | Guidance |
|---|---|
| **Rank r** | 8–16 for style/format; 32–64 for harder skills. Higher rank rarely helps and overfits small datasets. |
| **Alpha** | Commonly 2×r; it's a scaling factor — tune LR instead of fiddling with both |
| **Target modules** | Classic is `q_proj, v_proj`; **all-linear usually works better** for a small extra cost |
| **Dropout** | 0.05–0.1 on small datasets |

**QLoRA:** 4-bit NF4-quantized frozen base + LoRA adapters, with paged optimizers to survive memory spikes. **The biggest single memory win available** — and quality loss versus LoRA is small in practice, which is why it's the default when memory-bound rather than a compromise.

**When full fine-tuning is actually right:** genuine domain shift (a new language, or a very different modality of text), lots of data (100k+ examples), and you can afford the compute and the serving cost of a full separate model. **It is rarely right**, and a platform that makes it the default will burn its fleet.

## 6.6 Deep Dive C — Data curation is the bottleneck

**The #1 cause of failed fine-tunes is data, not hyperparameters.** Teams arrive expecting to tune learning rates; they should be cleaning their dataset.

**Quality beats quantity, decisively.** The LIMA result — 1,000 carefully curated examples outperforming far larger noisy sets — is worth citing by name. The platform should make it easy to inspect and hard to dump.

**The validation gate, all blocking:**
| Check | Why |
|---|---|
| **Schema + template conformance** | The prompt template at training must match serving exactly |
| **Exact + near-dedup (MinHash)** | Duplicates cause memorization and inflate eval |
| **Decontamination vs eval sets** | Otherwise your eval is meaningless |
| **PII scan** (Presidio) | The model will memorize and emit it |
| **Length distribution** | Truncation silently destroys long examples |
| **Task/class balance** | Skew produces a model that only does the majority case |
| **Train/eval split with no leakage** | Split by document or entity, not by row |

> **The failure that's silent and expensive: prompt template mismatch.** If training formats examples one way and serving formats them another — a different system prompt, different special tokens, a missing chat template — the model degrades badly and the cause is invisible in the metrics. **The platform should own templating end to end**, generating the serving config from the training config so they cannot diverge.

**Synthetic data** is legitimate — generate with a frontier model, filter aggressively, and have humans verify a sample. It's the same distillation loop as §4.4.

## 6.7 Deep Dive D — Forgetting and alignment regression

**Catastrophic forgetting:** narrow fine-tuning degrades general capability. Mitigations in order of practicality: **LoRA is inherently less destructive than full FT** (the base is frozen); mix 5–20% general instruction data into the training set as replay; use a lower learning rate and **fewer epochs — often 1–3, not 10**; and always evaluate general benchmarks alongside the task metric.

**Safety regression is the one people forget.** Fine-tuning on benign task data can measurably weaken alignment — the model becomes more compliant with harmful requests as a side effect of being trained to be more compliant generally. **Safety evals must be a blocking gate**, not a post-hoc check.

**Preference tuning, when you need it:**
| Method | Complexity | Use |
|---|---|---|
| **SFT only** | Low | Most cases — start and often stop here |
| ⭐ **SFT → DPO** | Medium | The practical default when you have preference pairs |
| **ORPO / KTO** | Medium | ORPO merges SFT and preference in one stage; KTO needs only binary good/bad, not pairs |
| **SFT → GRPO + RLVR** | Medium | When correctness is **programmatically checkable** — schema-valid extraction, SQL that executes to the right result, code that passes tests. The verifier *is* the reward, so no reward model and no value network; runs as LoRA on one GPU (`04` §7.6, §8.6) |
| **RLHF / PPO** | **High** | Only with a strong reward model and a team to maintain it. Rarely worth it internally. |

## 6.8 Failure modes

| Failure | Why | Mitigation |
|---|---|---|
| **Fine-tuned to inject knowledge** | Wrong tool | Triage flow (§6.4); route to RAG |
| **Worse than a good prompt** | No baseline was run | **Blocking eval gate vs few-shot** |
| **Template mismatch train/serve** | Two codepaths | Platform owns templating; generate serving config from training config |
| **Catastrophic forgetting** | Narrow data, too many epochs | Replay data, low LR, 1–3 epochs, general-capability eval |
| **Safety regression** | Alignment weakened as a side effect | Blocking safety evals |
| **Eval contamination** | Training data overlaps eval | Decontamination in the validation gate |
| **Overfitting** | Too many epochs on small data | Early stopping on held-out loss; the platform should default conservatively |
| **Adapter/base version mismatch** | Base upgraded, adapters not retested | Registry pins the triple; re-validate adapters on base upgrade |
| **OOM mid-run** | Memory estimated wrong | Pre-flight memory calculator from §6.2; fail fast at submission |
| **Irreproducible results** | Seed/data/config not captured | Hash the dataset; record everything in the registry |
| **Tenant data leakage** | Shared storage or a shared base adapter | Per-tenant isolation; never merge tenant data |

## 6.9 Evaluation

**The gate has three blocking checks, and the first is the important one:**

1. **Does it beat few-shot prompting on the same task?** If not, ship the prompt instead. This kills most bad projects before they consume serving capacity.
2. **Is general capability retained?** A small benchmark battery (reasoning, instruction-following, a general QA set) versus the base model, to catch forgetting.
3. **Do safety evals pass?** Versus the base model, on refusal and harmful-content sets.

Then task quality: a golden set with programmatic graders where possible, LLM-judge where not (with the bias mitigations from `07-agentic-system-design.md` §4.5), plus human review on a sample.

**Report cost and latency alongside quality** — a fine-tune that adds 3 points of accuracy and doubles serving cost may not be worth shipping, and the platform should surface that trade-off rather than reporting quality alone.

## 6.10 Cost and latency — where the platform actually pays

**Training is cheap** (§6.2: ~0.7 GPUs continuously for 30 teams). **Serving is where the money is**, and it's the platform's core economic argument:

| Approach | 30 fine-tuned 8B models |
|---|---|
| Separate deployment per model | **30 GPUs**, each mostly idle |
| ⭐ **Multi-adapter serving** (vLLM multi-LoRA / S-LoRA) | **1–2 GPUs** — one base model in memory, adapters swapped per request |

> **That's a ~15–20× cost reduction, and it only works because LoRA keeps the base weights shared.** It's also the strongest argument for mandating LoRA over full fine-tuning as platform policy: full fine-tunes cannot share a base, so each one costs a dedicated deployment. Naming this connection — a *training* decision driven by *serving* economics — is exactly the kind of cross-cutting reasoning that reads as senior.

**Other levers:** spot/preemptible GPUs for training with checkpoint-resume (60–80% saving, and training is interruptible); gradient checkpointing to trade compute for memory; packing short examples to fill sequences; and quotas so one team can't monopolize the fleet.

## 6.11 Corner Questions

**Q: A team wants to fine-tune so the model knows our internal documentation. Advise them.**
> That's RAG, not fine-tuning, and I'd push back clearly. Fine-tuning teaches form, not facts — the model blends what it learned rather than retrieving it, so answers are unreliable and unciteable; documentation changes mean a full retrain; and there's no way to attribute a claim to a source, which is usually what the use case actually needs. RAG gives freshness, citations, and access control for a fraction of the cost. The place fine-tuning *would* help alongside RAG is teaching a consistent output format or house style — so I'd suggest RAG for the knowledge and, if needed, a small fine-tune for the presentation layer.

**Q: The fine-tuned model performs worse than the base model with a good prompt.**
> Common, and it's why the platform makes the prompting baseline a blocking gate. Likely causes, in order: too little or too noisy data — the LIMA result is that 1,000 curated examples beat far larger noisy sets, so I'd inspect before tuning hyperparameters; too many epochs causing overfitting or forgetting; a prompt-template mismatch between training and serving, which is silent and brutal; or catastrophic forgetting where the task improved but general instruction-following collapsed. The right response is to fix the data and re-run once — and if it still loses, ship the prompt. That outcome is a success for the platform, not a failure.

**Q: Can you full fine-tune an 8B model on one 80 GB GPU?**
> Not with standard Adam. The memory is roughly 16 bytes per parameter — 2 for fp16 weights, 2 for gradients, 8 for fp32 Adam moments, 4 for master weights — so 8B needs about 128 GB before activations, which overflows an 80 GB card. Options: LoRA, which drops it to about 18 GB because only the adapter has gradients and optimizer state; QLoRA to about 7 GB; or full fine-tuning across multiple GPUs with FSDP or ZeRO-3 to shard the optimizer state. Gradient checkpointing and an 8-bit optimizer could squeeze full FT closer, but LoRA is almost always the right call — and the platform should run this calculation at submission time and reject impossible configs rather than failing four hours in.

**Q: The model got better at the task but worse at everything else.**
> Catastrophic forgetting, and it's why general-capability eval is a blocking gate rather than an afterthought. Fixes: mix 5–20% general instruction data back in as replay; lower the learning rate; cut epochs — often 1–3 is right and teams default to far more; and prefer LoRA over full fine-tuning since a frozen base is inherently less destructive. I'd also check whether the task data is unusually narrow in format, because training on a single rigid output shape teaches the model to produce it unconditionally. And I'd re-run safety evals, because forgetting and alignment regression usually travel together.

**Q: Thirty teams, thirty fine-tuned models. Serving cost is exploding.**
> Almost certainly one deployment per model, which is thirty GPUs each mostly idle. The fix is multi-adapter serving — vLLM's multi-LoRA or S-LoRA keep one base model resident and swap small adapters per request, collapsing thirty deployments into one or two GPUs. That only works if everyone used LoRA, which is the real argument for mandating it as platform policy: full fine-tunes can't share a base, so each one permanently costs a dedicated deployment. I'd also retire adapters that aren't serving traffic — registries accumulate abandoned models — and check whether some teams should be sharing one adapter rather than each tuning their own.

**Q: How much data do we need?**
> It depends on what you're teaching, and quality dominates quantity. For format or style, a few hundred to a thousand good examples is often enough — LIMA is the reference point. For a specific task like classification or extraction, low thousands. For a genuinely new skill, tens of thousands. For a new domain or language, you're into continued pretraining and billions of tokens, which is a different project. The practical advice I'd give a team is to start with 500 excellent examples rather than 50,000 scraped ones, measure against the prompting baseline, and only scale the dataset if the curve says more data helps — because usually the next win comes from cleaning what they have.

**Q: Same dataset, same config, two runs, different results.**
> Some variance is expected, but large variance means something isn't pinned. I'd check that the seed, data ordering, and library versions are all captured, that the dataset is hashed so it genuinely is the same data, and that any distributed run has deterministic reduction — multi-GPU non-determinism from non-associative float accumulation is a common source. GPU non-determinism in some kernels is real and usually not worth eliminating. The platform's job is to record the full triple of data hash, config, and seed in the registry so a run can be reproduced at all; if results still swing widely with everything pinned, that's a signal the dataset is too small and the model is landing in different minima.

## 6.12 Mid-level vs Senior

| | Mid-level | **Senior** |
|---|---|---|
| Framing | "Build a fine-tuning service" | **Triage first** — most requests should be RAG or a prompt |
| Method | "Use LoRA" | Derives the 16-bytes/param memory math; knows when QLoRA and full FT apply |
| Data | "Collect training data" | Curation as *the* bottleneck; blocking validation gate; LIMA |
| Template | Not mentioned | Platform owns templating so train/serve cannot diverge |
| Eval | "Check task accuracy" | Blocking gates: beats prompting · retains general capability · passes safety |
| Forgetting | Not mentioned | Replay, low LR, few epochs; safety regression as a first-class risk |
| Economics | Focuses on training cost | **Serving is the cost** — multi-adapter is ~15–20×, and it's why LoRA is mandated |

## 6.13 References for this case study

**Read first**
- ⭐ **LoRA: Low-Rank Adaptation of Large Language Models** — Hu et al., 2021 (arXiv 2106.09685) — the method the whole platform rests on
- ⭐ **QLoRA: Efficient Finetuning of Quantized LLMs** — Dettmers et al., 2023 (2305.14314) — 4-bit base + paged optimizers; the memory win in §6.2
- ⭐ **LIMA: Less Is More for Alignment** — Zhou et al., 2023 (2305.11206) — **1,000 curated examples beat far larger noisy sets.** Cite this when a team wants to scrape 100k rows.

**Alignment & preference tuning**
- **Training language models to follow instructions with human feedback (InstructGPT)** — Ouyang et al., 2022 (2203.02155) — the SFT → RM → PPO pipeline
- ⭐ **Direct Preference Optimization** — Rafailov et al., 2023 (2305.18290) — why you can usually skip PPO
- **ORPO** (2403.07691) · **KTO** (2402.01306) — single-stage and binary-feedback alternatives
- **Fine-tuning Aligned Language Models Compromises Safety** — Qi et al., 2023 (2310.03693) — **the §6.7 safety-regression result; makes the case for a blocking safety gate**

**Scale & serving**
- **ZeRO** — Rajbhandari et al., 2019 (1910.02054) · **PyTorch FSDP** — Zhao et al., 2023 (2304.11277)
- ⭐ **S-LoRA: Serving Thousands of Concurrent LoRA Adapters** — Sheng et al., 2023 (2311.03285) — the §6.10 economics
- **Punica: Multi-Tenant LoRA Serving** — Chen et al., 2023 (2310.18547)
- **DoRA: Weight-Decomposed Low-Rank Adaptation** — Liu et al., 2024 (2402.09353) — a stronger LoRA variant worth knowing
- **Scaling Down to Scale Up: A Guide to PEFT** — Lialin et al., 2023 (2303.15647) — the survey

**Code**
- ⭐ `huggingface/peft` (~17k) — LoRA/QLoRA/DoRA reference implementations
- ⭐ `huggingface/trl` (~11k) — SFT, DPO, ORPO, KTO, GRPO trainers
- `hiyouga/LLaMA-Factory` (~45k) — config-driven tuning; the model for platform self-service
- `axolotl-ai-cloud/axolotl` (~9k) — YAML-driven fine-tuning
- `vllm-project/vllm` (~55k) — multi-LoRA serving (§6.10)
- `bitsandbytes-foundation/bitsandbytes` (~7k) — 4-bit quantization and paged optimizers

---

# Case Study 7 — Content Moderation (Multimodal)

> *"Design the system that keeps policy-violating content off a platform serving 500M items a day."*

Multimodal classification, a precision/recall trade-off set by policy, and active learning on hard samples. The distinguishing feature: **there is no single operating point.** Different policies demand opposite error profiles, and a design with one threshold has already failed.

## 7.1 Requirements

**Ask:**
- "Which policies, and are they equally severe?" *(They aren't — that's the whole design.)*
- "Proactive scanning, or reactive on user reports?"
- "What's the human review capacity?" *(A hard constraint, like fraud.)*
- "Is there an appeals process, and regulatory exposure — DSA, COPPA?"

**Assume:** social + ads platform, **~15 policy categories**, multimodal (image, text, video), **500M items/day**, proactive scanning, ~1,000 reviewers, appeals required, EU DSA transparency obligations.

**Non-functional — and note there's no global target:**
- Proactive decision **p95 < 500 ms** pre-publish, or async with fast takedown
- **Per-policy operating points**, e.g. CSAM/terrorism → **recall ≈ 1.0** at any precision cost; hate speech → **precision ≥ 0.95** (wrongful removal is a censorship incident); spam → balanced
- Review queue sized to headcount
- Every action **auditable and appealable**

**Descope:** the policy definitions themselves, legal reporting workflows, advertiser billing.

> **Open by rejecting a single metric.** "Maximize F1 across policies" is wrong: a false negative on CSAM is a catastrophe, while a false positive on political speech is a front-page story. **One model can produce scores; the *decision layer* applies per-policy thresholds and per-policy actions.** Separating those is the same move as fraud's decision layer (§5.7), and it's what the round is testing.

## 7.2 Estimation

```
500M items/day ÷ 86,400 s ≈ 5,800/s average → ~20,000/s peak
Violation rate ~0.5%       ≈ 2.5M violations/day
```

**Compute is not the constraint:**
```
Multimodal encoder ~1,000 items/s/GPU batched
500M / 1,000 = 500,000 GPU-seconds ≈ 139 GPU-hours/day ≈ 6 GPUs continuous
```

**Human review is the constraint:**
```
1,000 reviewers × 500 items/day = 500,000 reviews/day = 0.1% of volume
```

> **The system must decide 99.9% of content automatically.** Only one item in a thousand can ever be seen by a human — so review is a *scarce routing target*, not a safety net, and the interesting design question becomes **which 0.1% is worth a human's attention** (§7.6). That framing is worth more than any model detail.

**Hash matching pays for itself before any ML runs.** Known violating content — previously-actioned CSAM, terrorist media, banned spam images — is matched by perceptual hash at essentially zero cost and near-perfect precision. If that catches 20–30% of violations, it removes them from model and reviewer load entirely.

## 7.3 Architecture

```
  Upload (image / text / video)
        │
        ▼
  ┌──────────────────────────────────────────┐
  │ TIER 0  HASH MATCH  (PDQ / TMK+PDQF)     │  known violations, ~0 cost
  │         → immediate action, no ML        │
  └────────────────┬─────────────────────────┘
                   ▼
  ┌──────────────────────────────────────────┐
  │ TIER 1  CHEAP FILTERS                    │  OCR · language ID · nudity
  │         drop obvious-benign early        │  detector · URL/domain lists
  └────────────────┬─────────────────────────┘
                   ▼
  ┌──────────────────────────────────────────┐
  │ TIER 2  MULTIMODAL CLASSIFIER            │  joint image+text encoder
  │  15 policy heads → 15 calibrated scores  │  (FLAVA / SigLIP-class)
  └────────────────┬─────────────────────────┘
                   ▼
  ┌──────────────────────────────────────────┐
  │ DECISION LAYER — per-policy thresholds   │
  │  allow │ demote │ age-gate │ label │     │  ← graduated response
  │  review │ remove │ account action        │
  └───┬──────────────────────┬───────────────┘
      │                      ▼
      │            REVIEW QUEUE (prioritized:
      │            severity × uncertainty × virality)
      │                      │
      ▼                      ▼
   published          reviewer decision ──▶ action + LABEL
                             │                    │
                          appeals ────────────────┤
                             │                    ▼
                    overturn rate      ACTIVE LEARNING → retrain
                    (precision signal)
                             ▲
                    PREVALENCE SAMPLING (§7.9)
                    random sample of ALL content → unbiased truth
```

### Recommended stack

| Component | ⭐ Default | Alternative | Flip when |
|---|---|---|---|
| **Hash matching** | ⭐ **PDQ** (images), **TMK+PDQF** (video) | MD5/SHA (exact only) | Always run this tier first — perceptual hashes survive re-encoding and cropping, exact hashes don't |
| **Multimodal encoder** | ⭐ **SigLIP** or **FLAVA**-class joint encoder | CLIP · late-fusion of separate encoders | **Joint beats late fusion** for meme-style violations (§7.5). |
| **Text-in-image** | ⭐ **OCR** (PaddleOCR / Tesseract) feeding the text branch | Ignore | Never ignore — most policy-violating memes carry their payload as embedded text |
| **Text policy model** | **XLM-RoBERTa** multilingual classifier | LLM for nuanced/contextual policies | LLM for low-volume, high-nuance categories (misinformation, context-dependent hate) where cost is affordable |
| **Video** | Frame sampling (keyframes + fixed interval) + audio transcription | Full temporal model | Full temporal only for policies that need motion (violence); sampling handles most |
| **Serving** | **Triton** with dynamic batching | Ray Serve | 20k/s peak needs real batching |
| **Decision layer** | Config-driven per-policy thresholds + actions | Hardcoded | **Must be config** — policy teams change thresholds without a model deploy |
| **Fast response** | **Rules / blocklists** alongside the model | Model-only | Same argument as fraud (§5.6): a new slur or scam format needs a fix in hours, not a retrain |
| **Review tooling** | Custom queue + prioritization service | Off-the-shelf | The prioritization logic *is* the product; don't outsource it |
| **Labeling / active learning** | Label Studio + uncertainty sampling | Prodigy | Feed reviewer decisions back automatically |

## 7.4 Deep Dive A — Per-policy operating points and graduated response

**One model, fifteen policies, fifteen different thresholds.** The table you should be able to sketch:

| Policy | Priority | Threshold posture | Default action |
|---|---|---|---|
| **CSAM / terrorism** | Recall ≈ 1.0 | Very low threshold; accept many FPs | Remove + report + human confirm |
| **Graphic violence** | Recall-leaning | Low | Age-gate or remove |
| **Adult content** | Balanced, region-dependent | Medium | Age-gate / demote |
| **Hate speech** | **Precision-leaning** | High threshold | Review before removal |
| **Misinformation** | **Precision-leaning**, context-heavy | Very high | Label / demote — rarely remove |
| **Spam** | Balanced, high volume | Medium | Demote / remove, low review |

> **The design element people miss: the action space is a ladder, not a binary.** Between "allow" and "remove" sit **demote (reduce reach), age-gate, add an informational label, restrict sharing, and route to review**. Graduated response lets you act on uncertain cases proportionately — you can demote at a confidence where you'd never remove. That converts a hard classification problem into a soft ranking problem, which is far more tractable and far more defensible publicly.

**Calibration is required**, not optional: thresholds only mean something if a 0.9 score corresponds to 90% precision. Recalibrate per policy and per language, since the same model is systematically overconfident in low-resource languages.

## 7.5 Deep Dive B — Multimodal fusion and the meme problem

**The canonical hard case: a benign image plus benign text that is violating in combination.** A photo of a desert and the caption "look how many people love you" are each innocuous; together they're an attack. **Unimodal models cannot catch this by construction**, and neither can late fusion that only combines independent scores.

| Approach | Catches cross-modal violations? | Cost |
|---|---|---|
| Text-only / image-only | No | Cheapest |
| **Late fusion** (separate encoders, combine scores) | Weakly — no interaction | Cheap, easy to ship |
| ⭐ **Joint multimodal encoder** (FLAVA / SigLIP-class) | **Yes** — cross-attention between modalities | Moderate |
| VLM (LLaVA-class) with policy prompt | Yes, plus reasoning | Expensive — reserve for review assist |

**Recommend a joint encoder** as the workhorse, with a VLM available for low-volume nuanced categories and for generating explanations that help reviewers.

**OCR is non-negotiable.** A large share of policy-violating image content carries its payload as rendered text — screenshots, memes, infographics. Without OCR feeding the text branch, you're blind to it. This is also the first thing adversaries exploit.

**Video:** sample keyframes plus fixed intervals, transcribe audio, aggregate per-frame scores with a max-and-mean pooling (max catches a single violating frame; mean catches sustained content). Full temporal models only where motion genuinely matters.

## 7.6 Deep Dive C — Human review as a designed subsystem

With 0.1% capacity, **queue prioritization is the highest-leverage component in the system.**

```
priority ≈ severity × uncertainty × reach
```
- **Severity** — policy tier; CSAM outranks spam absolutely
- **Uncertainty** — scores near the threshold are where a human adds the most information
- **Reach / virality** — a borderline post with 1M projected views matters more than a confident violation with 3 views. **Include predicted reach**, not just current views; acting after virality is too late.

**Reviewer wellbeing is a real design constraint**, not an HR footnote: blur-by-default with click-to-reveal, greyscale, exposure limits and rotation across content types, mandatory breaks, and counselling access. It also affects the system — reviewer accuracy degrades with fatigue, so throughput targets that ignore this produce worse labels.

**Reviewer decisions are your training labels, and they're biased.** Reviewers only see what the model flagged, so training on them reinforces the model's existing blind spots — the same feedback loop as recsys (§1.7) and fraud (§5.4). **Fix: inject a random sample of unflagged content into the queue.** It feels wasteful and it's the only way to discover what you're missing.

**Appeals close the loop.** The **appeal overturn rate is a direct precision signal** — if 30% of appealed removals are reinstated, your precision at that threshold is far worse than your offline metrics claim.

## 7.7 Deep Dive D — Adversarial evasion

Like fraud, the distribution is hostile — but here adversaries iterate publicly and share techniques.

| Evasion | Defense |
|---|---|
| Leetspeak, homoglyphs, zero-width chars | Unicode normalization (NFKC), confusable mapping, de-leeting before tokenization |
| Re-encoding, cropping, rotating images | **Perceptual hashing** (PDQ) rather than exact; augmentation during training |
| Text rendered into images | OCR (§7.5) |
| Splitting a violation across multiple posts | Session/account-level aggregation, not just per-item |
| Coordinated campaigns | Graph detection over accounts, timing, and shared assets |
| Novel slurs and coded language | **Blocklists shippable in hours** + rapid retrain loop |

**Coordinated inauthentic behaviour needs a different lens entirely** — no single item is violating, the *pattern* is. That's a network-detection problem layered on top of per-item classification, and worth naming as out of scope for the classifier but in scope for the system.

## 7.8 Failure modes

| Failure | Why | Mitigation |
|---|---|---|
| **Wrongful removal goes viral** | High-precision policy at too low a threshold | Precision-leaning thresholds + review-before-removal for speech policies; fast appeals |
| **Severe harm missed** | Recall gap on a rare policy | Recall ≈ 1.0 posture, hash tier, dedicated eval sets |
| **Reviewer label bias** | They only see flagged content | Inject random unflagged samples |
| **Low-resource language gap** | Training data concentrated in English | Per-language eval and calibration; targeted data collection; route to human where model is weak |
| **Demographic unfairness** | Dialect misread as toxic (a documented failure for AAVE) | Fairness audit by dialect/group; it's a documented, expected failure — test for it explicitly |
| **Review queue overflow** | Breaking news, coordinated event | Dynamic thresholds on queue depth; severity-first shedding |
| **Prevalence unknown** | No unbiased measurement | Random-sample prevalence estimation (§7.9) |
| **Evasion after a fix** | Adversary adapts | Blocklists in hours; monitor per-policy score drift |
| **Appeal backlog** | Appeals unstaffed | Size appeals capacity with removals; auto-reinstate on SLA breach for low-severity |

## 7.9 Evaluation

**Never report an aggregate F1.** Report **per-policy precision and recall at the deployed threshold**, segmented by language and region.

> **The metric that matters most, and the one candidates never mention: prevalence.** Precision and recall are computed on flagged content, so they can't tell you what fraction of *all* content is violating and got through. **Take a random sample of all content, have humans label it, and estimate the true violating rate — and the share of *views* that were violating.** This is Meta's published methodology, it's the only unbiased measure of the system's real performance, and it's the moderation analogue of fraud's randomized allow-through (§5.4).

Also track:
- **Appeal overturn rate** per policy — precision from the outside
- **Time-to-action** for severe categories — for CSAM this matters more than precision
- **Automation rate** — share actioned without human review, and whether it's rising safely
- **Reviewer agreement (IAA)** — your labelling ceiling
- **Per-language breakdown** — the aggregate is dominated by English and hides everything

## 7.10 Cost and latency

**500 ms budget:**
```
Hash match (PDQ lookup)              10 ms   ← exits early for known content
Cheap filters (lang ID, OCR trigger)  30 ms
OCR (when triggered)                 120 ms
Multimodal classifier (batched)       80 ms
Decision layer + logging              20 ms
Overhead                              60 ms
                                 ─────────
                                ~320 ms typical
```

| Lever | Impact |
|---|---|
| ⭐ **Hash tier first** | Removes 20–30% of violations from model + reviewer load at ~zero cost |
| ⭐ **Cheap-filter cascade** | Most content is obviously benign; don't run the big model on all of it |
| **Trigger OCR conditionally** | Only when text-likelihood is high — OCR is the expensive step |
| **Dynamic batching** | Essential at 20k/s |
| **Frame sampling for video** | Orders of magnitude vs full decode |
| **Distil the joint encoder** | 2–4× throughput at small quality cost, verified per policy |

**Human review dominates total cost.** 1,000 reviewers is a far larger line item than 6 GPUs — so **the highest-ROI engineering is anything that raises automation rate safely or improves queue prioritization**, not anything that shaves GPU cost.

## 7.11 Corner Questions

**Q: You wrongly remove a legitimate post and it becomes a censorship story. What in the design failed?**
> Probably the threshold posture and the action ladder. Speech policies like hate and misinformation need precision-leaning thresholds and should route to human review before removal rather than auto-removing — and for many of them the right action is demote or label, not remove. I'd check the appeal overturn rate for that policy, which is the honest precision signal, and whether the model was calibrated for that language and dialect specifically, since misreading dialect as toxic is a well-documented failure mode. The systemic fix is graduated response: at a confidence where removal is unjustifiable, demotion usually still is, and it fails far more gracefully.

**Q: A benign image with a benign caption is hateful in combination. How do you catch it?**
> Unimodal models can't, by construction, and neither can late fusion that only combines independent scores — each modality looks clean in isolation. You need a joint encoder with cross-attention between image and text so the representation captures the *combination*, which is what FLAVA-class models do and what the Hateful Memes benchmark was built to measure. OCR matters too, since much of this content renders its text into the image, which would otherwise bypass the text branch entirely. And I'd expect this category to have lower precision than others, so I'd route it to review rather than auto-remove.

**Q: How do you measure recall when by definition you don't know what you missed?**
> You sample randomly from *all* content, not from flagged content, and have humans label it. That gives you prevalence — the true violating rate and, more usefully, the share of views that were violating — and it's the only unbiased read on what's getting through. Precision and recall computed on flagged content are conditioned on the model's own decisions and cannot answer this. It's the same structural problem as fraud's blocked transactions, and the same shape of fix. It's expensive human labelling for content that's 99.5% benign, which is exactly why teams skip it and then can't say how well the system works.

**Q: A new coded slur appears Monday morning. How fast can you respond?**
> Hours, via blocklists and rules — not the model, which needs labelled data and a retrain cycle measured in days. That's why a rules layer sits alongside the classifier, same as fraud. In parallel: collect examples, get them labelled with priority, and fold them into the next training run so the rule can eventually be retired. I'd also want detection for the *emergence* pattern — sudden spikes in unfamiliar tokens co-occurring with already-flagged content — so novel terms surface for policy review before they're widespread rather than after.

**Q: You can review 0.1% of content. What gets reviewed?**
> Severity × uncertainty × predicted reach. Severity because CSAM outranks spam absolutely; uncertainty because scores near the threshold are where a human adds the most information, while confident cases are wasted reviewer time; and predicted reach because a borderline item heading for a million views matters far more than a certain violation with three. Predicted, not current — reviewing after something goes viral is too late. On top of that I'd reserve a slice of the queue for randomly sampled *unflagged* content, because otherwise reviewers only ever see what the model already suspects and the training labels reinforce its blind spots.

**Q: The model works well in English and badly in Burmese.**
> Expected, and it's the failure mode with the worst real-world consequences. Causes: training data concentration, weaker tokenization and pretraining for low-resource languages, and cultural context that doesn't transfer. Mitigations: evaluate and calibrate *per language* rather than in aggregate, since aggregate metrics are dominated by English; a multilingual encoder so labelled English data transfers; targeted data collection and native-speaker reviewers for high-risk languages; and lowering the automation threshold where the model is measurably weak, routing more to humans rather than pretending the score means the same thing. I'd treat per-language performance as a launch gate for operating in a market.

**Q: Your training labels come from reviewers. What's wrong with that?**
> Reviewers only see content the model flagged, so the labels describe the model's own decision boundary rather than the true distribution — train on them and you sharpen existing blind spots without ever discovering new ones. It's the same feedback loop as a recommender training on its own impressions. The fix is to inject randomly sampled unflagged content into the review queue so labels cover the whole distribution, and to use prevalence sampling as the ground truth against which flagged-content metrics are sanity-checked. I'd also track reviewer agreement, since labels with 80% IAA cap how good any model trained on them can be.

## 7.12 Mid-level vs Senior

| | Mid-level | **Senior** |
|---|---|---|
| Objective | "Maximize F1" | **Per-policy operating points**; opposite postures for CSAM vs speech |
| Actions | Allow / remove | **Graduated response** — demote, age-gate, label, review |
| Fusion | "Use a multimodal model" | Joint vs late fusion, *why* memes need cross-attention; OCR |
| Review | "Humans check flagged items" | Queue prioritization by severity × uncertainty × **predicted reach**; wellbeing as a constraint |
| Labels | "Reviewers label it" | Names the reviewer-label feedback loop; random unflagged injection |
| Measurement | Precision/recall | **Prevalence via random sampling**; appeal overturn rate |
| Fairness | Not mentioned | Per-language and per-dialect eval as a launch gate |

## 7.13 References for this case study

**Read first**
- ⭐ **The Hateful Memes Challenge** — Kiela et al., 2020 (arXiv 2005.04790) — **the benchmark built specifically to show unimodal and late-fusion models fail on cross-modal violations.** The empirical basis for §7.5.
- ⭐ **FLAVA: A Foundational Language And Vision Alignment Model** — Singh et al., 2021 (2112.04482)
- **Meta Community Standards Enforcement Reports** + their **prevalence methodology** — the public reference for §7.9, and the clearest statement of why prevalence rather than precision is the headline metric

**Models**
- **CLIP** — Radford et al., 2021 (2103.00020) · **SigLIP** — Zhai et al., 2023 (2303.15343) — the joint-encoder lineage
- **LLaVA** — Liu et al., 2023 (2304.08485) — VLMs for nuanced policy and reviewer assistance
- **PDQ & TMK+PDQF** — Meta's open perceptual hashing (github.com/facebook/ThreatExchange) — the Tier-0 design

**Fairness, evasion, policy**
- **Racial Bias in Hate Speech Detection** — Sap et al., ACL 2019 — the AAVE false-positive result; test for this explicitly
- **Perspective API / Jigsaw Toxic Comments** — widely used toxicity baselines and their documented biases
- **EU Digital Services Act** — the transparency and appeals obligations that shape §7.6 and §7.9
- **Deep Learning for Detecting Coordinated Inauthentic Behaviour** — network-level detection beyond per-item classification

**Code**
- `facebook/ThreatExchange` — PDQ, TMK, and the hash-matching tooling
- `open-mmlab/mmpretrain` · `mlfoundations/open_clip` (~11k) — multimodal encoders
- `PaddlePaddle/PaddleOCR` (~45k) — the OCR tier
- `unitaryai/detoxify` (~2k) — text toxicity baselines
- `HumanSignal/label-studio` (~21k) — review and active-learning loop

---

# Case Study 8 — Feature Store / ML Platform

> *"Forty teams, two hundred models. Design the feature platform."*

Referenced by every other case study in this file. It's also the design where **the correct answer is sometimes "don't build one"** — and being willing to say that is most of the signal.

## 8.1 Requirements

**Ask:**
- "How many teams and models, and do they share features?" *(Sharing is the only real justification.)*
- "Batch-only, or real-time serving?" *(Batch-only means you may not need a feature store at all.)*
- "What warehouse exists already?"
- "Who owns a feature — the producing team or the platform?" *(An org question that determines the design.)*

**Assume:** 40 teams, ~200 models, ~5,000 distinct features, **both batch training and real-time serving**, existing warehouse on S3/Iceberg, features shared across teams.

**Non-functional:**
- **Online read p99 < 20 ms** — it's on every prediction path in the company
- **Point-in-time correctness guaranteed by construction**, not by convention
- Features **discoverable and reusable** across teams
- **Backfill a new feature over 2 years of history** without a bespoke project
- Freshness SLA per feature, monitored

**Descope:** model training and serving, experiment tracking, the warehouse itself.

> **Say the honest thing early.** A feature store solves three problems: *training-serving skew*, *point-in-time correctness*, and *feature reuse across teams*. **If you have five models, one team, and batch-only scoring, you have none of those problems and a feature store is pure overhead** — a well-organized warehouse and a shared transformation library are better. The justification here is 40 teams and real-time serving. Naming the condition under which you'd *decline* to build it is the strongest opening.

## 8.2 Estimation

**Online read volume — the number that shapes the serving design:**
```
Aggregate model traffic ~50k QPS
× ~50 features per prediction
= 2.5M feature reads/second
```
> **You cannot issue 50 individual reads per prediction** — that's 50 network round trips inside a 20 ms budget. **Features must be co-located by entity so one key fetch returns the whole vector**, turning 2.5M reads/s into 50k multi-gets/s. That single decision is what makes the latency budget achievable.

**Online storage:**
```
10M entities × ~5,000 features × 8 B ≈ 400 GB  (dense upper bound)
```
Realistically far less — most features apply to a subset of entities — but it sizes the cluster: a Redis cluster or DynamoDB, not a single node.

**Backfill cost — quantify it, because teams always underestimate:**
```
2 years × daily snapshots × 10M entities = ~7.3B entity-days
```
A serious Spark/Flink job. The platform should estimate and surface this cost *before* the job runs, not after.

## 8.3 Architecture

```
  FEATURE DEFINITION (code, versioned, reviewed)
   entity · source · transformation · TTL · owner · freshness SLA
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
 ┌──────────────┐        ┌──────────────┐
 │ BATCH        │        │ STREAMING    │   same definition,
 │ materialize  │        │ materialize  │   two runtimes
 │ (Spark/SQL)  │        │ (Flink)      │
 └──────┬───────┘        └──────┬───────┘
        │                       │
        ▼                       ▼
 ┌────────────────┐      ┌────────────────┐
 │ OFFLINE STORE  │      │ ONLINE STORE   │
 │ Iceberg on S3  │      │ Redis / Dynamo │
 │ full history,  │      │ latest value,  │
 │ bitemporal     │      │ entity-keyed   │
 └───────┬────────┘      └───────┬────────┘
         │                       │
         ▼                       ▼
  POINT-IN-TIME JOIN       get_online_features()
  (training sets)           p99 < 20 ms
         │                       │
         ▼                       ▼
     TRAINING                 SERVING
         └──────── PARITY TESTS ─────────┘

  REGISTRY / CATALOG: discovery · lineage · ownership · usage · freshness
```

### Recommended stack

| Component | ⭐ Default | Alternative | Flip when |
|---|---|---|---|
| **Platform** | ⭐ **Feast** (OSS) | **Tecton** (managed) · build in-house | **Buy or adopt OSS.** Point-in-time correctness is subtle and easy to get wrong — reimplementing it is a classic waste. Build only if your access patterns are genuinely unusual. |
| **Offline store** | ⭐ **Iceberg/Delta on S3** | Snowflake · BigQuery | Iceberg gives time travel = reproducible training snapshots; warehouse if you're already centralized there |
| **Online store** | ⭐ **Redis** (cluster) | **DynamoDB** · Cassandra | DynamoDB past what Redis memory economics allow, or when you want zero ops; Redis for the sub-10 ms path |
| **Batch compute** | **Spark** | DuckDB · dbt + warehouse SQL | **DuckDB for single-node scale** — far simpler, and most feature jobs aren't actually big |
| **Streaming compute** | ⭐ **Flink** | Spark Structured Streaming | Flink for true event-time semantics, watermarks, and sub-second freshness |
| **Orchestration** | **Dagster** | Airflow | Dagster's asset model maps naturally onto "this feature derives from that table" |
| **Registry / catalog** | Feast registry + **DataHub** or **OpenMetadata** | Spreadsheet (genuinely, at small scale) | Catalog matters at 5,000 features; it's overhead at 50 |
| **Serving SDK** | Thin client, **multi-get by entity** | REST per feature | Never per-feature calls (§8.2) |
| **Monitoring** | Freshness SLA + null-rate + drift per feature | — | Silent staleness is the #1 production failure (§8.8) |

## 8.4 Deep Dive A — Point-in-time correctness

**The defining problem of a feature store, and the one most candidates cannot explain precisely.**

Training data needs each feature's value **as it was at label time** — not as it is now. The naive join is a disaster:

```sql
-- WRONG: leaks the future
SELECT l.user_id, l.label, f.total_purchases
FROM labels l JOIN features f USING (user_id)
```
A label from six months ago gets today's `total_purchases`, which **includes purchases made after the event you're predicting**. The model learns from the future, offline metrics look superb, production collapses.

```sql
-- RIGHT: as-of join on event time
SELECT l.user_id, l.label, f.total_purchases
FROM labels l
ASOF JOIN features f
  ON f.user_id = l.user_id AND f.event_ts <= l.event_ts
```

**The subtlety that separates senior answers — there are two timestamps, not one:**

| Timestamp | Meaning |
|---|---|
| **Event time** | When the fact became true |
| **Processing time** | When your pipeline *knew* it |

> **You must join on what was *available* at decision time, not what was *true*.** If a feature pipeline runs hourly, then at 10:30 the serving system saw the 10:00 value — even though the 10:25 event had already happened in the real world. **Training must reproduce that lag**, or the model is trained on fresher data than it will ever see in production, and degrades on deployment in a way that looks like nothing at all in your metrics.
>
> That means storing features **bitemporally** — valid-from/valid-to on event time *plus* the time the value landed. Getting this right is the single strongest technical point available in this case study.

## 8.5 Deep Dive B — Offline/online parity

The same feature computed twice, by two codepaths, will diverge. Three strategies:

| Strategy | Guarantee | Cost |
|---|---|---|
| **Single definition, dual materialization** (the Feast model) | Strong — one transformation spec, two runtimes | Requires transformations expressible in both |
| ⭐ **Log-and-wait** — log feature values **at serving time** and train on those logs | **Strongest possible** — training data *is* what was served, so skew is definitionally zero | You can't train until you've served; cold start needs a bootstrap |
| **Unified engine** (Flink for both batch and stream) | Strong | Ties you to one engine |

**Recommend log-and-wait for critical models**, dual materialization as the general default. And regardless of strategy, **run automated parity tests**: compute a sample of features both ways daily, diff them, and alert on divergence. Skew that isn't monitored is skew you'll discover from a model regression six weeks later.

## 8.6 Deep Dive C — The online path

**20 ms p99 for potentially 100+ features.** What makes it work:

1. **Entity-keyed vectors** — one key holds all features for a user; one round trip, not fifty (§8.2)
2. **Pipelined multi-get** across entity types (user + item + merchant fetched in parallel)
3. **Connection pooling and co-location** — the feature store in the same region and AZ as the model service
4. **Freshness tiers**, because not everything needs streaming:

| Tier | Latency | Example |
|---|---|---|
| **Batch** | daily | User long-term preference embeddings |
| **Streaming** | seconds | Velocity counters, session state |
| ⭐ **On-demand / request-time** | 0 | Derived from the request payload — ratios, differences, time-since |

**On-demand transformations are the underrated one.** Features computable from the request plus already-fetched features (e.g. `amount / user_avg_amount`) should be computed **in the SDK at request time**, never stored — they're free, always consistent, and can't go stale. Knowing which features *shouldn't* be materialized is a real design skill.

## 8.7 Deep Dive D — Governance at 5,000 features

The problems at this scale are organizational as much as technical:

- **Discovery** — a catalog with owner, description, freshness SLA, lineage, and **usage stats**. Without it, teams rebuild features that already exist, and you get eleven slightly different definitions of "user lifetime value."
- **Ownership** — every feature has a named owning team. Platform owns the *infrastructure*; teams own the *semantics*.
- **Deprecation** — you cannot delete a feature without knowing who consumes it. Usage tracking makes deletion possible; without it, features accumulate forever.
- **Breaking changes** — a shared feature's definition changing silently breaks every downstream model. Version features; require a migration path; notify consumers.
- **Access control** — PII-derived features need row- and column-level restrictions, and the training-set path must respect them too.
- **Cost attribution** — materialization cost charged to the owning team, or the fleet grows without bound.

**Feature sprawl is the characteristic failure of a successful platform.** Success creates 5,000 features, of which perhaps 800 are actually used. The catalog and usage tracking are what keep that from becoming unmanageable.

## 8.8 Failure modes

| Failure | Why | Mitigation |
|---|---|---|
| **Leakage via naive join** | Point-in-time violation | As-of joins enforced by the API; make the correct path the *only* path |
| **Trained on fresher data than served** | Processing-time lag ignored | Bitemporal storage; replicate serving lag in training (§8.4) |
| ⭐ **Silent staleness** | Pipeline fails; store serves last value forever | **Freshness SLA per feature with alerting** — the #1 real production failure |
| **Nulls masked by defaults** | Missing feature → 0, model reads it as signal | Distinguish missing from zero; monitor null rates; fail loudly |
| **Train/serve skew** | Two codepaths | Log-and-wait, or automated parity tests |
| **Breaking change to a shared feature** | No versioning | Version features; usage tracking; consumer notification |
| **Backfill cost explosion** | 2 years × 10M entities underestimated | Pre-flight cost estimate before the job runs |
| **Hot entity** | One entity read by every request | Cache hot keys in-process |
| **Feature sprawl** | No deprecation path | Usage stats; TTL on unused features |
| **Online store outage** | Every model depends on it | Graceful degradation to cached or default values, with the model told it's degraded |

## 8.9 Evaluation — platform SLOs

A platform's metrics are about *adoption and reliability*, not accuracy:

- **Online read p99** and error rate — the hard SLO
- **Freshness compliance** — % of features meeting their declared SLA
- **Parity test pass rate** — offline vs online divergence
- **Time-to-first-feature** for a new team — the adoption metric that matters most; if it's measured in weeks, teams route around you
- **Feature reuse rate** — % of features consumed by more than one model. **This is the number that justifies the platform's existence**; if it's near zero, you built a library with extra steps.
- **Null/coverage rates** per feature
- **Cost per feature per month**, attributed to owners

## 8.10 Cost and latency

| Lever | Impact |
|---|---|
| ⭐ **Entity-keyed multi-get** | The difference between 20 ms and 200 ms |
| ⭐ **On-demand transforms** instead of materializing | Free — removes storage and staleness entirely |
| **Freshness tiering** | Streaming everything is the most common overspend; most features don't need it |
| **TTL + deprecation** | Bounds unbounded online-store growth |
| **Incremental materialization** | Recompute deltas, not full snapshots |
| **DuckDB over Spark** for single-node jobs | Large — most feature jobs don't need a cluster |
| **Compress/quantize embedding features** | Embeddings dominate online storage |

## 8.11 Corner Questions

**Q: Explain point-in-time correctness with a concrete example.**
> Predicting churn with a feature "total purchases." A label from January joined naively to today's feature table gets the purchase count as of today, which includes eleven months of purchases that hadn't happened yet at prediction time. The model learns that high purchase counts predict the past, offline AUC looks excellent, and production is worthless. The fix is an as-of join on event time so each label sees only the value valid at its timestamp. The deeper subtlety is that there are two clocks: you need the value your *pipeline had computed* at that moment, not the value that was true in the world — if features refresh hourly, training must reproduce that staleness or the model is trained on data fresher than it will ever be served.

**Q: A feature pipeline fails silently at 3am. What happens?**
> The online store keeps serving the last written value, so every model quietly consumes an increasingly stale feature with no error anywhere — this is the most common serious failure in production feature systems, and it's invisible in model metrics for days. Mitigation is a declared freshness SLA per feature with monitoring on write timestamps and alerting when a feature exceeds its SLA. I'd also want models to be able to detect degraded features and fall back deliberately rather than trusting whatever they're handed. And the null-versus-stale distinction matters: a missing feature defaulting to zero is read by the model as real signal, so missing must be representable and monitored separately.

**Q: Team A changes a shared feature's definition. Twenty models silently degrade.**
> A governance failure, not a technical one. Features need versions, so a definition change creates `v2` rather than mutating `v1`, and consumers migrate deliberately. Usage tracking is the prerequisite — you can't notify consumers or assess blast radius if you don't know who reads what. Definitions should live in reviewed code with the owning team as required reviewer, and I'd want parity and distribution monitoring so a shift shows up as an alert rather than as twenty separate model regressions weeks later. The organizational rule underneath: the platform owns the infrastructure, teams own the semantics, and shared features need the same change discipline as a shared API.

**Q: Do we actually need a feature store?**
> Often not, and I'd want to establish that before building. It solves exactly three problems — training-serving skew, point-in-time correctness, and cross-team reuse. With five models, one team, and batch scoring, you have none of them, and a well-organized warehouse plus a shared transformation library is simpler and better. The threshold is real-time serving plus multiple teams sharing features; below that, the platform's operational cost exceeds its benefit. Here, 40 teams and a 20 ms online path clear that bar comfortably. And even then I'd adopt Feast or Tecton rather than build, because point-in-time correctness is subtle enough that reimplementing it is a predictable way to ship leakage.

**Q: Online p99 is 80 ms and the budget is 20 ms.**
> Almost certainly per-feature round trips — 50 sequential reads inside a 20 ms budget is impossible regardless of how fast the store is. The fix is entity-keyed storage so one key returns the whole feature vector, and parallel multi-gets across entity types rather than sequential ones. After that: co-locate the store in the same AZ as the model service, use connection pooling, and move anything derivable from the request payload into on-demand transforms so it's never fetched at all. If it's still slow, I'd look at whether embedding features are dominating payload size and quantize them, and cache hot entities in-process.

**Q: How do you guarantee offline and online produce the same value?**
> The strongest guarantee is not to compute it twice: log the feature values actually used at serving time and train on those logs, which makes skew definitionally impossible. The trade-off is you can't train until you've served, so new features need a bootstrap period. Where that's impractical, a single feature definition materialized by two runtimes gets you most of the way, provided transformations are expressible in both. Either way I'd run automated parity tests — recompute a sample both ways daily and alert on divergence — because skew that isn't monitored surfaces as an unexplained model regression months later, and by then nobody connects the two.

## 8.12 Mid-level vs Senior

| | Mid-level | **Senior** |
|---|---|---|
| Justification | "Feature stores are best practice" | **Names the three problems it solves — and when not to build it** |
| Point-in-time | "Avoid leakage" | As-of joins; **event time vs processing time**; bitemporal storage |
| Parity | "Use the same code" | Log-and-wait as the strongest guarantee; automated parity tests |
| Serving | "Read from Redis" | Entity-keyed multi-get; on-demand transforms; freshness tiers |
| Failure | Not mentioned | **Silent staleness** as the top production risk; SLA + alerting |
| Governance | Not mentioned | Ownership, versioning, deprecation, usage tracking, cost attribution |
| Success metric | "It works" | **Feature reuse rate** and time-to-first-feature |

## 8.13 References for this case study

**Read first**
- ⭐ **Uber — Michelangelo: Machine Learning Platform** (eng.uber.com) — the paper that defined the category, and still the clearest statement of why offline/online parity needs infrastructure
- ⭐ **Feast documentation** — particularly point-in-time joins and the registry model; the reference implementation of §8.4
- **Designing Machine Learning Systems** — Chip Huyen, chapters on feature engineering and data distribution shifts

**Concepts**
- **Hidden Technical Debt in Machine Learning Systems** — Sculley et al., NeurIPS 2015 — **the "glue code" and "pipeline jungle" sections are exactly what a feature platform exists to prevent**
- **Tecton engineering blog** — feature freshness tiers, on-demand transformations, backfill strategies
- **Airbnb — Zipline: Declarative Feature Engineering** — a strong published take on point-in-time correctness
- **Data Cascades in High-Stakes AI** — Sambasivan et al., CHI 2021 — how upstream data failures compound silently
- **Iceberg / Delta Lake** table-format docs — time travel as the mechanism for reproducible training snapshots

**Code**
- ⭐ `feast-dev/feast` (~6k) — read the point-in-time join and materialization engine
- `dagster-io/dagster` (~13k) — asset-based orchestration with lineage
- `apache/iceberg` (~7k) — time-travel semantics
- `great-expectations/great_expectations` (~10k) — data quality gates on feature pipelines
- `datahub-project/datahub` (~10k) — catalog, lineage, ownership at scale
- `duckdb/duckdb` (~30k) — the single-node alternative to Spark for most feature jobs

---

# Shared Reference Material

> **Per-case-study references live inside each case study** — §1.13, §2.14, §3.13, §4.13, §5.13, §6.13, §7.13, §8.13. This section holds only what's cross-cutting.

## Books
- ⭐ **Designing Machine Learning Systems** — Chip Huyen. The framework at the top of this file, properly.
- **AI Engineering** — Chip Huyen. The GenAI-era companion: eval, RAG, serving.
- **Designing Data-Intensive Applications** — Kleppmann. Ch. 5–9 for the storage and streaming primitives underneath all of this.

## Cross-cutting papers
- ⭐ **Hidden Technical Debt in Machine Learning Systems** — Sculley et al., NeurIPS 2015 — glue code, pipeline jungles, feedback loops. Still the best short statement of why ML systems rot.
- **The ML Test Score** — Breck et al., 2017 — a production-readiness rubric you can quote in interviews
- **Rules of Machine Learning** — Zinkevich (Google) — 43 rules, and Rule #1 is "don't be afraid to launch a product without ML"
- **Data Cascades in High-Stakes AI** — Sambasivan et al., CHI 2021 — how upstream data problems compound
- **Machine Learning: The High-Interest Credit Card of Technical Debt** — Sculley et al., 2014

## Repos
- ⭐ `eugeneyan/applied-ml` (~28k) — curated production ML writeups by company and problem type. The best single index.
- ⭐ `stas00/ml-engineering` (~15k) — training and serving at scale, honestly written
- `alirezadir/Machine-Learning-Interviews` (~12k) — ML system design interview prep
- `chiphuyen/machine-learning-systems-design` (~10k) — case-study exercises
- `visenger/awesome-mlops` (~13k) — MLOps tooling landscape

## Companion documents
- **`07-agentic-system-design.md`** — agent-shaped systems: support agents, memory, coding agents, eval & guardrails
- **`00-system-design-curriculum.md`** — the syllabus: §1 ML SD framework, §2 infra primitives, §2B data pipelines, §5B the 2026 interview format
