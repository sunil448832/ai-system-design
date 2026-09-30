---
title: "00 — ML / AI System Design Curriculum"
subtitle: "Target: Google / Meta / Netflix / Nvidia — Senior ML (L5 / E5 / IC4-5)"
companions:
  - "07-agentic-system-design.md — worked agentic case studies for §3"
  - "08-ml-system-design.md — worked ML case studies for §1.5, §3.3, §3.4"
  - "01–06 — depth behind the index: model (01, 02), data (03), training (04), serving (05), agents (06)"
---

> ### Read this first — priority order
>
> **1. Coding / DSA (daily, §6)** — the most common rejection reason. Not ML, but it was 3 of 5 rounds on a real Google ML loop (§5A). It rarely levels you *up*; it's what fails ML candidates. Keep the daily drumbeat.
> **2. ML system design (§1)** — the leveling round. Sets your offer.
> **3. Agentic / LLM system design (§3)** — now folded into the ML design round. **Table stakes in 2026, not bonus.**
> **4. Behavioral** — a fully scored round, and at Google it now includes a technical design conversation (§5B.1).
> **5. Infra primitives (§2)** — supporting knowledge. You can't design a feature store or an LLM gateway without it.
>
> **⚠️ Read §5B before planning anything.** The interview format changed materially in 2026 — AI-assisted coding rounds, code-comprehension rounds, and in-person rounds returning. §1–§3 are timeless content; §5B is what changed around them.

## Contents

| § | Section | Priority |
|---|---|---|
| 0 | The two design rounds you'll face | — |
| 1 | **ML system design** | **Highest** |
| 2 | Infra primitives ML systems require | Medium |
| 2B | **Data pipelines — ML, RAG, agentic** | **High** |
| 3 | **Agentic & LLM system design** | **Highest** |
| 4 | Company-by-company | — |
| 5A | Calibration: a real Google L5 ML loop (2025) | — |
| 5B | **What changed in 2026** | **Read first** |
| 6 | The 8-week plan | — |
| 7 | Scoring rubrics | — |
| 8 | Resources | — |

---

# 0. The Two Design Rounds You'll Face

ML loops now contain two distinguishable design conversations. Know which one you're in — the frameworks differ.

| | **ML system design** | **Agentic / LLM system design** |
|---|---|---|
| **Sounds like** | "Design YouTube recommendations" | "Design an agent that answers questions over our databases" |
| **Unit of thought** | Data → features → model → serving | Task → tools → control loop → guardrails |
| **Core artifact** | Pipeline + feedback loops | Orchestration + eval harness |
| **Scored on** | End-to-end judgment, metrics, skew | Safety, reliability, evaluation, cost |
| **Killer mistake** | Ignoring training-serving skew | "A system prompt tells it not to" |
| **Framework** | §1.1 | §3.1 |

Both run 45–60 min. Both are won in the deep dive, not the overview.

---

# 1. ML System Design — The Leveling Round

You do this daily. What you likely lack is the *canonical framing* that lets an interviewer tick their rubric. Learn the skeleton so hard it's automatic, then spend your thinking budget on the trade-offs.

### 1.1 The framework

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

**Step 1 is where most candidates lose the round.** "Increase engagement" is not an ML objective. Translate it: *business goal* (watch time) → *ML objective* (predict P(watch > 30s | user, video)) → *proxy label* (a 30s-watch event) → then immediately name the gap between the proxy and the goal. Optimizing clicks gives you clickbait. Saying that unprompted is a strong senior signal.

**The three things candidates always miss:**
1. **Training-serving skew** — the same feature computed two ways in two codepaths. Fix: one feature definition materialized to both an offline and an online store; log the *features actually served* rather than recomputing them at training time.
2. **Feedback loops** — your recommender trains on clicks it caused, so it reinforces what it already shows. Fix: exploration (ε-greedy / bandits), position-bias correction, a small randomized holdout that never sees the model.
3. **Always start with a non-ML baseline.** "Most-popular" is a shockingly strong recommender baseline. Stating it shows engineering maturity, not naivety.

**Label strategy** deserves its own 60 seconds in any design: explicit vs implicit feedback, delayed labels (a fraud chargeback lands 60 days later — so what do you train on today?), negative sampling (a feed shows 20 items; the 19 unclicked ones are weak negatives, not true negatives), and class imbalance.

### 1.2 Driving the clock

45 minutes. If you don't timebox, you run out of time in the deep dive — which is the part that actually gets scored.

```
0–5    Requirements    Business goal, scale, latency budget, explicit non-goals
5–10   Metrics + data  Offline/online metrics, label source, volume
10–20  Pipeline        Data → features → model → serving, end to end
20–35  Deep dive       Interviewer picks, or you offer the interesting one
35–45  Failure modes   Drift, skew, feedback loops, rollback
```

**Say what you're descoping.** "I won't cover the annotation tooling or the cost model unless you want them." That reads as judgment, not laziness.

**Offer the interesting deep dive** rather than waiting: "The real problem here is the delayed-label loop — want me to go into how we train when conversions land 60 days later?"

Deep dive means two or three concrete options, the trade-off, and **your pick with a reason**. Not "we could use A or B," but "A costs more at serving time but avoids skew entirely; given the team size, I'll take A."

### 1.3 Estimation for ML systems

Round aggressively. The point is to answer *does this fit on the hardware we have?*

**The formula:** `avg QPS = DAU × requests_per_day / 100,000` (1 day ≈ 100k s). Peak ≈ 3× avg.

**Embedding index sizing** — the number you'll need constantly:
```
100M items × 768 dims × 4 B (fp32)  = 307 GB   → doesn't fit one GPU/host
                        × 1 B (int8) =  77 GB   → now it does
```
This is why quantization is a design decision, not an optimization. A 4-bit vector quantizer (e.g. TurboQuant) gives ~8× compression; validate it at nDCG parity before claiming it.

**KV cache sizing** — the real constraint in LLM serving:
```
bytes/token = 2 (K,V) × layers × kv_heads × head_dim × dtype_bytes
Llama-3-8B:   2 × 32 × 8 × 128 × 2 B = 128 KB/token
→ 8k context ≈ 1 GiB ≈ 1.07 GB per concurrent sequence
```
So an 80 GB GPU holding a 16 GB model has ~64 GB left ÷ ~1.07 GB ≈ ~60 concurrent 8k sequences. That single calculation answers "how many users per GPU."

**Model weights:** `params × bytes_per_param`. A 7B model is 14 GB at fp16, 7 GB at fp8, 3.5 GB at int4.

**Latency budget** — decompose it out loud. A 200 ms p99 budget might be: 10 ms network, 20 ms feature fetch, 30 ms retrieval, 100 ms ranking, 40 ms slack. Now you know which stage you're allowed to make expensive.

**Reference latencies:**

| Operation | Time |
|---|---|
| Main memory reference | 100 ns |
| Redis / in-memory feature lookup | 0.5–2 ms |
| Same-datacenter round trip | 0.5 ms |
| ANN search over 100M vectors | 5–20 ms |
| GBDT inference (100s of trees) | 1–5 ms |
| Small transformer encoder (GPU) | 5–20 ms |
| LLM time-to-first-token (7B, short prompt) | 50–200 ms |
| LLM per-output-token | 10–30 ms |
| Cross-region round trip | 100–150 ms |

### 1.4 The two-stage retrieval pattern

Nearly every large-scale ranking problem is:
```
Millions of candidates
  → Retrieval / candidate generation (cheap, high recall, ~ms, ANN over embeddings)
  → Ranking (expensive, high precision, hundreds of candidates)
  → Re-ranking (diversity, business rules, freshness, dedup)
```
Two-tower models for retrieval (user tower + item tower; item embeddings precomputed into an ANN index, user tower run at request time), GBDT or a deep cross network for ranking. **This one pattern answers "design YouTube recs," "design feed ranking," "design search," and "design ads."** Learn it once, reuse it in four interviews.

Know the ANN trade-off: **HNSW** (fast, memory-hungry, awkward to update) vs **IVF-PQ** (compressed, tunable recall). And know that retrieval recall must be measured *separately* from final ranking quality, or you'll tune the wrong stage.

### 1.5 ML SD case studies — practice list

> **📄 Worked designs live in [`08-ml-system-design.md`](08-ml-system-design.md)** — recommendations, RAG, LLM serving, NER, fraud, fine-tuning, moderation, and feature store, designed end to end with corner questions.

**Ranking / recsys (Meta & Netflix core)**
1. **Video recommendations** (YouTube/Netflix) — two-stage pattern, cold start, diversity
2. **Feed ranking** (Meta) — multi-objective (click + like + comment + time-spent), value model
3. **Ads CTR prediction** — calibration, auction integration, budget pacing, delayed conversions
4. **Search ranking** — query understanding, BM25 + semantic hybrid, learning-to-rank
5. **People You May Know** — graph features at massive scale

**Classification / trust & safety**
6. **Content moderation** — multimodal, precision/recall trade-off at policy thresholds, human review queue, active learning on hard samples
7. **Fraud / abuse detection** — extreme imbalance, delayed labels, adversarial drift
8. Spam and harmful-content classification

**ML infra / platform**
9. **Feature store** — offline/online parity, point-in-time correctness, backfills
10. **ML training platform** — multi-tenant GPU scheduling, experiment tracking, reproducibility
11. **A/B experimentation platform** — assignment, sticky bucketing, sequential testing, interference
12. **Model serving platform** — multi-model, autoscaling, canary, shadow eval, rollback
13. **Distributed training platform** — data/tensor/pipeline parallelism, NCCL all-reduce, checkpointing, straggler handling, elastic training
14. **GPU cluster scheduler** (Nvidia) — gang scheduling, topology awareness (NVLink islands), preemption, fairshare, fragmentation
15. **ML monitoring system** — drift detection, per-model metrics, cardinality control

**Vision / multimodal**
16. Visual search / image similarity at scale
17. Face recognition at 1M-gallery scale — indexing, threshold calibration, privacy
18. Video understanding pipeline — frame sampling, cost control

---

# 2. Infra Primitives ML Systems Require

Not general system design — only the primitives that show up *inside* ML systems. Google's ML round explicitly probes batch-vs-real-time serving, and you cannot answer "design a feature store" without these.

### 2.1 Batch vs streaming vs real-time — the question you *will* be asked

| Mode | Latency | Use when | ML example |
|---|---|---|---|
| **Batch / precompute** | hours | Candidate set is small and stable; features change slowly | Precomputed recs, nightly embeddings |
| **Streaming** | seconds | Features must reflect recent behavior | Session features, counters, fraud signals |
| **Real-time inference** | ms | Input is only known at request time | Ranking, search, LLM generation |

Almost every production system is **hybrid**: precompute what you can (item embeddings), compute at request time what you must (user context, ranking). Say that, then justify the split by latency budget and freshness requirement. "Precompute unless freshness demands otherwise" is the defensible default.

### 2.2 Feature infrastructure

- **Offline store** (warehouse/lake: Parquet on S3, Athena) for training. **Online store** (Redis/DynamoDB) for serving. **One definition materialized to both** — this is the whole point of a feature store.
- **Point-in-time correctness** — when building training data you must join features *as they were available* at decision/prediction time — by event time **and** ingestion/processing time — not merely what was true then, and certainly not as they are now. That needs bitemporal storage or log-and-wait (see [`08-ml-system-design.md`](08-ml-system-design.md) Case Study 8). Getting this wrong is the most common silent source of leakage.
- **Backfills** — a new feature needs history recomputed. Budget for it.
- **Freshness vs cost** — streaming features cost more; only pay where the model actually gains.

### 2.3 Caching in ML systems

- **What's cacheable:** item embeddings, precomputed candidates, feature vectors, model outputs for repeat queries, **LLM prefix/prompt caches** (huge for agents that resend a long stable system prompt each step).
- **Cache-aside** is the default. Watch **staleness** — a cached feature is a stale feature, and stale features are skew.
- **Stampede / thundering herd** — a popular key expires and every request hits the backend at once. Fix with jittered TTLs, request coalescing (single-flight), and probabilistic early refresh. Matters more when the backend is a GPU.
- **Hot keys** — one viral item, one huge tenant. Replicate the hot key across nodes.

### 2.4 Queues and streams

- **Kafka** (replayable log, partitioned, ordered *within a partition only*) for feature pipelines, training-data collection, and async inference. **SQS-style queues** for competing consumers and job dispatch.
- **At-least-once + idempotency** is the practical default. "Exactly-once" is a claim you should be able to puncture.
- **Idempotency** — make every pipeline step and every agent tool retry-safe, via idempotency keys or dedup windows. Non-negotiable when the caller is a non-deterministic model that may retry.
- **Backpressure** — when inference can't keep up, you must shed or queue, not silently pile up. GPUs make this acute because you can't scale out in seconds.

### 2.5 Storage and sharding for ML

- **Blob store (S3)** — training data, model artifacts, checkpoints. Stream Parquet rather than materializing locally (DuckDB does this well).
- **Vector DB** — ANN index; shard by item ID, replicate for QPS. Rebuild vs incremental update is a real trade-off.
- **Warehouse** (Athena/BigQuery/Snowflake) — offline features, analysis.
- **Sharding** — hash-partition by entity ID for even load; watch for hot tenants. Resharding a live ANN index is painful, so size ahead.
- **Replication lag** = stale online features. Name it before the interviewer does.

### 2.6 Serving reliability

- **Timeouts, retries with exponential backoff and jitter, circuit breakers, bulkheads.** A slow model service will exhaust upstream threads and cascade.
- **Load shedding** — under overload, reject early rather than degrade everyone. For ranking, fall back to a cheaper model or precomputed results; a degraded response beats a timeout.
- **Autoscaling on GPUs** — GPUs take *minutes* to come up and are expensive to idle. Queueing, admission control, and headroom matter far more than in CPU services.
- **Graceful degradation ladder:** full model → smaller/quantized model → cached results → heuristic baseline. Have this ready; interviewers love it.

### 2.7 Observability for ML

- **Service health:** RED (Rate, Errors, Duration) — and always p50/p95/**p99**, never averages.
- **Resource health:** GPU utilization, memory, batch occupancy, queue depth.
- **Model health:** input drift (PSI/KL on feature distributions), prediction drift, **delayed** ground-truth metrics, per-segment performance.
- **SLI / SLO / error budget** — tie rollback decisions to a stated threshold, not to vibes.

### 2.8 Deployment safety for models

Shadow mode (serve traffic, log predictions, don't act) → canary (1% → 5% → 50%) → A/B with guardrail metrics → full rollout, with **one-click rollback to the previous model version**. Model *and* feature-pipeline versions must be pinned together, or you'll roll back the model onto features it never saw.

---

# 2B. Data Pipelines — for ML, for RAG, for Agents

The most under-prepared area in ML interviews, and the one closest to your actual production experience. Every ML system design answer has a data pipeline in it; most candidates wave at it and move on. Don't.

## 2B.1 The ML training data pipeline

```
Sources → Ingest → Validate → Transform → Feature compute → Store (offline + online)
                                                                    ↓
                                                          Training-set assembly
```

**Ingestion.** Batch pulls (JDBC, S3 listings), CDC from operational DBs (Debezium/binlog tailing), event streams (Kafka), third-party APIs. The design question is always *push or pull, and how do you know you got everything?*

**Orchestration.** Airflow / Dagster / Flink. What interviewers actually probe:
- **Idempotency** — rerunning a task must not double-count. Partition by date and overwrite the partition rather than appending.
- **Backfills** — a new feature needs 2 years of history recomputed. Can your DAG run a date range without special-casing? Budget the compute.
- **Retries and partial failure** — one shard fails; do you rerun everything or just that partition?
- **Late-arriving data** — an event lands 3 hours after its timestamp. Watermarks, grace periods, and whether you recompute the affected window.

**Data quality gates — name these unprompted.** A silent data-quality failure degrades a model for weeks before anyone notices, which makes this a *reliability* topic, not a hygiene one:
- Schema validation and **contracts** with upstream producers
- Volume checks (row counts vs expected range), null-rate and cardinality checks
- Distribution checks against a reference window (this is drift detection at the pipeline layer)
- **Freshness SLAs** with alerts — "this table is 6 hours stale" must page someone
- Fail the pipeline loudly rather than training on bad data

**Reproducibility and lineage.** Dataset versioning (Delta/Iceberg time travel, DVC), immutable snapshots so a training run can be re-executed exactly, and lineage from prediction back to the rows that produced it. When a model regresses, the first question is "what changed in the data" — you need to be able to answer it.

**Cost and efficiency.** Columnar formats (Parquet), partition pruning, predicate pushdown, and **streaming rather than materializing locally**. Using DuckDB as cross-source glue — joining DB tables against S3 files and streaming Parquet straight to S3, scaling to millions of rows without ever landing them locally — is a strong, specific answer here.

**Labeling pipelines** (their own design problem): human annotation tooling, inter-annotator agreement, gold/audit sets, **active learning loops that route hard samples to reviewers**, weak supervision, and label versioning. Say how labels get *corrected*, not just created.

## 2B.2 The RAG ingestion pipeline

Retrieval quality is set here, not at query time. This pipeline is where "RAG doesn't work" is actually decided.

```
Parse → Clean → Dedup → Chunk → Enrich → Embed → Index → (incremental sync)
```

**Parse.** PDFs, HTML, tables, scans, images. Layout-aware parsing (Docling, PyMuPDF), OCR for scans, table extraction, and multimodal handling for charts and figures — a small VLM (e.g. Phi-3 Vision on vLLM) handles these. Real corpora are noisy, multi-column, and multi-page; say so.

**Clean and dedup.** Boilerplate stripping, normalization, and **near-duplicate detection** (MinHash/SimHash) — duplicated content poisons retrieval by crowding out diverse results with N copies of the same passage.

**Chunk.** Fixed-size vs semantic vs **structure-aware** (respecting sections, tables, code blocks). Overlap to avoid severing context at boundaries. The trade-off: small chunks retrieve precisely but lose context; large chunks carry context but dilute the embedding. Structure-aware wins on real documents. Attach metadata to every chunk — source, section, timestamp, and **ACLs**.

**Embed at scale.** Batch for GPU throughput, pick the model deliberately (e.g. BGE-M3 dense, ColBERT late-interaction multivectors, bm42 sparse, CLIP for images), and budget the cost: 10M chunks × ~500 tokens is a real compute bill.

**Index and keep it in sync.** This is the part candidates never think about:
- **Incremental upserts and deletes** — deleted source documents must produce tombstones, or you serve content that no longer exists (and may have been deleted for legal reasons)
- **Change detection** — hashes/etags/CDC to find what actually changed instead of reprocessing everything
- **The re-embedding problem** — *changing the embedding model invalidates the entire index.* You must re-embed the whole corpus, then **blue/green swap** to a new index with no downtime. Naming this unprompted is a strong signal; it's the single most expensive operation in a RAG system's life.
- **Freshness SLA** — how long between a document changing and retrieval reflecting it?
- **ACL propagation** — permissions captured at ingest, enforced at query (§3.4)

**Evaluate the pipeline itself,** separately from the LLM: a golden set of queries with known relevant documents, measured on retrieval **recall@k** and nDCG. Validating a quantized index at nDCG/recall parity across BEIR datasets is exactly this discipline.

## 2B.3 Agentic data pipelines — LLM-authored ETL

The frontier version. The design question: **who writes the transformation logic — a human, or the model?**

| | Hard-coded recipes | **LLM-authored steps** |
|---|---|---|
| Coverage | Only anticipated cases | Arbitrary new sources/schemas |
| Reliability | Deterministic | Needs validation gates |
| Maintenance | Grows without bound | Prompt + tools |
| Failure mode | Silent gap | Wrong code — *catchable* |

The senior insight: **an LLM writing per-step code is only safe if every step is validated before it commits.** The pattern that makes it work:

1. **Discovery/profiling first** — inspect schemas, types, null rates, sample rows; cache this (content-addressed) so repeated runs are cheap
2. **Plan** — decompose the natural-language task into typed steps
3. **Sample-first execution** — run each generated step against a small sample, validate the output shape and row counts, *then* run at full scale
4. **Confidence gating** — low confidence routes to human clarification instead of guessing
5. **Streaming execution** — cross-source joins that never materialize locally (DuckDB as the glue; Parquet straight to S3)
6. **Sandboxed execution** — §3.2; LLM-authored code is untrusted code
7. **Checkpoint per step** — a failure at step 7 resumes at step 7

**What interviewers probe:** how you validate generated code, what happens when it's wrong, how you bound cost when a plan explodes into 40 steps, and how you make the whole thing resumable. Strong answers are concrete: validator gates, AST allow-lists, sample-first execution, externalized run state (e.g. Redis), and ownership routing so cancel/resume works across pods.

## 2B.4 Data pipeline case studies

1. **A training data pipeline for a ranking model** — CDC → events → features → point-in-time training sets
2. **A RAG ingestion pipeline over millions of PDFs**
3. **A re-embedding / index migration** — zero-downtime swap at corpus scale
4. **A labeling + active-learning loop**
5. **An agentic ETL system** — natural language → validated, sandboxed, streaming pipeline
6. **A data quality / drift monitoring system** — checks, thresholds, alerting, and who gets paged

---

# 3. Agentic & LLM System Design — Now Table Stakes

As of 2026 this is **expected**, not bonus: ML design rounds routinely include a RAG, LLM-gateway, or agent-orchestration prompt (§5B.3). Knowing it no longer differentiates you — *not* knowing it disqualifies you.

**What still differentiates you is operational depth.** Most candidates have read about agents; few have run one in production. If you have, lead with failure modes and evaluation methodology — that's the part nobody can fake.

### 3.1 The agentic system design framework

```
1. Task & autonomy    What decides what? Where's the human? Autonomy ladder.
2. Decomposition      Single agent vs planner/executor vs multi-agent — and WHY
3. Action space       Tool design, schemas, error surfaces, idempotency
4. Context & memory   Context budget, retrieval, state externalization
5. Control loop       ReAct / plan-execute / reflect; termination; step budget
6. Safety             Sandbox, least privilege, egress, prompt injection
7. HITL               Where to pause, approval gates, pause/resume mechanics
8. Reliability        Retries on stochastic failure, checkpointing, partial progress
9. Evaluation         Outcome vs trajectory eval, golden tasks, regression suites
10. Cost & latency    Token budget, model routing/cascade, caching, parallelism
11. Observability     Step-level tracing, replay, debugging non-determinism
```

### 3.2 The judgment calls interviewers probe

**"Why multi-agent instead of one agent with more tools?"**
There are two primary defensible reasons, and most candidates can't name them:
1. **Context isolation** — a sub-agent burns its own context on a messy subtask and returns a clean summary, keeping the orchestrator's context uncontaminated.
2. **Parallelism** — genuinely independent subtasks run concurrently.

Plus a third, security one:
3. **Permission separation** — sub-agents with genuinely different privileges (a read-only "reader" vs a narrowly-scoped "writer") form a real security boundary, not just an organizational one (see [`06-agent-architecture.md`](06-agent-architecture.md) §3.4).

Everything else ("separation of concerns," "specialist prompts") is usually better solved with one agent and better tools. **Say this.** Arguing against the fashionable answer where warranted is a strong signal. Then name the costs: multi-agent multiplies token spend, makes failures harder to attribute, and introduces coordination bugs.

**"How do you make an agent reliable when the model is non-deterministic?"**
You don't make the *model* reliable — you make the *system* reliable around it:
- Externalize state so any worker can resume (e.g. a Redis run store)
- Make every tool idempotent, or key it so a retry is safe
- Checkpoint after each step so a failure resumes rather than restarts
- Validate before committing — sample-first execution, validator gates
- Bound the loop: max steps, max tokens, max wall-clock, plus a no-progress detector

**"How do you stop it doing something destructive?"**
Capability-based, not prompt-based. Most candidates answer "a system prompt telling it not to." The right answer is that **prompt-level restrictions are not a security boundary**, because the input is untrusted:
- Isolated sandbox for model-generated code — **gVisor or a Firecracker microVM by default**, scratch-confined filesystem. An OS namespace sandbox (bubblewrap) shares the host kernel, so it is a deliberate latency trade-off, acceptable only with the layers below (egress broker, no credentials inside, read-only DB, AST allow-list). See [`06-agent-architecture.md`](06-agent-architecture.md) §10 and [`07-agentic-system-design.md`](07-agentic-system-design.md) Case Study 3.
- Egress broker — network allow-list, not open internet
- Session-scoped, least-privilege credentials (STS) that expire
- Read-only database paths for anything analytical
- Static analysis of generated code (AST allow-list) before execution
- Human approval gates on irreversible actions

**"How do you evaluate it?" — the question that separates levels.**
- **Outcome eval** — did the final artifact match ground truth? Needs a graded task registry over real datasets, with order-insensitive, type-tolerant comparison.
- **Trajectory eval** — was the *path* sane? Tool-call correctness, wasted steps, cost per task.
- **LLM-as-judge** — and its known biases: position, verbosity, self-preference. Mitigate with pairwise comparison, randomized order, explicit rubrics, and a human-labeled calibration set.
- **Regression suites in CI** — catch the release that quietly drops pass rate from 86% to 71%.
- Report **pass rate + mean F1 + cost + p95 latency together**. A quality number alone is not an eval.

### 3.3 LLM serving & inference (Nvidia's home turf)

> **📄 Depth in [`05-inference-serving.md`](05-inference-serving.md); worked design in [`08-ml-system-design.md`](08-ml-system-design.md) Case Study 3.**

- **Two SLOs, not one:** TTFT (time to first token — prefill-bound, compute-heavy) vs TPOT (time per output token — decode-bound, **memory-bandwidth-bound**). Batching raises decode *throughput* enormously and does nothing for TTFT; per-request TPOT stays flat or gets slightly worse. This distinction alone marks you as someone who has served models.
- **Continuous / in-flight batching** — the single biggest throughput win over static batching
- **PagedAttention / KV cache** — the real memory constraint at long context; paging fixes fragmentation. Size it with the §1.3 formula.
- **Parallelism:** tensor (intra-layer, needs NVLink-class interconnect), pipeline (inter-layer, bubble overhead), expert (MoE), data
- **Speculative decoding** — draft model proposes, target verifies; wins when acceptance rate is high
- **Quantization:** FP8 / INT4 / AWQ / GPTQ — throughput and memory vs quality. FP8 is the safe production default on Hopper-class GPUs.
- **Prefix / prompt caching** — enormous for agents, which resend a long stable system prompt every step
- **Autoscaling on a scarce, slow-to-boot resource** — see §2.6

### 3.4 RAG at production scale — the query path

Ingestion is §2B.2. This is what happens per request.

> **📄 Worked design in [`08-ml-system-design.md`](08-ml-system-design.md) Case Study 2; retrieval components in [`06-agent-architecture.md`](06-agent-architecture.md) §7.**

```
Query → Rewrite/expand → Hybrid retrieve → Filter (ACL) → Rerank → Assemble context → Generate → Cite
```

**Query understanding.** Raw user queries are bad search queries. Rewriting for context (resolving "it" and "that one" against conversation history), decomposing multi-part questions, and optionally HyDE (embed a *hypothetical answer* rather than the question, since answers live closer to answers in embedding space).

**Hybrid retrieval.** Dense (semantic) + sparse/BM25 (exact terms, rare tokens, IDs, error codes) fused with **Reciprocal Rank Fusion**. Dense alone fails on exact identifiers; sparse alone fails on paraphrase. Dense (e.g. BGE-M3) + sparse (e.g. bm42) + ColBERT late-interaction together *is* the strong answer.

**Filtering.** Metadata filters (date, source, type) and **ACL filtering** — the hard part nobody mentions. In an enterprise corpus, permissions must be enforced at query time without destroying recall, and without leaking that a document exists. Pre-filtering is correct but can gut ANN recall; post-filtering is cheap but may return an empty page. The usual answer: partition the index by coarse permission group, then post-filter fine-grained.

**Rerank.** A cross-encoder or ColBERT late-interaction pass over the top ~100 candidates. This is where most of the quality gain lives, and it costs one extra model call — a trade-off worth naming explicitly.

**Context assembly.** Token budget allocation, deduplication across chunks, ordering (models attend unevenly across long contexts — put the strongest evidence at the edges, not buried mid-context), and carrying source IDs through so citations are verifiable rather than generated.

**Evaluate the two stages separately:**
- *Retrieval:* recall@k, nDCG, MRR against a golden query set
- *Generation:* faithfulness/groundedness (is every claim supported by retrieved context?), answer relevance, citation correctness

**The failure mode to name unprompted: most "RAG hallucinations" are retrieval failures, not generation failures.** If the right chunk never arrived, no prompt engineering saves you. Instrument both stages or you'll spend a month tuning the wrong one.

**Agentic RAG** — the 2026 variant. Instead of one retrieve-then-generate pass, the model decides *whether* to retrieve, issues multiple targeted searches, and iterates until it has enough. Better recall on complex questions; costs more tokens and latency, and needs a step budget. Know when the simple pipeline is the right answer — usually it is.

### 3.5 Agentic case studies to practice

> **📄 Worked designs live in [`07-agentic-system-design.md`](07-agentic-system-design.md)** — agentic case studies designed end to end with corner questions (written so far: customer-support agent, agent memory, coding agent, evaluation & guardrails platform). Use that file for practice; use this section as the index.

1. **An agent answering analytical questions over a company's databases and files** ← *strong showcase if you've built one*
2. **A coding agent** — repo context, edit-verify loop, test execution, sandboxing ← *worked in `07` Case Study 3*
3. **A customer-support agent** — tool use against real systems, escalation, HITL, containment rate ← *worked in `07` Case Study 1*
4. **A deep-research agent** — parallel sub-agents, source dedup, synthesis, citation
5. **An agent evaluation platform** ← *worked in `07` Case Study 4*
6. **An LLM gateway / router** — model cascade, caching, rate limits, fallback, cost attribution
7. **A RAG assistant over enterprise docs** — permissions-aware retrieval ← *worked in [`08-ml-system-design.md`](08-ml-system-design.md) Case Study 2*
8. **A multimodal document-extraction pipeline** — layout-aware parsing (Docling), OCR, VLMs for figures
9. **An agent memory system** — what to write, consolidation, retrieval, forgetting ← *worked in `07` Case Study 2*

> **Prepare two showcase stories from your own work — rehearse both to 4 minutes, with numbers. Good shapes:**
> **(a) Safe code-executing agent** — sandbox (gVisor/Firecracker microVM by default; a bubblewrap namespace sandbox only as a deliberate latency trade-off backed by the other layers), egress broker, session-scoped credentials, read-only DB gate, AST allow-list, sample-first validator gating. Almost nobody has this answer.
> **(b) Agent evaluation at scale** — graded tasks on real data, an order-insensitive programmatic grader, LLM root-cause adjudication of genuine errors vs spec-variants, pass rate and mean F1 reported together, an LLM auto-responder driving HITL unattended.
> Both are *systems* answers to questions most candidates answer with vibes.

---

# 4. Company-by-Company

### Google — ML SWE / ML Infrastructure (L4 / L5)
- **Rounds (2025 baseline):** **3 coding**, 1 ML system design, 1 Googlyness & Leadership, then team matching. Coding is *three of five* technical rounds even for ML roles.
- **2026 changes (§5B.1):** a **code-comprehension round** with Gemini may replace a traditional coding round, and **Googleyness now includes a technical design conversation** about your prior work. Rollout is partial — prepare for both formats.
- **Coding topics that recur:** graphs (topological sort), DP with memoization, backtracking + caching, sliding window, heaps. Medium/Hard, Google-tagged.
- **ML SD round is the classic framework (§1.1)**, not agentic: model serving architecture, feature pipelines & feature stores, A/B testing for models, monitoring + retraining triggers, **batch vs real-time inference trade-offs** (§2.1).
- **Style:** Fundamentals-first. Correctness, precise complexity analysis, clean compilable code.
- **Lineage worth knowing:** TFX, Borg, Vertex AI, Pathways.
- **Prep weight:** **Coding 40% · ML SD 30% · Behavioral 20% · ML depth 10%.**
- **Trap:** two "lean hire" coding scores triggers an extra coding round from the hiring committee. Coding is the gate that actually fails ML candidates here.

### Meta — ML Engineer (E5 / E6)
- **Rounds (2026):** **1 traditional coding + 1 AI-assisted coding + 1 ML design + 1 behavioral.** The AI-assisted round is the big change — §5B.1/§5B.2.
- **Style:** Fast, candidate-driven, ~35 effective minutes. They score **driving** — if you wait to be led, you fail. Depth on 1–2 components beats breadth across ten.
- **ML topics:** recsys and ranking dominate — feed ranking, ads CTR, retrieval. Multi-objective optimization, calibration, real-time features.
- **Lineage:** FBLearner, PyTorch, DLRM.
- **Prep weight:** Coding (incl. AI-assisted) 35% · ML SD 35% · Behavioral 15% · Systems 15%.
- **Trap:** being too broad. Meta wants a clear decision and a defense of it.

### Netflix — ML / Personalization / ML Platform (Senior / Staff)
- **Style:** The most conversational of the four. Fewer puzzles, more "tell me about a system you owned, what broke, what you changed." They screen hard for **judgment and ownership**.
- **Topics:** personalization and recsys, A/B experimentation at scale (they are the industry benchmark), streaming data (Kafka/Flink), ML platform resilience.
- **Prep weight:** Production war stories 40% · ML SD 30% · ML infra 20% · Coding 10%.
- **Trap:** vagueness about your own work. They will drill into *your* incidents. Have exact numbers.

### Nvidia — ML Infra / Inference / Training (varies hugely by org — ask your recruiter)
- **Style:** Performance-obsessed and hardware-aware. C++ common in some orgs; Python fine for ML/DL software teams.
- **Topics:** §3.3 is the syllabus — memory-bound vs compute-bound, arithmetic intensity, NCCL collectives, tensor/pipeline parallelism, Triton & TensorRT-LLM, KV cache, continuous batching, NVLink/InfiniBand topology.
- **Prep weight:** Domain depth (GPU/ML systems) 40% · Coding 30% · ML SD 20% · Other 10%.
- **Trap:** treating the GPU as a black box. Be able to say "decode is memory-bandwidth-bound, so batching raises throughput until we saturate HBM — it doesn't help TTFT."

---

# 5A. Calibration: a real Google L5 ML-Infra loop

A successful L5 hire (6 YOE, ML/AI, US, ML Infrastructure) reported this loop. Treat it as **n=1** — it was shared in a post that also advertises a paid prep product — but it matches the general Google ML pattern.

| Round | Content | Their outcome |
|---|---|---|
| 1–2 | Coding — backtracking w/ caching, sliding window variation | **Lean hire ×2** |
| 3 | Coding — topological sort | Strong positive |
| 4 | **ML system design (45 min, NLP application)** | Strong positive |
| 5 | Googlyness & Leadership — pure STAR behavioral | Strong positive |
| — | Team matching: 2 conversational calls, ~1 month | Passed |
| Extra | Hiring committee added a 4th coding round over the two lean-hires | Passed |

Result: L5, not down-leveled, ~20% comp bump. Timeline ~3 months. Prep was 2–3 weeks, 2–3 LeetCode problems/day, Google-tagged Medium/Hard, focused on graphs / DP+memo / sliding window.

**Three things this should change:**

1. **Coding is heavier than ML candidates usually assume.** It is easy to rate coding "Medium — still a gate, but rarely sets your level." For a Google ML loop that is **understated**: 3 of 5 rounds (4 of 6 after the committee), and the *only* area scoring below "strong." Note the asymmetry — coding rarely levels you up, but it's the most common way to get rejected or down-leveled.

2. **The ML SD round is the classic framework, not the agentic one.** The probes map exactly onto §1.1. Your agentic depth is your differentiator *in the deep dive and behavioral* — not a substitute for knowing feature stores and retraining triggers cold.

3. **Behavioral is a full scored round.** Googlyness covers cross-functional collaboration, ambiguity, conflict, technical decisions with business impact, and mentorship. Five distinct stories. Write them out.

> ⚠️ **Caveat:** this loop ran in **2025**. The format changed in 2026 — see §5B. Accurate on *content*; outdated on *format*.

---

# 5B. What Changed in 2026 — AI in the Interview Itself

Researched July 2026. The biggest shift in technical hiring in a decade, and it cuts in two opposite directions at once.

## 5B.1 Some companies now hand you an AI; others are cracking down

**Google.** Piloting an AI-assisted coding interview with **Gemini** as the approved assistant. Three changes to the loop:
- A new **"code comprehension" round** — read, debug, and optimize an *existing codebase* with Gemini available. Described internally as "human-led, AI-assisted."
- **Googleyness & Leadership now includes a technical design conversation** about your own prior engineering work. No longer purely behavioral.
- An open-ended, ambiguous-problem round (junior candidates).

Scored on **"AI fluency": prompt engineering, output validation, debugging** — using AI for well-defined subtasks while retaining ownership. Pilot is junior/mid-level on select US teams; standardization estimated at 12–18 months, so **prepare for both formats**. Context: Sundar Pichai stated in April 2026 that ~75% of new code at Google is AI-generated and human-approved, up from 50% the previous autumn.

**Meta.** Further along. Since an October 2025 pilot, **one of the two onsite coding rounds is AI-enabled** for a growing share of candidates:
- 60 minutes in a special CoderPad with the assistant embedded
- Model roster as reported: **Claude Sonnet 4/4.5, Opus 4, GPT-5, Gemini 2.5 Pro, Llama 4 Maverick** (candidates report picking Claude Sonnet or Gemini; GPT-5 has been slow there). *The roster changes — ask what's available and pick the strongest model you know how to drive.*
- **Three phases:** ① fix bugs in an existing codebase (type-casting, off-by-one, bad conditionals) → ② core implementation, 120+ lines, BFS/DFS/backtracking/greedy → ③ optimize for larger inputs
- Scored on **Problem Solving · Code Quality · Verification · Communication**. Using the AI is optional and *frequency of use is not scored* — judgment and verification are.
- **The Meta MLE onsite is now: 1 traditional coding + 1 AI-assisted coding + 1 ML design + 1 behavioral.**

**The counter-trend — a crackdown everywhere else.** AI-assisted cheating roughly doubled from 15% → 35% of candidates between June and Dec 2025; one analysis of 19,368 interviews in early 2026 flagged **38.5%** for suspected AI assistance. In response: Google reinstated at least one **in-person** round, Amazon requires a **signed no-unauthorized-AI pledge**, and Gartner reports **72.4%** of recruiting leaders now interview in person. Proctoring watches tab-switching, keystroke rhythm, and gaze.

> **Operating rule: never use AI in an interview unless it is explicitly offered.** The upside is small; being flagged ends the loop and can blacklist you. When it *is* offered, using it well is scored — and refusing to touch it is a missed signal.

## 5B.2 How to perform in an AI-enabled coding round

This rewards daily AI-assisted coding practice — but only if you avoid the standard failure modes.

**What gets you dinged:**
- Prompting *"solve this problem"* instead of aiming at a specific subtask
- Accepting generated code you can't explain — the interviewer **will** ask you to walk through it
- Skipping the read-through in Phase 1 and debugging blind
- Over-polishing Phase 1 and never reaching Phase 3, where differentiation lives
- Going silent while reading AI output

**What scores:**
- Say your approach out loud *before* you prompt anything
- Use AI for mechanical bulk (boilerplate, a known data structure, test scaffolding); keep the algorithmic choice yourself
- Write information-rich prompts with full context, constraints, and the signature you want
- **Verify visibly** — run it, add edge cases, say "I don't trust this branch, let me test empty input"
- Recognize the algorithm yourself. AI fluency doesn't substitute for knowing it's a topological sort; it accelerates you *after* you know.

## 5B.3 What changed in the ML design round

The classic framework (§1.1) is still the backbone, but loops now commonly include a GenAI component. Newly standard prompts:

- **A RAG pipeline for enterprise document search** (§3.4)
- **An LLM gateway with rate limiting and cost allocation** (§3.5 #6)
- **A multi-tenant agent orchestration platform** (§3.5 #1)
- Real-time fraud detection with streaming inference

For agentic prompts they specifically probe **tool use and function calling** (how the model selects and validates tools), **multi-agent orchestration** (when to split work and how agents coordinate), and **observability** (tracing, debugging, and measuring an agent in production).

The most quoted line from 2026 guides: ***"evaluation methodology is the new system design."*** Interviewers now weight cost, latency, guardrails, and monitoring above the architecture diagram. If you have built an eval system, **lead with it.**

## 5B.4 Net effect on your prep

| Shift | What you do differently |
|---|---|
| AI-enabled coding rounds at Meta (coming at Google) | Add the §6 drill. Practice reading/debugging unfamiliar code — Meta's Phase 1 and Google's whole comprehension round. |
| Coding still the top rejection reason | The daily 2-problem drumbeat stays. Algorithmic recognition still gates you. |
| Googleyness now has a technical design conversation | Your STAR stories need a whiteboard version. Prep to diagram your main system from memory. |
| ML design absorbed GenAI/agentic | §3 is table stakes, not bonus. Differentiate on failure modes and eval. |
| Eval > architecture | Rehearse the eval story as a first-class system design, not a footnote. |
| In-person rounds returning | Practice on a physical whiteboard. Budget for travel/visa logistics. |

---

# 6. An 8-Week Plan

Assumes ~2 hr/weekday + a 3 hr weekend block. **Every weekday: 2 LeetCode problems, non-negotiable** — the drumbeat under everything else, because coding is the most common rejection reason (§5A). The remaining ~1 hr/day goes to the design column.

| Week | Daily coding (~45–60 min) | Design focus (~1 hr) | Weekend deliverable |
|---|---|---|---|
| 1 | Arrays, hashing, two pointers, sliding window | **§1.1 framework** — memorize the 10 steps; §1.3 estimation | 2 timed ML SD: content moderation, video recs |
| 2 | Graphs: BFS/DFS, **topological sort** | §2.2 feature infra + **§2B.1 ML data pipelines** (idempotency, backfills, quality gates) | Timed: feature store + training pipeline |
| 3 | DP + memoization, backtracking + caching | A/B testing, §2.7 monitoring, drift, retraining triggers | Timed: A/B experimentation platform |
| 4 | Heaps, intervals, binary search | **§3.1 agentic framework** + **§2B.3 agentic ETL** + your 2 showcase stories | Timed: agentic data-analyst system |
| 5 | Trees, tries, mixed review | **§2B.2 RAG ingestion** + §3.4 RAG query path + §3.2 eval | Timed: enterprise RAG, ingestion **and** query, ACL-aware |
| 6 | Google-tagged Medium/Hard, timed 45 min | §3.3 LLM serving — TTFT/TPOT, batching, KV cache | Timed: LLM inference platform (Nvidia-flavored) |
| 7 | Google-tagged Hard, timed | §2 infra primitives + §2.1 batch vs real-time + **5 STAR stories written out** | Timed: feed ranking + behavioral dry run |
| 8 | 1 timed problem/day, stay warm | — | **5 full mocks, recorded** (2 ML SD, 1 traditional coding, 1 **AI-enabled** coding, 1 behavioral) |

**Rules that make this work:**
- **70% of design time is timed, out-loud practice. Reading is 30%.** Design knowledge you can't narrate under pressure scores zero.
- **Coding: patterns, not volume.** ~120 well-chosen Google/Meta-tagged Medium/Hard beats 400 random. Redo failures after 3 days.
- **One day a week, run the AI-enabled drill** (§5B.2) instead of solo coding: 45 min, Claude open, state approach before prompting, prompt only for subtasks, explain every generated line aloud.
- **Once a week, do a code-comprehension rep:** open an unfamiliar OSS repo (vLLM, CrewAI, LangChain), pick a file, explain what it does and where you'd add a feature. Google's new round and Meta's Phase 1 — almost nobody practices it.
- **Record yourself** on at least 4 designs. You'll hear the hedging and option-listing immediately.
- Write the 5 Googlyness stories by end of week 7 — STAR, with numbers, **each with a whiteboard-diagram version** (§5B.4).

---

# 7. Scoring Rubrics

Interviewers fill in something close to these. Optimize for them directly.

### 7.1 Any design round

| Signal | Weak | Strong |
|---|---|---|
| Requirements | Jumps to a solution | Scopes, quantifies, states non-goals |
| Estimation | "It'll be a lot" | Numbers on the board, driving the design |
| Structure | Wanders | Visible framework, manages the clock |
| Trade-offs | Lists options | **Picks one and defends it** |
| Depth | Stays at box level | Goes 3 levels into one component |
| Failure thinking | Only happy path | Unprompted failure analysis |
| Communication | Silent or rambling | Thinks aloud, checks in, adapts |
| Ownership | "You'd use Kafka" | "I'd use Kafka **because** ordering per user matters" |

The biggest score jump available: **stop enumerating options and start making decisions.** "We could use X or Y" is mid-level. "I'll use X because of constraint Z; I'd switch to Y if Z changed" is senior.

### 7.2 ML system design — additional signals

| Signal | Weak | Strong |
|---|---|---|
| Problem framing | Starts with the model | Business goal → ML objective → proxy label → names the gap |
| Baseline | Jumps to a transformer | Non-ML baseline first, justifies each step up |
| Data & labels | Assumes clean labels | Label source, delay, imbalance, negative sampling, leakage |
| Data pipeline | "We'd ETL it" | Idempotent, backfillable, quality-gated, versioned for reproducibility |
| Data quality | Assumed good | Named checks + freshness SLA + who gets paged when it breaks |
| Features | Lists features | Feature store, **offline/online parity**, point-in-time correctness |
| Training-serving skew | Never mentioned | Raised unprompted, with a concrete fix |
| Evaluation | Offline metric only | Offline → shadow → A/B → guardrails, and what triggers rollback |
| Feedback loops | Ignored | Names the loop, proposes exploration / bias correction |
| Serving | "Deploy the model" | Latency budget, batch vs real-time **justified**, caching, quantization |
| Monitoring | "We'd monitor it" | Specific drift signals, thresholds, retraining triggers |

### 7.3 Agentic / LLM design — the 2026 signals

| Signal | Weak | Strong |
|---|---|---|
| Architecture choice | "Multi-agent, obviously" | Justifies single vs multi on **context isolation / parallelism**, names the cost |
| RAG quality | "We chunk and embed" | Structure-aware chunking, hybrid + rerank, **re-embedding/index migration plan** |
| Retrieval vs generation | Blames "hallucination" | Instruments the two stages separately; knows most failures are retrieval |
| Safety | "A system prompt tells it not to" | **Capability-based:** sandbox, egress broker, scoped creds, approval gates |
| Reliability | "Retry on failure" | Externalized state, idempotent tools, checkpointing, step/token/time bounds |
| Evaluation | "We'd use an LLM judge" | Outcome **and** trajectory eval, golden tasks, judge-bias mitigation, CI regression |
| Cost & latency | Not mentioned | Token budget, model cascade, prefix caching, cost per task tracked |
| Observability | "We'd log it" | Step-level tracing, replay, failure attributed to a specific tool call |

### 7.4 AI-enabled coding round

| Signal | Weak | Strong |
|---|---|---|
| Ownership | "Solve this problem" prompt | Names the algorithm first, prompts only for subtasks |
| Verification | Accepts output, moves on | Runs it, adds edge cases, says what they don't trust |
| Comprehension | Can't explain generated code | Walks any line on request |
| Pacing | Perfects Phase 1, never reaches Phase 3 | Budgets time to reach optimization |
| Communication | Silent while reading output | Narrates what they asked for and why |

---

# 8. Resources

**ML / agentic — highest priority**
- **Designing Machine Learning Systems** — Chip Huyen. The §1.1 framework, properly.
- **AI Engineering** — Chip Huyen. The GenAI-era companion: eval, RAG, serving.
- [alirezadir/Machine-Learning-Interviews](https://github.com/alirezadir/machine-learning-interviews) — best free ML system design repo; worked case studies.
- [Exponent — ML system design guide (2026)](https://www.tryexponent.com/blog/machine-learning-system-design-interview-guide) · [Meta MLE guide](https://www.tryexponent.com/guides/meta-machine-learning-engineer-interview)
- **Eval-focused reading** — since "evaluation methodology is the new system design" (§5B.3), read vendor eval docs (Anthropic, OpenAI) and LLM-as-judge bias papers.
- **vLLM docs** — PagedAttention, continuous batching. You use it; read the internals for §3.3 vocabulary.

**2026 format changes — read before your first loop**
- [Hello Interview — Meta's AI-enabled coding interview](https://www.hellointerview.com/blog/meta-ai-enabled-coding) — most detailed breakdown of the three phases
- [interviewing.io — how to use AI in Meta's round, with real prompts](https://interviewing.io/blog/how-to-use-ai-in-meta-s-ai-assisted-coding-interview-with-real-prompts-and-examples)
- [Exponent — Google's AI-assisted coding interview (2026)](https://www.tryexponent.com/blog/google-ai-coding-interview)
- [Computerworld — in-person interviews return](https://www.computerworld.com/article/4044734/to-counter-ai-cheating-companies-bring-back-in-person-job-interviews.html) · [Fabric — state of AI interview cheating 2026](https://fabrichq.ai/blogs/state-of-ai-interview-cheating-in-2026-insights-from-19-368-interviews)

**Infra fundamentals (only as far as §2 needs)**
- **Designing Data-Intensive Applications** — Kleppmann. Chapters 5–9 cover replication, partitioning, consistency, and batch/stream processing. Read for §2, not cover to cover.
- **The Google SRE Book** — SLOs, cascading failure, load shedding (free online).

**Coding**
- LeetCode Premium, **Google/Meta-tagged Medium/Hard**. Patterns over volume (§6).
- NeetCode 150 for structured pattern coverage.

> **Source note:** §5B reflects reporting as of **July 2026** on formats still rolling out. Model rosters, pilot scope, and round structure are all moving. **Ask your recruiter what your specific loop contains** — every company will tell you, and it's the only source that's actually current for you.
