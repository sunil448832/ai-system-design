# AI System Design

Study notes for senior ML / AI system design interviews (Google, Meta, Netflix, Nvidia). They cover the LLM stack from the model's internals up to production agents, plus worked interview case studies.

## Files

| File | What it covers |
|---|---|
| [00-system-design-curriculum.md](00-system-design-curriculum.md) | Interview syllabus and plan: ML and agentic design frameworks, infra primitives, data pipelines, company-by-company notes, 2026 format changes, an 8-week plan, scoring rubrics |
| **Foundations: the model** | |
| [01-transformer-architecture.md](01-transformer-architecture.md) | Transformer internals with math, shapes and code: tokenization, RoPE, attention, GQA/MLA, FFN, KV cache, prefill vs decode, FLOPs, MoE, vision encoders |
| [02-modern-transformer-architectures.md](02-modern-transformer-architectures.md) | How Llama 3, Qwen3, DeepSeek-V3, Mistral, Gemma and GPT-OSS instantiate those components, and what each choice costs at serving time |
| **Training** | |
| [03-training-data-pipeline.md](03-training-data-pipeline.md) | Pretraining data: collection, extraction, filtering, dedup, decontamination, mixing, packing, long-context data |
| [04-training-methods.md](04-training-methods.md) | Losses and optimizers; SFT, LoRA, reward models, PPO, DPO, GRPO/RLVR, distillation; parallelism; 27B and 2T training runs |
| **Serving** | |
| [05-inference-serving.md](05-inference-serving.md) | GPU hardware, memory budgeting, batching, PagedAttention, caching, quantization, speculative decoding, disaggregation, parallelism, cost model, engine comparison |
| **Agents** | |
| [06-agent-architecture.md](06-agent-architecture.md) | Part I: every component of a production agent (context, control loop, tools, memory, guardrails, sandboxing, reliability). Part II: agent evaluation |
| **Interview case studies** | |
| [07-agentic-system-design.md](07-agentic-system-design.md) | Agentic designs: customer support agent, agent memory, coding agent, eval & guardrails platform. Case studies 5–7 are not written yet |
| [08-ml-system-design.md](08-ml-system-design.md) | ML designs: recommendations, enterprise RAG, LLM inference platform, NER/extraction, fraud, fine-tuning platform, content moderation, feature store |

## How to use

- **Preparing for interviews:** start with `00` for the plan and frameworks, then practise with `07` and `08`.
- **Building depth:** read `01` → `05` in order. Each builds on the one before, and `05` relies on the KV-cache and prefill/decode material in `01`.
- **Agent work:** read `06`, then practise with `07`.

## The typical senior ML loop

Expect 4–6 stages over 4–8 weeks. Formats vary by company and are changing fast (AI-assisted coding, code-comprehension rounds, a return to in-person); see `00` §4–§5B.

| Stage | Length | What it assesses |
|---|---|---|
| Recruiter screen | 30 min | Background, motivation, compensation, logistics |
| Technical phone screen | 45–60 min | Coding, sometimes a short ML conversation |
| Coding ×2–3 | 45 min each | Data structures and algorithms; a gate that rarely sets your level but often causes rejections |
| ML system design | 45–60 min | Data → features → model → serving → monitoring; **sets your level** |
| ML breadth / depth | 45–60 min | Your projects, modelling trade-offs, maths intuition |
| Behavioral | 45 min | Ownership, conflict, incidents, mentoring (STAR stories with numbers) |
| Problem scoping *(forward-deployed / applied roles)* | 45–60 min | Turning a vague customer need into a scoped technical plan |
| Hiring manager, team match | — | Fit, then offer |

## More resources

`00` §8 lists the core reading. Also useful:

- *Machine Learning System Design Interview* — Ali Aminian and Alex Xu. Worked designs in interview format.
- [System Design Primer](https://github.com/donnemartin/system-design-primer) — distributed-systems fundamentals for the classic design round.
- [Anthropic — Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) — workflow vs agent patterns; pairs with `06`.
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course) — hands-on transformers, fine-tuning and RAG; pairs with `01`–`04`.
- [NeetCode](https://neetcode.io) — pattern-based coding practice.
- [IGotAnOffer — FAANG interview questions](https://igotanoffer.com/blogs/tech/faang-interview-questions) — question banks for SWE, MLE and AI roles.

Figures such as model configs, hardware specs and interview formats are accurate as of 2026. Check them against primary sources before you quote them.
