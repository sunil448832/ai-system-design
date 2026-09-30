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

Figures such as model configs, hardware specs and interview formats are accurate as of 2026. Check them against primary sources before you quote them.
