---
title: "Design Patterns for Python, ML & AI — The Complete Guide"
permalink: /
---

# 🏗️ Design Patterns for Python, ML & AI — The Complete Guide

> **From Principles to Production**: The most important software design patterns — GoF classics, Pythonic idioms, PyTorch-specific patterns, MLOps architecture patterns, and the emerging vocabulary of agentic AI systems. A reference for writing ML code that is reusable, testable, and production-ready.

## Who This Is For

You write Python. You train models. Maybe you've shipped something to production. But **design patterns** — the proven vocabulary of software architecture — are what separate a collection of scripts from a system you can reason about, test, and extend. This guide covers the complete landscape: from the Gang of Four patterns applied to real ML code, through PyTorch-specific idioms and MLOps architecture, to the design patterns behind modern LLM pipelines and AI agent systems. This is the companion to the [Software Engineering for ML](https://github.com/karam-nus/software) guide — where that guide covers process and tooling, this one covers structure and design.

## 📋 Table of Contents

| # | Chapter | What You'll Learn |
|---|---------|-------------------|
| **Foundations** | | |
| 1 | [Introduction to Design Patterns](./01_introduction.md) | What patterns are, GoF history, pattern taxonomy (creational / structural / behavioral), how they apply to ML and AI |
| 2 | [SOLID Principles in Python](./02_solid_principles.md) | All five SOLID principles with ML-specific examples — SRP in dataset classes, OCP in model heads, LSP in transforms, ISP in callbacks, DIP in trainers |
| 3 | [Python Idioms & Protocols as Patterns](./03_python_idioms.md) | Context managers, descriptors, generators, ABCs, `typing.Protocol`, dataclasses, `__slots__` — language-level patterns unique to Python |
| **Creational Patterns** | | |
| 4 | [Factory & Abstract Factory](./04_factory.md) | Model factories, loss factories, dataset factories, transform factories — instantiation from config with zero coupling |
| 5 | [Builder Pattern](./05_builder.md) | Training pipeline builders, config builders, step-by-step model assemblers — constructing complex objects without telescoping constructors |
| 6 | [Singleton, Borg & Registry](./06_registry.md) | Config registry, model registry, service locator, plugin system — managing global state safely and testably |
| 7 | [Prototype & Object Pool](./07_prototype.md) | Weight initialization from pretrained, model cloning, GPU memory pools, tensor buffer reuse |
| 8 | [Dependency Injection](./08_dependency_injection.md) | Inversion of control, constructor injection, interface-based design, testability, modular ML component wiring |
| **Structural Patterns** | | |
| 9 | [Composite & Module Hierarchy](./09_composite.md) | `nn.Module` as a composite tree, recursive `forward()`, pipeline composition, nested transforms |
| 10 | [Adapter & Bridge](./10_adapter_bridge.md) | Framework bridges (HuggingFace ↔ PyTorch), data format adapters, separating abstraction from implementation |
| 11 | [Decorator & Wrapper](./11_decorator.md) | Augmentation wrappers, timing, caching, model instrumentation — Python `@decorator` vs structural Decorator pattern |
| 12 | [Facade & Service Layer](./12_facade.md) | High-level training APIs, HuggingFace `Trainer`, `Pipeline`, unified model interfaces that hide complexity |
| 13 | [Proxy & Lazy Loading](./13_proxy.md) | Remote model proxies, lazy tensor loading, virtual proxies, protection proxies, mock objects in testing |
| 14 | [Flyweight & Parameter Sharing](./14_flyweight.md) | Shared embedding tables, weight tying (input/output embeddings), cross-layer parameter sharing |
| **Behavioral Patterns** | | |
| 15 | [Strategy Pattern](./15_strategy.md) | Interchangeable loss functions, optimizers, samplers, regularizers, data augmentation policies — plug-and-play components |
| 16 | [Observer, Callback & Event System](./16_observer.md) | Training callbacks, PyTorch hooks, early stopping, logging, W&B/TensorBoard integration as observers |
| 17 | [Template Method & Training Loop](./17_template_method.md) | Skeleton algorithm, fill-in training steps, PyTorch Lightning's `training_step`, `Trainer` abstraction |
| 18 | [Chain of Responsibility & Transform Pipelines](./18_chain.md) | `torchvision.transforms.Compose`, preprocessing chains, middleware stacks, filter pipelines |
| 19 | [Command & Job Scheduler](./19_command.md) | Experiment queuing, task encapsulation, undo/redo for training steps, reproducible job dispatch |
| 20 | [State Machine & Lifecycle](./20_state.md) | Model lifecycle (init → training → eval → serving), training phase FSMs, warmup/cooldown, agent state management |
| 21 | [Iterator, Generator & Lazy Datasets](./21_iterator.md) | `DataLoader` internals, `IterableDataset`, streaming datasets, WebDataset, lazy evaluation for large-scale data |
| 22 | [Visitor & Graph Traversal](./22_visitor.md) | Model introspection, FLOPs counting, ONNX export, pruning masks, `nn.Module.apply()` pattern |
| **PyTorch Design Patterns** | | |
| 23 | [nn.Module Composition Patterns](./23_module_composition.md) | `Sequential`, `ModuleList`, `ModuleDict`, skip connections, multi-head outputs, forward design principles |
| 24 | [Hook Pattern](./24_hooks.md) | Forward/backward hooks, activation extraction, gradient analysis, feature visualization, Grad-CAM internals |
| 25 | [Custom Training Components](./25_custom_components.md) | Custom loss functions, custom optimizers, LR schedulers, samplers, collate functions — the PyTorch extension points |
| 26 | [Checkpoint & Serialization](./26_checkpoint.md) | `state_dict` save/load, safetensors, versioning strategies, resumable training, distributed checkpointing |
| 27 | [Distributed & Parallel Patterns](./27_distributed.md) | DDP, FSDP, pipeline parallelism, tensor parallelism, ZeRO stages, device placement strategies |
| **ML Pipeline & MLOps Patterns** | | |
| 28 | [Data Pipeline & DAG Patterns](./28_data_pipeline.md) | Kedro, Airflow, Prefect, Metaflow — DAG composition, data lineage, incremental processing, DVC integration |
| 29 | [Configuration Management Pattern](./29_configuration.md) | Hydra, OmegaConf, YAML hierarchies, structured configs, feature flags, environment-specific overrides |
| 30 | [Experiment Tracking & Reproducibility](./30_experiment_tracking.md) | MLflow, W&B, run management, artifact versioning, lineage graphs, determinism — seeds, hashes, containers |
| 31 | [Model Serving Patterns](./31_serving.md) | REST/gRPC serving, batch vs online inference, shadow deployment, A/B testing, canary releases, model versioning |
| 32 | [Resilience Patterns for ML](./32_resilience.md) | Circuit breaker, bulkhead, retry with backoff, fallback models, health checks, graceful degradation under load |
| **Agentic & LLM Design Patterns** | | |
| 33 | [ReAct & Chain-of-Thought](./33_react.md) | Observe-think-act loop, scratchpad reasoning, iterative self-correction, step-by-step prompting patterns |
| 34 | [Router & Dispatcher Pattern](./34_router.md) | LLM routing, multi-model switching, intent classification, conditional generation pipelines |
| 35 | [Orchestrator-Worker Pattern](./35_orchestrator_worker.md) | Task delegation, parallel sub-agent execution, result aggregation, supervisor-worker agent architectures |
| 36 | [Evaluator-Optimizer Pattern](./36_evaluator_optimizer.md) | Self-reflection loops, quality scoring, auto-refinement, Constitutional AI, process reward models |
| 37 | [RAG & Retrieval Patterns](./37_rag.md) | Retrieval-augmented generation, dense/hybrid retrieval, reranking, context compression, self-RAG |
| 38 | [Multi-Agent Communication](./38_multi_agent.md) | Message passing, blackboard pattern, publish-subscribe, shared state management, agent handoffs |
| 39 | [Memory & Context Management](./39_memory.md) | Working memory, episodic & semantic memory, KV-cache eviction strategies, context window compression |
| **Appendices** | | |
| A | [Pattern Quick Reference](./appendix_a_quick_reference.md) | All 39 patterns on one page — classification, intent, participants, and key ML/AI use case |
| B | [Anti-Patterns in ML](./appendix_b_antipatterns.md) | God module, spaghetti training loop, hardcoded config, data leakage, premature optimization — and how to refactor each |
| C | [Glossary](./appendix_c_glossary.md) | Every pattern term, acronym, and concept defined — from Abstract Factory to ZeRO |

## 🗺️ Learning Path

<div class="diagram">
<div class="diagram-title">Recommended Learning Path</div>
<div class="flow">
  <div class="flow-node accent wide">📖 Ch 1–3: Foundations — Patterns, SOLID & Python Idioms</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">🏭 Ch 4–8: Creational — How Objects Are Made</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">🏗️ Ch 9–14: Structural — How Components Are Composed</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">🔁 Ch 15–22: Behavioral — How Components Communicate</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">🔥 Ch 23–27: PyTorch Patterns — nn.Module, Hooks, Distributed</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">⚙️ Ch 28–32: MLOps Patterns — Pipelines, Config, Serving</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node pink wide">🤖 Ch 33–39: Agentic & LLM Patterns — ReAct, RAG, Multi-Agent</div>
</div>
</div>

## ⚡ Quick Start Paths

### Path A: "I'm an ML researcher who wants to write better code" (8 chapters)

1. [01 — Introduction to Design Patterns](./01_introduction.md) — the vocabulary
2. [02 — SOLID Principles in Python](./02_solid_principles.md) — the foundation
3. [04 — Factory & Abstract Factory](./04_factory.md) — model/dataset instantiation
4. [09 — Composite & Module Hierarchy](./09_composite.md) — nn.Module design
5. [15 — Strategy Pattern](./15_strategy.md) — interchangeable components
6. [16 — Observer, Callback & Event System](./16_observer.md) — training hooks
7. [23 — nn.Module Composition Patterns](./23_module_composition.md) — PyTorch-specific
8. [24 — Hook Pattern](./24_hooks.md) — introspection and debugging

### Path B: "I'm building production ML systems" (8 chapters)

1. [01 — Introduction to Design Patterns](./01_introduction.md) — the vocabulary
2. [06 — Singleton, Borg & Registry](./06_registry.md) — global state management
3. [08 — Dependency Injection](./08_dependency_injection.md) — testable components
4. [12 — Facade & Service Layer](./12_facade.md) — clean API boundaries
5. [28 — Data Pipeline & DAG Patterns](./28_data_pipeline.md) — pipeline design
6. [29 — Configuration Management](./29_configuration.md) — Hydra and OmegaConf
7. [31 — Model Serving Patterns](./31_serving.md) — REST, gRPC, shadow deployment
8. [32 — Resilience Patterns for ML](./32_resilience.md) — circuit breakers, fallbacks

### Path C: "I'm building LLM apps and AI agents" (8 chapters)

1. [01 — Introduction to Design Patterns](./01_introduction.md) — the vocabulary
2. [15 — Strategy Pattern](./15_strategy.md) — interchangeable LLM components
3. [18 — Chain of Responsibility & Transform Pipelines](./18_chain.md) — prompt pipelines
4. [33 — ReAct & Chain-of-Thought](./33_react.md) — reasoning patterns
5. [34 — Router & Dispatcher](./34_router.md) — multi-model routing
6. [35 — Orchestrator-Worker](./35_orchestrator_worker.md) — multi-agent design
7. [37 — RAG & Retrieval Patterns](./37_rag.md) — retrieval-augmented systems
8. [39 — Memory & Context Management](./39_memory.md) — agent memory

### Path D: "I want to understand PyTorch architecture deeply" (7 chapters)

1. [03 — Python Idioms & Protocols](./03_python_idioms.md) — protocols and ABCs
2. [09 — Composite & Module Hierarchy](./09_composite.md) — nn.Module tree
3. [11 — Decorator & Wrapper](./11_decorator.md) — wrapping modules
4. [21 — Iterator, Generator & Lazy Datasets](./21_iterator.md) — DataLoader design
5. [23 — nn.Module Composition Patterns](./23_module_composition.md) — composition idioms
6. [24 — Hook Pattern](./24_hooks.md) — forward/backward hooks
7. [27 — Distributed & Parallel Patterns](./27_distributed.md) — DDP, FSDP, tensor parallel

### Path E: "I want the complete picture" (full guide)

Read chapters 1 through 39 in order. Each group builds on the previous. Use the appendices as reference material throughout.

## 📚 Prerequisites

Before diving in, you should be comfortable with:

- **Python** — classes, inheritance, decorators, context managers, generators, type hints
- **PyTorch** — `nn.Module`, tensors, `forward()`, optimizers, basic training loops
- **ML basics** — loss functions, gradient descent, backpropagation, data loaders
- **Command line** — terminal, pip/conda, virtual environments

## 📚 Key References

This guide draws on the canonical design patterns literature and applied ML engineering resources:

| Resource | Why It Matters |
|----------|----------------|
| **Design Patterns** — Gamma, Helm, Johnson, Vlissides (GoF) | The original 23 patterns — the vocabulary every engineer must know |
| **Fluent Python** — Luciano Ramalho | Python-specific idioms: descriptors, protocols, generators, data model |
| **Architecture Patterns with Python** — Percival & Gregory | Repository, Unit of Work, Events, CQRS applied to Python backends |
| **Machine Learning Design Patterns** — Lakshmanan, Robinson, Munn | 30 patterns for data representation, training, resilience, reproducibility in ML |
| **Designing Machine Learning Systems** — Chip Huyen | End-to-end ML system architecture: feature stores, serving, monitoring |
| **Clean Architecture** — Robert C. Martin | Dependency rules, component principles, screaming architecture |
| **Designing Data-Intensive Applications** — Martin Kleppmann | Distributed patterns: replication, partitioning, event streams |
| **PyTorch Documentation & Internals** | `nn.Module`, hooks, DDP, FSDP, distributed patterns |
| **Martin Fowler's bliki** | Patterns of Enterprise Application Architecture, refactoring catalog |
| **Anthropic / OpenAI Agent research** | ReAct, Constitutional AI, tool use, multi-agent communication patterns |

## 🧩 How This Guide Is Organized

Design patterns in this guide span **seven dimensions**:

<div class="diagram">
<div class="diagram-title">Seven Dimensions of Design Patterns</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">📐</div>
    <div class="card-title">Foundations</div>
    <div class="card-desc">GoF taxonomy, SOLID principles, Python-specific protocols — the mental model before any pattern</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🏭</div>
    <div class="card-title">Creational</div>
    <div class="card-desc">Factory, Builder, Registry, Prototype, DI — how objects and components are created and wired</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">🏗️</div>
    <div class="card-title">Structural</div>
    <div class="card-desc">Composite, Adapter, Decorator, Facade, Proxy, Flyweight — how components are composed and connected</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">🔁</div>
    <div class="card-title">Behavioral</div>
    <div class="card-desc">Strategy, Observer, Template Method, Chain, Command, State, Iterator, Visitor — how components communicate and collaborate</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🔥</div>
    <div class="card-title">PyTorch Patterns</div>
    <div class="card-desc">nn.Module composition, hooks, custom components, checkpointing, distributed training</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-icon">⚙️</div>
    <div class="card-title">MLOps Patterns</div>
    <div class="card-desc">Data pipelines, configuration, experiment tracking, serving, resilience — production ML architecture</div>
  </div>
  <div class="diagram-card pink">
    <div class="card-icon">🤖</div>
    <div class="card-title">Agentic & LLM Patterns</div>
    <div class="card-desc">ReAct, Router, Orchestrator-Worker, RAG, Multi-Agent — the design vocabulary of modern AI systems</div>
  </div>
</div>
</div>

## 📝 Changelog

| Date | Changes |
|------|---------|
| May 2026 | Initial structure — 39 chapters + 3 appendices |

---

*Last updated: May 2026*
