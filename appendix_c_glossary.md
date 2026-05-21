---
title: "Appendix C — Glossary"
---

[← Back to Table of Contents](./README.md)

# Appendix C — Glossary

> *"If you wish to converse with me, define your terms." — Voltaire*

This glossary defines every key term used across all 39 chapters, listed alphabetically.

---

**Abstract Factory**
A creational design pattern that provides an interface for creating *families* of related objects without specifying their concrete classes. In ML, used to create compatible sets of model, optimizer, and scheduler objects (e.g., a `TransformerFactory` that produces a matching encoder, decoder, and tokenizer). See Chapter 04.

**Adapter**
A structural pattern that converts the interface of a class into another interface expected by a client. Common use: wrapping a Hugging Face model so it exposes the same interface as a custom `nn.Module`, or bridging a NumPy-based preprocessing step into a PyTorch pipeline. See Chapter 10.

**Agent**
An autonomous software entity that perceives its environment (via tool calls, retrieved documents, or messages), reasons (via LLM), acts (via tool execution or message sending), and updates its state. Multi-step agents chain many perception-reasoning-action cycles. See Chapters 33–39.

**Anti-pattern**
A commonly used but consistently problematic design choice. Unlike a bug, an anti-pattern appears intentional and often "works" in simple cases, but creates maintenance, scalability, or correctness problems as systems grow. See Appendix B.

**Attention Mechanism**
A neural network component that computes a weighted sum of values, where weights are determined by query-key similarity. Self-attention allows each token to attend to all other tokens in a sequence, forming the core of transformer architectures.

**Autograd**
PyTorch's automatic differentiation engine. Records operations in a computational graph during the forward pass; traverses the graph in reverse during `.backward()` to compute gradients. Custom gradient computations are defined via `torch.autograd.Function`.

**Behavioral Pattern**
One of the three GoF pattern categories. Behavioral patterns characterise the ways objects communicate and collaborate: Strategy, Observer, Template Method, Chain of Responsibility, Command, State, Iterator, and Visitor. See Chapters 15–22.

**BM25**
Best Match 25 — a sparse retrieval function based on TF-IDF that ranks documents by term frequency, inverse document frequency, and document-length normalisation. Parameters k1 (term saturation) and b (length normalisation) are typically set to 1.5 and 0.75. See Chapter 37.

**Bridge**
A structural pattern that decouples an abstraction from its implementation so both can vary independently. In ML, separates a high-level training interface from the low-level compute backend (CPU, GPU, TPU). See Chapter 10.

**Builder**
A creational pattern that separates the construction of a complex object from its representation using a fluent step-by-step interface. In ML, used to assemble model configurations or experiment specs with sensible defaults and validation. See Chapter 05.

**Chain of Responsibility**
A behavioral pattern where a request is passed along a chain of handlers; each handler decides whether to process it or forward it. In ML: middleware pipelines, LLM tool routing, preprocessing chains. See Chapter 18.

**Checkpoint**
A snapshot of training state (model weights, optimizer state, scheduler state, epoch/step number) persisted to disk so training can be resumed after interruption or used for fine-tuning. Implemented via `torch.save` / `torch.load`. See Chapter 26.

**Circuit Breaker**
A resilience pattern that "trips" open after a configurable number of consecutive failures, stopping further attempts and allowing a downstream service to recover. Returns to "closed" (normal) after a cool-down period. See Chapter 32.

**CoT (Chain-of-Thought)**
A prompting technique where the LLM is asked to reason step-by-step before answering. Improves accuracy on complex reasoning tasks. Variants include Zero-shot CoT ("think step by step"), Few-shot CoT (with worked examples), and Tree-of-Thought. See Chapter 33.

**Composite**
A structural pattern that composes objects into tree (part-whole) hierarchies so clients treat individual objects and compositions uniformly. In PyTorch, `nn.Sequential` and `nn.ModuleList` implement the Composite pattern. See Chapter 09.

**Constitutional AI**
A technique from Anthropic where an LLM critiques and revises its own outputs against a list of explicit principles (a "constitution"), iteratively improving safety and helpfulness without human labelling at each step. See Chapter 36.

**Context Window**
The finite sequence of tokens an LLM can process in a single forward pass. Current models range from 4K to 1M+ tokens. Information outside the context window must be retrieved (RAG), summarised (compression), or discarded. See Chapter 39.

**Creational Pattern**
One of the three GoF pattern categories. Creational patterns abstract object instantiation: Factory Method, Builder, Registry, Prototype, Dependency Injection. See Chapters 04–08.

**DAG (Directed Acyclic Graph)**
A graph with directed edges and no cycles. In ML: the computational graph of a model (nodes = operations, edges = tensor dependencies), pipeline DAGs (each stage is a node), and LangGraph agent workflows all use DAGs.

**DDP (DistributedDataParallel)**
PyTorch's primary distributed training module. Each process holds a complete model replica; gradients are synchronised via AllReduce after each backward pass. Scales near-linearly to many GPUs/nodes. See Chapter 27.

**Decorator** *(pattern)*
A structural pattern that attaches additional responsibilities to an object dynamically by wrapping it. In Python, implemented via higher-order functions or the `@decorator` syntax. In PyTorch: gradient checkpointing wrappers, timing wrappers, and PEFT (LoRA) wrappers use this pattern. See Chapter 11.

**Dependency Injection**
A creational pattern where an object's dependencies are provided externally (constructor injection, setter injection) rather than created internally. Enables testability, modularity, and runtime swapping of implementations. See Chapter 08.

**Descriptor**
A Python protocol (`__get__`, `__set__`, `__delete__`) for defining managed attributes. PyTorch's `nn.Module` uses descriptors internally to register parameters and sub-modules. See Chapter 03.

**DIP (Dependency Inversion Principle)**
The "D" of SOLID: high-level modules should not depend on low-level modules; both should depend on abstractions. In Python, implemented via `Protocol` or `abc.ABC`. See Chapter 02.

**Distributed Training**
Training a model across multiple compute devices (GPUs, TPUs, or nodes) to overcome single-device memory and compute limits. Strategies include DDP, FSDP, ZeRO, pipeline parallelism, and tensor parallelism. See Chapter 27.

**DVC (Data Version Control)**
An open-source tool for versioning datasets, model artifacts, and experiments alongside Git. Tracks large files in remote storage (S3, GCS, etc.) using lightweight pointer files in the Git repo. See Chapter 30.

**Episodic Memory**
In agent architectures: a persistent store of past interactions (observations, actions, outcomes) retrievable by semantic similarity. Typically implemented with a vector database. Analogous to human autobiographical memory. See Chapter 39.

**Evaluator-Optimizer**
An agentic pattern where a generator produces output, an evaluator scores it, and an optimizer/refiner improves it based on the evaluation — forming an iterative feedback loop. Includes Self-Reflection, Constitutional AI, PRM, and Best-of-N as sub-variants. See Chapter 36.

**Factory Method**
A creational pattern that defines an interface for creating an object but lets subclasses decide which class to instantiate. In ML: `@classmethod` constructors, model registries, and `from_pretrained` methods are factory methods. See Chapter 04.

**Facade**
A structural pattern that provides a simplified interface to a complex subsystem. The Hugging Face `Trainer`, PyTorch Lightning `LightningModule`, and Keras `model.fit` are all Facade implementations. See Chapter 12.

**Flyweight**
A structural pattern that shares common state among many fine-grained objects to reduce memory. In PyTorch: `nn.Embedding` weight tying (input and output embeddings share the same weight matrix) is a classic Flyweight application. See Chapter 14.

**FSDP (Fully Sharded Data Parallel)**
A PyTorch distributed training strategy that shards model parameters, gradients, and optimizer states across all ranks, achieving near-optimal memory efficiency for large models. Based on the ZeRO-3 algorithm from Microsoft DeepSpeed. See Chapter 27.

**Generator** *(Python)*
A function that uses `yield` to produce values lazily. Enables memory-efficient iteration over datasets, token streams, or pipeline outputs without materialising the entire sequence. Fundamental to Python's iterator protocol. See Chapter 03.

**GoF (Gang of Four)**
Erich Gamma, Richard Helm, Ralph Johnson, and John Vlissides — authors of *Design Patterns: Elements of Reusable Object-Oriented Software* (1994), which catalogued 23 foundational patterns across Creational, Structural, and Behavioral categories.

**Hooks** *(PyTorch)*
Functions registered with `register_forward_hook`, `register_forward_pre_hook`, or `register_backward_hook` that are called automatically during the forward or backward pass. Used for feature extraction, gradient monitoring, activation visualisation, and PEFT adapters. See Chapter 24.

**HyDE (Hypothetical Document Embeddings)**
A RAG technique that generates a hypothetical answer to a query using an LLM, embeds the hypothesis, and retrieves documents nearest to the hypothesis embedding rather than the query itself — improving recall for complex queries. See Chapter 37.

**ISP (Interface Segregation Principle)**
The "I" of SOLID: clients should not be forced to depend on interfaces they do not use. In Python: use fine-grained `Protocol` classes rather than one large abstract base class. See Chapter 02.

**Iterator**
A behavioral pattern that provides sequential access to elements of a collection without exposing its internal structure. In Python: the `__iter__` / `__next__` protocol; in PyTorch: `DataLoader` is an iterator over batched dataset elements. See Chapter 21.

**KV-Cache**
A memory optimisation in transformer inference where the key and value matrices for all previously processed tokens are cached between decoding steps. Eliminates redundant recomputation; memory usage grows linearly with sequence length. See Chapter 39.

**LangGraph**
A library built on LangChain for building stateful multi-agent workflows as directed graphs. Nodes are agent functions; edges define control flow including conditional branching and cycles. See Chapter 38.

**LoRA (Low-Rank Adaptation)**
A parameter-efficient fine-tuning technique that freezes pre-trained weights and adds trainable low-rank decomposition matrices to attention layers. Reduces trainable parameters by 10,000× while achieving comparable fine-tuning quality.

**LSP (Liskov Substitution Principle)**
The "L" of SOLID: objects of a subtype must be substitutable for objects of the supertype without altering program correctness. Violated when overriding a method changes its pre/post-conditions. See Chapter 02.

**Mixins**
Small classes designed to be inherited alongside a primary base class to add a focused set of behaviours without creating deep inheritance hierarchies. In Python ML: logging mixins, serialisation mixins, device-management mixins. See Chapter 03.

**nn.Module**
The base class for all neural network components in PyTorch. Provides parameter registration, device management, serialisation, gradient tracking, and hook infrastructure. All custom layers, models, and model components inherit from `nn.Module`. See Chapters 23–25.

**Observer**
A behavioral pattern that defines a one-to-many dependency so that when one object changes state, all its dependents are notified automatically. In ML: training callbacks, metric loggers, early stopping triggers, and W&B integrations follow this pattern. See Chapter 16.

**OCP (Open-Closed Principle)**
The "O" of SOLID: software entities should be open for extension but closed for modification. Achieved in Python via inheritance, composition, and the Strategy or Decorator patterns. See Chapter 02.

**OmegaConf / Hydra**
OmegaConf is a YAML/Python configuration library with structured configs and interpolation. Hydra builds on top, adding hierarchical config composition, command-line overrides, and multi-run sweep support. Together they implement the Configuration pattern. See Chapter 29.

**ONNX (Open Neural Network Exchange)**
An open format for representing ML models. Allows models trained in PyTorch, TensorFlow, or other frameworks to be exported and run in optimised inference runtimes (ONNX Runtime, TensorRT) across hardware platforms. See Chapter 31.

**Orchestrator-Worker**
An agentic pattern where an orchestrator decomposes a complex task into independent sub-tasks and dispatches them to worker agents or tools — optionally in parallel. Workers return results; the orchestrator synthesises the final output. See Chapter 35.

**Pattern**
A general, reusable solution to a commonly occurring problem within a given context. Patterns are templates, not implementations — they must be adapted to specific circumstances. A pattern has a name, intent, motivation, structure, and known uses.

**Pipeline**
A sequence of processing stages where the output of each stage is the input to the next. In ML: data pipelines (ingest → clean → featurise → split), training pipelines (load → forward → backward → update), and inference pipelines (preprocess → model → postprocess). See Chapter 28.

**Process Reward Model (PRM)**
A reward model that assigns a quality score to each intermediate reasoning step in a chain-of-thought response, rather than only scoring the final answer. Enables more accurate selection and verification of reasoning chains. See Chapter 36.

**Prototype**
A creational pattern that creates new objects by copying an existing prototype instance. In Python: `copy.deepcopy`; in ML: cloning model configs or weight-initialised networks with modified hyperparameters. See Chapter 07.

**Protocol** *(Python)*
A `typing.Protocol` class that defines structural subtyping (duck typing with static analysis support). An object satisfies a Protocol if it has the required attributes and methods, without explicit inheritance. Enables Dependency Injection and Strategy patterns with no base class coupling. See Chapter 02.

**Proxy**
A structural pattern that provides a surrogate object to control access to another object. In ML: lazy model loaders (load weights only on first call), remote model API wrappers, and transparent caching proxies. See Chapter 13.

**RAG (Retrieval-Augmented Generation)**
An architecture that augments LLM generation by retrieving relevant documents from an external knowledge store at query time and inserting them into the prompt. Reduces hallucination and enables up-to-date, citable responses. See Chapter 37.

**ReAct**
A prompting strategy that interleaves Reasoning (Thought) and Acting (tool calls / Action) steps, with Observations fed back into the prompt. Enables LLMs to use external tools while maintaining a coherent reasoning trace. See Chapter 33.

**Registry**
A creational pattern that maintains a central dictionary mapping string names to classes or factory callables. Enables dynamic plugin registration and name-based object creation. Implemented via `@register` decorators or class `__init_subclass__`. See Chapter 06.

**Resilience**
The capacity of an ML system to detect, tolerate, and recover from failures in training, serving, or data pipelines. Patterns include retry with exponential backoff, circuit breaker, bulkhead, timeout, and graceful degradation. See Chapter 32.

**RRF (Reciprocal Rank Fusion)**
A rank aggregation method that combines multiple ranked lists by summing `1 / (k + rank)` for each document across lists, where k is a smoothing constant (typically 60). Used in hybrid retrieval to merge dense and sparse rankings. See Chapter 37.

**Self-RAG**
An extension of RAG where the model learns to decide *when* to retrieve using special reflection tokens (`[Retrieve]`, `[ISREL]`, `[ISSUP]`, `[ISUSE]`), enabling adaptive retrieval only when needed. See Chapter 37.

**Singleton**
A creational pattern that ensures only one instance of a class exists. In ML: model registries, device managers, and configuration singletons. Often better replaced by module-level state or dependency injection in Python. See Chapter 06 (Registry).

**SOLID**
An acronym for five object-oriented design principles: **S**ingle Responsibility, **O**pen-Closed, **L**iskov Substitution, **I**nterface Segregation, **D**ependency Inversion. Forms the foundation for maintainable ML system design. See Chapter 02.

**SRP (Single Responsibility Principle)**
The "S" of SOLID: a class should have only one reason to change — it should encapsulate only one aspect of the system's behaviour. Violated by the God Module anti-pattern. See Chapters 02, Appendix B.

**State**
A behavioral pattern that allows an object to alter its behaviour when its internal state changes — appearing to change its class. In ML: training-phase management (train / eval / serving), agent lifecycle (idle / running / waiting / done). See Chapter 20.

**Strategy**
A behavioral pattern that defines a family of algorithms, encapsulates each one, and makes them interchangeable. In ML: swappable loss functions, optimizers, samplers, and regularisers. Implemented via `Protocol` or callable injection. See Chapter 15.

**Template Method**
A behavioral pattern that defines the skeleton of an algorithm in a base class, deferring some steps to subclasses. In PyTorch: `nn.Module.forward` and training loop base classes follow this pattern. See Chapter 17.

**Tensor Parallelism**
A distributed training strategy that shards individual weight matrices (e.g., attention projections) across multiple GPUs, requiring all-gather/reduce-scatter communication within each layer. Used by Megatron-LM and large MoE models. See Chapter 27.

**TorchScript**
A way to serialise and optimise PyTorch models by tracing or scripting them into an intermediate representation that can be saved and run independently of Python. Used for production serving. See Chapter 31.

**Visitor**
A behavioral pattern that lets you define new operations on elements of an object structure without changing the classes of those elements. In PyTorch: `torch.fx` graph traversal, model analysis passes (counting parameters, finding BatchNorm layers) use the Visitor pattern. See Chapter 22.

**Working Memory**
In agent architectures: the active context window passed to the LLM on each call. Contains the current conversation, retrieved documents, tool results, and system instructions. Limited by the context window size. See Chapter 39.

**ZeRO (Zero Redundancy Optimizer)**
Microsoft's memory optimisation strategy for large-scale distributed training. ZeRO-1 shards optimizer states; ZeRO-2 adds gradient sharding; ZeRO-3 (implemented as FSDP in PyTorch) also shards model parameters across all ranks. See Chapter 27.

---

*Last updated: May 2026*
