---
title: "Appendix A — Pattern Quick Reference"
---

[← Back to Table of Contents](./README.md)

# Appendix A — Pattern Quick Reference

> *"A pattern describes a problem which occurs over and over again in our environment, and then describes the core of the solution." — Christopher Alexander*

---

## A.1 All 39 Patterns at a Glance

| # | Pattern | Category | Intent | Key ML/AI Use Case | Key Python Construct |
|---|---------|----------|--------|--------------------|----------------------|
| 01 | Introduction | Foundations | Establish vocabulary for design patterns in software | Mental model for all chapters | — |
| 02 | SOLID Principles | Foundations | Five principles for maintainable OO design | Model class architecture | Abstract base classes, `Protocol` |
| 03 | Python Idioms | Foundations | Pythonic patterns: generators, context managers, descriptors | Efficient data pipeline utilities | `yield`, `__enter__/__exit__`, `@property` |
| 04 | Factory Method | Creational | Define an interface for creating objects; let subclasses decide which class to instantiate | Model registry, optimizer factory | `@classmethod`, `abc.ABC` |
| 05 | Builder | Creational | Separate the construction of a complex object from its representation | Model config construction, experiment setup | Method chaining, `dataclass` |
| 06 | Registry | Creational | Central map from name string to class or callable | Plugin systems, `nn.Module` lookups | `dict`, `@register` decorator |
| 07 | Prototype | Creational | Create new objects by copying an existing prototype | Model weight initialisation, config cloning | `copy.deepcopy`, `__copy__` |
| 08 | Dependency Injection | Creational | Provide dependencies from outside rather than instantiating them internally | Swappable optimisers, loss functions, data loaders | Constructor injection, `Protocol` |
| 09 | Composite | Structural | Compose objects into tree structures to represent part-whole hierarchies | Sequential `nn.Module`, nested pipelines | `nn.Sequential`, `nn.ModuleList` |
| 10 | Adapter & Bridge | Structural | Convert interface of a class into another; decouple abstraction from implementation | Wrapping legacy models, multi-framework support | Wrapper class, `__call__` |
| 11 | Decorator | Structural | Attach additional responsibilities dynamically | Gradient checkpointing, timing, logging wrappers | `functools.wraps`, `@decorator` |
| 12 | Facade | Structural | Provide a simplified interface to a complex subsystem | High-level training API, Hugging Face `Trainer` | Thin wrapper class |
| 13 | Proxy | Structural | Provide a surrogate to control access | Lazy model loading, remote model APIs, caching | `__getattr__`, `__call__` forwarding |
| 14 | Flyweight | Structural | Use sharing to support large numbers of fine-grained objects | Token embedding tables, shared weight tying | `nn.Embedding`, interning |
| 15 | Strategy | Behavioral | Define a family of algorithms; make them interchangeable | Swappable loss functions, samplers, schedulers | `Protocol`, callable injection |
| 16 | Observer | Behavioral | Define a one-to-many dependency so objects are notified of state changes | Training callbacks, metric logging hooks | `Callback`, event system |
| 17 | Template Method | Behavioral | Define skeleton of algorithm in base class; defer steps to subclasses | `nn.Module.forward`, training loop base | `abc.abstractmethod` |
| 18 | Chain of Responsibility | Behavioral | Pass request along a chain of handlers until one handles it | Middleware pipelines, LLM tool routing | Linked handler list |
| 19 | Command | Behavioral | Encapsulate a request as an object | Undo/redo, experiment replay, CLI | Command objects, `__call__` |
| 20 | State | Behavioral | Allow object to alter its behaviour when internal state changes | Training phase management, agent lifecycle | `Enum`, state machine |
| 21 | Iterator | Behavioral | Provide sequential access to elements without exposing representation | DataLoader, lazy dataset pipelines | `__iter__`, `__next__`, `yield` |
| 22 | Visitor | Behavioral | Define new operations without changing classes they operate on | Model analysis, layer-wise ops, export | `accept/visit`, `torch.fx` |
| 23 | Module Composition | PyTorch | Compose `nn.Module` submodules declaratively | Building neural network architectures | `nn.ModuleList`, `nn.ModuleDict` |
| 24 | Hooks | PyTorch | Register callbacks on forward/backward passes for introspection | Feature extraction, gradient monitoring | `register_forward_hook`, `register_backward_hook` |
| 25 | Custom Components | PyTorch | Extend PyTorch with custom autograd, layers, or ops | Novel activations, custom attention | `torch.autograd.Function`, `nn.Module` |
| 26 | Checkpoint | MLOps | Persist training state for resumption and fault tolerance | Model saving, experiment recovery | `torch.save`, `torch.load` |
| 27 | Distributed Training | MLOps | Scale training across multiple GPUs/nodes | DDP, FSDP, ZeRO, tensor parallelism | `torch.distributed`, `FSDP` |
| 28 | Data Pipeline | MLOps | Efficient, reproducible data ingestion and transformation | Streaming datasets, feature stores | `Dataset`, `DataLoader`, `torch.utils.data` |
| 29 | Configuration | MLOps | Separate config from code; compose configs hierarchically | Hyperparameter management | OmegaConf, Hydra, Pydantic |
| 30 | Experiment Tracking | MLOps | Record parameters, metrics, and artifacts for reproducibility | MLflow, W&B, DVC | `mlflow.log_param`, `wandb.log` |
| 31 | Serving | MLOps | Deploy models as reliable, scalable inference services | FastAPI endpoints, ONNX runtime, TorchServe | ONNX, TorchScript, `@app.post` |
| 32 | Resilience | MLOps | Design training and serving to recover from faults | Retry, circuit breaker, bulkhead | `tenacity`, circuit breaker pattern |
| 33 | React | Agentic | Interleave reasoning (Thought) with tool use (Action) and observation | ReAct agents, LangChain `AgentExecutor` | Prompt templates, tool calling |
| 34 | Router | Agentic | Classify input and route to the appropriate specialised sub-agent or tool | Intent classification, multi-agent dispatch | Classifier + routing logic |
| 35 | Orchestrator-Worker | Agentic | Orchestrator decomposes tasks; workers execute in parallel | Parallel tool calls, MapReduce for agents | `asyncio.gather`, thread pool |
| 36 | Evaluator-Optimizer | Agentic | Generate → Evaluate → Refine loop for quality improvement | Self-reflection, Constitutional AI, Best-of-N | Feedback loop, rubric judge |
| 37 | RAG & Retrieval | Agentic | Retrieve relevant context at inference time to ground LLM outputs | FAISS/Chroma retrieval, hybrid search | DenseRetriever, BM25, RRF |
| 38 | Multi-Agent Communication | Agentic | Agents communicate via messages, blackboard, or pub-sub to collaborate | LangGraph, AutoGen GroupChat | MessageBus, EventBus |
| 39 | Memory & Context Management | Agentic | Store, compress, and retrieve information across agent steps | Episodic/semantic memory, KV-cache | SlidingWindowMemory, EpisodicMemory |

---

## A.2 Patterns by Category

<div class="diagram-grid cols-3">
  <div class="diagram-card blue">
    <div class="card-icon">📖</div>
    <div class="card-title">Foundations (01–03)</div>
    <div class="card-desc">
      <span class="badge">01</span> Introduction<br/>
      <span class="badge">02</span> SOLID Principles<br/>
      <span class="badge">03</span> Python Idioms
    </div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🏗️</div>
    <div class="card-title">Creational (04–08)</div>
    <div class="card-desc">
      <span class="badge creational">04</span> Factory Method<br/>
      <span class="badge creational">05</span> Builder<br/>
      <span class="badge creational">06</span> Registry<br/>
      <span class="badge creational">07</span> Prototype<br/>
      <span class="badge creational">08</span> Dependency Injection
    </div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">🔧</div>
    <div class="card-title">Structural (09–14)</div>
    <div class="card-desc">
      <span class="badge structural">09</span> Composite<br/>
      <span class="badge structural">10</span> Adapter &amp; Bridge<br/>
      <span class="badge structural">11</span> Decorator<br/>
      <span class="badge structural">12</span> Facade<br/>
      <span class="badge structural">13</span> Proxy<br/>
      <span class="badge structural">14</span> Flyweight
    </div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">🔄</div>
    <div class="card-title">Behavioral (15–22)</div>
    <div class="card-desc">
      <span class="badge behavioral">15</span> Strategy<br/>
      <span class="badge behavioral">16</span> Observer<br/>
      <span class="badge behavioral">17</span> Template Method<br/>
      <span class="badge behavioral">18</span> Chain of Responsibility<br/>
      <span class="badge behavioral">19</span> Command<br/>
      <span class="badge behavioral">20</span> State<br/>
      <span class="badge behavioral">21</span> Iterator<br/>
      <span class="badge behavioral">22</span> Visitor
    </div>
  </div>
  <div class="diagram-card teal">
    <div class="card-icon">🔥</div>
    <div class="card-title">PyTorch (23–25)</div>
    <div class="card-desc">
      <span class="badge pytorch">23</span> Module Composition<br/>
      <span class="badge pytorch">24</span> Hooks<br/>
      <span class="badge pytorch">25</span> Custom Components
    </div>
  </div>
  <div class="diagram-card pink">
    <div class="card-icon">⚙️</div>
    <div class="card-title">MLOps (26–32)</div>
    <div class="card-desc">
      <span class="badge mlops">26</span> Checkpoint<br/>
      <span class="badge mlops">27</span> Distributed Training<br/>
      <span class="badge mlops">28</span> Data Pipeline<br/>
      <span class="badge mlops">29</span> Configuration<br/>
      <span class="badge mlops">30</span> Experiment Tracking<br/>
      <span class="badge mlops">31</span> Serving<br/>
      <span class="badge mlops">32</span> Resilience
    </div>
  </div>
  <div class="diagram-card yellow">
    <div class="card-icon">🤖</div>
    <div class="card-title">Agentic (33–39)</div>
    <div class="card-desc">
      <span class="badge agentic">33</span> ReAct<br/>
      <span class="badge agentic">34</span> Router<br/>
      <span class="badge agentic">35</span> Orchestrator-Worker<br/>
      <span class="badge agentic">36</span> Evaluator-Optimizer<br/>
      <span class="badge agentic">37</span> RAG &amp; Retrieval<br/>
      <span class="badge agentic">38</span> Multi-Agent Communication<br/>
      <span class="badge agentic">39</span> Memory &amp; Context Management
    </div>
  </div>
</div>

---

## A.3 "Which Pattern Solves My Problem?" Decision Table

<table class="compare-table">
  <thead>
    <tr>
      <th>Problem Symptom</th>
      <th>Recommended Pattern(s)</th>
      <th>Chapter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>I need to create objects without specifying the exact class</td>
      <td>Factory Method, Registry</td>
      <td>04, 06</td>
    </tr>
    <tr>
      <td>My config object has too many constructor parameters</td>
      <td>Builder</td>
      <td>05</td>
    </tr>
    <tr>
      <td>I want to clone a model with different hyperparameters</td>
      <td>Prototype</td>
      <td>07</td>
    </tr>
    <tr>
      <td>My class creates its own dependencies (hard to test)</td>
      <td>Dependency Injection</td>
      <td>08</td>
    </tr>
    <tr>
      <td>I need to build a neural network from sub-modules</td>
      <td>Composite, Module Composition</td>
      <td>09, 23</td>
    </tr>
    <tr>
      <td>I need to wrap a third-party model with a different interface</td>
      <td>Adapter</td>
      <td>10</td>
    </tr>
    <tr>
      <td>I want to add timing, logging, or caching without modifying the class</td>
      <td>Decorator, Proxy</td>
      <td>11, 13</td>
    </tr>
    <tr>
      <td>The training API is too complex to use directly</td>
      <td>Facade</td>
      <td>12</td>
    </tr>
    <tr>
      <td>I want to share embedding weights across multiple modules</td>
      <td>Flyweight</td>
      <td>14</td>
    </tr>
    <tr>
      <td>I want to swap optimisers or loss functions at runtime</td>
      <td>Strategy</td>
      <td>15</td>
    </tr>
    <tr>
      <td>I want to log metrics or stop early during training</td>
      <td>Observer (Callbacks)</td>
      <td>16</td>
    </tr>
    <tr>
      <td>I have a training loop skeleton with customisable steps</td>
      <td>Template Method</td>
      <td>17</td>
    </tr>
    <tr>
      <td>I need to route requests through a sequence of handlers</td>
      <td>Chain of Responsibility</td>
      <td>18</td>
    </tr>
    <tr>
      <td>I want undo/redo or replay of training actions</td>
      <td>Command</td>
      <td>19</td>
    </tr>
    <tr>
      <td>My model behaves differently in train/eval/serving phases</td>
      <td>State</td>
      <td>20</td>
    </tr>
    <tr>
      <td>I need to stream a huge dataset without loading it all</td>
      <td>Iterator, Data Pipeline</td>
      <td>21, 28</td>
    </tr>
    <tr>
      <td>I want to analyse model layers without changing them</td>
      <td>Visitor, Hooks</td>
      <td>22, 24</td>
    </tr>
    <tr>
      <td>I need to inspect or modify gradients during backprop</td>
      <td>Hooks</td>
      <td>24</td>
    </tr>
    <tr>
      <td>I need a custom layer with a novel gradient computation</td>
      <td>Custom Components</td>
      <td>25</td>
    </tr>
    <tr>
      <td>Training crashes and I lose progress</td>
      <td>Checkpoint, Resilience</td>
      <td>26, 32</td>
    </tr>
    <tr>
      <td>Training is too slow on a single GPU</td>
      <td>Distributed Training</td>
      <td>27</td>
    </tr>
    <tr>
      <td>I have too many magic numbers in my training script</td>
      <td>Configuration (Hydra/OmegaConf)</td>
      <td>29</td>
    </tr>
    <tr>
      <td>I can't reproduce last week's experiment results</td>
      <td>Experiment Tracking</td>
      <td>30</td>
    </tr>
    <tr>
      <td>I need to deploy my model as a low-latency API</td>
      <td>Serving (ONNX, TorchServe)</td>
      <td>31</td>
    </tr>
    <tr>
      <td>My downstream API keeps failing and crashing the pipeline</td>
      <td>Resilience (Circuit Breaker)</td>
      <td>32</td>
    </tr>
    <tr>
      <td>I want an LLM that can use external tools</td>
      <td>ReAct</td>
      <td>33</td>
    </tr>
    <tr>
      <td>I need to send queries to different specialist models</td>
      <td>Router</td>
      <td>34</td>
    </tr>
    <tr>
      <td>I have many independent sub-tasks that can run in parallel</td>
      <td>Orchestrator-Worker</td>
      <td>35</td>
    </tr>
    <tr>
      <td>My LLM outputs are inconsistent quality</td>
      <td>Evaluator-Optimizer, Best-of-N</td>
      <td>36</td>
    </tr>
    <tr>
      <td>My LLM hallucinates facts</td>
      <td>RAG &amp; Retrieval</td>
      <td>37</td>
    </tr>
    <tr>
      <td>I need multiple specialised LLMs to collaborate</td>
      <td>Multi-Agent Communication</td>
      <td>38</td>
    </tr>
    <tr>
      <td>My agent forgets earlier parts of a long conversation</td>
      <td>Memory &amp; Context Management</td>
      <td>39</td>
    </tr>
  </tbody>
</table>

---

*Last updated: May 2026*
