---
title: "Chapter 1 — Introduction to Design Patterns"
---

[← Back to Table of Contents](./README.md)

# Chapter 1 — Introduction to Design Patterns

> *"Each pattern describes a problem which occurs over and over again in our environment, and then describes the core of the solution to that problem, in such a way that you can use this solution a million times over, without ever doing it the same way twice."*
> — Christopher Alexander, *A Pattern Language*, 1977

---

## What Is a Design Pattern?

A **design pattern** is a general, reusable solution to a commonly occurring problem within a given context in software design. It is not a finished design that can be transformed directly into code — rather, it is a description or template for how to solve a problem that can be used in many different situations.

The concept did not originate in software engineering. Christopher Alexander, an architect and design theorist, introduced the notion in his 1977 book *A Pattern Language* and its companion *The Timeless Way of Building* (1979). Alexander observed that successful human environments — towns, buildings, rooms — were built from recurring structural patterns, and that naming and cataloguing these patterns gave architects and planners a shared vocabulary for solving design problems.

Software architects noticed the analogy immediately. The "Gang of Four" (GoF) — Erich Gamma, Richard Helm, Ralph Johnson, and John Vlissides — formalized this idea for object-oriented software in their landmark 1994 book *Design Patterns: Elements of Reusable Object-Oriented Software*. Their 23 patterns became a lingua franca for software architects the world over, and remain the canonical reference today.

### The GoF Definition

The Gang of Four defined a design pattern as having four essential elements:

1. **Pattern Name** — A handle to describe a design problem, its solution, and consequences. Naming a pattern immediately enriches our design vocabulary and lets us discuss designs at a higher level of abstraction.
2. **Problem** — Describes when to apply the pattern: the context and conditions under which the pattern is applicable.
3. **Solution** — Describes the elements that make up the design, their relationships, responsibilities, and collaborations. It does not describe a concrete implementation; it is an abstract description.
4. **Consequences** — The results and trade-offs of applying the pattern, including impacts on flexibility, extensibility, portability, and system resource usage.

### Why "Pattern" and Not Just "Best Practice"?

Best practices are often high-level, informal advice ("keep functions small", "use meaningful variable names"). Patterns are more structured: they have names, documented contexts, forces, solutions, and trade-offs. They are relational — a pattern solves a tension between competing design forces. And they are composable — complex systems are built from multiple patterns working together.

---

## The Gang of Four: Who They Are and What They Wrote

<div class="diagram">
<div class="diagram-title">The Gang of Four</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-icon">📖</div>
    <div class="card-title">Erich Gamma</div>
    <div class="card-desc">Lead author and architect. Later co-created JUnit and led the Eclipse project. Now at Microsoft working on VS Code.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🔬</div>
    <div class="card-title">Richard Helm</div>
    <div class="card-desc">Expert in object-oriented analysis and design. Co-authored the book during his time at IBM Research.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🎓</div>
    <div class="card-title">Ralph Johnson</div>
    <div class="card-desc">Professor at University of Illinois. A founder of the patterns movement and an expert in refactoring and frameworks.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">⚙️</div>
    <div class="card-title">John Vlissides</div>
    <div class="card-desc">Researcher at IBM T.J. Watson Research Center. Contributed foundational work on Unidraw, a structured graphics framework.</div>
  </div>
</div>
</div>

Published in 1994, *Design Patterns: Elements of Reusable Object-Oriented Software* catalogued 23 patterns across three categories. The book sold over half a million copies and has been translated into more than a dozen languages. Its influence on how software engineers think about and communicate design is immeasurable.

---

## Why Patterns Matter for ML/AI Code

Machine learning code has a reputation for being notoriously difficult to maintain. Research code especially tends to be monolithic, deeply coupled, and hard to reproduce. As ML systems mature from prototype notebooks to production pipelines, the lack of structure becomes a serious liability.

Patterns address exactly the problems ML practitioners face most often:

- **Reproducibility**: Factories and Builders make it easy to reconstruct experiments from configuration, rather than hoping you can re-read a notebook.
- **Extensibility**: Open/Closed designs let you add new model architectures, loss functions, or augmentation strategies without touching core training loops.
- **Testability**: Dependency Inversion lets you swap real models for mocks in unit tests, running thousands of tests in milliseconds instead of waiting for GPU inference.
- **Team collaboration**: Named patterns give teams a shared vocabulary. "Use a Strategy here" is immediately understood; "do that thing where you swap the algorithm at runtime" is not.
- **Scalability**: Composable, decoupled code scales from a single GPU to a multi-node training cluster with minimal structural change.

<div class="callout info">
<strong>ML Pattern Adoption Trajectory</strong><br/>
Many ML libraries have organically converged on GoF patterns without explicitly naming them. PyTorch's <code>nn.Module</code> uses Composite. Hugging Face's <code>AutoModel</code> uses Factory. Lightning's <code>Trainer</code> uses Template Method. Recognizing these patterns helps you use the libraries more effectively and build libraries others will love.
</div>

---

## The Three Categories of GoF Patterns

<div class="diagram">
<div class="diagram-title">Pattern Categories</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">🏗️</div>
    <div class="card-title">Creational</div>
    <div class="card-desc">Deal with object creation mechanisms. Abstract the instantiation process, making the system independent of how objects are created, composed, and represented.<br/><br/><strong>Patterns:</strong> Singleton, Factory Method, Abstract Factory, Builder, Prototype</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🔧</div>
    <div class="card-title">Structural</div>
    <div class="card-desc">Deal with object composition — creating relationships between objects to form larger structures. Use inheritance and composition to achieve new functionality.<br/><br/><strong>Patterns:</strong> Adapter, Bridge, Composite, Decorator, Facade, Flyweight, Proxy</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🔄</div>
    <div class="card-title">Behavioral</div>
    <div class="card-desc">Characterize the ways in which classes or objects interact and distribute responsibility. Concerned with algorithms and assignment of responsibilities.<br/><br/><strong>Patterns:</strong> Chain of Responsibility, Command, Interpreter, Iterator, Mediator, Memento, Observer, State, Strategy, Template Method, Visitor</div>
  </div>
</div>
</div>

---

## The 23 GoF Patterns

| Name | Category | Intent |
|------|----------|--------|
| **Abstract Factory** | Creational | Provide an interface for creating families of related objects without specifying their concrete classes |
| **Builder** | Creational | Separate the construction of a complex object from its representation |
| **Factory Method** | Creational | Define an interface for creating an object, but let subclasses decide which class to instantiate |
| **Prototype** | Creational | Specify kinds of objects to create using a prototypical instance, and create new objects by copying this prototype |
| **Singleton** | Creational | Ensure a class has only one instance and provide a global point of access to it |
| **Adapter** | Structural | Convert the interface of a class into another interface clients expect |
| **Bridge** | Structural | Decouple an abstraction from its implementation so the two can vary independently |
| **Composite** | Structural | Compose objects into tree structures to represent part-whole hierarchies |
| **Decorator** | Structural | Attach additional responsibilities to an object dynamically |
| **Facade** | Structural | Provide a simplified interface to a complex subsystem |
| **Flyweight** | Structural | Use sharing to support large numbers of fine-grained objects efficiently |
| **Proxy** | Structural | Provide a surrogate or placeholder for another object to control access to it |
| **Chain of Responsibility** | Behavioral | Pass the request along a chain of handlers until one handles it |
| **Command** | Behavioral | Encapsulate a request as an object, parameterizing clients with different requests |
| **Interpreter** | Behavioral | Given a language, define a representation for its grammar with an interpreter |
| **Iterator** | Behavioral | Provide a way to access elements of an aggregate object sequentially |
| **Mediator** | Behavioral | Define an object that encapsulates how a set of objects interact |
| **Memento** | Behavioral | Capture and externalize an object's internal state so it can be restored later |
| **Observer** | Behavioral | Define a one-to-many dependency between objects |
| **State** | Behavioral | Allow an object to alter its behavior when its internal state changes |
| **Strategy** | Behavioral | Define a family of algorithms, encapsulate each one, and make them interchangeable |
| **Template Method** | Behavioral | Define the skeleton of an algorithm in an operation, deferring some steps to subclasses |
| **Visitor** | Behavioral | Represent an operation to be performed on elements of an object structure |

---

## A History of Design Patterns

<div class="diagram">
<div class="diagram-title">Timeline: Design Patterns History</div>
<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-year">1977</div>
    <div class="timeline-title">Christopher Alexander — <em>A Pattern Language</em></div>
    <div class="timeline-desc">Architect Christopher Alexander publishes his landmark work on 253 architectural and urban design patterns, planting the seed for the concept of recurring design solutions.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">1987</div>
    <div class="timeline-title">Ward Cunningham & Kent Beck — First Software Patterns</div>
    <div class="timeline-desc">Beck and Cunningham apply Alexander's ideas to Smalltalk user interface design, presenting "Using Pattern Languages for Object-Oriented Programs" at OOPSLA.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">1991</div>
    <div class="timeline-title">Erich Gamma — Doctoral Dissertation</div>
    <div class="timeline-desc">Gamma's dissertation on object-oriented design at the University of Zurich forms the backbone of what will become the GoF book.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">1994</div>
    <div class="timeline-title">GoF — <em>Design Patterns: Elements of Reusable Object-Oriented Software</em></div>
    <div class="timeline-desc">The Gang of Four publish their canonical catalog of 23 patterns. The book quickly becomes one of the most influential software engineering texts ever written.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">1995</div>
    <div class="timeline-title">Portland Pattern Repository — First Pattern Wiki</div>
    <div class="timeline-desc">Ward Cunningham launches the WikiWikiWeb, the world's first wiki, to host the Portland Pattern Repository — a community-edited catalog of software patterns.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2002</div>
    <div class="timeline-title">Martin Fowler — <em>Patterns of Enterprise Application Architecture</em></div>
    <div class="timeline-desc">Fowler extends the pattern vocabulary to enterprise application concerns: ORM, Repository, Service Layer, Unit of Work, and dozens more.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2010s</div>
    <div class="timeline-title">ML Frameworks Emerge</div>
    <div class="timeline-desc">PyTorch (2016), TensorFlow (2015), Keras, and others organically embed GoF patterns (Composite, Observer, Strategy, Factory) into their APIs without always naming them.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2020s</div>
    <div class="timeline-title">ML-Specific Patterns Codified</div>
    <div class="timeline-desc">The ML community begins explicitly cataloguing patterns for MLOps, model serving, data pipelines, and LLM-based agentic systems. This book is part of that movement.</div>
  </div>
</div>
</div>

---

## Pattern Anatomy: The Seven Elements

Every GoF pattern is documented using a consistent seven-element structure. Understanding this anatomy helps you read pattern descriptions and apply them correctly.

<div class="diagram">
<div class="diagram-title">Pattern Anatomy</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-icon">🎯</div>
    <div class="card-title">Intent</div>
    <div class="card-desc">A short statement of what the pattern does and the design problem it addresses. This is the one-sentence elevator pitch.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">💡</div>
    <div class="card-title">Motivation</div>
    <div class="card-desc">A concrete scenario illustrating the design problem and how the pattern solves it. Makes the abstract concrete before the formal description.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">✅</div>
    <div class="card-title">Applicability</div>
    <div class="card-desc">Describes the situations in which the pattern is applicable. Helps you recognise when this is the right tool for the job.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">🏛️</div>
    <div class="card-title">Structure</div>
    <div class="card-desc">A graphical representation of the classes in the pattern using a class diagram, showing their relationships and collaborations.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">👥</div>
    <div class="card-title">Participants</div>
    <div class="card-desc">The classes and objects participating in the design pattern and their responsibilities.</div>
  </div>
  <div class="diagram-card teal">
    <div class="card-icon">⚖️</div>
    <div class="card-title">Consequences</div>
    <div class="card-desc">The results and trade-offs of applying the pattern. Documents both what you gain and what you give up.</div>
  </div>
  <div class="diagram-card pink">
    <div class="card-icon">🛠️</div>
    <div class="card-title">Implementation</div>
    <div class="card-desc">Hints, techniques, and pitfalls to be aware of when implementing the pattern in a specific language or context.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-icon">🔗</div>
    <div class="card-title">Related Patterns</div>
    <div class="card-desc">Other patterns that are closely related to this one — often used together, or representing alternative solutions to the same problem.</div>
  </div>
</div>
</div>

---

## How Patterns Work Together

Patterns are not islands. Real systems are built from networks of collaborating patterns, each solving a different aspect of the overall design problem.

<div class="diagram">
<div class="diagram-title">Pattern Relationships in an ML Training System</div>
<div class="flow">
  <div class="flow-node accent wide">Config File (YAML/JSON)</div>
  <div class="flow-arrow">↓ parsed by</div>
  <div class="flow-node blue">Builder<br/><small>assembles pipeline components</small></div>
  <div class="flow-arrow">↓ calls</div>
  <div class="flow-h">
    <div class="flow-node green narrow">Factory Method<br/><small>creates Model</small></div>
    <div class="flow-node green narrow">Factory Method<br/><small>creates Optimizer</small></div>
    <div class="flow-node green narrow">Abstract Factory<br/><small>creates DataPipeline</small></div>
  </div>
  <div class="flow-arrow">↓ assembled into</div>
  <div class="flow-node purple wide">Trainer (Template Method)<br/><small>defines training loop skeleton</small></div>
  <div class="flow-arrow">↓ uses</div>
  <div class="flow-h">
    <div class="flow-node orange narrow">Strategy<br/><small>loss function</small></div>
    <div class="flow-node orange narrow">Observer<br/><small>metrics/callbacks</small></div>
    <div class="flow-node orange narrow">Composite<br/><small>nn.Module tree</small></div>
  </div>
  <div class="flow-arrow">↓ produces</div>
  <div class="flow-node teal wide">Trained Model + Checkpoint (Memento)</div>
</div>
</div>

This is a typical production ML training system. Notice how many patterns appear simultaneously — each solving a distinct problem without stepping on each other's toes.

---

## Why Python and ML Need Patterns

Python's dynamic nature and ML's experimental culture both push against structure. Notebooks encourage throw-away code; rapid prototyping discourages abstraction. But as projects grow, this debt compounds rapidly.

### The Scaling Problem

A research codebase starts with 500 lines. A production ML system has 50,000. The patterns that work at 500 lines — globals, monolithic scripts, hardcoded values — collapse at 50,000. Structural patterns pay for themselves at scale.

### The Reproducibility Problem

ML experiments must be reproducible. If your model creation logic is buried in ad-hoc code, reproducing a run from six months ago becomes archaeology. Factory patterns plus configuration management make every experiment reproducible from a config file alone.

### The Collaboration Problem

As ML teams grow beyond one person, code must be readable and modifiable by people who didn't write it. Named patterns are documentation. "This is a Strategy pattern for the loss function" tells a new engineer exactly where to look and exactly what to change.

### The Testing Problem

ML code is notoriously hard to unit-test. But well-structured ML code with Dependency Inversion, Strategy, and Factory patterns is just as testable as any other Python code. You can swap real models for fakes, real data loaders for fixtures, and test logic in milliseconds.

---

## Pattern Anti-Patterns

<div class="callout warn">
<strong>⚠️ When Patterns Become Problems</strong><br/><br/>
<strong>Overengineering</strong>: Applying patterns where a simple function would do. A two-line data transform does not need a Strategy pattern. Resist the urge to pattern-ify everything.<br/><br/>
<strong>Premature Abstraction</strong>: Abstracting before you understand the problem. Write concrete code first; abstract when you see the same structure appearing three times (the "Rule of Three").<br/><br/>
<strong>Pattern Obsession</strong>: Naming and applying patterns for their own sake, making code harder to read for those unfamiliar with the catalog.<br/><br/>
<strong>Wrong Pattern</strong>: Using a Singleton when you need a Factory, or a Decorator when you need a Composite. Misapplied patterns create real structural damage.<br/><br/>
<strong>The fix</strong>: Use patterns to solve real, observed design problems. Start simple. Extract patterns only when the complexity of the alternative outweighs the cost of the pattern.
</div>

---

## The ML/AI Pattern Landscape: Beyond GoF

The 23 GoF patterns are foundational, but the ML/AI world has developed its own pattern vocabulary. This book covers all of the following:

<div class="diagram">
<div class="diagram-title">ML/AI Pattern Landscape</div>
<div class="flow">
  <div class="flow-node accent wide">ML/AI Design Patterns</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-h">
    <div class="flow-node blue narrow">
      <span class="badge creational">Creational</span><br/>
      Factory, Builder,<br/>Prototype, Singleton
    </div>
    <div class="flow-node green narrow">
      <span class="badge structural">Structural</span><br/>
      Adapter, Composite,<br/>Decorator, Proxy
    </div>
    <div class="flow-node purple narrow">
      <span class="badge behavioral">Behavioral</span><br/>
      Strategy, Observer,<br/>Template, Command
    </div>
  </div>
  <div class="flow-arrow">↓ extended by</div>
  <div class="flow-h">
    <div class="flow-node orange narrow">
      <span class="badge pytorch">PyTorch</span><br/>
      Module, Hook,<br/>Custom Autograd,<br/>DataLoader
    </div>
    <div class="flow-node teal narrow">
      <span class="badge mlops">MLOps</span><br/>
      Pipeline, Registry,<br/>Experiment Tracker,<br/>Feature Store
    </div>
    <div class="flow-node pink narrow">
      <span class="badge agentic">Agentic</span><br/>
      ReAct, Tool Use,<br/>Memory, Multi-Agent,<br/>RAG
    </div>
  </div>
  <div class="flow-arrow">↓</div>
  <div class="flow-node cyan wide">Python Idioms: Protocols, Descriptors, Context Managers, Generators</div>
</div>
</div>

---

## The Simplest Pattern: A Factory Function

Let's make patterns concrete immediately. The Factory is arguably the most universally useful pattern in ML code. Here is its simplest Python form — a function that creates objects based on a string key.

```python
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any


# ── Model definitions ─────────────────────────────────────────────────────────

class SimpleMLP(nn.Module):
    """A straightforward multi-layer perceptron."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleCNN(nn.Module):
    """A minimal convolutional network for image tasks."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x).flatten(1)
        return self.classifier(feat)


# ── Registry ──────────────────────────────────────────────────────────────────

# Maps string keys to (class, required_kwargs)
_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "mlp": SimpleMLP,
    "cnn": SimpleCNN,
}


# ── Factory function ──────────────────────────────────────────────────────────

def create_model(model_type: str, **kwargs: Any) -> nn.Module:
    """
    Factory function: instantiate a model by name.

    Args:
        model_type: One of the registered model keys.
        **kwargs:   Constructor arguments forwarded to the model class.

    Returns:
        An initialised nn.Module.

    Raises:
        ValueError: When ``model_type`` is not registered.

    Example::

        model = create_model("mlp", input_dim=784, hidden_dim=256, output_dim=10)
        cnn   = create_model("cnn", in_channels=3, num_classes=100)
    """
    if model_type not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model type '{model_type}'. Available: {available}"
        )
    cls = _MODEL_REGISTRY[model_type]
    return cls(**kwargs)


# ── Usage ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Driven entirely by config — no if/elif chains in calling code
    configs = [
        {"model_type": "mlp", "input_dim": 784, "hidden_dim": 256, "output_dim": 10},
        {"model_type": "cnn", "in_channels": 3, "num_classes": 10},
    ]

    for cfg in configs:
        model_type = cfg.pop("model_type")
        model = create_model(model_type, **cfg)
        print(f"{model_type}: {sum(p.numel() for p in model.parameters()):,} params")
```

This 60-line example captures the essence of the Factory pattern:
- **Clients** (`__main__` block) never import concrete model classes directly.
- **The registry** is the single source of truth for which model types exist.
- **Adding a new model** requires registering it in `_MODEL_REGISTRY` — existing client code requires zero changes.
- **Configuration-driven**: the entire model selection is driven by a string in a config dict.

We will return to this pattern in much greater depth in Chapter 4, adding decorator-based registration, Abstract Factory for whole data-pipeline families, and Hydra integration.

---

## Summary

Design patterns are named, documented solutions to recurring design problems. Born in architecture and formalized by the Gang of Four in 1994, they provide a shared vocabulary and a catalogue of proven designs. In ML/AI systems, patterns are not academic exercises — they are practical tools for building reproducible, extensible, testable, and maintainable code.

This book covers all 23 GoF patterns through the lens of Python and ML, then extends to PyTorch-specific patterns, MLOps infrastructure patterns, and modern agentic AI patterns. The journey begins with the foundations: SOLID principles and Python idioms, which underpin every pattern we will study.

<div class="callout tip">
<strong>💡 How to Read This Book</strong><br/>
Each chapter is self-contained — you can jump to a specific pattern when you need it. But the first three chapters (Introduction, SOLID Principles, Python Idioms) form essential groundwork. If you're new to patterns, read them in order. If you're experienced, use the Table of Contents as a reference index.
</div>

---

**Next: [Chapter 2 — SOLID Principles in Python →](./02_solid_principles.md)**

*Last updated: May 2026*
