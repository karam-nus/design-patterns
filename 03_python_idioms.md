---
title: "Chapter 3 — Python Idioms & Protocols as Patterns"
---

[← Back to Table of Contents](./README.md)

# Chapter 3 — Python Idioms & Protocols as Patterns

> *"Python is a language that tries to be consistent and readable, and to have as few surprises as possible. The language has a coherent philosophy, and it is a good fit for the problems it was designed to solve."*
> — Guido van Rossum

---

## Why Python Has Language-Level Patterns

The Gang of Four's 23 patterns were formulated primarily in the context of statically typed, compiled languages like C++ and Java — languages where you must explicitly build what Python provides natively. Python's dynamic type system, first-class functions, special method protocol (`__dunder__` methods), and structural subtyping mean that many classic GoF patterns either simplify dramatically or disappear entirely.

Python replaces some patterns with language idioms:

- The **Iterator** pattern → Python `__iter__`/`__next__` + `for` loops
- The **Observer** pattern → Python generators and the `yield` keyword
- The **Strategy** pattern → first-class functions and lambdas
- The **Template Method** pattern → `abc.abstractmethod` in base classes
- The **Singleton** pattern → module-level singletons and `__new__`

But Python also introduces its own idiom-level patterns that have no direct GoF equivalent — patterns born of the language's design philosophy and deeply embedded in the idioms every experienced Python programmer uses daily.

<div class="diagram">
<div class="diagram-title">Python Idioms as Patterns</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">🔒</div>
    <div class="card-title">Context Managers</div>
    <div class="card-desc">Resource acquisition and release with guaranteed cleanup. The <code>with</code> statement pattern.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">📋</div>
    <div class="card-title">Descriptors</div>
    <div class="card-desc">Reusable attribute validation and computed properties via <code>__get__</code>/<code>__set__</code>/<code>__delete__</code>.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🌊</div>
    <div class="card-title">Generators</div>
    <div class="card-desc">Lazy evaluation and streaming data pipelines with <code>yield</code>.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">🔷</div>
    <div class="card-title">Abstract Base Classes</div>
    <div class="card-desc">Formal contracts via <code>abc.ABC</code> and <code>@abstractmethod</code>.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">🦆</div>
    <div class="card-title">typing.Protocol</div>
    <div class="card-desc">Structural subtyping — duck typing with static analysis support.</div>
  </div>
  <div class="diagram-card teal">
    <div class="card-icon">🗃️</div>
    <div class="card-title">Dataclasses</div>
    <div class="card-desc">Typed, validated configuration objects with auto-generated boilerplate.</div>
  </div>
  <div class="diagram-card pink">
    <div class="card-icon">⚡</div>
    <div class="card-title">__slots__</div>
    <div class="card-desc">Memory-efficient objects for high-throughput data structures.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-icon">📞</div>
    <div class="card-title">__call__ Protocol</div>
    <div class="card-desc">Making objects behave as functions — transforms, losses, layers.</div>
  </div>
  <div class="diagram-card yellow">
    <div class="card-icon">🖨️</div>
    <div class="card-title">__repr__ / __str__</div>
    <div class="card-desc">Self-documenting objects with rich string representations.</div>
  </div>
</div>
</div>

---

## 1. Context Managers: `__enter__` / `__exit__`

### The Pattern

A context manager is an object that defines the runtime context for execution of a `with` statement. It handles resource acquisition and release with a guarantee: the `__exit__` method is *always* called, even if an exception is raised inside the `with` block.

This is Python's implementation of the Resource Acquisition Is Initialization (RAII) idiom from C++, but simpler and more explicit.

### ML Application: Training Mode and Device Management

```python
from __future__ import annotations

import contextlib
import time
from contextlib import contextmanager
from typing import Generator, Literal

import torch
import torch.nn as nn


# ── Class-based context manager ───────────────────────────────────────────────

class ModelInferenceContext:
    """
    Switches a model to eval mode and disables gradient computation
    for the duration of the block; restores original state on exit.

    Usage::

        with ModelInferenceContext(model):
            predictions = model(inputs)
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._was_training: bool = False
        self._grad_ctx: contextlib.AbstractContextManager | None = None

    def __enter__(self) -> nn.Module:
        self._was_training = self.model.training
        self.model.eval()
        self._grad_ctx = torch.no_grad()
        self._grad_ctx.__enter__()
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        if self._grad_ctx is not None:
            self._grad_ctx.__exit__(exc_type, exc_val, exc_tb)
        if self._was_training:
            self.model.train()
        return False  # do not suppress exceptions


# ── Generator-based context manager ──────────────────────────────────────────

@contextmanager
def cuda_memory_tracking(label: str = "") -> Generator[None, None, None]:
    """
    Tracks peak CUDA memory allocated during a block.
    Useful for memory profiling training steps.
    """
    if not torch.cuda.is_available():
        yield
        return

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()

    yield

    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    prefix = f"[{label}] " if label else ""
    print(
        f"{prefix}Δmem: {(after - before) / 1e6:+.1f} MB  "
        f"peak: {peak / 1e6:.1f} MB"
    )


@contextmanager
def timer(label: str = "block") -> Generator[None, None, None]:
    """Simple wall-clock timer context manager."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    print(f"[{label}] {elapsed * 1000:.2f} ms")


# ── Usage ─────────────────────────────────────────────────────────────────────

class DeviceContext:
    """
    Temporarily moves a model to a different device.
    Ensures the model returns to the original device even on exception.
    """

    def __init__(self, model: nn.Module, device: str | torch.device) -> None:
        self.model = model
        self.target_device = torch.device(device)
        self._original_device: torch.device | None = None

    def __enter__(self) -> nn.Module:
        # Detect original device from first parameter
        try:
            self._original_device = next(self.model.parameters()).device
        except StopIteration:
            self._original_device = torch.device("cpu")
        self.model.to(self.target_device)
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        if self._original_device is not None:
            self.model.to(self._original_device)
        return False


# Example usage pattern
# model = MyModel()
# with ModelInferenceContext(model) as eval_model:
#     with cuda_memory_tracking("forward pass"):
#         logits = eval_model(batch)
```

---

## 2. Descriptors: Validated Hyperparameter Attributes

### The Pattern

A descriptor is any object that defines `__get__`, `__set__`, or `__delete__`. When a descriptor is assigned as a *class* attribute, Python routes all attribute access on *instances* through these methods. This enables attribute-level validation, type coercion, lazy computation, and logging — without the client code knowing anything special is happening.

Descriptors power Python's `@property`, `classmethod`, `staticmethod`, and the entire Django model field system.

### ML Application: Config Class with Validated Hyperparameters

```python
from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


# ── Generic typed, validated descriptor ──────────────────────────────────────

class Bounded(Generic[T]):
    """
    Descriptor that enforces numeric bounds and type on an attribute.

    Example::

        class Config:
            learning_rate = Bounded(float, lo=1e-7, hi=1.0)
            batch_size    = Bounded(int,   lo=1,    hi=8192)
    """

    def __init__(
        self,
        dtype: type,
        lo: T | None = None,
        hi: T | None = None,
        default: T | None = None,
    ) -> None:
        self.dtype = dtype
        self.lo = lo
        self.hi = hi
        self.default = default
        self._attr: str = ""  # set by __set_name__

    def __set_name__(self, owner: type, name: str) -> None:
        # Called when the class body is processed
        # Stores the mangled private name to avoid collisions
        self._attr = f"_bounded_{name}"

    def __get__(self, obj: Any, objtype: type | None = None) -> T:
        if obj is None:
            return self  # type: ignore[return-value]
        return getattr(obj, self._attr, self.default)

    def __set__(self, obj: Any, value: Any) -> None:
        value = self.dtype(value)  # coerce type
        if self.lo is not None and value < self.lo:
            raise ValueError(f"{self._attr}: {value} < minimum {self.lo}")
        if self.hi is not None and value > self.hi:
            raise ValueError(f"{self._attr}: {value} > maximum {self.hi}")
        setattr(obj, self._attr, value)

    def __delete__(self, obj: Any) -> None:
        setattr(obj, self._attr, self.default)


class OneOf:
    """Descriptor that restricts an attribute to a fixed set of values."""

    def __init__(self, *choices: Any) -> None:
        self.choices = frozenset(choices)
        self._attr = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self._attr = f"_oneof_{name}"

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return getattr(obj, self._attr, next(iter(self.choices)))

    def __set__(self, obj: Any, value: Any) -> None:
        if value not in self.choices:
            raise ValueError(f"Expected one of {sorted(str(c) for c in self.choices)}, got {value!r}")
        setattr(obj, self._attr, value)


# ── Config class using descriptors ────────────────────────────────────────────

class TrainingConfig:
    """
    Typed, validated training configuration.
    All validation is handled transparently by descriptors.
    """

    learning_rate: float = Bounded(float, lo=1e-7, hi=1.0, default=1e-3)  # type: ignore[assignment]
    weight_decay: float  = Bounded(float, lo=0.0,  hi=1.0, default=1e-4)  # type: ignore[assignment]
    batch_size: int      = Bounded(int,   lo=1,    hi=8192, default=32)    # type: ignore[assignment]
    num_epochs: int      = Bounded(int,   lo=1,    hi=10_000, default=100) # type: ignore[assignment]
    optimizer: str       = OneOf("adam", "adamw", "sgd", "rmsprop")        # type: ignore[assignment]
    scheduler: str       = OneOf("cosine", "step", "plateau", "none")      # type: ignore[assignment]

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        fields = ["learning_rate", "weight_decay", "batch_size",
                  "num_epochs", "optimizer", "scheduler"]
        items = ", ".join(f"{f}={getattr(self, f)!r}" for f in fields)
        return f"TrainingConfig({items})"


# ── Usage ─────────────────────────────────────────────────────────────────────

cfg = TrainingConfig(learning_rate=3e-4, batch_size=64, optimizer="adamw")
print(cfg)  # TrainingConfig(learning_rate=0.0003, ...)

try:
    cfg.learning_rate = 2.0  # raises ValueError: above maximum 1.0
except ValueError as e:
    print(f"Validation error: {e}")

try:
    cfg.optimizer = "adagrad"  # raises ValueError: not in choices
except ValueError as e:
    print(f"Validation error: {e}")
```

---

## 3. Generators and Lazy Evaluation

### The Pattern

A generator is a function that yields values one at a time, suspending execution between yields. Python generators implement the Iterator protocol (`__iter__` and `__next__`) automatically. Their key property is **lazy evaluation**: values are computed on demand, not all at once.

For ML data pipelines, this is enormously important: datasets can be larger than RAM, preprocessing can be expensive, and batching strategies vary. Generators let you compose arbitrary pipeline stages without materialising intermediate results.

### ML Application: Streaming Dataset Generator

```python
from __future__ import annotations

import gzip
import json
import os
import random
from pathlib import Path
from typing import Generator, Iterable, Iterator, TypeVar

import torch

T = TypeVar("T")


# ── Streaming file reader ─────────────────────────────────────────────────────

def read_jsonl(path: str | Path) -> Generator[dict, None, None]:
    """Lazily yields records from a .jsonl or .jsonl.gz file."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ── Pipeline stages as generator transformers ─────────────────────────────────

def filter_records(
    records: Iterable[dict],
    min_length: int = 10,
) -> Generator[dict, None, None]:
    """Yield only records with text longer than min_length."""
    for rec in records:
        if len(rec.get("text", "")) >= min_length:
            yield rec


def tokenize_records(
    records: Iterable[dict],
    tokenizer,  # any callable that takes str → list[int]
    max_length: int = 512,
) -> Generator[dict, None, None]:
    """Tokenize the 'text' field of each record."""
    for rec in records:
        tokens = tokenizer(rec["text"])[:max_length]
        yield {**rec, "input_ids": tokens, "length": len(tokens)}


def batch_records(
    records: Iterable[T],
    batch_size: int,
) -> Generator[list[T], None, None]:
    """Collect records into fixed-size batches."""
    batch: list[T] = []
    for rec in records:
        batch.append(rec)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # yield final partial batch
        yield batch


def shuffle_buffer(
    records: Iterable[T],
    buffer_size: int = 1000,
    seed: int = 42,
) -> Generator[T, None, None]:
    """
    Approximate streaming shuffle using a reservoir buffer.
    Provides randomness without loading the full dataset into memory.
    """
    rng = random.Random(seed)
    buffer: list[T] = []
    for rec in records:
        buffer.append(rec)
        if len(buffer) >= buffer_size:
            idx = rng.randrange(len(buffer))
            yield buffer[idx]
            buffer[idx] = buffer[-1]
            buffer.pop()
    rng.shuffle(buffer)
    yield from buffer


# ── Composing a full pipeline ─────────────────────────────────────────────────

def build_streaming_pipeline(
    data_path: str,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    shuffle: bool = True,
) -> Generator[list[dict], None, None]:
    """
    Compose a lazy streaming data pipeline.
    Memory usage is bounded by buffer_size, not dataset size.
    """
    records = read_jsonl(data_path)
    records = filter_records(records, min_length=10)
    records = tokenize_records(records, tokenizer, max_length)
    if shuffle:
        records = shuffle_buffer(records, buffer_size=10_000)
    yield from batch_records(records, batch_size)


# ── Infinite training generator ───────────────────────────────────────────────

def infinite_dataloader(
    dataset_path: str,
    tokenizer,
    batch_size: int = 32,
) -> Generator[list[dict], None, None]:
    """Endlessly cycle over the dataset, re-shuffling each epoch."""
    epoch = 0
    while True:
        yield from build_streaming_pipeline(
            dataset_path, tokenizer, batch_size,
            shuffle=True,
        )
        epoch += 1
        print(f"Completed epoch {epoch}")
```

---

## 4. Abstract Base Classes: Defining ML Component Contracts

### The Pattern

`abc.ABC` (Abstract Base Class) combined with `@abstractmethod` lets you define a *nominal* interface in Python: a class that declares methods that must be overridden by concrete subclasses. Unlike `Protocol` (structural), ABC uses inheritance. Trying to instantiate an abstract class raises `TypeError`.

ABCs are appropriate when you want *inheritance-based* polymorphism and a nominal type hierarchy, and when all implementors will explicitly subclass the base.

### ML Application: Formal Component Contracts

```python
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


# ── Abstract base for all augmentation policies ───────────────────────────────

class AugmentationPolicy(abc.ABC):
    """
    Formal contract for data augmentation strategies.
    All subclasses must implement `augment` and `describe`.
    """

    @abc.abstractmethod
    def augment(self, image: torch.Tensor) -> torch.Tensor:
        """Apply augmentation; return tensor of same shape."""
        ...

    @abc.abstractmethod
    def describe(self) -> str:
        """Return a human-readable description of this policy."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.describe()})"


class NoAugmentation(AugmentationPolicy):
    def augment(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def describe(self) -> str:
        return "identity"


class GaussianNoiseAugmentation(AugmentationPolicy):
    def __init__(self, std: float = 0.1) -> None:
        self.std = std

    def augment(self, image: torch.Tensor) -> torch.Tensor:
        return image + torch.randn_like(image) * self.std

    def describe(self) -> str:
        return f"gaussian_noise(std={self.std})"


# ── Abstract base for metrics ─────────────────────────────────────────────────

class Metric(abc.ABC):
    """Contract for streaming metrics (compute incrementally, then finalise)."""

    @abc.abstractmethod
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate predictions and targets."""
        ...

    @abc.abstractmethod
    def compute(self) -> float:
        """Return the final metric value."""
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset all accumulated state."""
        ...

    def __call__(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> float:
        self.update(predictions, targets)
        result = self.compute()
        self.reset()
        return result


class Accuracy(Metric):
    def __init__(self) -> None:
        self._correct = 0
        self._total = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        predicted_classes = predictions.argmax(dim=-1)
        self._correct += (predicted_classes == targets).sum().item()
        self._total += targets.numel()

    def compute(self) -> float:
        return self._correct / self._total if self._total > 0 else 0.0

    def reset(self) -> None:
        self._correct = 0
        self._total = 0
```

---

## 5. `typing.Protocol`: Structural Subtyping

### The Pattern

`typing.Protocol` (PEP 544, Python 3.8+) enables *structural subtyping* — a form of duck typing with static analysis support. A class satisfies a Protocol if it has the required methods and attributes, without needing to explicitly inherit from it. This is how Go's interfaces work.

Protocols are appropriate when you want duck typing but also want `mypy`/`pyright` to catch violations. They're ideal for library code where you don't control the implementors.

### ML Application: Duck-Typed Interfaces

```python
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn


# ── Protocols as structural interfaces ───────────────────────────────────────

@runtime_checkable
class Schedulable(Protocol):
    """Any object with a step() and get_last_lr() method."""
    def step(self) -> None: ...
    def get_last_lr(self) -> list[float]: ...


@runtime_checkable
class Saveable(Protocol):
    """Any object that knows how to save and load itself."""
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


@runtime_checkable
class Forwardable(Protocol):
    """Any object that can process a tensor (nn.Module satisfies this)."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


# ── These classes satisfy the protocols WITHOUT inheriting from them ──────────

class CustomScheduler:
    """
    A custom LR scheduler that satisfies Schedulable
    without inheriting from it or any torch class.
    """
    def __init__(self, base_lr: float, decay: float = 0.95) -> None:
        self.lr = base_lr
        self.decay = decay

    def step(self) -> None:
        self.lr *= self.decay

    def get_last_lr(self) -> list[float]:
        return [self.lr]


class ExponentialMovingAverage:
    """
    EMA for model weights.
    Satisfies Saveable without inheriting from nn.Module.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.shadow = state["shadow"]
        self.decay = state["decay"]


# ── Functions written against protocols accept anything that satisfies them ───

from typing import Any


def run_scheduler_epoch(scheduler: Schedulable, num_steps: int) -> list[float]:
    """Works with torch schedulers AND CustomScheduler — no isinstance needed."""
    lrs = []
    for _ in range(num_steps):
        scheduler.step()
        lrs.extend(scheduler.get_last_lr())
    return lrs


def checkpoint_object(obj: Saveable, path: str) -> None:
    """Saves any Saveable — nn.Module, EMA, optimizer, etc."""
    import pickle
    with open(path, "wb") as f:
        pickle.dump(obj.state_dict(), f)


# Runtime check with @runtime_checkable
scheduler = CustomScheduler(base_lr=1e-3)
print(isinstance(scheduler, Schedulable))  # True
```

---

## 6. Dataclasses: Typed Config Objects

### The Pattern

`@dataclass` (Python 3.7+) auto-generates `__init__`, `__repr__`, `__eq__`, and optionally `__hash__` from class-level type annotations. Combined with `__post_init__` for validation, frozen dataclasses, and `field(default_factory=...)` for mutable defaults, dataclasses are the standard pattern for typed configuration objects in modern Python ML code.

### ML Application: Validated Config Hierarchy

```python
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if not (0 < self.betas[0] < 1 and 0 < self.betas[1] < 1):
            raise ValueError(f"betas must be in (0, 1), got {self.betas}")


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_steps: int = 500
    max_steps: int = 10_000
    min_lr_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.warmup_steps >= self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be < max_steps ({self.max_steps})"
            )


@dataclass
class ModelConfig:
    architecture: str = "transformer"
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1
    vocab_size: int = 50_257

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )


@dataclass
class TrainingConfig:
    """Top-level config; composes sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    batch_size: int = 32
    num_epochs: int = 10
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.gradient_clip <= 0:
            raise ValueError(f"gradient_clip must be positive, got {self.gradient_clip}")

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten nested config to a flat dict for logging."""
        result: dict[str, Any] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if hasattr(val, "__dataclass_fields__"):
                for inner_f in fields(val):
                    result[f"{f.name}.{inner_f.name}"] = getattr(val, inner_f.name)
            else:
                result[f.name] = val
        return result


# ── Usage ─────────────────────────────────────────────────────────────────────

cfg = TrainingConfig(
    model=ModelConfig(hidden_dim=512, num_layers=6, num_heads=8),
    optimizer=OptimizerConfig(lr=1e-4),
    batch_size=64,
)
print(cfg)
print(cfg.to_flat_dict())
```

---

## 7. `__slots__`: Memory-Efficient Objects

### The Pattern

By default, Python stores instance attributes in a `__dict__` (a hash map). `__slots__` replaces this with a fixed layout of memory slots — more like a C struct. The benefits: ~40-60% less memory per object, faster attribute access, and prevention of accidental typos in attribute names.

Slots are most valuable when you create millions of lightweight objects — batches, tokens, data points, trajectory steps in RL.

### ML Application: Memory-Efficient Data Structures

```python
from __future__ import annotations

import sys
from dataclasses import dataclass

import torch


# ── Without slots: Python dict overhead ──────────────────────────────────────

class TokenWithoutSlots:
    def __init__(self, token_id: int, position: int, attention_mask: int) -> None:
        self.token_id = token_id
        self.position = position
        self.attention_mask = attention_mask


# ── With slots: ~50% smaller footprint ───────────────────────────────────────

class Token:
    """
    Memory-efficient token representation.
    For a 512-token sequence repeated across 100k examples,
    __slots__ saves tens of megabytes vs. plain __dict__.
    """

    __slots__ = ("token_id", "position", "attention_mask")

    def __init__(self, token_id: int, position: int, attention_mask: int) -> None:
        self.token_id = token_id
        self.position = position
        self.attention_mask = attention_mask


# ── RL trajectory step with slots ────────────────────────────────────────────

class Transition:
    """
    A single RL transition (s, a, r, s', done).
    Millions of these are stored in a replay buffer — slots matter.
    """

    __slots__ = ("state", "action", "reward", "next_state", "done")

    def __init__(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


# ── Demonstration ─────────────────────────────────────────────────────────────

t_no_slots = TokenWithoutSlots(42, 0, 1)
t_slots = Token(42, 0, 1)

print(f"Without __slots__: {sys.getsizeof(t_no_slots.__dict__) + sys.getsizeof(t_no_slots)} bytes")
print(f"With __slots__:    {sys.getsizeof(t_slots)} bytes")
```

---

## 8. `__call__` Protocol: Making Objects Callable

### The Pattern

Any Python object with a `__call__` method can be called as if it were a function. This is central to PyTorch's `nn.Module`, scikit-learn's transformers, and most ML frameworks. Callable objects are more powerful than plain functions because they can carry state — learned parameters, accumulated statistics, configuration.

### ML Application: Transforms, Loss Functions, and Layers

```python
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Stateful callable: a transform with a configurable kernel ─────────────────

class GaussianBlur:
    """
    A stateful callable that applies Gaussian blur.
    Stores the precomputed kernel — an advantage over a plain function.
    """

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0) -> None:
        self.kernel_size = kernel_size
        # Precompute kernel once at construction time
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = gauss / gauss.sum()
        self._kernel = torch.outer(kernel_1d, kernel_1d).unsqueeze(0).unsqueeze(0)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply blur; handles batched and single images."""
        c = image.shape[-3]
        kernel = self._kernel.expand(c, 1, -1, -1).to(image.device)
        if image.dim() == 3:
            image = image.unsqueeze(0)
            return F.conv2d(image, kernel, padding=self.kernel_size // 2, groups=c).squeeze(0)
        return F.conv2d(image, kernel, padding=self.kernel_size // 2, groups=c)

    def __repr__(self) -> str:
        return f"GaussianBlur(kernel_size={self.kernel_size})"


# ── Callable loss function with class weighting ───────────────────────────────

class WeightedCrossEntropyLoss:
    """
    Wraps nn.CrossEntropyLoss with class-frequency-based weighting.
    Callable interface matches the loss_fn: (preds, targets) -> scalar pattern.
    """

    def __init__(self, class_counts: list[int]) -> None:
        counts = torch.tensor(class_counts, dtype=torch.float)
        # Inverse frequency weighting
        weights = 1.0 / (counts + 1)
        weights = weights / weights.sum() * len(counts)
        self._loss_fn = nn.CrossEntropyLoss(weight=weights)

    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return self._loss_fn(predictions, targets)

    def __repr__(self) -> str:
        return f"WeightedCrossEntropyLoss(n_classes={len(self._loss_fn.weight)})"


# ── Pipeline of callables ─────────────────────────────────────────────────────

class TransformPipeline:
    """Chains callables; each callable takes and returns a tensor."""

    def __init__(self, *transforms: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self._transforms = list(transforms)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self._transforms:
            x = t(x)
        return x

    def __len__(self) -> int:
        return len(self._transforms)

    def __repr__(self) -> str:
        names = [getattr(t, "__name__", repr(t)) for t in self._transforms]
        return f"TransformPipeline([{', '.join(names)}])"
```

---

## 9. `__repr__` and `__str__`: Self-Documenting Models

### The Pattern

`__repr__` should return an unambiguous string that, ideally, could reconstruct the object. `__str__` should return a human-readable string for display. In ML code, rich representations make debugging, logging, and experiment tracking dramatically easier.

### ML Application: Model and Config Pretty-Printing

```python
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn as nn


class ModelSummary:
    """Mixin that adds rich __repr__ to any nn.Module."""

    def extra_repr(self) -> str:
        """Override in subclasses to add custom parameter strings."""
        return ""

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        # Reuse nn.Module's built-in tree repr and augment it
        base = super().__repr__()  # type: ignore[misc]
        n = self.num_parameters()
        return f"{base}\n  Trainable parameters: {n:,}"


class TransformerBlock(ModelSummary, nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_heads={self.n_heads}, dropout={self.dropout}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


@dataclass
class ExperimentConfig:
    """Config with a rich __str__ for experiment logging."""
    run_name: str
    model_type: str
    learning_rate: float
    batch_size: int
    num_epochs: int
    tags: list[str]

    def __str__(self) -> str:
        lines = [f"Experiment: {self.run_name}", "=" * 40]
        for f in fields(self):
            lines.append(f"  {f.name:<20} = {getattr(self, f.name)!r}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        args = ", ".join(
            f"{f.name}={getattr(self, f.name)!r}" for f in fields(self)
        )
        return f"ExperimentConfig({args})"


# ── Usage ─────────────────────────────────────────────────────────────────────
block = TransformerBlock(d_model=512, n_heads=8)
print(block)

cfg = ExperimentConfig(
    run_name="gpt2-small-finetune",
    model_type="transformer",
    learning_rate=3e-4,
    batch_size=32,
    num_epochs=5,
    tags=["nlp", "finetuning", "baseline"],
)
print(cfg)
print(repr(cfg))
```

---

## GoF Pattern → Python Idiom Equivalence

| GoF Pattern | Python Idiom Equivalent | Notes |
|-------------|------------------------|-------|
| **Iterator** | `__iter__` / `__next__` / generators | Built into the language; `for x in obj` works automatically |
| **Template Method** | `abc.abstractmethod` | Abstract base classes with concrete and abstract methods |
| **Strategy** | First-class functions / callables | Pass a function or callable object instead of a Strategy class |
| **Observer** | `yield` / generators / callbacks | Generator coroutines or callback lists replace explicit Observer hierarchy |
| **Decorator (GoF)** | `@functools.wraps` / class decorators | Python's `@` syntax is syntactic sugar for wrapping |
| **Singleton** | Module-level variable | Modules are loaded once; a module-level object is a singleton |
| **Prototype** | `copy.deepcopy` | Python's copy protocol handles deep cloning |
| **Command** | `functools.partial` / closures | Encapsulate a call with its arguments as a callable |
| **Chain of Responsibility** | Generator pipeline / middleware | Each stage `yield from` the next |
| **Memento** | `__getstate__` / `__setstate__` / `pickle` | Python's pickling protocol is the canonical memento mechanism |
| **Flyweight** | `__slots__` + interning | `sys.intern` for strings; `__slots__` for lightweight objects |
| **Composite** | `__iter__` on container classes | Any iterable container acts as a Composite node |
| **Facade** | Module with a public API | A Python module naturally facades its internal complexity |

---

<div class="callout tip">
<strong>💡 When to Use Each Idiom</strong><br/><br/>
<strong>Context Manager</strong>: Any time you have setup/teardown, resource acquisition/release, or want to guarantee cleanup on exceptions.<br/><br/>
<strong>Descriptor</strong>: When the same validation or transformation logic applies to multiple attributes across multiple classes — don't repeat it in every property setter.<br/><br/>
<strong>Generator</strong>: Data larger than memory, lazy pipelines, infinite sequences. If you're building a list and consuming it once, convert to a generator.<br/><br/>
<strong>ABC</strong>: When you want a nominal type hierarchy, all implementors will inherit from the base, and you want instantiation to fail for incomplete implementations.<br/><br/>
<strong>Protocol</strong>: When you don't control the implementors, or when you want duck typing with static analysis. Prefer Protocol over ABC for library code.<br/><br/>
<strong>Dataclass</strong>: For configuration objects, value objects, and any class that is primarily a container of typed attributes.<br/><br/>
<strong>__slots__</strong>: When creating millions of lightweight objects (tokens, transitions, events). Profile first — premature slot optimisation is a real trap.<br/><br/>
<strong>__call__</strong>: When an object needs to behave as a function but also needs to carry state or configuration. The nn.Module pattern.<br/><br/>
<strong>__repr__</strong>: Always. Good representations make debugging faster and logs more useful.
</div>

---

**Next: [Chapter 4 — Factory & Abstract Factory →](./04_factory.md)**

*Last updated: May 2026*
