---
title: "Chapter 2 — SOLID Principles in Python"
---

[← Back to Table of Contents](./README.md)

# Chapter 2 — SOLID Principles in Python

> *"The goal of software architecture is to minimize the human resources required to build and maintain the required system."*
> — Robert C. Martin, *Clean Architecture*, 2017

---

## What Is SOLID?

SOLID is an acronym coined by Robert C. Martin ("Uncle Bob") that collects five foundational principles of object-oriented design. Each principle targets a specific class of structural problem that causes code to become brittle, rigid, and hard to change over time.

| Letter | Principle | One-Line Definition |
|--------|-----------|---------------------|
| **S** | Single Responsibility Principle | A class should have only one reason to change |
| **O** | Open/Closed Principle | Open for extension, closed for modification |
| **L** | Liskov Substitution Principle | Subtypes must be substitutable for their base types |
| **I** | Interface Segregation Principle | Clients should not be forced to depend on interfaces they don't use |
| **D** | Dependency Inversion Principle | Depend on abstractions, not concretions |

These principles are not rules to be applied mechanically — they are design forces whose tension must be balanced. A codebase that violates all five is unmaintainable. A codebase that rigidly enforces all five becomes over-engineered. The goal is to apply them deliberately, knowing what problem each one solves.

---

## Why SOLID Matters for ML Code

<div class="callout info">
<strong>The ML Code Debt Crisis</strong><br/>
Google's 2015 paper "Hidden Technical Debt in Machine Learning Systems" identified that machine learning code is surrounded by a vast amount of configuration, data processing, feature engineering, and serving infrastructure — and that this surrounding code is where most of the debt accumulates. SOLID principles directly address the structural problems that create this debt: monolithic classes, rigid type hierarchies, and concrete dependencies that make refactoring painful.
</div>

ML code is particularly susceptible to SOLID violations because:

- **Rapid iteration**: Experiments are hacked together quickly, with SRP violations (one class doing data loading, augmentation, batching, and caching) being the most common result.
- **Deep inheritance hierarchies**: Framework-driven design (subclassing `nn.Module`, `pl.LightningModule`, `BaseEstimator`) creates inheritance chains where LSP violations are easy to introduce.
- **Monolithic trainers**: Production training scripts often evolve into God Objects that know about every component — violating both SRP and ISP.
- **Hard-coded dependencies**: `optimizer = torch.optim.Adam(...)` buried in a training loop violates DIP and makes testing and swapping components nearly impossible.

---

## S — Single Responsibility Principle

### The Principle

> *"A class should have only one reason to change."*

A class has a single responsibility when its entire body of code serves one, and only one, actor — a specific stakeholder or part of the system that might ask for changes. If two different kinds of change would require modifying the same class, it has multiple responsibilities.

The key word is **reason**: not "one method" or "one task", but one *reason to change*. A `UserReport` class that knows both how to calculate payroll and how to format it for the screen has two reasons to change: the HR department (changing payroll rules) and the UI team (changing report formatting).

### ML Example: The Overloaded Dataset Class

This is one of the most common SRP violations in ML code — a Dataset class that does too much:

```python
# ── BEFORE: SRP violation — one class with too many reasons to change ─────────

import os
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ImageDatasetDoingEverything(Dataset):
    """
    BAD EXAMPLE: This class violates SRP.
    Responsibilities mixed together:
      1. File discovery (scanning directories)
      2. Data loading (reading images from disk)
      3. Augmentation (applying transforms)
      4. Caching (storing loaded images in memory)
      5. Statistics logging (tracking load times)
      6. Train/val splitting (deciding which samples go where)
    Any change to ANY of these concerns requires touching this class.
    """

    def __init__(self, root: str, split: str = "train", cache: bool = True):
        self.root = root
        self.split = split
        self.cache = cache
        self._mem_cache: dict[int, torch.Tensor] = {}

        # Responsibility 1: File discovery
        all_paths = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith((".jpg", ".png")):
                    all_paths.append(os.path.join(dirpath, fn))

        # Responsibility 6: Splitting (logic buried inside __init__)
        random.seed(42)
        random.shuffle(all_paths)
        split_idx = int(len(all_paths) * 0.8)
        self.paths = all_paths[:split_idx] if split == "train" else all_paths[split_idx:]

        # Responsibility 3: Augmentation defined inline
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Responsibility 5: Statistics logging setup
        self.load_times: list[float] = []

    def __getitem__(self, idx: int) -> torch.Tensor:
        import time

        if idx in self._mem_cache:            # Responsibility 4: Cache lookup
            return self._mem_cache[idx]

        start = time.perf_counter()
        img = Image.open(self.paths[idx]).convert("RGB")  # Responsibility 2: Loading
        tensor = self.transform(img)          # Responsibility 3: Augmentation
        elapsed = time.perf_counter() - start

        self.load_times.append(elapsed)       # Responsibility 5: Logging

        if self.cache:
            self._mem_cache[idx] = tensor     # Responsibility 4: Caching

        return tensor

    def __len__(self) -> int:
        return len(self.paths)
```

Now let's refactor to honour SRP:

```python
# ── AFTER: SRP respected — each class has one reason to change ────────────────

from __future__ import annotations

import os
import random
import time
from typing import Callable, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Responsibility 1: File discovery — changes if directory layout changes
class ImageFileFinder:
    """Finds image file paths under a root directory."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def find(self, root: str) -> list[str]:
        paths: list[str] = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in self.EXTENSIONS:
                    paths.append(os.path.join(dirpath, fn))
        return sorted(paths)


# Responsibility 6: Splitting — changes if split strategy changes
class RandomSplitter:
    """Splits a sequence into train/val portions."""

    def __init__(self, train_fraction: float = 0.8, seed: int = 42) -> None:
        self.train_fraction = train_fraction
        self.seed = seed

    def split(self, items: list[str]) -> tuple[list[str], list[str]]:
        shuffled = items.copy()
        random.Random(self.seed).shuffle(shuffled)
        n_train = int(len(shuffled) * self.train_fraction)
        return shuffled[:n_train], shuffled[n_train:]


# Responsibility 2: Data loading — changes if image format or backend changes
class ImageLoader:
    """Loads a single image from disk as a PIL Image."""

    def load(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")


# Responsibility 4: Caching — changes if caching strategy changes
class TensorCache:
    """In-memory cache mapping integer indices to tensors."""

    def __init__(self) -> None:
        self._store: dict[int, torch.Tensor] = {}

    def get(self, idx: int) -> torch.Tensor | None:
        return self._store.get(idx)

    def set(self, idx: int, tensor: torch.Tensor) -> None:
        self._store[idx] = tensor


# Responsibility 5: Timing telemetry — changes if observability requirements change
class LoadTimeTelemetry:
    """Tracks image load durations for performance monitoring."""

    def __init__(self) -> None:
        self._times: list[float] = []

    def record(self, duration: float) -> None:
        self._times.append(duration)

    @property
    def mean_ms(self) -> float:
        if not self._times:
            return 0.0
        return 1000 * sum(self._times) / len(self._times)


# Responsibility 3: Augmentation — changes if augmentation pipeline changes
def build_train_transform() -> Callable:
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# Orchestrator: thin Dataset that delegates to each specialist
class ImageDataset(Dataset):
    """Clean Dataset: knows only how to orchestrate its collaborators."""

    def __init__(
        self,
        paths: Sequence[str],
        transform: Callable,
        cache: TensorCache | None = None,
        telemetry: LoadTimeTelemetry | None = None,
    ) -> None:
        self.paths = list(paths)
        self.transform = transform
        self._cache = cache
        self._telemetry = telemetry
        self._loader = ImageLoader()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._cache:
            cached = self._cache.get(idx)
            if cached is not None:
                return cached

        t0 = time.perf_counter()
        img = self._loader.load(self.paths[idx])
        tensor = self.transform(img)
        if self._telemetry:
            self._telemetry.record(time.perf_counter() - t0)
        if self._cache:
            self._cache.set(idx, tensor)

        return tensor


# ── Assembly ──────────────────────────────────────────────────────────────────

def build_datasets(root: str) -> tuple[ImageDataset, ImageDataset]:
    finder = ImageFileFinder()
    splitter = RandomSplitter(train_fraction=0.8)
    transform = build_train_transform()
    cache = TensorCache()
    telemetry = LoadTimeTelemetry()

    all_paths = finder.find(root)
    train_paths, val_paths = splitter.split(all_paths)

    train_ds = ImageDataset(train_paths, transform, cache=cache, telemetry=telemetry)
    val_ds = ImageDataset(val_paths, transform)
    return train_ds, val_ds
```

---

## O — Open/Closed Principle

### The Principle

> *"Software entities should be open for extension, but closed for modification."*
> — Bertrand Meyer, *Object Oriented Software Construction*, 1988

A module is **open** for extension when its behaviour can be changed to meet new requirements. It is **closed** for modification when its source code is stable — other code can depend on it without fearing that it will change. These goals are achieved through abstraction: define a stable interface, implement it in many concrete ways.

In Python, the OCP is typically realised with abstract base classes, `typing.Protocol`, or the Strategy pattern.

### ML Example: Extending a Model with New Output Heads

```python
# ── BEFORE: OCP violation — adding a new task requires modifying the base ──────

class MultiTaskModel(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, task: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.task = task
        # Adding new task types requires modifying THIS class
        if task == "classification":
            self.head = torch.nn.Linear(512, 10)
        elif task == "regression":
            self.head = torch.nn.Linear(512, 1)
        # Adding "segmentation" means opening this file and editing it
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)
```

```python
# ── AFTER: OCP respected — new heads extend without modifying the base ─────────

from __future__ import annotations

import abc
from typing import Protocol

import torch
import torch.nn as nn


# Stable abstraction — never needs to change when new heads are added
class ModelHead(abc.ABC):
    """Abstract output head. Closed for modification; new heads extend it."""

    @abc.abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        ...

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        ...


# Concrete extensions — each is self-contained; backbone is never touched
class ClassificationHead(nn.Module, ModelHead):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)

    @property
    def output_dim(self) -> int:
        return self.fc.out_features


class RegressionHead(nn.Module, ModelHead):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features).squeeze(-1)

    @property
    def output_dim(self) -> int:
        return 1


# Adding segmentation: a new file, zero changes to existing code
class SegmentationHead(nn.Module, ModelHead):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(feature_dim, num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv(features)

    @property
    def output_dim(self) -> int:
        return self.conv.out_channels


# The backbone model — CLOSED: never changes when we add new heads
class BackboneModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: ModelHead) -> None:
        super().__init__()
        self.backbone = backbone
        # head is the stable abstraction — any ModelHead implementation works
        self.head = head  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)  # type: ignore[operator]


# ── Usage ─────────────────────────────────────────────────────────────────────
# backbone = torchvision.models.resnet50(pretrained=False)
# backbone.fc = nn.Identity()  # remove final classifier
#
# clf_model = BackboneModel(backbone, ClassificationHead(2048, 100))
# reg_model  = BackboneModel(backbone, RegressionHead(2048))
# seg_model  = BackboneModel(backbone, SegmentationHead(2048, 21))
```

---

## L — Liskov Substitution Principle

### The Principle

> *"Functions that use pointers or references to base classes must be able to use objects of derived classes without knowing it."*
> — Barbara Liskov, 1987

The LSP is the most subtle of the five principles. It says that wherever you use a base type, you must be able to use any subtype in its place without the program behaving incorrectly. Subtypes must *strengthen* their parents' postconditions, not *weaken* them.

LSP violations typically manifest as:
- `isinstance` checks in client code to handle subtypes differently
- Overridden methods that accept a narrower set of inputs or raise exceptions that the parent doesn't declare
- Subclass methods that return a different type than the parent declares

### ML Example: Dataset Transforms

```python
# ── BEFORE: LSP violation — a subclass changes the contract ───────────────────

class Transform:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Returns a transformed tensor of the same shape."""
        return image


class NormalizingTransform(Transform):
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return (image - image.mean()) / (image.std() + 1e-8)


class MultiOutputTransform(Transform):
    """LSP VIOLATION: returns a tuple, not a single tensor."""
    def __call__(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Breaks any code written against the Transform contract
        aug = image + torch.randn_like(image) * 0.1
        return image, aug
```

```python
# ── AFTER: LSP respected — all transforms honour the same contract ─────────────

from __future__ import annotations

import abc

import torch
import torch.nn as nn
from torchvision import transforms as T


class ImageTransform(abc.ABC):
    """
    Contract:
      - Accepts a float tensor of shape (C, H, W) with values in [0, 1].
      - Returns a float tensor of shape (C, H, W).
      - Must be idempotent on dtype (input float → output float).
    Any concrete transform that satisfies this contract can substitute any other.
    """

    @abc.abstractmethod
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        ...


class Normalize(ImageTransform):
    def __init__(self, mean: list[float], std: list[float]) -> None:
        self._fn = T.Normalize(mean, std)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self._fn(image)  # same shape, same dtype — LSP satisfied


class RandomHFlip(ImageTransform):
    def __init__(self, p: float = 0.5) -> None:
        self._p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self._p:
            return torch.flip(image, dims=[-1])
        return image  # same shape, same dtype — LSP satisfied


class ComposeTransforms(ImageTransform):
    """Composite of transforms — also satisfies ImageTransform contract."""

    def __init__(self, transforms: list[ImageTransform]) -> None:
        self._transforms = transforms

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        for t in self._transforms:
            image = t(image)  # each call satisfies contract, so composition does too
        return image


# Client code written against the abstract ImageTransform
def augment_batch(
    images: torch.Tensor,
    transform: ImageTransform,   # accepts ANY conforming subtype
) -> torch.Tensor:
    return torch.stack([transform(img) for img in images])


# All subtypes are substitutable without changing augment_batch
pipeline = ComposeTransforms([
    RandomHFlip(p=0.5),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

---

## I — Interface Segregation Principle

### The Principle

> *"Clients should not be forced to depend upon interfaces they do not use."*

A fat interface — one with many methods — forces all implementors to implement all methods, even the irrelevant ones. When the interface changes, all implementing classes must change. The ISP says to split fat interfaces into narrow, cohesive ones. Each client should see only the methods it needs.

### ML Example: The Fat Trainer Interface

```python
# ── BEFORE: ISP violation — a God interface ───────────────────────────────────

class TrainerInterface(abc.ABC):
    """Fat interface: forces every trainer to implement unrelated methods."""

    # Core training
    @abc.abstractmethod
    def train_step(self, batch: dict) -> torch.Tensor: ...
    @abc.abstractmethod
    def val_step(self, batch: dict) -> dict: ...

    # Logging (unrelated to training logic)
    @abc.abstractmethod
    def log_metric(self, name: str, value: float) -> None: ...
    @abc.abstractmethod
    def log_image(self, name: str, image: torch.Tensor) -> None: ...

    # Checkpointing (unrelated to training logic)
    @abc.abstractmethod
    def save_checkpoint(self, path: str) -> None: ...
    @abc.abstractmethod
    def load_checkpoint(self, path: str) -> None: ...

    # Distributed training (not relevant to all trainers)
    @abc.abstractmethod
    def sync_gradients(self) -> None: ...
    @abc.abstractmethod
    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor: ...
```

```python
# ── AFTER: ISP respected — small, cohesive interfaces ─────────────────────────

from __future__ import annotations

import abc
from typing import Protocol

import torch


# Interface 1: Core training loop logic
class Trainable(Protocol):
    def train_step(self, batch: dict) -> torch.Tensor: ...
    def val_step(self, batch: dict) -> dict: ...


# Interface 2: Metrics and artefact logging
class Loggable(Protocol):
    def log_metric(self, name: str, value: float) -> None: ...
    def log_image(self, name: str, image: torch.Tensor) -> None: ...


# Interface 3: Checkpoint persistence
class Checkpointable(Protocol):
    def save_checkpoint(self, path: str) -> None: ...
    def load_checkpoint(self, path: str) -> None: ...


# Interface 4: Distributed training concerns
class Distributable(Protocol):
    def sync_gradients(self) -> None: ...
    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor: ...


# A simple local trainer only needs to implement Trainable + Loggable
class LocalTrainer:
    def train_step(self, batch: dict) -> torch.Tensor:
        ...  # implementation

    def val_step(self, batch: dict) -> dict:
        ...  # implementation

    def log_metric(self, name: str, value: float) -> None:
        print(f"{name}: {value:.4f}")

    def log_image(self, name: str, image: torch.Tensor) -> None:
        pass  # local trainer doesn't log images


# A distributed trainer implements everything it needs
class DistributedTrainer(LocalTrainer):
    def sync_gradients(self) -> None:
        torch.distributed.barrier()

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        torch.distributed.all_reduce(tensor)
        return tensor

    def save_checkpoint(self, path: str) -> None:
        ...  # implementation

    def load_checkpoint(self, path: str) -> None:
        ...  # implementation


# Client code declares only what it needs
def run_training_loop(trainer: Trainable, train_batches: list[dict]) -> None:
    """This function only needs Trainable — it doesn't care about checkpointing."""
    for batch in train_batches:
        loss = trainer.train_step(batch)


def save_experiment(trainer: Checkpointable, path: str) -> None:
    """This function only needs Checkpointable."""
    trainer.save_checkpoint(path)
```

---

## D — Dependency Inversion Principle

### The Principle

> *"High-level modules should not depend on low-level modules. Both should depend on abstractions. Abstractions should not depend on details. Details should depend on abstractions."*

The DIP is the most architecturally significant of the five principles. It separates policy (business logic) from mechanism (implementation details) by introducing an abstraction layer between them. High-level code (the `Trainer`) should not directly import low-level code (a specific `torch.optim.Adam`). Both should depend on an abstract `Optimizer` protocol.

### ML Example: Trainer with Abstracted Optimizer

```python
# ── BEFORE: DIP violation — Trainer hard-wires Adam ───────────────────────────

import torch
import torch.nn as nn


class BadTrainer:
    """DIP violation: directly coupled to torch.optim.Adam."""

    def __init__(self, model: nn.Module, lr: float = 1e-3) -> None:
        self.model = model
        # Hard dependency on a concrete class — can't swap without editing here
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

```python
# ── AFTER: DIP respected — Trainer depends on abstraction ─────────────────────

from __future__ import annotations

import abc
from typing import Iterator, Protocol

import torch
import torch.nn as nn


# Abstraction: what the Trainer needs from an optimizer
class OptimizerInterface(Protocol):
    def zero_grad(self) -> None: ...
    def step(self) -> None: ...


# Abstraction: what the Trainer needs from a scheduler
class SchedulerInterface(Protocol):
    def step(self) -> None: ...
    def get_last_lr(self) -> list[float]: ...


# Abstraction: what the Trainer needs from a loss function
class LossFunction(Protocol):
    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor: ...


# High-level module — depends only on abstractions (Protocols)
class Trainer:
    """
    High-level training policy.
    Depends on abstractions; concrete classes are injected from outside.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: OptimizerInterface,
        loss_fn: LossFunction,
        scheduler: SchedulerInterface | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        return loss.item()


# ── Wiring (the "composition root") — done once, at the entry point ───────────

def build_trainer(model: nn.Module, lr: float = 1e-3) -> Trainer:
    """
    Concrete dependencies are created here, at the application boundary,
    and injected into the high-level Trainer. The Trainer never imports torch.optim.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    loss_fn = nn.CrossEntropyLoss()
    return Trainer(model, optimizer, loss_fn, scheduler)


# ── Testing becomes trivial ───────────────────────────────────────────────────

class MockOptimizer:
    """Fake optimizer for unit tests — no GPU, no gradients needed."""
    def __init__(self) -> None:
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self) -> None:
        self.zero_grad_calls += 1

    def step(self) -> None:
        self.step_calls += 1


def test_trainer_calls_optimizer():
    model = nn.Linear(10, 2)
    mock_opt = MockOptimizer()
    trainer = Trainer(model, mock_opt, nn.CrossEntropyLoss())
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    trainer.train_step(x, y)
    assert mock_opt.zero_grad_calls == 1
    assert mock_opt.step_calls == 1
```

---

## SOLID Summary: Before/After

<div class="diagram">
<div class="diagram-title">SOLID Principles — Before and After</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card red">
    <div class="card-icon">❌</div>
    <div class="card-title">Before: SRP Violation</div>
    <div class="card-desc">Dataset that loads files, applies augmentation, manages cache, splits data, and logs metrics — one class doing six jobs.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">✅</div>
    <div class="card-title">After: SRP Respected</div>
    <div class="card-desc">FileFinder, Splitter, Loader, Cache, Telemetry, Transform — each class has one job, one reason to change.</div>
  </div>
  <div class="diagram-card red">
    <div class="card-icon">❌</div>
    <div class="card-title">Before: OCP Violation</div>
    <div class="card-desc">Model with if/elif chain for task type — adding segmentation requires modifying the model class.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">✅</div>
    <div class="card-title">After: OCP Respected</div>
    <div class="card-desc">Abstract ModelHead interface — add SegmentationHead in a new file; backbone code untouched.</div>
  </div>
  <div class="diagram-card red">
    <div class="card-icon">❌</div>
    <div class="card-title">Before: LSP Violation</div>
    <div class="card-desc">Transform subclass that returns a tuple instead of a tensor — breaks all code written against the base class contract.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">✅</div>
    <div class="card-title">After: LSP Respected</div>
    <div class="card-desc">All transforms return tensors of the same shape and dtype — any transform can substitute any other in augment_batch.</div>
  </div>
  <div class="diagram-card red">
    <div class="card-icon">❌</div>
    <div class="card-title">Before: ISP Violation</div>
    <div class="card-desc">God TrainerInterface with 8 methods — a simple local trainer must stub out distributed sync methods it will never use.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">✅</div>
    <div class="card-title">After: ISP Respected</div>
    <div class="card-desc">Four small protocols: Trainable, Loggable, Checkpointable, Distributable — each client uses only what it needs.</div>
  </div>
  <div class="diagram-card red">
    <div class="card-icon">❌</div>
    <div class="card-title">Before: DIP Violation</div>
    <div class="card-desc">Trainer hard-codes Adam optimizer — can't swap to SGD, can't mock in tests, can't extend without editing Trainer.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">✅</div>
    <div class="card-title">After: DIP Respected</div>
    <div class="card-desc">Trainer depends on OptimizerInterface Protocol — inject Adam, SGD, or MockOptimizer without changing Trainer.</div>
  </div>
</div>
</div>

---

## SOLID Violations Checklist

| Principle | Violation Symptom | How to Fix |
|-----------|-------------------|------------|
| **SRP** | Class has more than one "section" separated by blank lines and different comments | Extract each concern into its own class |
| **SRP** | Class name contains "And" (e.g., `LoaderAndProcessor`) | Split into two classes |
| **OCP** | Adding a feature requires editing an existing class | Introduce an abstraction; add new implementations |
| **OCP** | `if task == "x": ... elif task == "y":` inside an `__init__` | Replace with a Strategy/Factory pattern |
| **LSP** | `isinstance` checks in client code to handle subtypes specially | Fix the subtype's contract violation |
| **LSP** | Subclass method raises `NotImplementedError` for methods it "doesn't need" | You have the wrong hierarchy; use composition |
| **ISP** | Implementing class has many `pass` or `raise NotImplementedError` stubs | Split the interface into smaller, focused protocols |
| **ISP** | Client imports an interface for one method but the interface has twenty | Use a narrow Protocol with only the needed methods |
| **DIP** | `import torch.optim.Adam` at the top of a high-level module | Inject the optimizer; depend on a Protocol |
| **DIP** | Tight coupling makes unit tests require GPU / network / filesystem | Introduce abstractions; inject fakes in tests |

---

## How SOLID Principles Reinforce Each Other

<div class="diagram">
<div class="diagram-title">SOLID Principle Relationships</div>
<div class="flow">
  <div class="flow-node accent wide">Goal: Maintainable, Extensible ML Systems</div>
  <div class="flow-arrow">↑</div>
  <div class="flow-h">
    <div class="flow-node blue narrow">DIP<br/><small>Abstractions make<br/>SRP classes<br/>independently<br/>deployable</small></div>
    <div class="flow-node green narrow">ISP<br/><small>Narrow interfaces<br/>enable DIP and<br/>make LSP easier<br/>to satisfy</small></div>
    <div class="flow-node purple narrow">LSP<br/><small>Correct hierarchies<br/>make OCP<br/>extension safe<br/>and predictable</small></div>
  </div>
  <div class="flow-arrow">↑</div>
  <div class="flow-h">
    <div class="flow-node orange narrow">OCP<br/><small>Extension without<br/>modification<br/>requires SRP<br/>to be in place</small></div>
    <div class="flow-node teal narrow">SRP<br/><small>Single-responsibility<br/>classes are the<br/>building blocks<br/>for everything else</small></div>
  </div>
  <div class="flow-arrow">↑</div>
  <div class="flow-node cyan wide">Foundation: One class, one reason to change</div>
</div>
</div>

The principles form a hierarchy. SRP is the foundation — without it, no other principle can be fully realised. OCP builds on SRP by requiring stable abstractions. LSP ensures that OCP's extension points actually work safely. ISP keeps the abstractions from becoming bloated. DIP ties everything together by ensuring that the high-level policy never depends directly on the low-level detail.

<div class="callout tip">
<strong>💡 Practical Application</strong><br/>
Don't apply all five principles to every class from day one. Start by applying SRP aggressively — most other violations stem from classes with multiple responsibilities. Once SRP is in place, the need for OCP, ISP, and DIP patterns usually becomes obvious and the refactoring straightforward.
</div>

---

**Next: [Chapter 3 — Python Idioms & Protocols as Patterns →](./03_python_idioms.md)**

*Last updated: May 2026*
