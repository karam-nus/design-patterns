---
title: "Chapter 15 — Strategy Pattern"
---
[← Back to Table of Contents](./README.md)

# Chapter 15 — Strategy Pattern

> *"Define a family of algorithms, encapsulate each one, and make them interchangeable. Strategy lets the algorithm vary independently from the clients that use it."*
> — Gang of Four

---

## Intent and Motivation

Every ML system eventually accumulates a tangle of `if/elif` chains:

```python
# The symptom: a switch forest
if config.loss == "cross_entropy":
    loss = F.cross_entropy(logits, labels)
elif config.loss == "focal":
    loss = focal_loss(logits, labels, gamma=2.0)
elif config.loss == "label_smoothing":
    loss = label_smoothing_loss(logits, labels, smoothing=0.1)
# ... 8 more branches ...
```

Every time a new loss function is added, this function grows. Testing requires exercising every branch. Adding a parameter to one loss function requires touching this central dispatcher. The **Strategy pattern** eliminates the switch: each algorithm becomes an independent object, injected into the context at construction or runtime.

---

## GoF Strategy: UML Structure

<div class="diagram">
  <div class="diagram-title">Strategy Pattern — UML</div>
  <div class="uml-row">
    <div class="uml-box" style="border-color: var(--accent)">
      <div class="uml-title">Context (Trainer)</div>
      <div class="uml-section">
        <div class="uml-item">- strategy: Strategy</div>
      </div>
      <div class="uml-section">
        <div class="uml-item">+ set_strategy(s: Strategy)</div>
        <div class="uml-item">+ execute_strategy()</div>
      </div>
    </div>
    <div class="uml-box">
      <div class="uml-title">«interface»<br>Strategy</div>
      <div class="uml-section">
        <div class="uml-item">+ execute(*args) → result</div>
      </div>
    </div>
  </div>
  <div class="uml-row" style="margin-top:1rem">
    <div class="uml-box">
      <div class="uml-title">ConcreteStrategyA</div>
      <div class="uml-section">
        <div class="uml-item">+ execute(*args)</div>
      </div>
    </div>
    <div class="uml-box">
      <div class="uml-title">ConcreteStrategyB</div>
      <div class="uml-section">
        <div class="uml-item">+ execute(*args)</div>
      </div>
    </div>
    <div class="uml-box">
      <div class="uml-title">ConcreteStrategyC</div>
      <div class="uml-section">
        <div class="uml-item">+ execute(*args)</div>
      </div>
    </div>
  </div>
</div>

---

## Python Implementation: Protocol vs ABC vs Callable

Python offers three idiomatic ways to define a strategy interface. Each has trade-offs:

```python
from __future__ import annotations

import abc
from typing import Protocol, runtime_checkable
import torch
import torch.nn as nn


# ── Option 1: Protocol (structural subtyping, no inheritance required) ──────
@runtime_checkable
class LossStrategy(Protocol):
    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        ...


# ── Option 2: ABC (nominal subtyping, explicit registration) ─────────────
class LossStrategyABC(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the loss."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ── Option 3: Simple callable (duck-typed, most Pythonic for functions) ──
# Any function with signature (logits, labels) -> Tensor is a valid strategy.
# No base class or Protocol registration needed.

# All three work — choose Protocol for library APIs (documents the contract
# without imposing inheritance), ABC when you need mixins or abstract properties,
# and plain callables for configuration-driven pipelines.
```

---

## Strategy for Loss Functions

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Protocol
import math


class LossStrategy(Protocol):
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor: ...


class CrossEntropyLoss:
    """Standard cross-entropy strategy."""
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)


class FocalLoss:
    """Focal loss: down-weights easy examples, focuses on hard ones."""
    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            focal = self.alpha * focal
        return focal.mean()


class LabelSmoothingLoss:
    """Label smoothing: prevents overconfident predictions."""
    def __init__(self, smoothing: float = 0.1, num_classes: int | None = None):
        self.smoothing = smoothing
        self.num_classes = num_classes

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        n_classes = self.num_classes or logits.size(-1)
        confidence = 1.0 - self.smoothing
        smooth_val = self.smoothing / (n_classes - 1)

        # Build smoothed distribution
        one_hot = torch.zeros_like(logits).scatter_(-1, labels.unsqueeze(-1), 1)
        smooth_labels = one_hot * confidence + (1 - one_hot) * smooth_val

        log_probs = F.log_softmax(logits, dim=-1)
        return -(smooth_labels * log_probs).sum(dim=-1).mean()


class SymmetricCrossEntropyLoss:
    """SCE: robust to label noise via bidirectional cross-entropy."""
    def __init__(self, alpha: float = 0.1, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        probs = F.softmax(logits, dim=-1)
        one_hot = F.one_hot(labels, n_classes).float()

        # Forward CE
        ce = F.cross_entropy(logits, labels)
        # Reverse CE
        rce = -(probs * torch.log(one_hot.clamp(1e-4, 1.0))).sum(dim=-1).mean()
        return self.alpha * ce + self.beta * rce
```

### Plugging the Strategy into a Trainer

```python
class Trainer:
    """Context: uses an injected LossStrategy without knowing which one."""
    def __init__(
        self,
        model: nn.Module,
        loss_fn: LossStrategy,
        optimizer: torch.optim.Optimizer,
    ):
        self.model = model
        self.loss_fn = loss_fn          # injected strategy
        self.optimizer = optimizer

    def train_step(self, batch: dict) -> float:
        self.optimizer.zero_grad()
        logits = self.model(batch["input_ids"])
        loss = self.loss_fn(logits, batch["labels"])   # strategy called here
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def set_loss_strategy(self, loss_fn: LossStrategy) -> None:
        """Runtime strategy swap."""
        self.loss_fn = loss_fn


# Dependency injection at construction:
trainer_ce   = Trainer(model, CrossEntropyLoss(), optimizer)
trainer_fl   = Trainer(model, FocalLoss(gamma=2.0), optimizer)
trainer_ls   = Trainer(model, LabelSmoothingLoss(smoothing=0.1), optimizer)
```

---

## Strategy for Optimizers

```python
from typing import Protocol, Iterable
import torch
import torch.nn as nn


class OptimizerStrategy(Protocol):
    def __call__(
        self,
        parameters: Iterable[nn.Parameter],
        lr: float,
        **kwargs,
    ) -> torch.optim.Optimizer: ...


class AdamStrategy:
    def __call__(self, parameters, lr: float, **kwargs) -> torch.optim.Optimizer:
        return torch.optim.Adam(parameters, lr=lr, **kwargs)


class AdamWStrategy:
    def __init__(self, weight_decay: float = 0.01):
        self.weight_decay = weight_decay

    def __call__(self, parameters, lr: float, **kwargs) -> torch.optim.Optimizer:
        return torch.optim.AdamW(parameters, lr=lr,
                                  weight_decay=self.weight_decay, **kwargs)


class SGDMomentumStrategy:
    def __init__(self, momentum: float = 0.9, nesterov: bool = True):
        self.momentum = momentum
        self.nesterov = nesterov

    def __call__(self, parameters, lr: float, **kwargs) -> torch.optim.Optimizer:
        return torch.optim.SGD(parameters, lr=lr,
                                momentum=self.momentum,
                                nesterov=self.nesterov, **kwargs)


class LionStrategy:
    """Lion optimizer — requires `lion-pytorch` package."""
    def __call__(self, parameters, lr: float, **kwargs) -> torch.optim.Optimizer:
        try:
            from lion_pytorch import Lion
            return Lion(parameters, lr=lr, **kwargs)
        except ImportError:
            raise ImportError("Install lion-pytorch: pip install lion-pytorch")


class TrainerWithOptimizerStrategy:
    def __init__(
        self,
        model: nn.Module,
        optimizer_strategy: OptimizerStrategy,
        lr: float = 3e-4,
    ):
        self.model = model
        self.optimizer = optimizer_strategy(model.parameters(), lr=lr)

    def replace_optimizer(
        self,
        new_strategy: OptimizerStrategy,
        lr: float | None = None,
    ) -> None:
        """Switch optimizer strategy mid-training (rare but valid)."""
        current_lr = lr or self.optimizer.param_groups[0]["lr"]
        self.optimizer = new_strategy(self.model.parameters(), lr=current_lr)
```

---

## Strategy for Data Augmentation

```python
from typing import Protocol, Callable
import torch
from PIL import Image


class AugmentationStrategy(Protocol):
    def __call__(self, image: Image.Image) -> torch.Tensor: ...


class BaselineAugmentation:
    """Minimal augmentation: resize and normalise only."""
    def __init__(self, size: int = 224):
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)


class StandardAugmentation:
    """Random flip + crop + colour jitter."""
    def __init__(self, size: int = 224):
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)


class RandAugmentStrategy:
    """RandAugment: random choice from a set of augmentation policies."""
    def __init__(self, num_ops: int = 2, magnitude: int = 9, size: int = 224):
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.RandAugment(num_ops=num_ops, magnitude=magnitude),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)


class MixupStrategy:
    """Mixup augmentation applied at batch level."""
    def __init__(self, alpha: float = 0.2, base: AugmentationStrategy | None = None):
        self.alpha = alpha
        self.base = base or BaselineAugmentation()

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.base(image)   # per-sample; mixup applied in collate_fn

    def mix_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        import numpy as np
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        idx = torch.randperm(x.size(0))
        mixed_x = lam * x + (1 - lam) * x[idx]
        return mixed_x, y, y[idx], lam


class AugmentedDataset:
    """Context: uses an injected AugmentationStrategy."""
    def __init__(self, dataset, strategy: AugmentationStrategy):
        self.dataset = dataset
        self.strategy = strategy              # injected

    def set_strategy(self, strategy: AugmentationStrategy) -> None:
        self.strategy = strategy              # swap at runtime

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        return self.strategy(image), label   # strategy called here

    def __len__(self) -> int:
        return len(self.dataset)
```

---

## Strategy for Samplers

```python
import torch
from torch.utils.data import Sampler, RandomSampler, WeightedRandomSampler
from typing import Protocol, Iterator
import numpy as np


class SamplerStrategy(Protocol):
    def __call__(self, dataset) -> Sampler: ...


class RandomSamplerStrategy:
    def __call__(self, dataset) -> Sampler:
        return RandomSampler(dataset)


class WeightedRandomSamplerStrategy:
    """Sample in proportion to inverse class frequency (for imbalanced datasets)."""
    def __init__(self, labels: list[int]):
        self.labels = labels

    def __call__(self, dataset) -> Sampler:
        class_counts = np.bincount(self.labels)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = torch.tensor(
            [class_weights[label] for label in self.labels], dtype=torch.float
        )
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )


class ClassBalancedSamplerStrategy:
    """Sample equal numbers from each class per epoch."""
    def __init__(self, labels: list[int], samples_per_class: int = 100):
        self.labels = labels
        self.samples_per_class = samples_per_class

    def __call__(self, dataset) -> Sampler:
        from collections import defaultdict
        class_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            class_indices[label].append(idx)

        class BalancedSampler(Sampler):
            def __init__(self, class_indices, n_per_class):
                self.class_indices = class_indices
                self.n = n_per_class

            def __iter__(self) -> Iterator[int]:
                indices = []
                for cls_idx_list in self.class_indices.values():
                    sampled = np.random.choice(
                        cls_idx_list,
                        size=self.n,
                        replace=len(cls_idx_list) < self.n
                    )
                    indices.extend(sampled.tolist())
                np.random.shuffle(indices)
                return iter(indices)

            def __len__(self) -> int:
                return self.n * len(self.class_indices)

        return BalancedSampler(class_indices, self.samples_per_class)
```

---

## Strategy vs Switch Statements — Before/After

**Before:** the switch forest — every new algorithm grows this function:

```python
# ❌ Before Strategy pattern
def compute_loss(logits, labels, loss_type: str, **kwargs) -> torch.Tensor:
    if loss_type == "cross_entropy":
        return F.cross_entropy(logits, labels)
    elif loss_type == "focal":
        gamma = kwargs.get("gamma", 2.0)
        ce = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** gamma * ce).mean()
    elif loss_type == "label_smoothing":
        smoothing = kwargs.get("smoothing", 0.1)
        # ... 10 lines of implementation ...
        pass
    elif loss_type == "symmetric_ce":
        # ... 15 more lines ...
        pass
    elif loss_type == "sce_focal_mix":
        # ... combines two of the above, hardcoded ...
        pass
    else:
        raise ValueError(f"Unknown loss: {loss_type}")
    # Problems: untestable in isolation, violates Open/Closed, hard to combine
```

**After:** each algorithm is independent, composable, and testable:

```python
# ✅ After Strategy pattern
class Trainer:
    def __init__(self, model, loss_fn: LossStrategy, ...):
        self.loss_fn = loss_fn   # any callable (logits, labels) → scalar

    def train_step(self, batch):
        logits = self.model(batch["input_ids"])
        return self.loss_fn(logits, batch["labels"])  # ← no if/elif

# Adding a new loss: create a new class, zero changes to Trainer
class DiceLoss:
    def __call__(self, logits, labels):
        probs = torch.sigmoid(logits)
        intersection = (probs * labels).sum()
        return 1 - (2 * intersection + 1) / (probs.sum() + labels.sum() + 1)

# Zero lines changed in Trainer to support DiceLoss:
trainer = Trainer(model, DiceLoss(), optimizer)
```

---

## Combining Strategy with Factory — Config-Driven Selection

```python
from dataclasses import dataclass
from typing import Any


@dataclass
class TrainingConfig:
    loss: str = "cross_entropy"
    loss_kwargs: dict = None
    optimizer: str = "adamw"
    optimizer_kwargs: dict = None
    augmentation: str = "standard"

    def __post_init__(self):
        self.loss_kwargs = self.loss_kwargs or {}
        self.optimizer_kwargs = self.optimizer_kwargs or {}


class StrategyFactory:
    """
    Factory + Strategy: config string → concrete strategy object.
    New strategies register themselves; the factory needs no changes.
    """
    _loss_registry: dict[str, type] = {}
    _optimizer_registry: dict[str, type] = {}
    _augmentation_registry: dict[str, type] = {}

    @classmethod
    def register_loss(cls, name: str):
        def decorator(strategy_cls):
            cls._loss_registry[name] = strategy_cls
            return strategy_cls
        return decorator

    @classmethod
    def register_optimizer(cls, name: str):
        def decorator(strategy_cls):
            cls._optimizer_registry[name] = strategy_cls
            return strategy_cls
        return decorator

    @classmethod
    def create_loss(cls, name: str, **kwargs) -> LossStrategy:
        if name not in cls._loss_registry:
            raise KeyError(f"Unknown loss strategy '{name}'. "
                           f"Available: {list(cls._loss_registry)}")
        return cls._loss_registry[name](**kwargs)

    @classmethod
    def create_optimizer(cls, name: str, **kwargs) -> OptimizerStrategy:
        if name not in cls._optimizer_registry:
            raise KeyError(f"Unknown optimizer strategy '{name}'.")
        return cls._optimizer_registry[name](**kwargs)


# Register strategies via decorator:
@StrategyFactory.register_loss("cross_entropy")
class CrossEntropyLoss:
    def __call__(self, logits, labels):
        return F.cross_entropy(logits, labels)

@StrategyFactory.register_loss("focal")
class FocalLoss:
    def __init__(self, gamma: float = 2.0):
        self.gamma = gamma
    def __call__(self, logits, labels):
        ce = F.cross_entropy(logits, labels, reduction="none")
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


@StrategyFactory.register_optimizer("adamw")
class AdamWStrategy:
    def __init__(self, weight_decay: float = 0.01):
        self.weight_decay = weight_decay
    def __call__(self, parameters, lr: float, **kw):
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=self.weight_decay, **kw)


def build_trainer_from_config(model: nn.Module, cfg: TrainingConfig) -> Trainer:
    """Wire everything together from config — zero if/elif."""
    loss_fn  = StrategyFactory.create_loss(cfg.loss, **cfg.loss_kwargs)
    opt_strat = StrategyFactory.create_optimizer(cfg.optimizer, **cfg.optimizer_kwargs)
    optimizer = opt_strat(model.parameters(), lr=3e-4)
    return Trainer(model, loss_fn, optimizer)


# From YAML config (loaded via OmegaConf / Hydra):
cfg = TrainingConfig(
    loss="focal",
    loss_kwargs={"gamma": 1.5},
    optimizer="adamw",
    optimizer_kwargs={"weight_decay": 0.05},
)
trainer = build_trainer_from_config(model, cfg)
```

---

## Runtime Strategy Switching

Switching strategy mid-training based on live metrics is a powerful technique:

```python
class AdaptiveTrainer:
    """
    Context that switches loss strategy based on training dynamics.
    Example: start with label smoothing, switch to focal when loss plateaus.
    """
    def __init__(self, model: nn.Module, initial_loss: LossStrategy):
        self.model = model
        self.loss_fn = initial_loss
        self._loss_history: list[float] = []
        self._strategy_log: list[tuple[int, str]] = []

    def set_strategy(self, loss_fn: LossStrategy, step: int) -> None:
        self._strategy_log.append((step, loss_fn.__class__.__name__))
        self.loss_fn = loss_fn
        print(f"[Step {step}] Switched loss strategy → {loss_fn.__class__.__name__}")

    def train_step(self, batch: dict, step: int) -> float:
        logits = self.model(batch["input_ids"])
        loss = self.loss_fn(logits, batch["labels"])
        loss_val = loss.item()
        self._loss_history.append(loss_val)

        # Auto-switch: if loss hasn't improved for 200 steps, switch to focal
        if len(self._loss_history) >= 200:
            recent = self._loss_history[-200:]
            if max(recent) - min(recent) < 0.005:  # plateau threshold
                if not isinstance(self.loss_fn, FocalLoss):
                    self.set_strategy(FocalLoss(gamma=2.0), step)

        loss.backward()
        return loss_val

    def strategy_timeline(self) -> list[tuple[int, str]]:
        return self._strategy_log.copy()
```

---

## Flow Diagram

<div class="diagram">
  <div class="diagram-title">Strategy Pattern — Runtime Flow</div>
  <div class="flow">
    <div class="flow-node accent wide">Context (Trainer): <code>self.loss_fn(logits, labels)</code></div>
    <div class="flow-arrow accent">↓ delegates to injected strategy</div>
    <div class="flow-node blue wide">Strategy Interface: <code>LossStrategy.__call__</code></div>
    <div class="flow-arrow blue">↓ resolved at runtime</div>
    <div class="flow-h">
      <div class="flow-node orange narrow">ConcreteA<br>CrossEntropy<br>Loss</div>
      <div class="flow-node purple narrow">ConcreteB<br>Focal<br>Loss</div>
      <div class="flow-node green narrow">ConcreteC<br>LabelSmoothing<br>Loss</div>
      <div class="flow-node teal narrow">ConcreteD<br>SymmetricCE<br>Loss</div>
    </div>
    <div class="flow-arrow accent">↑ returns scalar loss tensor</div>
    <div class="flow-node accent wide">Context calls .backward() on result</div>
  </div>
</div>

---

## Strategy Families in ML

<div class="diagram-grid cols-3">
  <div class="diagram-card orange">
    <div class="card-icon">📉</div>
    <div class="card-title">Loss Strategies</div>
    <div class="card-desc">
      <strong>CrossEntropy</strong> — standard classification<br>
      <strong>Focal</strong> — class imbalance<br>
      <strong>LabelSmoothing</strong> — overconfidence<br>
      <strong>DiceLoss</strong> — segmentation<br>
      <strong>CTCLoss</strong> — sequence alignment<br>
      <strong>ContrastiveLoss</strong> — metric learning
    </div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">⚙️</div>
    <div class="card-title">Optimizer Strategies</div>
    <div class="card-desc">
      <strong>SGD+momentum</strong> — CV training<br>
      <strong>Adam</strong> — adaptive LR<br>
      <strong>AdamW</strong> — weight decay fix<br>
      <strong>Adan</strong> — aggressive convergence<br>
      <strong>Lion</strong> — sign-based updates<br>
      <strong>Adafactor</strong> — memory-efficient
    </div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🎨</div>
    <div class="card-title">Augmentation Strategies</div>
    <div class="card-desc">
      <strong>Baseline</strong> — resize + normalize<br>
      <strong>Standard</strong> — flip + jitter<br>
      <strong>RandAugment</strong> — random policy<br>
      <strong>AutoAugment</strong> — learned policy<br>
      <strong>Mixup</strong> — label interpolation<br>
      <strong>CutMix</strong> — patch mixing
    </div>
  </div>
</div>

---

## Comparison: Strategy vs Template Method vs Command

<table class="compare-table">
  <thead>
    <tr>
      <th>Dimension</th>
      <th>Strategy</th>
      <th>Template Method</th>
      <th>Command</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>What varies</strong></td>
      <td>The whole algorithm</td>
      <td>Steps within an algorithm skeleton</td>
      <td>A request (action + parameters)</td>
    </tr>
    <tr>
      <td><strong>Relationship</strong></td>
      <td>Composition (has-a)</td>
      <td>Inheritance (is-a)</td>
      <td>Composition (has-a)</td>
    </tr>
    <tr>
      <td><strong>Context knows algorithm?</strong></td>
      <td>No — only the interface</td>
      <td>Yes — defines the skeleton</td>
      <td>No — just calls execute()</td>
    </tr>
    <tr>
      <td><strong>Swappable at runtime?</strong></td>
      <td>Yes</td>
      <td>No (fixed at class definition)</td>
      <td>Yes — commands are objects</td>
    </tr>
    <tr>
      <td><strong>ML example</strong></td>
      <td>Swap loss function mid-training</td>
      <td>train() calls abstract train_step()</td>
      <td>Queued training job with undo</td>
    </tr>
    <tr>
      <td><strong>Granularity</strong></td>
      <td>One whole algorithm</td>
      <td>One step in a fixed skeleton</td>
      <td>One operation with state</td>
    </tr>
  </tbody>
</table>

---

## Complete Integration: Pluggable Training Pipeline

```python
"""
Putting it all together: a fully configurable training pipeline
where every axis of variation is a Strategy.
"""
from dataclasses import dataclass, field
from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class PipelineConfig:
    loss: str = "focal"
    loss_kwargs: dict = field(default_factory=lambda: {"gamma": 2.0})
    optimizer: str = "adamw"
    optimizer_kwargs: dict = field(default_factory=lambda: {"weight_decay": 0.01})
    augmentation: str = "randaugment"
    sampler: str = "class_balanced"
    lr: float = 3e-4
    epochs: int = 10


class FlexibleTrainer:
    """
    Context that wires together all strategy dimensions.
    Adding a new loss/optimizer/augmentation requires zero changes here.
    """
    def __init__(
        self,
        model: nn.Module,
        loss_fn: LossStrategy,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        eval_loader: DataLoader | None = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader

    def fit(self, epochs: int) -> list[float]:
        losses = []
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                logits = self.model(batch["input_ids"].to("cpu"))
                loss = self.loss_fn(logits, batch["labels"].to("cpu"))
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            avg = epoch_loss / len(self.train_loader)
            losses.append(avg)
            print(f"Epoch {epoch+1}/{epochs} loss={avg:.4f}")
        return losses


def build_from_config(model: nn.Module, cfg: PipelineConfig) -> FlexibleTrainer:
    loss_fn   = StrategyFactory.create_loss(cfg.loss, **cfg.loss_kwargs)
    opt_strat = StrategyFactory.create_optimizer(cfg.optimizer, **cfg.optimizer_kwargs)
    optimizer = opt_strat(model.parameters(), lr=cfg.lr)
    # ... build DataLoaders with sampler strategy ...
    return FlexibleTrainer(model, loss_fn, optimizer, train_loader=...)


# Changing loss, optimizer, augmentation requires changing only the config dict —
# zero changes to FlexibleTrainer, zero changes to any strategy class.
configs = [
    PipelineConfig(loss="cross_entropy", optimizer="sgd"),
    PipelineConfig(loss="focal",         optimizer="adamw", loss_kwargs={"gamma": 1.5}),
    PipelineConfig(loss="label_smoothing", optimizer="lion", loss_kwargs={"smoothing": 0.05}),
]
```

---

*Last updated: May 2026*
