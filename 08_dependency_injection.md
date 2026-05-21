---
title: "Chapter 8 — Dependency Injection"
---
[← Back to Table of Contents](./README.md)

# Chapter 8 — Dependency Injection

> *"Don't call us, we'll call you — the Hollywood Principle applied to object wiring."*

---

## 8.1 What Is Dependency Injection?

**Dependency Injection (DI)** is the practice of supplying an object's collaborators from the outside, rather than letting the object create them itself. It is a specific application of the **Inversion of Control (IoC)** principle.

### A Simple Analogy

Imagine ordering coffee in two different cafés:

- **Without DI**: You walk into the back room, grind the beans yourself, and operate the espresso machine directly. You are tightly coupled to that specific machine.
- **With DI**: You sit down, a barista brings you coffee. You don't know (or care) whether it came from a La Marzocco or a Jura. You depend on an *interface* ("bring me a flat white"), not an *implementation*.

In code terms:
- **Without DI**: `self.optimizer = torch.optim.Adam(params)`
- **With DI**: `self.optimizer = optimizer` (passed in from outside)

<div class="diagram">
  <div class="diagram-title">Inversion of Control</div>
  <div class="flow flow-h">
    <div class="flow-node accent wide">❌ Tightly Coupled<br><small>Trainer creates Adam directly</small></div>
    <div class="flow-arrow accent">→</div>
    <div class="flow-node green wide">✅ DI Applied<br><small>Trainer receives optimizer as arg</small></div>
  </div>
</div>

---

## 8.2 The Problem: Tightly Coupled ML Components

```python
# ❌ Tightly coupled — hard to test, hard to swap components
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class BadTrainer:
    """
    Anti-pattern: Trainer hardcodes every dependency.
    - Can't swap optimizer without editing this class
    - Can't test without a real GPU and real data
    - Can't reuse with a different dataset
    - Can't swap the logger
    """

    def __init__(self, model: nn.Module):
        self.model = model.cuda()               # hardcoded device!

        # Hardcoded dataset
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST("./data", train=True, transform=transform, download=True)
        self.loader = DataLoader(dataset, batch_size=64, num_workers=4)

        # Hardcoded optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # Hardcoded loss
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for x, y in self.loader:
            x, y = x.cuda(), y.cuda()          # hardcoded device
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(x), y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            print(f"loss: {loss.item():.4f}")  # hardcoded logger (print!)
        return total_loss / len(self.loader)
```

**Problems with this approach:**

| Issue | Impact |
|---|---|
| Hardcoded `Adam` | Can't benchmark against `SGD` without modifying the class |
| Hardcoded `MNIST` | Can't reuse trainer for CIFAR-10 |
| Hardcoded `cuda()` | Tests fail on CPU-only CI machines |
| Hardcoded `print` | Can't redirect to W&B, TensorBoard, or suppress in tests |
| Hardcoded `CrossEntropyLoss` | Can't try focal loss |

---

## 8.3 Constructor Injection

**Constructor injection** passes all dependencies through `__init__`. This is the most common and recommended form.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """
    ✅ Constructor injection — all dependencies passed at init time.
    The class expresses its needs clearly in its signature.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(x), y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)
```

```python
# Wiring at application entry point
model     = SmallCNN(num_classes=10)
loader    = DataLoader(mnist_train, batch_size=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

trainer = Trainer(model, loader, optimizer, criterion, device="cuda")
```

---

## 8.4 Method Injection

**Method injection** passes a dependency as a parameter to a *specific method* rather than at construction time. Use this when the dependency varies per-call.

```python
import torch
from typing import Callable


class EvaluatorWithMethodInjection:
    """
    The metric function is injected at evaluation time.
    Different callers can pass different metrics.
    """

    def __init__(self, model: nn.Module, loader: DataLoader, device: str = "cpu"):
        self.model = model.to(device)
        self.loader = loader
        self.device = device

    def evaluate(
        self,
        metric_fn: Callable[[torch.Tensor, torch.Tensor], float],
    ) -> float:
        """metric_fn is injected — caller decides how to score."""
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in self.loader:
                x = x.to(self.device)
                preds = self.model(x).argmax(dim=-1).cpu()
                all_preds.append(preds)
                all_labels.append(y)
        preds  = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        return metric_fn(preds, labels)


# Inject different metrics at call time
def accuracy(preds, labels):
    return (preds == labels).float().mean().item()

def top_k_accuracy(k: int):
    def _metric(preds, labels):
        return (preds[:k] == labels[:k]).float().mean().item()
    return _metric

score = evaluator.evaluate(accuracy)
score5 = evaluator.evaluate(top_k_accuracy(5))
```

---

## 8.5 Property Injection

**Property injection** sets dependencies after construction using properties or setters. Useful when a dependency is optional or only known after `__init__`.

```python
import torch.nn as nn
from typing import Optional


class ModelWrapper:
    """
    The logger and profiler are optional dependencies,
    set after construction via properties.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._logger: Optional[object] = None
        self._profiler: Optional[object] = None

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, value):
        if value is not None and not hasattr(value, "log"):
            raise TypeError(f"Logger must have a .log() method, got {type(value)}")
        self._logger = value

    @property
    def profiler(self):
        return self._profiler

    @profiler.setter
    def profiler(self, value):
        self._profiler = value

    def forward(self, x):
        if self._profiler:
            self._profiler.start()
        out = self.model(x)
        if self._profiler:
            self._profiler.stop()
        if self._logger:
            self._logger.log({"forward_shape": list(out.shape)})
        return out
```

---

## 8.6 Interface-Based DI with ABCs and Protocols

Defining **contracts** (interfaces) allows the injected dependencies to be swapped for mocks, fakes, or alternative implementations:

```python
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
import torch
from torch.utils.data import DataLoader


# ── Protocols (structural subtyping — no inheritance required) ───────────────

@runtime_checkable
class DataLoaderProtocol(Protocol):
    def __iter__(self): ...
    def __len__(self) -> int: ...


@runtime_checkable
class OptimizerProtocol(Protocol):
    def zero_grad(self) -> None: ...
    def step(self) -> None: ...
    param_groups: list[dict]


@runtime_checkable
class SchedulerProtocol(Protocol):
    def step(self, metrics: float | None = None) -> None: ...
    def get_last_lr(self) -> list[float]: ...


@runtime_checkable
class LoggerProtocol(Protocol):
    def log(self, metrics: dict[str, float], step: int) -> None: ...
    def log_hyperparams(self, params: dict) -> None: ...
    def finalize(self) -> None: ...


# ── Abstract base classes (for richer default behaviour) ─────────────────────

class BaseLogger(ABC):
    @abstractmethod
    def log(self, metrics: dict[str, float], step: int) -> None: ...

    @abstractmethod
    def log_hyperparams(self, params: dict) -> None: ...

    def finalize(self) -> None:
        pass   # optional hook


class BaseCheckpointer(ABC):
    @abstractmethod
    def save(self, state: dict, is_best: bool = False) -> None: ...

    @abstractmethod
    def load(self, path: str) -> dict: ...
```

---

## 8.7 Full ML Trainer with DI — Complete Working Example

```python
from __future__ import annotations
import time
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Iterator
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# Protocols & Interfaces
# ══════════════════════════════════════════════════════════════════════════════

class IDataLoader(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator: ...
    @abstractmethod
    def __len__(self) -> int: ...


class IOptimizer(ABC):
    @abstractmethod
    def zero_grad(self) -> None: ...
    @abstractmethod
    def step(self) -> None: ...
    @property
    @abstractmethod
    def param_groups(self) -> list[dict]: ...


class IScheduler(ABC):
    @abstractmethod
    def step(self, metrics: float | None = None) -> None: ...
    @abstractmethod
    def get_last_lr(self) -> list[float]: ...


class ILogger(ABC):
    @abstractmethod
    def log(self, metrics: dict[str, float], step: int) -> None: ...
    def log_hyperparams(self, params: dict) -> None: pass
    def finalize(self) -> None: pass


class ICheckpointer(ABC):
    @abstractmethod
    def save(self, state: dict, filename: str) -> None: ...
    @abstractmethod
    def load(self, filename: str) -> dict: ...


# ══════════════════════════════════════════════════════════════════════════════
# Concrete Implementations
# ══════════════════════════════════════════════════════════════════════════════

class TorchDataLoader(IDataLoader):
    """Wraps a standard PyTorch DataLoader."""
    def __init__(self, loader: torch.utils.data.DataLoader):
        self._loader = loader
    def __iter__(self): return iter(self._loader)
    def __len__(self): return len(self._loader)


class TorchOptimizer(IOptimizer):
    """Wraps a standard PyTorch Optimizer."""
    def __init__(self, optimizer: torch.optim.Optimizer):
        self._opt = optimizer
    def zero_grad(self): self._opt.zero_grad()
    def step(self):      self._opt.step()
    @property
    def param_groups(self): return self._opt.param_groups


class CosineScheduler(IScheduler):
    """Cosine annealing LR scheduler."""
    def __init__(self, optimizer: torch.optim.Optimizer, T_max: int):
        self._sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)
    def step(self, metrics=None): self._sched.step()
    def get_last_lr(self): return self._sched.get_last_lr()


class ReduceOnPlateauScheduler(IScheduler):
    """ReduceLROnPlateau scheduler."""
    def __init__(self, optimizer: torch.optim.Optimizer, patience: int = 5):
        self._sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=patience
        )
    def step(self, metrics=None): self._sched.step(metrics)
    def get_last_lr(self):
        return [pg["lr"] for pg in self._sched.optimizer.param_groups]


class ConsoleLogger(ILogger):
    """Logs metrics to stdout."""
    def log(self, metrics: dict[str, float], step: int) -> None:
        parts = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"[step {step:05d}] {parts}")
    def log_hyperparams(self, params: dict) -> None:
        print(f"[hparams] {params}")


class FileLogger(ILogger):
    """Appends metrics to a CSV file."""
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._headers_written = False

    def log(self, metrics: dict[str, float], step: int) -> None:
        row = {"step": step, **metrics}
        if not self._headers_written:
            self._path.write_text(",".join(row.keys()) + "\n")
            self._headers_written = True
        with self._path.open("a") as f:
            f.write(",".join(str(v) for v in row.values()) + "\n")

    def log_hyperparams(self, params: dict) -> None:
        with self._path.with_suffix(".hparams.json").open("w") as f:
            import json; json.dump(params, f, indent=2)

    def finalize(self) -> None:
        print(f"[FileLogger] Saved metrics to {self._path}")


class LocalCheckpointer(ICheckpointer):
    """Saves/loads PyTorch state dicts to disk."""
    def __init__(self, save_dir: str | Path = "./checkpoints"):
        self._dir = Path(save_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: dict, filename: str) -> None:
        torch.save(state, self._dir / filename)

    def load(self, filename: str) -> dict:
        return torch.load(self._dir / filename, map_location="cpu")


# ══════════════════════════════════════════════════════════════════════════════
# The Trainer — depends only on interfaces
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainerConfig:
    max_epochs: int = 10
    log_every_n_steps: int = 50
    checkpoint_every_n_epochs: int = 1
    device: str = "cpu"
    grad_clip_norm: float = 1.0


class Trainer:
    """
    A fully DI-wired trainer.
    Every dependency is an interface — concrete classes are injected externally.
    This makes the Trainer testable with mocks and swappable for any framework.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: IDataLoader,
        val_loader: IDataLoader,
        optimizer: IOptimizer,
        scheduler: IScheduler,
        logger: ILogger,
        checkpointer: ICheckpointer,
        config: TrainerConfig | None = None,
    ):
        self.cfg = config or TrainerConfig()
        self.model = model.to(self.cfg.device)
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.checkpointer = checkpointer
        self._global_step = 0

    def fit(self) -> None:
        self.logger.log_hyperparams(vars(self.cfg))
        best_val_loss = float("inf")

        for epoch in range(1, self.cfg.max_epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch()
            val_loss   = self._val_epoch()
            elapsed    = time.time() - t0

            self.scheduler.step(metrics=val_loss)
            lr = self.scheduler.get_last_lr()[0]

            self.logger.log(
                {"train_loss": train_loss, "val_loss": val_loss,
                 "lr": lr, "epoch_time_s": elapsed},
                step=epoch,
            )

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            if epoch % self.cfg.checkpoint_every_n_epochs == 0:
                self.checkpointer.save(
                    {"epoch": epoch, "model": self.model.state_dict(),
                     "val_loss": val_loss},
                    filename=f"epoch_{epoch:03d}{'_best' if is_best else ''}.pt",
                )

        self.logger.finalize()

    def _train_epoch(self) -> float:
        self.model.train()
        total, count = 0.0, 0
        for x, y in self.train_loader:
            x = x.to(self.cfg.device) if hasattr(x, "to") else x
            y = y.to(self.cfg.device) if hasattr(y, "to") else y
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip_norm
            )
            self.optimizer.step()
            total += loss.item()
            count += 1
            self._global_step += 1
            if self._global_step % self.cfg.log_every_n_steps == 0:
                self.logger.log({"step_loss": loss.item()}, step=self._global_step)
        return total / max(count, 1)

    def _val_epoch(self) -> float:
        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.cfg.device) if hasattr(x, "to") else x
                y = y.to(self.cfg.device) if hasattr(y, "to") else y
                loss = self.criterion(self.model(x), y)
                total += loss.item()
                count += 1
        return total / max(count, 1)
```

```python
# ── Wiring at application entry point ────────────────────────────────────────
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Create dummy data
X_train = torch.randn(1000, 784)
y_train = torch.randint(0, 10, (1000,))
X_val   = torch.randn(200, 784)
y_val   = torch.randint(0, 10, (200,))

train_ds = TensorDataset(X_train, y_train)
val_ds   = TensorDataset(X_val, y_val)

model     = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
criterion = nn.CrossEntropyLoss()
torch_opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

trainer = Trainer(
    model        = model,
    criterion    = criterion,
    train_loader = TorchDataLoader(DataLoader(train_ds, batch_size=64)),
    val_loader   = TorchDataLoader(DataLoader(val_ds, batch_size=128)),
    optimizer    = TorchOptimizer(torch_opt),
    scheduler    = ReduceOnPlateauScheduler(torch_opt, patience=3),
    logger       = ConsoleLogger(),
    checkpointer = LocalCheckpointer("./checkpoints"),
    config       = TrainerConfig(max_epochs=5, device="cpu"),
)

trainer.fit()
```

---

## 8.8 DI Containers in Python — `dependency-injector`

For large applications with many dependencies, a DI container manages the wiring automatically:

```python
# pip install dependency-injector
from dependency_injector import containers, providers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ApplicationContainer(containers.DeclarativeContainer):
    """
    Declares the entire object graph.
    Changing one line here rewires the whole application.
    """
    config = providers.Configuration()

    # Model
    model = providers.Singleton(
        nn.Sequential,
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

    # Optimizer — depends on model
    optimizer = providers.Singleton(
        torch.optim.AdamW,
        params=model.provided.parameters.call(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )

    # Logger
    logger = providers.Singleton(ConsoleLogger)

    # Checkpointer
    checkpointer = providers.Singleton(LocalCheckpointer, save_dir="./ckpts")

    # Trainer — depends on all above
    trainer = providers.Factory(
        Trainer,
        model=model,
        criterion=providers.Singleton(nn.CrossEntropyLoss),
        optimizer=providers.Object(None),   # placeholder; set below
        logger=logger,
        checkpointer=checkpointer,
    )


container = ApplicationContainer()
container.config.from_dict({
    "optimizer": {"lr": 1e-3, "weight_decay": 1e-2}
})
```

---

## 8.9 Testing with DI — Injecting Mocks and Fakes

DI makes unit testing vastly easier. Inject lightweight fakes instead of real dependencies:

```python
import pytest
from collections import defaultdict


# ── Fake implementations for testing ─────────────────────────────────────────

class FakeDataLoader(IDataLoader):
    """Returns a fixed batch — no real data needed."""
    def __init__(self, num_batches: int = 3, batch_size: int = 4, num_classes: int = 10):
        self._batches = [
            (torch.randn(batch_size, 784), torch.randint(0, num_classes, (batch_size,)))
            for _ in range(num_batches)
        ]
    def __iter__(self): return iter(self._batches)
    def __len__(self): return len(self._batches)


class FakeOptimizer(IOptimizer):
    """Records calls — no actual gradient updates."""
    def __init__(self):
        self.zero_grad_calls = 0
        self.step_calls = 0
        self._param_groups = [{"lr": 1e-3}]

    def zero_grad(self): self.zero_grad_calls += 1
    def step(self):      self.step_calls += 1

    @property
    def param_groups(self): return self._param_groups


class FakeScheduler(IScheduler):
    def step(self, metrics=None): pass
    def get_last_lr(self): return [1e-3]


class FakeLogger(ILogger):
    """Captures all logged metrics — assert on them in tests."""
    def __init__(self):
        self.logs: list[tuple[dict, int]] = []
        self.hparams: dict = {}

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.logs.append((metrics, step))

    def log_hyperparams(self, params: dict) -> None:
        self.hparams.update(params)


class FakeCheckpointer(ICheckpointer):
    def __init__(self): self.saved: dict[str, dict] = {}
    def save(self, state: dict, filename: str) -> None: self.saved[filename] = state
    def load(self, filename: str) -> dict: return self.saved[filename]


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_trainer_calls_optimizer():
    model     = nn.Linear(784, 10)
    fake_opt  = FakeOptimizer()
    fake_log  = FakeLogger()
    fake_ckpt = FakeCheckpointer()

    trainer = Trainer(
        model        = model,
        criterion    = nn.CrossEntropyLoss(),
        train_loader = FakeDataLoader(num_batches=5),
        val_loader   = FakeDataLoader(num_batches=2),
        optimizer    = fake_opt,
        scheduler    = FakeScheduler(),
        logger       = fake_log,
        checkpointer = fake_ckpt,
        config       = TrainerConfig(max_epochs=2),
    )
    trainer.fit()

    # 5 batches × 2 epochs = 10 optimizer steps
    assert fake_opt.step_calls == 10
    assert fake_opt.zero_grad_calls == 10
    assert len(fake_log.logs) >= 2          # at least one log per epoch
    assert len(fake_ckpt.saved) == 2        # one checkpoint per epoch


def test_trainer_logs_hyperparams():
    fake_log = FakeLogger()
    trainer = Trainer(
        model        = nn.Linear(784, 10),
        criterion    = nn.CrossEntropyLoss(),
        train_loader = FakeDataLoader(num_batches=2),
        val_loader   = FakeDataLoader(num_batches=1),
        optimizer    = FakeOptimizer(),
        scheduler    = FakeScheduler(),
        logger       = fake_log,
        checkpointer = FakeCheckpointer(),
        config       = TrainerConfig(max_epochs=1),
    )
    trainer.fit()
    assert "max_epochs" in fake_log.hparams
```

---

## 8.10 Comparison: DI vs Service Locator vs Registry

<div class="compare-table">

| Aspect | Dependency Injection | Service Locator | Registry |
|---|---|---|---|
| **Dependencies visible in** | Constructor / method signature | Hidden (resolved inside class) | Config dict / string key |
| **Testability** | ⭐⭐⭐ Excellent | ⭐⭐ Good | ⭐⭐ Good |
| **Coupling** | Low — depends on interface | Medium — depends on locator | Medium — depends on registry |
| **Boilerplate** | More wiring code | Less | Less |
| **Discovery** | Explicit | Implicit | Semi-explicit (key lookup) |
| **Best for** | Complex apps, unit testing | Legacy codebases, gradual migration | Plugin systems, factory lookup |
| **ML example** | Trainer receives Optimizer | Trainer calls `locator.get(IOptimizer)` | `optimizer_registry.build("adamw")` |

</div>

---

## 8.11 Flow Diagram

<div class="diagram">
  <div class="diagram-title">DI: Interface → Implementations → Injected into Trainer</div>
  <div class="flow">
    <div class="flow-node accent wide">IOptimizer<br><small>&lt;&lt;interface&gt;&gt;</small></div>
    <div class="flow-arrow accent">↓ implements</div>
    <div class="flow-node green wide">TorchOptimizer / FakeOptimizer / SparseOptimizer<br><small>concrete implementations</small></div>
    <div class="flow-arrow green">↓ injected via constructor</div>
    <div class="flow-node blue wide">Trainer.__init__(optimizer: IOptimizer)<br><small>depends on interface, not concrete class</small></div>
    <div class="flow-arrow blue">↓ uses</div>
    <div class="flow-node purple wide">optimizer.zero_grad() / optimizer.step()<br><small>polymorphic dispatch</small></div>
    <div class="flow-arrow purple">↓ swap at will</div>
    <div class="flow-node teal wide">Production: AdamW &nbsp;|&nbsp; Test: FakeOptimizer<br><small>same Trainer code, different behaviour</small></div>
  </div>
</div>

---

## 8.12 DI Best Practices & Anti-Patterns

<div class="callout tip">
<strong>💡 DI Best Practices</strong>
<ul>
<li><strong>Depend on interfaces, not implementations</strong> — use ABCs or Protocols as type hints.</li>
<li><strong>Inject at the outermost boundary</strong> — wire dependencies in <code>main.py</code> or a container, not deep inside your business logic.</li>
<li><strong>Keep constructors simple</strong> — avoid logic in <code>__init__</code>; constructors should only assign, not compute.</li>
<li><strong>Prefer constructor injection</strong> over property injection — dependencies are explicit and required, not optional surprises.</li>
<li><strong>One responsibility per class</strong> — if a class needs 6+ injected dependencies, it's doing too much.</li>
</ul>
</div>

<div class="callout warn">
<strong>⚠ DI Anti-Patterns</strong>
<ul>
<li><strong>Injecting the container itself</strong> — passing the DI container into a class turns it into a Service Locator and hides dependencies.</li>
<li><strong>Constructor over-injection</strong> — 10+ constructor parameters is a sign the class violates SRP.</li>
<li><strong>Circular dependencies</strong> — A depends on B depends on A. Break with a provider, factory, or event-driven approach.</li>
<li><strong>Injecting primitives</strong> — inject <code>Config</code> objects, not raw <code>int</code> / <code>float</code> values. Primitives don't document their purpose.</li>
<li><strong>Mutable singleton state via DI</strong> — a <code>Singleton</code>-scoped mutable object shared across threads is a concurrency bug waiting to happen.</li>
</ul>
</div>

---

## Summary

<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">🏗️</div>
    <div class="card-title">Constructor Injection</div>
    <div class="card-desc">Pass all required dependencies in __init__. Most explicit, most testable. The default choice.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">📨</div>
    <div class="card-title">Method Injection</div>
    <div class="card-desc">Pass per-call dependencies as method arguments. Best for strategies that vary per invocation.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🔧</div>
    <div class="card-title">Property Injection</div>
    <div class="card-desc">Set optional dependencies after construction. Use sparingly — hidden dependencies are a code smell.</div>
  </div>
</div>

---

*Last updated: May 2026*
