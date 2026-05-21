---
title: "Chapter 6 — Singleton, Borg & Registry"
---
[← Back to Table of Contents](./README.md)

# Chapter 6 — Singleton, Borg & Registry

> *"There can be only one … unless you need shared state, in which case there can be many with one mind."*

---

## 6.1 The Singleton Pattern

The Singleton pattern ensures a class has **exactly one instance** and provides a global access point to it. It is one of the most discussed — and most controversial — patterns in software design.

### Classic Implementation: Module-Level Instance

The simplest and most Pythonic singleton is a **module-level instance**. Python's import system caches modules after the first import, so any object created at module scope is naturally a singleton.

```python
# config.py  — module-level singleton
class _Config:
    def __init__(self):
        self.learning_rate: float = 1e-3
        self.batch_size: int = 32
        self.device: str = "cuda"
        self.max_epochs: int = 100

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"Unknown config key: {k}")
            setattr(self, k, v)

    def __repr__(self):
        fields = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"Config({fields})"

# The single instance — created once when module is first imported
config = _Config()
```

```python
# Usage across different modules
from config import config

config.update(learning_rate=5e-4, batch_size=64)
print(config.learning_rate)   # 0.0005
print(config.batch_size)      # 64
```

### Classic Implementation: Metaclass Singleton

For explicit control over instantiation, a metaclass approach enforces the singleton at the class level:

```python
import threading

class SingletonMeta(type):
    """Thread-safe singleton metaclass."""
    _instances: dict = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # Double-checked locking for performance
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


class DatabaseConnection(metaclass=SingletonMeta):
    def __init__(self, url: str = "postgresql://localhost/ml_experiments"):
        self.url = url
        self._connection = None
        print(f"[DB] New connection to {url}")   # prints only once

    def connect(self):
        if self._connection is None:
            self._connection = f"<Connection to {self.url}>"
        return self._connection


# Verify singleton behaviour
db1 = DatabaseConnection()
db2 = DatabaseConnection()
assert db1 is db2          # True — same object
print(db1.connect())
```

### Classic Implementation: `__new__`-Based Singleton

```python
class ExperimentTracker:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def __init__(self, run_dir: str = "./runs"):
        if self._initialised:
            return                         # prevent re-initialisation
        self.run_dir = run_dir
        self.metrics: dict[str, list] = {}
        self._initialised = True

    def log(self, key: str, value: float):
        self.metrics.setdefault(key, []).append(value)
```

---

## 6.2 Python Module as a Singleton — The Idiomatic Way

Python modules are singletons by design. The interpreter caches modules in `sys.modules` and never executes module code twice.

```python
# metrics_store.py
from collections import defaultdict
from typing import Any

_store: dict[str, list[Any]] = defaultdict(list)

def record(metric: str, value: Any) -> None:
    _store[metric].append(value)

def get(metric: str) -> list[Any]:
    return list(_store[metric])

def reset() -> None:
    _store.clear()

def summary() -> dict[str, Any]:
    return {k: {"count": len(v), "last": v[-1]} for k, v in _store.items()}
```

```python
# train.py
import metrics_store

metrics_store.record("loss", 0.432)
metrics_store.record("accuracy", 0.87)

# eval.py — same process, same data
import metrics_store
print(metrics_store.summary())
# {'loss': {'count': 1, 'last': 0.432}, 'accuracy': {'count': 1, 'last': 0.87}}
```

This is the **preferred pattern** in Python: no metaclass magic, fully thread-safe under the GIL, and trivially testable by calling `reset()`.

---

## 6.3 The Borg (Monostate) Pattern

The **Borg** pattern, coined by Alex Martelli, inverts the Singleton approach. Instead of restricting the number of *instances*, it shares *state* across all instances. Each call to the constructor returns a **different object** but they all share the same `__dict__`.

```python
class Borg:
    """All instances share the same state dictionary."""
    _shared_state: dict = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class HyperParameters(Borg):
    def __init__(self):
        super().__init__()
        # Only initialise if this is the first-ever instance
        if not self._shared_state:
            self.lr: float = 1e-3
            self.momentum: float = 0.9
            self.weight_decay: float = 1e-4
            self.warmup_steps: int = 1000

    def __repr__(self):
        return f"HyperParameters({self._shared_state})"


hp1 = HyperParameters()
hp2 = HyperParameters()

hp1.lr = 5e-4            # Mutates shared state
assert hp2.lr == 5e-4    # hp2 sees the change
assert hp1 is not hp2    # But they are different objects
print(hp1)
print(hp2)
```

### When to Prefer Borg over Singleton

- When subclassing matters — each subclass can have its **own** shared state:

```python
class BaseConfig(Borg):
    _shared_state: dict = {}   # Base class state

class TrainingConfig(BaseConfig):
    _shared_state: dict = {}   # Separate namespace from BaseConfig!

class InferenceConfig(BaseConfig):
    _shared_state: dict = {}   # Another separate namespace

t = TrainingConfig()
t.batch_size = 64

i = InferenceConfig()
i.batch_size = 1          # Does NOT affect TrainingConfig

print(t.batch_size)       # 64
print(i.batch_size)       # 1
```

---

## 6.4 The Registry Pattern

A **Registry** is a global catalog that maps string names to classes (or factories). It solves the problem of *configuring which implementation to use at runtime* without importing everything upfront.

<div class="diagram">
  <div class="diagram-title">Registry Pattern — Core Concept</div>
  <div class="flow flow-h">
    <div class="flow-node accent wide">Registry<br><small>name → class</small></div>
    <div class="flow-arrow accent">→</div>
    <div class="flow-node green">register()<br><small>add entry</small></div>
    <div class="flow-arrow accent">→</div>
    <div class="flow-node blue">lookup()<br><small>retrieve class</small></div>
    <div class="flow-arrow accent">→</div>
    <div class="flow-node purple">instantiate()<br><small>create object</small></div>
  </div>
</div>

### Base Registry Implementation

```python
from __future__ import annotations
import threading
from typing import TypeVar, Type, Callable, Any

T = TypeVar("T")


class Registry(Generic[T]):
    """
    A thread-safe registry mapping string keys to classes or factories.

    Usage:
        model_registry = Registry("models")

        @model_registry.register("resnet50")
        class ResNet50(nn.Module): ...

        cls = model_registry.lookup("resnet50")
        model = cls(num_classes=10)
    """

    def __init__(self, name: str):
        self._name = name
        self._catalog: dict[str, type[T]] = {}
        self._lock = threading.Lock()

    def register(self, key: str) -> Callable[[type[T]], type[T]]:
        """Decorator that registers a class under the given key."""
        def decorator(cls: type[T]) -> type[T]:
            with self._lock:
                if key in self._catalog:
                    raise KeyError(
                        f"[{self._name}] Key '{key}' already registered "
                        f"by {self._catalog[key].__name__}"
                    )
                self._catalog[key] = cls
            return cls
        return decorator

    def register_alias(self, key: str, alias: str) -> None:
        """Register an additional name pointing to the same class."""
        with self._lock:
            if key not in self._catalog:
                raise KeyError(f"[{self._name}] Key '{key}' not found")
            self._catalog[alias] = self._catalog[key]

    def lookup(self, key: str) -> type[T]:
        with self._lock:
            if key not in self._catalog:
                available = ", ".join(sorted(self._catalog))
                raise KeyError(
                    f"[{self._name}] Unknown key '{key}'. "
                    f"Available: [{available}]"
                )
            return self._catalog[key]

    def build(self, key: str, *args: Any, **kwargs: Any) -> T:
        """Shorthand: lookup + instantiate in one call."""
        return self.lookup(key)(*args, **kwargs)

    def list(self) -> list[str]:
        with self._lock:
            return sorted(self._catalog.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._catalog

    def __len__(self) -> int:
        return len(self._catalog)

    def __repr__(self) -> str:
        return f"Registry('{self._name}', entries={self.list()})"
```

### Model Registry Example

```python
from __future__ import annotations
import torch.nn as nn
from registry import Registry

model_registry: Registry[nn.Module] = Registry("models")


@model_registry.register("mlp")
class MLP(nn.Module):
    def __init__(self, in_features: int = 784, hidden: int = 256, out: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out),
        )

    def forward(self, x):
        return self.net(x.flatten(1))


@model_registry.register("cnn")
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


# Runtime selection from config / CLI
def build_model(config: dict) -> nn.Module:
    arch = config["architecture"]       # e.g. "mlp" or "cnn"
    params = config.get("params", {})
    return model_registry.build(arch, **params)

print(model_registry.list())
# ['cnn', 'mlp']
```

### Loss Registry Example

```python
import torch
import torch.nn as nn

loss_registry: Registry[nn.Module] = Registry("losses")


@loss_registry.register("cross_entropy")
class CrossEntropyLoss(nn.CrossEntropyLoss):
    pass


@loss_registry.register("focal")
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()


@loss_registry.register("label_smoothing")
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1, num_classes: int = 10):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = nn.functional.log_softmax(logits, dim=-1)
        smooth_val = self.smoothing / (self.num_classes - 1)
        with torch.no_grad():
            one_hot = torch.zeros_like(log_prob).scatter_(1, targets.unsqueeze(1), 1)
            soft_targets = one_hot * (1 - self.smoothing) + smooth_val
        return -(soft_targets * log_prob).sum(dim=-1).mean()
```

### Optimizer Registry Example

```python
import torch.optim as optim

optimizer_registry: Registry[optim.Optimizer] = Registry("optimizers")

optimizer_registry._catalog = {
    "adam":   optim.Adam,
    "adamw":  optim.AdamW,
    "sgd":    optim.SGD,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
}

def build_optimizer(params, config: dict) -> optim.Optimizer:
    name = config.pop("name")
    return optimizer_registry.build(name, params, **config)

# Config-driven construction
opt = build_optimizer(
    model.parameters(),
    {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-2}
)
```

---

## 6.5 Thread-Safety Considerations

In multi-threaded training (e.g., multi-worker data loading, async logging), the registry must guard against concurrent writes during registration.

```python
import threading
import time
from concurrent.futures import ThreadPoolExecutor

augmentation_registry: Registry = Registry("augmentations")

# Simulate concurrent plugin loading
def load_plugin(name: str, cls):
    time.sleep(0.001)                          # simulate I/O delay
    augmentation_registry._catalog[name] = cls  # UNSAFE without lock

# Safe version already shown above via self._lock in register()
# Here we demonstrate double-checked locking explicitly:

_aug_lock = threading.Lock()
_aug_store: dict[str, type] = {}


def safe_register(name: str, cls: type) -> None:
    if name not in _aug_store:               # fast path (no lock)
        with _aug_lock:
            if name not in _aug_store:       # re-check inside lock
                _aug_store[name] = cls


def safe_lookup(name: str) -> type:
    try:
        return _aug_store[name]
    except KeyError:
        raise KeyError(f"Augmentation '{name}' not registered") from None
```

### Read-Write Lock for High-Read Registries

When lookups vastly outnumber registrations (typical at inference time), a `RWLock` reduces contention:

```python
import threading


class RWLock:
    """Allows multiple concurrent readers, exclusive writers."""

    def __init__(self):
        self._readers = 0
        self._read_lock = threading.Lock()
        self._write_lock = threading.Lock()

    def acquire_read(self):
        with self._read_lock:
            self._readers += 1
            if self._readers == 1:
                self._write_lock.acquire()

    def release_read(self):
        with self._read_lock:
            self._readers -= 1
            if self._readers == 0:
                self._write_lock.release()

    def acquire_write(self):
        self._write_lock.acquire()

    def release_write(self):
        self._write_lock.release()
```

---

## 6.6 Plugin System Using Registry

A plugin system lets external packages register new implementations at import time, enabling fully dynamic extensibility.

```python
# plugin_base.py
from __future__ import annotations
import importlib
import pkgutil
from pathlib import Path
from registry import Registry

backbone_registry: Registry = Registry("backbones")
head_registry: Registry = Registry("heads")


def discover_plugins(package_name: str) -> None:
    """
    Walk all sub-modules of `package_name` and import them.
    Each module's top-level code runs, triggering @register decorators.
    """
    package = importlib.import_module(package_name)
    package_path = Path(package.__file__).parent

    for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
        full_name = f"{package_name}.{module_name}"
        importlib.import_module(full_name)
        print(f"  [plugins] Loaded {full_name}")
```

```python
# my_project/backbones/resnet.py  — a plugin
from plugin_base import backbone_registry
import torch.nn as nn


@backbone_registry.register("resnet18")
class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision.models as tv
        base = tv.resnet18(pretrained=pretrained)
        # Strip the final FC layer — output is (B, 512, 1, 1)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.out_channels = 512

    def forward(self, x):
        return self.features(x)
```

```python
# main.py
from plugin_base import discover_plugins, backbone_registry

discover_plugins("my_project.backbones")  # auto-imports all backbone files

backbone_cls = backbone_registry.lookup("resnet18")
backbone = backbone_cls(pretrained=False)
print(backbone_registry.list())           # ['resnet18', ...]
```

```python
# Entry-points based plugin discovery (for installed packages)
from importlib.metadata import entry_points

def load_entry_point_plugins(group: str = "myproject.backbones") -> None:
    eps = entry_points(group=group)
    for ep in eps:
        module = ep.load()        # triggers registration side-effects
        print(f"  [entry_point] Loaded {ep.name} from {ep.value}")
```

---

## 6.7 The Service Locator Pattern — Registry as DI-lite

A Service Locator is a Registry whose values are **instances** rather than classes. It provides a centralized place to resolve runtime services.

```python
from typing import TypeVar, Type

S = TypeVar("S")


class ServiceLocator:
    """Maps interface types to concrete singleton instances."""

    _services: dict[type, object] = {}

    @classmethod
    def register(cls, interface: type, instance: object) -> None:
        cls._services[interface] = instance

    @classmethod
    def resolve(cls, interface: Type[S]) -> S:
        if interface not in cls._services:
            raise KeyError(f"No service registered for {interface.__name__}")
        return cls._services[interface]  # type: ignore[return-value]


# --- Interfaces ---
class ILogger:
    def log(self, msg: str) -> None: ...

class ICheckpointer:
    def save(self, path: str, state: dict) -> None: ...
    def load(self, path: str) -> dict: ...

# --- Concrete implementations ---
class WandbLogger(ILogger):
    def log(self, msg: str) -> None:
        print(f"[wandb] {msg}")

class LocalCheckpointer(ICheckpointer):
    def save(self, path: str, state: dict) -> None:
        import torch; torch.save(state, path)
    def load(self, path: str) -> dict:
        import torch; return torch.load(path)

# --- Wire up at application entry point ---
ServiceLocator.register(ILogger, WandbLogger())
ServiceLocator.register(ICheckpointer, LocalCheckpointer())

# --- Consumer (no direct dependency on concrete class) ---
class Trainer:
    def __init__(self):
        self._logger = ServiceLocator.resolve(ILogger)
        self._ckpt   = ServiceLocator.resolve(ICheckpointer)

    def train_step(self, loss: float):
        self._logger.log(f"loss={loss:.4f}")
```

---

## 6.8 UML Diagrams

<div class="uml-row">
  <div class="uml-box">
    <div class="uml-title">Singleton</div>
    <div class="uml-section">Class</div>
    <div class="uml-item">- _instance: Singleton</div>
    <div class="uml-item">- _lock: Lock</div>
    <div class="uml-section">Methods</div>
    <div class="uml-item">+ __new__() → Singleton</div>
    <div class="uml-item">+ get_instance() → Singleton</div>
    <div class="uml-item">+ operation() → None</div>
  </div>

  <div class="uml-box">
    <div class="uml-title">Borg (Monostate)</div>
    <div class="uml-section">Class</div>
    <div class="uml-item">- _shared_state: dict</div>
    <div class="uml-item">+ __dict__ → _shared_state</div>
    <div class="uml-section">Effect</div>
    <div class="uml-item">instance1 ≠ instance2</div>
    <div class="uml-item">instance1.__dict__</div>
    <div class="uml-item">  IS instance2.__dict__</div>
  </div>

  <div class="uml-box">
    <div class="uml-title">Registry</div>
    <div class="uml-section">Attributes</div>
    <div class="uml-item">- _catalog: dict[str, type]</div>
    <div class="uml-item">- _lock: Lock</div>
    <div class="uml-section">Methods</div>
    <div class="uml-item">+ register(key) → decorator</div>
    <div class="uml-item">+ lookup(key) → type</div>
    <div class="uml-item">+ build(key, **kw) → obj</div>
    <div class="uml-item">+ list() → list[str]</div>
  </div>
</div>

---

## 6.9 Comparison Table

<div class="compare-table">

| Feature | Singleton | Borg | Registry |
|---|---|---|---|
| **Purpose** | One instance | Shared state | Name → class map |
| **Identity** | Single object | Multiple objects | N/A |
| **State** | Per-instance | All shared | Catalog |
| **Subclassable** | Fragile | Yes (separate state) | Yes |
| **Thread-safe (built-in)** | Needs care | Needs care | With Lock |
| **Testable** | Hard | Medium | Easy |
| **Typical ML use** | Config object | Hyperparameters | Model/loss/opt factory |
| **Python idiom** | Module-level var | Borg class | Module-level dict |

</div>

---

## 6.10 Singleton Anti-Patterns in ML

<div class="callout warn">
<strong>⚠ Global Mutable State</strong><br>
Singletons that hold mutable training state create <strong>hidden dependencies</strong> — a function's behaviour changes based on external state that is invisible in its signature. This makes code non-deterministic, harder to test, and nearly impossible to parallelise safely.

<strong>Symptoms:</strong>
<ul>
<li>Tests pass in isolation but fail when run together</li>
<li>Training behaviour differs between runs despite same config</li>
<li>Impossible to run two experiments in the same process</li>
</ul>

<strong>Remedy:</strong> Pass config/state explicitly as arguments; use the Registry to select implementations but inject instances via constructor.
</div>

```python
# ❌ Anti-pattern: global mutable state hidden inside a class
_global_step = 0

class Trainer:
    def step(self, loss):
        global _global_step
        _global_step += 1          # hidden mutation — test pollution!
        wandb.log({"step": _global_step, "loss": loss})

# ✅ Idiomatic: explicit state, no globals
class Trainer:
    def __init__(self):
        self._step = 0

    def step(self, loss, logger):
        self._step += 1
        logger.log({"step": self._step, "loss": loss})
```

---

## 6.11 Flow Diagram: Plugin Lifecycle

<div class="diagram">
  <div class="diagram-title">Plugin Discovery → Registration → Lookup → Instantiation</div>
  <div class="flow">
    <div class="flow-node accent wide">Plugin Discovery<br><small>pkgutil / entry_points scan</small></div>
    <div class="flow-arrow accent">↓</div>
    <div class="flow-node green wide">Module Import<br><small>importlib.import_module()</small></div>
    <div class="flow-arrow green">↓</div>
    <div class="flow-node blue wide">@register Decorator Runs<br><small>registry._catalog[key] = cls</small></div>
    <div class="flow-arrow accent">↓</div>
    <div class="flow-node purple wide">Lookup at Runtime<br><small>registry.lookup("resnet18")</small></div>
    <div class="flow-arrow purple">↓</div>
    <div class="flow-node teal wide">Instantiation<br><small>cls(**config_params)</small></div>
    <div class="flow-arrow accent">↓</div>
    <div class="flow-node orange wide">Running Model<br><small>model.forward(x)</small></div>
  </div>
</div>

---

## 6.12 Putting It All Together — Unified Config + Registry System

```python
# full_system.py
from __future__ import annotations
import threading
from typing import Any, TypeVar, Generic, Type, Callable

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str):
        self._name = name
        self._catalog: dict[str, type[T]] = {}
        self._lock = threading.Lock()

    def register(self, key: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            with self._lock:
                self._catalog[key] = cls
            return cls
        return decorator

    def build(self, key: str, **kwargs: Any) -> T:
        with self._lock:
            cls = self._catalog.get(key)
        if cls is None:
            raise KeyError(f"[{self._name}] '{key}' not found. Available: {self.list()}")
        return cls(**kwargs)

    def list(self) -> list[str]:
        return sorted(self._catalog)


# Registries
models     = Registry("models")
losses     = Registry("losses")
optimizers = Registry("optimizers")
schedulers = Registry("schedulers")


def build_experiment(cfg: dict) -> tuple:
    """Wire together an experiment from a plain dict config."""
    import torch, torch.nn as nn, torch.optim as optim

    model = models.build(cfg["model"]["name"], **cfg["model"].get("params", {}))
    loss_fn = losses.build(cfg["loss"]["name"], **cfg["loss"].get("params", {}))

    opt_cfg = cfg["optimizer"].copy()
    opt_name = opt_cfg.pop("name")
    optimizer_cls = {"adam": optim.Adam, "adamw": optim.AdamW, "sgd": optim.SGD}[opt_name]
    optimizer = optimizer_cls(model.parameters(), **opt_cfg)

    return model, loss_fn, optimizer


# Example config (could come from YAML/JSON/CLI)
experiment_config = {
    "model":     {"name": "mlp", "params": {"hidden": 512}},
    "loss":      {"name": "focal", "params": {"gamma": 2.0}},
    "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-2},
}
```

---

## Summary

<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">1️⃣</div>
    <div class="card-title">Singleton</div>
    <div class="card-desc">One instance via module, metaclass, or __new__. Best as a module-level variable in Python.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🧬</div>
    <div class="card-title">Borg</div>
    <div class="card-desc">Multiple instances, one shared state dict. Subclass-friendly, ideal for hierarchical configs.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">📚</div>
    <div class="card-title">Registry</div>
    <div class="card-desc">String → class catalog. The cornerstone of config-driven ML systems and plugin architectures.</div>
  </div>
</div>

| Pattern | Identity | State | Best for |
|---|---|---|---|
| Module singleton | Shared | Shared | App-wide constants, simple config |
| Borg | Separate | Shared | Hierarchical config with subclasses |
| Registry | N/A | Catalog | Plugin systems, config-driven factories |

<div class="callout tip">
<strong>💡 Recommendation for ML Projects</strong><br>
Use a <strong>module-level singleton</strong> for config, a <strong>Registry</strong> for model/loss/optimizer selection, and avoid metaclass Singletons unless you have a genuine need. Combine with a <strong>Service Locator</strong> for runtime service wiring in larger codebases.
</div>

---

*Last updated: May 2026*
