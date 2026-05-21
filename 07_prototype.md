---
title: "Chapter 7 — Prototype & Object Pool"
---
[← Back to Table of Contents](./README.md)

# Chapter 7 — Prototype & Object Pool

> *"Rather than constructing an object from scratch, clone a known-good one and adjust it — nature has been doing this for billions of years."*

---

## 7.1 The Prototype Pattern

The **Prototype** pattern specifies the kinds of objects to create using a *prototypical instance*, and creates new objects by copying it. Instead of calling a constructor (which may involve expensive initialisation), you start from a known-good template and diverge from there.

<div class="diagram">
  <div class="diagram-title">Why Clone Instead of Construct?</div>
  <div class="flow flow-h">
    <div class="flow-node accent wide">New Construction<br><small>allocate → init weights → load pretrained → validate</small></div>
    <div class="flow-arrow accent">vs</div>
    <div class="flow-node green wide">Clone Prototype<br><small>copy state dict → adjust head → done</small></div>
  </div>
</div>

In ML, the prototype pattern appears everywhere:
- Cloning a pretrained model for fine-tuning
- Creating N ensemble members from one trained base
- Copying a hyperparameter configuration before mutation
- Replicating a data-augmentation pipeline per worker

---

## 7.2 `copy.copy()` vs `copy.deepcopy()`

Python's `copy` module provides two levels of cloning:

```python
import copy
import torch
import torch.nn as nn

# ── Shallow copy ─────────────────────────────────────────────────────────────
# Creates a new container but SHARES the inner objects

layer = nn.Linear(128, 64)
shallow = copy.copy(layer)

# Weight tensors are shared — mutating one affects both!
shallow.weight.data.fill_(0.0)
print(layer.weight.data.norm())   # 0.0 — oops, same tensor

# ── Deep copy ────────────────────────────────────────────────────────────────
# Recursively duplicates every nested object

layer2 = nn.Linear(128, 64)
deep = copy.deepcopy(layer2)

deep.weight.data.fill_(0.0)
print(layer2.weight.data.norm())  # original unchanged ✓
```

### Shallow vs Deep: What Gets Shared?

| Object type | `copy.copy()` | `copy.deepcopy()` |
|---|---|---|
| Primitive (int, float, str) | Same object (immutable, safe) | Same object |
| List / dict | New container, same elements | New container + new elements |
| `nn.Module` | Same `weight` tensors | Independent `weight` tensors |
| `torch.Tensor` | Same storage | New storage |
| File handle / socket | Shared (risky) | Usually raises error |
| `numpy.ndarray` | Shared buffer | Independent buffer |

### ML Example: Config Mutation with Shallow Copy

```python
from dataclasses import dataclass, field
import copy

@dataclass
class AugConfig:
    p_flip: float = 0.5
    p_color: float = 0.3
    crop_size: tuple[int, int] = (224, 224)
    mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])

base_cfg = AugConfig()

# Shallow copy — lists are SHARED
shallow_cfg = copy.copy(base_cfg)
shallow_cfg.mean[0] = 0.0        # mutates base_cfg.mean too!
print(base_cfg.mean)             # [0.0, 0.456, 0.406]  ← corrupted

# Deep copy — lists are INDEPENDENT
deep_cfg = copy.deepcopy(base_cfg)
deep_cfg.mean[0] = 0.0
print(base_cfg.mean)             # [0.485, 0.456, 0.406]  ← safe
```

---

## 7.3 UML Diagram

<div class="uml-row">
  <div class="uml-box">
    <div class="uml-title">&lt;&lt;interface&gt;&gt; Prototype</div>
    <div class="uml-section">Methods</div>
    <div class="uml-item">+ clone() → Prototype</div>
  </div>

  <div class="uml-box">
    <div class="uml-title">ConcretePrototype</div>
    <div class="uml-section">Attributes</div>
    <div class="uml-item">- state: dict</div>
    <div class="uml-item">- weights: Tensor</div>
    <div class="uml-section">Methods</div>
    <div class="uml-item">+ clone() → ConcretePrototype</div>
    <div class="uml-item">+ __copy__() → self</div>
    <div class="uml-item">+ __deepcopy__(memo) → self</div>
  </div>

  <div class="uml-box">
    <div class="uml-title">Client</div>
    <div class="uml-section">Attributes</div>
    <div class="uml-item">- prototype: Prototype</div>
    <div class="uml-section">Methods</div>
    <div class="uml-item">+ operation() → Prototype</div>
    <div class="uml-item">  ↳ return prototype.clone()</div>
  </div>
</div>

---

## 7.4 The `__copy__` and `__deepcopy__` Protocols

You can customise how Python clones your objects by implementing these dunder methods:

```python
import copy
import torch
import torch.nn as nn
from typing import Any


class ModelPrototype(nn.Module):
    """
    A neural network that knows how to clone itself efficiently.
    Implements both shallow and deep copy protocols.
    """

    def __init__(self, arch: str, num_classes: int):
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.head = nn.Linear(128, num_classes)
        self._metadata: dict[str, Any] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x.flatten(1)))

    def __copy__(self) -> "ModelPrototype":
        """
        Shallow clone: new module with shared backbone weights.
        Useful when backbone is frozen — saves memory.
        """
        cls = type(self)
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)   # share references
        # Give it a fresh head (only part we'll train)
        new.head = nn.Linear(self.head.in_features, self.head.out_features)
        return new

    def __deepcopy__(self, memo: dict) -> "ModelPrototype":
        """
        Deep clone: fully independent copy, including all weights.
        memo dict prevents infinite recursion for circular references.
        """
        cls = type(self)
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, copy.deepcopy(v, memo))
        return new

    def clone_for_finetune(self, new_num_classes: int) -> "ModelPrototype":
        """Domain-specific clone: keep backbone, replace head."""
        cloned = copy.deepcopy(self)
        cloned.head = nn.Linear(128, new_num_classes)
        cloned.num_classes = new_num_classes
        return cloned
```

---

## 7.5 Weight Initialisation from Pretrained — Prototype-Based Model Cloning

The most common prototype pattern in ML is cloning a pretrained model for fine-tuning:

```python
import torch
import torch.nn as nn
import copy
from pathlib import Path


class PretrainedPrototype:
    """
    Manages a library of pretrained model prototypes.
    Clones on demand so the original weights are never mutated.
    """

    def __init__(self):
        self._prototypes: dict[str, nn.Module] = {}

    def register(self, name: str, model: nn.Module) -> None:
        """Store a frozen copy as the canonical prototype."""
        frozen = copy.deepcopy(model)
        for p in frozen.parameters():
            p.requires_grad_(False)
        self._prototypes[name] = frozen

    def clone(self, name: str, trainable: bool = True) -> nn.Module:
        """Return a fresh deep copy ready for fine-tuning."""
        if name not in self._prototypes:
            raise KeyError(f"No prototype named '{name}'")
        cloned = copy.deepcopy(self._prototypes[name])
        if trainable:
            for p in cloned.parameters():
                p.requires_grad_(True)
        return cloned

    def save(self, name: str, path: str | Path) -> None:
        torch.save(self._prototypes[name].state_dict(), path)

    def load(self, name: str, path: str | Path, model_cls: type, **kwargs) -> None:
        model = model_cls(**kwargs)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        self.register(name, model)


# ── Usage ────────────────────────────────────────────────────────────────────
prototype_store = PretrainedPrototype()

# Register a pretrained model (loaded from disk / HuggingFace / etc.)
import torchvision.models as tv
resnet = tv.resnet50(pretrained=True)
prototype_store.register("resnet50_imagenet", resnet)

# Clone for CIFAR-10 fine-tuning
ft_model = prototype_store.clone("resnet50_imagenet")
ft_model.fc = nn.Linear(2048, 10)           # replace head for 10 classes

# The prototype is untouched
proto_params = sum(p.sum().item() for p in prototype_store._prototypes["resnet50_imagenet"].parameters())
ft_params    = sum(p.sum().item() for p in ft_model.parameters())
# They may differ after the head replacement
```

### Using `state_dict()` as a Serialised Prototype

```python
def state_dict_clone(model: nn.Module) -> nn.Module:
    """Clone via state_dict — avoids Python object graph issues."""
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)

    clone = copy.deepcopy(model)          # copy architecture
    clone.load_state_dict(torch.load(buffer, map_location="cpu"))
    return clone


def disk_prototype(model: nn.Module, path: Path) -> None:
    """Serialise prototype to disk for cross-process cloning."""
    torch.save({
        "state_dict": model.state_dict(),
        "arch":       type(model).__name__,
    }, path)


def load_disk_prototype(path: Path, model: nn.Module) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    return copy.deepcopy(model)
```

---

## 7.6 Model Cloning for Ensemble Creation

Ensembles require N independent models starting from the same pretrained checkpoint but diverging due to different random seeds, data shuffles, or fine-tuning procedures:

```python
import torch
import torch.nn as nn
import copy
import random
from typing import Iterator


class EnsembleFactory:
    """
    Creates N diverse model clones from a single prototype.
    Each member diverges via different random seed and optional head noise.
    """

    def __init__(self, prototype: nn.Module):
        self._prototype = copy.deepcopy(prototype)
        for p in self._prototype.parameters():
            p.requires_grad_(False)   # freeze the canonical copy

    def create_members(
        self,
        n: int,
        seeds: list[int] | None = None,
        head_noise_std: float = 0.01,
    ) -> list[nn.Module]:
        seeds = seeds or list(range(n))
        members = []
        for seed in seeds[:n]:
            torch.manual_seed(seed)
            random.seed(seed)

            member = copy.deepcopy(self._prototype)
            for p in member.parameters():
                p.requires_grad_(True)

            # Slightly randomise the final layer to break symmetry
            if hasattr(member, "fc"):
                with torch.no_grad():
                    member.fc.weight += torch.randn_like(member.fc.weight) * head_noise_std
                    member.fc.bias   += torch.randn_like(member.fc.bias)   * head_noise_std

            members.append(member)
        return members

    def ensemble_predict(
        self,
        models: list[nn.Module],
        x: torch.Tensor,
        method: str = "mean",
    ) -> torch.Tensor:
        with torch.no_grad():
            preds = torch.stack([m(x) for m in models], dim=0)  # (N, B, C)

        if method == "mean":
            return preds.mean(dim=0)
        elif method == "vote":
            votes = preds.argmax(dim=-1)            # (N, B)
            return torch.mode(votes, dim=0).values
        raise ValueError(f"Unknown method '{method}'")


# ── Usage ────────────────────────────────────────────────────────────────────
base_model = tv.resnet18(pretrained=True)
base_model.fc = nn.Linear(512, 10)

factory = EnsembleFactory(base_model)
ensemble = factory.create_members(n=5, seeds=[42, 43, 44, 45, 46])

x = torch.randn(8, 3, 224, 224)
predictions = factory.ensemble_predict(ensemble, x, method="mean")
print(predictions.shape)   # torch.Size([8, 10])
```

---

## 7.7 The Object Pool Pattern

An **Object Pool** pre-allocates a set of expensive objects and leases them to consumers, returning them to the pool when done. This avoids the overhead of repeated allocation and GC.

<div class="diagram">
  <div class="diagram-title">Object Pool Lifecycle</div>
  <div class="flow">
    <div class="flow-node accent wide">Pool Initialisation<br><small>pre-allocate N objects</small></div>
    <div class="flow-arrow accent">↓</div>
    <div class="flow-node green wide">acquire()<br><small>lease an idle object</small></div>
    <div class="flow-arrow green">↓</div>
    <div class="flow-node blue wide">Use Object<br><small>compute / forward / load</small></div>
    <div class="flow-arrow accent">↓</div>
    <div class="flow-node purple wide">release()<br><small>return to pool</small></div>
    <div class="flow-arrow purple">↓ (loop)</div>
    <div class="flow-node teal wide">Pool Reuse<br><small>no re-allocation cost</small></div>
  </div>
</div>

### GPU Memory Pool — Pre-allocating Tensor Buffers

```python
import torch
import threading
from contextlib import contextmanager
from collections import deque


class TensorPool:
    """
    Pre-allocates fixed-size CUDA tensors and leases them.
    Eliminates repeated cudaMalloc / cudaFree cycles during training.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        pool_size: int = 8,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ):
        self._shape = shape
        self._device = device
        self._lock = threading.Lock()
        self._available: deque[torch.Tensor] = deque()
        self._all: list[torch.Tensor] = []

        for _ in range(pool_size):
            buf = torch.empty(shape, dtype=dtype, device=device)
            self._available.append(buf)
            self._all.append(buf)

    def acquire(self) -> torch.Tensor:
        with self._lock:
            if not self._available:
                # Expand pool automatically if exhausted
                buf = torch.empty(self._shape, device=self._device)
                self._all.append(buf)
                return buf
            return self._available.popleft()

    def release(self, tensor: torch.Tensor) -> None:
        with self._lock:
            self._available.append(tensor)

    @contextmanager
    def lease(self):
        """Context manager for automatic release."""
        buf = self.acquire()
        try:
            yield buf
        finally:
            self.release(buf)

    @property
    def available_count(self) -> int:
        return len(self._available)


# ── Usage ────────────────────────────────────────────────────────────────────
batch_pool = TensorPool(shape=(32, 3, 224, 224), pool_size=4, device="cpu")

with batch_pool.lease() as buf:
    buf.normal_()                         # fill with random data
    output = model(buf)                   # forward pass
    # buf auto-released when exiting context
```

### Worker Pool for Data Loading

```python
import queue
import threading
from typing import Any, Callable, Iterator


class DataWorkerPool:
    """
    A pool of worker threads that pre-fetch and preprocess batches.
    Decouples data loading from model forward passes.
    """

    def __init__(
        self,
        worker_fn: Callable[..., Any],
        num_workers: int = 4,
        queue_size: int = 16,
    ):
        self._fn = worker_fn
        self._task_queue: queue.Queue = queue.Queue(maxsize=queue_size * 2)
        self._result_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._workers: list[threading.Thread] = []
        self._stop_event = threading.Event()

        for _ in range(num_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._workers.append(t)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                task = self._task_queue.get(timeout=0.1)
                result = self._fn(task)
                self._result_queue.put(result)
            except queue.Empty:
                continue

    def submit(self, task: Any) -> None:
        self._task_queue.put(task)

    def fetch(self, timeout: float = 5.0) -> Any:
        return self._result_queue.get(timeout=timeout)

    def shutdown(self) -> None:
        self._stop_event.set()
        for t in self._workers:
            t.join(timeout=2.0)


def preprocess_batch(paths: list[str]) -> torch.Tensor:
    """Simulate loading and preprocessing images."""
    return torch.randn(len(paths), 3, 224, 224)   # placeholder

pool = DataWorkerPool(worker_fn=preprocess_batch, num_workers=4)
```

### CUDA Stream Pool

```python
import torch


class CUDAStreamPool:
    """
    Maintains a pool of CUDA streams for concurrent kernel dispatch.
    Multiple streams enable overlapping computation and data transfer.
    """

    def __init__(self, num_streams: int = 4):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        self._streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self._lock = threading.Lock()
        self._available = list(range(num_streams))
        self._in_use: dict[int, torch.cuda.Stream] = {}

    def acquire(self) -> tuple[int, torch.cuda.Stream]:
        with self._lock:
            if not self._available:
                raise RuntimeError("All CUDA streams in use")
            idx = self._available.pop(0)
            self._in_use[idx] = self._streams[idx]
            return idx, self._streams[idx]

    def release(self, idx: int) -> None:
        with self._lock:
            self._in_use.pop(idx, None)
            self._available.append(idx)

    @contextmanager
    def stream(self):
        idx, s = self.acquire()
        try:
            with torch.cuda.stream(s):
                yield s
        finally:
            s.synchronize()
            self.release(idx)


# ── Usage ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    stream_pool = CUDAStreamPool(num_streams=4)
    with stream_pool.stream() as s:
        result = model(batch.cuda())   # runs on a pooled CUDA stream
```

---

## 7.8 Prototype Registry — Combining Prototype + Registry

```python
from __future__ import annotations
import copy
import torch.nn as nn
from typing import Any


class PrototypeRegistry:
    """
    Stores named model prototypes and creates clones on demand.
    Combines the Registry (name lookup) and Prototype (clone) patterns.
    """

    def __init__(self):
        self._prototypes: dict[str, nn.Module] = {}

    def register(self, name: str, model: nn.Module) -> None:
        """Deep-copy before storing so the original is never aliased."""
        self._prototypes[name] = copy.deepcopy(model)
        # Freeze the canonical prototype
        for p in self._prototypes[name].parameters():
            p.requires_grad_(False)

    def clone(self, name: str, **overrides: Any) -> nn.Module:
        """Return a trainable deep-copy with optional attribute overrides."""
        if name not in self._prototypes:
            raise KeyError(f"Prototype '{name}' not registered. "
                           f"Available: {list(self._prototypes)}")
        cloned = copy.deepcopy(self._prototypes[name])
        for p in cloned.parameters():
            p.requires_grad_(True)
        for attr, val in overrides.items():
            setattr(cloned, attr, val)
        return cloned

    def list(self) -> list[str]:
        return sorted(self._prototypes)

    def __repr__(self) -> str:
        entries = ", ".join(f"'{k}'" for k in self.list())
        return f"PrototypeRegistry([{entries}])"


# ── Build a shared registry ───────────────────────────────────────────────────
proto_registry = PrototypeRegistry()

resnet18 = tv.resnet18(pretrained=True)
proto_registry.register("resnet18", resnet18)

mobilenet = tv.mobilenet_v3_small(pretrained=True)
proto_registry.register("mobilenet_v3_small", mobilenet)

# Clone for different downstream tasks
cifar_model  = proto_registry.clone("resnet18")
cifar_model.fc = nn.Linear(512, 10)

imagenet_ft  = proto_registry.clone("resnet18")
imagenet_ft.fc = nn.Linear(512, 200)   # Tiny-ImageNet

print(proto_registry)
# PrototypeRegistry(['mobilenet_v3_small', 'resnet18'])
```

---

## 7.9 Flow Diagram: Prototype-Based Fine-Tuning Pipeline

<div class="diagram">
  <div class="diagram-title">Original Model → Prototype Registry → Clone → Fine-tune</div>
  <div class="flow">
    <div class="flow-node accent wide">Pretrained Model<br><small>HuggingFace / torchvision / custom</small></div>
    <div class="flow-arrow accent">↓ register()</div>
    <div class="flow-node green wide">Prototype Registry<br><small>frozen deep copy stored</small></div>
    <div class="flow-arrow green">↓ clone()</div>
    <div class="flow-node blue wide">Trainable Clone<br><small>independent weights, requires_grad=True</small></div>
    <div class="flow-arrow blue">↓ replace head</div>
    <div class="flow-node purple wide">Task-Specific Model<br><small>new classifier / decoder / head</small></div>
    <div class="flow-arrow purple">↓ fine-tune</div>
    <div class="flow-node teal wide">Fine-tuned Model<br><small>saved checkpoint</small></div>
    <div class="flow-arrow accent">↓ register (optional)</div>
    <div class="flow-node orange wide">New Prototype<br><small>fine-tuned checkpoint as next prototype</small></div>
  </div>
</div>

---

## 7.10 Comparison Tables

### Object Creation Strategies

<div class="compare-table">

| Aspect | Factory | Builder | Prototype |
|---|---|---|---|
| **When to use** | Many variants of one type | Complex step-by-step construction | Copy & adjust known-good object |
| **Configuration** | Constructor args / config dict | Fluent builder calls | Clone + post-clone mutation |
| **Expense of init** | Moderate | High (many steps) | Low (copy is cheap) |
| **Independence of copies** | Always independent | Always independent | Depends: deep vs shallow |
| **ML example** | `model_registry.build("resnet")` | `TrainerBuilder().set_lr(…).build()` | `proto_registry.clone("resnet18")` |
| **State sharing risk** | None | None | High with shallow copy |

</div>

### Construction Performance (Relative)

<div class="compare-table">

| Method | Relative Time | Memory overhead | Notes |
|---|---|---|---|
| `nn.Module()` from scratch | 1.0× (baseline) | 0 | Full init, random weights |
| `copy.deepcopy(model)` | ~1.2× | 2× params | Copies Python objects too |
| `state_dict` clone | ~1.4× | 2× params | Safe across devices |
| `io.BytesIO` serialize | ~1.8× | 3× peak | Safest, no object sharing |
| Object pool (reuse) | ~0.02× | 0 | No alloc — just reset |
| `torch.save` / `torch.load` | ~3–5× | 1× disk + 2× RAM | For cross-process |

</div>

---

## 7.11 Summary

<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-icon">🧬</div>
    <div class="card-title">Prototype</div>
    <div class="card-desc">Clone a known-good object instead of constructing from scratch. Essential for ensemble creation, fine-tuning, and config branching. Always deep-copy unless you explicitly want shared state.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🏊</div>
    <div class="card-title">Object Pool</div>
    <div class="card-desc">Pre-allocate expensive objects (GPU tensors, CUDA streams, workers) and reuse them. Eliminates allocation overhead in tight training loops. Use context managers for safe lease/release.</div>
  </div>
</div>

<div class="callout tip">
<strong>💡 Deep Copy Pitfalls</strong><br>
<code>copy.deepcopy</code> on large models can be slow and use 2× memory. For large models (>1B params), prefer <code>state_dict</code>-based cloning or serialise to disk. Always benchmark before choosing a cloning strategy.
</div>

---

*Last updated: May 2026*
