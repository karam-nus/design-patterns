---
title: "Chapter 9 — Composite & Module Hierarchy"
---
[← Back to Table of Contents](./README.md)

# Chapter 9 — Composite & Module Hierarchy

> *"A leaf and a branch are both nodes of the same tree — the trunk doesn't need to know which it's holding."*

---

## 9.1 The Composite Pattern

The **Composite** pattern lets you compose objects into tree structures to represent part-whole hierarchies. It enables clients to treat individual objects (leaves) and compositions (branches) **uniformly** through the same interface.

The two core rules are:
1. **Leaves** and **Composites** share the same interface.
2. A **Composite** holds a collection of children — each child may be a Leaf or another Composite.

This recursive structure is everywhere in deep learning: `nn.Module` is the most influential implementation of the Composite pattern in any ML framework.

<div class="diagram">
  <div class="diagram-title">Composite Pattern — Part-Whole Hierarchy</div>
  <div class="flow">
    <div class="flow-node accent wide">Component (abstract)<br><small>forward(x) — uniform interface</small></div>
    <div class="flow-arrow accent">↓ inherits</div>
    <div class="flow-node flow-h">
      <div class="flow-node green">Leaf<br><small>nn.Linear, nn.Conv2d, nn.ReLU</small></div>
      <div class="flow-node blue wide">Composite<br><small>nn.Sequential, ResBlock, Encoder</small></div>
    </div>
    <div class="flow-arrow blue">↓ Composite contains</div>
    <div class="flow-node purple wide">children: list[Component]<br><small>any mix of Leaf and Composite</small></div>
  </div>
</div>

---

## 9.2 UML Diagram

<div class="uml-row">
  <div class="uml-box">
    <div class="uml-title">&lt;&lt;abstract&gt;&gt; Component</div>
    <div class="uml-section">Methods</div>
    <div class="uml-item">+ forward(x: Tensor) → Tensor</div>
    <div class="uml-item">+ parameters() → Iterator</div>
    <div class="uml-item">+ train(mode: bool) → self</div>
    <div class="uml-item">+ to(device) → self</div>
  </div>

  <div class="uml-box">
    <div class="uml-title">Leaf (e.g. nn.Linear)</div>
    <div class="uml-section">Attributes</div>
    <div class="uml-item">- weight: Parameter</div>
    <div class="uml-item">- bias: Parameter</div>
    <div class="uml-section">Methods</div>
    <div class="uml-item">+ forward(x) → Tensor</div>
    <div class="uml-item">  ↳ no children</div>
  </div>

  <div class="uml-box">
    <div class="uml-title">Composite (e.g. nn.Sequential)</div>
    <div class="uml-section">Attributes</div>
    <div class="uml-item">- _modules: OrderedDict</div>
    <div class="uml-section">Methods</div>
    <div class="uml-item">+ add_module(name, module)</div>
    <div class="uml-item">+ forward(x) → Tensor</div>
    <div class="uml-item">  ↳ calls each child.forward()</div>
    <div class="uml-item">+ children() → Iterator</div>
    <div class="uml-item">+ modules() → Iterator</div>
  </div>
</div>

---

## 9.3 `nn.Module` as the Composite Pattern in PyTorch

`nn.Module` is PyTorch's implementation of Component. It serves as both Leaf and Composite — any module can contain children, and any module has a `forward()` method.

### Tree Traversal Methods

```python
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class SimpleEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = ResidualBlock(base_ch)
        self.layer2 = ResidualBlock(base_ch)
        self.pool   = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.pool(x).flatten(1)


encoder = SimpleEncoder()

# children() — immediate children only (1 level deep)
print("── children() ───────────────────────────────")
for name, child in encoder.named_children():
    print(f"  {name}: {type(child).__name__}")
# stem: Sequential
# layer1: ResidualBlock
# layer2: ResidualBlock
# pool: AdaptiveAvgPool2d

# modules() — full recursive tree (depth-first)
print("\n── modules() ────────────────────────────────")
for name, mod in encoder.named_modules():
    indent = "  " * name.count(".")
    print(f"  {indent}{name or 'encoder'}: {type(mod).__name__}")

# parameters() — all leaf tensors that require grad
total_params = sum(p.numel() for p in encoder.parameters())
print(f"\nTotal parameters: {total_params:,}")
```

### The `forward()` Uniform Interface

The power of the Composite pattern: code that calls `forward()` doesn't need to know if it's talking to a single layer or a 152-layer network:

```python
def run_inference(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Works identically for a single Linear layer or an entire ResNet."""
    module.eval()
    with torch.no_grad():
        return module(x)    # dispatches to module.forward()

# Same function works for both:
single_layer = nn.Linear(512, 10)
full_network = SimpleEncoder()

run_inference(single_layer, torch.randn(4, 512))
run_inference(full_network, torch.randn(4, 3, 64, 64))
```

---

## 9.4 Building a Composite Transform Pipeline

Transforms are a natural Composite: individual transforms are Leaves, `Compose` is a Composite.

```python
import torch
import torchvision.transforms.functional as TF
from abc import ABC, abstractmethod
from typing import Any


class Transform(ABC):
    """Component — uniform interface for all transforms."""
    @abstractmethod
    def __call__(self, x: Any) -> Any: ...
    def __repr__(self): return type(self).__name__


# ── Leaf transforms ──────────────────────────────────────────────────────────

class Normalise(Transform):
    def __init__(self, mean: list[float], std: list[float]):
        self.mean = mean
        self.std  = std
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return TF.normalize(x, self.mean, self.std)
    def __repr__(self): return f"Normalise(mean={self.mean})"


class RandomHorizontalFlip(Transform):
    def __init__(self, p: float = 0.5):
        self.p = p
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            return TF.hflip(x)
        return x


class RandomCrop(Transform):
    def __init__(self, size: int, padding: int = 4):
        self.size = size
        self.padding = padding
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        padded = TF.pad(x, self.padding)
        return TF.center_crop(padded, self.size)


class CutOut(Transform):
    """Randomly mask out square patches."""
    def __init__(self, n_holes: int = 1, length: int = 16):
        self.n_holes = n_holes
        self.length  = length

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        mask = torch.ones_like(x)
        for _ in range(self.n_holes):
            cy = torch.randint(h, (1,)).item()
            cx = torch.randint(w, (1,)).item()
            y1, y2 = max(0, cy - self.length//2), min(h, cy + self.length//2)
            x1, x2 = max(0, cx - self.length//2), min(w, cx + self.length//2)
            mask[..., y1:y2, x1:x2] = 0
        return x * mask


# ── Composite transform ───────────────────────────────────────────────────────

class Compose(Transform):
    """Composite — chains transforms sequentially."""
    def __init__(self, transforms: list[Transform]):
        self._transforms = transforms

    def __call__(self, x: Any) -> Any:
        for t in self._transforms:
            x = t(x)
        return x

    def append(self, t: Transform) -> "Compose":
        return Compose(self._transforms + [t])

    def __repr__(self):
        inner = "\n  ".join(repr(t) for t in self._transforms)
        return f"Compose(\n  {inner}\n)"


# ── Nested Composite: augmentation policy with sub-pipelines ─────────────────

class RandomChoice(Transform):
    """Randomly applies one of N sub-pipelines (Composite of Composites)."""
    def __init__(self, transforms: list[Transform]):
        self._transforms = transforms

    def __call__(self, x: Any) -> Any:
        idx = torch.randint(len(self._transforms), (1,)).item()
        return self._transforms[idx](x)


train_transform = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomCrop(size=32, padding=4),
    RandomChoice([
        CutOut(n_holes=1, length=8),
        CutOut(n_holes=2, length=4),
    ]),
    Normalise(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
])

print(train_transform)
```

---

## 9.5 Recursive `forward()` — How PyTorch Walks the Module Tree

PyTorch's `Module.__call__` does more than just invoke `forward()` — it runs hooks, handles training mode, etc. The tree walk is depth-first:

```python
import torch.nn as nn
from typing import Callable


def trace_forward(module: nn.Module, depth: int = 0) -> Callable:
    """
    Monkey-patch a module tree to trace forward calls.
    Illustrates depth-first tree traversal order.
    """
    prefix = "  " * depth
    original_forward = module.forward

    def traced(*args, **kwargs):
        print(f"{prefix}→ {type(module).__name__}.forward()")
        out = original_forward(*args, **kwargs)
        print(f"{prefix}← {type(module).__name__} done, shape={list(out.shape)}")
        return out

    module.forward = traced

    for child in module.children():
        trace_forward(child, depth + 1)

    return module


# Usage (illustrative — modifies the module in place)
# small_net = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
# trace_forward(small_net)
# small_net(torch.randn(3, 10))
```

---

## 9.6 Full Example: ResNet-like Architecture as a Composite Tree

```python
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional


class ConvBnReLU(nn.Module):
    """Leaf-level building block: Conv → BN → ReLU."""
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class BottleneckBlock(nn.Module):
    """Composite: 1×1 → 3×3 → 1×1 + skip connection."""
    expansion = 4

    def __init__(self, in_ch: int, mid_ch: int, stride: int = 1):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.path = nn.Sequential(
            ConvBnReLU(in_ch, mid_ch, kernel=1, padding=0),
            ConvBnReLU(mid_ch, mid_ch, kernel=3, stride=stride),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.path(x) + self.skip(x))


class ResStage(nn.Module):
    """Composite: a sequence of BottleneckBlocks at one resolution."""
    def __init__(self, in_ch: int, mid_ch: int, n_blocks: int, stride: int = 1):
        super().__init__()
        blocks = [BottleneckBlock(in_ch, mid_ch, stride)]
        out_ch = mid_ch * BottleneckBlock.expansion
        for _ in range(1, n_blocks):
            blocks.append(BottleneckBlock(out_ch, mid_ch))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ResNetBackbone(nn.Module):
    """
    Top-level Composite.
    Architecture: stem → stage1 → stage2 → stage3 → stage4 → avgpool
    """

    def __init__(
        self,
        layers: list[int] = (3, 4, 6, 3),
        num_classes: int = 1000,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBnReLU(3, 64, kernel=7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage1 = ResStage(64,  64,  layers[0], stride=1)
        self.stage2 = ResStage(256, 128, layers[1], stride=2)
        self.stage3 = ResStage(512, 256, layers[2], stride=2)
        self.stage4 = ResStage(1024, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc     = nn.Linear(2048, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        return self.fc(x.flatten(1))


resnet50 = ResNetBackbone(layers=[3, 4, 6, 3])
x = torch.randn(2, 3, 224, 224)
out = resnet50(x)
print(out.shape)   # torch.Size([2, 1000])
```

---

## 9.7 `add_module()` and `__setattr__` — How Modules Are Registered

When you assign a `nn.Module` as an attribute, PyTorch intercepts via `__setattr__`:

```python
class CustomComposite(nn.Module):
    def __init__(self):
        super().__init__()
        # These are equivalent — both call add_module() internally:
        self.linear = nn.Linear(10, 5)          # via __setattr__
        self.add_module("activation", nn.ReLU()) # explicit

        # Parameters are also auto-registered:
        self.scale = nn.Parameter(torch.ones(5))

        # Plain tensor — NOT tracked (no grad, no .to(device)):
        self.buffer_data = torch.zeros(5)        # ← not tracked

        # Registered buffer — moved with .to(device) but not a param:
        self.register_buffer("running_mean", torch.zeros(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x)) * self.scale + self.running_mean


model = CustomComposite()

print("Named modules:")
for n, m in model.named_modules():
    if n: print(f"  {n}: {type(m).__name__}")

print("\nNamed parameters:")
for n, p in model.named_parameters():
    print(f"  {n}: {p.shape}")

print("\nNamed buffers:")
for n, b in model.named_buffers():
    print(f"  {n}: {b.shape}")
```

---

## 9.8 Module Tree Visualisation

```python
import torch.nn as nn


def print_module_tree(
    module: nn.Module,
    name: str = "root",
    depth: int = 0,
    max_depth: int = 10,
) -> None:
    """Pretty-print the full module tree with parameter counts."""
    if depth > max_depth:
        return
    indent  = "  " * depth
    connector = "└─" if depth > 0 else ""
    param_count = sum(p.numel() for p in module.parameters(recurse=False))
    buf_count   = sum(b.numel() for b in module.buffers(recurse=False))
    extra = f" [params={param_count:,}, buffers={buf_count}]" if (param_count or buf_count) else ""
    print(f"{indent}{connector} {name} ({type(module).__name__}){extra}")

    for child_name, child in module.named_children():
        print_module_tree(child, child_name, depth + 1, max_depth)


# Example output for ResNetBackbone:
# root (ResNetBackbone)
#   └─ stem (Sequential)
#     └─ 0 (ConvBnReLU)
#       └─ conv (Conv2d) [params=9,408]
#       └─ bn (BatchNorm2d) [params=128, buffers=128]
#       └─ relu (ReLU)
#     └─ 1 (MaxPool2d)
#   └─ stage1 (ResStage)
#     └─ blocks (Sequential)
#       └─ 0 (BottleneckBlock)
#         ...


def count_parameters(module: nn.Module) -> dict[str, int]:
    """Return per-submodule parameter counts."""
    counts = {}
    for name, mod in module.named_modules():
        n = sum(p.numel() for p in mod.parameters(recurse=False))
        if n > 0:
            counts[name] = n
    total = sum(p.numel() for p in module.parameters())
    counts["__total__"] = total
    return counts
```

---

## 9.9 `nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict` as Specialised Composites

<div class="compare-table">

| Container | Children access | Order guaranteed | Dynamic add/remove | Typical use |
|---|---|---|---|---|
| `nn.Sequential` | Via `forward()` chain | Yes | No (rebuild needed) | Fixed linear pipeline |
| `nn.ModuleList` | By integer index | Yes | Yes (`append`, `insert`) | Variable-depth architectures, ensembles |
| `nn.ModuleDict` | By string key | Python 3.7+ | Yes | Multi-task heads, named branches |

</div>

```python
import torch
import torch.nn as nn


# ── nn.Sequential ─────────────────────────────────────────────────────────────
mlp = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
out = mlp(torch.randn(8, 784))


# ── nn.ModuleList ─────────────────────────────────────────────────────────────
class DynamicResNet(nn.Module):
    """Depth is determined at runtime — ModuleList enables this."""

    def __init__(self, n_blocks: int, channels: int = 64):
        super().__init__()
        self.stem   = ConvBnReLU(3, channels, kernel=3, padding=1)
        self.blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(n_blocks)]
        )
        self.head   = nn.Linear(channels, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:           # explicit iteration — full flexibility
            x = block(x)
        return self.head(x.mean(dim=[-2, -1]))


# ── nn.ModuleDict ─────────────────────────────────────────────────────────────
class MultiTaskHead(nn.Module):
    """Separate output heads for different tasks, accessed by name."""

    def __init__(self, in_features: int, tasks: dict[str, int]):
        super().__init__()
        self.heads = nn.ModuleDict({
            task: nn.Linear(in_features, n_classes)
            for task, n_classes in tasks.items()
        })

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {task: head(features) for task, head in self.heads.items()}


mt_head = MultiTaskHead(512, {"age_regression": 1, "gender_cls": 2, "emotion_cls": 7})
feats = torch.randn(4, 512)
outputs = mt_head(feats)
# {"age_regression": Tensor[4,1], "gender_cls": Tensor[4,2], "emotion_cls": Tensor[4,7]}
```

---

## 9.10 Pipeline Composition: `Compose` as a Composite for Transforms

```python
from torchvision import transforms as T

# torchvision.transforms.Compose IS the Composite pattern
train_pipeline = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# We can nest Composites — a sub-pipeline inside a pipeline
augmentation = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(15),
])

preprocessing = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5], [0.5]),
])

# Compose of Composites — the outer Compose doesn't know/care
full_pipeline = T.Compose([augmentation, preprocessing])
```

---

## 9.11 Module Hierarchy Tree Diagram

```
ResNetBackbone
├── stem (Sequential)
│   ├── [0] ConvBnReLU
│   │   ├── conv  (Conv2d)        7×7, 3→64, stride=2
│   │   ├── bn    (BatchNorm2d)   64
│   │   └── relu  (ReLU)
│   └── [1] MaxPool2d             3×3, stride=2
│
├── stage1 (ResStage)             3 × BottleneckBlock, 64 channels
│   └── blocks (Sequential)
│       ├── [0] BottleneckBlock
│       │   ├── path (Sequential): ConvBnReLU→ConvBnReLU→Conv2d→BN
│       │   ├── skip (Sequential): Conv2d→BN  [projection]
│       │   └── relu (ReLU)
│       ├── [1] BottleneckBlock   (skip = Identity)
│       └── [2] BottleneckBlock   (skip = Identity)
│
├── stage2 (ResStage)             4 × BottleneckBlock, 128 channels
├── stage3 (ResStage)             6 × BottleneckBlock, 256 channels
├── stage4 (ResStage)             3 × BottleneckBlock, 512 channels
│
├── avgpool (AdaptiveAvgPool2d)   output (1, 1)
└── fc      (Linear)              2048 → 1000
```

---

## 9.12 When the Composite Pattern Breaks

<div class="callout warn">
<strong>⚠ Deeply Nested Trees — Memory & Performance Implications</strong>

<ul>
<li><strong>Memory overhead</strong>: Every <code>nn.Module</code> object carries Python overhead (~200–500 bytes). A 1000-layer model can add meaningful baseline memory even before counting parameters.</li>
<li><strong>Slow <code>state_dict()</code></strong>: Collecting parameters from deeply nested trees is O(N) in module count, not just parameter count. Thousands of tiny modules can slow checkpointing.</li>
<li><strong>Hook explosion</strong>: Global hooks registered with <code>register_forward_hook</code> fire for every node — O(N) hook calls per forward pass in deep trees.</li>
<li><strong>Gradient accumulation quirks</strong>: With <code>torch.compile</code>, deeply nested composites may inhibit fusion across boundaries. Prefer flat <code>nn.Sequential</code> where possible.</li>
<li><strong>Serialisation fragility</strong>: Renaming a submodule breaks <code>load_state_dict()</code> unless you use <code>strict=False</code> and manually remap keys.</li>
</ul>

<strong>Guideline</strong>: Prefer wide, shallow trees over deep, narrow ones. Merge tiny leaf modules into functional blocks using <code>torch.nn.functional</code> where performance matters.
</div>

---

## Summary

<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-icon">🌲</div>
    <div class="card-title">Composite Pattern</div>
    <div class="card-desc">Leaves and composites share the same interface. Clients treat any node uniformly. PyTorch's nn.Module is the canonical ML implementation.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🔗</div>
    <div class="card-title">Module Registration</div>
    <div class="card-desc">Assigning an nn.Module as an attribute auto-registers it. Use register_buffer() for non-parameter tensors. ModuleList/Dict for dynamic collections.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🔄</div>
    <div class="card-title">Tree Traversal</div>
    <div class="card-desc">children() for direct children, modules() for full depth-first tree, parameters() for leaf tensors. All powered by the Composite structure.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">⚡</div>
    <div class="card-title">Specialised Composites</div>
    <div class="card-desc">Sequential for fixed pipelines, ModuleList for variable depth, ModuleDict for named branches. Each trades flexibility for clarity.</div>
  </div>
</div>

---

*Last updated: May 2026*
