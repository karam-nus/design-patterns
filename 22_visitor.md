---
title: "Chapter 22 — Visitor & Graph Traversal"
---

[← Back to Table of Contents](./README.md)

# Chapter 22 — Visitor & Graph Traversal

> *"Add new operations to objects without modifying them — the visitor knocks; the element answers."*

<span class="badge behavioral">Behavioral</span> <span class="badge pytorch">PyTorch</span>

---

## 22.1 Intent

The **Visitor** pattern lets you define a new operation on a family of objects without altering their classes. Instead of putting the operation inside each class, you write a *visitor* object that knows how to handle each type. Each element "accepts" a visitor and dispatches to the appropriate handler.

In deep learning, this maps onto **model graph traversal**: inspecting, profiling, pruning, quantising, or exporting a neural network — all operations that touch every node in a computational graph without modifying the node classes themselves.

---

## 22.2 UML Structure

<div class="diagram">
  <div class="diagram-title">Visitor Pattern — Class Structure</div>
  <div class="uml-row">
    <div class="uml-box">
      <div class="uml-title">«abstract» Visitor</div>
      <div class="uml-section">
        <div class="uml-item">+ visit_linear(e: Linear)</div>
        <div class="uml-item">+ visit_conv(e: Conv2d)</div>
        <div class="uml-item">+ visit_bn(e: BatchNorm2d)</div>
      </div>
    </div>
    <div class="uml-box">
      <div class="uml-title">ParamCountVisitor</div>
      <div class="uml-section">
        <div class="uml-item">+ visit_linear(e)</div>
        <div class="uml-item">+ visit_conv(e)</div>
        <div class="uml-item">+ visit_bn(e)</div>
        <div class="uml-item">─────────────</div>
        <div class="uml-item">total: int = 0</div>
      </div>
    </div>
    <div class="uml-box">
      <div class="uml-title">«abstract» Element</div>
      <div class="uml-section">
        <div class="uml-item">+ accept(v: Visitor)</div>
      </div>
    </div>
  </div>
</div>

---

## 22.3 Double Dispatch — Why Single Dispatch Fails

In a statically-typed system, `visitor.visit(element)` selects the method based only on the *static* type of `element`. If you have `Linear`, `Conv2d`, and `BatchNorm2d` all passed as `nn.Module`, single dispatch calls the same method for all.

**Double dispatch** solves this by letting the *element* call back into the visitor with its concrete type:

```python
# Single dispatch problem:
def visit(module: nn.Module) -> None:
    # At runtime, module could be Linear, Conv2d, etc.
    # We can't dispatch cleanly without isinstance chains
    if isinstance(module, nn.Linear):
        handle_linear(module)
    elif isinstance(module, nn.Conv2d):
        handle_conv(module)
    # ... fragile, closed to extension


# Double dispatch solution:
class LinearElement(nn.Linear):
    def accept(self, visitor: "Visitor") -> None:
        visitor.visit_linear(self)   # dispatches to correct method

class Conv2dElement(nn.Conv2d):
    def accept(self, visitor: "Visitor") -> None:
        visitor.visit_conv(self)
```

In Python we achieve the same effect more cleanly with a dispatch table or `singledispatchmethod`, since we can inspect types at runtime efficiently.

---

## 22.4 `nn.Module.apply()` as Visitor

PyTorch provides `Module.apply(fn)` — a built-in depth-first visitor that calls `fn` on every submodule.

```python
import torch
import torch.nn as nn
from typing import Callable

# --- Weight initialisation visitor ---
def kaiming_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Linear(64, 10),
)
model.apply(kaiming_init)


# --- Norm freezing visitor ---
def freeze_norms(module: nn.Module) -> None:
    """Freeze all BatchNorm and LayerNorm layers."""
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d,
                           nn.BatchNorm3d, nn.LayerNorm)):
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)


model.apply(freeze_norms)
```

### Composing Multiple Visitors

```python
def compose_visitors(*fns: Callable[[nn.Module], None]) -> Callable[[nn.Module], None]:
    """Combine multiple visitor functions into one."""
    def combined(module: nn.Module) -> None:
        for fn in fns:
            fn(module)
    return combined

model.apply(compose_visitors(kaiming_init, freeze_norms))
```

---

## 22.5 Model Introspection Visitors

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Type

class ModelVisitor(ABC):
    """Abstract base for model visitors."""

    @abstractmethod
    def visit(self, name: str, module: nn.Module) -> None: ...

    def run(self, model: nn.Module) -> "ModelVisitor":
        for name, module in model.named_modules():
            self.visit(name, module)
        return self


# --- Concrete Visitor 1: Parameter Counter ---
@dataclass
class ParamCountVisitor(ModelVisitor):
    total: int = 0
    trainable: int = 0
    by_layer: dict = field(default_factory=dict)

    def visit(self, name: str, module: nn.Module) -> None:
        params = sum(p.numel() for p in module.parameters(recurse=False))
        trainable = sum(p.numel() for p in module.parameters(recurse=False)
                        if p.requires_grad)
        if params > 0:
            self.by_layer[name or "root"] = {"total": params, "trainable": trainable}
            self.total += params
            self.trainable += trainable

    def report(self) -> str:
        return (f"Total parameters: {self.total:,}\n"
                f"Trainable: {self.trainable:,}\n"
                f"Frozen: {self.total - self.trainable:,}")


# --- Concrete Visitor 2: Layer Type Collector ---
@dataclass
class LayerTypeVisitor(ModelVisitor):
    type_counts: dict = field(default_factory=dict)
    layers: list = field(default_factory=list)

    def visit(self, name: str, module: nn.Module) -> None:
        type_name = type(module).__name__
        self.type_counts[type_name] = self.type_counts.get(type_name, 0) + 1
        self.layers.append((name, type_name))


# --- Concrete Visitor 3: FLOP Counter (simplified) ---
@dataclass
class FLOPCountVisitor(ModelVisitor):
    total_flops: int = 0
    _hooks: list = field(default_factory=list)

    def visit(self, name: str, module: nn.Module) -> None:
        def _hook(mod, inp, out):
            if isinstance(mod, nn.Linear):
                batch = inp[0].shape[0]
                self.total_flops += 2 * batch * mod.in_features * mod.out_features
            elif isinstance(mod, nn.Conv2d):
                batch, c_out, h, w = out.shape
                c_in = mod.in_channels
                kh, kw = mod.kernel_size
                self.total_flops += 2 * batch * c_out * h * w * c_in * kh * kw
        self._hooks.append(module.register_forward_hook(_hook))

    def clear_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# Usage
import torchvision.models as tvm

resnet = tvm.resnet18(weights=None)

param_visitor = ParamCountVisitor().run(resnet)
print(param_visitor.report())

layer_visitor = LayerTypeVisitor().run(resnet)
for type_name, count in sorted(layer_visitor.type_counts.items()):
    print(f"  {type_name}: {count}")
```

---

## 22.6 Pruning Mask Visitor

```python
import torch.nn.utils.prune as prune

class PruningVisitor(ModelVisitor):
    """Apply magnitude pruning to all Conv2d and Linear layers."""

    def __init__(self, amount: float = 0.3) -> None:
        self.amount = amount
        self.pruned_layers: list[str] = []

    def visit(self, name: str, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            prune.l1_unstructured(module, name="weight", amount=self.amount)
            self.pruned_layers.append(name)

    def make_permanent(self, model: nn.Module) -> None:
        """Remove pruning reparametrisation — bake masks in."""
        for name, module in model.named_modules():
            if name in self.pruned_layers:
                try:
                    prune.remove(module, "weight")
                except ValueError:
                    pass


model = tvm.resnet18(weights=None)
pruner = PruningVisitor(amount=0.4)
pruner.run(model)
print(f"Pruned {len(pruner.pruned_layers)} layers")
pruner.make_permanent(model)
```

---

## 22.7 Quantisation Visitor

```python
import torch.quantization as tq

class QuantisationVisitor(ModelVisitor):
    """Attach quantisation stubs and configs to targeted layers."""

    def __init__(self, qconfig=None) -> None:
        self.qconfig = qconfig or tq.get_default_qconfig("fbgemm")
        self.configured: list[str] = []

    def visit(self, name: str, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.qconfig = self.qconfig
            self.configured.append(name)
        elif isinstance(module, (nn.ReLU, nn.ReLU6)):
            # Fuse-friendly — no qconfig needed directly
            pass


model = tvm.resnet18(weights=None)
model.eval()
quant_visitor = QuantisationVisitor()
quant_visitor.run(model)
tq.prepare(model, inplace=True)
# ... calibration data pass ...
tq.convert(model, inplace=True)
```

---

## 22.8 `torch.fx` as a Graph Visitor

`torch.fx` symbolically traces a model into an explicit IR graph, where every operation is a `Node` object. This enables surgical graph transformations.

```python
import torch.fx as fx
from torch.fx import GraphModule, Node

def count_relu_nodes(model: nn.Module) -> int:
    """Count all ReLU calls in the fx graph."""
    traced: GraphModule = fx.symbolic_trace(model)
    count = 0
    for node in traced.graph.nodes:
        if node.op == "call_module":
            submod = traced.get_submodule(node.target)
            if isinstance(submod, nn.ReLU):
                count += 1
        elif node.op == "call_function" and node.target is torch.relu:
            count += 1
    return count


# --- Custom transformation pass: replace ReLU with GELU ---
def replace_relu_with_gelu(model: nn.Module) -> GraphModule:
    traced = fx.symbolic_trace(model)

    for node in traced.graph.nodes:
        if node.op == "call_module":
            submod = traced.get_submodule(node.target)
            if isinstance(submod, nn.ReLU):
                # Replace in the module tree
                parts = node.target.rsplit(".", 1)
                parent = traced.get_submodule(parts[0]) if len(parts) > 1 else traced
                setattr(parent, parts[-1] if len(parts) > 1 else node.target,
                        nn.GELU())

    traced.recompile()
    return traced


simple_model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
transformed = replace_relu_with_gelu(simple_model)
print(transformed.graph)
```

---

## 22.9 Module Summary Table via Visitor

```python
from dataclasses import dataclass

@dataclass
class LayerSummary:
    name: str
    type: str
    output_shape: tuple
    params: int
    trainable_params: int


class SummaryVisitor:
    """Build a torchsummary-style table using hooks."""

    def __init__(self) -> None:
        self.rows: list[LayerSummary] = []
        self._hooks: list = []

    def run(self, model: nn.Module, input_size: tuple) -> "SummaryVisitor":
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules only
                self._attach_hook(name, module)

        dummy = torch.zeros(1, *input_size)
        model(dummy)

        for h in self._hooks:
            h.remove()
        return self

    def _attach_hook(self, name: str, module: nn.Module) -> None:
        def _hook(mod, inp, out):
            params = sum(p.numel() for p in mod.parameters())
            trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            shape = tuple(out.shape[1:]) if hasattr(out, "shape") else ()
            self.rows.append(LayerSummary(
                name=name or "root",
                type=type(mod).__name__,
                output_shape=shape,
                params=params,
                trainable_params=trainable,
            ))
        self._hooks.append(module.register_forward_hook(_hook))

    def print_table(self) -> None:
        print(f"{'Layer':<40} {'Type':<20} {'Output':<20} {'Params':>10}")
        print("-" * 92)
        total = 0
        for row in self.rows:
            print(f"{row.name:<40} {row.type:<20} {str(row.output_shape):<20} "
                  f"{row.params:>10,}")
            total += row.params
        print("=" * 92)
        print(f"{'Total parameters':<80} {total:>10,}")


model = tvm.resnet18(weights=None)
SummaryVisitor().run(model, (3, 224, 224)).print_table()
```

---

## 22.10 Flow Diagram

<div class="diagram">
  <div class="diagram-title">Visitor Traversal — Model Graph</div>
  <div class="flow">
    <div class="flow-node accent wide">Model Graph<br/><small>nn.Module tree</small></div>
    <div class="flow-arrow accent">→ accept(visitor)</div>
    <div class="flow-node blue">Visitor<br/><small>dispatch</small></div>
  </div>
  <div class="flow">
    <div class="flow-node green">ParamCounter<br/><small>count params</small></div>
    <div class="flow-node purple">FLOPCounter<br/><small>count ops</small></div>
    <div class="flow-node orange">Pruner<br/><small>set masks</small></div>
    <div class="flow-node teal">Quantiser<br/><small>attach config</small></div>
  </div>
</div>

---

## 22.11 Comparison Table

| Criterion | Visitor | Iterator | Decorator |
|-----------|---------|----------|-----------|
| Adds new operations | ✅ easily | ❌ changes traversal | ❌ wraps existing |
| Modifies element classes | ❌ no | ❌ no | ✅ wraps |
| Type-specific dispatch | ✅ explicit | ❌ uniform | ❌ uniform |
| Double dispatch | ✅ | ❌ | ❌ |
| Good for model graph ops | ✅ | For sequences | For single module |
| Python naturalness | Medium | High | High |

<div class="callout tip">
<strong>Tip:</strong> Prefer <code>apply()</code> for simple per-module operations (init, freeze, dtype cast). Use the full Visitor pattern when you need type-specific logic with state accumulation across multiple module types — e.g., building a summary table or a calibration profile.
</div>

---

## Summary

<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">🔀</div>
    <div class="card-title">Double Dispatch</div>
    <div class="card-desc">Element calls back into visitor with its concrete type — the key mechanism that makes Visitor work.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🔍</div>
    <div class="card-title">apply() — Built-in Visitor</div>
    <div class="card-desc">PyTorch's <code>apply(fn)</code> is a DFS visitor. Use it for init, freezing, and dtype changes.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🔬</div>
    <div class="card-title">torch.fx — IR Visitor</div>
    <div class="card-desc">Graph-level traversal for structural rewrites: replace ops, fuse layers, count FLOPs precisely.</div>
  </div>
</div>

*Last updated: May 2026*
