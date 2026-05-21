---
title: "Chapter 24 — Hook Pattern"
---

[← Back to Table of Contents](./README.md)

# Chapter 24 — Hook Pattern

> *"Hooks are the telescope through which you observe the inner life of a neural network — without disturbing the experiment."*

PyTorch's hook system is the **Observer pattern** applied directly to tensor computations. Instead of modifying model source code to inspect or alter intermediate values, you register callbacks that fire automatically during the forward or backward pass. This separation of concerns keeps model logic clean while enabling powerful instrumentation, debugging, and interpretability tools.

---

## What Are PyTorch Hooks?

A hook is a callable registered on a `nn.Module` or a `Tensor` that PyTorch invokes at a specific point in the computation graph. Hooks follow the classic Observer contract: the subject (module/tensor) maintains a list of observers (hooks) and notifies them when specific events occur.

<div class="diagram">
  <div class="diagram-title">Hook Types and Their Firing Points</div>
  <div class="flow">
    <div class="flow-node blue wide">Input Tensor</div>
    <div class="flow-arrow accent">→</div>
    <div class="flow-node orange wide">register_forward_pre_hook<br><small>fires before forward()</small></div>
    <div class="flow-arrow accent">→</div>
    <div class="flow-node accent wide">Module.forward()</div>
    <div class="flow-arrow accent">→</div>
    <div class="flow-node green wide">register_forward_hook<br><small>fires after forward()</small></div>
    <div class="flow-arrow accent">→</div>
    <div class="flow-node blue wide">Output Tensor</div>
  </div>
  <div class="flow" style="margin-top:1rem">
    <div class="flow-node blue wide">Output Gradient</div>
    <div class="flow-arrow purple">→</div>
    <div class="flow-node pink wide">register_full_backward_hook<br><small>fires during backward()</small></div>
    <div class="flow-arrow purple">→</div>
    <div class="flow-node blue wide">Input Gradient</div>
  </div>
</div>

---

## Three Hook Types

### 1. `register_forward_pre_hook`

**Signature:** `fn(module, input) -> None or modified_input`

Fires **before** `module.forward()` is called. The hook receives the module and the tuple of inputs. Returning a modified input tuple replaces the actual inputs passed to `forward()`.

```python
def pre_hook(module, input):
    # input is a tuple of tensors
    print(f"[pre_hook] {module.__class__.__name__}: input shapes = {[t.shape for t in input]}")
    # Return modified input or None (no modification)
    return None

handle = model.layer1.register_forward_pre_hook(pre_hook)
```

### 2. `register_forward_hook`

**Signature:** `fn(module, input, output) -> None or modified_output`

Fires **after** `module.forward()` completes. The hook receives module, inputs, and the output tensor(s). Returning a value replaces the output seen by downstream layers.

```python
def forward_hook(module, input, output):
    print(f"[hook] {module.__class__.__name__}: output shape = {output.shape}")
    # Optionally return a modified output
    return None

handle = model.layer2.register_forward_hook(forward_hook)
```

### 3. `register_full_backward_hook`

**Signature:** `fn(module, grad_input, grad_output) -> None or modified_grad_input`

Fires during **backpropagation**. `grad_output` is the gradient flowing into the module's output; `grad_input` is the gradient flowing back through the module's inputs.

```python
def backward_hook(module, grad_input, grad_output):
    grad_norm = grad_output[0].norm().item()
    print(f"[backward] {module.__class__.__name__}: grad_output norm = {grad_norm:.4f}")
    return None

handle = model.layer3.register_full_backward_hook(backward_hook)
```

<div class="callout warn">
<strong>Memory leak warning:</strong> Every registered hook holds a reference to the module. If you register hooks in a training loop without removing them, you accumulate hundreds of hooks and exhaust memory. Always store the handle and call <code>handle.remove()</code> when done.
</div>

<div class="callout tip">
Use the <code>RemovableHook</code> context manager (shown later in this chapter) to guarantee cleanup even if an exception is raised during the hooked computation.
</div>

---

## Hook Lifecycle

```python
import torch
import torch.nn as nn

model = nn.Linear(4, 2)

# Registration returns a RemovableHandle
handle = model.register_forward_hook(lambda m, i, o: print(o))

x = torch.randn(3, 4)
out = model(x)   # hook fires here

# Always clean up
handle.remove()

# After removal, hook no longer fires
out2 = model(x)  # silent
```

---

## Activation Extraction

One of the most common hook use-cases is extracting intermediate layer activations for visualization, probing classifiers, or nearest-neighbor retrieval.

```python
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, List, Optional


class ActivationExtractor:
    """
    Extracts intermediate activations from named layers using forward hooks.
    
    Usage:
        extractor = ActivationExtractor(model, layers=["layer1", "layer2.conv"])
        with extractor:
            output = model(x)
        activations = extractor.activations
    """

    def __init__(self, model: nn.Module, layers: Optional[List[str]] = None):
        self.model = model
        self.layers = layers
        self.activations: Dict[str, torch.Tensor] = OrderedDict()
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, name: str):
        def hook(module, input, output):
            # Detach to avoid holding the full computation graph
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach()
            elif isinstance(output, (tuple, list)):
                self.activations[name] = tuple(
                    t.detach() if isinstance(t, torch.Tensor) else t for t in output
                )
        return hook

    def register(self):
        """Register hooks on all target layers (or all modules if layers=None)."""
        self.activations.clear()
        for name, module in self.model.named_modules():
            if self.layers is None or name in self.layers:
                handle = module.register_forward_hook(self._make_hook(name))
                self._handles.append(handle)

    def remove(self):
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *args):
        self.remove()


# ── Example: ResNet feature extraction ──────────────────────────────────────
import torchvision.models as models

resnet = models.resnet50(weights=None)
target_layers = ["layer1", "layer2", "layer3", "layer4", "avgpool"]

x = torch.randn(2, 3, 224, 224)

with ActivationExtractor(resnet, layers=target_layers) as extractor:
    logits = resnet(x)

for layer_name, activation in extractor.activations.items():
    if isinstance(activation, torch.Tensor):
        print(f"{layer_name:20s}: {activation.shape}")
```

### Visualizing Activations

```python
import matplotlib.pyplot as plt
import numpy as np


def visualize_feature_maps(activations: torch.Tensor, layer_name: str, n_cols: int = 8):
    """Visualize the first batch item's feature maps from a conv layer."""
    # activations: (B, C, H, W)
    feat = activations[0].cpu().numpy()  # (C, H, W)
    n_maps = min(feat.shape[0], 64)
    n_rows = (n_maps + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    fig.suptitle(f"Feature maps — {layer_name}", fontsize=12)

    for idx in range(n_rows * n_cols):
        ax = axes[idx // n_cols][idx % n_cols]
        if idx < n_maps:
            ax.imshow(feat[idx], cmap="viridis")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"activations_{layer_name.replace('.', '_')}.png", dpi=150)
    plt.close()
```

---

## Gradient Analysis

Hooks on the backward pass enable per-layer gradient monitoring — essential for diagnosing vanishing or exploding gradients.

```python
import torch
import torch.nn as nn
from typing import Dict


class GradientMonitor:
    """
    Monitors per-layer gradient norms during backpropagation.
    Detects vanishing (norm < threshold) and exploding (norm > threshold) gradients.
    """

    def __init__(
        self,
        model: nn.Module,
        vanish_threshold: float = 1e-5,
        explode_threshold: float = 100.0,
    ):
        self.model = model
        self.vanish_threshold = vanish_threshold
        self.explode_threshold = explode_threshold
        self.grad_norms: Dict[str, float] = {}
        self._handles = []

    def _make_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                norm = grad_output[0].norm().item()
                self.grad_norms[name] = norm
                if norm < self.vanish_threshold:
                    print(f"⚠  Vanishing gradient in {name}: norm={norm:.2e}")
                elif norm > self.explode_threshold:
                    print(f"🔥 Exploding gradient in {name}: norm={norm:.2e}")
        return hook

    def register(self):
        self.grad_norms.clear()
        for name, module in self.model.named_modules():
            if len(list(module.parameters(recurse=False))) > 0:
                handle = module.register_full_backward_hook(self._make_hook(name))
                self._handles.append(handle)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def report(self):
        """Print a sorted summary of gradient norms."""
        print("\n── Gradient Norm Report ──────────────────")
        for name, norm in sorted(self.grad_norms.items(), key=lambda x: x[1]):
            bar = "█" * min(int(norm * 10), 50)
            print(f"  {name:35s} {norm:8.4f}  {bar}")
        print()

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *args):
        self.remove()


# ── Usage ────────────────────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 32),  nn.ReLU(),
    nn.Linear(32, 10),
)

monitor = GradientMonitor(model, vanish_threshold=1e-4, explode_threshold=10.0)

with monitor:
    x = torch.randn(16, 128)
    y = torch.randint(0, 10, (16,))
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()

monitor.report()
```

---

## Grad-CAM Implementation

Grad-CAM (Gradient-weighted Class Activation Mapping) uses both a **forward hook** (to capture feature maps) and a **backward hook** (to capture gradients) to produce visual explanations.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Registers forward and backward hooks on a target convolutional layer.
    After a forward+backward pass, computes a heatmap highlighting the
    spatial regions most influential for the predicted class.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = []

    def register(self):
        def save_features(module, input, output):
            self.feature_maps = output  # (B, C, H, W)

        def save_gradients(module, grad_input, grad_output):
            self.gradients = grad_output[0]  # (B, C, H, W)

        self._handles.append(
            self.target_layer.register_forward_hook(save_features)
        )
        self._handles.append(
            self.target_layer.register_full_backward_hook(save_gradients)
        )

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for input x.

        Returns:
            heatmap: (H, W) numpy array, values in [0, 1]
        """
        self.model.eval()
        self.register()

        try:
            # Forward pass
            logits = self.model(x)  # (B, num_classes)
            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()

            # Backward pass for the selected class
            self.model.zero_grad()
            score = logits[0, class_idx]
            score.backward()

            # Grad-CAM formula: global average pool gradients → weight each channel
            # gradients: (1, C, H, W)  feature_maps: (1, C, H, W)
            weights = self.gradients[0].mean(dim=(1, 2))  # (C,)
            cam = torch.zeros(self.feature_maps.shape[2:], device=x.device)
            for i, w in enumerate(weights):
                cam += w * self.feature_maps[0, i]

            cam = F.relu(cam)
            # Normalize to [0, 1]
            cam -= cam.min()
            if cam.max() > 0:
                cam /= cam.max()

            return cam.detach().cpu().numpy()

        finally:
            self.remove()

    def visualize(self, x: torch.Tensor, original_image: np.ndarray,
                  class_idx: Optional[int] = None, alpha: float = 0.5):
        import matplotlib.pyplot as plt
        import cv2

        heatmap = self(x, class_idx)
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]

        overlay = alpha * heatmap_colored + (1 - alpha) * (original_image / 255.0)
        overlay = np.clip(overlay, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(original_image)
        axes[0].set_title("Original")
        axes[1].imshow(heatmap_resized, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        return fig
```

---

## Model Instrumentation

### Layer Timing

Forward hooks can time every layer to locate inference bottlenecks:

```python
import time
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict


class LayerTimer:
    """Measure wall-clock time spent in each module's forward pass."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.timings: Dict[str, list] = defaultdict(list)
        self._handles = []
        self._start_times: Dict[str, float] = {}

    def _pre_hook(self, name):
        def hook(module, input):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._start_times[name] = time.perf_counter()
        return hook

    def _post_hook(self, name):
        def hook(module, input, output):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - self._start_times.get(name, 0)
            self.timings[name].append(elapsed * 1000)  # ms
        return hook

    def register(self):
        for name, module in self.model.named_modules():
            if name:  # skip root
                self._handles.append(module.register_forward_pre_hook(self._pre_hook(name)))
                self._handles.append(module.register_forward_hook(self._post_hook(name)))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def report(self, top_k: int = 10):
        import statistics
        print(f"\n── Top-{top_k} Slowest Layers (mean ms) ──────────────")
        ranked = sorted(
            [(n, statistics.mean(v)) for n, v in self.timings.items()],
            key=lambda x: -x[1]
        )[:top_k]
        for name, mean_ms in ranked:
            bar = "█" * min(int(mean_ms * 5), 40)
            print(f"  {name:40s} {mean_ms:7.3f} ms  {bar}")

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *args):
        self.remove()
```

### Memory Profiling Per Layer

```python
import torch
import torch.nn as nn
from typing import Dict


class MemoryProfiler:
    """Track peak GPU memory allocation introduced by each layer."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.memory_mb: Dict[str, float] = {}
        self._handles = []
        self._pre_mem: Dict[str, int] = {}

    def _pre_hook(self, name):
        def hook(module, input):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self._pre_mem[name] = torch.cuda.memory_allocated()
        return hook

    def _post_hook(self, name):
        def hook(module, input, output):
            if torch.cuda.is_available():
                post = torch.cuda.memory_allocated()
                delta = (post - self._pre_mem.get(name, post)) / 1e6
                self.memory_mb[name] = delta
        return hook

    def register(self):
        for name, module in self.model.named_modules():
            if name:
                self._handles.append(module.register_forward_pre_hook(self._pre_hook(name)))
                self._handles.append(module.register_forward_hook(self._post_hook(name)))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *args):
        self.remove()

    def report(self):
        print("\n── Memory Delta Per Layer ───────────────────")
        for name, mb in sorted(self.memory_mb.items(), key=lambda x: -x[1]):
            if abs(mb) > 0.01:
                print(f"  {name:40s} {mb:+8.2f} MB")
```

---

## Feature Injection and Activation Patching

Hooks that **return a modified output** allow you to patch activations mid-forward — useful for causal interventions in interpretability research, adversarial probing, and steering model behavior.

```python
import torch
import torch.nn as nn
from typing import Callable, Optional


class ActivationPatcher:
    """
    Replaces or perturbs a layer's output during the forward pass.
    
    Useful for:
    - Causal tracing (zero out one component, observe effect)
    - Concept activation vectors (steer toward a direction)
    - Ablation studies
    """

    def __init__(self, module: nn.Module, patch_fn: Callable[[torch.Tensor], torch.Tensor]):
        self.module = module
        self.patch_fn = patch_fn
        self._handle = None

    def _hook(self, module, input, output):
        if isinstance(output, torch.Tensor):
            return self.patch_fn(output)
        return output

    def __enter__(self):
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *args):
        if self._handle:
            self._handle.remove()
            self._handle = None


# ── Example: zero-ablate an attention head ───────────────────────────────────
def zero_head(head_idx: int, n_heads: int):
    """Return a patch function that zeros out one attention head."""
    def patch(output: torch.Tensor) -> torch.Tensor:
        # output: (B, seq_len, d_model) where d_model = n_heads * head_dim
        B, T, D = output.shape
        head_dim = D // n_heads
        out = output.clone()
        out[:, :, head_idx * head_dim:(head_idx + 1) * head_dim] = 0.0
        return out
    return patch


# ── Example: steer toward a concept activation vector ────────────────────────
def steer_toward(concept_vector: torch.Tensor, strength: float = 1.0):
    """Add a scaled concept vector to the activation."""
    direction = concept_vector / concept_vector.norm()
    def patch(output: torch.Tensor) -> torch.Tensor:
        return output + strength * direction.to(output.device)
    return patch
```

---

## `RemovableHook` Context Manager

A robust context manager that ensures hooks are always removed, even if an exception occurs:

```python
import torch.nn as nn
from typing import Callable, List, Optional, Union


class RemovableHook:
    """
    Context manager for safe hook lifecycle management.

    Supports registering multiple hooks across multiple modules
    and guarantees removal via __exit__ even on exceptions.

    Usage:
        with RemovableHook() as rh:
            rh.register_forward_hook(model.layer1, my_hook)
            rh.register_backward_hook(model.layer2, my_grad_hook)
            output = model(x)
        # All hooks removed here
    """

    def __init__(self):
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def register_forward_pre_hook(self, module: nn.Module, fn: Callable):
        handle = module.register_forward_pre_hook(fn)
        self._handles.append(handle)
        return handle

    def register_forward_hook(self, module: nn.Module, fn: Callable):
        handle = module.register_forward_hook(fn)
        self._handles.append(handle)
        return handle

    def register_backward_hook(self, module: nn.Module, fn: Callable):
        handle = module.register_full_backward_hook(fn)
        self._handles.append(handle)
        return handle

    def register_tensor_hook(self, tensor: torch.Tensor, fn: Callable):
        handle = tensor.register_hook(fn)
        self._handles.append(handle)
        return handle

    def remove_all(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_all()
        return False  # do not suppress exceptions


# ── Usage ────────────────────────────────────────────────────────────────────
import torch
model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

captured = {}

with RemovableHook() as rh:
    rh.register_forward_hook(model[0], lambda m, i, o: captured.update({"fc1": o.detach()}))
    rh.register_forward_hook(model[2], lambda m, i, o: captured.update({"fc2": o.detach()}))
    out = model(torch.randn(4, 10))

print("Captured:", {k: v.shape for k, v in captured.items()})
# All hooks removed automatically
```

---

## Hook-Based Probing for Interpretability

Probing classifiers test whether a linear classifier trained on intermediate activations can predict a linguistic or semantic property — revealing what information is encoded at each layer.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple


class LinearProbe(nn.Module):
    """Simple linear classifier for probing layer representations."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def extract_representations(
    model: nn.Module,
    layer_name: str,
    dataloader: DataLoader,
    device: torch.device,
    pool: str = "mean",  # "mean" | "cls" | "flatten"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract representations from a named layer across a dataset.

    Returns:
        representations: (N, D)
        labels: (N,)
    """
    all_reps, all_labels = [], []
    captured = {}

    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            captured["rep"] = output.detach()

    # Find and register hook on the named layer
    handle = None
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook)
            break

    if handle is None:
        raise ValueError(f"Layer '{layer_name}' not found in model.")

    model.eval()
    try:
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                model(batch_x)
                rep = captured["rep"]
                if rep.dim() == 3:  # (B, T, D)
                    if pool == "mean":
                        rep = rep.mean(dim=1)
                    elif pool == "cls":
                        rep = rep[:, 0, :]
                    else:
                        rep = rep.flatten(1)
                elif rep.dim() == 4:  # (B, C, H, W)
                    rep = rep.mean(dim=(2, 3))  # global average pool
                all_reps.append(rep.cpu())
                all_labels.append(batch_y)
    finally:
        handle.remove()

    return torch.cat(all_reps), torch.cat(all_labels)


def train_probe(representations: torch.Tensor, labels: torch.Tensor,
                num_classes: int, epochs: int = 30) -> Dict[str, float]:
    """Train a linear probe and return accuracy."""
    dataset = TensorDataset(representations, labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    probe = LinearProbe(representations.shape[1], num_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            criterion(probe(x), y).backward()
            optimizer.step()

    # Evaluate
    probe.eval()
    with torch.no_grad():
        logits = probe(representations)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()

    return {"accuracy": acc, "num_samples": len(labels), "dim": representations.shape[1]}
```

---

## Hook Comparison Table

| Hook Type | Fires When | Can Modify? | Use Cases |
|---|---|---|---|
| `register_forward_pre_hook` | Before `forward()` | Input tuple | Input normalization, conditional skip |
| `register_forward_hook` | After `forward()` | Output tensor | Activation capture, feature injection, timing |
| `register_full_backward_hook` | During `backward()` | Grad input | Gradient monitoring, gradient clipping per-layer |
| `register_hook` (tensor) | During `backward()` on that tensor | Gradient | Gradient logging, custom gradient flows |
| `register_state_dict_hook` | During `state_dict()` | State dict | Key renaming for compatibility |
| `register_load_state_dict_post_hook` | After `load_state_dict()` | N/A | Post-load initialization |

<div class="callout info">
<strong>Tensor hooks vs module hooks:</strong> <code>tensor.register_hook(fn)</code> fires when that specific tensor's gradient is computed, while module hooks fire for all parameters in the module. Tensor hooks are useful for inspecting gradients of intermediate activations (not just parameters).
</div>

---

## When to Use Each Hook

<div class="diagram">
  <div class="diagram-title">Hook Selection Guide</div>
  <div class="diagram-grid cols-3">
    <div class="diagram-card accent">
      <div class="card-icon">⬆️</div>
      <div class="card-title">forward_pre_hook</div>
      <div class="card-desc">Modify inputs before forward; inject noise; conditional routing; input logging</div>
    </div>
    <div class="diagram-card green">
      <div class="card-icon">📸</div>
      <div class="card-title">forward_hook</div>
      <div class="card-desc">Capture activations; apply post-processing; timing; memory profiling; feature injection</div>
    </div>
    <div class="diagram-card purple">
      <div class="card-icon">⬇️</div>
      <div class="card-title">backward_hook</div>
      <div class="card-desc">Monitor gradient norms; detect vanishing/exploding; per-layer gradient clipping; Grad-CAM</div>
    </div>
  </div>
</div>

---

## Summary

PyTorch hooks bring the Observer pattern to tensor computation graphs. They enable powerful instrumentation without touching model code:

- **Activation extraction** for visualization and probing
- **Gradient monitoring** to diagnose training pathologies
- **Grad-CAM** for interpretability heatmaps
- **Layer timing and memory profiling** for performance optimization
- **Activation patching** for causal interventions

The key discipline: **always remove hooks**. Use the `RemovableHook` context manager or `handle.remove()` in a `finally` block to prevent memory leaks.

*Last updated: May 2026*
