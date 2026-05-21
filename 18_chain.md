---
title: "Chapter 18 — Chain of Responsibility & Transform Pipelines"
---

[← Back to Table of Contents](./README.md)

# Chapter 18 — Chain of Responsibility & Transform Pipelines

> *"I'll handle it if I can; otherwise I'll pass it on. Someone in the chain will know what to do."*

Data rarely arrives in a model-ready form. Images need resizing and normalisation. Text needs cleaning, tokenisation, truncation and padding. Inference outputs need post-processing and decoding. The **Chain of Responsibility** pattern captures this reality: a sequence of handlers, each responsible for one transformation, organised so that data flows through them in order — or stops early when a handler rejects it.

<span class="badge structural">Structural</span>

---

## Intent

Avoid coupling the sender of a request to its receiver by giving **more than one object a chance to handle the request**. Chain the receiving objects and pass the request along the chain until an object handles it.

---

## UML Structure

<div class="diagram">
  <div class="diagram-title">Chain of Responsibility — Class Structure</div>
  <div class="uml-row">
    <div class="uml-box">
      <div class="uml-title">«abstract» Handler</div>
      <div class="uml-section">
        <div class="uml-item">– _next: Handler | None</div>
      </div>
      <div class="uml-section">
        <div class="uml-item">+ set_next(h: Handler) → Handler</div>
        <div class="uml-item">+ handle(request) → Any</div>
      </div>
    </div>
  </div>
  <div class="uml-row">
    <div class="uml-box">
      <div class="uml-title">ConcreteHandlerA</div>
      <div class="uml-section">
        <div class="uml-item">+ handle(request) → Any</div>
        <div class="uml-item"># can_handle(req) → bool</div>
      </div>
    </div>
    <div class="uml-box">
      <div class="uml-title">ConcreteHandlerB</div>
      <div class="uml-section">
        <div class="uml-item">+ handle(request) → Any</div>
        <div class="uml-item"># can_handle(req) → bool</div>
      </div>
    </div>
  </div>
</div>

The key design decision: when a handler *cannot* process the request, it calls `self._next.handle(request)` to pass it along. If no handler claims the request, it falls off the end of the chain (or raises an exception, depending on the contract).

---

## `torchvision.transforms.Compose` as Chain of Responsibility

`torchvision.transforms.Compose` is the most-used Chain of Responsibility in deep learning. Each transform is a handler; `Compose` passes the tensor/PIL image through them in sequence.

```python
from torchvision import transforms

# Standard ImageNet preprocessing chain
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),           # Handler 1
    transforms.RandomHorizontalFlip(p=0.5),      # Handler 2
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1), # Handler 3
    transforms.ToTensor(),                        # Handler 4
    transforms.Normalize(                         # Handler 5
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

Under the hood, `Compose.__call__` is simply:

```python
def __call__(self, img):
    for t in self.transforms:
        img = t(img)      # each handler transforms and passes on
    return img
```

---

## Building a Generic Processing Pipeline

A more expressive pipeline with metadata, error handling, and optional short-circuiting.

```python
from __future__ import annotations
import abc
import time
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class PipelineContext:
    """Carries data and metadata through the pipeline."""
    data: Any
    metadata: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    skipped_steps: list[str] = field(default_factory=list)
    should_stop: bool = False  # short-circuit flag


class PipelineStep(abc.ABC):
    """Abstract handler in the chain."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__
        self._next: "PipelineStep | None" = None

    def set_next(self, step: "PipelineStep") -> "PipelineStep":
        """Fluent API: returns next step to allow chaining."""
        self._next = step
        return step

    def handle(self, ctx: PipelineContext) -> PipelineContext:
        """Template: run this step, then pass to next (unless stopped)."""
        if ctx.should_stop:
            ctx.skipped_steps.append(self.name)
            if self._next:
                return self._next.handle(ctx)
            return ctx
        try:
            ctx = self.process(ctx)
        except Exception as exc:
            ctx.errors.append(f"{self.name}: {exc}")
            ctx = self.on_error(ctx, exc)
        if not ctx.should_stop and self._next:
            return self._next.handle(ctx)
        return ctx

    @abc.abstractmethod
    def process(self, ctx: PipelineContext) -> PipelineContext:
        """Apply this step's transformation."""

    def on_error(
        self, ctx: PipelineContext, exc: Exception
    ) -> PipelineContext:
        """Default error handler — stop the pipeline."""
        ctx.should_stop = True
        return ctx


class Pipeline:
    """Builds and runs a linear chain of steps."""

    def __init__(self) -> None:
        self._head: PipelineStep | None = None
        self._tail: PipelineStep | None = None
        self._steps: list[PipelineStep] = []

    def add_step(self, step: PipelineStep) -> "Pipeline":
        self._steps.append(step)
        if self._head is None:
            self._head = step
            self._tail = step
        else:
            self._tail.set_next(step)  # type: ignore[union-attr]
            self._tail = step
        return self  # fluent API

    def run(self, data: Any, **metadata) -> PipelineContext:
        if self._head is None:
            raise RuntimeError("Pipeline has no steps.")
        ctx = PipelineContext(data=data, metadata=metadata)
        t0 = time.perf_counter()
        result = self._head.handle(ctx)
        result.metadata["pipeline_time_ms"] = (time.perf_counter() - t0) * 1000
        return result

    def __repr__(self) -> str:
        names = " → ".join(s.name for s in self._steps)
        return f"Pipeline([{names}])"
```

---

## Text Preprocessing Chain for NLP

```python
import re
import torch


class TextCleanStep(PipelineStep):
    """Lowercase, strip, collapse whitespace."""

    def process(self, ctx: PipelineContext) -> PipelineContext:
        text: str = ctx.data
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        ctx.data = re.sub(r"\s+", " ", text)
        return ctx


class TokenizeStep(PipelineStep):
    """Tokenise using a HuggingFace tokeniser."""

    def __init__(self, tokenizer, **kwargs) -> None:
        super().__init__("Tokenize")
        self.tokenizer = tokenizer
        self.kwargs = kwargs

    def process(self, ctx: PipelineContext) -> PipelineContext:
        ctx.data = self.tokenizer(ctx.data, **self.kwargs)
        return ctx


class TruncateStep(PipelineStep):
    """Truncate token ids to max_length."""

    def __init__(self, max_length: int = 512) -> None:
        super().__init__("Truncate")
        self.max_length = max_length

    def process(self, ctx: PipelineContext) -> PipelineContext:
        ids = ctx.data["input_ids"]
        if len(ids) > self.max_length:
            ctx.data["input_ids"] = ids[: self.max_length]
            ctx.metadata["truncated"] = True
        return ctx


class PadStep(PipelineStep):
    """Pad token ids to a fixed length."""

    def __init__(self, pad_length: int = 512, pad_token_id: int = 0) -> None:
        super().__init__("Pad")
        self.pad_length = pad_length
        self.pad_token_id = pad_token_id

    def process(self, ctx: PipelineContext) -> PipelineContext:
        ids = ctx.data["input_ids"]
        padding = [self.pad_token_id] * (self.pad_length - len(ids))
        ctx.data["input_ids"] = ids + padding
        ctx.data["attention_mask"] = [1] * len(ids) + [0] * len(padding)
        return ctx


class TensorizeStep(PipelineStep):
    """Convert lists to PyTorch tensors."""

    def process(self, ctx: PipelineContext) -> PipelineContext:
        ctx.data = {
            k: torch.tensor(v, dtype=torch.long)
            for k, v in ctx.data.items()
            if isinstance(v, list)
        }
        return ctx


# Build the NLP pipeline
def build_nlp_pipeline(tokenizer, max_length: int = 128) -> Pipeline:
    return (
        Pipeline()
        .add_step(TextCleanStep())
        .add_step(TokenizeStep(tokenizer, add_special_tokens=True))
        .add_step(TruncateStep(max_length=max_length))
        .add_step(PadStep(pad_length=max_length))
        .add_step(TensorizeStep())
    )
```

---

## Model Inference Chain

A chain that takes raw input, runs inference, and returns decoded output.

```python
import numpy as np


class InputValidationStep(PipelineStep):
    """Validate and gate invalid inference requests."""

    def __init__(self, max_size_mb: float = 10.0) -> None:
        super().__init__("InputValidation")
        self.max_bytes = max_size_mb * 1024 * 1024

    def process(self, ctx: PipelineContext) -> PipelineContext:
        data = ctx.data
        size = data.nbytes if hasattr(data, "nbytes") else len(str(data))
        if size > self.max_bytes:
            raise ValueError(
                f"Input size {size / 1024 / 1024:.1f} MB exceeds limit."
            )
        return ctx


class PreprocessStep(PipelineStep):
    """Apply model-specific preprocessing."""

    def __init__(self, transform) -> None:
        super().__init__("Preprocess")
        self.transform = transform

    def process(self, ctx: PipelineContext) -> PipelineContext:
        ctx.data = self.transform(ctx.data)
        if ctx.data.ndim == 3:
            ctx.data = ctx.data.unsqueeze(0)  # add batch dimension
        return ctx


class ForwardStep(PipelineStep):
    """Run the model forward pass."""

    def __init__(self, model: torch.nn.Module, device: str = "cpu") -> None:
        super().__init__("Forward")
        self.model = model.to(device)
        self.device = device

    def process(self, ctx: PipelineContext) -> PipelineContext:
        x = ctx.data.to(self.device)
        self.model.eval()
        with torch.no_grad():
            ctx.data = self.model(x)
        return ctx


class PostprocessStep(PipelineStep):
    """Apply softmax and format raw logits."""

    def process(self, ctx: PipelineContext) -> PipelineContext:
        logits = ctx.data
        ctx.data = torch.softmax(logits, dim=-1)
        ctx.metadata["confidence"] = ctx.data.max().item()
        return ctx


class DecodeStep(PipelineStep):
    """Map class probabilities to human-readable label."""

    def __init__(self, id2label: dict[int, str]) -> None:
        super().__init__("Decode")
        self.id2label = id2label

    def process(self, ctx: PipelineContext) -> PipelineContext:
        probs = ctx.data.squeeze(0)
        class_idx = probs.argmax().item()
        ctx.data = {
            "label": self.id2label.get(class_idx, str(class_idx)),
            "confidence": probs[class_idx].item(),
            "top5": [
                (self.id2label.get(i, str(i)), p.item())
                for i, p in sorted(
                    enumerate(probs.tolist()),
                    key=lambda x: -x[1]
                )[:5]
            ],
        }
        return ctx
```

---

## Middleware Stack for ML Serving

Production inference servers use a middleware chain (WSGI/ASGI style) to wrap every request with cross-cutting concerns.

```python
from __future__ import annotations
from typing import Callable
import functools

# Type alias: middleware is a function that wraps a handler
Handler = Callable[[dict], dict]
Middleware = Callable[[Handler], Handler]


def auth_middleware(handler: Handler) -> Handler:
    """Reject requests without a valid API key."""
    @functools.wraps(handler)
    def wrapped(request: dict) -> dict:
        if request.get("api_key") != "valid-key-123":
            return {"error": "Unauthorized", "status": 401}
        return handler(request)
    return wrapped


def rate_limit_middleware(max_rpm: int = 60) -> Middleware:
    """Sliding-window rate limiter (simplified)."""
    import collections, time
    window: collections.deque[float] = collections.deque()

    def middleware(handler: Handler) -> Handler:
        @functools.wraps(handler)
        def wrapped(request: dict) -> dict:
            now = time.time()
            cutoff = now - 60.0
            while window and window[0] < cutoff:
                window.popleft()
            if len(window) >= max_rpm:
                return {"error": "Rate limit exceeded", "status": 429}
            window.append(now)
            return handler(request)
        return wrapped
    return middleware


def input_validation_middleware(schema: dict) -> Middleware:
    """Validate required fields are present."""
    def middleware(handler: Handler) -> Handler:
        @functools.wraps(handler)
        def wrapped(request: dict) -> dict:
            for field_name, field_type in schema.items():
                if field_name not in request:
                    return {"error": f"Missing field: {field_name}", "status": 400}
                if not isinstance(request[field_name], field_type):
                    return {
                        "error": f"Field '{field_name}' must be {field_type.__name__}",
                        "status": 422,
                    }
            return handler(request)
        return wrapped
    return middleware


def logging_middleware(handler: Handler) -> Handler:
    """Log request/response timing."""
    import time
    @functools.wraps(handler)
    def wrapped(request: dict) -> dict:
        t0 = time.perf_counter()
        response = handler(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[Middleware] {request.get('path', '/')} → "
              f"status={response.get('status', 200)} ({elapsed_ms:.1f}ms)")
        return response
    return wrapped


def build_inference_app(model_fn: Handler) -> Handler:
    """Compose middleware stack around the core inference handler."""
    app = model_fn
    # Applied in reverse — outermost wraps first
    app = input_validation_middleware({"input": list})(app)
    app = rate_limit_middleware(max_rpm=100)(app)
    app = auth_middleware(app)
    app = logging_middleware(app)
    return app


# Core inference handler
def inference_handler(request: dict) -> dict:
    predictions = [0.9, 0.1]  # placeholder
    return {"predictions": predictions, "status": 200}


serve = build_inference_app(inference_handler)

response = serve({"api_key": "valid-key-123", "input": [1.0, 2.0, 3.0], "path": "/predict"})
print(response)
```

---

## Short-Circuit Chains — Early Exit

```python
class ShortCircuitValidationStep(PipelineStep):
    """Stop the chain immediately if input quality is too low."""

    def __init__(self, min_tokens: int = 3) -> None:
        super().__init__("QualityGate")
        self.min_tokens = min_tokens

    def process(self, ctx: PipelineContext) -> PipelineContext:
        text: str = ctx.data
        token_count = len(text.split())
        if token_count < self.min_tokens:
            ctx.should_stop = True
            ctx.metadata["rejection_reason"] = (
                f"Too short: {token_count} tokens < {self.min_tokens}"
            )
        return ctx


class LanguageFilterStep(PipelineStep):
    """Reject non-English text."""

    def process(self, ctx: PipelineContext) -> PipelineContext:
        # Simplified: use langdetect in practice
        text: str = ctx.data
        non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / max(len(text), 1)
        if non_ascii_ratio > 0.5:
            ctx.should_stop = True
            ctx.metadata["rejection_reason"] = "Non-English content detected"
        return ctx
```

---

## Priority-Ordered Handler Chains

```python
class PrioritizedPipeline(Pipeline):
    """Handlers sorted by priority before execution."""

    def add_step(self, step: PipelineStep, priority: int = 0) -> "PrioritizedPipeline":
        self._steps.append((priority, step))  # type: ignore[arg-type]
        return self  # type: ignore[return-value]

    def build(self) -> "PrioritizedPipeline":
        sorted_steps = sorted(self._steps, key=lambda x: x[0], reverse=True)
        self._steps = [s for _, s in sorted_steps]
        # Re-chain
        self._head = self._steps[0] if self._steps else None
        for a, b in zip(self._steps, self._steps[1:]):
            a.set_next(b)
        self._tail = self._steps[-1] if self._steps else None
        return self
```

---

## Albumentations Augmentation Pipeline

Albumentations is a high-performance Chain of Responsibility for image augmentation.

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Build an augmentation chain
train_aug = A.Compose([
    A.RandomResizedCrop(height=224, width=224, scale=(0.08, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
    A.GaussianBlur(blur_limit=(3, 7), p=0.1),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_aug = A.Compose([
    A.Resize(height=256, width=256),
    A.CenterCrop(height=224, width=224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# With bounding boxes — Compose threads bbox info through the chain
bbox_aug = A.Compose(
    [
        A.RandomResizedCrop(height=512, width=512),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
)

# Apply
image = np.zeros((480, 640, 3), dtype=np.uint8)
bboxes = [[100, 100, 300, 300]]
labels = [1]
result = bbox_aug(image=image, bboxes=bboxes, labels=labels)
augmented_image = result["image"]
augmented_bboxes = result["bboxes"]
```

Albumentations' `Compose` also supports:
- `KeypointParams` for keypoint augmentation threading
- `ReplayCompose` for deterministic replay of augmentation sequences
- `OneOf` for randomly selecting one transform from a group

---

## `torchvision.transforms.v2` Pipeline

```python
import torchvision.transforms.v2 as T

# v2 works on tensors natively — no PIL round-trip
transform = T.Compose([
    T.RandomResizedCrop(224, antialias=True),
    T.RandomHorizontalFlip(p=0.5),
    T.TrivialAugmentWide(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.1),
])

# MixUp / CutMix as a probabilistic step in the chain
cutmix = T.CutMix(num_classes=1000)
mixup = T.MixUp(num_classes=1000)
cutmix_or_mixup = T.RandomChoice([cutmix, mixup])
```

---

## Flow Diagram

<div class="diagram">
  <div class="diagram-title">Processing Chain with Short-Circuit</div>
  <div class="flow">
    <div class="flow-node accent wide">Raw Input</div>
    <div class="flow-arrow accent">▼</div>
    <div class="flow-node green wide">Step 1: Validate / Quality Gate</div>
    <div class="flow-arrow purple">▼ pass  &nbsp;&nbsp;&nbsp; ╰── reject → Stop ✕</div>
    <div class="flow-node blue wide">Step 2: Clean / Normalise</div>
    <div class="flow-arrow">▼</div>
    <div class="flow-node teal wide">Step 3: Transform / Featurise</div>
    <div class="flow-arrow">▼</div>
    <div class="flow-node orange wide">Step 4: Tensorize / Encode</div>
    <div class="flow-arrow accent">▼</div>
    <div class="flow-node accent wide">Model-Ready Output ✓</div>
  </div>
</div>

---

## Comparison Table

<table class="compare-table">
  <thead>
    <tr>
      <th>Dimension</th>
      <th>Chain of Responsibility</th>
      <th>Pipeline (linear)</th>
      <th>Strategy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Handler coupling</td>
      <td>Linked list — each knows next</td>
      <td>Managed externally</td>
      <td>No chaining</td>
    </tr>
    <tr>
      <td>Short-circuit</td>
      <td>Yes — handler stops chain</td>
      <td>Only with explicit flag</td>
      <td>No</td>
    </tr>
    <tr>
      <td>Branching</td>
      <td>Possible per handler</td>
      <td>Usually linear</td>
      <td>No</td>
    </tr>
    <tr>
      <td>Order matters</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
    </tr>
    <tr>
      <td>ML example</td>
      <td>transforms.Compose, middleware</td>
      <td>Albumentations, sklearn Pipeline</td>
      <td>Loss function selection</td>
    </tr>
  </tbody>
</table>

---

## Key Takeaways

<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">🔗</div>
    <div class="card-title">Single Responsibility</div>
    <div class="card-desc">Each handler in a chain does exactly one thing — clean, tokenise, normalise. This makes individual steps testable and reusable.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">⛔</div>
    <div class="card-title">Short-Circuit Early</div>
    <div class="card-desc">Place validation and quality-gate steps at the front of the chain. Failing fast saves computation on downstream, expensive steps.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🔄</div>
    <div class="card-title">Compose, Don't Inherit</div>
    <div class="card-desc">Build pipelines by composing small, focused handlers rather than creating monolithic preprocessing classes that are hard to test in isolation.</div>
  </div>
</div>

---

*Last updated: May 2026*
