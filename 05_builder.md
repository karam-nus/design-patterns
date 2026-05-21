---
title: "Chapter 5 — Builder Pattern"
---

[← Back to Table of Contents](./README.md)

# Chapter 5 — Builder Pattern

> *"The Builder pattern separates the construction of a complex object from its representation so that the same construction process can create different representations."*
> — Erich Gamma et al., *Design Patterns*, 1994

---

## Intent and Motivation

The Builder pattern solves a specific and common problem: how do you construct a complex object step-by-step when the object has many optional components, the construction process is non-trivial, and different configurations should produce different but structurally valid results?

The Builder separates two concerns:
1. **What** is being built (the **Product**) — the final, fully configured object.
2. **How** it is built (the **Builder**) — the step-by-step construction process, validated and assembled piece by piece.

A **Director** (optional but common) orchestrates the Builder, defining which steps to call in which order to produce a specific "standard" product configuration.

### Why This Matters in ML

ML training pipelines are textbook Builder use cases. A complete training run requires:
- A model (with its own nested configuration)
- An optimizer (with learning rate, weight decay, betas)
- A learning rate scheduler (with warmup, decay type, milestones)
- One or more data loaders (train, val, test)
- A set of callbacks (checkpointing, early stopping, logging, profiling)
- Mixed precision settings, gradient clipping, gradient accumulation

This is far too many arguments for a single constructor. More importantly, many of these components are optional, their defaults depend on other components, and validation is only possible once all pieces are assembled (you can't validate the scheduler warmup until you know the number of training steps, which depends on the dataset size).

The Builder pattern handles this complexity elegantly.

---

## The Telescoping Constructor Problem

Before exploring the solution, let's see the problem clearly.

```python
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any


# ── The Telescoping Constructor Anti-Pattern ──────────────────────────────────

class Trainer:
    """
    Anti-pattern: constructor with too many parameters.
    Adding any new option requires changing the constructor signature,
    all call sites, and all documentation.
    This is sometimes called the "telescoping constructor" problem.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        train_loader: Any,
        val_loader: Any | None = None,
        scheduler: Any | None = None,
        gradient_clip: float = 1.0,
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = False,
        early_stopping_patience: int | None = None,
        checkpoint_dir: str | None = None,
        checkpoint_every_n_steps: int | None = None,
        log_every_n_steps: int = 10,
        max_epochs: int = 100,
        max_steps: int | None = None,
        device: str = "cuda",
        seed: int = 42,
        callbacks: list | None = None,
        compile_model: bool = False,
        ema_decay: float | None = None,
        warmup_steps: int = 0,
        # ... and it keeps growing
    ) -> None:
        # Impossible to validate coherently without seeing all parameters together
        if scheduler is None and warmup_steps > 0:
            raise ValueError("warmup_steps > 0 requires a scheduler")
        if checkpoint_every_n_steps is not None and checkpoint_dir is None:
            raise ValueError("checkpoint_every_n_steps requires checkpoint_dir")
        # ... more cross-parameter validation
        self.model = model
        self.optimizer = optimizer
        # ... 15 more assignments


# ── At the call site, this becomes unreadable ─────────────────────────────────

# trainer = Trainer(
#     model=model,
#     optimizer=optimizer,
#     loss_fn=loss_fn,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     scheduler=scheduler,
#     gradient_clip=1.0,
#     gradient_accumulation_steps=2,
#     mixed_precision=True,
#     early_stopping_patience=10,
#     checkpoint_dir="./checkpoints",
#     checkpoint_every_n_steps=500,
#     log_every_n_steps=20,
#     max_epochs=50,
#     device="cuda",
#     seed=123,
#     # ... are we forgetting any arguments?
# )
```

The telescoping constructor has four serious problems:
1. **Unreadable call sites**: a 15-argument constructor call is incomprehensible without looking at the signature.
2. **Error-prone defaults**: it's easy to forget an argument or confuse positional ones.
3. **Impossible validation**: cross-parameter invariants are hard to check because all parameters arrive simultaneously.
4. **Rigidity**: adding a new parameter is a breaking change.

---

## UML Structure

<div class="diagram">
<div class="diagram-title">Builder Pattern — Structure</div>
<div class="uml-row">
  <div class="uml-box">
    <div class="uml-title">Director</div>
    <div class="uml-section">
      <div class="uml-item">- _builder: Builder</div>
    </div>
    <div class="uml-section">
      <div class="uml-item">+ construct() → Product</div>
      <div class="uml-item">+ construct_minimal() → Product</div>
    </div>
  </div>
  <div class="uml-box">
    <div class="uml-title">«abstract» Builder</div>
    <div class="uml-section">
      <div class="uml-item">+ set_model() → Builder</div>
      <div class="uml-item">+ set_optimizer() → Builder</div>
      <div class="uml-item">+ set_scheduler() → Builder</div>
      <div class="uml-item">+ add_callback() → Builder</div>
      <div class="uml-item">+ build() → Product</div>
    </div>
  </div>
</div>
<div class="uml-row">
  <div class="uml-box">
    <div class="uml-title">TrainingPipelineBuilder</div>
    <div class="uml-section">
      <div class="uml-item">- _model: nn.Module | None</div>
      <div class="uml-item">- _optimizer: Optimizer | None</div>
      <div class="uml-item">- _scheduler: Scheduler | None</div>
      <div class="uml-item">- _callbacks: list[Callback]</div>
    </div>
    <div class="uml-section">
      <div class="uml-item">+ set_model() → Self</div>
      <div class="uml-item">+ set_optimizer() → Self</div>
      <div class="uml-item">+ add_callback() → Self</div>
      <div class="uml-item">+ build() → TrainingPipeline</div>
    </div>
  </div>
  <div class="uml-box">
    <div class="uml-title">TrainingPipeline (Product)</div>
    <div class="uml-section">
      <div class="uml-item">+ model: nn.Module</div>
      <div class="uml-item">+ optimizer: Optimizer</div>
      <div class="uml-item">+ scheduler: Scheduler</div>
      <div class="uml-item">+ train_loader: DataLoader</div>
      <div class="uml-item">+ callbacks: list[Callback]</div>
    </div>
    <div class="uml-section">
      <div class="uml-item">+ run(epochs: int) → None</div>
    </div>
  </div>
</div>
</div>

---

## Builder for ML: Training Pipeline Builder

```python
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Self

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ── Product ───────────────────────────────────────────────────────────────────

@dataclass
class TrainingPipeline:
    """
    The Product: a fully assembled, validated training pipeline.
    All fields are required — there is no such thing as a partially-built Pipeline.
    Construction is done exclusively through the Builder.
    """
    model: nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader | None
    scheduler: Any | None
    callbacks: list[Callable]
    gradient_clip: float
    gradient_accumulation_steps: int
    mixed_precision: bool
    max_epochs: int
    device: torch.device

    def run(self) -> None:
        """Execute the training loop."""
        self.model.to(self.device)
        scaler = torch.cuda.amp.GradScaler() if self.mixed_precision and self.device.type == "cuda" else None

        for epoch in range(self.max_epochs):
            self.model.train()
            accumulated_loss = 0.0

            for step, batch in enumerate(self.train_loader):
                # Gradient accumulation
                context = torch.cuda.amp.autocast() if scaler else torch.no_grad.__class__()
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    loss = loss / self.gradient_accumulation_steps

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                accumulated_loss += loss.item()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.gradient_clip > 0:
                        if scaler:
                            scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                    if scaler:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Fire step callbacks
                    for cb in self.callbacks:
                        if hasattr(cb, "on_step_end"):
                            cb.on_step_end(step=step, loss=accumulated_loss)
                    accumulated_loss = 0.0

            if self.scheduler:
                self.scheduler.step()

            # Fire epoch callbacks
            for cb in self.callbacks:
                if hasattr(cb, "on_epoch_end"):
                    cb.on_epoch_end(epoch=epoch, model=self.model)

        print("Training complete.")


# ── Builder ───────────────────────────────────────────────────────────────────

class TrainingPipelineBuilder:
    """
    Builder for TrainingPipeline.

    Enforces step-by-step construction with validation at each step.
    Call `build()` only when all required components are set.

    Example::

        pipeline = (
            TrainingPipelineBuilder()
            .set_model(model)
            .set_optimizer(torch.optim.AdamW, lr=3e-4)
            .set_loss(nn.CrossEntropyLoss())
            .set_train_loader(train_loader)
            .set_val_loader(val_loader)
            .set_scheduler(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=100)
            .add_callback(checkpoint_cb)
            .set_max_epochs(50)
            .enable_mixed_precision()
            .set_gradient_clip(1.0)
            .build()
        )
    """

    def __init__(self) -> None:
        # Required components
        self._model: nn.Module | None = None
        self._loss_fn: nn.Module | None = None
        self._train_loader: DataLoader | None = None

        # Optional components
        self._val_loader: DataLoader | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: Any | None = None
        self._callbacks: list[Callable] = []

        # Hyperparameters with defaults
        self._gradient_clip: float = 1.0
        self._gradient_accumulation_steps: int = 1
        self._mixed_precision: bool = False
        self._max_epochs: int = 100
        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # ── Required setters ──────────────────────────────────────────────────────

    def set_model(self, model: nn.Module) -> Self:
        """Set the model to train. Required."""
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model)}")
        self._model = model
        return self

    def set_loss(self, loss_fn: nn.Module) -> Self:
        """Set the loss function. Required."""
        self._loss_fn = loss_fn
        return self

    def set_train_loader(self, loader: DataLoader) -> Self:
        """Set the training data loader. Required."""
        self._train_loader = loader
        return self

    # ── Optional setters with validation ─────────────────────────────────────

    def set_optimizer(
        self,
        optimizer_cls: type,
        **kwargs: Any,
    ) -> Self:
        """
        Set the optimizer. If not called, defaults to AdamW with lr=1e-3.
        Must be called after set_model().
        """
        if self._model is None:
            raise RuntimeError(
                "set_model() must be called before set_optimizer()"
            )
        self._optimizer = optimizer_cls(self._model.parameters(), **kwargs)
        return self

    def set_val_loader(self, loader: DataLoader) -> Self:
        self._val_loader = loader
        return self

    def set_scheduler(
        self,
        scheduler_cls: type,
        **kwargs: Any,
    ) -> Self:
        """
        Set the learning rate scheduler.
        Must be called after set_optimizer().
        """
        if self._optimizer is None:
            raise RuntimeError(
                "set_optimizer() must be called before set_scheduler()"
            )
        self._scheduler = scheduler_cls(self._optimizer, **kwargs)
        return self

    def add_callback(self, callback: Callable) -> Self:
        """Add a training callback. Can be called multiple times."""
        self._callbacks.append(callback)
        return self

    # ── Hyperparameter setters ────────────────────────────────────────────────

    def set_gradient_clip(self, max_norm: float) -> Self:
        if max_norm <= 0:
            raise ValueError(f"gradient_clip must be positive, got {max_norm}")
        self._gradient_clip = max_norm
        return self

    def set_gradient_accumulation_steps(self, steps: int) -> Self:
        if steps < 1:
            raise ValueError(f"gradient_accumulation_steps must be >= 1, got {steps}")
        self._gradient_accumulation_steps = steps
        return self

    def enable_mixed_precision(self) -> Self:
        if not torch.cuda.is_available():
            import warnings
            warnings.warn("Mixed precision requested but CUDA is not available; ignoring.")
        else:
            self._mixed_precision = True
        return self

    def set_max_epochs(self, epochs: int) -> Self:
        if epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {epochs}")
        self._max_epochs = epochs
        return self

    def set_device(self, device: str | torch.device) -> Self:
        self._device = torch.device(device)
        return self

    # ── Build and validation ──────────────────────────────────────────────────

    def _validate(self) -> None:
        """Validate that all required components are set before building."""
        missing: list[str] = []
        if self._model is None:
            missing.append("model (call set_model())")
        if self._loss_fn is None:
            missing.append("loss_fn (call set_loss())")
        if self._train_loader is None:
            missing.append("train_loader (call set_train_loader())")
        if missing:
            raise RuntimeError(
                "Cannot build TrainingPipeline: missing required components:\n"
                + "\n".join(f"  - {m}" for m in missing)
            )

    def build(self) -> TrainingPipeline:
        """
        Validate all components and assemble the final TrainingPipeline.

        Raises:
            RuntimeError: If required components are missing.
        """
        self._validate()

        # Apply default optimizer if not set
        if self._optimizer is None:
            if self._model is None:
                raise RuntimeError("set_model() must be called before build()")
            self._optimizer = torch.optim.AdamW(
                self._model.parameters(), lr=1e-3, weight_decay=1e-4
            )

        return TrainingPipeline(
            model=self._model,  # type: ignore[arg-type]
            optimizer=self._optimizer,
            loss_fn=self._loss_fn,  # type: ignore[arg-type]
            train_loader=self._train_loader,  # type: ignore[arg-type]
            val_loader=self._val_loader,
            scheduler=self._scheduler,
            callbacks=list(self._callbacks),
            gradient_clip=self._gradient_clip,
            gradient_accumulation_steps=self._gradient_accumulation_steps,
            mixed_precision=self._mixed_precision,
            max_epochs=self._max_epochs,
            device=self._device,
        )

    def __repr__(self) -> str:
        status = {
            "model": "✓" if self._model else "✗",
            "optimizer": "✓" if self._optimizer else "⚙ default",
            "loss_fn": "✓" if self._loss_fn else "✗",
            "train_loader": "✓" if self._train_loader else "✗",
            "val_loader": "✓" if self._val_loader else "—",
            "scheduler": "✓" if self._scheduler else "—",
            "callbacks": str(len(self._callbacks)),
        }
        items = ", ".join(f"{k}={v}" for k, v in status.items())
        return f"TrainingPipelineBuilder({items})"
```

---

## Method Chaining / Fluent Interface

The builder's `set_*` and `add_*` methods each return `self` (or `Self` in Python 3.11+). This enables **method chaining** — a calling style where multiple method calls are chained together in a single expression:

```python
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── Example: minimal training pipeline via fluent interface ───────────────────

def build_classification_pipeline(
    input_dim: int,
    num_classes: int,
    n_samples: int = 1000,
) -> TrainingPipeline:
    """
    Build a complete training pipeline with a fluent interface.
    Every method call configures one aspect of the pipeline;
    build() assembles and validates the whole.
    """
    # Create toy data
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, num_classes, (n_samples,))
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=64)

    # Create model
    model = nn.Sequential(
        nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, num_classes),
    )

    # Fluent construction: reads like a recipe
    pipeline = (
        TrainingPipelineBuilder()
        .set_model(model)
        .set_optimizer(torch.optim.AdamW, lr=3e-4, weight_decay=1e-4)
        .set_loss(nn.CrossEntropyLoss())
        .set_train_loader(train_loader)
        .set_val_loader(val_loader)
        .set_scheduler(
            torch.optim.lr_scheduler.CosineAnnealingLR,
            T_max=50,
        )
        .set_gradient_clip(1.0)
        .set_gradient_accumulation_steps(2)
        .set_max_epochs(50)
        .build()
    )
    return pipeline


# ── Minimal pipeline (only required components) ───────────────────────────────

def build_minimal_pipeline() -> TrainingPipeline:
    """The builder enforces that required components are present."""
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    loader = DataLoader(TensorDataset(X, y), batch_size=16)

    return (
        TrainingPipelineBuilder()
        .set_model(nn.Linear(10, 2))
        .set_loss(nn.CrossEntropyLoss())
        .set_train_loader(loader)
        .build()
        # Optimizer defaults to AdamW; no scheduler; no callbacks
    )


# ── Director: defines standard pipeline configurations ───────────────────────

class PipelineDirector:
    """
    Director: encapsulates standard pipeline configurations.
    Clients can use the director for common setups without knowing
    which builder methods to call in which order.
    """

    @staticmethod
    def build_research_pipeline(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_training_steps: int,
    ) -> TrainingPipeline:
        """Standard research pipeline: AdamW + cosine + mixed precision."""
        return (
            TrainingPipelineBuilder()
            .set_model(model)
            .set_optimizer(torch.optim.AdamW, lr=1e-4, weight_decay=0.01)
            .set_loss(nn.CrossEntropyLoss(label_smoothing=0.1))
            .set_train_loader(train_loader)
            .set_val_loader(val_loader)
            .set_scheduler(
                torch.optim.lr_scheduler.OneCycleLR,
                max_lr=1e-3,
                total_steps=num_training_steps,
            )
            .set_gradient_clip(1.0)
            .set_gradient_accumulation_steps(4)
            .enable_mixed_precision()
            .set_max_epochs(100)
            .build()
        )

    @staticmethod
    def build_quick_experiment_pipeline(
        model: nn.Module,
        train_loader: DataLoader,
    ) -> TrainingPipeline:
        """Quick experiment pipeline: minimal config, fast iteration."""
        return (
            TrainingPipelineBuilder()
            .set_model(model)
            .set_loss(nn.CrossEntropyLoss())
            .set_train_loader(train_loader)
            .set_max_epochs(5)
            .build()
        )
```

---

## Config Builder: Programmatic Nested Configs

The Builder pattern also applies to configuration objects — especially when building nested configs programmatically (e.g., for hyperparameter search).

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentConfig:
    """The product: a fully-specified experiment configuration."""
    run_name: str
    model_arch: str
    model_kwargs: dict[str, Any]
    optimizer_name: str
    optimizer_kwargs: dict[str, Any]
    scheduler_name: str
    scheduler_kwargs: dict[str, Any]
    batch_size: int
    max_epochs: int
    seed: int
    tags: list[str]
    notes: str


class ExperimentConfigBuilder:
    """
    Builder for ExperimentConfig.
    Separates concerns: naming, model config, optimisation config, run config.
    Validates completeness before producing the config.
    """

    def __init__(self) -> None:
        self._run_name: str | None = None
        self._model_arch: str | None = None
        self._model_kwargs: dict[str, Any] = {}
        self._optimizer_name: str = "adamw"
        self._optimizer_kwargs: dict[str, Any] = {"lr": 1e-3}
        self._scheduler_name: str = "none"
        self._scheduler_kwargs: dict[str, Any] = {}
        self._batch_size: int = 32
        self._max_epochs: int = 100
        self._seed: int = 42
        self._tags: list[str] = []
        self._notes: str = ""

    def named(self, name: str) -> "ExperimentConfigBuilder":
        self._run_name = name
        return self

    def with_model(self, arch: str, **kwargs: Any) -> "ExperimentConfigBuilder":
        self._model_arch = arch
        self._model_kwargs = kwargs
        return self

    def with_optimizer(self, name: str, **kwargs: Any) -> "ExperimentConfigBuilder":
        valid = {"adam", "adamw", "sgd", "rmsprop", "lamb"}
        if name not in valid:
            raise ValueError(f"Optimizer '{name}' not supported. Choose from: {valid}")
        self._optimizer_name = name
        self._optimizer_kwargs = kwargs
        return self

    def with_scheduler(self, name: str, **kwargs: Any) -> "ExperimentConfigBuilder":
        self._scheduler_name = name
        self._scheduler_kwargs = kwargs
        return self

    def with_training(
        self,
        batch_size: int,
        max_epochs: int,
        seed: int = 42,
    ) -> "ExperimentConfigBuilder":
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {max_epochs}")
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._seed = seed
        return self

    def tag(self, *tags: str) -> "ExperimentConfigBuilder":
        self._tags.extend(tags)
        return self

    def note(self, text: str) -> "ExperimentConfigBuilder":
        self._notes = text
        return self

    def build(self) -> ExperimentConfig:
        if self._run_name is None:
            raise RuntimeError("Experiment must have a name. Call named('...')")
        if self._model_arch is None:
            raise RuntimeError("Model architecture must be set. Call with_model('...')")

        return ExperimentConfig(
            run_name=self._run_name,
            model_arch=self._model_arch,
            model_kwargs=self._model_kwargs,
            optimizer_name=self._optimizer_name,
            optimizer_kwargs=self._optimizer_kwargs,
            scheduler_name=self._scheduler_name,
            scheduler_kwargs=self._scheduler_kwargs,
            batch_size=self._batch_size,
            max_epochs=self._max_epochs,
            seed=self._seed,
            tags=list(self._tags),
            notes=self._notes,
        )


# ── Usage: fluent experiment specification ────────────────────────────────────

cfg = (
    ExperimentConfigBuilder()
    .named("vit-base-cifar10-sweep-001")
    .with_model("vit_base", patch_size=16, image_size=32, num_classes=10)
    .with_optimizer("adamw", lr=3e-4, weight_decay=0.05)
    .with_scheduler("cosine_warmup", warmup_steps=500, max_steps=5000)
    .with_training(batch_size=128, max_epochs=200, seed=42)
    .tag("vision", "vit", "cifar10", "sweep")
    .note("Baseline run for ViT hyperparameter sweep. No augmentation.")
    .build()
)

print(cfg)


# ── Hyperparameter sweep: build multiple configs programmatically ─────────────

def build_lr_sweep(
    lrs: list[float],
    base_builder: ExperimentConfigBuilder,
) -> list[ExperimentConfig]:
    """Generate one config per learning rate."""
    configs = []
    for lr in lrs:
        cfg = (
            ExperimentConfigBuilder()
            .named(f"sweep-lr-{lr:.0e}")
            .with_model("mlp", input_dim=784, hidden_dim=512, output_dim=10)
            .with_optimizer("adamw", lr=lr, weight_decay=1e-4)
            .with_training(batch_size=64, max_epochs=50)
            .tag("sweep", "learning-rate")
            .build()
        )
        configs.append(cfg)
    return configs


sweep_configs = build_lr_sweep([1e-2, 3e-3, 1e-3, 3e-4, 1e-4], None)
for c in sweep_configs:
    print(f"  {c.run_name}: lr={c.optimizer_kwargs['lr']}")
```

---

## Model Assembler Builder: Transformer with Optional Components

The Builder pattern is especially powerful when assembling complex models with optional submodules — such as a transformer that may or may not have LoRA adapters, custom attention, or a linear probe head.

```python
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


# ── Transformer components ────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class LoRAAdapter(nn.Module):
    """
    Low-Rank Adaptation (LoRA) adapter.
    Hu et al. (2021): "LoRA: Low-Rank Adaptation of Large Language Models."
    """

    def __init__(self, d_model: int, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(d_model, rank, bias=False)
        self.lora_B = nn.Linear(rank, d_model, bias=False)
        nn.init.zeros_(self.lora_B.weight)  # initialise B to zero (delta W = 0 at init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scaling


class TransformerModel(nn.Module):
    """The product: a fully assembled transformer."""

    def __init__(
        self,
        embedding: nn.Embedding,
        layers: list[nn.ModuleList],
        head: nn.Module | None,
        lora_adapters: nn.ModuleList | None,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.layers = nn.ModuleList([layer for sublist in layers for layer in sublist])
        self.head = head
        self.lora_adapters = lora_adapters

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        layer_idx = 0
        for i in range(0, len(self.layers), 2):
            attn = self.layers[i]
            ffn = self.layers[i + 1]
            x = attn(x)
            if self.lora_adapters is not None and layer_idx < len(self.lora_adapters):
                x = x + self.lora_adapters[layer_idx](x)
            x = ffn(x)
            layer_idx += 1
        if self.head is not None:
            x = self.head(x[:, 0, :])  # CLS pooling
        return x


# ── Transformer Builder ───────────────────────────────────────────────────────

class TransformerBuilder:
    """
    Builder for constructing a transformer with optional components.
    Supports: standard transformer, transformer + LoRA, transformer + classifier head.
    """

    def __init__(self) -> None:
        self._vocab_size: int | None = None
        self._d_model: int = 512
        self._n_heads: int = 8
        self._n_layers: int = 6
        self._dropout: float = 0.1
        self._ffn_expansion: int = 4
        self._num_classes: int | None = None
        self._lora_rank: int | None = None
        self._lora_alpha: float = 16.0

    def set_vocabulary(self, vocab_size: int) -> "TransformerBuilder":
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        self._vocab_size = vocab_size
        return self

    def set_dimensions(self, d_model: int, n_heads: int) -> "TransformerBuilder":
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self._d_model = d_model
        self._n_heads = n_heads
        return self

    def set_depth(self, n_layers: int) -> "TransformerBuilder":
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self._n_layers = n_layers
        return self

    def set_dropout(self, dropout: float) -> "TransformerBuilder":
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self._dropout = dropout
        return self

    def add_classification_head(self, num_classes: int) -> "TransformerBuilder":
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        self._num_classes = num_classes
        return self

    def add_lora_adapters(
        self,
        rank: int = 8,
        alpha: float = 16.0,
    ) -> "TransformerBuilder":
        """Attach LoRA adapters to every transformer layer."""
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")
        self._lora_rank = rank
        self._lora_alpha = alpha
        return self

    def build(self) -> TransformerModel:
        if self._vocab_size is None:
            raise RuntimeError(
                "Vocabulary size not set. Call set_vocabulary(vocab_size)."
            )

        embedding = nn.Embedding(self._vocab_size, self._d_model)

        layers: list[list[nn.Module]] = []
        for _ in range(self._n_layers):
            attn = MultiHeadSelfAttention(self._d_model, self._n_heads, self._dropout)
            ffn = FeedForwardNetwork(self._d_model, self._ffn_expansion, self._dropout)
            layers.append([attn, ffn])

        head: nn.Module | None = None
        if self._num_classes is not None:
            head = nn.Sequential(
                nn.LayerNorm(self._d_model),
                nn.Linear(self._d_model, self._num_classes),
            )

        lora_adapters: nn.ModuleList | None = None
        if self._lora_rank is not None:
            lora_adapters = nn.ModuleList([
                LoRAAdapter(self._d_model, self._lora_rank, self._lora_alpha)
                for _ in range(self._n_layers)
            ])

        return TransformerModel(embedding, layers, head, lora_adapters)


# ── Usage: three different transformers from one builder ──────────────────────

# Standard encoder
encoder = (
    TransformerBuilder()
    .set_vocabulary(30522)
    .set_dimensions(d_model=512, n_heads=8)
    .set_depth(6)
    .set_dropout(0.1)
    .build()
)

# Classification model
classifier = (
    TransformerBuilder()
    .set_vocabulary(30522)
    .set_dimensions(d_model=768, n_heads=12)
    .set_depth(12)
    .add_classification_head(num_classes=2)
    .build()
)

# Fine-tunable model with LoRA
lora_model = (
    TransformerBuilder()
    .set_vocabulary(50257)
    .set_dimensions(d_model=1024, n_heads=16)
    .set_depth(24)
    .add_lora_adapters(rank=16, alpha=32.0)
    .add_classification_head(num_classes=10)
    .build()
)

for name, model in [("encoder", encoder), ("classifier", classifier), ("lora_model", lora_model)]:
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name}: {n_params:,} trainable params")
```

---

## Flow: Step-by-Step Assembly

<div class="diagram">
<div class="diagram-title">Builder Assembly Process</div>
<div class="flow">
  <div class="flow-node accent wide">TrainingPipelineBuilder()</div>
  <div class="flow-arrow">↓ .set_model(model)</div>
  <div class="flow-node blue wide">Model registered ✓<br/><small>Validates: is nn.Module</small></div>
  <div class="flow-arrow">↓ .set_optimizer(AdamW, lr=3e-4)</div>
  <div class="flow-node green wide">Optimizer created ✓<br/><small>Requires: model already set</small></div>
  <div class="flow-arrow">↓ .set_loss(CrossEntropyLoss)</div>
  <div class="flow-node purple wide">Loss function registered ✓</div>
  <div class="flow-arrow">↓ .set_train_loader(loader)</div>
  <div class="flow-node orange wide">DataLoader registered ✓</div>
  <div class="flow-arrow">↓ .set_scheduler(CosineAnnealingLR)</div>
  <div class="flow-node teal wide">Scheduler created ✓<br/><small>Requires: optimizer already set</small></div>
  <div class="flow-arrow">↓ .add_callback(ckpt_cb).add_callback(log_cb)</div>
  <div class="flow-node pink wide">Callbacks registered: [checkpoint, logging]</div>
  <div class="flow-arrow">↓ .enable_mixed_precision().set_max_epochs(100)</div>
  <div class="flow-node cyan wide">Hyperparameters configured ✓</div>
  <div class="flow-arrow">↓ .build()</div>
  <div class="flow-node yellow wide">VALIDATION CHECKPOINT<br/><small>Checks all required fields; raises RuntimeError if incomplete</small></div>
  <div class="flow-arrow">↓ valid ✓</div>
  <div class="flow-node accent wide">TrainingPipeline (immutable product)</div>
</div>
</div>

---

## Builder as a Validation Checkpoint

One of the most underappreciated uses of the Builder pattern is as a **validation checkpoint**: the `build()` method is the single point where all cross-component invariants can be checked.

```python
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ValidatingPipelineBuilder(TrainingPipelineBuilder):
    """
    Extends TrainingPipelineBuilder with deep validation at build time.
    Catches configuration errors before any training begins.
    """

    def build(self) -> TrainingPipeline:
        # Run all standard validation
        self._validate()

        # ── Deep cross-component validation ───────────────────────────────────

        # Check 1: Mixed precision requires CUDA
        if self._mixed_precision and not torch.cuda.is_available():
            raise RuntimeError(
                "Mixed precision is enabled but CUDA is not available. "
                "Disable with .enable_mixed_precision(False) or run on GPU."
            )

        # Check 2: Model and data dimensionality alignment (if checkable)
        if self._train_loader is not None and self._model is not None:
            try:
                batch = next(iter(self._train_loader))
                if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                    sample_input = batch[0][:1]  # take one example
                    with torch.no_grad():
                        _ = self._model(sample_input)  # dry run
                    print("✓ Model/data dimensionality check passed")
            except Exception as e:
                raise RuntimeError(
                    f"Model/data compatibility check failed: {e}\n"
                    "Check that your model's input dimensions match the data."
                ) from e

        # Check 3: Gradient accumulation consistency
        if (
            self._gradient_accumulation_steps > 1
            and self._train_loader is not None
        ):
            dataset_size = len(self._train_loader.dataset)  # type: ignore[arg-type]
            effective_batch = (
                self._train_loader.batch_size * self._gradient_accumulation_steps
            )
            if effective_batch > dataset_size:
                import warnings
                warnings.warn(
                    f"Effective batch size ({effective_batch}) exceeds dataset size "
                    f"({dataset_size}). Training will be very noisy.",
                    stacklevel=2,
                )

        # Check 4: Scheduler + epochs consistency
        if (
            isinstance(self._scheduler, torch.optim.lr_scheduler.OneCycleLR)
            and self._train_loader is not None
        ):
            # OneCycleLR total_steps must match actual training
            pass  # complex validation omitted for brevity

        return super().build()
```

---

## Comparison: Builder vs. Constructor vs. Factory

| | Constructor | Factory | Builder |
|-|-------------|---------|---------|
| **When to use** | Simple objects, few arguments | Object type varies by config | Complex objects, many optional parts |
| **Argument style** | All at once | All at once | Step by step, fluent |
| **Validation** | In `__init__`, all at once | In factory function | In each setter + final `build()` |
| **Optional parts** | Default values in signature | Default values in factory | Separate optional setter methods |
| **Product variety** | One class | Many classes | One class, many configurations |
| **Readability** | Poor with > 5 args | Good | Excellent with fluent chaining |
| **Extensibility** | Edit constructor signature | Edit factory / registry | Add new setter methods |
| **ML use case** | Simple transforms, heads | Model architecture selection | Training pipelines, experiment configs |

---

<div class="callout tip">
<strong>✅ Use Builder When:</strong><br/><br/>
• Your object has more than ~5 constructor arguments, especially if many are optional.<br/>
• Some components must be created in a specific order (optimizer before scheduler).<br/>
• You need to validate cross-component invariants before the object is used.<br/>
• You want to support multiple "standard configurations" (via a Director) while keeping the builder flexible for custom configurations.<br/>
• You want the construction process to read like a recipe — each step self-documenting what it does.<br/>
• You're building a configuration object that will be serialised, replayed, or shared across experiments.
</div>

---

**Next: [Chapter 6 — Singleton & Prototype →](./06_singleton_prototype.md)**

*Last updated: May 2026*
