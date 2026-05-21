---
title: "Chapter 17 — Template Method & Training Loop"
---

[← Back to Table of Contents](./README.md)

# Chapter 17 — Template Method & Training Loop

> *"The skeleton is invariant; the flesh is variable. Define the former once, let subclasses provide the latter."*

Every ML framework you have ever used — PyTorch Lightning, Keras, Hugging Face Trainer — is built on the same foundation: a training loop whose **structure is fixed** but whose individual steps are **customisable**. That structure is the Template Method pattern. Understanding it not only explains why those frameworks feel familiar across different domains, but also gives you the tools to build your own extensible trainers.

<span class="badge behavioral">Behavioral</span>

---

## Intent

Define the **skeleton of an algorithm** in an operation, deferring some steps to subclasses. Template Method lets subclasses redefine certain steps of an algorithm without changing the algorithm's structure.

---

## UML Structure

<div class="diagram">
  <div class="diagram-title">Template Method — Class Structure</div>
  <div class="uml-row">
    <div class="uml-box">
      <div class="uml-title">AbstractClass</div>
      <div class="uml-section">
        <div class="uml-item">+ template_method()  ← final skeleton</div>
      </div>
      <div class="uml-section">
        <div class="uml-item"># primitive_op_1()  ← abstract</div>
        <div class="uml-item"># primitive_op_2()  ← abstract</div>
        <div class="uml-item"># hook()            ← optional override</div>
      </div>
    </div>
  </div>
  <div class="uml-row">
    <div class="uml-box">
      <div class="uml-title">ConcreteClassA</div>
      <div class="uml-section">
        <div class="uml-item"># primitive_op_1() → specific impl A</div>
        <div class="uml-item"># primitive_op_2() → specific impl A</div>
      </div>
    </div>
    <div class="uml-box">
      <div class="uml-title">ConcreteClassB</div>
      <div class="uml-section">
        <div class="uml-item"># primitive_op_1() → specific impl B</div>
        <div class="uml-item"># primitive_op_2() → specific impl B</div>
        <div class="uml-item"># hook() → custom behaviour B</div>
      </div>
    </div>
  </div>
</div>

There are two kinds of methods in the abstract class:

- **Abstract steps** — *must* be overridden; they represent variable parts with no sensible default.
- **Hook methods** — *may* be overridden; they have a default (often empty) implementation.

---

## The Training Loop as a Template Method

A vanilla supervised training loop always follows the same sequence:

1. **Setup** — instantiate model, optimiser, data loaders
2. **For each epoch:**
   a. For each batch in train set: forward → loss → backward → step
   b. Evaluate on validation set
   c. Log metrics / checkpoint / early-stop
3. **Teardown** — save final model, close writers

This sequence is the *template*. What varies across domains is: how the forward pass works, what the loss is, how validation is measured. The Template Method pattern draws a clean boundary between the invariant scaffold and the variable specifics.

---

## Full `BaseTrainer` Implementation

```python
from __future__ import annotations
import abc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Data transfer object passed through the training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    train_loss: float = float("inf")
    val_metric: float = float("inf")
    best_val_metric: float = float("inf")
    should_stop: bool = False
    history: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract base — the Template Method
# ---------------------------------------------------------------------------

class BaseTrainer(abc.ABC):
    """
    Template Method for supervised ML training.

    Invariant skeleton (template_method = fit):
        setup → [epoch_loop → [train_step × N → validate]] → teardown

    Abstract steps (must override):
        compute_loss, forward_pass, configure_optimizers

    Hook steps (may override):
        on_epoch_start, on_epoch_end, on_batch_end, log_metrics,
        get_val_metric
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        max_epochs: int = 20,
        device: str | torch.device = "cpu",
        checkpoint_dir: str = "checkpoints",
        grad_clip: float | None = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.grad_clip = grad_clip
        self.state = TrainState()
        self.optimizer: Optimizer | None = None
        self.scheduler = None

    # ------------------------------------------------------------------
    # TEMPLATE METHOD — the invariant skeleton.  Do not override.
    # ------------------------------------------------------------------

    def fit(self) -> TrainState:
        """Run the full training procedure.  This is the template method."""
        self.optimizer, self.scheduler = self.configure_optimizers()
        self._setup()
        self.on_train_start()

        for epoch in range(self.max_epochs):
            if self.state.should_stop:
                break
            self.state.epoch = epoch
            self.on_epoch_start(epoch)

            t0 = time.perf_counter()
            train_loss = self._run_train_epoch()
            self.state.train_loss = train_loss

            val_metric = self._run_validation()
            self.state.val_metric = val_metric

            epoch_time = time.perf_counter() - t0
            self.log_metrics(epoch, train_loss, val_metric, epoch_time)
            self.on_epoch_end(epoch, train_loss, val_metric)

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_metric": val_metric,
            }
            self.state.history.append(record)

            if val_metric < self.state.best_val_metric:
                self.state.best_val_metric = val_metric
                self._save_checkpoint("best.pt")

            if self.scheduler is not None:
                try:
                    self.scheduler.step(val_metric)
                except TypeError:
                    self.scheduler.step()

        self._teardown()
        self.on_train_end()
        return self.state

    # ------------------------------------------------------------------
    # ABSTRACT STEPS — subclasses MUST implement these
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute the scalar training loss."""

    @abc.abstractmethod
    def forward_pass(
        self, batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass.  Return (outputs, targets)."""

    @abc.abstractmethod
    def configure_optimizers(
        self,
    ) -> tuple[Optimizer, Any]:
        """Return (optimizer, scheduler).  Scheduler may be None."""

    # ------------------------------------------------------------------
    # HOOK METHODS — subclasses MAY override these
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        """Called once before training begins."""

    def on_train_end(self) -> None:
        """Called once after training ends."""

    def on_epoch_start(self, epoch: int) -> None:
        """Called at the start of each epoch."""

    def on_epoch_end(
        self, epoch: int, train_loss: float, val_metric: float
    ) -> None:
        """Called at the end of each epoch."""

    def on_batch_end(
        self, batch_idx: int, loss: float, outputs: torch.Tensor
    ) -> None:
        """Called after each training batch."""

    def log_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_metric: float,
        elapsed: float,
    ) -> None:
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_metric={val_metric:.4f} | "
            f"time={elapsed:.2f}s"
        )

    def get_val_metric(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute validation metric from accumulated predictions."""
        # Default: return loss as the metric
        return self.compute_loss(outputs, targets).item()

    # ------------------------------------------------------------------
    # Private scaffold — not meant for overriding
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _teardown(self) -> None:
        pass

    def _run_train_epoch(self) -> float:
        self.model.train()
        total_loss, n_batches = 0.0, 0
        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            outputs, targets = self.forward_pass(batch)
            loss = self.compute_loss(outputs, targets)
            loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.state.global_step += 1
            total_loss += loss.item()
            n_batches += 1
            self.on_batch_end(batch_idx, loss.item(), outputs)
        return total_loss / max(n_batches, 1)

    def _run_validation(self) -> float:
        if self.val_loader is None:
            return float("inf")
        self.model.eval()
        all_outputs, all_targets = [], []
        with torch.no_grad():
            for batch in self.val_loader:
                outputs, targets = self.forward_pass(batch)
                all_outputs.append(outputs)
                all_targets.append(targets)
        return self.get_val_metric(
            torch.cat(all_outputs), torch.cat(all_targets)
        )

    def _save_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": self.state.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_metric": self.state.val_metric,
            },
            path,
        )
```

---

## Concrete Subclasses

### ClassificationTrainer

```python
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ClassificationTrainer(BaseTrainer):
    """Template Method filled in for image/text classification."""

    def configure_optimizers(self):
        opt = AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-2)
        sched = ReduceLROnPlateau(opt, mode="min", patience=3, factor=0.5)
        return opt, sched

    def forward_pass(self, batch):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        logits = self.model(images)
        return logits, labels

    def compute_loss(self, outputs, targets):
        return F.cross_entropy(outputs, targets)

    def get_val_metric(self, outputs, targets):
        # Override to return accuracy instead of loss
        preds = outputs.argmax(dim=1)
        acc = (preds == targets).float().mean().item()
        return 1.0 - acc  # return as "error" so lower = better

    def on_epoch_end(self, epoch, train_loss, val_metric):
        acc = 1.0 - val_metric
        if acc > 0.99:
            print(f"[ClassificationTrainer] >99% accuracy at epoch {epoch}!")
            self.state.should_stop = True
```

### GenerationTrainer

```python
class GenerationTrainer(BaseTrainer):
    """Language model training with teacher forcing."""

    def __init__(self, *args, label_smoothing: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing

    def configure_optimizers(self):
        from torch.optim import Adam
        opt = Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.98))
        return opt, None

    def forward_pass(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        # Autoregressive: predict next token
        outputs = self.model(input_ids[:, :-1])  # (B, T-1, V)
        targets = input_ids[:, 1:].reshape(-1)   # (B*(T-1),)
        return outputs.reshape(-1, outputs.size(-1)), targets

    def compute_loss(self, outputs, targets):
        return F.cross_entropy(
            outputs, targets, label_smoothing=self.label_smoothing
        )

    def log_metrics(self, epoch, train_loss, val_metric, elapsed):
        perplexity = torch.exp(torch.tensor(train_loss)).item()
        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.4f} | "
            f"ppl={perplexity:.2f} | "
            f"val_ppl={torch.exp(torch.tensor(val_metric)).item():.2f}"
        )
```

### ContrastiveLearningTrainer

```python
class ContrastiveLearningTrainer(BaseTrainer):
    """SimCLR-style contrastive training."""

    def __init__(self, *args, temperature: float = 0.07, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def configure_optimizers(self):
        from torch.optim import LARS  # or SGD
        opt = torch.optim.SGD(
            self.model.parameters(), lr=0.3, momentum=0.9, weight_decay=1e-6
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs
        )
        return opt, sched

    def forward_pass(self, batch):
        (x1, x2), _ = batch  # two augmented views, labels ignored
        x1, x2 = x1.to(self.device), x2.to(self.device)
        z1, z2 = self.model(x1), self.model(x2)
        # Stack for NT-Xent loss
        return torch.stack([z1, z2], dim=1), None

    def compute_loss(self, outputs, targets):
        z1, z2 = outputs[:, 0], outputs[:, 1]
        return self._nt_xent(z1, z2)

    def _nt_xent(self, z1, z2):
        B = z1.size(0)
        z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)  # (2B, D)
        sim = torch.mm(z, z.T) / self.temperature            # (2B, 2B)
        # Mask out self-similarity
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, float("-inf"))
        labels = torch.arange(B, device=z.device)
        labels = torch.cat([labels + B, labels])
        return F.cross_entropy(sim, labels)
```

---

## PyTorch Lightning's `LightningModule` as Template Method

Lightning's `LightningModule` is the most widely used Template Method in the PyTorch ecosystem. The `Trainer` (the invoker) calls the template; `LightningModule` subclasses provide the abstract steps.

```python
import pytorch_lightning as pl


class LitClassifier(pl.LightningModule):
    """LightningModule: abstract steps filled in for classification."""

    def __init__(self, backbone: nn.Module, num_classes: int, lr: float = 3e-4):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.out_features, num_classes)
        self.lr = lr
        self.save_hyperparameters(ignore=["backbone"])

    # ---- Abstract step: forward ------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    # ---- Abstract step: training_step ------------------------------------
    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # ---- Abstract step: validation_step ---------------------------------
    def validation_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

    # ---- Abstract step: configure_optimizers ----------------------------
    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.lr)
        sched = {
            "scheduler": ReduceLROnPlateau(opt, patience=3),
            "monitor": "val_loss",
        }
        return [opt], [sched]

    # ---- Hook step (optional): on_train_epoch_end -----------------------
    def on_train_epoch_end(self) -> None:
        sch = self.lr_schedulers()
        if sch is not None:
            self.log("lr", sch.get_last_lr()[0])
```

Lightning's `Trainer.fit(model)` call is the template invocation — it calls `training_step`, `validation_step`, and `configure_optimizers` in the correct order, wiring in DDP, AMP, gradient clipping, and all other cross-cutting concerns without the user touching them.

---

## Data Preprocessing Pipeline as Template Method

Template Method applies beyond training — preprocessing pipelines have the same invariant structure.

```python
class BasePreprocessor(abc.ABC):
    """Template method for data preprocessing."""

    def process(self, raw_data: Any) -> Any:
        """Template method — fixed pipeline."""
        data = self.load(raw_data)
        data = self.clean(data)
        data = self.normalise(data)
        data = self.featurise(data)
        return self.finalise(data)

    @abc.abstractmethod
    def load(self, raw_data: Any) -> Any: ...

    @abc.abstractmethod
    def clean(self, data: Any) -> Any: ...

    def normalise(self, data: Any) -> Any:
        """Hook — identity by default."""
        return data

    @abc.abstractmethod
    def featurise(self, data: Any) -> Any: ...

    def finalise(self, data: Any) -> Any:
        """Hook — identity by default."""
        return data


class TextPreprocessor(BasePreprocessor):
    def load(self, raw_data):
        return raw_data if isinstance(raw_data, str) else str(raw_data)

    def clean(self, data):
        import re
        data = data.lower().strip()
        data = re.sub(r"[^\w\s]", " ", data)
        return re.sub(r"\s+", " ", data)

    def normalise(self, data):
        # Override hook: apply unicode normalisation
        import unicodedata
        return unicodedata.normalize("NFKC", data)

    def featurise(self, data):
        return data.split()  # simple whitespace tokenisation

    def finalise(self, data):
        # Pad / truncate to max_len
        max_len = 512
        return data[:max_len]


class ImagePreprocessor(BasePreprocessor):
    def load(self, raw_data):
        from PIL import Image
        return Image.open(raw_data).convert("RGB")

    def clean(self, data):
        return data  # images don't need text cleaning

    def normalise(self, data):
        import torchvision.transforms.functional as TF
        t = TF.to_tensor(data)
        return TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def featurise(self, data):
        return data  # tensor is the feature
```

---

## The Hollywood Principle

<div class="callout info">
<strong>The Hollywood Principle: "Don't call us, we'll call you."</strong><br>
In Template Method, the abstract class (the framework, the trainer) is in charge of the control flow. Subclasses provide implementations, but they never call the skeleton themselves. The high-level class calls into the low-level subclass — not the other way round.

This inverts the traditional library/application relationship. Instead of your code calling the framework, the framework calls your code at the right time. Lightning embodies this perfectly: you never call <code>training_step</code> yourself; the <code>Trainer</code> does.
</div>

---

## Flow Diagram

<div class="diagram">
  <div class="diagram-title">Template Method — Training Execution Flow</div>
  <div class="flow">
    <div class="flow-node accent wide">Trainer.fit()  ← Template Method Entry</div>
    <div class="flow-arrow accent">▼</div>
    <div class="flow-node blue wide">setup()  ← scaffold (private)</div>
    <div class="flow-arrow">▼  for each epoch</div>
    <div class="flow-node green wide">on_epoch_start()  ← hook</div>
    <div class="flow-arrow">▼</div>
    <div class="flow-h">
      <div class="flow-node purple">forward_pass()  ← abstract</div>
      <div class="flow-node orange">compute_loss()  ← abstract</div>
      <div class="flow-node teal">on_batch_end()  ← hook</div>
    </div>
    <div class="flow-arrow">▼</div>
    <div class="flow-node blue wide">validate()  → get_val_metric()  ← hook</div>
    <div class="flow-arrow">▼</div>
    <div class="flow-node green wide">on_epoch_end() + log_metrics()  ← hooks</div>
    <div class="flow-arrow accent">▼  after all epochs</div>
    <div class="flow-node accent wide">teardown()  ← scaffold (private)</div>
  </div>
</div>

---

## Template Method vs Strategy

These two patterns solve similar-sounding problems but from opposite directions.

<table class="compare-table">
  <thead>
    <tr>
      <th>Dimension</th>
      <th>Template Method</th>
      <th>Strategy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Mechanism</td>
      <td>Inheritance</td>
      <td>Composition / delegation</td>
    </tr>
    <tr>
      <td>Granularity</td>
      <td>Whole algorithm skeleton</td>
      <td>One interchangeable behaviour</td>
    </tr>
    <tr>
      <td>Extension point</td>
      <td>Subclass overrides methods</td>
      <td>Pass different strategy object</td>
    </tr>
    <tr>
      <td>Runtime switching</td>
      <td>No — class is fixed</td>
      <td>Yes — swap strategy object</td>
    </tr>
    <tr>
      <td>Best for</td>
      <td>Frameworks, training loops</td>
      <td>Algorithms, loss functions, samplers</td>
    </tr>
    <tr>
      <td>ML example</td>
      <td>Lightning LightningModule</td>
      <td>Loss function, optimizer selection</td>
    </tr>
  </tbody>
</table>

<div class="diagram">
  <div class="diagram-title">When to use which?</div>
  <div class="flow-h">
    <div class="flow-node green wide">The full loop structure varies<br>→ <strong>Template Method</strong><br><small>Subclass the trainer</small></div>
    <div class="flow-node blue wide">Only one behaviour varies<br>→ <strong>Strategy</strong><br><small>Inject a loss function</small></div>
  </div>
</div>

---

## Tips & Warnings

<div class="callout tip">
<strong>Keep template methods short.</strong> The skeleton should be a concise sequence of high-level calls — ideally under 20 lines. If the template method itself becomes long and complex, the invariant structure is unclear and the pattern loses its value.
</div>

<div class="callout warn">
<strong>Beware deep class hierarchies.</strong> Template Method encourages inheritance; it is easy to build ClassificationTrainer → RobustClassificationTrainer → FewShotRobustClassificationTrainer chains that become unmaintainable. Prefer shallow hierarchies (one level of subclassing) and compose behaviours via callbacks or strategies rather than stacking inheritance layers.
</div>

---

## Key Takeaways

<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">🦴</div>
    <div class="card-title">Skeleton First</div>
    <div class="card-desc">Write the invariant scaffold before the variable parts. The template method should read like a specification of what the algorithm does.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🎭</div>
    <div class="card-title">Abstract vs Hook</div>
    <div class="card-desc">Abstract methods enforce a contract; hooks offer optional customisation. Use abstract for steps with no sensible default; hooks for optional behaviour.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">⚡</div>
    <div class="card-title">Framework DNA</div>
    <div class="card-desc">Lightning, Keras, and HF Trainer are all Template Method. Recognising the pattern lets you extend any framework confidently.</div>
  </div>
</div>

---

*Last updated: May 2026*
