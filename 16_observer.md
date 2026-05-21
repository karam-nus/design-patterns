---
title: "Chapter 16 — Observer, Callback & Event System"
---

[← Back to Table of Contents](./README.md)

# Chapter 16 — Observer, Callback & Event System

> *"Don't call us, we'll call you — but only if something interesting happens."*

The **Observer** pattern is one of the most widely deployed patterns in machine learning infrastructure. Every time a training loop fires `on_epoch_end`, every time a PyTorch hook captures an activation, every time W&B logs a metric — the Observer pattern is at work. Understanding it deeply unlocks the ability to build extensible, loosely-coupled ML systems where new behaviours (logging, checkpointing, early stopping) can be added without touching the core training logic.

<span class="badge behavioral">Behavioral</span>

---

## Intent

Define a **one-to-many dependency** between objects so that when one object changes state, all its dependents are **notified and updated automatically**.

The object that holds the state is the **Subject** (or Publisher). The objects that want to know about changes are **Observers** (or Subscribers). Neither knows the concrete type of the other — they communicate through a stable interface.

---

## UML Structure

<div class="diagram">
  <div class="diagram-title">Observer — Class Structure</div>
  <div class="uml-row">
    <div class="uml-box">
      <div class="uml-title">Subject</div>
      <div class="uml-section">
        <div class="uml-item">– _observers: List[Observer]</div>
      </div>
      <div class="uml-section">
        <div class="uml-item">+ attach(o: Observer)</div>
        <div class="uml-item">+ detach(o: Observer)</div>
        <div class="uml-item">+ notify()</div>
      </div>
    </div>
    <div class="uml-box">
      <div class="uml-title">«interface» Observer</div>
      <div class="uml-section">
        <div class="uml-item">+ update(subject)</div>
      </div>
    </div>
  </div>
  <div class="uml-row">
    <div class="uml-box">
      <div class="uml-title">ConcreteSubject</div>
      <div class="uml-section">
        <div class="uml-item">– state: Any</div>
      </div>
      <div class="uml-section">
        <div class="uml-item">+ get_state()</div>
        <div class="uml-item">+ set_state(v)</div>
      </div>
    </div>
    <div class="uml-box">
      <div class="uml-title">ConcreteObserver</div>
      <div class="uml-section">
        <div class="uml-item">– cached_state: Any</div>
      </div>
      <div class="uml-section">
        <div class="uml-item">+ update(subject)</div>
      </div>
    </div>
  </div>
</div>

---

## Push vs Pull Notification Models

There are two philosophies for how the Subject communicates state to its Observers.

### Push Model

The Subject **sends data** to every observer on notification. Observers receive a payload they may or may not need. This couples the Subject to the data shape each Observer expects.

```python
class PushSubject:
    def __init__(self):
        self._observers: list = []
        self._state: dict = {}

    def attach(self, observer) -> None:
        self._observers.append(observer)

    def detach(self, observer) -> None:
        self._observers.remove(observer)

    def set_state(self, **kwargs) -> None:
        self._state.update(kwargs)
        self._notify()

    def _notify(self) -> None:
        # Push: send a copy of current state to every observer
        for obs in self._observers:
            obs.update(dict(self._state))


class PushObserver:
    def update(self, state: dict) -> None:
        loss = state.get("loss")
        print(f"[PushObserver] received loss={loss:.4f}")
```

### Pull Model

The Subject **passes itself** as the argument. Observers call back into the Subject to retrieve exactly what they need. More flexible but requires observers to know the Subject's public interface.

```python
class PullSubject:
    def __init__(self):
        self._observers: list = []
        self.epoch: int = 0
        self.loss: float = float("inf")
        self.metrics: dict = {}

    def attach(self, observer) -> None:
        self._observers.append(observer)

    def _notify(self) -> None:
        # Pull: pass self; observers query what they need
        for obs in self._observers:
            obs.update(self)


class PullObserver:
    def update(self, subject: "PullSubject") -> None:
        # Pull only what this observer cares about
        print(f"[PullObserver] epoch={subject.epoch}, loss={subject.loss:.4f}")
```

| Aspect | Push | Pull |
|---|---|---|
| Coupling | Medium — tied to payload shape | Low — observers use public API |
| Bandwidth | Sends everything every time | Observers fetch only what they need |
| Flexibility | Low — changing payload breaks observers | High |
| Common in ML | Lightning (push-like) | PyTorch hooks (pull) |

---

## Training Callbacks as Observers

The canonical ML application of Observer is the **callback system**. A `Trainer` is the Subject; `Callback` instances are Observers. Each callback method corresponds to a lifecycle event.

```python
from __future__ import annotations
import abc
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


# ---------------------------------------------------------------------------
# Abstract Observer
# ---------------------------------------------------------------------------

class TrainingCallback(abc.ABC):
    """Abstract base for all training callbacks (Observers)."""

    # Subject (Trainer) calls these hooks at lifecycle boundaries ----------

    def on_train_start(self, trainer: "Trainer") -> None:
        """Called once before the first epoch."""

    def on_train_end(self, trainer: "Trainer") -> None:
        """Called once after training completes or is stopped."""

    def on_epoch_start(self, trainer: "Trainer", epoch: int) -> None:
        """Called at the beginning of every epoch."""

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: dict) -> None:
        """Called at the end of every epoch with current metrics."""

    def on_batch_start(self, trainer: "Trainer", batch_idx: int) -> None:
        """Called before each training step."""

    def on_batch_end(
        self, trainer: "Trainer", batch_idx: int, loss: float
    ) -> None:
        """Called after each training step with the step loss."""


# ---------------------------------------------------------------------------
# Concrete Subject — Trainer
# ---------------------------------------------------------------------------

@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    should_stop: bool = False
    logs: dict = field(default_factory=dict)


class Trainer:
    """Concrete Subject.  Manages observers and drives the training loop."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        max_epochs: int = 10,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.state = TrainerState()
        self._callbacks: list[TrainingCallback] = []

    # Observer management --------------------------------------------------

    def add_callback(self, cb: TrainingCallback) -> "Trainer":
        self._callbacks.append(cb)
        return self  # fluent API

    def remove_callback(self, cb: TrainingCallback) -> None:
        self._callbacks.remove(cb)

    # Notification helpers -------------------------------------------------

    def _fire(self, event: str, *args, **kwargs) -> None:
        for cb in self._callbacks:
            getattr(cb, event)(self, *args, **kwargs)

    # Training loop --------------------------------------------------------

    def fit(self, train_loader, val_loader=None) -> None:
        self._fire("on_train_start")
        for epoch in range(self.max_epochs):
            if self.state.should_stop:
                break
            self.state.epoch = epoch
            self._fire("on_epoch_start", epoch)
            train_loss = self._run_epoch(train_loader)
            val_loss = self._validate(val_loader) if val_loader else None
            logs = {"train_loss": train_loss}
            if val_loss is not None:
                logs["val_loss"] = val_loss
            self.state.logs = logs
            self._fire("on_epoch_end", epoch, logs)
        self._fire("on_train_end")

    def _run_epoch(self, loader) -> float:
        self.model.train()
        total, count = 0.0, 0
        for batch_idx, batch in enumerate(loader):
            self._fire("on_batch_start", batch_idx)
            loss = self._train_step(batch)
            self.state.global_step += 1
            total += loss
            count += 1
            self._fire("on_batch_end", batch_idx, loss)
        return total / max(count, 1)

    def _train_step(self, batch) -> float:
        raise NotImplementedError

    def _validate(self, loader) -> float:
        raise NotImplementedError
```

### Concrete Callbacks

```python
# ---------------------------------------------------------------------------
# EarlyStoppingCallback
# ---------------------------------------------------------------------------

class EarlyStoppingCallback(TrainingCallback):
    """Stop training when a monitored metric stops improving."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._counter = 0
        self._best: float = float("inf") if mode == "min" else float("-inf")

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict) -> None:
        current = logs.get(self.monitor)
        if current is None:
            return
        improved = (
            current < self._best - self.min_delta
            if self.mode == "min"
            else current > self._best + self.min_delta
        )
        if improved:
            self._best = current
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                print(
                    f"[EarlyStopping] No improvement for {self.patience} epochs. "
                    f"Stopping at epoch {epoch}."
                )
                trainer.state.should_stop = True


# ---------------------------------------------------------------------------
# LRSchedulerCallback
# ---------------------------------------------------------------------------

class LRSchedulerCallback(TrainingCallback):
    """Step a learning-rate scheduler after each epoch."""

    def __init__(self, scheduler: _LRScheduler, monitor: str = "val_loss") -> None:
        self.scheduler = scheduler
        self.monitor = monitor

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict) -> None:
        metric = logs.get(self.monitor)
        if hasattr(self.scheduler, "step"):
            try:
                self.scheduler.step(metric)
            except TypeError:
                self.scheduler.step()
        lrs = [pg["lr"] for pg in trainer.optimizer.param_groups]
        print(f"[LRScheduler] epoch={epoch} lr={lrs}")


# ---------------------------------------------------------------------------
# CheckpointCallback
# ---------------------------------------------------------------------------

class CheckpointCallback(TrainingCallback):
    """Save a checkpoint whenever the monitored metric improves."""

    def __init__(
        self,
        dirpath: str | Path = "checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 3,
    ) -> None:
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self._best = float("inf") if mode == "min" else float("-inf")
        self._saved: list[tuple[float, Path]] = []

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict) -> None:
        metric = logs.get(self.monitor)
        if metric is None:
            return
        improved = (
            metric < self._best if self.mode == "min" else metric > self._best
        )
        if improved:
            self._best = metric
            self.dirpath.mkdir(parents=True, exist_ok=True)
            ckpt_path = self.dirpath / f"epoch={epoch}-{self.monitor}={metric:.4f}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": trainer.model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "metric": metric,
                },
                ckpt_path,
            )
            self._saved.append((metric, ckpt_path))
            self._saved.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
            # Evict old checkpoints beyond save_top_k
            while len(self._saved) > self.save_top_k:
                _, old_path = self._saved.pop()
                old_path.unlink(missing_ok=True)
            print(f"[Checkpoint] Saved {ckpt_path}")


# ---------------------------------------------------------------------------
# LoggingCallback
# ---------------------------------------------------------------------------

class LoggingCallback(TrainingCallback):
    """Print epoch-level metrics to stdout."""

    def __init__(self, log_every_n_epochs: int = 1) -> None:
        self.log_every_n_epochs = log_every_n_epochs
        self._t0: float = 0.0

    def on_train_start(self, trainer: Trainer) -> None:
        self._t0 = time.time()
        print(f"[Logger] Training started. max_epochs={trainer.max_epochs}")

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict) -> None:
        if epoch % self.log_every_n_epochs == 0:
            elapsed = time.time() - self._t0
            parts = " | ".join(f"{k}={v:.4f}" for k, v in logs.items())
            print(f"[Logger] Epoch {epoch:04d} | {parts} | elapsed={elapsed:.1f}s")

    def on_train_end(self, trainer: Trainer) -> None:
        total = time.time() - self._t0
        print(f"[Logger] Training finished in {total:.1f}s")
```

---

## PyTorch Hooks as Observers

PyTorch modules expose a hook system that lets you observe the forward and backward passes **without modifying the model**. Hooks are function-level observers.

### Available Hook Types

| Hook | When fired | Receives |
|---|---|---|
| `register_forward_pre_hook` | Before `forward()` | module, input |
| `register_forward_hook` | After `forward()` | module, input, output |
| `register_backward_hook` | After gradient computation | module, grad_in, grad_out |
| `register_full_backward_hook` | After backward (full) | module, grad_in, grad_out |

### Activation Extraction

```python
from collections import OrderedDict
from typing import Callable

import torch
import torch.nn as nn


class ActivationExtractor:
    """Extract intermediate activations from named layers using forward hooks."""

    def __init__(self, model: nn.Module, layer_names: list[str]) -> None:
        self.model = model
        self.activations: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._handles: list = []
        self._register(layer_names)

    def _register(self, layer_names: list[str]) -> None:
        named = dict(self.model.named_modules())
        for name in layer_names:
            if name not in named:
                raise ValueError(f"Layer '{name}' not found in model.")
            handle = named[name].register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def _make_hook(self, name: str) -> Callable:
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()


# Usage
model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)
# Name the layers for hook registration
model[0].register_forward_hook  # works on any nn.Module

extractor = ActivationExtractor(model, layer_names=["0", "2"])
x = torch.randn(4, 128)
with torch.no_grad():
    _ = model(x)

print(extractor.activations["0"].shape)  # (4, 64)
print(extractor.activations["2"].shape)  # (4, 10)
extractor.remove()
```

### Gradient Analysis Hook

```python
class GradientMonitor:
    """Monitor gradient norms per layer during backward pass."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.grad_norms: dict[str, list[float]] = {}
        self._handles: list = []
        self._register_all()

    def _register_all(self) -> None:
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules only
                handle = module.register_full_backward_hook(
                    self._make_grad_hook(name)
                )
                self._handles.append(handle)
                self.grad_norms[name] = []

    def _make_grad_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            for g in grad_output:
                if g is not None:
                    self.grad_norms[name].append(g.norm().item())
        return hook

    def summary(self) -> dict[str, float]:
        return {
            name: sum(norms) / len(norms)
            for name, norms in self.grad_norms.items()
            if norms
        }

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
```

---

## W&B and TensorBoard Integration as Observers

Experiment tracking tools integrate naturally as callbacks — pure observers that react to training events without coupling the trainer to any specific logging backend.

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import wandb as wandb_module
    from torch.utils.tensorboard import SummaryWriter


class WandbCallback(TrainingCallback):
    """Log metrics to Weights & Biases on each epoch end."""

    def __init__(
        self,
        project: str,
        name: str | None = None,
        config: dict | None = None,
        watch_model: bool = True,
        log_gradients: bool = False,
    ) -> None:
        self.project = project
        self.name = name
        self.config = config or {}
        self.watch_model = watch_model
        self.log_gradients = log_gradients
        self._run = None

    def on_train_start(self, trainer: Trainer) -> None:
        import wandb
        self._run = wandb.init(
            project=self.project,
            name=self.name,
            config=self.config,
        )
        if self.watch_model:
            log = "gradients" if self.log_gradients else None
            wandb.watch(trainer.model, log=log, log_freq=50)

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict) -> None:
        if self._run is not None:
            import wandb
            wandb.log({"epoch": epoch, **logs})

    def on_train_end(self, trainer: Trainer) -> None:
        if self._run is not None:
            self._run.finish()


class TensorBoardCallback(TrainingCallback):
    """Log scalars, histograms, and graphs to TensorBoard."""

    def __init__(
        self,
        log_dir: str = "runs",
        log_histograms: bool = True,
        histogram_freq: int = 5,
    ) -> None:
        self.log_dir = log_dir
        self.log_histograms = log_histograms
        self.histogram_freq = histogram_freq
        self._writer: "SummaryWriter | None" = None

    def on_train_start(self, trainer: Trainer) -> None:
        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=self.log_dir)
        # Log compute graph
        try:
            dummy = torch.randn(1, *trainer.model.input_shape)
            self._writer.add_graph(trainer.model, dummy)
        except Exception:
            pass

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict) -> None:
        if self._writer is None:
            return
        for key, value in logs.items():
            self._writer.add_scalar(key, value, global_step=epoch)
        if self.log_histograms and epoch % self.histogram_freq == 0:
            for name, param in trainer.model.named_parameters():
                self._writer.add_histogram(name, param, global_step=epoch)
                if param.grad is not None:
                    self._writer.add_histogram(
                        f"{name}.grad", param.grad, global_step=epoch
                    )

    def on_train_end(self, trainer: Trainer) -> None:
        if self._writer:
            self._writer.close()
```

---

## Typed Event Bus for ML Pipelines

For more complex pipelines where multiple subsystems need to communicate, a full **event bus** provides publish-subscribe decoupling with typed events.

```python
from __future__ import annotations
import dataclasses
from collections import defaultdict
from typing import Callable, Generic, Type, TypeVar

E = TypeVar("E")


@dataclasses.dataclass
class Event:
    """Base class for all domain events."""
    source: str = ""


@dataclasses.dataclass
class EpochEndEvent(Event):
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float | None = None


@dataclasses.dataclass
class ModelSavedEvent(Event):
    checkpoint_path: str = ""
    metric: float = 0.0


@dataclasses.dataclass
class TrainingStoppedEvent(Event):
    reason: str = "completed"
    final_epoch: int = 0


class EventBus:
    """Thread-safe typed publish-subscribe event bus."""

    def __init__(self) -> None:
        self._subscribers: dict[type, list[Callable]] = defaultdict(list)
        self._history: list[Event] = []

    def subscribe(
        self, event_type: Type[E], handler: Callable[[E], None]
    ) -> None:
        """Register handler to receive events of event_type."""
        self._subscribers[event_type].append(handler)

    def unsubscribe(
        self, event_type: Type[E], handler: Callable[[E], None]
    ) -> None:
        self._subscribers[event_type].remove(handler)

    def publish(self, event: Event) -> None:
        """Dispatch event to all registered handlers synchronously."""
        self._history.append(event)
        for handler in self._subscribers.get(type(event), []):
            handler(event)

    def replay(self, event_type: Type[E] | None = None) -> list[Event]:
        """Return event history, optionally filtered by type."""
        if event_type is None:
            return list(self._history)
        return [e for e in self._history if isinstance(e, event_type)]


# ---------------------------------------------------------------------------
# Usage in a pipeline
# ---------------------------------------------------------------------------

bus = EventBus()

# Subscribe handlers
bus.subscribe(EpochEndEvent, lambda e: print(f"Metric tracker: epoch={e.epoch}"))
bus.subscribe(EpochEndEvent, lambda e: print(f"Alerter: loss={e.train_loss:.4f}"))
bus.subscribe(ModelSavedEvent, lambda e: print(f"Registry: saved {e.checkpoint_path}"))
bus.subscribe(TrainingStoppedEvent, lambda e: print(f"Notifier: {e.reason}"))

# Publish from anywhere in the pipeline
bus.publish(EpochEndEvent(source="Trainer", epoch=1, train_loss=0.45, val_loss=0.51))
bus.publish(ModelSavedEvent(source="Checkpoint", checkpoint_path="ckpt/best.pt", metric=0.51))
bus.publish(TrainingStoppedEvent(source="EarlyStopping", reason="patience_exhausted", final_epoch=12))
```

---

## PyTorch Lightning's Callback System

Lightning's callback system is a polished, production-grade Observer implementation. The `Trainer` (Subject) fires dozens of precisely defined hooks; `Callback` subclasses are Observers.

```python
import pytorch_lightning as pl
import torch


class MetricsPlotCallback(pl.Callback):
    """A Lightning callback that accumulates and plots metrics."""

    def __init__(self) -> None:
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Pull model from the subject (trainer/pl_module)
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.train_losses.append(loss.item())

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        loss = trainer.callback_metrics.get("val_loss")
        if loss is not None:
            self.val_losses.append(loss.item())

    def on_train_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Could use matplotlib here; omitted for brevity
        print(f"Final train loss: {self.train_losses[-1]:.4f}")
        print(f"Final val   loss: {self.val_losses[-1]:.4f}")


class GradientClipMonitor(pl.Callback):
    """Warn when gradients are large before clipping."""

    def on_before_optimizer_step(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer
    ) -> None:
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        if total_norm > 10.0:
            pl_module.log("grad_norm_warning", total_norm)
```

Lightning hooks are more granular than the minimal callback above — there are over 40 hook methods covering optimizer steps, gradient clipping, prediction loops, and more. Every hook follows the Observer contract: the Trainer notifies; the Callback reacts.

---

## Flow Diagram

<div class="diagram">
  <div class="diagram-title">Trainer (Subject) → Observers</div>
  <div class="flow">
    <div class="flow-node accent wide">Trainer (Subject)<br><small>fires lifecycle events</small></div>
    <div class="flow-arrow accent">▼ notify(event)</div>
    <div class="flow-h">
      <div class="flow-node green narrow">EarlyStopping<br>Callback</div>
      <div class="flow-node blue narrow">Checkpoint<br>Callback</div>
      <div class="flow-node purple narrow">Logger /<br>W&B Callback</div>
      <div class="flow-node orange narrow">LRScheduler<br>Callback</div>
    </div>
  </div>
  <p style="text-align:center;margin-top:0.5rem;font-size:0.85rem;opacity:0.7;">Each observer reacts independently — no observer knows about the others.</p>
</div>

---

## Comparison Table

<table class="compare-table">
  <thead>
    <tr>
      <th>Dimension</th>
      <th>GoF Observer</th>
      <th>Callback List</th>
      <th>Event Bus</th>
      <th>PyTorch Hook</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Coupling</td>
      <td>Low (interface)</td>
      <td>Low (ABC)</td>
      <td>Very low (type only)</td>
      <td>Very low (function)</td>
    </tr>
    <tr>
      <td>Ordering</td>
      <td>Registration order</td>
      <td>Registration order</td>
      <td>Subscription order</td>
      <td>Registration order</td>
    </tr>
    <tr>
      <td>Event types</td>
      <td>Single update()</td>
      <td>Named methods</td>
      <td>Typed event classes</td>
      <td>forward / backward</td>
    </tr>
    <tr>
      <td>History / replay</td>
      <td>No</td>
      <td>No</td>
      <td>Yes (if stored)</td>
      <td>No</td>
    </tr>
    <tr>
      <td>Thread safety</td>
      <td>Manual</td>
      <td>Manual</td>
      <td>Lock required</td>
      <td>PyTorch-managed</td>
    </tr>
    <tr>
      <td>ML use case</td>
      <td>Generic model state</td>
      <td>Training lifecycle</td>
      <td>Distributed pipelines</td>
      <td>Activation / grad probing</td>
    </tr>
  </tbody>
</table>

---

## Thread-Safety in Observers

<div class="callout warn">
<strong>Warning — concurrent observer lists</strong><br>
If training runs on multiple threads (e.g., DataLoader workers, distributed setups, async logging), the observer list can be mutated while iteration is in progress. Use a <code>threading.Lock</code> around <code>_notify</code> and list mutations, or iterate over a snapshot copy.

```python
import threading

class ThreadSafeSubject:
    def __init__(self):
        self._observers = []
        self._lock = threading.Lock()

    def attach(self, obs):
        with self._lock:
            self._observers.append(obs)

    def _notify(self, *args, **kwargs):
        with self._lock:
            snapshot = list(self._observers)   # iterate a copy
        for obs in snapshot:
            obs.update(*args, **kwargs)
```

PyTorch Lightning handles this by restricting all callback calls to the main process in multi-GPU scenarios — a sound design decision worth copying.
</div>

---

## Key Takeaways

<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">🔔</div>
    <div class="card-title">Decoupling</div>
    <div class="card-desc">Subject knows nothing about observer implementations — add or remove callbacks without changing the Trainer.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🔗</div>
    <div class="card-title">Hooks vs Callbacks</div>
    <div class="card-desc">PyTorch hooks are function-level observers for tensor data; callbacks are object-level observers for training state.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">📡</div>
    <div class="card-title">Event Bus</div>
    <div class="card-desc">For multi-component pipelines, a typed event bus decouples publishers from subscribers better than direct callback lists.</div>
  </div>
</div>

---

*Last updated: May 2026*
