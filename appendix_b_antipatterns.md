---
title: "Appendix B — Anti-Patterns in ML"
---

[← Back to Table of Contents](./README.md)

# Appendix B — Anti-Patterns in ML

> *"Every anti-pattern has a tell: the code that seems to work, until it doesn't."*

<span class="badge behavioral">Behavioral</span> <span class="badge mlops">MLOps</span>

---

## Overview

An **anti-pattern** is a commonly used solution that appears reasonable but consistently causes problems in practice. This appendix catalogs the ten most common ML anti-patterns, each with symptoms, consequences, and a refactored solution.

---

## B.1 God Module

<span class="badge structural">Structural</span>

**Description:** A single `nn.Module` (or class) that contains the entire model, data preprocessing, training loop, evaluation logic, and prediction serving — hundreds or thousands of lines in one class.

**Symptoms:**
- `class MyModel(nn.Module)` has methods `train_epoch`, `load_data`, `evaluate`, `save`, `predict`, `visualise`
- Cannot unit-test any component independently
- Adding a new loss function requires understanding the entire class

**Consequences:** Untestable, unreusable, merge-conflict-prone, and impossible to swap components.

**Refactoring:** Apply *Single Responsibility Principle* and *Composite* pattern.

```python
# ❌ BEFORE — God Module
class GodModel(nn.Module):
    def __init__(self, data_path, lr=1e-3, epochs=10):
        super().__init__()
        self.data_path = data_path
        self.lr = lr
        self.epochs = epochs
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.fc = nn.Linear(64, 10)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def load_data(self):
        # 50 lines of data loading
        ...

    def forward(self, x):
        x = self.conv1(x)
        return self.fc(x.flatten(1))

    def train_loop(self):
        for epoch in range(self.epochs):
            for batch in self.load_data():
                # inline training logic
                ...

    def evaluate(self):
        # evaluation logic mixed with training state
        ...


# ✅ AFTER — Separated Responsibilities
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class ImageClassifier(nn.Module):
    """Pure model: only forward pass."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x).flatten(1))


class Trainer:
    """Handles the training loop only."""
    def __init__(self, model: nn.Module, optimizer, loss_fn, device="cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(x), y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)


class Evaluator:
    """Handles evaluation only."""
    def __init__(self, model: nn.Module, device="cpu"):
        self.model = model.to(device)
        self.device = device

    def accuracy(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total if total else 0.0
```

---

## B.2 Spaghetti Training Loop

<span class="badge behavioral">Behavioral</span>

**Description:** All training logic — data loading, model forward pass, loss computation, metric accumulation, checkpointing, learning-rate scheduling, early stopping, and logging — crammed into a single 400–600 line function.

**Symptoms:**
- `def train(): ...` is more than 100 lines
- Nested `if/else` blocks for different modes
- Global variables mutated inside the loop

**Consequences:** Impossible to reuse, debug, or extend any single concern.

**Refactoring:** Apply *Template Method* pattern; extract concerns into callbacks or handler objects.

```python
# ❌ BEFORE
def train(model, data, epochs, lr, save_path, log_path, patience, ...):
    optimizer = ...
    scheduler = ...
    best_loss = float("inf")
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        for batch in data["train"]:
            # forward, backward, step, accumulate metrics ...
            if step % 100 == 0:
                # log to file ...
                pass
        val_loss = ...
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
        scheduler.step()


# ✅ AFTER — Template Method + Callbacks
from abc import ABC, abstractmethod

class TrainingCallback:
    def on_epoch_end(self, epoch: int, metrics: dict) -> bool:
        """Return True to stop training."""
        return False

class EarlyStoppingCallback(TrainingCallback):
    def __init__(self, patience: int = 5, monitor: str = "val_loss"):
        self.patience = patience
        self.monitor = monitor
        self._best = float("inf")
        self._no_improve = 0

    def on_epoch_end(self, epoch: int, metrics: dict) -> bool:
        val = metrics.get(self.monitor, float("inf"))
        if val < self._best:
            self._best = val
            self._no_improve = 0
        else:
            self._no_improve += 1
        return self._no_improve >= self.patience

class CheckpointCallback(TrainingCallback):
    def __init__(self, path: str, monitor: str = "val_loss"):
        self.path = path
        self.monitor = monitor
        self._best = float("inf")

    def on_epoch_end(self, epoch: int, metrics: dict) -> bool:
        val = metrics.get(self.monitor, float("inf"))
        if val < self._best:
            self._best = val
            import torch
            torch.save(metrics.get("model_state"), self.path)
        return False

class TrainingLoop:
    def __init__(self, trainer, evaluator, scheduler=None, callbacks=None):
        self.trainer = trainer
        self.evaluator = evaluator
        self.scheduler = scheduler
        self.callbacks = callbacks or []

    def fit(self, train_loader, val_loader, epochs: int) -> list[dict]:
        history = []
        for epoch in range(epochs):
            train_loss = self.trainer.train_epoch(train_loader)
            val_acc = self.evaluator.accuracy(val_loader)
            metrics = {"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc}
            history.append(metrics)
            if self.scheduler:
                self.scheduler.step()
            if any(cb.on_epoch_end(epoch, metrics) for cb in self.callbacks):
                break
        return history
```

---

## B.3 Hardcoded Config

<span class="badge mlops">MLOps</span>

**Description:** Magic numbers — learning rates, batch sizes, architecture dimensions, file paths — are embedded directly in training code rather than centralised in a config.

**Symptoms:**
- `lr = 0.0003` appears in five different files
- Experiments differ only by hand-edited numbers
- Reproducing a previous run requires archaeology

**Consequences:** Non-reproducible experiments; easy to introduce inconsistencies across files.

**Refactoring:** Use *Configuration* pattern (Chapter 29) — OmegaConf, Hydra, or Pydantic.

```python
# ❌ BEFORE
def build_model():
    return nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=512, nhead=8, dropout=0.1),
        num_layers=6,
    )

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.98))


# ✅ AFTER
from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dropout: float = 0.1

@dataclass
class TrainConfig:
    lr: float = 3e-4
    betas: tuple = (0.9, 0.98)
    batch_size: int = 64
    max_epochs: int = 100
    model: ModelConfig = ModelConfig()

def build_model(cfg: ModelConfig) -> nn.Module:
    return nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.nhead, dropout=cfg.dropout),
        num_layers=cfg.num_layers,
    )
```

---

## B.4 Data Leakage

<span class="badge mlops">MLOps</span>

**Description:** Information from the test or validation set influences the training process — through normalisation statistics computed on the full dataset, feature selection driven by test labels, or preprocessing fitted on all data.

**Symptoms:**
- `scaler.fit(X)` where `X` includes test rows
- `SelectKBest` fitted before train/test split
- Suspiciously high test accuracy that drops in production

**Consequences:** Overly optimistic evaluation metrics; models that fail in deployment.

```python
# ❌ BEFORE — leakage through full-dataset scaler
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # ← test data included!

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)


# ✅ AFTER — fit scaler only on train split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)   # fit on train only
X_test = scaler.transform(X_test_raw)          # transform test with train statistics
```

---

## B.5 Premature Optimization

<span class="badge pytorch">PyTorch</span>

**Description:** Spending engineering effort on micro-optimisations (custom CUDA kernels, manual memory management, fused ops) before profiling to identify actual bottlenecks.

**Symptoms:**
- Weeks spent on a 2% speedup while the data loader is the 10× bottleneck
- `torch.compile` or `@torch.jit.script` applied everywhere without measurement

**Refactoring:** Profile first; optimise the measured bottleneck.

```python
# ✅ Profile before optimising
import torch
import torch.profiler as profiler

def profile_training_step(model, batch, loss_fn, optimizer):
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        optimizer.zero_grad()
        out = model(batch["input"])
        loss = loss_fn(out, batch["target"])
        loss.backward()
        optimizer.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # Only THEN decide what to optimise
```

---

## B.6 Experiment Graveyard

<span class="badge mlops">MLOps</span>

**Description:** Dozens of uncommitted experiment scripts (`train_v2_final_REAL_fixed3.py`) litter the repository with no record of what changed, what results were obtained, or whether they ran successfully.

**Symptoms:**
- Files named `train_backup.py`, `run_exp_42_modified.py`
- No tags or branches for experiment checkpoints
- "Which script produced the results in the paper?" is unanswerable

**Refactoring:** Use *Experiment Tracking* pattern (Chapter 30) and commit experiment configs, not script copies.

```python
# ✅ Track every experiment, never copy scripts
import mlflow

with mlflow.start_run(run_name="transformer_lr3e4_dropout0.1"):
    mlflow.log_params({"lr": 3e-4, "dropout": 0.1, "model": "transformer"})
    # ... training ...
    mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.pytorch.log_model(model, "model")
# All variants are tracked; no file duplication needed.
```

---

## B.7 Copy-Paste ML

<span class="badge mlops">MLOps</span>

**Description:** Duplicating entire training scripts and changing only two lines to try a different model architecture or dataset — leading to subtle divergence between scripts that should be identical.

**Symptoms:**
- `train_resnet.py` and `train_vgg.py` differ by one import and one line
- Bug fixes applied to one script but not all copies
- `grep` finds the same function implemented 6 slightly different ways

**Refactoring:** Extract common logic into shared modules; parametrise differences.

```python
# ❌ BEFORE — six near-identical training scripts

# ✅ AFTER — one script, parametrised
import torch.nn as nn

MODEL_REGISTRY = {
    "resnet18": lambda: torchvision.models.resnet18(pretrained=False),
    "vgg16":    lambda: torchvision.models.vgg16(pretrained=False),
    "efficientnet": lambda: torchvision.models.efficientnet_b0(pretrained=False),
}

def main(model_name: str, cfg: TrainConfig):
    model = MODEL_REGISTRY[model_name]()
    trainer = Trainer(model, ...)
    loop = TrainingLoop(trainer, ...)
    loop.fit(train_loader, val_loader, cfg.max_epochs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), required=True)
    args = parser.parse_args()
    main(args.model, TrainConfig())
```

---

## B.8 Metric Hacking

<span class="badge behavioral">Behavioral</span>

**Description:** Optimising a proxy metric (accuracy, F1 on a held-out set) through hyperparameter search until it looks great on the leaderboard, without understanding whether the metric reflects real-world performance.

**Symptoms:**
- 99% accuracy on a dataset that is 99% class-negative (trivial classifier)
- Optimising BLEU without checking whether translations are actually fluent
- Overfitting the validation set through repeated hyperparameter tuning

**Refactoring:** Track *multiple* diverse metrics; use out-of-sample evaluation, human evaluation, and business metrics in tandem.

```python
# ✅ Multi-metric evaluation to prevent metric hacking
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)

def full_evaluation(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1_macro":  f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics
```

---

## B.9 Silent Failure

<span class="badge mlops">MLOps</span>

**Description:** A bare `except Exception: pass` (or `continue`) swallows errors silently. Training proceeds with NaN loss, corrupted batches, or failed data loading — producing a model that appears to train but learns nothing.

**Symptoms:**
- Loss is always exactly `0.0` or `nan` after epoch 1
- Validation accuracy never improves despite many epochs
- No error messages despite obviously wrong outputs

**Consequences:** Wasted compute; silent model corruption; silent data pipeline failures.

```python
# ❌ BEFORE — silent failure
for batch in loader:
    try:
        loss = model(batch)
        loss.backward()
    except Exception:
        continue  # silently skip broken batches, loss may be NaN


# ✅ AFTER — explicit error handling with monitoring
import math
import logging

logger = logging.getLogger(__name__)

def safe_train_step(model, batch, optimizer, loss_fn, step: int) -> float:
    try:
        optimizer.zero_grad()
        out = model(batch["input"])
        loss = loss_fn(out, batch["target"])

        if not math.isfinite(loss.item()):
            logger.error("Non-finite loss %.4f at step %d — stopping.", loss.item(), step)
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

        loss.backward()

        # Check for exploding gradients
        total_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters()
                         if p.grad is not None) ** 0.5
        if total_norm > 1e4:
            logger.warning("Large gradient norm %.2f at step %d", total_norm, step)

        optimizer.step()
        return loss.item()

    except Exception:
        logger.exception("Training step %d failed", step)
        raise  # never swallow silently
```

---

## B.10 Monolithic Pipeline

<span class="badge mlops">MLOps</span>

**Description:** A single script that reads raw data, cleans it, engineers features, trains the model, evaluates it, and writes predictions — all in sequence with no boundaries between stages.

**Symptoms:**
- Re-running the full script to change only the model architecture (re-runs expensive data processing)
- Cannot run stages independently or in parallel
- Debugging requires stepping through 800 lines

**Refactoring:** Apply *Data Pipeline* pattern (Chapter 28) — separate stages with defined interfaces, caching, and independent execution.

```python
# ✅ AFTER — staged pipeline with caching
from pathlib import Path
import pickle

class PipelineStage:
    def __init__(self, name: str, cache_dir: Path):
        self.name = name
        self.cache_path = cache_dir / f"{name}.pkl"

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def cached_run(self, *args, force: bool = False, **kwargs):
        if self.cache_path.exists() and not force:
            with open(self.cache_path, "rb") as f:
                return pickle.load(f)
        result = self.run(*args, **kwargs)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(result, f)
        return result

class IngestStage(PipelineStage):
    def run(self, raw_path: str): ...

class FeatureStage(PipelineStage):
    def run(self, data): ...

class TrainStage(PipelineStage):
    def run(self, features): ...

def run_pipeline(raw_path: str, cache_dir: str = ".cache"):
    cache = Path(cache_dir)
    data     = IngestStage("ingest", cache).cached_run(raw_path)
    features = FeatureStage("features", cache).cached_run(data)
    model    = TrainStage("train", cache).cached_run(features)
    return model
```

---

## B.11 Summary Table

<table class="compare-table">
  <thead>
    <tr>
      <th>Anti-Pattern</th>
      <th>Primary Symptom</th>
      <th>Fix / Pattern</th>
      <th>Chapter Ref</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>God Module</strong></td>
      <td>1000-line class mixing model + training + data</td>
      <td>SRP + Composite</td>
      <td>02, 09</td>
    </tr>
    <tr>
      <td><strong>Spaghetti Loop</strong></td>
      <td>500-line train() function</td>
      <td>Template Method + Callbacks</td>
      <td>16, 17</td>
    </tr>
    <tr>
      <td><strong>Hardcoded Config</strong></td>
      <td>Magic numbers everywhere</td>
      <td>Configuration pattern</td>
      <td>29</td>
    </tr>
    <tr>
      <td><strong>Data Leakage</strong></td>
      <td>Scaler fitted on full dataset</td>
      <td>Strict train/val/test separation</td>
      <td>28</td>
    </tr>
    <tr>
      <td><strong>Premature Optimisation</strong></td>
      <td>Micro-optimising before profiling</td>
      <td>Profile first</td>
      <td>25</td>
    </tr>
    <tr>
      <td><strong>Experiment Graveyard</strong></td>
      <td>train_final_v3_real.py proliferation</td>
      <td>Experiment Tracking</td>
      <td>30</td>
    </tr>
    <tr>
      <td><strong>Copy-Paste ML</strong></td>
      <td>Duplicated scripts differ by 2 lines</td>
      <td>Registry + parametrisation</td>
      <td>06, 15</td>
    </tr>
    <tr>
      <td><strong>Metric Hacking</strong></td>
      <td>Great numbers, bad real-world performance</td>
      <td>Multi-metric + out-of-sample eval</td>
      <td>30</td>
    </tr>
    <tr>
      <td><strong>Silent Failure</strong></td>
      <td>NaN loss, training proceeds anyway</td>
      <td>Explicit error handling + NaN checks</td>
      <td>32</td>
    </tr>
    <tr>
      <td><strong>Monolithic Pipeline</strong></td>
      <td>Full re-run to change one stage</td>
      <td>Data Pipeline + stage caching</td>
      <td>28</td>
    </tr>
  </tbody>
</table>

<div class="callout warn">
  <div class="callout-icon">⚠️</div>
  <div class="callout-body">
    Anti-patterns are not crimes — they are useful shortcuts that have gone too far. Recognise them early and refactor incrementally rather than attempting a big-bang rewrite.
  </div>
</div>

---

*Last updated: May 2026*
