---
title: "Chapter 12 — Facade & Service Layer"
---
[← Back to Table of Contents](./README.md)

# Chapter 12 — Facade & Service Layer

> *"Provide a unified interface to a set of interfaces in a subsystem. Facade defines a higher-level interface that makes the subsystem easier to use."*
> — Gang of Four

---

## Intent and Motivation

Complex systems accrete complexity over time. A training pipeline eventually manages optimizers, schedulers, mixed-precision scaler, gradient accumulation, distributed process groups, logging backends, checkpoint saving, and early stopping — all at once. Without a Facade, every script that trains a model reimplements the same boilerplate in slightly different and incompatible ways.

**The Facade pattern** provides a single, simple entry point — `trainer.fit(model, dataset)` — that hides all of that machinery. The subsystems still exist and are still testable in isolation; they are simply orchestrated behind a clean surface.

<div class="diagram">
  <div class="diagram-title">The Complexity Iceberg</div>
  <div class="flow">
    <div class="flow-node accent wide">Client sees: <code>trainer.fit(model, data)</code></div>
    <div class="flow-arrow accent">↓ visible surface</div>
    <div class="flow-node blue wide" style="font-weight:bold">FACADE</div>
    <div class="flow-arrow" style="border-left: 3px dashed #888; height: 2rem"></div>
    <div class="flow-h">
      <div class="flow-node orange narrow">Optimizer<br>& Scheduler</div>
      <div class="flow-node purple narrow">AMP / FP16<br>Scaler</div>
      <div class="flow-node green narrow">DDP / FSDP<br>Process Group</div>
      <div class="flow-node teal narrow">Gradient<br>Accumulation</div>
      <div class="flow-node pink narrow">Logging<br>W&amp;B / TB</div>
      <div class="flow-node cyan narrow">Checkpoint<br>Manager</div>
    </div>
  </div>
</div>

---

## GoF Facade: UML Structure

<div class="diagram">
  <div class="diagram-title">Facade Pattern — UML</div>
  <div class="uml-row">
    <div class="uml-box">
      <div class="uml-title">Client</div>
      <div class="uml-section">
        <div class="uml-item">uses Facade only</div>
      </div>
    </div>
    <div class="uml-box" style="border-color: var(--accent)">
      <div class="uml-title">Facade</div>
      <div class="uml-section">
        <div class="uml-item">- subsystemA: SubsystemA</div>
        <div class="uml-item">- subsystemB: SubsystemB</div>
        <div class="uml-item">- subsystemC: SubsystemC</div>
      </div>
      <div class="uml-section">
        <div class="uml-item">+ simpleOperation()</div>
      </div>
    </div>
  </div>
  <div class="uml-row" style="margin-top:1rem">
    <div class="uml-box">
      <div class="uml-title">SubsystemA</div>
      <div class="uml-section">
        <div class="uml-item">+ operationA1()</div>
        <div class="uml-item">+ operationA2()</div>
      </div>
    </div>
    <div class="uml-box">
      <div class="uml-title">SubsystemB</div>
      <div class="uml-section">
        <div class="uml-item">+ operationB1()</div>
      </div>
    </div>
    <div class="uml-box">
      <div class="uml-title">SubsystemC</div>
      <div class="uml-section">
        <div class="uml-item">+ operationC1()</div>
        <div class="uml-item">+ operationC2()</div>
      </div>
    </div>
  </div>
</div>

---

## HuggingFace `Trainer` as a Facade

The HuggingFace `Trainer` is one of the most widely deployed Facade instances in all of ML. Behind a three-line call, it manages a staggering amount of complexity:

<div class="diagram-grid cols-3">
  <div class="diagram-card orange">
    <div class="card-icon">⚙️</div>
    <div class="card-title">Optimizer & Scheduler</div>
    <div class="card-desc">AdamW with weight decay, linear warmup + cosine decay, gradient accumulation steps — configured from <code>TrainingArguments</code></div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">⚡</div>
    <div class="card-title">Mixed Precision (AMP)</div>
    <div class="card-desc">FP16 / BF16 GradScaler, loss scaling, dynamic scaling factors, overflow detection</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🌐</div>
    <div class="card-title">Distributed Training</div>
    <div class="card-desc">DDP, FSDP, DeepSpeed ZeRO integration; process group init; gradient synchronisation across ranks</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">💾</div>
    <div class="card-title">Gradient Checkpointing</div>
    <div class="card-desc">Activation recomputation to trade compute for memory; enabled with one flag</div>
  </div>
  <div class="diagram-card teal">
    <div class="card-icon">📊</div>
    <div class="card-title">Logging & Reporting</div>
    <div class="card-desc">W&B, TensorBoard, MLflow, Comet integration; evaluation metrics; progress bars via tqdm</div>
  </div>
  <div class="diagram-card pink">
    <div class="card-icon">📁</div>
    <div class="card-title">Checkpoint Management</div>
    <div class="card-desc">Save-best-model logic, resume from checkpoint, model card generation, push-to-Hub</div>
  </div>
</div>

```python
# What the client sees — the Facade surface:
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    fp16=True,                          # triggers AMP subsystem
    gradient_accumulation_steps=4,      # triggers accumulation subsystem
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",                  # triggers logging subsystem
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics,
)

trainer.train()   # ← one call, ~2000 lines of subsystem code executes
```

---

## HuggingFace `pipeline()` as a Facade

`pipeline()` is a lighter Facade — it fuses tokenization, model inference, and post-processing into one callable:

```python
from transformers import pipeline

# Everything hidden: tokenizer init, padding, truncation,
# attention masks, model forward, argmax/softmax, label mapping
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

result = classifier("The training loss finally converged!")
# [{'label': 'POSITIVE', 'score': 0.9997}]

# What pipeline() actually does (simplified):
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

def manual_pipeline_equivalent(text: str, model_name: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Tokenisation subsystem
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    # Inference subsystem
    with torch.no_grad():
        logits = model(**inputs).logits

    # Post-processing subsystem
    probs = F.softmax(logits, dim=-1)
    label_id = probs.argmax(dim=-1).item()
    return {
        "label": model.config.id2label[label_id],
        "score": probs[0][label_id].item(),
    }
```

---

## Building Your Own Training Facade

A complete, production-style `train()` function that hides all training complexity:

```python
"""
training_facade.py — A Facade that hides optimizer, scheduler, AMP,
gradient clipping, evaluation, and checkpointing.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Public configuration surface of the Facade."""
    epochs: int = 10
    lr: float = 3e-4
    batch_size: int = 32
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = torch.cuda.is_available()
    gradient_accumulation_steps: int = 1
    eval_every_n_epochs: int = 1
    save_best: bool = True
    output_dir: str = "./checkpoints"
    patience: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    warmup_ratio: float = 0.1


def train(
    model: nn.Module,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    config: Optional[TrainConfig] = None,
    compute_metrics: Optional[Callable] = None,
) -> dict:
    """
    Facade entry point: train a model with one function call.

    Hides:
    - DataLoader construction with pinned memory and prefetch
    - Optimizer and learning rate scheduler setup
    - Mixed-precision (AMP) GradScaler
    - Gradient accumulation
    - Gradient norm clipping
    - Evaluation loop
    - Best-model checkpointing
    - Early stopping
    """
    cfg = config or TrainConfig()
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    model = model.to(cfg.device)

    # ── Subsystem: DataLoaders ────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device == "cuda",
        drop_last=True,
    )
    eval_loader = (
        DataLoader(eval_dataset, batch_size=cfg.batch_size * 2,
                   num_workers=cfg.num_workers, pin_memory=cfg.device == "cuda")
        if eval_dataset else None
    )

    # ── Subsystem: Optimizer ──────────────────────────────────────────────
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(params, lr=cfg.lr)

    # ── Subsystem: LR Scheduler ───────────────────────────────────────────
    total_steps = len(train_loader) * cfg.epochs // cfg.gradient_accumulation_steps
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
    )

    # ── Subsystem: AMP ────────────────────────────────────────────────────
    scaler = GradScaler(enabled=cfg.fp16)

    # ── Training loop (orchestration) ────────────────────────────────────
    best_metric = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "eval_metrics": []}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(cfg.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with autocast(enabled=cfg.fp16):
                outputs = model(**batch)
                loss = outputs.loss / cfg.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * cfg.gradient_accumulation_steps

        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)
        logger.info(f"Epoch {epoch}/{cfg.epochs} — loss: {avg_loss:.4f}")

        # ── Evaluation & checkpointing ────────────────────────────────
        if eval_loader and epoch % cfg.eval_every_n_epochs == 0:
            metrics = _evaluate(model, eval_loader, cfg, compute_metrics)
            history["eval_metrics"].append(metrics)
            logger.info(f"  eval: {metrics}")

            monitor = metrics.get("loss", avg_loss)
            if cfg.save_best and monitor < best_metric:
                best_metric = monitor
                patience_counter = 0
                torch.save(model.state_dict(),
                           os.path.join(cfg.output_dir, "best_model.pt"))
                logger.info(f"  ✓ saved best model (metric={monitor:.4f})")
            else:
                patience_counter += 1

            if patience_counter >= cfg.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    return history


def _evaluate(model, loader, cfg, compute_metrics):
    """Private subsystem — not exposed to callers."""
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(cfg.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            with autocast(enabled=cfg.fp16):
                outputs = model(**batch)
            total_loss += outputs.loss.item()
            if hasattr(outputs, "logits"):
                all_preds.extend(outputs.logits.argmax(-1).cpu().tolist())
            if "labels" in batch:
                all_labels.extend(batch["labels"].cpu().tolist())

    result = {"loss": total_loss / len(loader)}
    if compute_metrics and all_preds:
        result.update(compute_metrics({"predictions": all_preds, "label_ids": all_labels}))
    return result
```

---

## Service Layer Pattern

The Service Layer combines a Facade with business logic. It is the standard pattern for ML serving:

```python
"""
model_service.py — Service Layer that hides preprocessing, batching,
inference, and postprocessing behind a clean predict() surface.
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    predictions: list[Any]
    probabilities: list[list[float]]
    latency_ms: float
    model_version: str


class ModelService:
    """
    Service Layer / Facade for model inference.

    The caller only sees: service.predict(texts) → PredictionResult
    Everything else is internal subsystem logic.
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        label_map: dict[int, str],
        model_version: str = "1.0.0",
        max_batch_size: int = 32,
        device: str = "cpu",
    ):
        self._model = model.eval().to(device)
        self._tokenizer = tokenizer
        self._label_map = label_map
        self._version = model_version
        self._max_batch = max_batch_size
        self._device = device

    def predict(self, texts: list[str]) -> PredictionResult:
        """
        Public Facade surface: texts → structured predictions.
        Hides: validation, batching, tokenisation, inference, postprocessing.
        """
        start = time.perf_counter()

        # Subsystem 1: Input validation
        texts = self._validate(texts)

        # Subsystem 2: Batching
        all_probs = []
        for batch_texts in self._batch(texts):
            # Subsystem 3: Tokenisation / preprocessing
            inputs = self._preprocess(batch_texts)
            # Subsystem 4: Inference
            logits = self._infer(inputs)
            # Subsystem 5: Postprocessing
            all_probs.extend(self._postprocess(logits))

        predictions = [self._label_map[np.argmax(p)] for p in all_probs]
        latency = (time.perf_counter() - start) * 1000

        return PredictionResult(
            predictions=predictions,
            probabilities=all_probs,
            latency_ms=latency,
            model_version=self._version,
        )

    # ── Private subsystems ────────────────────────────────────────────────
    def _validate(self, texts: list[str]) -> list[str]:
        if not texts:
            raise ValueError("texts must be non-empty")
        return [str(t)[:512] for t in texts]  # truncate silently

    def _batch(self, texts: list[str]):
        for i in range(0, len(texts), self._max_batch):
            yield texts[i : i + self._max_batch]

    def _preprocess(self, texts: list[str]) -> dict:
        return self._tokenizer(
            texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        )

    def _infer(self, inputs: dict) -> torch.Tensor:
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            return self._model(**inputs).logits

    def _postprocess(self, logits: torch.Tensor) -> list[list[float]]:
        import torch.nn.functional as F
        return F.softmax(logits, dim=-1).cpu().numpy().tolist()
```

---

## Anti-Corruption Layer

A Facade can serve as a **boundary** — an anti-corruption layer — between legacy code and a new system. Rather than letting old concepts leak into the new domain model, the Facade translates:

```python
class LegacyDataLoaderFacade:
    """
    Anti-corruption layer: translates the old batch loading API
    (returns numpy arrays, uses string-keyed dicts with odd conventions)
    into the modern torch-based interface the rest of the system expects.
    """
    def __init__(self, legacy_loader):
        self._legacy = legacy_loader   # old system we don't control

    def __iter__(self):
        for old_batch in self._legacy.get_next_batch():
            # translate old format → new format
            yield {
                "input_ids": torch.from_numpy(old_batch["token_ids"]).long(),
                "attention_mask": torch.from_numpy(old_batch["mask"]).bool().long(),
                "labels": torch.tensor(old_batch["label_ints"]).long(),
            }

    def __len__(self) -> int:
        return self._legacy.num_batches()
```

---

## Facade vs Adapter vs Mediator

<table class="compare-table">
  <thead>
    <tr>
      <th>Dimension</th>
      <th>Facade</th>
      <th>Adapter</th>
      <th>Mediator</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Purpose</strong></td>
      <td>Simplify interface to a <em>subsystem</em></td>
      <td>Make incompatible interface <em>compatible</em></td>
      <td>Reduce coupling between <em>peers</em></td>
    </tr>
    <tr>
      <td><strong>Direction</strong></td>
      <td>Client → Facade → many subsystems</td>
      <td>Client → Adapter → one adaptee</td>
      <td>Components ↔ Mediator ↔ Components</td>
    </tr>
    <tr>
      <td><strong>Subsystem knows facade?</strong></td>
      <td>No</td>
      <td>No</td>
      <td>Yes (registers with mediator)</td>
    </tr>
    <tr>
      <td><strong>New interface?</strong></td>
      <td>Yes — simpler, higher-level</td>
      <td>Yes — matches target interface</td>
      <td>Yes — orchestration interface</td>
    </tr>
    <tr>
      <td><strong>ML example</strong></td>
      <td><code>Trainer.train()</code></td>
      <td>HF Datasets ↔ torch Dataset</td>
      <td>Event bus in a training loop</td>
    </tr>
  </tbody>
</table>

---

## Flow Diagram

<div class="diagram">
  <div class="diagram-title">Facade Pattern — Runtime Flow</div>
  <div class="flow">
    <div class="flow-node accent wide">Client: <code>trainer.fit(model, data)</code></div>
    <div class="flow-arrow accent">↓</div>
    <div class="flow-node blue wide" style="font-weight:bold">Facade (Trainer)</div>
    <div class="flow-arrow blue">↓ orchestrates</div>
    <div class="flow-h">
      <div class="flow-node orange narrow">Subsystem 1<br>DataLoader</div>
      <div class="flow-node purple narrow">Subsystem 2<br>Optimizer</div>
      <div class="flow-node green narrow">Subsystem 3<br>AMP Scaler</div>
      <div class="flow-node teal narrow">Subsystem 4<br>Checkpointer</div>
    </div>
    <div class="flow-arrow blue">↑ returns history dict</div>
    <div class="flow-node accent wide">Client receives result</div>
  </div>
</div>

---

## Layered Architecture Diagram

<div class="diagram">
  <div class="diagram-title">Service Layer in a Layered Architecture</div>
  <div class="flow">
    <div class="flow-node accent wide">Presentation Layer<br><small>REST API, CLI, Gradio UI</small></div>
    <div class="flow-arrow accent">↓ calls</div>
    <div class="flow-node blue wide">Facade / Service Layer<br><small>ModelService.predict(), train(), evaluate()</small></div>
    <div class="flow-arrow blue">↓ uses</div>
    <div class="flow-node purple wide">Domain Layer<br><small>Model, Dataset, Metrics, Loss functions</small></div>
    <div class="flow-arrow purple">↓ persists via</div>
    <div class="flow-node green wide">Infrastructure Layer<br><small>File I/O, S3, Database, ONNX Runtime, GPU</small></div>
  </div>
</div>

Each layer depends only on the layer below it. The Facade is the *seam* between the presentation layer's simplicity and the domain/infrastructure complexity beneath.

---

*Last updated: May 2026*
