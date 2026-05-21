---
title: "Chapter 26 — Checkpoint & Serialization"
---

[← Back to Table of Contents](./README.md)

# Chapter 26 — Checkpoint & Serialization

> *"A model that can't be saved, resumed, or audited is a liability. A checkpoint is the contract between your training run and the rest of the world."*

<span class="badge mlops">MLOps</span> <span class="badge pytorch">PyTorch</span>

---

## Why Checkpointing Matters

Training a large model takes hours, days, or weeks. Hardware fails. Quotas expire. A new idea demands a mid-run pivot. Without disciplined checkpointing, every interruption means starting over from scratch.

<div class="diagram">
  <div class="diagram-title">Three Pillars of Checkpointing</div>
  <div class="diagram-grid cols-3">
    <div class="diagram-card teal">
      <div class="card-icon">🔁</div>
      <div class="card-title">Resumability</div>
      <div class="card-desc">Restart from the last saved epoch — preserve optimizer momentum, LR schedule, RNG state</div>
    </div>
    <div class="diagram-card blue">
      <div class="card-icon">🏷️</div>
      <div class="card-title">Versioning</div>
      <div class="card-desc">Every checkpoint is an immutable snapshot tied to config, git hash, and metrics</div>
    </div>
    <div class="diagram-card purple">
      <div class="card-icon">🚀</div>
      <div class="card-title">Deployment</div>
      <div class="card-desc">Export weights for inference — safetensors, TorchScript, ONNX, or HuggingFace Hub</div>
    </div>
  </div>
</div>

---

## `state_dict` Pattern

PyTorch serialization revolves around one concept: the **state dict** — an `OrderedDict` mapping parameter/buffer names to tensors.

<div class="diagram">
  <div class="diagram-title">What Lives in a state_dict</div>
  <div class="flow flow-h">
    <div class="flow-node blue wide">Model<br><small>nn.Module</small></div>
    <div class="flow-arrow">→</div>
    <div class="flow-node teal wide">Parameters<br><small>weight, bias tensors</small></div>
    <div class="flow-arrow">+</div>
    <div class="flow-node purple wide">Buffers<br><small>running_mean, running_var<br>non-grad tensors</small></div>
  </div>
</div>

```python
import torch
import torch.nn as nn
from collections import OrderedDict

class SmallNet(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.fc1(x)))
        return self.fc2(x)


model = SmallNet(784, 10)

# Inspect state_dict keys
for key, tensor in model.state_dict().items():
    print(f"{key:40s}  shape={tuple(tensor.shape)}  dtype={tensor.dtype}")

# Save — never pickle the whole module
torch.save(model.state_dict(), "model.pt")

# Load — always instantiate first, then load weights
restored = SmallNet(784, 10)
restored.load_state_dict(torch.load("model.pt", weights_only=True))
restored.eval()
```

<div class="callout tip">
  <span class="callout-icon">💡</span>
  <div class="callout-body">Always pass <code>weights_only=True</code> to <code>torch.load</code>. It restricts unpickling to tensors only, eliminating arbitrary code execution via crafted checkpoints.</div>
</div>

---

## Optimizer State Dict

Optimizer state is *as important* as model weights for resumable training. Adam stores first and second moment estimates; SGD stores momentum buffers. Without them, the optimizer "forgets" its accumulated history and training takes extra epochs to recover.

```python
import torch
import torch.optim as optim

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

# Inspect optimizer state_dict structure
opt_sd = optimizer.state_dict()
# Keys: "state" (per-param tensors) and "param_groups" (hyperparams)
print(opt_sd["param_groups"][0].keys())
# dict_keys(['lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'params'])

# After a few backward passes, state tensors appear
for loss_val in [dummy_loss() for _ in range(3)]:
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()

for param_idx, state in opt_sd["state"].items():
    print(f"  param {param_idx}: {list(state.keys())}")
    # step, exp_avg, exp_avg_sq

# Save + load optimizer state
torch.save(optimizer.state_dict(), "optimizer.pt")
optimizer.load_state_dict(torch.load("optimizer.pt", weights_only=False))
```

<div class="callout warn">
  <span class="callout-icon">⚠️</span>
  <div class="callout-body"><code>weights_only=False</code> is required for optimizer state dicts because they contain Python integers and dicts, not just tensors. Only load optimizer checkpoints from trusted sources.</div>
</div>

---

## Full `CheckpointManager` Class

A production checkpoint manager needs to handle rotation (keep last N), best-model tracking, atomic writes, and rich metadata.

```python
import os
import json
import shutil
import hashlib
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
import torch


@dataclass
class CheckpointMeta:
    epoch: int
    step: int
    train_loss: float
    val_loss: float
    val_metric: float
    config: dict[str, Any]
    git_hash: str
    timestamp: str
    pytorch_version: str = field(default_factory=lambda: torch.__version__)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, path: str | Path) -> "CheckpointMeta":
        with open(path) as f:
            return cls(**json.load(f))


def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


class CheckpointManager:
    """Manages model checkpoints with rotation, best-tracking, and metadata."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        keep_last_n: int = 3,
        metric_mode: str = "min",        # "min" or "max"
        metric_name: str = "val_loss",
    ) -> None:
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.keep_last_n = keep_last_n
        self.metric_mode = metric_mode
        self.metric_name = metric_name
        self._best_metric: float = float("inf") if metric_mode == "min" else float("-inf")
        self._saved_epochs: list[int] = []

    # ------------------------------------------------------------------
    def save(self, meta: CheckpointMeta) -> Path:
        epoch = meta.epoch
        epoch_dir = self.dir / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(exist_ok=True)

        # Atomic write: write to temp, then rename
        tmp_weights = epoch_dir / "weights.tmp"
        torch.save(self.model.state_dict(), tmp_weights)
        tmp_weights.rename(epoch_dir / "weights.pt")

        torch.save(self.optimizer.state_dict(), epoch_dir / "optimizer.pt")
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), epoch_dir / "scheduler.pt")

        (epoch_dir / "meta.json").write_text(meta.to_json())

        self._saved_epochs.append(epoch)
        self._rotate()
        self._maybe_update_best(meta, epoch_dir)
        return epoch_dir

    # ------------------------------------------------------------------
    def _rotate(self) -> None:
        """Delete oldest checkpoints beyond keep_last_n."""
        while len(self._saved_epochs) > self.keep_last_n:
            oldest = self._saved_epochs.pop(0)
            oldest_dir = self.dir / f"epoch_{oldest:04d}"
            if oldest_dir.exists():
                # Don't delete if it's the best symlink target
                best_link = self.dir / "best"
                if best_link.is_symlink() and best_link.resolve() == oldest_dir.resolve():
                    continue
                shutil.rmtree(oldest_dir)

    # ------------------------------------------------------------------
    def _maybe_update_best(self, meta: CheckpointMeta, epoch_dir: Path) -> None:
        metric = meta.val_metric
        is_better = (
            metric < self._best_metric
            if self.metric_mode == "min"
            else metric > self._best_metric
        )
        if is_better:
            self._best_metric = metric
            best_link = self.dir / "best"
            if best_link.is_symlink():
                best_link.unlink()
            best_link.symlink_to(epoch_dir.name)

    # ------------------------------------------------------------------
    def load(
        self,
        path: str | Path,
        strict: bool = True,
        map_location: Optional[str] = None,
    ) -> CheckpointMeta:
        epoch_dir = Path(path)
        device = map_location or ("cuda" if torch.cuda.is_available() else "cpu")

        weights = torch.load(epoch_dir / "weights.pt", weights_only=True, map_location=device)
        missing, unexpected = self.model.load_state_dict(weights, strict=strict)
        if missing:
            print(f"[CheckpointManager] Missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"[CheckpointManager] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

        self.optimizer.load_state_dict(
            torch.load(epoch_dir / "optimizer.pt", weights_only=False, map_location=device)
        )
        if self.scheduler is not None and (epoch_dir / "scheduler.pt").exists():
            self.scheduler.load_state_dict(
                torch.load(epoch_dir / "scheduler.pt", weights_only=False)
            )

        return CheckpointMeta.from_json(epoch_dir / "meta.json")

    # ------------------------------------------------------------------
    def load_best(self, **kwargs: Any) -> CheckpointMeta:
        best_link = self.dir / "best"
        if not best_link.exists():
            raise FileNotFoundError(f"No best checkpoint at {best_link}")
        return self.load(best_link.resolve(), **kwargs)

    # ------------------------------------------------------------------
    def load_latest(self, **kwargs: Any) -> CheckpointMeta:
        dirs = sorted(self.dir.glob("epoch_*"))
        if not dirs:
            raise FileNotFoundError(f"No checkpoints in {self.dir}")
        return self.load(dirs[-1], **kwargs)

    # ------------------------------------------------------------------
    def list_checkpoints(self) -> list[dict[str, Any]]:
        result = []
        for d in sorted(self.dir.glob("epoch_*")):
            meta_file = d / "meta.json"
            if meta_file.exists():
                result.append(json.loads(meta_file.read_text()))
        return result
```

**Using the manager in a training loop:**

```python
import datetime

ckpt_mgr = CheckpointManager(
    checkpoint_dir="checkpoints/run_001",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    keep_last_n=3,
    metric_mode="min",
)

for epoch in range(start_epoch, num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss, val_metric = evaluate(model, val_loader)

    meta = CheckpointMeta(
        epoch=epoch,
        step=epoch * len(train_loader),
        train_loss=train_loss,
        val_loss=val_loss,
        val_metric=val_metric,
        config={"lr": 3e-4, "batch_size": 64, "arch": "SmallNet"},
        git_hash=_get_git_hash(),
        timestamp=datetime.datetime.utcnow().isoformat(),
    )
    saved_path = ckpt_mgr.save(meta)
    print(f"Epoch {epoch} | val_loss={val_loss:.4f} | saved → {saved_path}")
```

---

## Checkpoint Metadata

Rich metadata makes checkpoints *auditable*. Beyond weights, record everything needed to reproduce or understand the run.

<div class="diagram">
  <div class="diagram-title">Metadata Taxonomy</div>
  <div class="diagram-grid cols-3">
    <div class="diagram-card blue">
      <div class="card-icon">📊</div>
      <div class="card-title">Training State</div>
      <div class="card-desc">epoch · global step · train/val loss · best metric · LR at save time</div>
    </div>
    <div class="diagram-card green">
      <div class="card-icon">🔧</div>
      <div class="card-title">Reproducibility</div>
      <div class="card-desc">git hash · config dict · random seeds · dataset version hash</div>
    </div>
    <div class="diagram-card orange">
      <div class="card-icon">🕐</div>
      <div class="card-title">Provenance</div>
      <div class="card-desc">UTC timestamp · host name · PyTorch version · CUDA version</div>
    </div>
  </div>
</div>

```python
import platform
import hashlib

def build_meta(epoch, train_loss, val_loss, val_metric, config, dataset_path):
    # Stable hash of the dataset directory for lineage
    hasher = hashlib.sha256()
    for p in sorted(Path(dataset_path).rglob("*.json")):
        hasher.update(p.read_bytes())
    dataset_hash = hasher.hexdigest()[:12]

    return CheckpointMeta(
        epoch=epoch,
        step=epoch * 1000,
        train_loss=train_loss,
        val_loss=val_loss,
        val_metric=val_metric,
        config={
            **config,
            "dataset_hash": dataset_hash,
            "host": platform.node(),
            "cuda_version": torch.version.cuda,
        },
        git_hash=_get_git_hash(),
        timestamp=datetime.datetime.utcnow().isoformat(),
    )
```

---

## Safetensors Format

`pickle`-based `.pt` files allow arbitrary Python objects to execute during deserialization — a supply-chain attack vector. [safetensors](https://github.com/huggingface/safetensors) addresses this.

<div class="diagram">
  <div class="diagram-title">Safetensors File Layout</div>
  <div class="flow">
    <div class="flow-node teal wide">8-byte header length (little-endian uint64)</div>
    <div class="flow-arrow accent">↓</div>
    <div class="flow-node blue wide">JSON header — dtype, shape, data_offsets per tensor</div>
    <div class="flow-arrow accent">↓</div>
    <div class="flow-node green wide">Raw tensor bytes — zero-copy mmap-able, no deserialization</div>
  </div>
</div>

```python
# pip install safetensors
from safetensors.torch import save_file, load_file
from pathlib import Path

# Save
state = model.state_dict()
save_file(state, "model.safetensors")

# Load — safe, no code execution, supports memory-mapping
loaded_state = load_file("model.safetensors", device="cuda")
model.load_state_dict(loaded_state)

# Memory-mapped load (only loads requested tensors into RAM)
from safetensors import safe_open
with safe_open("model.safetensors", framework="pt", device="cpu") as f:
    embed_weight = f.get_tensor("embedding.weight")   # only this tensor
    print(embed_weight.shape)
```

**Benchmark comparison — 1 B parameter model:**

| Format | Save time | Load time | File size | Safe? |
|---|---|---|---|---|
| `torch.save` (pickle) | 18 s | 22 s | 4.0 GB | ❌ |
| safetensors | 12 s | **4 s** | 4.0 GB | ✅ |
| safetensors (mmap) | 12 s | **< 1 s** | 4.0 GB | ✅ |

---

## Versioned Checkpointing Strategy

<div class="diagram">
  <div class="diagram-title">Training Loop → Checkpoint Storage</div>
  <div class="flow">
    <div class="flow-node blue extra-wide">Training Loop<br><small>forward → loss → backward → step</small></div>
    <div class="flow-arrow accent">↓ end of epoch</div>
    <div class="flow-node teal wide">CheckpointManager.save(meta)</div>
    <div class="flow-arrow accent">↓</div>
    <div class="flow-h">
      <div class="flow-node green">best/<br><small>lowest val_loss symlink</small></div>
      <div class="flow-arrow green">→</div>
      <div class="flow-node orange">last/<br><small>most recent epoch</small></div>
      <div class="flow-arrow green">→</div>
      <div class="flow-node purple">epoch_NNNN/<br><small>periodic every K epochs</small></div>
    </div>
    <div class="flow-arrow accent">↓ rotation</div>
    <div class="flow-node accent wide">Keep last N — delete oldest non-best checkpoints</div>
  </div>
</div>

```python
class VersionedCheckpointManager(CheckpointManager):
    """Extends CheckpointManager with periodic epoch saving."""

    def __init__(self, *args, save_every_n_epochs: int = 5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_every_n = save_every_n_epochs

    def save(self, meta: CheckpointMeta) -> Path:
        path = super().save(meta)
        # Always symlink "last"
        last_link = self.dir / "last"
        if last_link.is_symlink():
            last_link.unlink()
        last_link.symlink_to(path.name)
        # Periodic hard-keep
        if meta.epoch % self.save_every_n == 0:
            keep_dir = self.dir / f"periodic_epoch_{meta.epoch:04d}"
            shutil.copytree(path, keep_dir, dirs_exist_ok=True)
        return path
```

---

## Partial Weight Loading

When fine-tuning a pretrained backbone or growing a model architecture, you need to load a *subset* of weights — ignoring mismatched or missing keys.

```python
def load_partial_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    prefix_remap: dict[str, str] | None = None,
    ignore_prefixes: list[str] | None = None,
    device: str = "cpu",
) -> tuple[list[str], list[str]]:
    """
    Load weights with flexible key remapping and prefix filtering.

    Returns:
        missing_keys, unexpected_keys
    """
    saved_sd = torch.load(checkpoint_path, weights_only=True, map_location=device)

    # Optional key remapping: e.g. {"encoder.": "backbone.encoder."}
    if prefix_remap:
        remapped = {}
        for k, v in saved_sd.items():
            new_k = k
            for old_prefix, new_prefix in prefix_remap.items():
                if k.startswith(old_prefix):
                    new_k = new_prefix + k[len(old_prefix):]
                    break
            remapped[new_k] = v
        saved_sd = remapped

    # Optional prefix filtering: skip keys from certain sub-modules
    if ignore_prefixes:
        saved_sd = {
            k: v for k, v in saved_sd.items()
            if not any(k.startswith(p) for p in ignore_prefixes)
        }

    missing, unexpected = model.load_state_dict(saved_sd, strict=False)
    print(f"Partial load: {len(missing)} missing, {len(unexpected)} unexpected")
    return missing, unexpected


# Example: load ViT backbone, skip classification head
missing, unexpected = load_partial_weights(
    model=finetune_model,
    checkpoint_path="vit_pretrained.pt",
    prefix_remap={"model.": ""},          # strip wrapper prefix
    ignore_prefixes=["head.", "classifier."],
)
```

<div class="callout info">
  <span class="callout-icon">ℹ️</span>
  <div class="callout-body">After partial loading, inspect <code>missing_keys</code>. Keys for newly added layers (e.g. a new classifier head) are expected to be missing and will be randomly initialized. Unexpected keys usually mean a mismatch in architecture.</div>
</div>

---

## DDP Checkpointing

With `DistributedDataParallel`, the model is wrapped and each rank holds an identical copy. Only **rank 0** should write to disk.

```python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def ddp_save_checkpoint(
    model: DDP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
    rank: int,
) -> None:
    """Only rank 0 saves; other ranks block until done."""
    if rank == 0:
        # Unwrap DDP to get the raw module state_dict
        state = {
            "model": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(state, path)
    # Barrier: ensure rank 0 finishes writing before training continues
    dist.barrier()


def ddp_load_checkpoint(
    model: DDP,
    optimizer: torch.optim.Optimizer,
    path: str,
    rank: int,
    map_location: str | None = None,
) -> int:
    """All ranks load from the same checkpoint file."""
    loc = map_location or f"cuda:{rank}"
    state = torch.load(path, weights_only=False, map_location=loc)
    # DDP wraps module, so load into model.module
    model.module.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return state["epoch"]


# Training loop skeleton (each process runs this after torchrun)
def train_ddp(rank: int, world_size: int) -> None:
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = SmallNet(784, 10).to(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    start_epoch = 0
    ckpt = "checkpoints/latest.pt"
    if os.path.exists(ckpt):
        start_epoch = ddp_load_checkpoint(model, optimizer, ckpt, rank) + 1

    for epoch in range(start_epoch, 100):
        # ... training ...
        ddp_save_checkpoint(model, optimizer, epoch, ckpt, rank)

    dist.destroy_process_group()
```

<div class="diagram">
  <div class="diagram-title">DDP Checkpoint Protocol</div>
  <div class="flow">
    <div class="flow-h">
      <div class="flow-node blue narrow">Rank 0</div>
      <div class="flow-node teal narrow">Rank 1</div>
      <div class="flow-node teal narrow">Rank 2</div>
      <div class="flow-node teal narrow">Rank 3</div>
    </div>
    <div class="flow-arrow accent">↓ end of epoch</div>
    <div class="flow-node blue wide">Rank 0: save model.module.state_dict() → disk</div>
    <div class="flow-arrow accent">↓ dist.barrier()</div>
    <div class="flow-node green wide">All ranks: resume training together</div>
  </div>
</div>

---

## FSDP State Dict Configuration

Fully Sharded Data Parallel shards parameters across ranks. Saving requires explicit configuration of how to gather the shards.

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp import ShardedStateDictConfig

# --- Option 1: Full state dict (all params gathered on rank 0) ---
def fsdp_save_full(model: FSDP, path: str, rank: int) -> None:
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state_dict = model.state_dict()
    if rank == 0:
        torch.save(state_dict, path)
    dist.barrier()


# --- Option 2: Sharded state dict (each rank saves its own shard) ---
def fsdp_save_sharded(model: FSDP, path_template: str, rank: int) -> None:
    cfg = ShardedStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, cfg):
        state_dict = model.state_dict()
    shard_path = path_template.format(rank=rank)
    torch.save(state_dict, shard_path)
    dist.barrier()


# --- Loading full state dict back ---
def fsdp_load_full(model: FSDP, path: str, rank: int) -> None:
    loc = f"cuda:{rank}"
    full_sd = torch.load(path, weights_only=True, map_location=loc)
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        model.load_state_dict(full_sd)
```

<div class="callout tip">
  <span class="callout-icon">💡</span>
  <div class="callout-body">For models too large to fit on rank 0 alone, use <strong>ShardedStateDictConfig</strong>. Each rank saves its shard; distributed loading reconstructs the model without materializing everything on one device.</div>
</div>

---

## Model Card Metadata

Every checkpoint should ship with a model card — a structured description of training data, intended use, evaluation metrics, and limitations.

```python
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

@dataclass
class ModelCard:
    model_name: str
    model_version: str
    description: str
    training_data: str
    evaluation_metrics: dict[str, float]
    intended_use: str
    limitations: list[str]
    license: str = "Apache-2.0"
    tags: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)

    def save(self, checkpoint_dir: str | Path) -> None:
        card_path = Path(checkpoint_dir) / "model_card.json"
        card_path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, checkpoint_dir: str | Path) -> "ModelCard":
        card_path = Path(checkpoint_dir) / "model_card.json"
        return cls(**json.loads(card_path.read_text()))


# Example usage
card = ModelCard(
    model_name="SmallNet-MNIST",
    model_version="1.2.0",
    description="Lightweight CNN for MNIST digit classification.",
    training_data="MNIST train split (60 000 images)",
    evaluation_metrics={"test_accuracy": 0.9921, "test_loss": 0.0241},
    intended_use="Educational demo and benchmark baseline",
    limitations=["Not suitable for production handwriting recognition"],
    tags=["vision", "classification", "mnist"],
    authors=["Alice <alice@example.com>"],
)
card.save("checkpoints/run_001/best")
```

---

## Comparison Table

<table class="compare-table">
  <thead>
    <tr>
      <th>Format</th>
      <th>Safety</th>
      <th>Speed</th>
      <th>Partial Load</th>
      <th>Cross-Language</th>
      <th>Best For</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>torch.save (pickle)</strong></td>
      <td>❌ Unsafe</td>
      <td>Moderate</td>
      <td>Full load only</td>
      <td>❌ Python only</td>
      <td>Quick local experiments</td>
    </tr>
    <tr>
      <td><strong>safetensors</strong></td>
      <td>✅ Safe</td>
      <td>⚡ Fast (mmap)</td>
      <td>✅ Per-tensor</td>
      <td>✅ Rust/Python/JS</td>
      <td>Production weight sharing</td>
    </tr>
    <tr>
      <td><strong>ONNX</strong></td>
      <td>✅ Safe</td>
      <td>Moderate</td>
      <td>❌</td>
      <td>✅ Any runtime</td>
      <td>Cross-framework inference</td>
    </tr>
    <tr>
      <td><strong>TorchScript</strong></td>
      <td>⚠️ Moderate</td>
      <td>Fast</td>
      <td>❌</td>
      <td>✅ C++ libtorch</td>
      <td>Mobile / embedded</td>
    </tr>
  </tbody>
</table>

---

## Resumable Training Checklist

<table class="compare-table">
  <thead>
    <tr><th>What to Save</th><th>When</th><th>Why</th></tr>
  </thead>
  <tbody>
    <tr><td>Model <code>state_dict</code></td><td>Every epoch (+ best)</td><td>Weights are the primary artifact</td></tr>
    <tr><td>Optimizer <code>state_dict</code></td><td>Every epoch</td><td>Moment estimates / momentum buffers</td></tr>
    <tr><td>LR Scheduler <code>state_dict</code></td><td>Every epoch</td><td>Correct LR at resume point</td></tr>
    <tr><td>Epoch / global step</td><td>Every epoch</td><td>Loop counter for resuming</td></tr>
    <tr><td>Python RNG state</td><td>Every epoch</td><td>Exact data order reproducibility</td></tr>
    <tr><td>CUDA RNG state</td><td>Every epoch</td><td>Dropout / stochastic layer reproducibility</td></tr>
    <tr><td>Config dict</td><td>Once (+ every ckpt)</td><td>Re-instantiate identical architecture</td></tr>
    <tr><td>Git hash</td><td>Once (+ every ckpt)</td><td>Pin exact code version</td></tr>
    <tr><td>Dataset hash</td><td>Once</td><td>Lineage — which data trained this model</td></tr>
    <tr><td>Best metric value</td><td>Every epoch</td><td>Best-model logic needs reference point</td></tr>
  </tbody>
</table>

```python
import random
import numpy as np

def save_full_training_state(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    best_metric: float,
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(state, path)


def load_full_training_state(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: str = "cpu",
) -> tuple[int, int, float]:
    state = torch.load(path, weights_only=False, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    rng = state.get("rng_state", {})
    if rng.get("python"):
        random.setstate(rng["python"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
    if rng.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return state["epoch"], state["global_step"], state["best_metric"]
```

---

## End-to-End Example

```python
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# ── Setup ──────────────────────────────────────────────────────────────
torch.manual_seed(42)
X = torch.randn(1000, 784)
y = torch.randint(0, 10, (1000,))
ds = TensorDataset(X, y)
loader = DataLoader(ds, batch_size=64, shuffle=True)

model = SmallNet(784, 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
criterion = nn.CrossEntropyLoss()

ckpt_mgr = CheckpointManager(
    checkpoint_dir=Path("checkpoints") / "demo",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    keep_last_n=3,
)

# ── Training loop ──────────────────────────────────────────────────────
for epoch in range(20):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    avg_loss = total_loss / len(loader)

    meta = CheckpointMeta(
        epoch=epoch,
        step=(epoch + 1) * len(loader),
        train_loss=avg_loss,
        val_loss=avg_loss * 1.05,          # simulated
        val_metric=avg_loss,
        config={"arch": "SmallNet", "lr": 1e-3},
        git_hash=_get_git_hash(),
        timestamp=datetime.datetime.utcnow().isoformat(),
    )
    ckpt_mgr.save(meta)

# ── Load best ──────────────────────────────────────────────────────────
best_meta = ckpt_mgr.load_best()
print(f"Best checkpoint: epoch={best_meta.epoch}, val_loss={best_meta.val_loss:.4f}")
```

<div class="callout tip">
  <span class="callout-icon">💡</span>
  <div class="callout-body">In CI/CD pipelines, load the best checkpoint, run your evaluation suite, and assert that metrics exceed a threshold before promoting the model to the model registry.</div>
</div>

---

## Summary

<div class="diagram">
  <div class="diagram-grid cols-2">
    <div class="diagram-card teal">
      <div class="card-icon">✅</div>
      <div class="card-title">Do</div>
      <div class="card-desc">Use safetensors · save optimizer state · use atomic writes · include metadata · rotate checkpoints · set weights_only=True</div>
    </div>
    <div class="diagram-card red">
      <div class="card-icon">❌</div>
      <div class="card-title">Avoid</div>
      <div class="card-desc">Pickling full modules · loading untrusted .pt files without weights_only · saving only weights without optimizer state · no rotation strategy</div>
    </div>
  </div>
</div>

*Last updated: May 2026*
