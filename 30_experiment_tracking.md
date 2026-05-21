---
title: "Chapter 30 — Experiment Tracking & Reproducibility"
---

[← Back to Table of Contents](./README.md)

# Chapter 30 — Experiment Tracking & Reproducibility

> *"An experiment you cannot reproduce is not a result — it's a rumor."*

---

## Why Experiment Tracking Matters

Machine learning development is inherently iterative. A researcher trains dozens of models before finding the right combination of architecture, hyperparameters, and data preprocessing. Without systematic tracking, this process degrades into a notebook graveyard: `model_final_v2_ACTUALLY_FINAL.ipynb` and a metrics spreadsheet last updated three weeks ago.

Proper experiment tracking delivers:

- **Regression debugging** — when model quality drops, compare the current run to the last good one param-by-param
- **Result reproducibility** — re-train the exact winning model six months later for a production release
- **Collaboration** — teammates see your runs in real time, compare to their own, and build on your findings
- **Compliance and auditing** — in regulated industries, you must prove which data and code produced a deployed model
- **Sweep efficiency** — understand which hyperparameters matter, rather than running blind grid searches

<div class="diagram">
<div class="diagram-title">The Reproducibility Stack</div>
<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-year">Layer 1</div>
    <div class="timeline-title">Random Seeds</div>
    <div class="timeline-desc">Fix Python, NumPy, PyTorch, and CUDA seeds before any computation. Non-determinism is the enemy of reproducibility.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Layer 2</div>
    <div class="timeline-title">Data Version</div>
    <div class="timeline-desc">Hash your dataset or pin a DVC tag. A model trained on v1.2 vs v1.3 of the data is a different model.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Layer 3</div>
    <div class="timeline-title">Code Version</div>
    <div class="timeline-desc">Log the git commit hash with every run. Dirty working trees should trigger a warning.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Layer 4</div>
    <div class="timeline-title">Config Snapshot</div>
    <div class="timeline-desc">Log the full resolved config — not just the overrides, but every effective value — as a run artifact.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Layer 5</div>
    <div class="timeline-title">Environment Snapshot</div>
    <div class="timeline-desc">Log <code>pip freeze</code> / conda env export. Library version differences silently change model behavior.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Layer 6</div>
    <div class="timeline-title">Hardware Info</div>
    <div class="timeline-desc">GPU model, CUDA version, and cuDNN version. CUDA kernel behavior can differ across hardware generations.</div>
  </div>
</div>
</div>

---

## Full Reproducibility Setup

Before any training code runs, lock down all sources of non-determinism:

```python
import os
import random
import hashlib
import subprocess
import platform
import sys
from typing import Optional

import numpy as np
import torch


def setup_reproducibility(seed: int = 42, deterministic: bool = True) -> None:
    """
    Configure all sources of randomness for reproducible training.

    Args:
        seed: Integer seed for all RNGs.
        deterministic: If True, force CUDA deterministic algorithms
                       (slower but fully reproducible).
    """
    # Python built-in RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # PyTorch CPU RNG
    torch.manual_seed(seed)

    # PyTorch GPU RNG (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)   # multi-GPU

    # Environment-level hash seed (affects Python hash randomization)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic and torch.cuda.is_available():
        # Use deterministic CUDA algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 1.11+ deterministic mode
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Required by some deterministic ops
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif not deterministic:
        # Allow cudnn autotuner for speed — non-deterministic
        torch.backends.cudnn.benchmark = True


def get_environment_snapshot() -> dict:
    """Capture the full environment for reproducibility logging."""
    snapshot = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        snapshot["cuda_version"] = torch.version.cuda
        snapshot["cudnn_version"] = str(torch.backends.cudnn.version())
        snapshot["gpu_count"] = torch.cuda.device_count()
        snapshot["gpu_names"] = [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ]

    # Installed packages
    try:
        result = subprocess.run(
            ["pip", "freeze"], capture_output=True, text=True
        )
        snapshot["pip_freeze"] = result.stdout
        # Hash the environment for quick comparison
        snapshot["env_hash"] = hashlib.sha256(
            result.stdout.encode()
        ).hexdigest()[:12]
    except Exception:
        snapshot["pip_freeze"] = "unavailable"
        snapshot["env_hash"] = "unknown"

    return snapshot


def get_git_state() -> dict:
    """Capture git state. Warns if the working tree is dirty."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        diff_stat = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()
        dirty = bool(diff_stat)

        if dirty:
            import warnings
            warnings.warn(
                "Git working tree is dirty. "
                "Results may not be reproducible from this commit alone. "
                "Commit or stash your changes before running experiments.",
                UserWarning,
                stacklevel=2,
            )

        return {
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
            "diff_stat": diff_stat or "(clean)",
        }
    except Exception:
        return {"commit": "unknown", "branch": "unknown", "dirty": True}
```

---

## MLflow — The Open-Source Standard

[MLflow](https://mlflow.org/) is the most widely deployed open-source experiment tracker. It provides a tracking server, model registry, and serving layer that integrates with every major ML framework.

### Core Tracking API

```python
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import torch
import torch.nn as nn
from pathlib import Path
import json
import tempfile
import os


def train_with_mlflow(cfg: dict, model: nn.Module, train_loader, val_loader):
    """Full training loop with comprehensive MLflow tracking."""

    # Configure tracking server (defaults to ./mlruns if not set)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment(cfg["experiment_name"])

    with mlflow.start_run(run_name=cfg.get("run_name")) as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # ── Log all hyperparameters ─────────────────────────────────────────
        # log_params accepts flat dict; use nested keys with dots
        mlflow.log_params({
            "model.name": cfg["model"]["name"],
            "model.num_classes": cfg["model"]["num_classes"],
            "model.pretrained": cfg["model"]["pretrained"],
            "training.lr": cfg["training"]["lr"],
            "training.epochs": cfg["training"]["epochs"],
            "training.batch_size": cfg["training"]["batch_size"],
            "training.optimizer": cfg["training"]["optimizer"],
            "training.weight_decay": cfg["training"]["weight_decay"],
            "training.scheduler": cfg["training"]["scheduler"]["name"],
            "seed": cfg["seed"],
        })

        # ── Log git and environment info as tags ────────────────────────────
        git = get_git_state()
        mlflow.set_tags({
            "git.commit": git["commit"],
            "git.branch": git["branch"],
            "git.dirty": str(git["dirty"]),
            "env.torch_version": torch.__version__,
            "env.cuda_version": torch.version.cuda or "cpu",
        })

        # ── Log the full config as an artifact ──────────────────────────────
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(cfg, f, indent=2)
            config_path = f.name
        mlflow.log_artifact(config_path, "config")
        os.unlink(config_path)

        # ── Training loop ───────────────────────────────────────────────────
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        criterion = nn.CrossEntropyLoss()
        best_val_acc = 0.0

        for epoch in range(cfg["training"]["epochs"]):
            # Training phase
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_acc = correct / total
            train_loss /= len(train_loader)

            # Validation phase
            val_loss, val_acc = evaluate(model, val_loader, criterion)

            # ── Log metrics at each epoch ────────────────────────────────────
            mlflow.log_metrics({
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }, step=epoch)

            print(
                f"Epoch {epoch:3d} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

            # ── Save best model ──────────────────────────────────────────────
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Log PyTorch model with input signature
                mlflow.pytorch.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=cfg["model"]["name"],
                )
                mlflow.log_metric("best_val_accuracy", best_val_acc, step=epoch)

        # ── Final summary metrics ────────────────────────────────────────────
        mlflow.log_metrics({
            "final/best_val_accuracy": best_val_acc,
            "final/total_epochs": cfg["training"]["epochs"],
        })

        return run_id, best_val_acc


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return total_loss / len(loader), correct / total
```

### MLflow Model Registry

```python
from mlflow.tracking import MlflowClient

def promote_model_to_production(
    model_name: str,
    run_id: str,
    min_accuracy: float = 0.90,
) -> bool:
    """
    Promote a model version to Production if it meets quality gates.
    Returns True if promotion succeeded.
    """
    client = MlflowClient()

    # Fetch the run to check metrics
    run = client.get_run(run_id)
    val_acc = run.data.metrics.get("final/best_val_accuracy", 0.0)

    if val_acc < min_accuracy:
        print(
            f"Model rejected: val_acc={val_acc:.3f} < threshold={min_accuracy}"
        )
        return False

    # Find the model version from this run
    versions = client.search_model_versions(f"name='{model_name}'")
    run_versions = [v for v in versions if v.run_id == run_id]

    if not run_versions:
        print(f"No model version found for run {run_id}")
        return False

    version = run_versions[0].version

    # Transition through staging first
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging",
        archive_existing_versions=False,
    )
    print(f"Model {model_name} v{version} → Staging")

    # Archive current production model
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,  # archive old Production
    )
    print(f"Model {model_name} v{version} → Production ✓")

    # Add description with key metrics
    client.update_model_version(
        name=model_name,
        version=version,
        description=(
            f"Val accuracy: {val_acc:.4f} | "
            f"Git: {run.data.tags.get('git.commit', 'unknown')[:8]}"
        ),
    )

    return True
```

---

## Weights & Biases — Rich Experiment Visualization

[W&B](https://wandb.ai/) excels at interactive dashboards, media logging (images, audio, tables), and hyperparameter sweep orchestration.

```python
import wandb
import torch
import numpy as np
from PIL import Image


def train_with_wandb(cfg: dict, model, train_loader, val_loader):
    """Training loop with full W&B integration."""

    run = wandb.init(
        project=cfg["wandb_project"],
        entity=cfg.get("wandb_entity"),
        name=cfg.get("run_name"),
        config=cfg,          # log entire config dict
        tags=cfg.get("tags", []),
        notes=cfg.get("notes", ""),
        save_code=True,      # snapshot the calling script
    )

    # Watch model: log gradient/weight histograms every N batches
    wandb.watch(model, log="gradients", log_freq=100)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    best_val_acc = 0.0

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, sample_images, sample_preds = run_eval_epoch(
            model, val_loader, criterion
        )

        # ── Log scalar metrics ───────────────────────────────────────────────
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # ── Log sample predictions as a W&B Table ───────────────────────────
        if epoch % 10 == 0:
            columns = ["image", "pred", "true", "confidence"]
            data = [
                [
                    wandb.Image(img),
                    pred_label,
                    true_label,
                    confidence,
                ]
                for img, pred_label, true_label, confidence
                in zip(sample_images, sample_preds["labels"],
                       sample_preds["targets"], sample_preds["confidences"])
            ]
            wandb.log({"val/predictions": wandb.Table(columns=columns, data=data)})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save model as W&B artifact with lineage tracking
            artifact = wandb.Artifact(
                name=f"{cfg['model']['name']}-best",
                type="model",
                metadata={
                    "val_accuracy": val_acc,
                    "epoch": epoch,
                    "config_hash": cfg.get("config_hash", "unknown"),
                },
            )
            torch.save(model.state_dict(), "best_model.pth")
            artifact.add_file("best_model.pth")
            run.log_artifact(artifact)

    wandb.summary["best_val_accuracy"] = best_val_acc
    run.finish()
    return best_val_acc
```

### W&B Sweep Configuration

```yaml
# sweep_config.yaml — Bayesian hyperparameter search
program: train.py
method: bayes
metric:
  name: val/accuracy
  goal: maximize

parameters:
  training.lr:
    distribution: log_uniform_values
    min: 1.0e-5
    max: 1.0e-2

  training.weight_decay:
    distribution: log_uniform_values
    min: 1.0e-6
    max: 1.0e-3

  training.batch_size:
    values: [32, 64, 128, 256]

  model.dropout:
    distribution: uniform
    min: 0.0
    max: 0.5

  training.scheduler.name:
    values: [cosine, step, plateau]

early_terminate:
  type: hyperband
  min_iter: 5
  eta: 3
```

```python
import wandb
import yaml

def launch_sweep(sweep_config_path: str, count: int = 50) -> str:
    with open(sweep_config_path) as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep_config, project="my-project")
    print(f"Sweep ID: {sweep_id}")
    print(f"View at: https://wandb.ai/my-team/my-project/sweeps/{sweep_id}")

    # Launch agents (can be run on multiple machines)
    wandb.agent(sweep_id, function=sweep_train_fn, count=count)
    return sweep_id


def sweep_train_fn():
    """Called by the W&B sweep agent with sampled hyperparameters."""
    with wandb.init() as run:
        cfg = dict(wandb.config)
        model = build_model(cfg)
        train_loader, val_loader = build_loaders(cfg)
        val_acc = train_with_wandb(cfg, model, train_loader, val_loader)
        wandb.log({"final_val_acc": val_acc})
```

---

## Artifact Versioning Pattern

Every trained model should be traceable back to the exact data, code, and config that produced it. The artifact versioning pattern creates a **content-addressed lineage chain**:

```python
import hashlib
import json
import os
import subprocess
from pathlib import Path
from datetime import datetime, timezone


def hash_file(path: str, chunk_size: int = 65536) -> str:
    """SHA-256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def hash_directory(dir_path: str) -> str:
    """
    Stable hash of a directory: sorted file paths + sizes.
    Fast proxy for content hash — suitable for large datasets.
    """
    h = hashlib.sha256()
    root = Path(dir_path)
    for fpath in sorted(root.rglob("*")):
        if fpath.is_file():
            stat = fpath.stat()
            entry = f"{fpath.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}"
            h.update(entry.encode())
    return h.hexdigest()[:16]


def hash_config(cfg: dict) -> str:
    """Stable hash of a config dict."""
    canonical = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class ArtifactLineage:
    """
    Tracks the full lineage of a model artifact:
    data_hash + code_hash + config_hash → unique artifact ID.
    """

    def __init__(
        self,
        data_path: str,
        config: dict,
        code_root: str = ".",
    ):
        self.data_hash = hash_directory(data_path)
        self.config_hash = hash_config(config)
        self.git = self._get_git_info(code_root)

        # The artifact ID combines all three — if any changes, ID changes
        combined = f"{self.data_hash}:{self.config_hash}:{self.git['commit']}"
        self.artifact_id = hashlib.sha256(
            combined.encode()
        ).hexdigest()[:20]

        self.created_at = datetime.now(timezone.utc).isoformat()

    def _get_git_info(self, code_root: str) -> dict:
        try:
            commit = subprocess.check_output(
                ["git", "-C", code_root, "rev-parse", "HEAD"],
                text=True
            ).strip()
            dirty = bool(subprocess.check_output(
                ["git", "-C", code_root, "status", "--porcelain"],
                text=True
            ).strip())
            return {"commit": commit, "dirty": dirty}
        except Exception:
            return {"commit": "unknown", "dirty": True}

    def to_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "data_hash": self.data_hash,
            "config_hash": self.config_hash,
            "git_commit": self.git["commit"],
            "git_dirty": self.git["dirty"],
            "created_at": self.created_at,
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> dict:
        with open(path) as f:
            return json.load(f)


# Usage
lineage = ArtifactLineage(
    data_path="/data/imagenet",
    config=cfg,
    code_root=".",
)
print(f"Artifact ID: {lineage.artifact_id}")
lineage.save(f"artifacts/{lineage.artifact_id}/lineage.json")
```

---

## Git Integration in Experiment Tracking

```python
import subprocess
import tempfile
import os


def log_git_diff_as_artifact(run_dir: str) -> str | None:
    """
    Save the current git diff as an artifact.
    Useful when running experiments on uncommitted changes.
    Returns the path to the saved diff file, or None if clean.
    """
    try:
        diff = subprocess.check_output(
            ["git", "diff", "HEAD"], text=True
        )
        if not diff.strip():
            return None

        diff_path = os.path.join(run_dir, "uncommitted_changes.patch")
        os.makedirs(run_dir, exist_ok=True)
        with open(diff_path, "w") as f:
            f.write(diff)
        return diff_path
    except Exception:
        return None


def log_to_mlflow(run_id: str, data_path: str, config: dict) -> None:
    """Log git and artifact lineage info to an active MLflow run."""
    import mlflow

    lineage = ArtifactLineage(
        data_path=data_path,
        config=config,
    )

    mlflow.set_tags({
        "artifact.id": lineage.artifact_id,
        "artifact.data_hash": lineage.data_hash,
        "artifact.config_hash": lineage.config_hash,
        "git.commit": lineage.git["commit"],
        "git.dirty": str(lineage.git["dirty"]),
    })

    # Save and log lineage JSON
    lineage_path = f"lineage_{lineage.artifact_id}.json"
    lineage.save(lineage_path)
    mlflow.log_artifact(lineage_path, "lineage")
    os.unlink(lineage_path)

    # Log git diff if working tree is dirty
    diff_path = log_git_diff_as_artifact(".")
    if diff_path:
        mlflow.log_artifact(diff_path, "git")
        os.unlink(diff_path)
```

---

## DVC + MLflow Integration

[DVC](https://dvc.org/) versions datasets and models alongside code. Integrate it with MLflow to track data versions alongside model versions:

```python
import subprocess
import json
import mlflow


def get_dvc_data_version(dvc_file: str) -> dict:
    """
    Read the DVC file to get the content hash of a tracked dataset.
    Returns a dict with md5, size, and path.
    """
    try:
        result = subprocess.run(
            ["dvc", "status", dvc_file, "--json"],
            capture_output=True, text=True
        )
        return json.loads(result.stdout)
    except Exception:
        return {}


def train_with_dvc_mlflow(cfg: dict, dvc_data_path: str) -> None:
    """
    Training run that tracks DVC data version in MLflow.
    Ensures the model artifact is traceable to the exact data version.
    """
    # Pull latest data version specified in DVC lock file
    subprocess.run(["dvc", "pull", dvc_data_path], check=True)

    # Get DVC-tracked hash of the data
    dvc_info = get_dvc_data_version(f"{dvc_data_path}.dvc")

    mlflow.set_experiment(cfg["experiment_name"])

    with mlflow.start_run():
        # Log DVC data version info
        mlflow.set_tags({
            "dvc.data_path": dvc_data_path,
            "dvc.data_md5": dvc_info.get("md5", "unknown"),
            "dvc.data_size": str(dvc_info.get("size", "unknown")),
        })

        mlflow.log_params({
            "data.version": dvc_info.get("md5", "unknown")[:8],
            "data.path": dvc_data_path,
        })

        # Run training
        model = build_model(cfg["model"])
        train_loader, val_loader = build_dataloaders(cfg["dataset"])
        val_acc = run_training(model, train_loader, val_loader, cfg["training"])

        mlflow.log_metric("val/accuracy", val_acc)
        mlflow.pytorch.log_model(model, "model")
```

---

## Hyperparameter Sweep Patterns

<div class="diagram">
<div class="diagram-title">Sweep Strategy Comparison</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card blue">
    <div class="card-icon">📐</div>
    <div class="card-title">Grid Search</div>
    <div class="card-desc">Exhaustive search over all combinations. Guarantees finding the best point in the grid. Exponential cost — only practical with ≤3 params and small ranges.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🎲</div>
    <div class="card-title">Random Search</div>
    <div class="card-desc">Sample uniformly from the search space. Empirically outperforms grid search for the same budget because it explores the full range of each parameter independently.</div>
  </div>
  <div class="diagram-card accent">
    <div class="card-icon">🤖</div>
    <div class="card-title">Bayesian Search</div>
    <div class="card-desc">Build a surrogate model (Gaussian Process or TPE) over the objective and use it to pick the next promising point. Most sample-efficient for expensive experiments.</div>
  </div>
</div>
</div>

```python
from itertools import product
import random
from typing import Iterator


def grid_search(param_grid: dict) -> Iterator[dict]:
    """Yield all combinations of parameters."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    for combo in product(*values):
        yield dict(zip(keys, combo))


def random_search(
    param_space: dict,
    n_trials: int,
    seed: int = 42,
) -> Iterator[dict]:
    """
    Sample n_trials random configurations.

    param_space format:
        "lr": ("log_uniform", 1e-5, 1e-2)     # log-scale continuous
        "batch_size": ("choice", [32, 64, 128]) # categorical
        "dropout": ("uniform", 0.0, 0.5)        # linear continuous
    """
    rng = random.Random(seed)

    for _ in range(n_trials):
        config = {}
        for key, spec in param_space.items():
            dist_type = spec[0]
            if dist_type == "choice":
                config[key] = rng.choice(spec[1])
            elif dist_type == "uniform":
                config[key] = rng.uniform(spec[1], spec[2])
            elif dist_type == "log_uniform":
                import math
                log_low = math.log(spec[1])
                log_high = math.log(spec[2])
                config[key] = math.exp(rng.uniform(log_low, log_high))
            elif dist_type == "int":
                config[key] = rng.randint(spec[1], spec[2])
        yield config


# Example usage
param_space = {
    "training.lr": ("log_uniform", 1e-5, 1e-2),
    "training.weight_decay": ("log_uniform", 1e-6, 1e-3),
    "training.batch_size": ("choice", [32, 64, 128, 256]),
    "model.dropout": ("uniform", 0.0, 0.5),
}

for i, trial_cfg in enumerate(random_search(param_space, n_trials=20)):
    print(f"Trial {i}: {trial_cfg}")
    # Run training with trial_cfg, log to MLflow
```

---

## Experiment Comparison

Build structured comparison tables from MLflow run data:

```python
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


def build_experiment_table(
    experiment_name: str,
    metric_cols: list[str] | None = None,
    param_cols: list[str] | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Build a comparison DataFrame from all runs in an experiment.
    Sorted by best validation accuracy.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment not found: {experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val/accuracy DESC"],
        max_results=top_n,
    )

    metric_cols = metric_cols or [
        "val/accuracy", "val/loss", "train/accuracy", "final/best_val_accuracy"
    ]
    param_cols = param_cols or [
        "model.name", "training.lr", "training.batch_size",
        "training.optimizer", "training.scheduler"
    ]

    rows = []
    for run in runs:
        row = {
            "run_id": run.info.run_id[:8],
            "run_name": run.info.run_name or "unnamed",
            "status": run.info.status,
            "duration_min": (
                (run.info.end_time - run.info.start_time) / 60000
                if run.info.end_time else None
            ),
        }
        for col in param_cols:
            row[f"param/{col}"] = run.data.params.get(col, "—")
        for col in metric_cols:
            row[f"metric/{col}"] = run.data.metrics.get(col)
        row["git_commit"] = run.data.tags.get("git.commit", "unknown")[:8]
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values("metric/val/accuracy", ascending=False)


# Usage
table = build_experiment_table("imagenet_classification")
print(table.to_string(index=False))
```

---

## The Full Tracking Pipeline

<div class="diagram">
<div class="diagram-title">Experiment Tracking Flow</div>
<div class="flow">
  <div class="flow-h">
    <div class="flow-node blue narrow">Code<br/><small>git commit</small></div>
    <div class="flow-node green narrow">Data<br/><small>DVC hash</small></div>
    <div class="flow-node purple narrow">Config<br/><small>Hydra yaml</small></div>
  </div>
  <div class="flow-arrow accent">↓ lineage = hash(code + data + config)</div>
  <div class="flow-node accent wide">MLflow / W&B Run<br/><small>run_id = lineage.artifact_id</small></div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-h">
    <div class="flow-node blue narrow">Params<br/><small>log_params()</small></div>
    <div class="flow-node green narrow">Metrics<br/><small>log_metric()</small></div>
    <div class="flow-node orange narrow">Artifacts<br/><small>log_artifact()</small></div>
    <div class="flow-node purple narrow">Model<br/><small>log_model()</small></div>
  </div>
  <div class="flow-arrow green">↓ quality gate</div>
  <div class="flow-node green wide">Model Registry<br/><small>Staging → Production</small></div>
</div>
</div>

---

## Comparison: Experiment Tracking Platforms

<table class="compare-table">
<thead>
<tr>
  <th>Platform</th>
  <th>Self-Hosted</th>
  <th>Sweeps</th>
  <th>Model Registry</th>
  <th>Media Logging</th>
  <th>DVC Integration</th>
  <th>Best For</th>
</tr>
</thead>
<tbody>
<tr>
  <td><strong>MLflow</strong></td>
  <td>✅ Native</td>
  <td>⚠️ Via plugins</td>
  <td>✅ Built-in</td>
  <td>⚠️ Basic</td>
  <td>✅ Good</td>
  <td>Open-source teams, enterprise</td>
</tr>
<tr>
  <td><strong>W&B</strong></td>
  <td>✅ Enterprise</td>
  <td>✅ Native sweeps</td>
  <td>✅ Artifacts</td>
  <td>✅ Rich (images, 3D, audio)</td>
  <td>✅ Good</td>
  <td>Research, collaboration</td>
</tr>
<tr>
  <td><strong>Comet ML</strong></td>
  <td>✅ On-premise</td>
  <td>✅ Optimizer</td>
  <td>✅ Model registry</td>
  <td>✅ Good</td>
  <td>⚠️ Limited</td>
  <td>Teams needing compliance</td>
</tr>
<tr>
  <td><strong>Neptune</strong></td>
  <td>✅ On-premise</td>
  <td>⚠️ Via Neptune-Optuna</td>
  <td>✅ Model registry</td>
  <td>✅ Good</td>
  <td>✅ Good</td>
  <td>MLOps-heavy workflows</td>
</tr>
<tr>
  <td><strong>TensorBoard</strong></td>
  <td>✅ Always</td>
  <td>❌</td>
  <td>❌</td>
  <td>✅ Images, embeddings</td>
  <td>❌</td>
  <td>Quick local visualization</td>
</tr>
</tbody>
</table>

<div class="diagram-grid cols-2">
  <div class="diagram-card blue">
    <div class="card-icon">🏗️</div>
    <div class="card-title">MLflow for Platform Teams</div>
    <div class="card-desc">Best when you control the infrastructure and need a model registry integrated with Spark, Delta Lake, or Azure ML. The open-source standard for enterprise ML platforms.</div>
  </div>
  <div class="diagram-card accent">
    <div class="card-icon">🔬</div>
    <div class="card-title">W&B for Research Teams</div>
    <div class="card-desc">Best for fast-moving research where rich visualization, collaborative dashboards, and Bayesian sweeps are more important than on-premise control.</div>
  </div>
</div>

---

## Summary

<div class="callout tip">
<div class="callout-icon">✅</div>
<div class="callout-body">
<strong>Reproducibility Checklist:</strong>
<ul>
<li>Call <code>setup_reproducibility(seed)</code> before any compute — fix all RNGs</li>
<li>Log git commit hash and warn loudly if the working tree is dirty</li>
<li>Hash your dataset and include it in the run artifact ID</li>
<li>Log the full resolved config, not just overrides</li>
<li>Log <code>pip freeze</code> as an artifact on every run</li>
<li>Use DVC to version datasets alongside code in git</li>
<li>Use the Model Registry with quality gates before promoting to production</li>
<li>Store the lineage chain: <code>data_hash + code_hash + config_hash → artifact_id</code></li>
</ul>
</div>
</div>

<div class="callout warn">
<div class="callout-icon">⚠️</div>
<div class="callout-body">
<strong>Perfect reproducibility is harder than it looks.</strong> Even with fixed seeds, results can differ across CUDA versions, GPU architectures, and PyTorch releases. Document the full hardware and software stack. Use containers (Docker) for the strongest reproducibility guarantees.
</div>
</div>

---

*Last updated: May 2026*
