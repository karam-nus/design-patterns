---
title: "Chapter 29 — Configuration Management Pattern"
---

[← Back to Table of Contents](./README.md)

# Chapter 29 — Configuration Management Pattern

> *"Hardcoded values are technical debt with compound interest. Every constant you bury in code is a future emergency."*

---

## Why Configuration Management Matters

Every ML project starts the same way: a learning rate typed directly into a `model.fit()` call, a file path copy-pasted three times, an API key checked into a config comment "just temporarily." Then the project grows. A colleague tries to reproduce your results on a different machine. The staging environment needs different database credentials. A sweep needs to try 50 learning rates. Suddenly the codebase is a maze of `if os.getenv('ENV') == 'prod':` scattered across a dozen files.

Configuration management is the discipline of separating **what your code does** from **the parameters that control it**. Done well, it delivers:

- **Reproducibility** — every run is fully described by its config; git commit + config hash = deterministic result
- **Multi-environment portability** — the same codebase runs on a laptop, CI server, and production cluster
- **Experiment agility** — hyperparameter sweeps, ablations, and A/B tests become single-line CLI overrides
- **Security** — secrets never touch version control
- **Auditability** — configs are logged alongside metrics, so regressions are diagnosable

<div class="diagram">
<div class="diagram-title">The Config Hierarchy — From Chaos to Structure</div>
<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-year">Level 0</div>
    <div class="timeline-title">Hardcoded Values</div>
    <div class="timeline-desc">Constants scattered in source files. Breaks on any environment change. Zero reproducibility.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Level 1</div>
    <div class="timeline-title">Environment Variables</div>
    <div class="timeline-desc">Secrets and deployment-specific values via <code>os.getenv()</code>. Good for credentials, poor for structured ML configs.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Level 2</div>
    <div class="timeline-title">.env Files + python-dotenv</div>
    <div class="timeline-desc">Local overrides committed to <code>.gitignore</code>. Bridges secrets into env vars without exposing them.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Level 3</div>
    <div class="timeline-title">YAML / JSON Config Files</div>
    <div class="timeline-desc">Structured, human-readable configs with hierarchy. Supports inheritance and comments (YAML).</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Level 4</div>
    <div class="timeline-title">OmegaConf / Hydra</div>
    <div class="timeline-desc">Typed, composable configs with CLI overrides, interpolation, and multirun sweeps.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Level 5</div>
    <div class="timeline-title">Structured Configs + Pydantic</div>
    <div class="timeline-desc">Dataclass or Pydantic schemas with validation, type coercion, and IDE autocomplete.</div>
  </div>
</div>
</div>

---

## Environment Variables — The Foundation Layer

Environment variables are the universal primitive. Every deployment system (Docker, Kubernetes, CI/CD) supports them. They are the right place for **secrets, deployment targets, and feature toggles** — not for nested ML hyperparameters.

```python
import os
from dotenv import load_dotenv

# Load .env file for local development; no-op in production
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")
ENV = os.getenv("ENV", "development")  # default to 'development'

if not DATABASE_URL:
    raise EnvironmentError(
        "DATABASE_URL must be set. "
        "Copy .env.example to .env and fill in values."
    )
```

A `.env.example` file (committed) documents required variables; `.env` (git-ignored) holds actual values:

```yaml
# .env.example — commit this
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
OPENAI_API_KEY=sk-...
ENV=development
WANDB_API_KEY=
```

<div class="callout tip">
<div class="callout-icon">💡</div>
<div class="callout-body">
<strong>Rule of thumb:</strong> if a value changes between machines or contains a secret, use an env var. If it changes between experiments, use a config file. If it changes between runs of the same experiment, use a CLI override.
</div>
</div>

---

## OmegaConf — Structured YAML with Superpowers

[OmegaConf](https://omegaconf.readthedocs.io/) is the config backbone of Hydra (and useful standalone). It wraps Python dicts and YAML files in a `DictConfig` object that supports **dot-access, interpolation, merging, and readonly locking**.

### Basic DictConfig

```python
from omegaconf import OmegaConf, DictConfig

# Create from dict
cfg = OmegaConf.create({
    "model": {
        "name": "resnet50",
        "num_classes": 10,
        "pretrained": True,
    },
    "training": {
        "lr": 1e-3,
        "epochs": 100,
        "batch_size": 64,
    },
    "paths": {
        "data": "/data/imagenet",
        "checkpoint": "/checkpoints/${model.name}",  # interpolation!
    }
})

# Dot-access (fails fast on typos — unlike dict.get())
print(cfg.model.name)           # resnet50
print(cfg.paths.checkpoint)     # /checkpoints/resnet50

# Resolved YAML output
print(OmegaConf.to_yaml(cfg))
```

### Variable Interpolation

OmegaConf interpolation (`${key.path}`) enables configs to reference each other, eliminating duplication:

```python
cfg = OmegaConf.create({
    "experiment": {
        "name": "resnet_baseline",
        "output_dir": "runs/${experiment.name}",
        "checkpoint_dir": "${experiment.output_dir}/checkpoints",
        "log_dir": "${experiment.output_dir}/logs",
    },
    "model": {
        "backbone": "resnet50",
        "checkpoint_path": "${experiment.checkpoint_dir}/best.pth",
    }
})

# All paths stay consistent even if experiment.name changes
print(cfg.model.checkpoint_path)
# runs/resnet_baseline/checkpoints/best.pth
```

### Loading and Merging YAML Files

```python
from omegaconf import OmegaConf

# Load individual configs
base_cfg = OmegaConf.load("configs/base.yaml")
model_cfg = OmegaConf.load("configs/models/resnet50.yaml")
train_cfg = OmegaConf.load("configs/training/adamw.yaml")

# Merge: later configs override earlier ones
cfg = OmegaConf.merge(base_cfg, model_cfg, train_cfg)

# CLI overrides as the final layer
cli_overrides = OmegaConf.from_dotlist([
    "training.lr=5e-4",
    "training.epochs=50",
])
cfg = OmegaConf.merge(cfg, cli_overrides)

# Lock to prevent accidental mutation after setup
OmegaConf.set_readonly(cfg, True)

print(OmegaConf.to_yaml(cfg))
```

### Structured Configs (Dataclass Backend)

```python
from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING

@dataclass
class ModelConfig:
    name: str = MISSING          # Required — no default
    num_classes: int = 1000
    pretrained: bool = True
    dropout: float = 0.0

@dataclass
class TrainingConfig:
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    weight_decay: float = 1e-4
    scheduler: str = "cosine"

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42

# Schema-validated DictConfig
cfg = OmegaConf.structured(Config)
cfg.model.name = "resnet50"     # OK
cfg.model.num_classes = "oops"  # Raises ValidationError — wrong type!

# Merge YAML on top of structured schema (YAML values validated against types)
yaml_overrides = OmegaConf.load("experiments/run_001.yaml")
cfg = OmegaConf.merge(cfg, yaml_overrides)
```

---

## Hydra — Config Composition at Scale

[Hydra](https://hydra.cc/) is Facebook's framework built on OmegaConf. Its killer features are **config groups** (swappable config modules), **CLI overrides**, and **multirun sweeps** — all without touching Python code.

### Project Layout

```
my_project/
├── conf/
│   ├── config.yaml              # Root config with defaults list
│   ├── model/
│   │   ├── resnet50.yaml
│   │   ├── vit_base.yaml
│   │   └── efficientnet_b4.yaml
│   ├── training/
│   │   ├── adamw.yaml
│   │   └── sgd.yaml
│   ├── dataset/
│   │   ├── imagenet.yaml
│   │   └── cifar10.yaml
│   └── env/
│       ├── local.yaml
│       ├── staging.yaml
│       └── prod.yaml
├── train.py
└── evaluate.py
```

### Root Config

```yaml
# conf/config.yaml
defaults:
  - model: resnet50        # load conf/model/resnet50.yaml
  - training: adamw        # load conf/training/adamw.yaml
  - dataset: imagenet      # load conf/dataset/imagenet.yaml
  - env: local             # load conf/env/local.yaml
  - _self_                 # root config values come last

experiment_name: baseline_run
seed: 42
output_dir: outputs/${experiment_name}
```

### Config Group Files

```yaml
# conf/model/resnet50.yaml
name: resnet50
num_classes: 1000
pretrained: true
dropout: 0.1
backbone_frozen: false
```

```yaml
# conf/model/vit_base.yaml
name: vit_base_patch16_224
num_classes: 1000
pretrained: true
dropout: 0.0
patch_size: 16
embed_dim: 768
num_heads: 12
```

```yaml
# conf/training/adamw.yaml
optimizer: adamw
lr: 1.0e-3
weight_decay: 1.0e-4
epochs: 100
batch_size: 64
gradient_clip: 1.0
scheduler:
  name: cosine
  warmup_epochs: 5
  min_lr: 1.0e-6
```

```yaml
# conf/dataset/imagenet.yaml
name: imagenet
root: /data/imagenet
num_workers: 8
image_size: 224
augmentation:
  random_crop: true
  random_flip: true
  color_jitter: 0.4
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

### The Hydra Main Function

```python
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import logging

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> float:
    """Main training entry point. cfg is fully composed and validated."""

    log.info(f"Starting experiment: {cfg.experiment_name}")
    log.info(f"Full config:\n{OmegaConf.to_yaml(cfg)}")

    # Seed everything
    torch.manual_seed(cfg.seed)

    # Build model from config
    model = build_model(cfg.model)

    # Build optimizer
    optimizer = build_optimizer(model, cfg.training)

    # Build dataloaders
    train_loader, val_loader = build_dataloaders(cfg.dataset)

    best_val_acc = 0.0
    for epoch in range(cfg.training.epochs):
        train_one_epoch(model, optimizer, train_loader, cfg)
        val_acc = evaluate(model, val_loader)

        log.info(f"Epoch {epoch}: val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, cfg.output_dir)

    return best_val_acc  # returned value used by Hydra multirun


def build_model(model_cfg: DictConfig):
    import timm
    return timm.create_model(
        model_cfg.name,
        pretrained=model_cfg.pretrained,
        num_classes=model_cfg.num_classes,
        drop_rate=model_cfg.dropout,
    )


def build_optimizer(model, train_cfg: DictConfig):
    params = model.parameters()
    if train_cfg.optimizer == "adamw":
        return torch.optim.AdamW(
            params,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
    elif train_cfg.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=train_cfg.lr,
            momentum=0.9,
            weight_decay=train_cfg.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {train_cfg.optimizer}")


if __name__ == "__main__":
    train()
```

### CLI Overrides — Zero Code Changes

```bash
# Basic override
python train.py training.lr=5e-4 training.epochs=50

# Swap entire config group
python train.py model=vit_base dataset=cifar10

# Override nested values
python train.py training.scheduler.warmup_epochs=10

# Add a new key not in the schema (requires +)
python train.py +training.gradient_accumulation=4

# Multirun sweep — tries all combinations
python train.py --multirun \
    training.lr=1e-3,5e-4,1e-4 \
    model=resnet50,vit_base

# Bayesian sweep via Ax sweeper plugin
python train.py --multirun \
    hydra/sweeper=ax \
    "training.lr=interval(1e-5, 1e-2)" \
    "training.weight_decay=interval(1e-5, 1e-2)"
```

<div class="callout info">
<div class="callout-icon">ℹ️</div>
<div class="callout-body">
Hydra automatically creates timestamped output directories (<code>outputs/2026-05-14/10-23-45/</code>) for each run and saves the resolved config as <code>.hydra/config.yaml</code>. You can reproduce any run exactly by pointing at that saved config.
</div>
</div>

---

## Structured Configs with Full Validation

For production ML systems, use Python dataclasses (or Pydantic) as the config schema. This gives you IDE autocomplete, type checking, and runtime validation.

```python
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from omegaconf import MISSING, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore

# ─── Schema Definitions ────────────────────────────────────────────────────────

@dataclass
class AugmentationConfig:
    random_crop: bool = True
    random_flip: bool = True
    color_jitter: float = 0.4
    normalize_mean: List[float] = field(
        default_factory=lambda: [0.485, 0.456, 0.406]
    )
    normalize_std: List[float] = field(
        default_factory=lambda: [0.229, 0.224, 0.225]
    )

@dataclass
class DatasetConfig:
    name: str = MISSING
    root: str = MISSING
    num_workers: int = 4
    image_size: int = 224
    augmentation: AugmentationConfig = field(
        default_factory=AugmentationConfig
    )

@dataclass
class SchedulerConfig:
    name: Literal["cosine", "step", "plateau"] = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    step_size: int = 30       # for 'step' scheduler
    gamma: float = 0.1        # for 'step' scheduler

@dataclass
class TrainingConfig:
    optimizer: Literal["adamw", "sgd", "adam"] = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 64
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def __post_init__(self):
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if not (1 <= self.batch_size <= 4096):
            raise ValueError(f"batch_size out of range: {self.batch_size}")

@dataclass
class ModelConfig:
    name: str = MISSING
    num_classes: int = 1000
    pretrained: bool = True
    dropout: float = 0.0
    backbone_frozen: bool = False

@dataclass
class ExperimentConfig:
    experiment_name: str = MISSING
    seed: int = 42
    output_dir: str = "outputs/${experiment_name}"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

# ─── Register with ConfigStore ─────────────────────────────────────────────────

cs = ConfigStore.instance()
cs.store(name="config_schema", node=ExperimentConfig)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: ExperimentConfig) -> None:
    # cfg is fully typed — IDE knows cfg.training.lr is a float
    print(f"LR: {cfg.training.lr}")
    print(f"Scheduler: {cfg.training.scheduler.name}")
```

---

## Config Composition Pattern

Real projects compose configs from multiple independent files that can be mixed and matched:

<div class="diagram">
<div class="diagram-title">Config Composition Flow</div>
<div class="flow">
  <div class="flow-node blue wide">base.yaml<br/><small>seed, paths, logging</small></div>
  <div class="flow-arrow"></div>
  <div class="flow-node green wide">model/resnet50.yaml<br/><small>arch, pretrained, dropout</small></div>
  <div class="flow-arrow"></div>
  <div class="flow-node purple wide">training/adamw.yaml<br/><small>lr, epochs, scheduler</small></div>
  <div class="flow-arrow"></div>
  <div class="flow-node orange wide">dataset/imagenet.yaml<br/><small>root, augmentation</small></div>
  <div class="flow-arrow accent">merge →</div>
  <div class="flow-node accent extra-wide">Resolved DictConfig<br/><small>interpolated, validated, readonly</small></div>
</div>
</div>

```python
from omegaconf import OmegaConf
import hashlib, json

def compose_config(
    base: str = "configs/base.yaml",
    model: str = "configs/model/resnet50.yaml",
    training: str = "configs/training/adamw.yaml",
    dataset: str = "configs/dataset/imagenet.yaml",
    overrides: list[str] | None = None,
) -> OmegaConf:
    """Compose a config from layered YAML files with optional CLI-style overrides."""
    cfg = OmegaConf.merge(
        OmegaConf.load(base),
        OmegaConf.load(model),
        OmegaConf.load(training),
        OmegaConf.load(dataset),
    )
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    OmegaConf.set_readonly(cfg, True)
    return cfg


def config_hash(cfg) -> str:
    """Stable hash of the resolved config for reproducibility tracking."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    canonical = json.dumps(cfg_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# Usage
cfg = compose_config(
    model="configs/model/vit_base.yaml",
    overrides=["training.lr=5e-4", "training.epochs=50"],
)
run_id = config_hash(cfg)
print(f"Run ID: {run_id}")  # e.g. "a3f9b2c11d04"
```

---

## Feature Flags in ML Configs

Feature flags let you toggle experimental code paths without code changes — essential for gradual rollouts and A/B testing in ML pipelines:

```python
from dataclasses import dataclass, field
from omegaconf import DictConfig

@dataclass
class FeatureFlags:
    # Experimental techniques
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = False
    use_flash_attention: bool = False
    use_compile: bool = False           # torch.compile()
    use_ema_weights: bool = False

    # Data pipeline flags
    use_prefetch: bool = True
    use_webdataset: bool = False

    # Monitoring flags
    profile_first_batch: bool = False
    log_gradient_norms: bool = False
    log_weight_histograms: bool = False


def apply_feature_flags(model, optimizer, cfg: DictConfig):
    flags = cfg.features

    if flags.use_mixed_precision:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    if flags.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if flags.use_compile:
        import torch
        model = torch.compile(model, mode="reduce-overhead")

    if flags.use_ema_weights:
        from torch_ema import ExponentialMovingAverage
        ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    else:
        ema = None

    return model, scaler, ema
```

```yaml
# conf/features/experimental.yaml
use_mixed_precision: true
use_gradient_checkpointing: true
use_flash_attention: true
use_compile: false      # unstable — leave off by default
use_ema_weights: true
profile_first_batch: false
log_gradient_norms: true
```

---

## Environment-Specific Config Layers

Production ML systems run across dev, staging, and prod with different resource limits, logging verbosity, and storage backends:

```yaml
# conf/env/local.yaml
_target_: configs.env.LocalEnv
storage:
  type: local
  checkpoint_root: ./outputs/checkpoints
  artifact_root: ./outputs/artifacts
logging:
  level: DEBUG
  use_wandb: false
  use_mlflow: false
resources:
  num_gpus: 1
  dataloader_workers: 2
  batch_size_multiplier: 1
```

```yaml
# conf/env/staging.yaml
storage:
  type: s3
  bucket: ml-experiments-staging
  checkpoint_root: s3://ml-experiments-staging/checkpoints
  artifact_root: s3://ml-experiments-staging/artifacts
logging:
  level: INFO
  use_wandb: true
  wandb_project: my-project-staging
  use_mlflow: true
  mlflow_uri: http://mlflow-staging:5000
resources:
  num_gpus: 4
  dataloader_workers: 8
  batch_size_multiplier: 4
```

```yaml
# conf/env/prod.yaml
storage:
  type: s3
  bucket: ml-experiments-prod
  checkpoint_root: s3://ml-experiments-prod/checkpoints
  artifact_root: s3://ml-experiments-prod/artifacts
logging:
  level: WARNING
  use_wandb: true
  wandb_project: my-project-prod
  use_mlflow: true
  mlflow_uri: http://mlflow-prod:5000
resources:
  num_gpus: 8
  dataloader_workers: 16
  batch_size_multiplier: 8
```

```python
import os
from omegaconf import DictConfig

def get_effective_batch_size(cfg: DictConfig) -> int:
    """Scales batch size by environment resource multiplier."""
    return cfg.training.batch_size * cfg.env.resources.batch_size_multiplier

def setup_storage(cfg: DictConfig) -> dict:
    """Returns storage paths adapted to current environment."""
    env = cfg.env.storage.type
    if env == "local":
        return {
            "checkpoint_dir": cfg.env.storage.checkpoint_root,
            "artifact_dir": cfg.env.storage.artifact_root,
        }
    elif env == "s3":
        return {
            "checkpoint_dir": cfg.env.storage.checkpoint_root,
            "artifact_dir": cfg.env.storage.artifact_root,
            "bucket": cfg.env.storage.bucket,
        }
    raise ValueError(f"Unknown storage type: {env}")
```

---

## Config Versioning and Hashing for Reproducibility

A config hash uniquely identifies an experiment configuration. Store it alongside every run so you can reproduce results exactly months later:

```python
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def get_git_info() -> dict:
    """Capture git state at run time."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip())
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception:
        return {"commit": "unknown", "branch": "unknown", "dirty": True}


def create_run_manifest(cfg: DictConfig, data_hash: str) -> dict:
    """
    Create a reproducibility manifest that captures everything
    needed to recreate this exact run.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_json = json.dumps(cfg_dict, sort_keys=True, default=str)
    cfg_hash = hashlib.sha256(cfg_json.encode()).hexdigest()[:16]

    manifest = {
        "run_id": cfg_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": cfg_hash,
        "data_hash": data_hash,
        "git": get_git_info(),
        "config": cfg_dict,
        "python_version": __import__("sys").version,
        "torch_version": __import__("torch").__version__,
    }
    return manifest


def save_run_manifest(manifest: dict, output_dir: str) -> Path:
    out = Path(output_dir) / "run_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    return out


def hash_dataset(dataset_path: str) -> str:
    """Hash a dataset directory for change detection."""
    hasher = hashlib.sha256()
    path = Path(dataset_path)

    # Hash sorted file listing + sizes (fast proxy for content hash)
    for fpath in sorted(path.rglob("*")):
        if fpath.is_file():
            stat = fpath.stat()
            hasher.update(f"{fpath.name}:{stat.st_size}".encode())

    return hasher.hexdigest()[:16]
```

---

## Secrets Management

<div class="callout danger">
<div class="callout-icon">🚨</div>
<div class="callout-body">
<strong>Never log secrets.</strong> API keys, database passwords, and auth tokens must never appear in log files, MLflow artifacts, W&B configs, or committed YAML files. A secret that touches a log is a compromised secret.
</div>
</div>

```python
import os
import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class SecretsConfig:
    """
    Secrets are NEVER stored in config files.
    They live exclusively in environment variables.
    This class provides typed access with validation.
    """
    wandb_api_key: str
    openai_api_key: Optional[str]
    db_password: str
    s3_secret_key: str

    @classmethod
    def from_env(cls) -> "SecretsConfig":
        required = {
            "wandb_api_key": "WANDB_API_KEY",
            "db_password": "DB_PASSWORD",
            "s3_secret_key": "AWS_SECRET_ACCESS_KEY",
        }
        missing = [
            env_var for _, env_var in required.items()
            if not os.getenv(env_var)
        ]
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {missing}\n"
                "Copy .env.example to .env and fill in values."
            )
        return cls(
            wandb_api_key=os.environ["WANDB_API_KEY"],
            openai_api_key=os.getenv("OPENAI_API_KEY"),  # optional
            db_password=os.environ["DB_PASSWORD"],
            s3_secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )

    def __repr__(self) -> str:
        # Mask secrets in all string representations
        return (
            f"SecretsConfig("
            f"wandb_api_key=*****, "
            f"db_password=*****, "
            f"s3_secret_key=*****)"
        )


def safe_log_config(cfg, secrets: SecretsConfig) -> None:
    """Log the config dict, asserting no secret values appear."""
    import json
    from omegaconf import OmegaConf
    cfg_str = OmegaConf.to_yaml(cfg)

    # Paranoia check: ensure no secret values leaked into config
    sensitive_values = [
        secrets.wandb_api_key,
        secrets.db_password,
        secrets.s3_secret_key,
    ]
    if secrets.openai_api_key:
        sensitive_values.append(secrets.openai_api_key)

    for secret in sensitive_values:
        if secret and len(secret) > 4 and secret in cfg_str:
            raise RuntimeError(
                "SECURITY: A secret value was found in the config output! "
                "Check your config for hardcoded credentials."
            )

    log.info(f"Config:\n{cfg_str}")
```

---

## Full Config Flow Diagram

<div class="diagram">
<div class="diagram-title">Hydra Config Pipeline</div>
<div class="flow">
  <div class="flow-node blue">CLI Args<br/><small>--multirun, overrides</small></div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-node accent wide">Hydra Compose<br/><small>defaults list resolution</small></div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-h">
    <div class="flow-node green narrow">base.yaml</div>
    <div class="flow-node blue narrow">model/*.yaml</div>
    <div class="flow-node purple narrow">training/*.yaml</div>
    <div class="flow-node orange narrow">dataset/*.yaml</div>
    <div class="flow-node teal narrow">env/*.yaml</div>
  </div>
  <div class="flow-arrow accent">↓ merge + interpolate</div>
  <div class="flow-node accent wide">Resolved DictConfig<br/><small>typed, validated, readonly</small></div>
  <div class="flow-arrow green">↓</div>
  <div class="flow-node green wide">Trainer / Pipeline<br/><small>receives cfg object</small></div>
  <div class="flow-arrow"></div>
  <div class="flow-node teal wide">Run Manifest<br/><small>config_hash + git + data_hash</small></div>
</div>
</div>

---

## Comparison: Config Approaches

<table class="compare-table">
<thead>
<tr>
  <th>Approach</th>
  <th>Type Safety</th>
  <th>Interpolation</th>
  <th>CLI Override</th>
  <th>Composition</th>
  <th>Sweep Support</th>
  <th>Best For</th>
</tr>
</thead>
<tbody>
<tr>
  <td><strong>argparse</strong></td>
  <td>⚠️ Manual</td>
  <td>❌</td>
  <td>✅ Native</td>
  <td>❌</td>
  <td>❌</td>
  <td>Simple scripts</td>
</tr>
<tr>
  <td><strong>dataclasses</strong></td>
  <td>✅ Full</td>
  <td>❌</td>
  <td>⚠️ Manual</td>
  <td>⚠️ Via inheritance</td>
  <td>❌</td>
  <td>Typed schemas</td>
</tr>
<tr>
  <td><strong>OmegaConf</strong></td>
  <td>✅ With structs</td>
  <td>✅ Native</td>
  <td>⚠️ Via dotlist</td>
  <td>✅ merge()</td>
  <td>❌</td>
  <td>YAML pipelines</td>
</tr>
<tr>
  <td><strong>Hydra</strong></td>
  <td>✅ With structs</td>
  <td>✅ Native</td>
  <td>✅ Native</td>
  <td>✅ Config groups</td>
  <td>✅ --multirun</td>
  <td>ML experiments</td>
</tr>
<tr>
  <td><strong>Pydantic Settings</strong></td>
  <td>✅ Full</td>
  <td>⚠️ Limited</td>
  <td>⚠️ Via env vars</td>
  <td>⚠️ Via nesting</td>
  <td>❌</td>
  <td>Web services</td>
</tr>
</tbody>
</table>

<div class="diagram-grid cols-3">
  <div class="diagram-card blue">
    <div class="card-icon">🔧</div>
    <div class="card-title">Use OmegaConf when…</div>
    <div class="card-desc">You need YAML merging and interpolation without the full Hydra overhead. Great for library authors and pipeline builders.</div>
  </div>
  <div class="diagram-card accent">
    <div class="card-icon">🚀</div>
    <div class="card-title">Use Hydra when…</div>
    <div class="card-desc">You run experiments with config groups, need CLI overrides, or want multirun sweeps. The gold standard for ML research codebases.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🔒</div>
    <div class="card-title">Use Pydantic Settings when…</div>
    <div class="card-desc">You're building a FastAPI service where configs come primarily from environment variables and you need strict validation with clear error messages.</div>
  </div>
</div>

---

## Summary

<div class="callout tip">
<div class="callout-icon">✅</div>
<div class="callout-body">
<strong>Config Management Checklist:</strong>
<ul>
<li>Secrets live in environment variables only — never in YAML or code</li>
<li>Use OmegaConf interpolation to eliminate duplicated paths</li>
<li>Compose configs from small, focused YAML modules (model / training / dataset / env)</li>
<li>Lock configs with <code>set_readonly()</code> after initialization to catch accidental mutation</li>
<li>Hash every resolved config and store it in your run manifest</li>
<li>Use Hydra <code>--multirun</code> for hyperparameter sweeps instead of custom loop code</li>
<li>Use <code>MISSING</code> for required fields — fail fast rather than using wrong defaults</li>
</ul>
</div>
</div>

The pattern scales from single-GPU experiments to distributed training clusters. The same `train.py` that runs on your laptop with `env=local` runs in production with `env=prod` — no code changes, just config swaps.

---

*Last updated: May 2026*
