---
title: "Chapter 19 — Command & Job Scheduler"
---

[← Back to Table of Contents](./README.md)

# Chapter 19 — Command & Job Scheduler

> *"Wrap a request in an object. Now you can queue it, log it, undo it, or retry it."*

ML workflows are full of discrete, well-defined units of work: train a model, evaluate a checkpoint, export to ONNX, run a hyperparameter sweep. The **Command** pattern turns each of these operations into a first-class object — something that can be stored in a queue, scheduled, retried, audited, and even undone. It is the conceptual backbone of every experiment scheduler from SLURM to Prefect.

<span class="badge behavioral">Behavioral</span>

---

## Intent

Encapsulate a **request as an object**, thereby letting you parameterise clients with different requests, queue or log requests, and support undoable operations.

---

## UML Structure

<div class="diagram">
  <div class="diagram-title">Command — Class Structure</div>
  <div class="uml-row">
    <div class="uml-box">
      <div class="uml-title">«interface» Command</div>
      <div class="uml-section">
        <div class="uml-item">+ execute() → Any</div>
        <div class="uml-item">+ undo() → None</div>
      </div>
    </div>
    <div class="uml-box">
      <div class="uml-title">Invoker</div>
      <div class="uml-section">
        <div class="uml-item">– queue: list[Command]</div>
      </div>
      <div class="uml-section">
        <div class="uml-item">+ enqueue(cmd: Command)</div>
        <div class="uml-item">+ run_all()</div>
      </div>
    </div>
  </div>
  <div class="uml-row">
    <div class="uml-box">
      <div class="uml-title">ConcreteCommand</div>
      <div class="uml-section">
        <div class="uml-item">– receiver: Receiver</div>
        <div class="uml-item">– params: dict</div>
      </div>
      <div class="uml-section">
        <div class="uml-item">+ execute() → Any</div>
        <div class="uml-item">+ undo() → None</div>
      </div>
    </div>
    <div class="uml-box">
      <div class="uml-title">Receiver</div>
      <div class="uml-section">
        <div class="uml-item">+ train(config)</div>
        <div class="uml-item">+ evaluate(checkpoint)</div>
        <div class="uml-item">+ export(format)</div>
      </div>
    </div>
  </div>
</div>

---

## ML Experiment Commands

```python
from __future__ import annotations
import abc
import copy
import datetime
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Command status
# ---------------------------------------------------------------------------

class CommandStatus(Enum):
    PENDING   = auto()
    RUNNING   = auto()
    SUCCEEDED = auto()
    FAILED    = auto()
    CANCELLED = auto()


# ---------------------------------------------------------------------------
# Base Command
# ---------------------------------------------------------------------------

@dataclass
class CommandResult:
    command_id: str
    status: CommandStatus
    output: Any = None
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def elapsed(self) -> float | None:
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return None


class MLCommand(abc.ABC):
    """Abstract Command for ML operations."""

    def __init__(self, description: str = "") -> None:
        self.id: str = str(uuid.uuid4())[:8]
        self.description = description or self.__class__.__name__
        self.status: CommandStatus = CommandStatus.PENDING
        self.result: CommandResult | None = None
        self._created_at: float = time.time()

    @abc.abstractmethod
    def execute(self) -> CommandResult:
        """Run the command.  Return a CommandResult."""

    def undo(self) -> None:
        """Reverse the effect of execute().  Not all commands support undo."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support undo."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, status={self.status.name})"


# ---------------------------------------------------------------------------
# Receiver — the model training system
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    model_name: str
    dataset: str
    learning_rate: float = 3e-4
    batch_size: int = 32
    max_epochs: int = 10
    seed: int = 42
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ExperimentSystem:
    """Receiver — knows how to actually run ML operations."""

    def __init__(self, output_dir: str = "experiments") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._registry: dict[str, dict] = {}

    def train(self, config: ExperimentConfig) -> dict:
        """Simulate a training run."""
        run_id = f"{config.model_name}_{int(time.time())}"
        artifact = {
            "run_id": run_id,
            "config": config.to_dict(),
            "best_val_loss": 0.45 + (config.learning_rate * 100 % 0.3),
            "checkpoint": str(self.output_dir / f"{run_id}/best.pt"),
        }
        self._registry[run_id] = artifact
        return artifact

    def evaluate(self, checkpoint: str, dataset: str = "test") -> dict:
        """Simulate evaluation of a checkpoint."""
        return {
            "checkpoint": checkpoint,
            "dataset": dataset,
            "accuracy": 0.91,
            "f1": 0.89,
        }

    def export(self, checkpoint: str, format: str = "onnx") -> dict:
        """Simulate model export."""
        out_path = str(Path(checkpoint).with_suffix(f".{format}"))
        return {"source": checkpoint, "target": out_path, "format": format}

    def delete_run(self, run_id: str) -> None:
        """Undo a training run by removing its registry entry."""
        self._registry.pop(run_id, None)
        print(f"[System] Deleted run {run_id} from registry.")


# ---------------------------------------------------------------------------
# Concrete Commands
# ---------------------------------------------------------------------------

class TrainCommand(MLCommand):
    """Command: train a model with a given config."""

    def __init__(
        self, system: ExperimentSystem, config: ExperimentConfig
    ) -> None:
        super().__init__(f"Train {config.model_name} on {config.dataset}")
        self.system = system
        self.config = config
        self._run_id: str | None = None

    def execute(self) -> CommandResult:
        self.status = CommandStatus.RUNNING
        result = CommandResult(
            command_id=self.id,
            status=CommandStatus.RUNNING,
            started_at=time.time(),
        )
        try:
            artifact = self.system.train(self.config)
            self._run_id = artifact["run_id"]
            result.output = artifact
            result.status = CommandStatus.SUCCEEDED
            self.status = CommandStatus.SUCCEEDED
        except Exception as exc:
            result.error = str(exc)
            result.status = CommandStatus.FAILED
            self.status = CommandStatus.FAILED
        finally:
            result.finished_at = time.time()
        self.result = result
        return result

    def undo(self) -> None:
        if self._run_id:
            self.system.delete_run(self._run_id)
            self.status = CommandStatus.CANCELLED


class EvaluateCommand(MLCommand):
    """Command: evaluate a checkpoint on a dataset."""

    def __init__(
        self,
        system: ExperimentSystem,
        checkpoint: str,
        dataset: str = "test",
    ) -> None:
        super().__init__(f"Evaluate {checkpoint} on {dataset}")
        self.system = system
        self.checkpoint = checkpoint
        self.dataset = dataset

    def execute(self) -> CommandResult:
        self.status = CommandStatus.RUNNING
        result = CommandResult(
            command_id=self.id,
            status=CommandStatus.RUNNING,
            started_at=time.time(),
        )
        try:
            metrics = self.system.evaluate(self.checkpoint, self.dataset)
            result.output = metrics
            result.status = CommandStatus.SUCCEEDED
            self.status = CommandStatus.SUCCEEDED
        except Exception as exc:
            result.error = str(exc)
            result.status = CommandStatus.FAILED
            self.status = CommandStatus.FAILED
        finally:
            result.finished_at = time.time()
        self.result = result
        return result


class ExportCommand(MLCommand):
    """Command: export a model to a deployment format."""

    def __init__(
        self,
        system: ExperimentSystem,
        checkpoint: str,
        format: str = "onnx",
    ) -> None:
        super().__init__(f"Export {checkpoint} to {format}")
        self.system = system
        self.checkpoint = checkpoint
        self.format = format

    def execute(self) -> CommandResult:
        self.status = CommandStatus.RUNNING
        result = CommandResult(
            command_id=self.id,
            status=CommandStatus.RUNNING,
            started_at=time.time(),
        )
        try:
            artifact = self.system.export(self.checkpoint, self.format)
            result.output = artifact
            result.status = CommandStatus.SUCCEEDED
            self.status = CommandStatus.SUCCEEDED
        except Exception as exc:
            result.error = str(exc)
            result.status = CommandStatus.FAILED
            self.status = CommandStatus.FAILED
        finally:
            result.finished_at = time.time()
        self.result = result
        return result
```

---

## Job Queue / Experiment Scheduler

```python
import threading
from collections import deque


class ExperimentQueue:
    """Invoker — manages a queue of MLCommand objects."""

    def __init__(self, max_retries: int = 2) -> None:
        self._queue: deque[MLCommand] = deque()
        self._history: list[MLCommand] = []
        self._undo_stack: list[MLCommand] = []
        self.max_retries = max_retries
        self._lock = threading.Lock()

    def enqueue(self, cmd: MLCommand) -> "ExperimentQueue":
        with self._lock:
            self._queue.append(cmd)
        print(f"[Queue] Enqueued: {cmd}")
        return self

    def run_next(self) -> CommandResult | None:
        with self._lock:
            if not self._queue:
                return None
            cmd = self._queue.popleft()

        for attempt in range(1, self.max_retries + 2):
            result = cmd.execute()
            self._history.append(cmd)
            if result.status == CommandStatus.SUCCEEDED:
                self._undo_stack.append(cmd)
                print(
                    f"[Queue] ✓ {cmd} succeeded in {result.elapsed:.3f}s "
                    f"(attempt {attempt})"
                )
                return result
            if attempt <= self.max_retries:
                print(f"[Queue] ✗ {cmd} failed. Retrying ({attempt}/{self.max_retries})…")
            else:
                print(f"[Queue] ✗ {cmd} failed after {self.max_retries + 1} attempts.")
        return result

    def run_all(self) -> list[CommandResult]:
        results = []
        while self._queue:
            r = self.run_next()
            if r:
                results.append(r)
        return results

    def undo_last(self) -> None:
        if not self._undo_stack:
            print("[Queue] Nothing to undo.")
            return
        cmd = self._undo_stack.pop()
        try:
            cmd.undo()
            print(f"[Queue] Undid: {cmd}")
        except NotImplementedError:
            print(f"[Queue] {cmd} does not support undo.")

    def status_report(self) -> str:
        lines = ["=== Experiment Queue Report ==="]
        lines.append(f"Pending:   {len(self._queue)}")
        lines.append(f"Completed: {len(self._history)}")
        lines.append("")
        for cmd in self._history:
            icon = "✓" if cmd.status == CommandStatus.SUCCEEDED else "✗"
            elapsed = cmd.result.elapsed if cmd.result else None
            t = f"{elapsed:.3f}s" if elapsed else "N/A"
            lines.append(f"  {icon} [{cmd.id}] {cmd.description} ({t})")
        return "\n".join(lines)
```

---

## Undo/Redo for Hyperparameter Tuning

Some commands can be reversed — restoring a model to a previous configuration state.

```python
class HyperparameterState:
    """Receiver — holds the current hyperparameter configuration."""

    def __init__(self, lr: float = 1e-3, batch_size: int = 32) -> None:
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = 1e-4

    def __repr__(self) -> str:
        return (
            f"HyperparameterState("
            f"lr={self.lr}, batch_size={self.batch_size}, "
            f"wd={self.weight_decay})"
        )


class SetLRCommand(MLCommand):
    """Command: change learning rate (reversible)."""

    def __init__(self, state: HyperparameterState, new_lr: float) -> None:
        super().__init__(f"Set LR to {new_lr}")
        self.state = state
        self.new_lr = new_lr
        self._old_lr: float | None = None

    def execute(self) -> CommandResult:
        self._old_lr = self.state.lr
        self.state.lr = self.new_lr
        self.status = CommandStatus.SUCCEEDED
        print(f"[SetLR] {self._old_lr} → {self.new_lr}")
        return CommandResult(
            command_id=self.id,
            status=CommandStatus.SUCCEEDED,
            output={"lr": self.new_lr},
        )

    def undo(self) -> None:
        if self._old_lr is not None:
            self.state.lr = self._old_lr
            print(f"[SetLR undo] {self.new_lr} → {self._old_lr}")


class SetBatchSizeCommand(MLCommand):
    """Command: change batch size (reversible)."""

    def __init__(self, state: HyperparameterState, new_bs: int) -> None:
        super().__init__(f"Set batch size to {new_bs}")
        self.state = state
        self.new_bs = new_bs
        self._old_bs: int | None = None

    def execute(self) -> CommandResult:
        self._old_bs = self.state.batch_size
        self.state.batch_size = self.new_bs
        self.status = CommandStatus.SUCCEEDED
        print(f"[SetBatch] {self._old_bs} → {self.new_bs}")
        return CommandResult(
            command_id=self.id,
            status=CommandStatus.SUCCEEDED,
            output={"batch_size": self.new_bs},
        )

    def undo(self) -> None:
        if self._old_bs is not None:
            self.state.batch_size = self._old_bs
            print(f"[SetBatch undo] {self.new_bs} → {self._old_bs}")


# Tuning session with undo/redo
hp_state = HyperparameterState()
queue = ExperimentQueue()
queue.enqueue(SetLRCommand(hp_state, 1e-4))
queue.enqueue(SetBatchSizeCommand(hp_state, 64))
queue.run_all()
print(hp_state)             # lr=0.0001, batch_size=64
queue.undo_last()           # undo SetBatchSize
print(hp_state)             # lr=0.0001, batch_size=32
```

---

## Macro Commands — Composing Multiple Steps

```python
class MacroCommand(MLCommand):
    """Composite command that runs several commands in sequence."""

    def __init__(
        self,
        commands: list[MLCommand],
        description: str = "MacroCommand",
        stop_on_failure: bool = True,
    ) -> None:
        super().__init__(description)
        self.commands = commands
        self.stop_on_failure = stop_on_failure

    def execute(self) -> CommandResult:
        self.status = CommandStatus.RUNNING
        results = []
        for cmd in self.commands:
            r = cmd.execute()
            results.append(r)
            if r.status != CommandStatus.SUCCEEDED and self.stop_on_failure:
                self.status = CommandStatus.FAILED
                return CommandResult(
                    command_id=self.id,
                    status=CommandStatus.FAILED,
                    error=f"Sub-command {cmd} failed: {r.error}",
                    output=results,
                )
        self.status = CommandStatus.SUCCEEDED
        return CommandResult(
            command_id=self.id,
            status=CommandStatus.SUCCEEDED,
            output=results,
        )

    def undo(self) -> None:
        # Undo in reverse order
        for cmd in reversed(self.commands):
            try:
                cmd.undo()
            except NotImplementedError:
                pass


# Full experiment pipeline as a single macro command
system = ExperimentSystem()
config = ExperimentConfig("resnet50", "imagenet", learning_rate=1e-3)

train_cmd = TrainCommand(system, config)
# EvaluateCommand will be wired to the checkpoint after training
pipeline = MacroCommand(
    commands=[train_cmd],
    description="ResNet50 full experiment pipeline",
)
result = pipeline.execute()
if result.status == CommandStatus.SUCCEEDED:
    checkpoint = result.output[0].output["checkpoint"]
    eval_cmd = EvaluateCommand(system, checkpoint)
    export_cmd = ExportCommand(system, checkpoint, "onnx")
    MacroCommand([eval_cmd, export_cmd], "Post-train pipeline").execute()
```

---

## Command History and Audit Trail

```python
import json
from pathlib import Path


class AuditingQueue(ExperimentQueue):
    """ExperimentQueue that writes a JSON audit log."""

    def __init__(self, log_path: str = "audit.jsonl", **kwargs) -> None:
        super().__init__(**kwargs)
        self.log_path = Path(log_path)

    def run_next(self) -> CommandResult | None:
        result = super().run_next()
        if result is not None:
            self._write_log(result)
        return result

    def _write_log(self, result: CommandResult) -> None:
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "command_id": result.command_id,
            "status": result.status.name,
            "output_keys": (
                list(result.output.keys())
                if isinstance(result.output, dict)
                else None
            ),
            "error": result.error,
            "elapsed_s": result.elapsed,
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
```

---

## Async Command Execution

```python
from concurrent.futures import ThreadPoolExecutor, as_completed


class ParallelExperimentQueue:
    """Run multiple independent commands concurrently."""

    def __init__(self, max_workers: int = 4) -> None:
        self._commands: list[MLCommand] = []
        self.max_workers = max_workers

    def add(self, cmd: MLCommand) -> "ParallelExperimentQueue":
        self._commands.append(cmd)
        return self

    def run_all(self) -> list[CommandResult]:
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_cmd = {
                executor.submit(cmd.execute): cmd for cmd in self._commands
            }
            for future in as_completed(future_to_cmd):
                cmd = future_to_cmd[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"[Parallel] {cmd} → {result.status.name}")
                except Exception as exc:
                    print(f"[Parallel] {cmd} raised: {exc}")
        return results


# Run a hyperparameter sweep in parallel
system = ExperimentSystem()
parallel_queue = ParallelExperimentQueue(max_workers=4)

for lr in [1e-2, 1e-3, 1e-4, 1e-5]:
    cfg = ExperimentConfig("bert-base", "sst2", learning_rate=lr)
    parallel_queue.add(TrainCommand(system, cfg))

sweep_results = parallel_queue.run_all()
best = min(
    (r for r in sweep_results if r.status == CommandStatus.SUCCEEDED),
    key=lambda r: r.output.get("best_val_loss", float("inf")),
    default=None,
)
if best:
    print(f"Best run: {best.output['run_id']} val_loss={best.output['best_val_loss']:.4f}")
```

---

## Hydra Multirun as Command Pattern

Hydra's `multirun` feature treats each configuration override as a command object. When you run:

```bash
python train.py --multirun lr=1e-3,1e-4 batch_size=32,64
```

Hydra generates 4 `(lr, batch_size)` combinations, each wrapped in a job object (command) and dispatched to a launcher (invoker). The `BasicLauncher` runs them sequentially; `RayLauncher` or `SlurmLauncher` run them in parallel.

```python
# With Hydra's Python API
from hydra._internal.utils import create_config_loader
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base=None)
def train(cfg: DictConfig) -> float:
    """Each multirun trial is an implicit Command invocation."""
    # cfg is the receiver's configuration
    system = ExperimentSystem()
    config = ExperimentConfig(
        model_name=cfg.model.name,
        dataset=cfg.data.name,
        learning_rate=cfg.optimizer.lr,
        batch_size=cfg.data.batch_size,
    )
    cmd = TrainCommand(system, config)
    result = cmd.execute()
    val_loss = result.output.get("best_val_loss", float("inf"))
    return val_loss  # Hydra captures this for sweep optimisation
```

---

## Flow Diagram

<div class="diagram">
  <div class="diagram-title">Command Pattern — ML Job Scheduler</div>
  <div class="flow">
    <div class="flow-node accent wide">Client (experiment script / CLI)</div>
    <div class="flow-arrow accent">▼ creates &amp; enqueues commands</div>
    <div class="flow-node blue wide">ExperimentQueue (Invoker)</div>
    <div class="flow-arrow">▼ dequeues &amp; calls execute()</div>
    <div class="flow-h">
      <div class="flow-node green narrow">TrainCommand</div>
      <div class="flow-node orange narrow">EvaluateCommand</div>
      <div class="flow-node purple narrow">ExportCommand</div>
    </div>
    <div class="flow-arrow purple">▼ delegate to</div>
    <div class="flow-node teal wide">ExperimentSystem (Receiver)</div>
    <div class="flow-arrow accent">▼ writes</div>
    <div class="flow-node accent wide">AuditLog + Artifact Registry</div>
  </div>
</div>

---

## Comparison Table

<table class="compare-table">
  <thead>
    <tr>
      <th>Dimension</th>
      <th>Command</th>
      <th>Strategy</th>
      <th>Template Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Encapsulates</td>
      <td>A whole request / operation</td>
      <td>One algorithm / behaviour</td>
      <td>Algorithm skeleton</td>
    </tr>
    <tr>
      <td>Undo support</td>
      <td>Yes — store previous state</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <td>Queueing</td>
      <td>Natural — commands are objects</td>
      <td>Awkward</td>
      <td>Awkward</td>
    </tr>
    <tr>
      <td>Composition</td>
      <td>MacroCommand</td>
      <td>No native composition</td>
      <td>No</td>
    </tr>
    <tr>
      <td>ML use case</td>
      <td>Job scheduler, sweep, undo HP</td>
      <td>Loss function, sampler</td>
      <td>Training loop skeleton</td>
    </tr>
  </tbody>
</table>

---

## Real-World ML Job Schedulers

| Scheduler | Command equivalent | Invoker equivalent | Notes |
|---|---|---|---|
| **SLURM** | Job script + `sbatch` | SLURM daemon | Shell scripts as commands |
| **Ray** | `remote` function / Actor task | Ray scheduler | Tasks are first-class objects |
| **Prefect** | `@task` decorated function | Flow / Runner | Full audit trail built-in |
| **Hydra multirun** | Config override combination | Launcher plugin | Native sweep support |
| **MLflow Projects** | `mlflow run` entry point | Backend executor | Reproducible runs as commands |

Each system independently arrived at the same insight: **wrap the unit of work in an object so it can be stored, scheduled, and tracked**.

---

## Key Takeaways

<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">📦</div>
    <div class="card-title">First-Class Operations</div>
    <div class="card-desc">Making operations first-class objects unlocks queuing, retry, audit logging, and undo — none of which are possible with raw function calls.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">↩️</div>
    <div class="card-title">Undo for Free</div>
    <div class="card-desc">Commands that store previous state enable undo — invaluable in interactive hyperparameter tuning or AutoML systems where you want to backtrack.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">⚡</div>
    <div class="card-title">Parallelism is Easy</div>
    <div class="card-desc">Independent Command objects map directly to ThreadPoolExecutor or Ray tasks. Sweep parallelism is structural, not an afterthought.</div>
  </div>
</div>

---

*Last updated: May 2026*
