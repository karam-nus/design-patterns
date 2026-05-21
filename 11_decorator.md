---
title: "Chapter 11 — Decorator & Wrapper"
---
[← Back to Table of Contents](./README.md)

# Chapter 11 — Decorator & Wrapper

> *"Attach additional responsibilities to an object dynamically. Decorators provide a flexible alternative to subclassing for extending functionality."*
> — Gang of Four

---

## The Critical Distinction: Two Meanings of "Decorator"

Python overloads the word "decorator" in a way that trips up almost every ML engineer coming from a design-patterns background. There are **two completely different things**:

| Concept | What it is | Syntax |
|---|---|---|
| **Python `@decorator`** | Syntactic sugar — a callable that wraps another callable at *definition* time | `@my_func` above a `def` or `class` |
| **GoF Decorator Pattern** | A *structural* OOP pattern — a class that wraps another object, implementing the same interface, adding behaviour at *runtime* | Typically a class with `__init__(self, component)` |

Python `@` syntax can *implement* the GoF pattern, but most uses of `@` in Python have nothing to do with GoF Decorator. `@property`, `@staticmethod`, `@torch.no_grad()` are Python decorators but are **not** GoF Decorator pattern instances.

---

## GoF Decorator: Intent and Structure

**Intent:** Attach additional responsibilities to an object dynamically. Decorators provide a flexible alternative to subclassing for extending functionality.

**Key insight:** The decorator *wraps* a component and implements the *same interface*, so the client cannot tell the difference. Decorators can be stacked infinitely.

<div class="diagram">
  <div class="diagram-title">GoF Decorator — UML Structure</div>
  <div class="uml-row">
    <div class="uml-box">
      <div class="uml-title">«interface»<br>Component</div>
      <div class="uml-section">
        <div class="uml-item">+ operation()</div>
      </div>
    </div>
  </div>
  <div class="uml-row" style="margin-top:1rem">
    <div class="uml-box">
      <div class="uml-title">ConcreteComponent</div>
      <div class="uml-section">
        <div class="uml-item">+ operation()</div>
      </div>
    </div>
    <div class="uml-box">
      <div class="uml-title">Decorator</div>
      <div class="uml-section">
        <div class="uml-item">- component: Component</div>
      </div>
      <div class="uml-section">
        <div class="uml-item">+ operation()</div>
      </div>
    </div>
  </div>
  <div class="uml-row" style="margin-top:1rem">
    <div class="uml-box">
      <div class="uml-title">ConcreteDecoratorA</div>
      <div class="uml-section">
        <div class="uml-item">+ operation()</div>
        <div class="uml-item">+ addedBehaviourA()</div>
      </div>
    </div>
    <div class="uml-box">
      <div class="uml-title">ConcreteDecoratorB</div>
      <div class="uml-section">
        <div class="uml-item">- addedState: T</div>
      </div>
      <div class="uml-section">
        <div class="uml-item">+ operation()</div>
        <div class="uml-item">+ addedBehaviourB()</div>
      </div>
    </div>
  </div>
</div>

---

## Python `@functools.wraps` — Preserving Metadata

When you wrap a function, Python loses the original function's `__name__`, `__doc__`, and `__module__`. Always use `@functools.wraps` to preserve them:

```python
import functools
import time

def timing(func):
    @functools.wraps(func)          # copies __name__, __doc__, __wrapped__
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timing
def train_epoch(model, loader):
    """Train for one epoch."""
    ...

# Without @functools.wraps:
# train_epoch.__name__ == 'wrapper'   ← wrong
# With @functools.wraps:
# train_epoch.__name__ == 'train_epoch'  ← correct
print(train_epoch.__name__)     # 'train_epoch'
print(train_epoch.__wrapped__)  # the original unwrapped function
```

---

## ML Decorators: Python `@` Syntax

### Timing Decorator for Model Forward Passes

```python
import functools
import time
import torch

def timed(func=None, *, log_memory: bool = False):
    """Decorator that times function execution, optionally logging GPU memory."""
    if func is None:
        # Called with arguments: @timed(log_memory=True)
        return functools.partial(timed, log_memory=log_memory)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        result = func(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        mem_str = ""
        if log_memory and torch.cuda.is_available():
            mem_mb = torch.cuda.max_memory_allocated() / 1e6
            mem_str = f" | peak GPU mem: {mem_mb:.1f} MB"

        print(f"[TIMING] {func.__qualname__}: {elapsed*1000:.2f}ms{mem_str}")
        return result

    return wrapper


class ResNet(torch.nn.Module):
    @timed(log_memory=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

# Or wrap any callable:
@timed
def run_inference(model, batch):
    with torch.no_grad():
        return model(batch)
```

### Caching Decorator for Expensive Computations

```python
import functools
import hashlib
import pickle
from pathlib import Path
from typing import Any

def disk_cache(cache_dir: str = ".cache"):
    """Decorator that caches function results to disk (pickle)."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Build a stable cache key from arguments
            key_data = pickle.dumps((func.__qualname__, args, sorted(kwargs.items())))
            key = hashlib.sha256(key_data).hexdigest()[:16]
            cache_path = Path(cache_dir) / f"{func.__name__}_{key}.pkl"
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            if cache_path.exists():
                print(f"[CACHE] HIT {func.__name__} ({key})")
                with open(cache_path, "rb") as f:
                    return pickle.load(f)

            print(f"[CACHE] MISS {func.__name__} ({key}) — computing...")
            result = func(*args, **kwargs)
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
            return result
        return wrapper
    return decorator


# Built-in LRU cache for in-memory caching
@functools.lru_cache(maxsize=128)
def get_tokenizer(model_name: str):
    """Cache tokenizer instances — loading is expensive."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


@disk_cache(cache_dir=".feature_cache")
def extract_features(dataset_path: str, model_name: str) -> Any:
    """Expensive feature extraction cached to disk."""
    from transformers import AutoModel
    import torch
    model = AutoModel.from_pretrained(model_name)
    # ... expensive computation ...
    return features
```

### Retry Decorator for Flaky Data Loading

```python
import functools
import time
import logging
from typing import Type

logger = logging.getLogger(__name__)

def retry(
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    jitter: float = 0.1,
):
    """Retry decorator with exponential backoff and jitter."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {exc}"
                        )
                        raise
                    import random
                    wait = current_delay + random.uniform(0, jitter)
                    logger.warning(
                        f"{func.__name__} attempt {attempt} failed: {exc}. "
                        f"Retrying in {wait:.2f}s..."
                    )
                    time.sleep(wait)
                    current_delay *= backoff
        return wrapper
    return decorator


@retry(
    exceptions=(ConnectionError, TimeoutError, OSError),
    max_attempts=5,
    delay=0.5,
    backoff=2.0,
)
def load_shard_from_remote(shard_id: int, url: str) -> bytes:
    """Load a dataset shard, retrying on network errors."""
    import urllib.request
    with urllib.request.urlopen(f"{url}/shard_{shard_id}.bin", timeout=30) as r:
        return r.read()
```

### Logging Decorator for Model Calls

```python
import functools
import logging
import time
from typing import Any

def log_calls(
    logger: logging.Logger | None = None,
    level: int = logging.DEBUG,
    log_args: bool = False,
    log_result_shape: bool = True,
):
    """Log entry, exit, and timing for every call."""
    _logger = logger or logging.getLogger(__name__)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args_str = ""
            if log_args:
                # Summarise tensor shapes instead of printing values
                def _fmt(x: Any) -> str:
                    import torch
                    if isinstance(x, torch.Tensor):
                        return f"Tensor{list(x.shape)}"
                    return repr(x)
                args_str = f"({', '.join(_fmt(a) for a in args)})"

            _logger.log(level, f"→ {func.__qualname__}{args_str}")
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000

            result_str = ""
            if log_result_shape:
                import torch
                if isinstance(result, torch.Tensor):
                    result_str = f" → Tensor{list(result.shape)}"
                elif isinstance(result, dict):
                    result_str = f" → dict({list(result.keys())})"

            _logger.log(level, f"← {func.__qualname__} [{elapsed:.1f}ms]{result_str}")
            return result
        return wrapper
    return decorator
```

---

## Structural Decorator (GoF Pattern) for `nn.Module`

### `ModelWrapper`: Adding Timing, Logging, and Profiling

This is the **true GoF Decorator** — a class that implements the same interface (`nn.Module`) and wraps another module:

```python
import time
import logging
from typing import Any
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelWrapper(nn.Module):
    """
    GoF Decorator: wraps any nn.Module and adds cross-cutting behaviour
    without touching the wrapped module's code.
    """
    def __init__(
        self,
        module: nn.Module,
        *,
        enable_timing: bool = True,
        enable_logging: bool = True,
        enable_profiling: bool = False,
    ):
        super().__init__()
        self._wrapped = module
        self.enable_timing = enable_timing
        self.enable_logging = enable_logging
        self.enable_profiling = enable_profiling
        self._call_count = 0
        self._total_time = 0.0

    # Delegate parameter/buffer access to wrapped module
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self._call_count += 1

        if self.enable_logging:
            input_shapes = [
                list(a.shape) for a in args if isinstance(a, torch.Tensor)
            ]
            logger.debug(
                f"[{self._wrapped.__class__.__name__}] call #{self._call_count} "
                f"input shapes: {input_shapes}"
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        if self.enable_profiling:
            with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]) as prof:
                output = self._wrapped(*args, **kwargs)
            logger.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
        else:
            output = self._wrapped(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self._total_time += elapsed

        if self.enable_timing:
            logger.debug(
                f"[{self._wrapped.__class__.__name__}] forward: {elapsed*1000:.2f}ms "
                f"(avg: {self._total_time/self._call_count*1000:.2f}ms)"
            )
        return output

    def stats(self) -> dict:
        return {
            "call_count": self._call_count,
            "total_time_s": self._total_time,
            "avg_time_ms": (self._total_time / max(self._call_count, 1)) * 1000,
        }

    # Transparent attribute access — delegates to wrapped module
    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._wrapped, name)


# Usage — wrapping is transparent to training code:
base_model = torch.hub.load("pytorch/vision", "resnet50", pretrained=False)
model = ModelWrapper(base_model, enable_timing=True, enable_logging=True)

# model behaves exactly like a nn.Module:
output = model(torch.randn(4, 3, 224, 224))
print(model.stats())
```

### Augmentation Wrapper for Datasets

```python
from torch.utils.data import Dataset
from typing import Callable, Any
import torch


class AugmentedDataset(Dataset):
    """
    GoF Decorator for Dataset: wraps any Dataset and adds
    on-the-fly augmentation without modifying the source dataset.
    """
    def __init__(
        self,
        dataset: Dataset,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        augment_prob: float = 1.0,
    ):
        self._dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.augment_prob = augment_prob

    def __len__(self) -> int:
        return len(self._dataset)              # delegate length

    def __getitem__(self, idx: int) -> Any:
        sample, label = self._dataset[idx]     # delegate fetch

        import random
        if self.transform and random.random() < self.augment_prob:
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(label)
        return sample, label

    # Delegate any unknown attribute to the wrapped dataset
    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)


# Stack decorators — augmentation on top of subset on top of base
from torch.utils.data import Subset
from torchvision import transforms

base = MyImageDataset("/data/imagenet")
subset = Subset(base, indices=range(1000))
augmented = AugmentedDataset(
    subset,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
    ]),
)
```

### Gradient Clipping Wrapper Around Optimizer

```python
import torch
from torch.optim import Optimizer
from typing import Iterable


class GradientClippingOptimizer:
    """
    GoF Decorator for Optimizer: adds gradient clipping before every step.
    Wraps any optimizer without modifying it.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        parameters: Iterable[torch.nn.Parameter],
        max_norm: float = 1.0,
        norm_type: float = 2.0,
    ):
        self._optimizer = optimizer
        self._parameters = list(parameters)
        self.max_norm = max_norm
        self.norm_type = norm_type
        self._clip_events = []

    def step(self, closure=None):
        # Added behaviour: clip gradients before the update
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._parameters, self.max_norm, norm_type=self.norm_type
        )
        self._clip_events.append(float(grad_norm))
        return self._optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = True):
        return self._optimizer.zero_grad(set_to_none=set_to_none)

    def grad_norm_history(self) -> list[float]:
        return self._clip_events.copy()

    # Transparent delegation for everything else
    def __getattr__(self, name):
        return getattr(self._optimizer, name)


# Usage:
base_opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
opt = GradientClippingOptimizer(base_opt, model.parameters(), max_norm=1.0)
# Training loop uses opt.step() / opt.zero_grad() exactly as before
```

---

## Stacking Decorators — Order Matters

<div class="diagram">
  <div class="diagram-title">Decorator Stack — Execution Order</div>
  <div class="flow">
    <div class="flow-node accent wide">Client call: model(x)</div>
    <div class="flow-arrow accent">↓ enters outermost first</div>
    <div class="flow-node orange wide">LoggingWrapper.forward(x)<br><small>logs input shapes</small></div>
    <div class="flow-arrow"></div>
    <div class="flow-node purple wide">TimingWrapper.forward(x)<br><small>starts timer</small></div>
    <div class="flow-arrow"></div>
    <div class="flow-node blue wide">CachingWrapper.forward(x)<br><small>checks cache</small></div>
    <div class="flow-arrow"></div>
    <div class="flow-node green wide">BaseModel.forward(x)<br><small>actual computation</small></div>
    <div class="flow-arrow green">↑ returns innermost first</div>
    <div class="flow-node blue wide">CachingWrapper stores result</div>
    <div class="flow-arrow"></div>
    <div class="flow-node purple wide">TimingWrapper records elapsed</div>
    <div class="flow-arrow"></div>
    <div class="flow-node orange wide">LoggingWrapper logs output shape</div>
    <div class="flow-arrow accent">↑ result returns to Client</div>
    <div class="flow-node accent wide">Client receives output</div>
  </div>
</div>

```python
# Stacking Python @decorators — applied bottom-up, executed top-down
@log_calls(log_args=True)      # outermost: applied last, executes first
@timed(log_memory=True)        # middle
@retry(max_attempts=3)         # innermost decorator: applied first
def load_checkpoint(path: str):
    return torch.load(path)

# Equivalent to:
load_checkpoint = log_calls(log_args=True)(
    timed(log_memory=True)(
        retry(max_attempts=3)(load_checkpoint)
    )
)

# Stacking GoF Decorators (object wrappers):
model = BaseResNet()
model = CachingWrapper(model, cache_size=256)
model = TimingWrapper(model, log_every_n=100)
model = LoggingWrapper(model, logger=my_logger)
# Call order: Logging → Timing → Caching → BaseResNet → Caching → Timing → Logging
```

---

## Introspecting Decorated Functions

```python
import inspect
import functools

def double_wrap(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@double_wrap
@double_wrap
def my_fn():
    """My docstring."""
    pass

# __wrapped__ always points to the immediately inner function
print(my_fn.__wrapped__)            # wrapper (the inner double_wrap)
print(my_fn.__wrapped__.__wrapped__) # my_fn (the original)

# inspect.unwrap() traverses the full chain
original = inspect.unwrap(my_fn)
print(original)                     # <function my_fn>
print(original.__doc__)             # "My docstring."

# Checking if something is decorated
def is_decorated(func) -> bool:
    return hasattr(func, "__wrapped__")
```

---

## Comparison: Python Decorator vs GoF Decorator vs Mixin

<table class="compare-table">
  <thead>
    <tr>
      <th>Dimension</th>
      <th>Python @decorator</th>
      <th>GoF Decorator (class)</th>
      <th>Mixin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>What it wraps</strong></td>
      <td>Functions, methods, classes</td>
      <td>Objects (same interface)</td>
      <td>Classes (via inheritance)</td>
    </tr>
    <tr>
      <td><strong>When applied</strong></td>
      <td>Definition time (static)</td>
      <td>Runtime (dynamic)</td>
      <td>Definition time (static)</td>
    </tr>
    <tr>
      <td><strong>Interface preserved?</strong></td>
      <td>Only with @wraps</td>
      <td>Yes — by design</td>
      <td>Yes — via inheritance</td>
    </tr>
    <tr>
      <td><strong>Stackable?</strong></td>
      <td>Yes, but order matters</td>
      <td>Yes, unlimited nesting</td>
      <td>Yes, MRO determines order</td>
    </tr>
    <tr>
      <td><strong>Access to wrapped object</strong></td>
      <td>Via closure / __wrapped__</td>
      <td>Via self._component</td>
      <td>Via super()</td>
    </tr>
    <tr>
      <td><strong>State per wrapper</strong></td>
      <td>In closure variables</td>
      <td>In decorator instance</td>
      <td>Shared with class instance</td>
    </tr>
    <tr>
      <td><strong>ML use case</strong></td>
      <td>timing, retry, cache, log</td>
      <td>ModelWrapper, AugDataset</td>
      <td>SaveMixin, LogMixin</td>
    </tr>
  </tbody>
</table>

---

## When NOT to Use Decorators

<div class="callout warn">
<strong>Avoid decorators when:</strong>
<ul>
  <li><strong>The added behaviour is core logic</strong>, not a cross-cutting concern. If timing is part of your algorithm, don't hide it in a decorator.</li>
  <li><strong>Debugging becomes painful</strong> — deeply stacked decorators produce confusing tracebacks. Use <code>inspect.unwrap()</code> sparingly; it's a smell that your stack is too deep.</li>
  <li><strong>You need fine-grained control over each layer</strong> — if Decorator A and Decorator B interact (e.g., the cache must know about the timer), a simple class is cleaner than two wrappers that reach through each other.</li>
  <li><strong>Performance matters at microsecond scale</strong> — function call overhead from stacked Python decorators adds up. In tight inference loops, inlining is faster.</li>
  <li><strong>The wrapped interface changes often</strong> — if <code>Component</code>'s API is unstable, maintaining a parallel decorator hierarchy is expensive.</li>
  <li><strong>You have more than 3-4 stacked decorators</strong> — this is a complexity smell. Consolidate into a single configurable wrapper class.</li>
</ul>
</div>

---

## Complete Example: Composable Training Decorators

```python
"""
Putting it all together: a composable decorator stack for a training loop.
"""
import functools
import logging
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trainer")


# 1. Python @decorator for cross-cutting concerns
def logged(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        logger.info(f"Finished {func.__name__}")
        return result
    return wrapper


def timed(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        logger.info(f"{func.__name__}: {(time.perf_counter()-start)*1000:.1f}ms")
        return result
    return wrapper


# 2. GoF Decorator — wraps nn.Module, same interface
class DropoutInjector(nn.Module):
    """Decorator that injects dropout into any model for MC-dropout inference."""
    def __init__(self, model: nn.Module, p: float = 0.1):
        super().__init__()
        self._model = model
        self._dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._dropout(self._model(x))   # wrap output

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)


# 3. Compose them:
@logged
@timed
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
    return correct / total


base = nn.Linear(128, 10)
mc_model = DropoutInjector(base, p=0.15)   # GoF Decorator
# evaluate() uses Python @logged + @timed decorators
```

---

*Last updated: May 2026*
