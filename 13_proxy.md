---
title: "Chapter 13 — Proxy & Lazy Loading"
---
[← Back to Table of Contents](./README.md)

# Chapter 13 — Proxy & Lazy Loading

> *"Provide a surrogate or placeholder for another object to control access to it."*
> — Gang of Four

---

## Intent and Motivation

A model with 7 billion parameters takes 30+ seconds to load from disk. A remote inference server is not always reachable. A shared model should not be called more than 100 times per second. These are access-control problems — and the Proxy pattern solves all of them using the same structure: an object that *looks* like the real thing but intercepts access to add a layer of control.

The Proxy implements the same interface as the real subject, so the client never needs to change its code. Swapping a `LazyModel` proxy for a `RealModel` is transparent.

---

## Four Proxy Types

<div class="diagram-grid cols-4">
  <div class="diagram-card orange">
    <div class="card-icon">😴</div>
    <div class="card-title">Virtual Proxy</div>
    <div class="card-desc">Defers expensive initialisation until the object is actually needed (lazy loading). Saves startup time and memory.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🌐</div>
    <div class="card-title">Remote Proxy</div>
    <div class="card-desc">Represents an object in a different process or machine. Handles serialisation, network calls, and error handling transparently.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">🔒</div>
    <div class="card-title">Protection Proxy</div>
    <div class="card-desc">Controls access based on permissions, rate limits, or operational mode (e.g., read-only inference replica).</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">⚡</div>
    <div class="card-title">Caching Proxy</div>
    <div class="card-desc">Memoises results for identical inputs. Avoids redundant computation for repeated requests with the same data.</div>
  </div>
</div>

---

## GoF Proxy: UML Structure

<div class="diagram">
  <div class="diagram-title">Proxy Pattern — UML</div>
  <div class="uml-row">
    <div class="uml-box">
      <div class="uml-title">«interface»<br>Subject</div>
      <div class="uml-section">
        <div class="uml-item">+ request()</div>
      </div>
    </div>
  </div>
  <div class="uml-row" style="margin-top:1rem">
    <div class="uml-box">
      <div class="uml-title">RealSubject</div>
      <div class="uml-section">
        <div class="uml-item">+ request()</div>
      </div>
    </div>
    <div class="uml-box" style="border-color: var(--accent)">
      <div class="uml-title">Proxy</div>
      <div class="uml-section">
        <div class="uml-item">- real_subject: RealSubject</div>
      </div>
      <div class="uml-section">
        <div class="uml-item">+ request()</div>
        <div class="uml-item">- check_access()</div>
        <div class="uml-item">- log_access()</div>
      </div>
    </div>
  </div>
</div>

---

## Virtual Proxy / Lazy Loading in ML

### LazyModel — Loading Weights Only on First Forward Pass

Large models (LLaMA-70B, Stable Diffusion XL) have startup times of 30–120 seconds. A `LazyModel` proxy lets you import and pass around model *references* without paying the load cost until the first actual call:

```python
from __future__ import annotations

import threading
import torch
import torch.nn as nn
from typing import Any


class LazyModel(nn.Module):
    """
    Virtual Proxy for any nn.Module.
    The real model is not loaded until the first forward() call.
    Thread-safe via double-checked locking.
    """
    def __init__(self, model_path: str, model_class: type, **model_kwargs):
        super().__init__()
        self._model_path = model_path
        self._model_class = model_class
        self._model_kwargs = model_kwargs
        self._real_model: nn.Module | None = None
        self._lock = threading.Lock()

    def _load(self) -> nn.Module:
        """Lazy initialisation — load once, reuse forever."""
        if self._real_model is None:
            with self._lock:
                if self._real_model is None:   # double-checked
                    print(f"[LazyModel] Loading {self._model_class.__name__} from {self._model_path}...")
                    model = self._model_class(**self._model_kwargs)
                    state = torch.load(self._model_path, map_location="cpu")
                    model.load_state_dict(state)
                    model.eval()
                    self._real_model = model
                    print(f"[LazyModel] Loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return self._real_model

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._load()(*args, **kwargs)   # load on first call

    def __getattr__(self, name: str) -> Any:
        """Transparent attribute access — delegates to real model if loaded."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if self._real_model is not None:
                return getattr(self._real_model, name)
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}' "
                f"(model not yet loaded; call forward() first)"
            )

    def is_loaded(self) -> bool:
        return self._real_model is not None


# Usage — model is NOT loaded at import time:
model = LazyModel("./checkpoints/best.pt", MyTransformer, hidden=768, layers=12)

# ... pass model around freely, inject into services ...

# Only now does the 30-second load happen:
output = model(torch.randn(1, 128))
print(model.is_loaded())   # True
```

### Lazy Tensor Loading — Large Datasets On Demand

```python
from torch.utils.data import Dataset
from pathlib import Path
import torch
import numpy as np
from typing import Any


class LazyShardedDataset(Dataset):
    """
    Virtual Proxy dataset: 100GB dataset loaded shard-by-shard on access.
    Only one shard lives in memory at a time.
    """
    def __init__(self, shard_dir: str, shard_size: int = 10_000):
        self._shard_dir = Path(shard_dir)
        self._shard_size = shard_size
        self._shard_paths = sorted(self._shard_dir.glob("shard_*.npy"))
        self._current_shard_idx = -1
        self._current_shard: np.ndarray | None = None

        # Read index from a small metadata file — no data loaded yet
        meta_path = self._shard_dir / "meta.json"
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        self._total_len = meta["total_samples"]

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        shard_idx = idx // self._shard_size
        local_idx = idx % self._shard_size

        # Lazy load — only swap shard when needed
        if shard_idx != self._current_shard_idx:
            self._current_shard = np.load(self._shard_paths[shard_idx], mmap_mode="r")
            self._current_shard_idx = shard_idx

        sample = self._current_shard[local_idx]
        return torch.from_numpy(sample[:-1].copy()).float(), int(sample[-1])


class LazyTensor:
    """
    __getattr__-based lazy proxy for a single tensor.
    The tensor is loaded from disk only when an attribute is first accessed.
    """
    def __init__(self, path: str):
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_tensor", None)

    def _load(self) -> torch.Tensor:
        t = object.__getattribute__(self, "_tensor")
        if t is None:
            path = object.__getattribute__(self, "_path")
            t = torch.load(path)
            object.__setattr__(self, "_tensor", t)
        return t

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __repr__(self) -> str:
        t = object.__getattribute__(self, "_tensor")
        if t is None:
            return f"LazyTensor(path={object.__getattribute__(self, '_path')!r}, not loaded)"
        return f"LazyTensor({t!r})"
```

### `__getattr__`-Based Generic Lazy Proxy

```python
class LazyProxy:
    """
    Generic lazy proxy using __getattr__. The factory is called only
    when the first attribute is accessed. Works for any object.
    """
    def __init__(self, factory):
        object.__setattr__(self, "_factory", factory)
        object.__setattr__(self, "_instance", None)

    def _get_instance(self):
        instance = object.__getattribute__(self, "_instance")
        if instance is None:
            factory = object.__getattribute__(self, "_factory")
            instance = factory()
            object.__setattr__(self, "_instance", instance)
        return instance

    def __getattr__(self, name):
        return getattr(self._get_instance(), name)

    def __call__(self, *args, **kwargs):
        return self._get_instance()(*args, **kwargs)

    def __repr__(self):
        instance = object.__getattribute__(self, "_instance")
        if instance is None:
            return f"LazyProxy(not yet initialised)"
        return repr(instance)


# Usage:
tokenizer = LazyProxy(lambda: __import__("transformers").AutoTokenizer.from_pretrained("bert-base-uncased"))
# Tokenizer is NOT loaded yet.
encoded = tokenizer("Hello world")   # loaded here, transparently
```

---

## Remote Proxy in ML

### Proxy for a Model Served Over REST

When a model is too large for the inference pod and lives on a dedicated server, a remote proxy makes the remote call look like a local `nn.Module`:

```python
import requests
import torch
import torch.nn as nn
import numpy as np
from typing import Any


class RemoteModelProxy(nn.Module):
    """
    Remote Proxy: wraps a model deployed as a REST service.
    The caller's code is identical to calling a local nn.Module.
    """
    def __init__(
        self,
        endpoint: str,
        timeout: float = 10.0,
        api_key: str | None = None,
    ):
        super().__init__()
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
        payload = {
            "input_ids": input_ids.cpu().numpy().tolist(),
            **{k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v
               for k, v in kwargs.items()},
        }
        response = requests.post(
            f"{self._endpoint}/predict",
            json=payload,
            headers=self._headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        return torch.tensor(data["logits"])

    def health_check(self) -> bool:
        try:
            r = requests.get(f"{self._endpoint}/health", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False


# Client code looks identical whether model is local or remote:
model = RemoteModelProxy("https://ml-inference.internal:8080", api_key="token-xxx")
logits = model(input_ids=torch.tensor([[101, 2054, 2003, 2023, 102]]))
```

---

## Protection Proxy

### Rate-Limited Model Proxy

```python
import time
import threading
import torch
import torch.nn as nn
from collections import deque
from typing import Any


class RateLimitedProxy(nn.Module):
    """
    Protection Proxy: limits the model to max_calls_per_second.
    Raises RuntimeError if the rate is exceeded.
    """
    def __init__(
        self,
        model: nn.Module,
        max_calls_per_second: float = 100.0,
        burst_capacity: int = 10,
    ):
        super().__init__()
        self._model = model
        self._rate = max_calls_per_second
        self._burst = burst_capacity
        self._tokens = float(burst_capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        self._call_count = 0
        self._rejected_count = 0

    def _acquire_token(self) -> bool:
        """Token bucket algorithm."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._burst,
                self._tokens + elapsed * self._rate
            )
            self._last_refill = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if not self._acquire_token():
            self._rejected_count += 1
            raise RuntimeError(
                f"Rate limit exceeded ({self._rate:.0f} calls/s). "
                f"Rejected {self._rejected_count} calls total."
            )
        self._call_count += 1
        return self._model(*args, **kwargs)

    def stats(self) -> dict:
        return {"accepted": self._call_count, "rejected": self._rejected_count}


class ReadOnlyModelProxy(nn.Module):
    """
    Protection Proxy: prevents any modification to the wrapped model.
    Forces inference-only usage (no training, no weight updates).
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        object.__setattr__(self, "_model", model.eval())
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad_(False)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        with torch.no_grad():
            return object.__getattribute__(self, "_model")(*args, **kwargs)

    def train(self, mode: bool = True):
        if mode:
            raise PermissionError(
                "ReadOnlyModelProxy: cannot set model to training mode. "
                "Use the original model reference for training."
            )
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        raise PermissionError(
            f"ReadOnlyModelProxy: cannot set attribute '{name}'. "
            "This proxy is read-only."
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_model"), name)
```

---

## Caching Proxy

```python
import hashlib
import pickle
from typing import Any
import torch
import torch.nn as nn


class CachingModelProxy(nn.Module):
    """
    Caching Proxy: memoises forward pass results for identical inputs.
    Useful for embeddings, encoders, and any deterministic model.
    """
    def __init__(self, model: nn.Module, maxsize: int = 256):
        super().__init__()
        self._model = model.eval()
        self._cache: dict[str, Any] = {}
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0

    def _make_key(self, *args: Any, **kwargs: Any) -> str:
        """Create a stable hash key from tensor contents."""
        parts = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                parts.append(arg.cpu().numpy().tobytes())
            else:
                parts.append(pickle.dumps(arg))
        for k, v in sorted(kwargs.items()):
            parts.append(k.encode())
            if isinstance(v, torch.Tensor):
                parts.append(v.cpu().numpy().tobytes())
            else:
                parts.append(pickle.dumps(v))
        return hashlib.sha256(b"".join(parts)).hexdigest()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        key = self._make_key(*args, **kwargs)

        if key in self._cache:
            self._hits += 1
            return self._cache[key]

        self._misses += 1
        with torch.no_grad():
            result = self._model(*args, **kwargs)

        # Simple LRU eviction: drop oldest entry when full
        if len(self._cache) >= self._maxsize:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = result
        return result

    def cache_stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "cache_size": len(self._cache),
        }

    def clear_cache(self):
        self._cache.clear()
        self._hits = self._misses = 0
```

---

## Mock Objects in Testing as Proxies

`unittest.mock.MagicMock` is a proxy pattern in disguise — it intercepts any attribute access or call and records it, without involving the real subject:

```python
from unittest.mock import MagicMock, patch
import torch


def test_service_calls_model_once_per_request():
    """Use MagicMock as a Proxy to verify interaction patterns."""
    mock_model = MagicMock()
    mock_model.return_value = torch.zeros(2, 10)  # fake logits

    service = ModelService(
        model=mock_model,
        tokenizer=MagicMock(),
        label_map={0: "neg", 1: "pos"},
    )
    service.predict(["hello", "world"])

    # MagicMock recorded the call — verify without loading a real model:
    mock_model.assert_called_once()
    call_kwargs = mock_model.call_args.kwargs
    assert "input_ids" in call_kwargs


def test_lazy_model_loads_only_once(tmp_path):
    """Verify lazy proxy loads exactly once even with multiple calls."""
    load_count = 0

    def expensive_factory():
        nonlocal load_count
        load_count += 1
        return MagicMock(return_value=torch.zeros(1))

    proxy = LazyProxy(expensive_factory)

    for _ in range(10):
        proxy(torch.randn(1))

    assert load_count == 1, "Factory must be called exactly once"
```

---

## Flow Diagram

<div class="diagram">
  <div class="diagram-title">Proxy Pattern — Runtime Flow</div>
  <div class="flow">
    <div class="flow-node accent wide">Client: <code>model(x)</code></div>
    <div class="flow-arrow accent">↓</div>
    <div class="flow-node blue wide">Proxy.forward(x)<br><small>intercepts the call</small></div>
    <div class="flow-arrow blue">↓ check</div>
    <div class="flow-h">
      <div class="flow-node orange narrow">Rate limit<br>exceeded?</div>
      <div class="flow-node purple narrow">Cache<br>hit?</div>
      <div class="flow-node teal narrow">Already<br>loaded?</div>
    </div>
    <div class="flow-arrow green">↓ pass through</div>
    <div class="flow-node green wide">Real Subject (Model)<br><small>actual computation</small></div>
    <div class="flow-arrow green">↑ result</div>
    <div class="flow-node blue wide">Proxy stores in cache / logs / updates stats</div>
    <div class="flow-arrow accent">↑</div>
    <div class="flow-node accent wide">Client receives result (transparently)</div>
  </div>
</div>

---

## Proxy Comparison Table

<table class="compare-table">
  <thead>
    <tr>
      <th>Proxy Type</th>
      <th>Problem Solved</th>
      <th>ML Use Case</th>
      <th>Key Mechanism</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Virtual</strong></td>
      <td>Expensive creation</td>
      <td>Load 70B model lazily</td>
      <td>Load on first access</td>
    </tr>
    <tr>
      <td><strong>Remote</strong></td>
      <td>Object in another process</td>
      <td>gRPC/REST model server</td>
      <td>Serialise + HTTP/gRPC call</td>
    </tr>
    <tr>
      <td><strong>Protection</strong></td>
      <td>Unauthorised access</td>
      <td>Rate limiting, read-only</td>
      <td>Check before delegating</td>
    </tr>
    <tr>
      <td><strong>Caching</strong></td>
      <td>Repeated identical calls</td>
      <td>Embedding cache, KV store</td>
      <td>Hash inputs, store results</td>
    </tr>
  </tbody>
</table>

---

<div class="callout tip">
<strong>When lazy loading significantly speeds up development and testing</strong>

Loading a 7B-parameter model takes 45–90 seconds. With a <code>LazyProxy</code>:
<ul>
  <li><strong>Unit tests</strong> that don't exercise the model path run instantly — the model is never loaded.</li>
  <li><strong>Import time</strong> for your module is milliseconds instead of minutes.</li>
  <li><strong>Multi-model services</strong> that rarely use all models at once save gigabytes of GPU memory.</li>
  <li><strong>Development iteration</strong> becomes much faster: code changes, linting, and type-checking run without waiting for model I/O.</li>
</ul>
Rule of thumb: if your model takes >5 seconds to load, wrap it in a <code>LazyProxy</code> for any non-training context.
</div>

---

*Last updated: May 2026*
