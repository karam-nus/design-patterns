---
title: "Chapter 32 — Resilience Patterns for ML"
---

[← Back to Table of Contents](./README.md)

# Chapter 32 — Resilience Patterns for ML

> *"Anything that can go wrong will go wrong — and in ML systems, the blast radius is larger than you think."*

ML systems fail in ways that ordinary web services do not. A GPU runs out of memory mid-batch. A feature store returns stale vectors. A model checkpoint is silently corrupted. A downstream API times out at peak load. Resilience patterns are the engineering vocabulary for surviving these failures gracefully — keeping predictions flowing, degrading softly, and recovering automatically.

<div class="callout info">
<span class="callout-icon">ℹ️</span>
<div class="callout-body">
Resilience is not about preventing failures — it is about limiting their blast radius and recovering quickly. Each pattern in this chapter targets a different failure mode common to ML inference and training pipelines.
</div>
</div>

---

## Why Resilience Matters in ML Systems

Traditional web services fail predictably: a database connection drops, a timeout fires, a 500 is returned. ML systems add a new layer of exotic failure modes on top.

<div class="diagram">
<div class="diagram-title">ML-Specific Failure Modes</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card red">
    <div class="card-icon">💥</div>
    <div class="card-title">GPU OOM</div>
    <div class="card-desc">CUDA out-of-memory mid-inference; entire process may die without recovery logic</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">🌐</div>
    <div class="card-title">Network Failures</div>
    <div class="card-desc">Feature store, embedding API, or model registry unavailable during critical path</div>
  </div>
  <div class="diagram-card yellow">
    <div class="card-icon">🗑️</div>
    <div class="card-title">Corrupt Batches</div>
    <div class="card-desc">NaN gradients, malformed inputs, or truncated records poisoning a training run</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">⏱️</div>
    <div class="card-title">Inference Timeout</div>
    <div class="card-desc">Autoregressive models generating beyond SLA; synchronous callers blocked indefinitely</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">🔄</div>
    <div class="card-title">Model Load Failure</div>
    <div class="card-desc">Checkpoint missing, incompatible weights, or registry unreachable at startup</div>
  </div>
  <div class="diagram-card teal">
    <div class="card-icon">📉</div>
    <div class="card-title">Cascading Overload</div>
    <div class="card-desc">Retry storms amplifying upstream pressure; one slow model degrades the whole fleet</div>
  </div>
</div>
</div>

The patterns in this chapter address these failure modes systematically.

---

## Pattern 1 — Circuit Breaker

The Circuit Breaker prevents a system from repeatedly calling a failing dependency. Inspired by electrical circuit breakers, it has three states: **CLOSED** (normal operation), **OPEN** (failing fast), and **HALF_OPEN** (testing recovery).

<div class="diagram">
<div class="diagram-title">Circuit Breaker State Machine</div>
<div class="flow">
  <div class="flow-node green">CLOSED<br/><small>Requests flow normally</small></div>
  <div class="flow-arrow accent">failure threshold exceeded →</div>
  <div class="flow-node red">OPEN<br/><small>Fail fast, no calls made</small></div>
  <div class="flow-arrow purple">← reset timeout elapsed</div>
</div>
<div class="flow">
  <div class="flow-node orange">HALF_OPEN<br/><small>One probe request allowed</small></div>
  <div class="flow-arrow green">success → CLOSED</div>
  <div class="flow-arrow accent">failure → OPEN</div>
</div>
</div>

### Implementation

```python
import time
import threading
import logging
from enum import Enum
from functools import wraps
from typing import Callable, Optional, Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Raised when the circuit is OPEN and the call is rejected."""
    pass


class CircuitBreaker:
    """
    Thread-safe circuit breaker for ML inference calls.

    Args:
        failure_threshold: consecutive failures before opening the circuit
        recovery_timeout: seconds to wait before entering HALF_OPEN
        success_threshold: consecutive successes in HALF_OPEN to close circuit
        name: human-readable label for logs/metrics
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2,
        name: str = "circuit",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._get_state()

    def _get_state(self) -> CircuitState:
        """Internal — must be called with lock held."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - (self._last_failure_time or 0)
            if elapsed >= self.recovery_timeout:
                logger.info("[%s] Recovery timeout elapsed → HALF_OPEN", self.name)
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
        return self._state

    def call(self, func: Callable, *args, **kwargs) -> Any:
        with self._lock:
            state = self._get_state()

            if state == CircuitState.OPEN:
                raise CircuitBreakerError(
                    f"Circuit '{self.name}' is OPEN — failing fast"
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except CircuitBreakerError:
            raise
        except Exception as exc:
            self._on_failure(exc)
            raise

    def _on_success(self):
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    logger.info("[%s] Circuit CLOSED after recovery", self.name)
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
            else:
                self._failure_count = 0

    def _on_failure(self, exc: Exception):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            logger.warning(
                "[%s] Failure %d/%d: %s",
                self.name, self._failure_count, self.failure_threshold, exc,
            )
            if self._failure_count >= self.failure_threshold:
                if self._state != CircuitState.OPEN:
                    logger.error("[%s] Circuit OPEN", self.name)
                self._state = CircuitState.OPEN

    def __call__(self, func: Callable) -> Callable:
        """Use as a decorator."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper


# --- Usage example ---

inference_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=20.0,
    name="primary-model",
)


def call_primary_model(payload: dict) -> dict:
    # Real inference call here
    return {"prediction": 0.92}


def predict_with_breaker(payload: dict) -> dict:
    try:
        return inference_breaker.call(call_primary_model, payload)
    except CircuitBreakerError:
        logger.warning("Primary circuit open — serving fallback")
        return {"prediction": None, "fallback": True}
```

<div class="callout tip">
<span class="callout-icon">💡</span>
<div class="callout-body">
Tune <code>failure_threshold</code> and <code>recovery_timeout</code> separately for each downstream dependency. A GPU inference service needs a longer recovery timeout than a lightweight embedding API.
</div>
</div>

---

## Pattern 2 — Retry with Exponential Backoff

Transient failures — network blips, rate limits, momentary GPU stalls — are best handled with retries. Exponential backoff prevents thundering-herd retry storms by spreading retries over time.

```python
import time
import random
import logging
from functools import wraps
from typing import Tuple, Type

logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator: retry with exponential backoff + optional jitter.

    Args:
        max_attempts: total attempts including the first
        base_delay: initial wait in seconds
        max_delay: cap on wait between retries
        backoff_factor: multiplier applied each retry
        jitter: add random noise to avoid synchronized retries
        retryable_exceptions: only retry these exception types
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exc: Exception = RuntimeError("No attempts made")
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    wait = min(delay, max_delay)
                    if jitter:
                        wait *= (0.5 + random.random())
                    logger.warning(
                        "%s attempt %d/%d failed (%s) — retrying in %.2fs",
                        func.__name__, attempt, max_attempts, exc, wait,
                    )
                    time.sleep(wait)
                    delay *= backoff_factor
            raise last_exc
        return wrapper
    return decorator


# --- Usage ---

@retry(max_attempts=4, base_delay=0.5, retryable_exceptions=(ConnectionError, TimeoutError))
def fetch_embeddings(texts: list[str]) -> list[list[float]]:
    """Call remote embedding API with automatic retry."""
    # ... HTTP call here
    return [[0.1, 0.2] for _ in texts]
```

---

## Pattern 3 — Bulkhead

The Bulkhead pattern isolates failure domains by giving each subsystem its own resource pool. If the image-captioning model is saturated, the text-classification model is unaffected.

```python
import concurrent.futures
import logging
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)


class Bulkhead:
    """
    Limits concurrent calls to an ML service using a bounded thread pool.

    Args:
        name: label for this bulkhead (e.g., "image-model")
        max_concurrent: maximum simultaneous calls
        queue_size: additional requests to queue before rejecting
        timeout: per-call timeout in seconds
    """

    def __init__(
        self,
        name: str,
        max_concurrent: int = 4,
        queue_size: int = 8,
        timeout: float = 30.0,
    ):
        self.name = name
        self.timeout = timeout
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent,
            thread_name_prefix=f"bulkhead-{name}",
        )
        self._semaphore = __import__("threading").BoundedSemaphore(
            max_concurrent + queue_size
        )

    def call(self, func: Callable, *args, **kwargs) -> Any:
        acquired = self._semaphore.acquire(blocking=False)
        if not acquired:
            raise RuntimeError(
                f"Bulkhead '{self.name}' full — request rejected"
            )
        try:
            future = self._executor.submit(func, *args, **kwargs)
            return future.result(timeout=self.timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(
                f"Bulkhead '{self.name}': call exceeded {self.timeout}s"
            )
        finally:
            self._semaphore.release()

    def shutdown(self):
        self._executor.shutdown(wait=False)


# Separate bulkheads per model — failures don't bleed across
image_bulkhead = Bulkhead("image-model", max_concurrent=2)
text_bulkhead  = Bulkhead("text-model",  max_concurrent=8)
audio_bulkhead = Bulkhead("audio-model", max_concurrent=1)


def route_inference(modality: str, payload: dict) -> dict:
    if modality == "image":
        return image_bulkhead.call(run_image_model, payload)
    elif modality == "text":
        return text_bulkhead.call(run_text_model, payload)
    elif modality == "audio":
        return audio_bulkhead.call(run_audio_model, payload)
    raise ValueError(f"Unknown modality: {modality}")


def run_image_model(payload): return {"label": "cat"}
def run_text_model(payload):  return {"sentiment": "positive"}
def run_audio_model(payload): return {"transcript": "hello"}
```

---

## Pattern 4 — Fallback Model

When the primary model is unavailable, serve a lighter or cached result rather than returning an error to the user.

```python
import logging
import functools
from typing import Optional

logger = logging.getLogger(__name__)


class ModelFallbackChain:
    """
    Tries models in order, returning the first successful result.
    Caches the last successful response for each input hash.
    """

    def __init__(self, cache_size: int = 128):
        from collections import OrderedDict
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self._cache_size = cache_size

    def _cache_put(self, key: int, value: dict):
        self._cache[key] = value
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def predict(self, payload: dict) -> dict:
        payload_hash = hash(str(sorted(payload.items())))

        for model_fn, label in [
            (self._call_gpt4,       "gpt-4"),
            (self._call_gpt35,      "gpt-3.5"),
            (self._call_local_llm,  "local-llm"),
        ]:
            try:
                result = model_fn(payload)
                result["model_used"] = label
                self._cache_put(payload_hash, result)
                return result
            except Exception as exc:
                logger.warning("Fallback: %s failed (%s), trying next", label, exc)

        # Last resort: cached response
        cached = self._cache.get(payload_hash)
        if cached:
            logger.warning("All models failed — serving cached response")
            return {**cached, "model_used": "cache", "stale": True}

        return {"prediction": None, "model_used": "none", "error": True}

    def _call_gpt4(self, payload):       raise ConnectionError("demo")  # stub
    def _call_gpt35(self, payload):      raise ConnectionError("demo")  # stub
    def _call_local_llm(self, payload):  return {"prediction": "fallback answer"}


fallback_chain = ModelFallbackChain()
```

---

## Pattern 5 — Health Check

A `/health` endpoint gives orchestrators (Kubernetes, load balancers) a signal about model readiness.

```python
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading
import time
import torch


class ModelHealthState:
    def __init__(self):
        self.model_loaded = False
        self.last_inference_ok = True
        self.last_check_time = time.time()
        self.error_message: str = ""

    @property
    def healthy(self) -> bool:
        return self.model_loaded and self.last_inference_ok


health_state = ModelHealthState()


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/ready":
            self._handle_ready()
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_health(self):
        status = 200 if health_state.model_loaded else 503
        body = json.dumps({
            "status": "ok" if health_state.model_loaded else "degraded",
            "model_loaded": health_state.model_loaded,
            "last_inference_ok": health_state.last_inference_ok,
            "error": health_state.error_message,
        }).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def _handle_ready(self):
        status = 200 if health_state.healthy else 503
        self.send_response(status)
        self.end_headers()

    def log_message(self, *args):
        pass  # suppress default logging


def start_health_server(port: int = 8080):
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def load_model_with_health_tracking(checkpoint_path: str):
    try:
        health_state.model_loaded = False
        # model = torch.load(checkpoint_path)
        health_state.model_loaded = True
        return None  # return model
    except Exception as exc:
        health_state.error_message = str(exc)
        raise
```

---

## Pattern 6 — Timeout

Wrap any inference call with a deadline. Autoregressive models generating long sequences can block callers indefinitely without one.

```python
import signal
import threading
import concurrent.futures
import logging
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)


class InferenceTimeout(Exception):
    pass


def with_timeout(func: Callable, timeout_secs: float, *args, **kwargs) -> Any:
    """
    Run func(*args, **kwargs) with a hard timeout.
    Works in both main and non-main threads.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_secs)
        except concurrent.futures.TimeoutError:
            raise InferenceTimeout(
                f"{func.__name__} exceeded {timeout_secs}s deadline"
            )


def timeout(seconds: float):
    """Decorator version of with_timeout."""
    def decorator(func: Callable) -> Callable:
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):
            return with_timeout(func, seconds, *args, **kwargs)
        return wrapper
    return decorator


@timeout(seconds=5.0)
def generate_text(prompt: str, max_tokens: int = 256) -> str:
    """LLM generation with a 5-second hard deadline."""
    # ... model.generate(prompt, max_tokens=max_tokens)
    return "Generated response"


# Use at call site
def safe_generate(prompt: str) -> str:
    try:
        return generate_text(prompt)
    except InferenceTimeout:
        logger.warning("Generation timed out — returning empty string")
        return ""
```

---

## Pattern 7 — Graceful Degradation

When the system is overloaded, return a simplified or cached response rather than failing hard.

```python
import time
import threading
import logging
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class LoadShedder:
    """
    Tracks request latency and activates degraded mode under high load.

    In degraded mode, requests bypass expensive models and use a fast
    heuristic or cached response instead.
    """

    def __init__(
        self,
        latency_threshold_ms: float = 500.0,
        window_size: int = 50,
        degraded_ratio: float = 0.8,
    ):
        self._latencies: deque = deque(maxlen=window_size)
        self._threshold = latency_threshold_ms
        self._degraded_ratio = degraded_ratio
        self._lock = threading.Lock()

    def record_latency(self, latency_ms: float):
        with self._lock:
            self._latencies.append(latency_ms)

    @property
    def is_degraded(self) -> bool:
        with self._lock:
            if len(self._latencies) < 10:
                return False
            p80 = sorted(self._latencies)[int(len(self._latencies) * self._degraded_ratio)]
            return p80 > self._threshold

    def predict(self, payload: dict) -> dict:
        if self.is_degraded:
            logger.warning("System degraded — using fast heuristic")
            return self._fast_heuristic(payload)
        return self._full_model(payload)

    def _fast_heuristic(self, payload: dict) -> dict:
        return {"prediction": "unknown", "degraded": True, "confidence": 0.0}

    def _full_model(self, payload: dict) -> dict:
        start = time.monotonic()
        result = {"prediction": "positive", "confidence": 0.87}
        self.record_latency((time.monotonic() - start) * 1000)
        return result
```

---

## Pattern 8 — OOM Recovery

CUDA out-of-memory errors are recoverable if you catch them, reduce the batch size, and retry.

```python
import logging
import torch
from typing import Callable, Any

logger = logging.getLogger(__name__)


class OOMRecovery:
    """
    Catches CUDA OOM errors, halves the batch size, and retries.

    Args:
        min_batch_size: stop retrying below this threshold
        max_retries: maximum reduction attempts
    """

    def __init__(self, min_batch_size: int = 1, max_retries: int = 4):
        self.min_batch_size = min_batch_size
        self.max_retries = max_retries

    def run(self, model_fn: Callable, batch: list, **kwargs) -> Any:
        current_batch = batch
        for attempt in range(self.max_retries + 1):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return model_fn(current_batch, **kwargs)

            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise  # not an OOM — propagate

                if len(current_batch) <= self.min_batch_size:
                    logger.error("OOM at minimum batch size %d — giving up", self.min_batch_size)
                    raise

                new_size = max(len(current_batch) // 2, self.min_batch_size)
                logger.warning(
                    "CUDA OOM (attempt %d) — reducing batch %d → %d",
                    attempt + 1, len(current_batch), new_size,
                )
                current_batch = current_batch[:new_size]

        raise RuntimeError("Exceeded OOM retry budget")


oom_recovery = OOMRecovery(min_batch_size=1, max_retries=4)


def run_batch_inference(batch: list) -> list:
    def model_forward(b):
        # torch model call here
        return [{"score": 0.9}] * len(b)

    return oom_recovery.run(model_forward, batch)
```

<div class="callout warn">
<span class="callout-icon">⚠️</span>
<div class="callout-body">
Always call <code>torch.cuda.empty_cache()</code> before retrying after an OOM. The freed memory may not be immediately available otherwise, causing repeated failures even with a smaller batch.
</div>
</div>

---

## Pattern 9 — Corrupt Batch Handler

During training, a single malformed example (NaN values, wrong dtype, extreme values) can crash an entire run. Skip bad batches, log them, and continue.

```python
import logging
import torch
from typing import Iterator, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BatchStats:
    total: int = 0
    skipped: int = 0
    skip_reasons: list = field(default_factory=list)

    @property
    def skip_rate(self) -> float:
        return self.skipped / max(self.total, 1)


def validate_batch(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[bool, str]:
    """Return (is_valid, reason_if_invalid)."""
    if torch.isnan(inputs).any():
        return False, "NaN in inputs"
    if torch.isinf(inputs).any():
        return False, "Inf in inputs"
    if torch.isnan(targets).any():
        return False, "NaN in targets"
    if inputs.shape[0] == 0:
        return False, "Empty batch"
    if inputs.shape[0] != targets.shape[0]:
        return False, f"Shape mismatch: {inputs.shape[0]} vs {targets.shape[0]}"
    return True, ""


def safe_training_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: Any,
    max_skip_rate: float = 0.05,
) -> BatchStats:
    stats = BatchStats()
    criterion = torch.nn.CrossEntropyLoss()

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        stats.total += 1

        valid, reason = validate_batch(inputs, targets)
        if not valid:
            stats.skipped += 1
            stats.skip_reasons.append((batch_idx, reason))
            logger.warning("Skipping batch %d: %s", batch_idx, reason)

            if stats.skip_rate > max_skip_rate:
                raise RuntimeError(
                    f"Skip rate {stats.skip_rate:.1%} exceeds threshold "
                    f"{max_skip_rate:.1%} — data pipeline issue likely"
                )
            continue

        try:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if torch.isnan(loss):
                stats.skipped += 1
                stats.skip_reasons.append((batch_idx, "NaN loss"))
                logger.warning("NaN loss at batch %d — skipping gradient step", batch_idx)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        except Exception as exc:
            stats.skipped += 1
            stats.skip_reasons.append((batch_idx, str(exc)))
            logger.error("Batch %d failed: %s", batch_idx, exc, exc_info=True)

    logger.info(
        "Training complete: %d/%d batches processed (%.1f%% skipped)",
        stats.total - stats.skipped, stats.total, stats.skip_rate * 100,
    )
    return stats
```

---

## Full Request Flow with Resilience Layers

<div class="diagram">
<div class="diagram-title">Resilience-Wrapped Inference Pipeline</div>
<div class="flow">
  <div class="flow-node blue wide">Incoming Request</div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-node teal wide">Timeout Guard<br/><small>5s hard deadline</small></div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-node orange wide">Bulkhead<br/><small>max 4 concurrent calls</small></div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-node purple wide">Circuit Breaker<br/><small>CLOSED / OPEN / HALF_OPEN</small></div>
</div>
<div class="flow">
  <div class="flow-arrow accent">↙ circuit CLOSED</div>
  <div class="flow-arrow accent">↘ circuit OPEN</div>
</div>
<div class="flow">
  <div class="flow-node green">Primary Model<br/><small>GPU inference</small></div>
  <div class="flow-node red">Fallback Model<br/><small>cached / smaller model</small></div>
</div>
<div class="flow">
  <div class="flow-arrow accent">↓ success</div>
  <div class="flow-arrow accent">↓ fail → retry</div>
</div>
<div class="flow">
  <div class="flow-node blue wide">Response</div>
</div>
</div>

---

## Comparison Table

<table class="compare-table">
<thead>
<tr>
  <th>Pattern</th>
  <th>Failure Mode Addressed</th>
  <th>Recovery Strategy</th>
  <th>Complexity</th>
  <th>Best For</th>
</tr>
</thead>
<tbody>
<tr>
  <td><strong>Circuit Breaker</strong></td>
  <td>Cascading failures to slow dependency</td>
  <td>Fail fast + timed recovery</td>
  <td>Medium</td>
  <td>External APIs, model servers</td>
</tr>
<tr>
  <td><strong>Retry + Backoff</strong></td>
  <td>Transient network / rate-limit errors</td>
  <td>Repeat with increasing delay</td>
  <td>Low</td>
  <td>Feature stores, embedding APIs</td>
</tr>
<tr>
  <td><strong>Bulkhead</strong></td>
  <td>Resource exhaustion bleeding across services</td>
  <td>Bounded thread/process pools</td>
  <td>Low–Medium</td>
  <td>Multi-model serving</td>
</tr>
<tr>
  <td><strong>Timeout</strong></td>
  <td>Long-tail inference latency</td>
  <td>Hard deadline + exception</td>
  <td>Low</td>
  <td>All synchronous calls</td>
</tr>
<tr>
  <td><strong>Fallback</strong></td>
  <td>Primary model unavailable</td>
  <td>Serve smaller model or cache</td>
  <td>Medium</td>
  <td>User-facing predictions</td>
</tr>
<tr>
  <td><strong>Graceful Degradation</strong></td>
  <td>Overload / sustained high latency</td>
  <td>Heuristic fast path</td>
  <td>Medium</td>
  <td>High-traffic inference</td>
</tr>
<tr>
  <td><strong>OOM Recovery</strong></td>
  <td>CUDA out-of-memory</td>
  <td>Reduce batch size, retry</td>
  <td>Low</td>
  <td>GPU batch inference</td>
</tr>
<tr>
  <td><strong>Corrupt Batch Handler</strong></td>
  <td>NaN / malformed training data</td>
  <td>Skip bad batches, continue</td>
  <td>Low</td>
  <td>Training loops</td>
</tr>
</tbody>
</table>

---

## Composing the Patterns

Real-world ML services stack multiple resilience layers. The recommended composition order from outermost to innermost:

<div class="diagram">
<div class="diagram-grid cols-2">
  <div class="diagram-card blue">
    <div class="card-icon">1️⃣</div>
    <div class="card-title">Timeout (outermost)</div>
    <div class="card-desc">Every call has a deadline. Enforced before any other logic runs.</div>
  </div>
  <div class="diagram-card teal">
    <div class="card-icon">2️⃣</div>
    <div class="card-title">Bulkhead</div>
    <div class="card-desc">Bound concurrency so a surge doesn't starve all threads.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">3️⃣</div>
    <div class="card-title">Circuit Breaker</div>
    <div class="card-desc">Trip on sustained failure; stop hammering a broken dependency.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">4️⃣</div>
    <div class="card-title">Retry (innermost)</div>
    <div class="card-desc">Retry only transient errors — inside the circuit breaker's view.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">5️⃣</div>
    <div class="card-title">Fallback (always present)</div>
    <div class="card-desc">If everything above fails, serve a degraded but non-error response.</div>
  </div>
  <div class="diagram-card red">
    <div class="card-icon">6️⃣</div>
    <div class="card-title">Health Check (side channel)</div>
    <div class="card-desc">Liveness and readiness probes run independently of the request path.</div>
  </div>
</div>
</div>

```python
import time
import logging

logger = logging.getLogger(__name__)

# Compose all patterns into a single resilient predict function
timeout_guard   = lambda f: timeout(seconds=5.0)(f)
model_bulkhead  = Bulkhead("primary", max_concurrent=4)
model_breaker   = CircuitBreaker(failure_threshold=5, name="primary")
fallback        = ModelFallbackChain()


def resilient_predict(payload: dict) -> dict:
    """
    Outer: timeout → bulkhead → circuit breaker → retry → fallback
    """
    @timeout(seconds=5.0)
    def _inner():
        try:
            return model_bulkhead.call(
                lambda: model_breaker.call(
                    lambda: _retry_predict(payload)
                )
            )
        except (CircuitBreakerError, RuntimeError, TimeoutError) as exc:
            logger.warning("Primary path failed (%s) — using fallback", exc)
            return fallback.predict(payload)
    return _inner()


@retry(max_attempts=2, base_delay=0.1, retryable_exceptions=(ConnectionError,))
def _retry_predict(payload: dict) -> dict:
    return {"prediction": "ok", "confidence": 0.95}
```

<div class="callout tip">
<span class="callout-icon">💡</span>
<div class="callout-body">
Export key metrics from each resilience layer — circuit state transitions, OOM recovery counts, bulkhead rejection rates — to your observability platform. Silent failures are the most dangerous kind.
</div>
</div>

---

## Summary

<div class="diagram">
<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-year">Circuit Breaker</div>
    <div class="timeline-title">Stop cascading failures</div>
    <div class="timeline-desc">Trip after N failures; recover after a timeout; probe with HALF_OPEN.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Retry</div>
    <div class="timeline-title">Recover from transient faults</div>
    <div class="timeline-desc">Exponential backoff with jitter prevents thundering herds.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Bulkhead</div>
    <div class="timeline-title">Contain resource exhaustion</div>
    <div class="timeline-desc">Separate thread pools per model; overload in one doesn't starve others.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Fallback</div>
    <div class="timeline-title">Serve something useful</div>
    <div class="timeline-desc">Cached result, smaller model, or heuristic beats an HTTP 500.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">OOM Recovery</div>
    <div class="timeline-title">Survive GPU memory pressure</div>
    <div class="timeline-desc">Catch CUDA OOM, halve batch size, retry — down to minimum batch.</div>
  </div>
</div>
</div>

<span class="badge mlops">MLOps</span> <span class="badge behavioral">Behavioral</span>

*Last updated: May 2026*
