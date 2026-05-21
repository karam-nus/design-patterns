---
title: "Chapter 31 — Model Serving Patterns"
---

[← Back to Table of Contents](./README.md)

# Chapter 31 — Model Serving Patterns

> *"A model that isn't deployed is a research paper. A model that crashes under load is a liability. The gap between the two is serving engineering."*

---

## Serving Taxonomy

Before choosing a serving architecture, understand the three fundamental paradigms:

<div class="diagram">
<div class="diagram-title">Serving Modalities</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card blue">
    <div class="card-icon">📦</div>
    <div class="card-title">Batch Serving</div>
    <div class="card-desc">Process a dataset offline. High throughput, latency doesn't matter. Use for: daily recommendations, fraud scoring on yesterday's transactions, offline embeddings. Schedule with Airflow/Prefect.</div>
  </div>
  <div class="diagram-card accent">
    <div class="card-icon">⚡</div>
    <div class="card-title">Online (REST/gRPC) Serving</div>
    <div class="card-desc">Synchronous request-response. P99 latency matters. Use for: real-time classification, search ranking, chatbots. Deploy behind a load balancer. Horizontal scaling via replicas.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🌊</div>
    <div class="card-title">Streaming Serving</div>
    <div class="card-desc">Consume from a Kafka/Kinesis topic, score, and publish results. Continuous, event-driven. Use for: fraud detection on payment stream, live feed moderation, telemetry anomaly detection.</div>
  </div>
</div>
</div>

<table class="compare-table">
<thead>
<tr>
  <th>Dimension</th>
  <th>Batch</th>
  <th>Online</th>
  <th>Streaming</th>
</tr>
</thead>
<tbody>
<tr>
  <td><strong>Latency</strong></td>
  <td>Hours–Days</td>
  <td>1–100ms</td>
  <td>10ms–1s</td>
</tr>
<tr>
  <td><strong>Throughput</strong></td>
  <td>Very High</td>
  <td>Medium</td>
  <td>High</td>
</tr>
<tr>
  <td><strong>Freshness</strong></td>
  <td>Stale</td>
  <td>Real-time</td>
  <td>Near real-time</td>
</tr>
<tr>
  <td><strong>Infra Complexity</strong></td>
  <td>Low</td>
  <td>Medium</td>
  <td>High</td>
</tr>
<tr>
  <td><strong>Cost</strong></td>
  <td>Low (spot GPUs)</td>
  <td>Medium (on-demand)</td>
  <td>High (always-on)</td>
</tr>
</tbody>
</table>

---

## REST Serving with FastAPI

FastAPI is the standard for Python ML serving: async, fast, auto-documented, and trivially containerized.

```python
# serve/app.py
import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import numpy as np

log = logging.getLogger(__name__)


# ── Request / Response Models ───────────────────────────────────────────────────

class PredictRequest(BaseModel):
    inputs: list[list[float]] = Field(
        ...,
        description="Batch of input feature vectors",
        min_items=1,
        max_items=256,
    )
    model_version: str = Field(
        default="v1",
        description="Model version to use for prediction",
    )
    return_probabilities: bool = Field(
        default=False,
        description="If true, return class probabilities alongside top-1",
    )

    @validator("inputs")
    def validate_input_shape(cls, v):
        expected_dim = 512   # example feature dimension
        for i, row in enumerate(v):
            if len(row) != expected_dim:
                raise ValueError(
                    f"Input {i} has {len(row)} features, expected {expected_dim}"
                )
        return v


class Prediction(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float] | None = None


class PredictResponse(BaseModel):
    request_id: str
    predictions: list[Prediction]
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    device: str


# ── Model Registry ──────────────────────────────────────────────────────────────

class ModelRegistry:
    """Thread-safe registry of loaded model versions."""

    def __init__(self):
        self._models: dict[str, nn.Module] = {}
        self._lock = asyncio.Lock()
        self._label_map: dict[str, list[str]] = {}

    async def load(
        self,
        version: str,
        checkpoint_path: str,
        label_names: list[str],
        device: str = "cuda",
    ) -> None:
        async with self._lock:
            if version in self._models:
                return
            model = build_model_from_checkpoint(checkpoint_path)
            model.eval()
            model = model.to(device)
            self._models[version] = model
            self._label_map[version] = label_names
            log.info(f"Loaded model version {version} on {device}")

    def get(self, version: str) -> tuple[nn.Module, list[str]]:
        if version not in self._models:
            raise KeyError(f"Model version '{version}' not loaded")
        return self._models[version], self._label_map[version]

    @property
    def versions(self) -> list[str]:
        return list(self._models.keys())


# ── Application Startup / Shutdown ─────────────────────────────────────────────

registry = ModelRegistry()
start_time = time.monotonic()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, release on shutdown."""
    log.info("Loading models...")
    await registry.load(
        version="v1",
        checkpoint_path="/models/v1/best.pth",
        label_names=["cat", "dog", "bird"],
        device=DEVICE,
    )
    await registry.load(
        version="v2",
        checkpoint_path="/models/v2/best.pth",
        label_names=["cat", "dog", "bird"],
        device=DEVICE,
    )
    log.info(f"Models loaded: {registry.versions}")
    yield
    log.info("Shutting down — releasing model memory")
    registry._models.clear()


# ── FastAPI App ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ML Model Server",
    version="1.0.0",
    description="Production model serving API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness + readiness probe."""
    return HealthResponse(
        status="healthy" if registry.versions else "loading",
        model_loaded=bool(registry.versions),
        model_version=registry.versions[0] if registry.versions else "none",
        uptime_seconds=time.monotonic() - start_time,
        device=DEVICE,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Run inference on a batch of inputs."""
    request_id = str(uuid.uuid4())[:8]
    t0 = time.monotonic()

    try:
        model, label_names = registry.get(request.model_version)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Build input tensor
    inputs = torch.tensor(request.inputs, dtype=torch.float32).to(DEVICE)

    with torch.inference_mode():
        logits = model(inputs)
        probs = torch.softmax(logits, dim=-1)
        confidences, indices = probs.max(dim=-1)

    predictions = []
    for i in range(len(request.inputs)):
        pred = Prediction(
            label=label_names[indices[i].item()],
            confidence=float(confidences[i].item()),
        )
        if request.return_probabilities:
            pred.probabilities = {
                label_names[j]: float(probs[i, j].item())
                for j in range(len(label_names))
            }
        predictions.append(pred)

    latency_ms = (time.monotonic() - t0) * 1000

    return PredictResponse(
        request_id=request_id,
        predictions=predictions,
        model_version=request.model_version,
        latency_ms=latency_ms,
    )


@app.post("/v1/predict", response_model=PredictResponse)
async def predict_v1(request: PredictRequest):
    """Versioned endpoint — always routes to v1."""
    request.model_version = "v1"
    return await predict(request)


@app.post("/v2/predict", response_model=PredictResponse)
async def predict_v2(request: PredictRequest):
    """Versioned endpoint — always routes to v2."""
    request.model_version = "v2"
    return await predict(request)
```

Run it:
```bash
uvicorn serve.app:app --host 0.0.0.0 --port 8080 --workers 4
```

---

## gRPC Serving

gRPC is preferred when latency is critical, clients are internal services, or you need streaming predictions. Define the contract in a `.proto` file:

```python
# proto/predict.proto (conceptual — not executed directly)
# syntax = "proto3";
# service Predictor {
#   rpc Predict (PredictRequest) returns (PredictResponse);
#   rpc PredictStream (stream PredictRequest) returns (stream PredictResponse);
# }
# message PredictRequest {
#   repeated float inputs = 1;
#   string model_version = 2;
# }
# message PredictResponse {
#   string label = 1;
#   float confidence = 2;
#   string request_id = 3;
# }

import grpc
from concurrent import futures
import torch


class PredictorServicer:
    """gRPC servicer implementation."""

    def __init__(self, model: torch.nn.Module, label_names: list[str]):
        self.model = model
        self.label_names = label_names
        self.model.eval()

    def Predict(self, request, context):
        inputs = torch.tensor(
            [request.inputs], dtype=torch.float32
        )
        with torch.inference_mode():
            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=-1)
            conf, idx = probs.max(dim=-1)

        # In real code, return the generated protobuf response object
        return {
            "label": self.label_names[idx.item()],
            "confidence": float(conf.item()),
            "request_id": "grpc-" + str(hash(str(request.inputs)))[:6],
        }


def serve_grpc(model, label_names, port: int = 50051):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_receive_message_length", 16 * 1024 * 1024),
            ("grpc.max_send_message_length", 16 * 1024 * 1024),
        ],
    )
    # add_PredictorServicer_to_server(PredictorServicer(model, label_names), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"gRPC server running on port {port}")
    server.wait_for_termination()
```

<div class="callout info">
<div class="callout-icon">ℹ️</div>
<div class="callout-body">
<strong>REST vs gRPC:</strong> Use REST for external-facing APIs and debugging simplicity. Use gRPC for internal microservice communication where you need lower latency, binary encoding (smaller payloads), and bidirectional streaming.
</div>
</div>

---

## Request Batching

Individual inference requests are inefficient on GPU. Dynamic batching collects concurrent requests into a batch, maximizing GPU utilization:

```python
import asyncio
import time
import torch
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PendingRequest:
    inputs: torch.Tensor
    future: asyncio.Future
    arrival_time: float = field(default_factory=time.monotonic)


class DynamicBatcher:
    """
    Collects individual inference requests and dispatches them as batches.

    Key parameters:
        max_batch_size: Maximum number of requests in a single forward pass.
        max_wait_ms: Maximum time to wait for a full batch before dispatching.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        max_batch_size: int = 32,
        max_wait_ms: float = 10.0,
        device: str = "cuda",
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_ms / 1000.0
        self.device = device

        self._queue: list[PendingRequest] = []
        self._lock = asyncio.Lock()
        self._batch_event = asyncio.Event()

    async def predict(self, inputs: list[float]) -> dict:
        """
        Submit a single request. Returns when the batch it belongs to
        has been processed.
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        tensor = torch.tensor([inputs], dtype=torch.float32)

        async with self._lock:
            self._queue.append(PendingRequest(inputs=tensor, future=future))
            # Signal the batch worker that new data arrived
            self._batch_event.set()

        return await future

    async def run_batch_worker(self) -> None:
        """Background coroutine — collect and dispatch batches continuously."""
        while True:
            # Wait for at least one request
            await self._batch_event.wait()
            self._batch_event.clear()

            # Give stragglers a chance to arrive (up to max_wait_ms)
            await asyncio.sleep(self.max_wait_s)

            async with self._lock:
                if not self._queue:
                    continue
                # Take up to max_batch_size requests
                batch = self._queue[:self.max_batch_size]
                self._queue = self._queue[self.max_batch_size:]

            await self._dispatch_batch(batch)

    async def _dispatch_batch(self, batch: list[PendingRequest]) -> None:
        """Run the model on a collected batch and resolve each future."""
        inputs = torch.cat([req.inputs for req in batch], dim=0).to(self.device)

        try:
            with torch.inference_mode():
                logits = self.model(inputs)
                probs = torch.softmax(logits, dim=-1)
                confs, indices = probs.max(dim=-1)

            for i, req in enumerate(batch):
                if not req.future.done():
                    req.future.set_result({
                        "label_idx": indices[i].item(),
                        "confidence": float(confs[i].item()),
                        "batch_size": len(batch),
                    })

        except Exception as exc:
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)
```

---

## Shadow Deployment

Shadow mode runs a new model version alongside production without serving its results to users. Outputs are logged and compared offline:

```python
import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ShadowResult:
    request_id: str
    prod_output: Any
    shadow_output: Any
    match: bool
    prod_latency_ms: float
    shadow_latency_ms: float


class ShadowRouter:
    """
    Routes every request to the production model.
    Asynchronously sends the same request to the shadow model.
    Shadow results are never returned to the user.
    """

    def __init__(
        self,
        prod_model,
        shadow_model,
        shadow_sample_rate: float = 1.0,
        result_callback=None,
    ):
        self.prod = prod_model
        self.shadow = shadow_model
        self.sample_rate = shadow_sample_rate
        self.result_callback = result_callback or self._default_log

    async def predict(self, inputs) -> Any:
        import time

        # Always run production model
        t0 = time.monotonic()
        prod_output = await self._run_model(self.prod, inputs)
        prod_latency = (time.monotonic() - t0) * 1000

        # Sample shadow traffic
        if random.random() < self.sample_rate:
            asyncio.create_task(
                self._run_shadow(inputs, prod_output, prod_latency)
            )

        return prod_output  # user only sees production output

    async def _run_shadow(self, inputs, prod_output, prod_latency_ms):
        import time
        import uuid

        t0 = time.monotonic()
        try:
            shadow_output = await self._run_model(self.shadow, inputs)
            shadow_latency = (time.monotonic() - t0) * 1000

            result = ShadowResult(
                request_id=str(uuid.uuid4())[:8],
                prod_output=prod_output,
                shadow_output=shadow_output,
                match=(prod_output == shadow_output),
                prod_latency_ms=prod_latency_ms,
                shadow_latency_ms=shadow_latency,
            )
            await self.result_callback(result)
        except Exception as e:
            log.warning(f"Shadow model failed (non-fatal): {e}")

    async def _run_model(self, model, inputs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, model, inputs)

    async def _default_log(self, result: ShadowResult):
        log.info(
            f"shadow_compare | req={result.request_id} "
            f"match={result.match} "
            f"prod_ms={result.prod_latency_ms:.1f} "
            f"shadow_ms={result.shadow_latency_ms:.1f}"
        )
```

---

## A/B Testing Pattern

Route a fraction of traffic to the new model and measure real user outcomes:

```python
import hashlib
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class ABConfig:
    model_a_version: str     # control
    model_b_version: str     # treatment
    traffic_to_b: float      # e.g. 0.10 = 10% to B
    experiment_id: str


class ABRouter:
    """
    Deterministic A/B router based on user/request ID hashing.
    The same user always gets the same model — avoids inconsistent UX.
    """

    def __init__(self, model_registry, ab_config: ABConfig):
        self.registry = model_registry
        self.config = ab_config

    def _get_variant(self, routing_key: str) -> str:
        """Hash the routing key to deterministically assign A or B."""
        hash_input = f"{self.config.experiment_id}:{routing_key}"
        h = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (h % 10000) / 10000.0  # [0, 1)
        return "B" if bucket < self.config.traffic_to_b else "A"

    async def predict(self, inputs, routing_key: str) -> dict:
        variant = self._get_variant(routing_key)
        version = (
            self.config.model_b_version if variant == "B"
            else self.config.model_a_version
        )

        model, label_names = self.registry.get(version)

        import torch
        tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=-1)
            conf, idx = probs.max(dim=-1)

        result = {
            "label": label_names[idx.item()],
            "confidence": float(conf.item()),
            "variant": variant,
            "model_version": version,
            "experiment_id": self.config.experiment_id,
        }

        # Log for downstream analysis
        log.info(
            f"ab_test | exp={self.config.experiment_id} "
            f"key={routing_key} variant={variant} "
            f"label={result['label']} conf={result['confidence']:.3f}"
        )

        return result
```

---

## Canary Release

A canary gradually shifts traffic to a new version, monitoring error rates and latency at each step:

<div class="diagram">
<div class="diagram-title">Canary Rollout Stages</div>
<div class="flow">
  <div class="flow-node blue wide">100% → v1<br/><small>Stable production</small></div>
  <div class="flow-arrow accent">↓ deploy v2, shift 5%</div>
  <div class="flow-node green wide">95% → v1 / 5% → v2<br/><small>Monitor error rate + latency</small></div>
  <div class="flow-arrow green">↓ metrics OK → shift 25%</div>
  <div class="flow-node green wide">75% → v1 / 25% → v2<br/><small>Wider canary</small></div>
  <div class="flow-arrow green">↓ metrics OK → shift 50%</div>
  <div class="flow-node green wide">50% → v1 / 50% → v2<br/><small>Half traffic</small></div>
  <div class="flow-arrow green">↓ metrics OK → full rollout</div>
  <div class="flow-node accent wide">100% → v2<br/><small>v1 retired</small></div>
  <div class="flow-arrow purple">↕ rollback if p99 > threshold</div>
  <div class="flow-node purple wide">100% → v1<br/><small>Instant rollback</small></div>
</div>
</div>

```python
import time
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class CanaryConfig:
    stable_version: str
    canary_version: str
    canary_weight: float = 0.05     # start at 5%
    max_weight: float = 1.0
    step_size: float = 0.05         # increase by 5% each step
    step_interval_seconds: int = 300  # wait 5 min between steps
    error_rate_threshold: float = 0.01  # rollback if >1% errors
    p99_latency_threshold_ms: float = 200.0


class CanaryController:
    def __init__(self, registry, config: CanaryConfig):
        self.registry = registry
        self.cfg = config
        self._error_counts: dict[str, int] = {"stable": 0, "canary": 0}
        self._request_counts: dict[str, int] = {"stable": 0, "canary": 0}
        self._last_step_time = time.monotonic()

    def should_route_to_canary(self) -> bool:
        import random
        return random.random() < self.cfg.canary_weight

    async def predict(self, inputs) -> dict:
        use_canary = self.should_route_to_canary()
        version = self.cfg.canary_version if use_canary else self.cfg.stable_version
        slot = "canary" if use_canary else "stable"

        t0 = time.monotonic()
        try:
            model, labels = self.registry.get(version)
            output = self._run_model(model, labels, inputs)
            self._request_counts[slot] += 1
            latency_ms = (time.monotonic() - t0) * 1000
            output["latency_ms"] = latency_ms
            return output
        except Exception as e:
            self._error_counts[slot] += 1
            self._request_counts[slot] += 1
            self._check_rollback()
            raise

    def _run_model(self, model, labels, inputs):
        import torch
        tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=-1)
            conf, idx = probs.max(dim=-1)
        return {"label": labels[idx.item()], "confidence": float(conf.item())}

    def _check_rollback(self):
        n = self._request_counts["canary"]
        if n < 100:
            return  # not enough data yet
        error_rate = self._error_counts["canary"] / n
        if error_rate > self.cfg.error_rate_threshold:
            log.error(
                f"CANARY ROLLBACK: error_rate={error_rate:.3f} "
                f"exceeds threshold={self.cfg.error_rate_threshold}"
            )
            self.cfg.canary_weight = 0.0

    def maybe_advance(self):
        """Call periodically to advance the canary weight."""
        elapsed = time.monotonic() - self._last_step_time
        if elapsed < self.cfg.step_interval_seconds:
            return

        n = self._request_counts["canary"]
        if n < 50:
            return  # too few samples

        error_rate = self._error_counts["canary"] / max(n, 1)
        if error_rate > self.cfg.error_rate_threshold:
            return  # don't advance if errors are elevated

        new_weight = min(
            self.cfg.canary_weight + self.cfg.step_size,
            self.cfg.max_weight,
        )
        log.info(
            f"Canary advancing: {self.cfg.canary_weight:.0%} → {new_weight:.0%}"
        )
        self.cfg.canary_weight = new_weight
        self._last_step_time = time.monotonic()
```

---

## Model Versioning in Serving

Explicit version routing in your API lets clients pin to a stable version during migrations:

```python
from fastapi import FastAPI, APIRouter

app = FastAPI()

# Version-specific routers
v1_router = APIRouter(prefix="/v1", tags=["v1"])
v2_router = APIRouter(prefix="/v2", tags=["v2"])


@v1_router.post("/predict")
async def predict_v1(request: PredictRequest):
    """Stable v1 API — guaranteed backward-compatible until EOL 2027-01."""
    request.model_version = "resnet50-v1.3"
    return await _run_inference(request)


@v2_router.post("/predict")
async def predict_v2(request: PredictRequest):
    """v2 API — new response format with top-5 predictions."""
    request.model_version = "vit-base-v2.1"
    response = await _run_inference(request)
    # v2 adds top-5 instead of top-1
    return enrich_with_topk(response, k=5)


app.include_router(v1_router)
app.include_router(v2_router)

# Alias: /predict always points to latest stable
app.include_router(v2_router, prefix="")
```

---

## Blue/Green Deployment

<div class="diagram">
<div class="diagram-title">Blue/Green Deployment Flow</div>
<div class="flow">
  <div class="flow-node blue wide">Load Balancer<br/><small>routes 100% traffic</small></div>
  <div class="flow-arrow accent">↓ currently points to</div>
  <div class="flow-h">
    <div class="flow-node blue extra-wide">🟦 Blue (Active)<br/>model v1 — serving 100%</div>
    <div class="flow-node green extra-wide">🟩 Green (Standby)<br/>model v2 — warm, not serving</div>
  </div>
  <div class="flow-arrow green">↓ green passes smoke tests</div>
  <div class="flow-node accent wide">DNS / LB flip: 100% → Green<br/><small>instant cutover, &lt;1s downtime</small></div>
  <div class="flow-h">
    <div class="flow-node green extra-wide">🟩 Green (Active)<br/>model v2 — serving 100%</div>
    <div class="flow-node blue extra-wide">🟦 Blue (Standby)<br/>model v1 — held for rollback</div>
  </div>
  <div class="flow-arrow purple">↑ rollback: flip LB back to Blue</div>
</div>
</div>

---

## TorchServe

TorchServe is PyTorch's official model serving framework. It handles model packaging, versioning, and dynamic batching natively:

```python
# custom_handler.py — TorchServe handler pattern
import torch
import json
import logging
from ts.torch_handler.base_handler import BaseHandler

log = logging.getLogger(__name__)


class ImageClassifierHandler(BaseHandler):
    """
    Custom TorchServe handler for image classification.
    Packaging: torch-model-archiver --model-name resnet50 \
        --version 1.0 \
        --model-file model.py \
        --serialized-file resnet50.pth \
        --handler custom_handler.py \
        --extra-files index_to_name.json
    """

    def __init__(self):
        super().__init__()
        self.label_map = None

    def initialize(self, context):
        """Called once when the worker starts."""
        super().initialize(context)
        # Load label map from packaged extra file
        label_file = context.manifest.get("model", {}).get(
            "labelFile", "index_to_name.json"
        )
        try:
            with open(label_file) as f:
                self.label_map = json.load(f)
        except FileNotFoundError:
            log.warning("Label map not found, using indices as labels")
            self.label_map = None

    def preprocess(self, data):
        """Convert raw request bytes to a tensor batch."""
        from torchvision import transforms
        from PIL import Image
        import io

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        tensors = []
        for row in data:
            image_bytes = row.get("data") or row.get("body")
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensors.append(transform(img))

        return torch.stack(tensors)

    def inference(self, data):
        """Run forward pass."""
        with torch.inference_mode():
            logits = self.model(data.to(self.device))
        return torch.softmax(logits, dim=-1)

    def postprocess(self, probs):
        """Convert probabilities to labeled response."""
        results = []
        confs, indices = probs.topk(5, dim=-1)
        for i in range(probs.size(0)):
            top5 = []
            for rank in range(5):
                idx = indices[i, rank].item()
                label = (
                    self.label_map.get(str(idx), str(idx))
                    if self.label_map else str(idx)
                )
                top5.append({
                    "label": label,
                    "confidence": float(confs[i, rank].item()),
                })
            results.append(top5)
        return results
```

---

## Triton Inference Server

NVIDIA Triton serves models from TensorRT, ONNX, PyTorch, and TensorFlow with a unified gRPC/HTTP interface and GPU-native dynamic batching:

```python
# Model repository layout:
# model_repository/
# └── resnet50/
#     ├── config.pbtxt
#     └── 1/
#         └── model.onnx

# config.pbtxt
TRITON_CONFIG = """
name: "resnet50"
platform: "onnxruntime_onnx"
max_batch_size: 64

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 5000
}

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0, 1 ]
  }
]
"""

# Python client
import tritonclient.http as httpclient
import numpy as np


def triton_predict(
    server_url: str,
    model_name: str,
    image_array: np.ndarray,
) -> np.ndarray:
    """
    Run inference via Triton HTTP client.
    image_array shape: (batch, 3, 224, 224), dtype float32
    """
    client = httpclient.InferenceServerClient(url=server_url)

    # Check server and model are ready
    assert client.is_server_live(), "Triton server not live"
    assert client.is_model_ready(model_name), f"{model_name} not ready"

    # Build input
    inputs = httpclient.InferInput("input", image_array.shape, "FP32")
    inputs.set_data_from_numpy(image_array)

    # Build output request
    outputs = httpclient.InferRequestedOutput("output")

    # Synchronous inference
    result = client.infer(model_name, inputs=[inputs], outputs=[outputs])

    return result.as_numpy("output")  # shape: (batch, 1000)
```

<div class="callout tip">
<div class="callout-icon">💡</div>
<div class="callout-body">
Export PyTorch models to ONNX for Triton: <code>torch.onnx.export(model, dummy_input, "model.onnx", opset_version=17, dynamic_axes={"input": {0: "batch"}})</code>. This unlocks TensorRT optimization and cross-framework serving under one Triton instance.
</div>
</div>

---

## Latency vs Throughput Trade-offs

<table class="compare-table">
<thead>
<tr>
  <th>Technique</th>
  <th>Latency Impact</th>
  <th>Throughput Impact</th>
  <th>Memory Impact</th>
  <th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
  <td><strong>Dynamic Batching</strong></td>
  <td>⬆️ +5–50ms wait</td>
  <td>⬆️ 5–20x higher</td>
  <td>➡️ Neutral</td>
  <td>Essential for GPU utilization</td>
</tr>
<tr>
  <td><strong>FP16 Inference</strong></td>
  <td>⬇️ ~2x faster</td>
  <td>⬆️ ~2x higher</td>
  <td>⬇️ 2x smaller</td>
  <td>Use <code>model.half()</code> or TensorRT FP16</td>
</tr>
<tr>
  <td><strong>INT8 Quantization</strong></td>
  <td>⬇️ 3–4x faster</td>
  <td>⬆️ 3–4x higher</td>
  <td>⬇️ 4x smaller</td>
  <td>Requires calibration, small accuracy hit</td>
</tr>
<tr>
  <td><strong>torch.compile()</strong></td>
  <td>⬇️ 10–30% faster</td>
  <td>⬆️ 10–30% higher</td>
  <td>⬆️ Slightly higher</td>
  <td>Warm-up cost; best with fixed shapes</td>
</tr>
<tr>
  <td><strong>Model Replicas</strong></td>
  <td>➡️ Neutral</td>
  <td>⬆️ Linear scaling</td>
  <td>⬆️ Linear higher</td>
  <td>Horizontal scale for CPU-bound workloads</td>
</tr>
<tr>
  <td><strong>Async Inference</strong></td>
  <td>⬇️ P99 lower</td>
  <td>⬆️ Higher concurrency</td>
  <td>➡️ Neutral</td>
  <td>Free when using FastAPI + async handlers</td>
</tr>
</tbody>
</table>

---

## Full Serving Architecture Diagram

<div class="diagram">
<div class="diagram-title">Production Serving Flow with Deployment Strategies</div>
<div class="flow">
  <div class="flow-node blue wide">Client<br/><small>REST / gRPC request</small></div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-node accent extra-wide">Load Balancer / API Gateway<br/><small>auth, rate limiting, routing</small></div>
  <div class="flow-arrow accent">↓ deployment strategy</div>
  <div class="flow-h">
    <div class="flow-node purple narrow">Shadow<br/><small>silent comparison</small></div>
    <div class="flow-node green narrow">Canary<br/><small>N% new model</small></div>
    <div class="flow-node blue narrow">A/B Test<br/><small>user-keyed split</small></div>
    <div class="flow-node teal narrow">Blue/Green<br/><small>instant cutover</small></div>
  </div>
  <div class="flow-arrow green">↓</div>
  <div class="flow-h">
    <div class="flow-node blue wide">Model Server v1<br/><small>FastAPI / TorchServe / Triton</small></div>
    <div class="flow-node green wide">Model Server v2<br/><small>FastAPI / TorchServe / Triton</small></div>
  </div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-node teal extra-wide">Monitoring<br/><small>latency · error rate · input drift · output drift</small></div>
  <div class="flow-arrow purple">↑ rollback signal</div>
  <div class="flow-node blue wide">Client Response<br/><small>prediction + metadata</small></div>
</div>
</div>

---

## Monitoring Pattern

A deployed model degrades silently without monitoring. Log latency, error rates, and input drift on every request:

```python
import time
import logging
import hashlib
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    latency_ms: float
    model_version: str
    success: bool
    input_shape: tuple
    timestamp: float = field(default_factory=time.time)


class ServingMonitor:
    """
    Tracks serving metrics and detects input distribution drift.
    Thread-safe for concurrent request handling.
    """

    def __init__(
        self,
        window_size: int = 1000,
        drift_threshold: float = 3.0,
        p99_alert_ms: float = 200.0,
        error_rate_alert: float = 0.05,
    ):
        self._window = deque(maxlen=window_size)
        self._lock = Lock()
        self.drift_threshold = drift_threshold
        self.p99_alert_ms = p99_alert_ms
        self.error_rate_alert = error_rate_alert

        # Reference distribution for drift detection (set from training data)
        self._ref_mean: np.ndarray | None = None
        self._ref_std: np.ndarray | None = None

    def set_reference_distribution(
        self, mean: np.ndarray, std: np.ndarray
    ) -> None:
        """Set the training data statistics for drift comparison."""
        self._ref_mean = mean
        self._ref_std = np.clip(std, 1e-8, None)

    def record(self, metrics: RequestMetrics) -> None:
        with self._lock:
            self._window.append(metrics)
        self._check_alerts(metrics)

    def record_input(self, inputs: np.ndarray) -> dict:
        """
        Compute per-feature z-scores against reference distribution.
        Returns drift report.
        """
        if self._ref_mean is None:
            return {"drift_detected": False, "reason": "no reference set"}

        current_mean = inputs.mean(axis=0)
        z_scores = np.abs((current_mean - self._ref_mean) / self._ref_std)
        max_z = float(z_scores.max())
        drifted_features = int((z_scores > self.drift_threshold).sum())

        drift_detected = max_z > self.drift_threshold

        if drift_detected:
            log.warning(
                f"INPUT DRIFT DETECTED: max_z={max_z:.2f} "
                f"drifted_features={drifted_features}/{len(z_scores)}"
            )

        return {
            "drift_detected": drift_detected,
            "max_z_score": max_z,
            "drifted_feature_count": drifted_features,
        }

    def _check_alerts(self, latest: RequestMetrics) -> None:
        with self._lock:
            recent = list(self._window)

        if len(recent) < 50:
            return  # not enough data

        latencies = [r.latency_ms for r in recent if r.success]
        error_rate = sum(1 for r in recent if not r.success) / len(recent)

        if latencies:
            p99 = np.percentile(latencies, 99)
            if p99 > self.p99_alert_ms:
                log.warning(
                    f"LATENCY ALERT: p99={p99:.1f}ms > "
                    f"threshold={self.p99_alert_ms}ms "
                    f"(window_size={len(recent)})"
                )

        if error_rate > self.error_rate_alert:
            log.error(
                f"ERROR RATE ALERT: rate={error_rate:.3f} > "
                f"threshold={self.error_rate_alert} "
                f"(window_size={len(recent)})"
            )

    def summary(self) -> dict:
        """Return current monitoring summary stats."""
        with self._lock:
            recent = list(self._window)

        if not recent:
            return {"status": "no data"}

        latencies = [r.latency_ms for r in recent if r.success]
        error_count = sum(1 for r in recent if not r.success)

        return {
            "window_size": len(recent),
            "error_rate": error_count / len(recent),
            "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
            "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
            "latency_p99_ms": float(np.percentile(latencies, 99)) if latencies else None,
            "latency_mean_ms": float(np.mean(latencies)) if latencies else None,
            "model_versions": list({r.model_version for r in recent}),
        }


# Integrate into FastAPI
monitor = ServingMonitor(p99_alert_ms=150.0, error_rate_alert=0.02)


@app.middleware("http")
async def monitoring_middleware(request, call_next):
    t0 = time.monotonic()
    success = True
    try:
        response = await call_next(request)
        return response
    except Exception:
        success = False
        raise
    finally:
        latency_ms = (time.monotonic() - t0) * 1000
        monitor.record(RequestMetrics(
            latency_ms=latency_ms,
            model_version="v1",
            success=success,
            input_shape=(0,),
        ))


@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint."""
    return monitor.summary()
```

---

## Summary

<div class="callout tip">
<div class="callout-icon">✅</div>
<div class="callout-body">
<strong>Model Serving Checklist:</strong>
<ul>
<li>Expose <code>/health</code> for liveness/readiness probes — Kubernetes requires it</li>
<li>Version your API (<code>/v1/predict</code>, <code>/v2/predict</code>) and model registry separately</li>
<li>Use shadow mode before any A/B test to validate the new model is numerically correct</li>
<li>Implement dynamic batching for GPU serving — single-request GPU calls are 10x less efficient</li>
<li>Monitor P99 latency and error rate with a rolling window — alert before users notice</li>
<li>Track input distribution drift — silent data shift is the most common cause of model degradation</li>
<li>Keep the previous model warm in blue/green or canary — instant rollback is priceless at 3am</li>
</ul>
</div>
</div>

<div class="diagram-grid cols-2">
  <div class="diagram-card blue">
    <div class="card-icon">🚀</div>
    <div class="card-title">Start Here</div>
    <div class="card-desc">FastAPI + dynamic batching + <code>/health</code> endpoint covers 80% of production use cases. Add TorchServe or Triton when you need multi-model management or TensorRT acceleration.</div>
  </div>
  <div class="diagram-card accent">
    <div class="card-icon">🛡️</div>
    <div class="card-title">Safety First</div>
    <div class="card-desc">Deploy with shadow mode → canary → blue/green. Never replace a production model in a single atomic swap. Every new model version is a hypothesis until traffic validates it.</div>
  </div>
</div>

---

*Last updated: May 2026*
