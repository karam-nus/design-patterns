---
title: "Chapter 27 — Distributed & Parallel Patterns"
---

[← Back to Table of Contents](./README.md)

# Chapter 27 — Distributed & Parallel Patterns

> *"A single GPU is a ceiling. Distribution is the staircase."*

<span class="badge mlops">MLOps</span> <span class="badge pytorch">PyTorch</span>

---

## Why Distribution?

Modern foundation models have billions of parameters. A single A100 (80 GB) cannot hold GPT-3's 175 B parameters, let alone train them. Even when a model *fits* on one device, training speed is bounded by memory bandwidth and compute throughput of that device.

Distribution solves three distinct problems:

<div class="diagram">
  <div class="diagram-title">Why We Distribute</div>
  <div class="diagram-grid cols-3">
    <div class="diagram-card blue">
      <div class="card-icon">🧠</div>
      <div class="card-title">Memory Capacity</div>
      <div class="card-desc">Model parameters + gradients + optimizer states exceed a single GPU's VRAM</div>
    </div>
    <div class="diagram-card teal">
      <div class="card-icon">⚡</div>
      <div class="card-title">Compute Throughput</div>
      <div class="card-desc">More GPUs = more FLOPS = faster training at the same batch size</div>
    </div>
    <div class="diagram-card purple">
      <div class="card-icon">📦</div>
      <div class="card-title">Effective Batch Size</div>
      <div class="card-desc">Larger effective batches stabilize training and improve generalization</div>
    </div>
  </div>
</div>

---

## The Four Parallelism Dimensions

<div class="diagram">
  <div class="diagram-title">Four Axes of Parallelism</div>
  <div class="diagram-grid cols-4">
    <div class="diagram-card blue">
      <div class="card-icon">🗂️</div>
      <div class="card-title">Data Parallel</div>
      <div class="card-desc">Each GPU holds the full model. Data is split across GPUs. Gradients are all-reduced.</div>
    </div>
    <div class="diagram-card teal">
      <div class="card-icon">🔗</div>
      <div class="card-title">Pipeline Parallel</div>
      <div class="card-desc">Model layers split across GPUs in a pipeline. Micro-batches flow through stages.</div>
    </div>
    <div class="diagram-card purple">
      <div class="card-icon">✂️</div>
      <div class="card-title">Tensor Parallel</div>
      <div class="card-desc">Individual weight matrices split across GPUs. Column/row partitioning of linear layers.</div>
    </div>
    <div class="diagram-card orange">
      <div class="card-icon">🧩</div>
      <div class="card-title">Expert Parallel</div>
      <div class="card-desc">MoE experts distributed across GPUs. Only activated experts incur communication.</div>
    </div>
  </div>
</div>

---

## Data Parallel (DDP)

`DistributedDataParallel` is the most common parallelism strategy. Every rank holds a **full copy** of the model. The training batch is split; each rank computes its local gradient. An all-reduce synchronizes gradients across ranks before the optimizer step.

<div class="diagram">
  <div class="diagram-title">DDP All-Reduce Gradient Sync</div>
  <div class="flow">
    <div class="flow-h">
      <div class="flow-node blue">GPU 0<br><small>batch shard 0</small></div>
      <div class="flow-node blue">GPU 1<br><small>batch shard 1</small></div>
      <div class="flow-node blue">GPU 2<br><small>batch shard 2</small></div>
      <div class="flow-node blue">GPU 3<br><small>batch shard 3</small></div>
    </div>
    <div class="flow-arrow accent">↓ forward + backward (local gradients)</div>
    <div class="flow-node teal extra-wide">Ring All-Reduce — average gradients across all ranks</div>
    <div class="flow-arrow accent">↓ synchronized gradients</div>
    <div class="flow-h">
      <div class="flow-node green">optimizer.step()</div>
      <div class="flow-node green">optimizer.step()</div>
      <div class="flow-node green">optimizer.step()</div>
      <div class="flow-node green">optimizer.step()</div>
    </div>
    <div class="flow-arrow accent">↓ identical weights on all ranks</div>
  </div>
</div>

```python
# launch with: torchrun --nproc_per_node=4 train_ddp.py

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler


class MLP(nn.Module):
    def __init__(self, in_dim: int = 784, hidden: int = 512, out_dim: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def setup_ddp(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def build_dataloader(rank: int, world_size: int, batch_size: int = 64) -> DataLoader:
    torch.manual_seed(0)
    X = torch.randn(4096, 784)
    y = torch.randint(0, 10, (4096,))
    dataset = TensorDataset(X, y)
    # DistributedSampler ensures each rank sees a non-overlapping data shard
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)


def train_ddp(rank: int, world_size: int, num_epochs: int = 5) -> None:
    setup_ddp(rank, world_size)

    model = MLP().to(rank)
    # find_unused_parameters=False is faster when all params are used
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loader = build_dataloader(rank, world_size)

    for epoch in range(num_epochs):
        loader.sampler.set_epoch(epoch)   # re-shuffle per epoch
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(rank), yb.to(rank)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()               # gradients all-reduced automatically
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch} complete — loss={loss.item():.4f}")

    cleanup_ddp()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train_ddp, args=(world_size,), nprocs=world_size)
```

<div class="callout tip">
  <span class="callout-icon">💡</span>
  <div class="callout-body">Always call <code>sampler.set_epoch(epoch)</code> before iterating the DataLoader. Without it, every epoch sees the same ordering, which can hurt generalization.</div>
</div>

---

## FSDP — Fully Sharded Data Parallel

DDP replicates the full model on every rank. For large models this wastes memory. **FSDP** shards parameters, gradients, *and* optimizer states across ranks — each rank stores `1/N` of everything.

<div class="diagram">
  <div class="diagram-title">FSDP Sharding Diagram</div>
  <div class="flow">
    <div class="flow-node teal extra-wide">Full Model: 10 B params — 40 GB fp32</div>
    <div class="flow-arrow accent">↓ FSDP shards across 8 GPUs</div>
    <div class="flow-h">
      <div class="flow-node blue narrow">GPU 0<br>1.25 B params<br>5 GB</div>
      <div class="flow-node blue narrow">GPU 1<br>1.25 B params<br>5 GB</div>
      <div class="flow-node blue narrow">GPU 2<br>1.25 B params<br>5 GB</div>
      <div class="flow-node blue narrow">GPU 3<br>1.25 B params<br>5 GB</div>
    </div>
    <div class="flow-arrow purple">↓ all-gather before forward; reduce-scatter after backward</div>
    <div class="flow-node green extra-wide">Each rank runs forward with temporarily gathered params (freed immediately)</div>
  </div>
</div>

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import (
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
import functools


# Transformer block to be individually wrapped by FSDP
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ff(x))


class SmallTransformer(nn.Module):
    def __init__(self, num_layers: int = 6, d_model: int = 512) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(d_model) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, 0])          # CLS token


def wrap_with_fsdp(rank: int) -> FSDP:
    model = SmallTransformer().to(rank)

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )

    # Wrap each TransformerBlock independently — critical for memory efficiency
    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    return FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,      # ZeRO-3 equivalent
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,    # overlap comms
        device_id=rank,
    )
```

---

## ZeRO Stage Comparison

Microsoft DeepSpeed's ZeRO (Zero Redundancy Optimizer) progressively shards more state across ranks:

<table class="compare-table">
  <thead>
    <tr>
      <th>ZeRO Stage</th>
      <th>What's Sharded</th>
      <th>Memory Saving (vs DDP)</th>
      <th>Communication Overhead</th>
      <th>PyTorch Equivalent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Stage 1</strong></td>
      <td>Optimizer states only</td>
      <td>~4× (optimizer states)</td>
      <td>Low</td>
      <td>N/A (manual or DeepSpeed)</td>
    </tr>
    <tr>
      <td><strong>Stage 2</strong></td>
      <td>Optimizer states + gradients</td>
      <td>~8× (gradients + optimizer)</td>
      <td>Moderate</td>
      <td><code>FSDP(ShardingStrategy.SHARD_GRAD_OP)</code></td>
    </tr>
    <tr>
      <td><strong>Stage 3</strong></td>
      <td>Optimizer states + gradients + parameters</td>
      <td>~64× on 64 GPUs</td>
      <td>Higher (all-gather on fwd)</td>
      <td><code>FSDP(ShardingStrategy.FULL_SHARD)</code></td>
    </tr>
  </tbody>
</table>

---

## Pipeline Parallelism

Pipeline parallelism partitions the model *vertically* — layers are split across GPUs. GPipe overlaps computation across micro-batches to hide pipeline bubbles.

<div class="diagram">
  <div class="diagram-title">GPipe Micro-Batch Pipeline</div>
  <div class="flow">
    <div class="flow-h">
      <div class="flow-node blue narrow">GPU 0<br>Layers 0–3</div>
      <div class="flow-node teal narrow">GPU 1<br>Layers 4–7</div>
      <div class="flow-node purple narrow">GPU 2<br>Layers 8–11</div>
      <div class="flow-node orange narrow">GPU 3<br>Layers 12–15</div>
    </div>
    <div class="flow-arrow accent">↓ micro-batches fill the pipeline</div>
    <div class="flow-node green extra-wide">μ-batch 1 → GPU0 → GPU1 → GPU2 → GPU3 → loss</div>
    <div class="flow-node green extra-wide">μ-batch 2 → GPU0 (while GPU1 runs μ-batch 1) → ...</div>
    <div class="flow-arrow accent">↓ gradients flow backward through stages</div>
  </div>
</div>

```python
import torch
import torch.nn as nn
from torch.distributed.pipeline.sync import Pipe

# Build a sequential model with explicit device placement per stage
stage0 = nn.Sequential(
    nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 512)
).to("cuda:0")

stage1 = nn.Sequential(
    nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128)
).to("cuda:1")

stage2 = nn.Sequential(
    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)
).to("cuda:2")

# Pipe requires an nn.Sequential of device-placed stages
model_pipe = nn.Sequential(stage0, stage1, stage2)

# chunks = number of micro-batches; more chunks → less bubble, more memory
pipe_model = Pipe(model_pipe, chunks=8)

# Forward pass — input must be on first stage device
x = torch.randn(256, 784, device="cuda:0")
output = pipe_model(x).local_value()    # output lives on last stage device
```

<div class="callout warn">
  <span class="callout-icon">⚠️</span>
  <div class="callout-body">Pipeline parallelism introduces a <em>bubble</em> — idle time at the start and end of each micro-batch batch. The bubble fraction is <code>(p-1)/m</code> where <code>p</code> is pipeline depth and <code>m</code> is number of micro-batches. Use large <code>m</code> to amortize.</div>
</div>

---

## Tensor Parallelism

Tensor parallelism splits individual weight matrices across GPUs. A linear layer `Y = XW` can be column-partitioned (`W = [W_0 | W_1]`) or row-partitioned. Megatron-LM uses both for attention.

<div class="diagram">
  <div class="diagram-title">Column + Row Linear Partition (Megatron Style)</div>
  <div class="flow">
    <div class="flow-node blue extra-wide">Input X — broadcast to all tensor-parallel ranks</div>
    <div class="flow-arrow accent">↓</div>
    <div class="flow-h">
      <div class="flow-node teal wide">GPU 0<br>W_col[:, :d/2]<br>column partition</div>
      <div class="flow-node teal wide">GPU 1<br>W_col[:, d/2:]<br>column partition</div>
    </div>
    <div class="flow-arrow accent">↓ local GeLU</div>
    <div class="flow-h">
      <div class="flow-node purple wide">GPU 0<br>W_row[:d/2, :]<br>row partition</div>
      <div class="flow-node purple wide">GPU 1<br>W_row[d/2:, :]<br>row partition</div>
    </div>
    <div class="flow-arrow accent">↓ all-reduce</div>
    <div class="flow-node green extra-wide">Full output Y — no inter-GPU communication inside the block</div>
  </div>
</div>

```python
import torch
import torch.nn as nn
import torch.distributed as dist


class ColumnParallelLinear(nn.Module):
    """Column-split linear: W[:, local_start:local_end]."""

    def __init__(self, in_features: int, out_features: int, rank: int, world_size: int) -> None:
        super().__init__()
        assert out_features % world_size == 0
        self.local_out = out_features // world_size
        self.rank = rank
        self.world_size = world_size
        self.weight = nn.Parameter(torch.empty(self.local_out, in_features))
        self.bias = nn.Parameter(torch.zeros(self.local_out))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Row-split linear: W[local_start:local_end, :] — reduces after forward."""

    def __init__(self, in_features: int, out_features: int, rank: int, world_size: int) -> None:
        super().__init__()
        assert in_features % world_size == 0
        self.local_in = in_features // world_size
        self.weight = nn.Parameter(torch.empty(out_features, self.local_in))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.linear(x, self.weight)
        # All-reduce sums partial results across tensor-parallel ranks
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out + self.bias
```

---

## Device Placement with Accelerate

For models that span multiple GPUs but don't need full sharding, Hugging Face Accelerate's `device_map="auto"` handles placement automatically.

```python
# pip install accelerate
from accelerate import Accelerator, init_empty_weights, infer_auto_device_map
from accelerate import load_checkpoint_and_dispatch
import torch

# Pattern 1: Accelerate-managed training loop
accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=4)

model = MLP(784, 512, 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loader = build_dataloader(0, 1)     # single-process loader

model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

for xb, yb in loader:
    with accelerator.accumulate(model):
        loss = nn.CrossEntropyLoss()(model(xb), yb)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

# Pattern 2: Auto device-map for inference (large models)
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",          # spread across available GPUs (and CPU if needed)
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,     # build model on meta device first
)
print(model.hf_device_map)     # shows which layer → which device
```

---

## Mixed Precision

Mixed precision trains in fp16/bf16 but accumulates gradients in fp32, dramatically reducing memory and increasing throughput on Tensor Core hardware.

<div class="diagram">
  <div class="diagram-title">Mixed Precision Data Flow</div>
  <div class="flow">
    <div class="flow-node blue wide">fp32 master weights (optimizer)</div>
    <div class="flow-arrow accent">↓ cast to fp16 for forward</div>
    <div class="flow-node teal wide">fp16 forward pass — 2× faster on Tensor Cores</div>
    <div class="flow-arrow accent">↓</div>
    <div class="flow-node purple wide">fp16 loss → GradScaler multiplies loss</div>
    <div class="flow-arrow accent">↓</div>
    <div class="flow-node orange wide">fp16 backward — scaled gradients avoid underflow</div>
    <div class="flow-arrow accent">↓ GradScaler unscales + checks for inf/nan</div>
    <div class="flow-node green wide">fp32 optimizer step — update master weights</div>
  </div>
</div>

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

model = SmallTransformer().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = GradScaler()                  # manages loss scaling for fp16

for epoch in range(num_epochs):
    for xb, yb in loader:
        xb, yb = xb.cuda(), yb.cuda()
        optimizer.zero_grad(set_to_none=True)

        # autocast selects fp16 for matmuls, keeps fp32 for reductions
        with autocast(dtype=torch.float16):
            logits = model(xb)
            loss = nn.CrossEntropyLoss()(logits, yb)

        scaler.scale(loss).backward()          # scale before backward
        scaler.unscale_(optimizer)             # unscale before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)                 # skip step if inf/nan
        scaler.update()                        # adjust scale factor


# BFloat16 — preferred on Ampere+ (no GradScaler needed)
with autocast(dtype=torch.bfloat16):
    logits = model(xb)
    loss = nn.CrossEntropyLoss()(logits, yb)

loss.backward()
optimizer.step()
```

<div class="callout info">
  <span class="callout-icon">ℹ️</span>
  <div class="callout-body"><strong>bf16 vs fp16</strong>: bf16 has the same dynamic range as fp32 (8 exponent bits) but lower precision (7 mantissa bits). It rarely requires loss scaling. Prefer bf16 on Ampere (A100) or newer hardware.</div>
</div>

---

## Collective Operations

<table class="compare-table">
  <thead>
    <tr>
      <th>Operation</th>
      <th>Direction</th>
      <th>What It Does</th>
      <th>Typical Use</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>all_reduce</code></td>
      <td>Many → Many</td>
      <td>Reduce (sum/mean) across all ranks; every rank gets result</td>
      <td>DDP gradient averaging</td>
    </tr>
    <tr>
      <td><code>all_gather</code></td>
      <td>Many → Many</td>
      <td>Gather tensors from all ranks; every rank gets concatenation</td>
      <td>FSDP forward pass (gather shards)</td>
    </tr>
    <tr>
      <td><code>broadcast</code></td>
      <td>One → Many</td>
      <td>Send tensor from src rank to all ranks</td>
      <td>Model init sync, hyperparams</td>
    </tr>
    <tr>
      <td><code>scatter</code></td>
      <td>One → Many</td>
      <td>Distribute chunks from src rank to each rank</td>
      <td>Dataset sharding from rank 0</td>
    </tr>
    <tr>
      <td><code>reduce_scatter</code></td>
      <td>Many → Many</td>
      <td>Reduce then scatter chunks; each rank gets reduced shard</td>
      <td>FSDP backward pass (gradient sharding)</td>
    </tr>
    <tr>
      <td><code>reduce</code></td>
      <td>Many → One</td>
      <td>Reduce across all ranks; only dst rank gets result</td>
      <td>Metric aggregation on rank 0</td>
    </tr>
  </tbody>
</table>

```python
import torch
import torch.distributed as dist

# Example: aggregate validation loss across all ranks
def gather_metric(local_value: float, device: int) -> float:
    tensor = torch.tensor(local_value, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / dist.get_world_size()).item()


# Example: all-gather for distributed evaluation (e.g. concatenate predictions)
def all_gather_tensors(local_tensor: torch.Tensor) -> torch.Tensor:
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, local_tensor)
    return torch.cat(gathered, dim=0)
```

---

## Gradient Accumulation

When the per-GPU batch size is constrained by memory, gradient accumulation achieves a larger *effective* batch size by accumulating gradients over multiple micro-steps before calling `optimizer.step()`.

```python
MICRO_BATCH = 16       # fits in GPU memory
EFFECTIVE_BATCH = 256  # desired effective batch size
ACCUMULATION_STEPS = EFFECTIVE_BATCH // MICRO_BATCH  # = 16

model = SmallTransformer().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = GradScaler()

optimizer.zero_grad(set_to_none=True)
for step, (xb, yb) in enumerate(loader):
    xb, yb = xb.cuda(), yb.cuda()

    with autocast(dtype=torch.float16):
        loss = nn.CrossEntropyLoss()(model(xb), yb)
        # Scale loss to keep gradients comparable to non-accumulated baseline
        loss = loss / ACCUMULATION_STEPS

    scaler.scale(loss).backward()

    if (step + 1) % ACCUMULATION_STEPS == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

<div class="callout tip">
  <span class="callout-icon">💡</span>
  <div class="callout-body">In DDP, gradient accumulation interacts with the all-reduce: call <code>model.no_sync()</code> context manager on all accumulation steps except the last to prevent redundant all-reduce communications.</div>
</div>

```python
# DDP-aware gradient accumulation
for step, (xb, yb) in enumerate(loader):
    xb, yb = xb.cuda(rank), yb.cuda(rank)
    is_last_step = (step + 1) % ACCUMULATION_STEPS == 0

    # Suppress all-reduce on intermediate accumulation steps
    ctx = model.no_sync() if not is_last_step else contextlib.nullcontext()
    with ctx:
        with autocast(dtype=torch.float16):
            loss = nn.CrossEntropyLoss()(model(xb), yb) / ACCUMULATION_STEPS
        scaler.scale(loss).backward()

    if is_last_step:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

---

## Combining Strategies: 3D Parallelism

Large-scale training often combines all three axis types. Here is the topology for a 512-GPU training run:

<div class="diagram">
  <div class="diagram-title">3D Parallelism Topology</div>
  <div class="diagram-grid cols-3">
    <div class="diagram-card blue">
      <div class="card-icon">🗂️</div>
      <div class="card-title">Data Parallel</div>
      <div class="card-desc">64 data-parallel replicas — each holds 1 pipeline</div>
    </div>
    <div class="diagram-card teal">
      <div class="card-icon">🔗</div>
      <div class="card-title">Pipeline Parallel</div>
      <div class="card-desc">8 pipeline stages — each holds 1 tensor-parallel group</div>
    </div>
    <div class="diagram-card purple">
      <div class="card-icon">✂️</div>
      <div class="card-title">Tensor Parallel</div>
      <div class="card-desc">8-way tensor split per layer — 64 × 8 × 8 = 512 GPUs total</div>
    </div>
  </div>
</div>

```python
# Simplified Accelerate + FSDP + gradient accumulation combo
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=8,
    # For FSDP, pass fsdp_plugin:
    # fsdp_plugin=FullyShardedDataParallelPlugin(...)
)

model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

for step, (xb, yb) in enumerate(train_loader):
    with accelerator.accumulate(model):   # handles no_sync() automatically
        logits = model(xb)
        loss = nn.CrossEntropyLoss()(logits, yb)
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```

---

## All-Reduce Visualization

<div class="diagram">
  <div class="diagram-title">All-Reduce Across 4 GPUs</div>
  <div class="flow">
    <div class="flow-h">
      <div class="flow-node blue narrow">GPU 0<br>g₀</div>
      <div class="flow-node blue narrow">GPU 1<br>g₁</div>
      <div class="flow-node blue narrow">GPU 2<br>g₂</div>
      <div class="flow-node blue narrow">GPU 3<br>g₃</div>
    </div>
    <div class="flow-arrow accent">↓ ring reduce-scatter (each rank sends/receives one chunk)</div>
    <div class="flow-h">
      <div class="flow-node teal narrow">sum_0</div>
      <div class="flow-node teal narrow">sum_1</div>
      <div class="flow-node teal narrow">sum_2</div>
      <div class="flow-node teal narrow">sum_3</div>
    </div>
    <div class="flow-arrow accent">↓ ring all-gather (broadcast reduced chunks)</div>
    <div class="flow-h">
      <div class="flow-node green narrow">g₀+g₁+g₂+g₃</div>
      <div class="flow-node green narrow">g₀+g₁+g₂+g₃</div>
      <div class="flow-node green narrow">g₀+g₁+g₂+g₃</div>
      <div class="flow-node green narrow">g₀+g₁+g₂+g₃</div>
    </div>
  </div>
</div>

---

## Summary

<div class="diagram">
  <div class="diagram-title">Choose Your Parallelism Strategy</div>
  <div class="timeline">
    <div class="timeline-item">
      <div class="timeline-year">Step 1</div>
      <div class="timeline-title">Single GPU — baseline</div>
      <div class="timeline-desc">Profile memory and throughput. Use mixed precision and gradient accumulation first.</div>
    </div>
    <div class="timeline-item">
      <div class="timeline-year">Step 2</div>
      <div class="timeline-title">DDP — 2–8 GPUs</div>
      <div class="timeline-desc">Model fits on one GPU. Scale out with data parallelism. Minimal code changes.</div>
    </div>
    <div class="timeline-item">
      <div class="timeline-year">Step 3</div>
      <div class="timeline-title">FSDP — 8–128 GPUs</div>
      <div class="timeline-desc">Model too large for one GPU. Shard parameters + optimizer states. Use auto_wrap_policy.</div>
    </div>
    <div class="timeline-item">
      <div class="timeline-year">Step 4</div>
      <div class="timeline-title">3D Parallel — 128+ GPUs</div>
      <div class="timeline-desc">Foundation model scale. Combine data + pipeline + tensor parallelism via Megatron or DeepSpeed.</div>
    </div>
  </div>
</div>

*Last updated: May 2026*
