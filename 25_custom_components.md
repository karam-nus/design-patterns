---
title: "Chapter 25 — Custom Training Components"
---

[← Back to Table of Contents](./README.md)

# Chapter 25 — Custom Training Components

> *"PyTorch's true power lies not in what it provides out of the box, but in how cleanly it lets you replace any piece."*

## Extension Points Overview

<div class="diagram">
<div class="diagram-title">PyTorch Extension Points</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card orange">
    <div class="card-icon">📉</div>
    <div class="card-title">Loss Functions</div>
    <div class="card-desc">Subclass nn.Module, implement forward(pred, target)</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">⚡</div>
    <div class="card-title">Optimizers</div>
    <div class="card-desc">Subclass torch.optim.Optimizer, implement step()</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">📅</div>
    <div class="card-title">LR Schedulers</div>
    <div class="card-desc">Subclass _LRScheduler, implement get_lr()</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🎲</div>
    <div class="card-title">Samplers</div>
    <div class="card-desc">Subclass Sampler, implement __iter__ and __len__</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-icon">📦</div>
    <div class="card-title">Collate Functions</div>
    <div class="card-desc">Any callable: List[Sample] → Batch tensor dict</div>
  </div>
  <div class="diagram-card teal">
    <div class="card-icon">🔄</div>
    <div class="card-title">Transforms</div>
    <div class="card-desc">Any callable or nn.Module: sample → sample</div>
  </div>
</div>
</div>

## Custom Loss Functions

### FocalLoss

Focal loss addresses class imbalance by down-weighting easy examples.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance.
    
    Lin et al. 2017: https://arxiv.org/abs/1708.02002
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], targets: [B]
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss  # "none"
```

### DiceLoss

```python
class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""

    def __init__(self, smooth: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Flatten spatial dims
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1).float()

        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        loss = 1.0 - dice

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
```

### NT-Xent (Contrastive) Loss

```python
class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy loss for SimCLR."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        # z1, z2: [B, D] — two views of same batch
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        z = F.normalize(z, dim=1)

        sim = torch.mm(z, z.t()) / self.temperature  # [2B, 2B]

        # Mask out self-similarity
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, float("-inf"))

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)]).to(z.device)
        return F.cross_entropy(sim, labels)
```

## Custom Optimizers

```python
class Lion(torch.optim.Optimizer):
    """Lion optimizer — Evolved Sign Momentum.
    
    Chen et al. 2023: https://arxiv.org/abs/2302.06675
    Uses sign of gradient (like SignSGD) with momentum tracking.
    """

    def __init__(self, params, lr: float = 1e-4, betas=(0.9, 0.99),
                 weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # Initialize momentum buffer
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                # Update: p = p - lr * (sign(beta1*m + (1-beta1)*g) + wd*p)
                update = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1)
                p.add_(update.sign_(), alpha=-lr)
                if wd != 0:
                    p.add_(p, alpha=-lr * wd)

                # Update momentum with beta2
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss
```

## Custom LR Schedulers

```python
from torch.optim.lr_scheduler import _LRScheduler
import math


class CosineWarmupScheduler(_LRScheduler):
    """Linear warmup + cosine decay scheduler."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr_ratio: float = 0.1, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            scale = step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        return [base_lr * scale for base_lr in self.base_lrs]
```

## Custom Samplers

```python
from torch.utils.data import Sampler
import random


class ClassBalancedSampler(Sampler):
    """Samples equal number of examples per class each epoch."""

    def __init__(self, labels: list, samples_per_class: int):
        self.labels = labels
        self.samples_per_class = samples_per_class

        # Group indices by class
        from collections import defaultdict
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)
        self.classes = list(self.class_indices.keys())

    def __iter__(self):
        indices = []
        for cls in self.classes:
            cls_idx = self.class_indices[cls]
            if len(cls_idx) >= self.samples_per_class:
                sampled = random.sample(cls_idx, self.samples_per_class)
            else:
                sampled = random.choices(cls_idx, k=self.samples_per_class)
            indices.extend(sampled)
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.classes) * self.samples_per_class
```

## Custom Collate Functions

```python
from torch.nn.utils.rnn import pad_sequence


def variable_length_collate(batch):
    """Collate variable-length sequences with padding."""
    # batch: list of dicts with 'input_ids', 'attention_mask', 'labels'
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    labels    = [item["label"] for item in batch]

    # Pad to max length in batch
    padded_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = (padded_ids != 0).long()

    return {
        "input_ids":      padded_ids,
        "attention_mask": attention_mask,
        "labels":         torch.tensor(labels),
    }
```

## Extension Point Reference

| Component | Base Class | Key Method | Common Use Case |
|-----------|-----------|------------|-----------------|
| Loss | `nn.Module` | `forward(pred, target)` | Custom objective functions |
| Optimizer | `torch.optim.Optimizer` | `step()` | Novel update rules |
| LR Scheduler | `_LRScheduler` | `get_lr()` | Custom warmup/decay |
| Sampler | `Sampler` | `__iter__`, `__len__` | Class balancing, curriculum |
| Collate fn | `Callable` | `__call__(batch)` | Variable-length padding |
| Transform | `nn.Module` or `Callable` | `forward(x)` or `__call__(x)` | Custom augmentation |

<div class="callout tip">
<span class="callout-icon">✅</span>
<div class="callout-body">Always validate custom losses with <code>torch.autograd.gradcheck(loss_fn, inputs)</code> to verify gradient correctness before using in training.</div>
</div>

---

*Last updated: May 2026*
