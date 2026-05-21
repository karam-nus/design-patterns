---
title: "Chapter 23 — nn.Module Composition Patterns"
---

[← Back to Table of Contents](./README.md)

# Chapter 23 — nn.Module Composition Patterns

> *"A neural network is a program. `nn.Module` is its function abstraction."*

<span class="badge pytorch">PyTorch</span> <span class="badge structural">Structural</span>

---

## 23.1 Why Module Composition Is Central

`nn.Module` is PyTorch's fundamental unit of computation and state. Every layer, every sub-network, every full model is an `nn.Module`. Composition — nesting modules inside modules — is how complex architectures are built from simple, reusable parts.

The composition patterns in this chapter solve recurring structural problems:

- **How** to arrange layers sequentially
- **How** to hold dynamic collections of sub-modules
- **How** to express skip connections, branching, and conditional routing
- **How** to share weights across different parts of a network

Understanding these patterns makes you fluent in reading and writing PyTorch model code at any scale.

---

## 23.2 `nn.Sequential` — Ordered Composition

`nn.Sequential` chains modules so that the output of one becomes the input of the next. It is the simplest composition pattern.

```python
import torch
import torch.nn as nn
from typing import Optional

# Basic sequential stack
backbone = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((1, 1)),
)

x = torch.randn(4, 3, 224, 224)
print(backbone(x).shape)  # torch.Size([4, 128, 1, 1])
```

### Named Sequential with OrderedDict

```python
from collections import OrderedDict

named_block = nn.Sequential(OrderedDict([
    ("conv1",  nn.Conv2d(64, 64, 3, padding=1)),
    ("bn1",    nn.BatchNorm2d(64)),
    ("relu1",  nn.ReLU(inplace=True)),
    ("conv2",  nn.Conv2d(64, 64, 3, padding=1)),
    ("bn2",    nn.BatchNorm2d(64)),
]))

# Access by name
print(named_block.conv1)
```

### Limitations of Sequential

- No skip connections — every layer must accept the previous layer's output
- No branching — single path only
- No conditional logic

For any of these, you need to write a custom `nn.Module`.

---

## 23.3 `nn.ModuleList` — Indexed Collection

`nn.ModuleList` holds an ordered list of modules that PyTorch tracks for parameter registration. Unlike a plain Python list, parameters inside a `ModuleList` appear in `model.parameters()`.

```python
class DynamicMLP(nn.Module):
    """MLP with a variable number of hidden layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1])
            for i in range(len(dims) - 1)
        ])
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:   # no activation on final layer
                x = self.dropout(self.activation(x))
        return x


mlp = DynamicMLP(784, [512, 256, 128], 10)
print(mlp(torch.randn(32, 784)).shape)  # torch.Size([32, 10])
```

### Dynamic Architecture with ModuleList

```python
class StochasticDepthNetwork(nn.Module):
    """Network where layers can be randomly dropped during training."""

    def __init__(self, num_layers: int, dim: int, drop_prob: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_layers)
        ])
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.training and torch.rand(1).item() < self.drop_prob:
                continue   # stochastic depth — skip layer
            x = torch.relu(layer(x)) + x
        return x
```

---

## 23.4 `nn.ModuleDict` — Named Collection

`nn.ModuleDict` maps string keys to modules. Ideal when sub-modules are selected by name at runtime.

```python
class MultiModalEncoder(nn.Module):
    """Encode different modalities using a named dict of encoders."""

    def __init__(self) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict({
            "image":  nn.Sequential(nn.Conv2d(3, 512, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten()),
            "text":   nn.EmbeddingBag(50_000, 512, mode="mean"),
            "audio":  nn.Sequential(nn.Conv1d(1, 512, 400, stride=160), nn.AdaptiveAvgPool1d(1), nn.Flatten()),
        })
        self.fusion = nn.Linear(512, 256)

    def forward(self, modality: str, data: torch.Tensor) -> torch.Tensor:
        embedding = self.encoders[modality](data)
        return self.fusion(embedding)


# Runtime modality selection
encoder = MultiModalEncoder()
img_embed = encoder("image", torch.randn(4, 3, 224, 224))
print(img_embed.shape)  # torch.Size([4, 256])
```

---

## 23.5 Skip Connections — ResNet Style

```python
class ResidualBlock(nn.Module):
    """Pre-activation residual block: BN → ReLU → Conv → BN → ReLU → Conv + skip."""

    def __init__(self, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )
        self.shortcut = (
            nn.Conv2d(channels, channels, 1, stride=stride, bias=False)
            if stride != 1 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.shortcut(x)


class ResNet(nn.Module):
    def __init__(self, blocks_per_stage: list[int] = [2, 2, 2, 2]) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stages = nn.ModuleList([
            nn.Sequential(*[ResidualBlock(64) for _ in range(n)])
            for n in blocks_per_stage
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(64, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.head(self.pool(x).flatten(1))
```

---

## 23.6 Multi-Head Outputs

```python
from typing import NamedTuple

class DetectionOutput(NamedTuple):
    classification: torch.Tensor    # [B, num_classes]
    bbox_regression: torch.Tensor   # [B, 4]
    objectness: torch.Tensor        # [B, 1]


class DetectionHead(nn.Module):
    """Multiple prediction heads from a shared feature backbone."""

    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.cls_head   = nn.Linear(512, num_classes)
        self.bbox_head  = nn.Linear(512, 4)
        self.obj_head   = nn.Linear(512, 1)

    def forward(self, features: torch.Tensor) -> DetectionOutput:
        shared = self.shared(features)
        return DetectionOutput(
            classification  = self.cls_head(shared),
            bbox_regression = self.bbox_head(shared),
            objectness      = torch.sigmoid(self.obj_head(shared)),
        )


head = DetectionHead(2048, 80)
features = torch.randn(8, 2048)
out = head(features)
print(out.classification.shape)   # [8, 80]
print(out.bbox_regression.shape)  # [8, 4]
```

---

## 23.7 Branching Architectures — Two-Tower

```python
class TwoTowerModel(nn.Module):
    """Dual-encoder for retrieval: query tower + document tower."""

    def __init__(self, vocab_size: int, embed_dim: int, out_dim: int) -> None:
        super().__init__()
        tower = lambda: nn.Sequential(
            nn.Embedding(vocab_size, embed_dim, padding_idx=0),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(embed_dim, nhead=8, batch_first=True),
                num_layers=4,
            ),
        )
        self.query_tower    = tower()
        self.document_tower = tower()
        self.query_proj    = nn.Linear(embed_dim, out_dim)
        self.document_proj = nn.Linear(embed_dim, out_dim)

    def encode_query(self, tokens: torch.Tensor) -> torch.Tensor:
        emb = self.query_tower[0](tokens)
        h = self.query_tower[1](emb)
        return nn.functional.normalize(self.query_proj(h[:, 0]), dim=-1)

    def encode_document(self, tokens: torch.Tensor) -> torch.Tensor:
        emb = self.document_tower[0](tokens)
        h = self.document_tower[1](emb)
        return nn.functional.normalize(self.document_proj(h[:, 0]), dim=-1)

    def forward(self, query_tokens: torch.Tensor,
                doc_tokens: torch.Tensor) -> torch.Tensor:
        q = self.encode_query(query_tokens)
        d = self.encode_document(doc_tokens)
        return (q * d).sum(dim=-1)   # cosine similarity scores
```

---

## 23.8 Shared Submodules — Weight Tying

```python
class TiedAutoEncoder(nn.Module):
    """Encoder and decoder share the same weight matrix (transposed)."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder_weight = nn.Parameter(torch.randn(latent_dim, input_dim) * 0.01)
        self.encoder_bias   = nn.Parameter(torch.zeros(latent_dim))
        self.decoder_bias   = nn.Parameter(torch.zeros(input_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.encoder_weight.T + self.encoder_bias)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.encoder_weight + self.decoder_bias   # transposed!

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# Standard weight tying (e.g., language model input/output embeddings)
class TiedLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True),
            num_layers=6,
        )
        # Output projection reuses the embedding weights — no extra parameters
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.transformer(self.embedding(tokens))
        # Tied projection: h @ E^T
        return h @ self.embedding.weight.T + self.output_bias
```

---

## 23.9 Dynamic Depth — Variable Layers

```python
class GrowableNetwork(nn.Module):
    """Network that can grow new layers progressively during training."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        self._add_layer()   # start with one layer

    def _add_layer(self) -> None:
        self.layers.append(nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.GELU(),
        ))

    def grow(self) -> None:
        """Add a new layer (call after N training steps)."""
        new_layer = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.GELU(),
        )
        # Initialise to near-identity to preserve learned representations
        nn.init.eye_(new_layer[0].weight)
        nn.init.zeros_(new_layer[0].bias)
        self.layers.append(new_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
```

---

## 23.10 Conditional Composition

```python
class MixtureOfExperts(nn.Module):
    """Route each token to top-k experts via a learned gating network."""

    def __init__(self, dim: int, num_experts: int, top_k: int = 2) -> None:
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        x_flat = x.view(-1, D)                        # [B*T, D]
        logits = self.gate(x_flat)                    # [B*T, E]
        weights, indices = logits.topk(self.top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)       # [B*T, K]

        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = indices[:, k]                 # [B*T]
            for e_id, expert in enumerate(self.experts):
                mask = (expert_idx == e_id)
                if mask.any():
                    out[mask] += weights[mask, k:k+1] * expert(x_flat[mask])

        return out.view(B, T, D)
```

---

## 23.11 Forward Design Principles

```python
from typing import Optional
import torch
from torch import Tensor

class WellDesignedModule(nn.Module):
    """Demonstrates forward() best practices."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.proj  = nn.Linear(dim, dim)
        self.drop  = nn.Dropout(0.1)

    def forward(
        self,
        x: Tensor,                           # type-annotated input
        mask: Optional[Tensor] = None,       # optional mask
    ) -> Tensor:                             # type-annotated output
        # 1. No state mutation (don't set self.last_output = ...)
        # 2. Use normed input, apply transform, residual
        normed = self.norm(x)
        projected = self.proj(normed)
        projected = self.drop(projected)
        if mask is not None:
            projected = projected.masked_fill(mask.unsqueeze(-1), 0.0)
        return x + projected               # residual — return, don't store
```

---

## 23.12 Module Composition for Transformers

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.qkv   = nn.Linear(d_model, 3 * d_model)
        self.out   = nn.Linear(d_model, d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.nhead, self.d_head)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)   # [B, H, T, D/H]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = self.d_head ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = self.drop(torch.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, d_model: int, nhead: int, d_ff: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadAttention(d_model, nhead, dropout)
        self.ff    = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Full transformer — N stacked blocks."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            TransformerBlock(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.token_emb.weight

    def forward(self, tokens: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.drop(self.token_emb(tokens) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.norm(x))


# Test
model = Transformer(vocab_size=50_257, d_model=512, nhead=8, num_layers=6)
tokens = torch.randint(0, 50_257, (2, 128))
logits = model(tokens)
print(logits.shape)  # [2, 128, 50257]
```

---

## 23.13 Pattern Overview

<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">📋</div>
    <div class="card-title">Sequential</div>
    <div class="card-desc">Ordered chain, no branching. Use for simple feed-forward stacks, preprocessing pipelines.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">📂</div>
    <div class="card-title">ModuleList</div>
    <div class="card-desc">Indexed collection. Use for variable-depth networks, transformer stacks, residual blocks.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🗂️</div>
    <div class="card-title">ModuleDict</div>
    <div class="card-desc">Named collection. Use for multi-modal encoders, multi-task heads, runtime routing.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">➕</div>
    <div class="card-title">Skip Connections</div>
    <div class="card-desc">Additive identity path. Essential for deep networks — ResNet, DenseNet, UNet.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">🧩</div>
    <div class="card-title">Shared Weights</div>
    <div class="card-desc">Same module in multiple forward positions. Weight tying in LMs, siamese networks.</div>
  </div>
  <div class="diagram-card teal">
    <div class="card-icon">🔀</div>
    <div class="card-title">Conditional</div>
    <div class="card-desc">Route through different sub-modules based on input. Mixture of Experts, conditional compute.</div>
  </div>
</div>

---

## Module Tree Diagram

<div class="diagram">
  <div class="diagram-title">Transformer Module Tree</div>
  <div class="flow">
    <div class="flow-node accent extra-wide">Transformer</div>
  </div>
  <div class="flow">
    <div class="flow-node blue">Embedding<br/><small>token</small></div>
    <div class="flow-node blue">Embedding<br/><small>position</small></div>
    <div class="flow-node green wide">ModuleList<br/><small>blocks[0..N]</small></div>
    <div class="flow-node purple">LayerNorm</div>
    <div class="flow-node orange">Linear<br/><small>head</small></div>
  </div>
  <div class="flow">
    <div class="flow-node teal wide">TransformerBlock</div>
    <div class="flow-arrow">×N</div>
  </div>
  <div class="flow">
    <div class="flow-node pink">LayerNorm ×2</div>
    <div class="flow-node cyan wide">MultiHeadAttention</div>
    <div class="flow-node yellow wide">FeedForward</div>
  </div>
</div>

*Last updated: May 2026*
