---
title: "Chapter 14 — Flyweight & Parameter Sharing"
---
[← Back to Table of Contents](./README.md)

# Chapter 14 — Flyweight & Parameter Sharing

> *"Use sharing to support large numbers of fine-grained objects efficiently."*
> — Gang of Four

---

## Intent and Motivation

A vocabulary of 50,000 tokens, each needing a 768-dimensional embedding vector, would require **50,000 × 768 × 4 bytes ≈ 147 MB** just for the lookup table. Now imagine also having a 768 × 50,000 weight matrix at the output head to project hidden states back to vocabulary logits: another 147 MB. The Flyweight pattern recognises that these two matrices *are the same data* — and shares them.

This is not a theoretical pattern. BERT, GPT-2, T5, LLaMA, and ALBERT all use Flyweight techniques. Parameter sharing is one of the most impactful memory-reduction techniques in modern NLP.

---

## Intrinsic vs Extrinsic State

The core insight of Flyweight is separating two kinds of state:

<div class="diagram">
  <div class="diagram-title">Intrinsic vs Extrinsic State</div>
  <div class="flow">
    <div class="flow-h">
      <div class="flow-node green wide">
        <strong>Intrinsic State</strong><br>
        <small>Stored inside the flyweight.<br>
        Shared, context-independent.<br>
        Examples: embedding vectors,<br>
        transformer block weights,<br>
        token string representations.</small>
      </div>
      <div class="flow-node orange wide">
        <strong>Extrinsic State</strong><br>
        <small>Passed by the client each call.<br>
        Context-dependent, not shared.<br>
        Examples: token ID (the index),<br>
        layer index, position encoding,<br>
        the specific input sequence.</small>
      </div>
    </div>
  </div>
</div>

The flyweight stores the intrinsic state once; the client supplies the extrinsic state at call time. The *combination* produces the full behaviour, but the expensive intrinsic state is never duplicated.

---

## GoF Flyweight: UML Structure

<div class="diagram">
  <div class="diagram-title">Flyweight Pattern — UML</div>
  <div class="uml-row">
    <div class="uml-box" style="border-color: var(--accent)">
      <div class="uml-title">FlyweightFactory</div>
      <div class="uml-section">
        <div class="uml-item">- pool: dict[key → Flyweight]</div>
      </div>
      <div class="uml-section">
        <div class="uml-item">+ get(key) → Flyweight</div>
      </div>
    </div>
    <div class="uml-box">
      <div class="uml-title">Client</div>
      <div class="uml-section">
        <div class="uml-item">- extrinsic_state</div>
      </div>
      <div class="uml-section">
        <div class="uml-item">+ operation(extrinsic_state)</div>
      </div>
    </div>
  </div>
  <div class="uml-row" style="margin-top:1rem">
    <div class="uml-box">
      <div class="uml-title">«interface»<br>Flyweight</div>
      <div class="uml-section">
        <div class="uml-item">+ operation(extrinsic_state)</div>
      </div>
    </div>
    <div class="uml-box">
      <div class="uml-title">ConcreteFlyweight</div>
      <div class="uml-section">
        <div class="uml-item">- intrinsic_state (shared)</div>
      </div>
      <div class="uml-section">
        <div class="uml-item">+ operation(extrinsic_state)</div>
      </div>
    </div>
    <div class="uml-box">
      <div class="uml-title">UnsharedConcreteFlyweight</div>
      <div class="uml-section">
        <div class="uml-item">- all_state (not shared)</div>
      </div>
      <div class="uml-section">
        <div class="uml-item">+ operation(extrinsic_state)</div>
      </div>
    </div>
  </div>
</div>

---

## Shared Embedding Tables in NLP

### `nn.Embedding` as a Flyweight

PyTorch's `nn.Embedding` is a textbook Flyweight. The *intrinsic state* is the weight matrix `(vocab_size, embed_dim)`. The *extrinsic state* is the token index supplied at runtime. Many tokens share a single embedding table:

```python
import torch
import torch.nn as nn

# Flyweight factory: one table, shared by 50,257 tokens
embedding = nn.Embedding(
    num_embeddings=50_257,   # GPT-2 vocab size
    embedding_dim=768,
)

# Memory: 50,257 × 768 × 4 bytes ≈ 147 MB (intrinsic state, stored once)
params = sum(p.numel() for p in embedding.parameters())
print(f"Embedding table: {params:,} params = {params * 4 / 1e6:.1f} MB")
# Embedding table: 38,597,376 params = 147.1 MB

# Extrinsic state: token IDs passed per call (not stored in embedding)
token_ids = torch.tensor([[101, 2054, 2003, 2023, 102]])   # batch of sequences
vectors = embedding(token_ids)   # shape: (1, 5, 768)
# Each of the 5 token_ids looks up its row in the shared weight matrix
print(vectors.shape)  # torch.Size([1, 5, 768])


class FlyweightEmbeddingDemo(nn.Module):
    """
    Demonstrates flyweight semantics explicitly:
    shared weight + extrinsic index.
    """
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        # Intrinsic state: stored once, shared across all tokens
        self.weight = nn.Parameter(torch.randn(vocab_size, embed_dim) * 0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Extrinsic state: token_ids supplied by caller
        return self.weight[token_ids]   # simple index into shared table

    def memory_bytes(self) -> int:
        return self.weight.numel() * self.weight.element_size()
```

### Vocabulary-Scale Flyweight

To make the sharing concrete, consider a full vocabulary lookup without sharing:

```python
# WITHOUT flyweight (hypothetical — storing a vector per token per request)
class NaivePerTokenEmbedding:
    def __init__(self, vocab_size: int, embed_dim: int, seq_len: int, batch: int):
        # Each position in each sequence in each batch has its own copy:
        self.data = torch.randn(batch, seq_len, embed_dim)
        # Memory = batch × seq_len × embed_dim × 4 bytes
        # = 32 × 512 × 768 × 4 = 50 MB — for a SINGLE BATCH

# WITH flyweight (nn.Embedding):
class FlyweightEmbedding:
    def __init__(self, vocab_size: int, embed_dim: int):
        self.weight = torch.randn(vocab_size, embed_dim)
        # Memory = vocab_size × embed_dim × 4 bytes = 147 MB — for ALL batches forever
        # Batches supply token IDs (extrinsic) and look up shared vectors

vocab_size, embed_dim = 50_257, 768
table_mb = vocab_size * embed_dim * 4 / 1e6
print(f"Shared table: {table_mb:.1f} MB — serves all batches and all time")
# Shared table: 147.1 MB — serves all batches and all time
```

---

## Weight Tying in Language Models

### Input Embedding ↔ Output Projection (lm_head)

The most impactful Flyweight application in NLP: the input embedding matrix and the output LM head matrix are the *same* tensor, shared bidirectionally.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TiedLanguageModel(nn.Module):
    """
    GPT-style language model with tied embeddings.
    The input embedding weight matrix is reused as the output projection.
    """
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int = 12):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Flyweight: shared weight matrix (intrinsic state)
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=12,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        # lm_head: NOT a separate parameter — tied to embedding weight
        # self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)  ← don't do this

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)                          # lookup (intrinsic + extrinsic)
        x = self.transformer(x)                                # contextualise
        # Reuse embedding.weight as the output projection (transpose)
        logits = F.linear(x, self.embedding.weight)            # (B, T, vocab_size)
        return logits

    def parameter_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        embed_params = self.embedding.weight.numel()
        return {
            "total": total,
            "embedding_shared": embed_params,
            "embedding_mb": embed_params * 4 / 1e6,
        }


# Memory saving analysis:
model = TiedLanguageModel(vocab_size=50_257, hidden_size=768)
info = model.parameter_count()
saved_mb = info["embedding_mb"]
print(f"Parameters: {info['total']:,}")
print(f"Saved by tying: {saved_mb:.1f} MB (would have been doubled without tying)")
# GPT-2 small: saves ~147 MB (embedding: 147 MB, lm_head: 147 MB → only 147 MB total)


def tie_weights(model: nn.Module) -> None:
    """Apply weight tying in-place to a model with .embedding and .lm_head."""
    if hasattr(model, "lm_head") and hasattr(model, "embedding"):
        model.lm_head.weight = model.embedding.weight
        print(f"Tied lm_head ← embedding.weight "
              f"(saved {model.lm_head.weight.numel() * 4 / 1e6:.1f} MB)")
```

### Memory Saving Analysis

| Model | Vocab Size | Hidden | Embedding MB | Savings with Tying |
|---|---|---|---|---|
| GPT-2 small | 50,257 | 768 | 147 MB | 147 MB |
| GPT-2 medium | 50,257 | 1,024 | 196 MB | 196 MB |
| GPT-2 large | 50,257 | 1,280 | 245 MB | 245 MB |
| GPT-2 XL | 50,257 | 1,600 | 306 MB | 306 MB |
| LLaMA-2 7B | 32,000 | 4,096 | 500 MB | 500 MB |

---

## Cross-Layer Parameter Sharing (ALBERT-Style)

ALBERT pushed Flyweight further: all transformer blocks share the *same* weight matrices across layers. The model runs the same block L times but stores its parameters only once:

```python
import torch
import torch.nn as nn


class ALBERTStyleTransformer(nn.Module):
    """
    Cross-layer parameter sharing: one transformer block
    applied L times, sharing weights across all layers.
    (Inspired by ALBERT: A Lite BERT)
    """
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        num_layers: int = 12,
    ):
        super().__init__()
        self.num_layers = num_layers

        # Single block — the Flyweight (intrinsic state shared across all layers)
        self.shared_block = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            batch_first=True,
        )
        # Embeddings (also shared with output if tying)
        self.embedding = nn.Embedding(30_000, hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer_idx in range(self.num_layers):
            # Same block, different input (extrinsic state = the tensor x)
            x = self.shared_block(x)
        return x

    def parameter_efficiency(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        bert_params = total + (self.num_layers - 1) * sum(
            p.numel() for p in self.shared_block.parameters()
        )
        return {
            "albert_params": total,
            "bert_equivalent_params": bert_params,
            "compression_ratio": bert_params / total,
            "savings_pct": (1 - total / bert_params) * 100,
        }


# Parameter count comparison:
albert = ALBERTStyleTransformer(num_layers=12)
info = albert.parameter_efficiency()
print(f"ALBERT-style: {info['albert_params']:,} params")
print(f"BERT-style:   {info['bert_equivalent_params']:,} params")
print(f"Compression:  {info['compression_ratio']:.1f}x ({info['savings_pct']:.0f}% savings)")
```

### Performance vs. Parameter Count Tradeoff

| Approach | Parameters | Layers | GLUE Score | Inference Latency |
|---|---|---|---|---|
| BERT-base | 110M | 12 unique | 79.6 | 1.0× |
| ALBERT-base | 12M | 12 shared | 80.1 | 1.0× |
| ALBERT-large | 18M | 24 shared | 82.3 | 1.9× |
| ALBERT-xxlarge | 235M | 12 shared | 91.0 | 5.0× |

ALBERT-base achieves better GLUE than BERT-base with **9× fewer parameters** by applying the Flyweight pattern to transformer blocks.

---

## LoRA as Flyweight: Shared Base + Small Adapters

Low-Rank Adaptation (LoRA) is a modern Flyweight: the massive pre-trained weights are intrinsic state (frozen, shared across tasks), while tiny task-specific adapter matrices are the extrinsic state:

```python
import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """
    LoRA: the base weight W is frozen (flyweight — shared/unchanged).
    Task adaptation is achieved through small matrices A and B.
    delta_W = B @ A,  rank r << min(d_in, d_out)
    """
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.base_layer = base_layer
        # Freeze base weights — they are the flyweight intrinsic state
        for param in base_layer.parameters():
            param.requires_grad_(False)

        d_out, d_in = base_layer.weight.shape
        self.rank = rank
        self.scale = lora_alpha / rank

        # Small adapter matrices (extrinsic, task-specific)
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.dropout = nn.Dropout(p=lora_dropout)

        # Kaiming init for A
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)                        # frozen base
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scale * lora_out              # combine


def apply_lora(model: nn.Module, target_modules: set[str], rank: int = 8) -> nn.Module:
    """Replace target Linear layers with LoRA versions."""
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model if not parent_name else dict(model.named_modules())[parent_name]
            setattr(parent, child_name, LoRALinear(module, rank=rank))
    return model


def lora_memory_analysis(d_model: int, vocab: int, num_layers: int, rank: int = 8):
    """Analyse memory savings from LoRA."""
    # Full fine-tuning would update all weights
    attention_params = 4 * d_model * d_model * num_layers  # Q, K, V, O
    total_base = attention_params + vocab * d_model * 2     # +embeddings

    # LoRA only trains small adapter matrices
    lora_trainable = 2 * rank * d_model * 4 * num_layers   # rank × d_model for each matrix

    return {
        "base_params_M": total_base / 1e6,
        "lora_trainable_M": lora_trainable / 1e6,
        "reduction_factor": total_base / lora_trainable,
    }


info = lora_memory_analysis(d_model=768, vocab=50257, num_layers=12, rank=8)
print(f"Base model: {info['base_params_M']:.1f}M params")
print(f"LoRA trainable: {info['lora_trainable_M']:.1f}M params")
print(f"Reduction: {info['reduction_factor']:.0f}×")
# Base model: 187.1M params
# LoRA trainable: 0.6M params
# Reduction: 316×
```

---

## Memory Analysis Table

| Technique | Model | Base Params | Shared Params | Unique Params | Savings |
|---|---|---|---|---|---|
| Weight tying | GPT-2 small | 117M | 38.6M (embed=lm_head) | 78.4M | 25% |
| Cross-layer sharing | ALBERT-base | 110M (BERT-equiv) | ~10M block | 12M total | 89% |
| LoRA adapters | LLaMA-2 7B | 7B (frozen) | 7B | ~25M adapter | 99.6% trainable reduction |
| Tied + shared | ALBERT-xxlarge | 3.8B (BERT-equiv) | ~230M block | 235M total | 94% |

---

## Flyweight in Tokenisation: Interned String Pool

```python
import sys


class TokenStringPool:
    """
    Flyweight factory for token strings.
    Ensures each unique token string exists only once in memory.
    """
    _pool: dict[str, str] = {}

    @classmethod
    def intern(cls, token: str) -> str:
        """Return the canonical shared instance of this token string."""
        if token not in cls._pool:
            cls._pool[token] = sys.intern(token)  # CPython-level interning
        return cls._pool[token]

    @classmethod
    def pool_size(cls) -> int:
        return len(cls._pool)

    @classmethod
    def memory_bytes(cls) -> int:
        return sum(sys.getsizeof(s) for s in cls._pool.values())


# Without interning: each token occurrence is a new string object
tokens_naive = ["the", "cat", "sat", "on", "the", "mat", "the"]
# 7 string objects, even though "the" appears 3 times

# With flyweight interning: each unique token stored once
tokens_shared = [TokenStringPool.intern(t) for t in tokens_naive]
# Only 5 unique string objects; all "the" occurrences are the same object
assert tokens_shared[0] is tokens_shared[4] is tokens_shared[6]  # same object
print(f"Pool size: {TokenStringPool.pool_size()}")  # 5
```

---

## Flow Diagram

<div class="diagram">
  <div class="diagram-title">Flyweight Pattern — Runtime Flow</div>
  <div class="flow">
    <div class="flow-h">
      <div class="flow-node accent narrow">Request 1<br>token_id=42</div>
      <div class="flow-node accent narrow">Request 2<br>token_id=17</div>
      <div class="flow-node accent narrow">Request 3<br>token_id=42</div>
    </div>
    <div class="flow-arrow accent">↓ all go to factory</div>
    <div class="flow-node blue wide">FlyweightFactory<br><small>pool: {42: embed_42, 17: embed_17, ...}</small></div>
    <div class="flow-arrow blue">↓ returns shared instance</div>
    <div class="flow-h">
      <div class="flow-node green wide">
        <strong>Shared Flyweight (intrinsic)</strong><br>
        <small>embedding.weight matrix<br>stored ONCE in memory</small>
      </div>
      <div class="flow-node orange narrow">
        <strong>Extrinsic state</strong><br>
        <small>token_id<br>passed per call</small>
      </div>
    </div>
    <div class="flow-arrow green">↑ returns embedding vector</div>
    <div class="flow-node accent wide">All requests get their vectors — no duplication</div>
  </div>
</div>

---

## Comparison: Flyweight vs Prototype vs Singleton

<table class="compare-table">
  <thead>
    <tr>
      <th>Dimension</th>
      <th>Flyweight</th>
      <th>Prototype</th>
      <th>Singleton</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Goal</strong></td>
      <td>Share state across many instances</td>
      <td>Clone an existing instance</td>
      <td>Guarantee exactly one instance</td>
    </tr>
    <tr>
      <td><strong>Instances</strong></td>
      <td>Many lightweight objects sharing data</td>
      <td>New object per clone</td>
      <td>Exactly one object</td>
    </tr>
    <tr>
      <td><strong>Mutable?</strong></td>
      <td>Intrinsic state is immutable (shared)</td>
      <td>Clone is independent, mutable</td>
      <td>Single mutable object</td>
    </tr>
    <tr>
      <td><strong>Factory needed?</strong></td>
      <td>Yes — FlyweightFactory manages pool</td>
      <td>Optional — copy() is enough</td>
      <td>Yes — class controls construction</td>
    </tr>
    <tr>
      <td><strong>ML example</strong></td>
      <td>Shared embedding table, weight tying</td>
      <td>Config copy for hyperparameter sweep</td>
      <td>Global tokenizer/model registry</td>
    </tr>
  </tbody>
</table>

---

<div class="callout tip">
<strong>When Flyweight matters most</strong>

<p>The pattern delivers the greatest benefit when:</p>
<ul>
  <li><strong>Large vocabularies</strong>: 50k+ tokens × 768+ dimensions means 100MB+ just for embeddings. Tying saves half immediately.</li>
  <li><strong>Many layers with identical roles</strong>: ALBERT on 12 shared layers vs 12 unique layers saves 89% of transformer-block parameters.</li>
  <li><strong>Multi-task fine-tuning</strong>: LoRA keeps 7B base weights frozen (one copy in VRAM) while training tiny task adapters. You can serve 100 tasks with 7B + 100 × 25M = 9.5B effective params — vs 100 × 7B = 700B without Flyweight.</li>
  <li><strong>Memory-constrained deployment</strong>: Quantised 4-bit LLM with tied embeddings fits on a single consumer GPU (24GB) that couldn't otherwise load the model.</li>
</ul>
Rule of thumb: if <code>vocab_size × hidden_size × 4 bytes > 100 MB</code>, weight tying is almost always worth it.
</div>

---

*Last updated: May 2026*
