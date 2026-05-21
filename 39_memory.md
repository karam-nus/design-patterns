---
title: "Chapter 39 — Memory & Context Management"
---

[← Back to Table of Contents](./README.md)

# Chapter 39 — Memory & Context Management

> *"Memory is the treasury and guardian of all things." — Cicero (adapted for transformer architectures)*

<span class="badge agentic">Agentic</span> <span class="badge pytorch">PyTorch</span>

---

## Overview

LLMs are **stateless**: each forward pass sees only what fits in the context window. Memory systems extend this fundamental limitation by managing what information is stored, compressed, or retrieved. This chapter surveys every major memory strategy from raw KV-cache mechanics to long-horizon episodic stores.

---

## 39.1 Memory Taxonomy

<div class="diagram">
  <div class="diagram-title">Four Types of Agent Memory</div>
  <div class="diagram-grid cols-4">
    <div class="diagram-card blue">
      <div class="card-icon">🧠</div>
      <div class="card-title">Working Memory</div>
      <div class="card-desc">The active context window. Fast, limited (~128K tokens). Discarded after each conversation.</div>
    </div>
    <div class="diagram-card green">
      <div class="card-icon">📖</div>
      <div class="card-title">Episodic Memory</div>
      <div class="card-desc">Records of past interactions stored in a vector DB. Slow to retrieve but persistent across sessions.</div>
    </div>
    <div class="diagram-card purple">
      <div class="card-icon">📚</div>
      <div class="card-title">Semantic Memory</div>
      <div class="card-desc">Structured world knowledge: facts, entities, relationships. Retrieved by similarity or lookup.</div>
    </div>
    <div class="diagram-card orange">
      <div class="card-icon">⚙️</div>
      <div class="card-title">Procedural Memory</div>
      <div class="card-desc">Skills and tool usage patterns. Encoded in prompts, few-shot examples, or fine-tuned weights.</div>
    </div>
  </div>
</div>

---

## 39.2 Working Memory — Token Budget Management

The context window is precious. Explicit budget management prevents silent truncation.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

@dataclass
class TokenBudget:
    """
    Manages token allocation across context sections.

    Typical allocation for a 128K model:
      system prompt   ~  2 000 tokens
      retrieved docs  ~ 60 000 tokens
      conversation    ~ 30 000 tokens
      output reserve  ~ 36 000 tokens
    """

    total: int = 128_000
    system_reserve: int = 2_000
    output_reserve: int = 4_000
    _used: int = 0

    @property
    def available(self) -> int:
        return self.total - self.system_reserve - self.output_reserve - self._used

    def charge(self, tokens: int) -> bool:
        """Return True if budget allows; deduct tokens."""
        if tokens > self.available:
            return False
        self._used += tokens
        return True

    def reset(self) -> None:
        self._used = 0

    def summary(self) -> dict:
        return {
            "total": self.total,
            "used": self._used,
            "available": self.available,
            "reserved_system": self.system_reserve,
            "reserved_output": self.output_reserve,
        }


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Approximate token count using tiktoken."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        # Fallback: ~4 chars per token
        return len(text) // 4


def truncate_to_budget(texts: list[str], budget: TokenBudget, model: str = "gpt-4o") -> list[str]:
    """Return texts in order, stopping when budget is exhausted."""
    selected = []
    for text in texts:
        tokens = count_tokens(text, model)
        if budget.charge(tokens):
            selected.append(text)
        else:
            break
    return selected
```

---

## 39.3 KV-Cache as Memory

During autoregressive generation, the transformer reuses **key-value matrices** from previous tokens. This is the most important memory mechanism for inference efficiency.

<div class="diagram">
  <div class="diagram-title">KV-Cache Mechanics</div>
  <div class="flow">
    <div class="flow-node accent wide">Prefill Phase<br/><small>All prompt tokens processed once; KVs stored in cache</small></div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node blue wide">Decode Phase<br/><small>Each new token attends to ALL cached KVs — no recomputation</small></div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node purple wide">Cache grows with each token<br/><small>Memory = 2 × layers × heads × head_dim × seq_len × dtype_bytes</small></div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node orange wide">Eviction needed when<br/><small>cache exceeds GPU VRAM budget</small></div>
  </div>
</div>

**KV-cache size formula:**

```
bytes = 2 × num_layers × num_kv_heads × head_dim × seq_len × bytes_per_element
```

For LLaMA-3 70B (FP16): ~2.6 GB per 1K tokens in cache.

---

## 39.4 KV-Cache Eviction Strategies

<table class="compare-table">
  <thead>
    <tr>
      <th>Strategy</th>
      <th>Policy</th>
      <th>Complexity</th>
      <th>Quality Impact</th>
      <th>Best For</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>FIFO</strong></td>
      <td>Evict oldest tokens first</td>
      <td>O(1)</td>
      <td>Loses early context (system prompt)</td>
      <td>Simple streaming</td>
    </tr>
    <tr>
      <td><strong>LRU</strong></td>
      <td>Evict least-recently attended tokens</td>
      <td>O(log n)</td>
      <td>Better; keeps attended tokens</td>
      <td>General purpose</td>
    </tr>
    <tr>
      <td><strong>H2O</strong></td>
      <td>Keep heavy-hitter tokens (cumulative attention score)</td>
      <td>O(n)</td>
      <td>Near-lossless at 80% compression</td>
      <td>Long contexts</td>
    </tr>
    <tr>
      <td><strong>SnapKV</strong></td>
      <td>Cluster similar KVs, keep one representative</td>
      <td>O(n log n)</td>
      <td>Excellent</td>
      <td>Repeated content</td>
    </tr>
    <tr>
      <td><strong>StreamingLLM</strong></td>
      <td>Keep first N (sink tokens) + last M tokens</td>
      <td>O(1)</td>
      <td>Good for generation; loses middle</td>
      <td>Infinite streaming</td>
    </tr>
  </tbody>
</table>

---

## 39.5 Context Window Compression

When conversation history grows beyond the budget, summarise earlier turns to free space.

```python
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class Turn:
    role: str   # "user" | "assistant" | "system"
    content: str
    token_count: int = 0

@dataclass
class ConversationCompressor:
    """
    Manages a long conversation by compressing older turns into a
    rolling summary when the context budget is approached.

    Attributes:
        summarise_fn:  (list[Turn]) -> str — produces a summary of given turns
        token_counter: str -> int
        max_tokens:    hard cap for total history
        compress_to:   target tokens to achieve after compression
    """

    summarise_fn: Callable[[list[Turn]], str]
    token_counter: Callable[[str], int]
    max_tokens: int = 32_000
    compress_to: int = 16_000

    _turns: list[Turn] = field(default_factory=list)
    _summary: str = ""

    def add(self, role: str, content: str) -> None:
        tc = self.token_counter(content)
        self._turns.append(Turn(role=role, content=content, token_count=tc))
        if self._total_tokens() > self.max_tokens:
            self._compress()

    def _total_tokens(self) -> int:
        return sum(t.token_count for t in self._turns) + self.token_counter(self._summary)

    def _compress(self) -> None:
        """Summarise oldest turns until we're below compress_to."""
        to_compress = []
        while self._total_tokens() > self.compress_to and self._turns:
            to_compress.append(self._turns.pop(0))
        if to_compress:
            new_summary = self.summarise_fn(to_compress)
            if self._summary:
                self._summary = f"[Earlier summary]: {self._summary}\n\n[Recent summary]: {new_summary}"
            else:
                self._summary = new_summary

    def get_context(self) -> list[dict]:
        """Return context-window-ready message list."""
        messages = []
        if self._summary:
            messages.append({"role": "system", "content": f"[Conversation summary]: {self._summary}"})
        messages.extend({"role": t.role, "content": t.content} for t in self._turns)
        return messages
```

---

## 39.6 Sliding Window Memory

For long-running conversations, keep the last N turns verbatim plus a summary of earlier history.

```python
from collections import deque
from dataclasses import dataclass, field

@dataclass
class SlidingWindowMemory:
    """
    Keeps the most recent `window_size` turns in full,
    summarises everything older into a compact buffer.
    """

    window_size: int = 10
    summarise_fn: Callable[[list[dict]], str] = lambda turns: str(turns)
    _window: deque = field(default_factory=lambda: deque(maxlen=10), repr=False)
    _summary: str = ""

    def __post_init__(self):
        self._window = deque(maxlen=self.window_size)

    def add(self, role: str, content: str) -> None:
        if len(self._window) == self.window_size:
            evicted = self._window[0]
            new_summary = self.summarise_fn([evicted])
            self._summary = (self._summary + " " + new_summary).strip()
        self._window.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        result = []
        if self._summary:
            result.append({"role": "system", "content": f"Earlier context: {self._summary}"})
        result.extend(self._window)
        return result

    def clear(self) -> None:
        self._window.clear()
        self._summary = ""
```

---

## 39.7 Episodic Memory — Vector DB Store

Episodic memory persists interaction history across sessions, retrievable by semantic similarity.

```python
import uuid
import time
from dataclasses import dataclass, field
import numpy as np

@dataclass
class EpisodicMemory:
    """
    Stores and retrieves past interactions using a vector database.
    Each memory = (text, embedding, metadata).
    """

    embed_fn: Callable[[str], np.ndarray]
    top_k: int = 5
    _memories: list[dict] = field(default_factory=list)

    def store(self, text: str, metadata: dict | None = None) -> str:
        memory_id = uuid.uuid4().hex[:8]
        embedding = self.embed_fn(text)
        self._memories.append({
            "id": memory_id,
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {},
            "timestamp": time.time(),
        })
        return memory_id

    def retrieve(self, query: str, min_score: float = 0.0) -> list[dict]:
        if not self._memories:
            return []
        q_emb = self.embed_fn(query)
        # Cosine similarity
        results = []
        for mem in self._memories:
            sim = float(np.dot(q_emb, mem["embedding"]) /
                        (np.linalg.norm(q_emb) * np.linalg.norm(mem["embedding"]) + 1e-9))
            if sim >= min_score:
                results.append({**mem, "score": sim})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:self.top_k]

    def forget(self, memory_id: str) -> bool:
        before = len(self._memories)
        self._memories = [m for m in self._memories if m["id"] != memory_id]
        return len(self._memories) < before
```

---

## 39.8 Semantic Memory — Knowledge Base

Semantic memory stores factual knowledge about entities and concepts, retrieved by key lookup or similarity.

```python
from dataclasses import dataclass, field

@dataclass
class SemanticMemory:
    """
    Structured knowledge base: entity → facts.
    Supports exact lookup (by entity name) and fuzzy lookup (by embedding).
    """

    embed_fn: Callable[[str], np.ndarray]
    _store: dict[str, dict] = field(default_factory=dict)

    def upsert(self, entity: str, facts: dict) -> None:
        key = entity.lower().strip()
        if key in self._store:
            self._store[key]["facts"].update(facts)
        else:
            self._store[key] = {
                "entity": entity,
                "facts": facts,
                "embedding": self.embed_fn(entity),
            }

    def get(self, entity: str) -> dict | None:
        return self._store.get(entity.lower().strip())

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        q_emb = self.embed_fn(query)
        scored = []
        for entry in self._store.values():
            sim = float(np.dot(q_emb, entry["embedding"]) /
                        (np.linalg.norm(q_emb) * np.linalg.norm(entry["embedding"]) + 1e-9))
            scored.append((entry, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{"entity": e["entity"], "facts": e["facts"], "score": s}
                for e, s in scored[:top_k]]
```

---

## 39.9 Memory-Augmented Generation

Combine the context window (working memory) with retrieved memories to maximise both recency and long-term recall.

```python
@dataclass
class MemoryAugmentedAgent:
    """
    Agent that maintains working, episodic, and semantic memory
    and constructs an augmented prompt from all three.
    """

    llm: Callable[[list[dict]], str]
    episodic: EpisodicMemory
    semantic: SemanticMemory
    working: SlidingWindowMemory
    max_context_tokens: int = 8_000
    token_counter: Callable[[str], int] = lambda t: len(t) // 4

    def chat(self, user_message: str) -> str:
        # 1. Retrieve relevant memories
        episodic_hits = self.episodic.retrieve(user_message, min_score=0.6)
        semantic_hits = self.semantic.search(user_message, top_k=3)

        # 2. Build augmented system message
        memory_block = ""
        if episodic_hits:
            memory_block += "\n[Relevant past interactions]:\n"
            memory_block += "\n".join(f"- {h['text']}" for h in episodic_hits[:3])
        if semantic_hits:
            memory_block += "\n[Relevant knowledge]:\n"
            for hit in semantic_hits:
                facts_str = "; ".join(f"{k}={v}" for k, v in hit["facts"].items())
                memory_block += f"- {hit['entity']}: {facts_str}\n"

        # 3. Compose messages
        messages = []
        if memory_block:
            messages.append({"role": "system", "content": memory_block.strip()})
        messages.extend(self.working.get_messages())
        messages.append({"role": "user", "content": user_message})

        # 4. Generate
        response = self.llm(messages)

        # 5. Store to episodic memory
        self.episodic.store(f"User: {user_message}\nAssistant: {response}")
        self.working.add("user", user_message)
        self.working.add("assistant", response)

        return response
```

---

## 39.10 Long-Context Strategies

<div class="diagram">
  <div class="diagram-title">Strategies for Long-Horizon Knowledge</div>
  <div class="flow">
    <div class="flow-node accent wide">Need to handle information beyond context window?</div>
    <div class="flow-arrow">↓</div>
    <div class="flow-h">
      <div class="flow-node blue">RAG<br/><small>Retrieve relevant chunks dynamically</small></div>
      <div class="flow-node green">Long-Context LLM<br/><small>128K–1M token window</small></div>
      <div class="flow-node purple">Hierarchical Summary<br/><small>Multi-level compression</small></div>
    </div>
    <div class="flow-h">
      <div class="flow-node teal narrow">Best for: dynamic, searchable corpora</div>
      <div class="flow-node orange narrow">Best for: entire docs, codebases</div>
      <div class="flow-node pink narrow">Best for: long chats, meetings</div>
    </div>
  </div>
</div>

---

## 39.11 Full Memory Flow

<div class="diagram">
  <div class="diagram-title">Memory-Augmented LLM Agent Flow</div>
  <div class="flow">
    <div class="flow-node accent wide">User Query</div>
    <div class="flow-arrow">↓</div>
    <div class="flow-h">
      <div class="flow-node blue">Working Memory<br/><small>context window turns</small></div>
      <div class="flow-node green">Episodic Memory<br/><small>vector DB retrieval</small></div>
      <div class="flow-node purple">Semantic Memory<br/><small>knowledge base lookup</small></div>
    </div>
    <div class="flow-arrow accent">↓ merged into prompt ↓</div>
    <div class="flow-node teal wide">LLM Generation</div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node orange">Response</div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node pink wide">Update Episodic Memory<br/><small>store (query, response) pair</small></div>
  </div>
</div>

---

## 39.12 Comparison Table

<table class="compare-table">
  <thead>
    <tr>
      <th>Approach</th>
      <th>Capacity</th>
      <th>Retrieval Latency</th>
      <th>Up-to-date?</th>
      <th>Cost</th>
      <th>Best For</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>In-context</strong></td>
      <td>128K tokens</td>
      <td>Zero</td>
      <td>Current session only</td>
      <td>High (tokens)</td>
      <td>Short tasks</td>
    </tr>
    <tr>
      <td><strong>RAG</strong></td>
      <td>Millions of docs</td>
      <td>50–200 ms</td>
      <td>Yes (index updates)</td>
      <td>Medium</td>
      <td>Knowledge grounding</td>
    </tr>
    <tr>
      <td><strong>Fine-tuning</strong></td>
      <td>Baked into weights</td>
      <td>Zero</td>
      <td>No (requires retraining)</td>
      <td>Very high</td>
      <td>Style, format, domain</td>
    </tr>
    <tr>
      <td><strong>Agent Memory</strong></td>
      <td>Scales with DB</td>
      <td>100–500 ms</td>
      <td>Yes (persistent)</td>
      <td>Low–Medium</td>
      <td>Long-horizon agents</td>
    </tr>
  </tbody>
</table>

<div class="callout tip">
  <div class="callout-icon">💡</div>
  <div class="callout-body">
    Combine approaches: use fine-tuning for <em>style</em>, RAG for <em>facts</em>, and agent memory for <em>user-specific context</em>. Each layer has different update frequencies and costs.
  </div>
</div>

---

*Last updated: May 2026*
