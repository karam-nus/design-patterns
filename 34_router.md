---
title: "Chapter 34 — Router & Dispatcher Pattern"
---

[← Back to Table of Contents](./README.md)

# Chapter 34 — Router & Dispatcher Pattern

> *"The right model for the right query — routing is the difference between paying for a sledgehammer and reaching for a scalpel."*

Not every query deserves a GPT-4-class response. A simple yes/no question doesn't need 100B parameters. A code generation request shouldn't go to a text-summarization model. The Router pattern sits in front of your model fleet and directs each request to the most appropriate handler — balancing cost, latency, accuracy, and capability.

<div class="callout info">
<span class="callout-icon">ℹ️</span>
<div class="callout-body">
A router is a <strong>meta-decision layer</strong>: it doesn't answer questions directly but decides <em>who should answer them</em>. Well-designed routing can cut inference costs by 60–80% with negligible quality loss on most traffic.
</div>
</div>

---

## Use Cases for Routing

<div class="diagram">
<div class="diagram-grid cols-3">
  <div class="diagram-card blue">
    <div class="card-icon">🗂️</div>
    <div class="card-title">Domain Routing</div>
    <div class="card-desc">Legal queries → legal model. Medical → clinical LLM. Code → code model.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">💰</div>
    <div class="card-title">Cost-Aware Routing</div>
    <div class="card-desc">Simple queries → cheap model. Complex queries → expensive model only if needed.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">🎯</div>
    <div class="card-title">Intent Routing</div>
    <div class="card-desc">Classify intent (question / command / chitchat) and route to the specialist handler.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">🔬</div>
    <div class="card-title">A/B Routing</div>
    <div class="card-desc">Split traffic between model versions for online evaluation of quality.</div>
  </div>
  <div class="diagram-card teal">
    <div class="card-icon">🧩</div>
    <div class="card-title">Modality Routing</div>
    <div class="card-desc">Detect images, audio, or code in the payload and send to the right pipeline.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-icon">⚡</div>
    <div class="card-title">Latency-Aware Routing</div>
    <div class="card-desc">Interactive queries → low-latency model. Batch jobs → high-quality async model.</div>
  </div>
</div>
</div>

---

## Router Architecture Overview

<div class="diagram">
<div class="diagram-title">Router Dispatch — Full Picture</div>
<div class="flow">
  <div class="flow-node blue wide">User Query</div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-node teal wide">Router<br/><small>classify → score → select</small></div>
</div>
<div class="flow">
  <div class="flow-arrow accent">↙</div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-arrow accent">↘</div>
</div>
<div class="flow">
  <div class="flow-node green narrow">GPT-4o<br/><small>complex reasoning</small></div>
  <div class="flow-node blue narrow">Claude<br/><small>long context</small></div>
  <div class="flow-node purple narrow">Local LLM<br/><small>fast / private</small></div>
  <div class="flow-node orange narrow">RAG Pipeline<br/><small>knowledge queries</small></div>
  <div class="flow-node teal narrow">Code Model<br/><small>programming tasks</small></div>
</div>
<div class="flow">
  <div class="flow-arrow accent">↓ unified response</div>
</div>
<div class="flow">
  <div class="flow-node blue wide">Response to User</div>
</div>
</div>

---

## Pattern 1 — Simple Intent Classifier Router

Classify the query into a known intent category and dispatch to the corresponding handler.

```python
import re
import logging
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class RouteRule:
    """A single routing rule: pattern → handler name."""
    pattern: re.Pattern
    handler: str
    priority: int = 0


class IntentClassifierRouter:
    """
    Rule-based router: matches query against regex patterns in priority order,
    dispatches to registered handlers.

    Args:
        default_handler: fallback handler name when no rule matches
    """

    def __init__(self, default_handler: str = "general"):
        self._rules: list[RouteRule] = []
        self._handlers: dict[str, Callable] = {}
        self.default_handler = default_handler

    def register_handler(self, name: str, func: Callable):
        self._handlers[name] = func

    def add_rule(self, pattern: str, handler: str, priority: int = 0):
        self._rules.append(RouteRule(
            pattern=re.compile(pattern, re.IGNORECASE),
            handler=handler,
            priority=priority,
        ))
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def route(self, query: str) -> dict:
        handler_name = self._classify(query)
        handler = self._handlers.get(handler_name)
        if handler is None:
            logger.warning("No handler registered for '%s', using default", handler_name)
            handler = self._handlers[self.default_handler]
        logger.info("Routing '%s...' → %s", query[:40], handler_name)
        result = handler(query)
        return {"response": result, "routed_to": handler_name}

    def _classify(self, query: str) -> str:
        for rule in self._rules:
            if rule.pattern.search(query):
                return rule.handler
        return self.default_handler


# --- Setup ---

router = IntentClassifierRouter(default_handler="general_llm")

# Register handlers
router.register_handler("code_model",    lambda q: f"[Code model response to: {q[:40]}]")
router.register_handler("math_model",    lambda q: f"[Math model response to: {q[:40]}]")
router.register_handler("rag_pipeline",  lambda q: f"[RAG response to: {q[:40]}]")
router.register_handler("general_llm",   lambda q: f"[General LLM response to: {q[:40]}]")
router.register_handler("chitchat_model",lambda q: "Hi there! How can I help?")

# Register rules (higher priority evaluated first)
router.add_rule(
    pattern=r"\b(code|function|class|debug|error|python|javascript|typescript|sql)\b",
    handler="code_model",
    priority=10,
)
router.add_rule(
    pattern=r"\b(calculate|solve|equation|integral|derivative|matrix|statistics)\b",
    handler="math_model",
    priority=10,
)
router.add_rule(
    pattern=r"\b(what is|who is|when did|where is|define|explain|according to)\b",
    handler="rag_pipeline",
    priority=5,
)
router.add_rule(
    pattern=r"\b(hello|hi|hey|how are you|good morning|thanks|thank you)\b",
    handler="chitchat_model",
    priority=8,
)

# Use it
result = router.route("Write a Python function to reverse a linked list")
print(result)  # routed_to: code_model
```

---

## Pattern 2 — Semantic Router

Use embedding similarity to route queries to the closest domain, without requiring exact keyword matches.

```python
import numpy as np
from typing import NamedTuple
from openai import OpenAI

client = OpenAI()


class SemanticRoute(NamedTuple):
    name: str
    examples: list[str]
    handler: Callable


class SemanticRouter:
    """
    Routes queries using cosine similarity between query embedding and
    pre-computed route prototype embeddings.

    Args:
        threshold: minimum similarity score to commit to a route
        embedding_model: OpenAI embedding model to use
    """

    def __init__(
        self,
        threshold: float = 0.75,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.threshold = threshold
        self.embedding_model = embedding_model
        self._routes: list[SemanticRoute] = []
        self._route_embeddings: list[np.ndarray] = []

    def add_route(self, route: SemanticRoute):
        embeddings = self._embed_batch(route.examples)
        prototype = np.mean(embeddings, axis=0)
        prototype /= np.linalg.norm(prototype)
        self._routes.append(route)
        self._route_embeddings.append(prototype)

    def route(self, query: str, fallback: Optional[Callable] = None) -> dict:
        query_emb = self._embed(query)
        similarities = [
            float(np.dot(query_emb, proto))
            for proto in self._route_embeddings
        ]
        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]

        if best_score < self.threshold:
            logger.info(
                "No route above threshold %.2f (best: %.2f) — using fallback",
                self.threshold, best_score,
            )
            handler = fallback or (lambda q: f"No confident route for: {q}")
            return {"response": handler(query), "routed_to": "fallback", "score": best_score}

        route = self._routes[best_idx]
        logger.info("Semantic route '%s' (score=%.3f)", route.name, best_score)
        return {
            "response": route.handler(query),
            "routed_to": route.name,
            "score": best_score,
        }

    def _embed(self, text: str) -> np.ndarray:
        response = client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        vec = np.array(response.data[0].embedding)
        return vec / np.linalg.norm(vec)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        response = client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        vecs = np.array([d.embedding for d in response.data])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms


# --- Setup ---
semantic_router = SemanticRouter(threshold=0.72)

semantic_router.add_route(SemanticRoute(
    name="medical",
    examples=[
        "What are the symptoms of diabetes?",
        "How does insulin work?",
        "What is the recommended dosage for ibuprofen?",
        "Explain the mechanism of statins",
    ],
    handler=lambda q: f"[Medical LLM] {q}",
))

semantic_router.add_route(SemanticRoute(
    name="legal",
    examples=[
        "What does habeas corpus mean?",
        "Explain GDPR compliance requirements",
        "What is the statute of limitations in California?",
        "How does intellectual property law apply to AI?",
    ],
    handler=lambda q: f"[Legal LLM] {q}",
))

semantic_router.add_route(SemanticRoute(
    name="finance",
    examples=[
        "What is dollar-cost averaging?",
        "How do I calculate compound interest?",
        "Explain options trading strategies",
        "What is the Sharpe ratio?",
    ],
    handler=lambda q: f"[Finance LLM] {q}",
))
```

---

## Pattern 3 — Cost-Aware Router

Route to a cheap model first; escalate to an expensive model only when the cheap model signals low confidence.

```python
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    name: str
    cost_per_1k_tokens: float
    max_complexity_score: float
    handler: Callable


class CostAwareRouter:
    """
    Tries models in ascending cost order.
    Escalates when the model's confidence is below the threshold.

    Args:
        confidence_threshold: minimum confidence to accept a model's response
    """

    def __init__(self, confidence_threshold: float = 0.85):
        self.confidence_threshold = confidence_threshold
        self._models: list[ModelConfig] = []

    def add_model(self, config: ModelConfig):
        self._models.append(config)
        self._models.sort(key=lambda m: m.cost_per_1k_tokens)

    def route(self, query: str) -> dict:
        complexity = self._estimate_complexity(query)
        logger.info("Query complexity estimate: %.2f", complexity)

        for model in self._models:
            if complexity > model.max_complexity_score:
                logger.debug("Skipping %s (complexity %.2f > max %.2f)",
                             model.name, complexity, model.max_complexity_score)
                continue

            response, confidence = model.handler(query)
            logger.info("Model %s confidence: %.2f", model.name, confidence)

            if confidence >= self.confidence_threshold:
                return {
                    "response": response,
                    "model_used": model.name,
                    "confidence": confidence,
                    "cost_tier": model.cost_per_1k_tokens,
                }
            logger.info("Escalating from %s (confidence too low)", model.name)

        # All models tried — return best we have
        model = self._models[-1]
        response, confidence = model.handler(query)
        return {
            "response": response,
            "model_used": model.name,
            "confidence": confidence,
            "cost_tier": model.cost_per_1k_tokens,
            "escalated": True,
        }

    def _estimate_complexity(self, query: str) -> float:
        """
        Heuristic complexity score 0–1 based on query characteristics.
        Replace with a learned classifier for production use.
        """
        score = 0.0
        words = query.split()
        score += min(len(words) / 100, 0.3)           # length
        score += 0.2 if "?" in query else 0.0          # question
        score += 0.3 if any(w in query.lower() for w in [
            "compare", "analyze", "explain", "why", "how", "evaluate"
        ]) else 0.0
        score += 0.2 if len(re.findall(r"[;,]", query)) > 3 else 0.0
        return min(score, 1.0)


# --- Example models (tuples of response + confidence) ---

def call_local_llm(query: str):     return (f"[Local] answer", 0.70)
def call_gpt35(query: str):         return (f"[GPT-3.5] answer", 0.88)
def call_gpt4(query: str):          return (f"[GPT-4o] answer", 0.97)


cost_router = CostAwareRouter(confidence_threshold=0.85)
cost_router.add_model(ModelConfig("local-llm", 0.0001, 0.5,  call_local_llm))
cost_router.add_model(ModelConfig("gpt-3.5",   0.002,  0.8,  call_gpt35))
cost_router.add_model(ModelConfig("gpt-4o",    0.015,  1.0,  call_gpt4))
```

---

## Pattern 4 — LLM-as-Router

Use a fast, cheap LLM to classify and route queries — more flexible than regex, cheaper than embeddings.

```python
import json
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)
client = OpenAI()

LLM_ROUTER_PROMPT = """\
You are a routing classifier. Given a user query, select the BEST handler from the list below.

Handlers:
- code_model: programming tasks, debugging, code review, algorithm questions
- math_model: calculations, equations, statistics, data analysis
- rag_pipeline: factual questions requiring knowledge retrieval
- creative_model: writing, brainstorming, storytelling, content generation
- general_llm: everything else

Respond with a JSON object: {{"handler": "<name>", "confidence": <0.0-1.0>, "reason": "<brief>"}}

Query: {query}
"""


class LLMRouter:
    """
    Uses a cheap LLM (e.g., GPT-4o-mini) to classify and route queries.

    Args:
        classifier_model: fast/cheap model for classification
        handlers: dict mapping handler names to callables
        fallback_handler: handler used when classification confidence is low
        min_confidence: threshold below which fallback is used
    """

    def __init__(
        self,
        classifier_model: str = "gpt-4o-mini",
        handlers: Optional[dict[str, Callable]] = None,
        fallback_handler: str = "general_llm",
        min_confidence: float = 0.6,
    ):
        self.classifier_model = classifier_model
        self.handlers = handlers or {}
        self.fallback_handler = fallback_handler
        self.min_confidence = min_confidence

    def route(self, query: str) -> dict:
        classification = self._classify(query)
        handler_name = classification.get("handler", self.fallback_handler)
        confidence = classification.get("confidence", 0.0)

        if confidence < self.min_confidence:
            logger.warning(
                "Low confidence %.2f for handler '%s' — using fallback",
                confidence, handler_name,
            )
            handler_name = self.fallback_handler

        handler = self.handlers.get(handler_name, self.handlers.get(self.fallback_handler))
        result = handler(query)
        return {
            "response": result,
            "routed_to": handler_name,
            "confidence": confidence,
            "reason": classification.get("reason", ""),
        }

    def _classify(self, query: str) -> dict:
        try:
            response = client.chat.completions.create(
                model=self.classifier_model,
                messages=[{
                    "role": "user",
                    "content": LLM_ROUTER_PROMPT.format(query=query),
                }],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as exc:
            logger.error("LLM routing failed: %s", exc)
            return {"handler": self.fallback_handler, "confidence": 0.0, "reason": str(exc)}
```

---

## Pattern 5 — Mixture of Experts (Soft Routing)

A learned gating function weights the contributions of multiple expert models, producing a blended output.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class GatingNetwork(nn.Module):
    """
    Learned routing: produces a probability distribution over experts
    given a query embedding.
    """

    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)


class MixtureOfExpertsRouter:
    """
    Soft routing: blend expert outputs weighted by the gating network.

    Args:
        experts: list of expert model callables
        embedding_dim: dimension of the query embedding
        top_k: number of experts to activate (sparse MoE); None = dense
    """

    def __init__(
        self,
        experts: List[Callable],
        embedding_dim: int,
        top_k: Optional[int] = None,
    ):
        self.experts = experts
        self.top_k = top_k
        self.gate = GatingNetwork(
            input_dim=embedding_dim,
            num_experts=len(experts),
        )

    def forward(
        self,
        query_embedding: torch.Tensor,
        query: str,
    ) -> dict:
        weights = self.gate(query_embedding)  # shape: (num_experts,)

        if self.top_k is not None:
            # Sparse: only top_k experts active
            topk_weights, topk_idx = torch.topk(weights, self.top_k)
            topk_weights = F.softmax(topk_weights, dim=-1)
            active = list(zip(topk_idx.tolist(), topk_weights.tolist()))
        else:
            active = list(enumerate(weights.tolist()))

        results = []
        for idx, weight in active:
            expert_result = self.experts[idx](query)
            results.append((expert_result, weight))

        return {
            "expert_weights": {i: w for i, w in active},
            "results": results,
            "top_expert": max(active, key=lambda x: x[1])[0],
        }
```

---

## Pattern 6 — Conditional Generation Pipeline

A simple if/elif dispatch on parsed intent — readable, maintainable, and fast.

```python
from enum import Enum, auto
import dataclasses


class QueryIntent(Enum):
    CODE_GENERATION  = auto()
    CODE_EXPLANATION = auto()
    SUMMARIZATION    = auto()
    TRANSLATION      = auto()
    QUESTION_ANSWER  = auto()
    CREATIVE_WRITING = auto()
    UNKNOWN          = auto()


@dataclasses.dataclass
class ParsedQuery:
    original: str
    intent: QueryIntent
    language: Optional[str] = None
    target_language: Optional[str] = None


def parse_intent(query: str) -> ParsedQuery:
    q = query.lower()
    if any(kw in q for kw in ["write a function", "implement", "create a class", "code that"]):
        return ParsedQuery(query, QueryIntent.CODE_GENERATION)
    if any(kw in q for kw in ["explain this code", "what does this do", "review"]):
        return ParsedQuery(query, QueryIntent.CODE_EXPLANATION)
    if any(kw in q for kw in ["summarize", "tldr", "brief overview", "key points"]):
        return ParsedQuery(query, QueryIntent.SUMMARIZATION)
    if any(kw in q for kw in ["translate", "in french", "in spanish", "in german"]):
        target = "french" if "french" in q else "spanish" if "spanish" in q else "german"
        return ParsedQuery(query, QueryIntent.TRANSLATION, target_language=target)
    if "?" in query or any(kw in q for kw in ["what", "who", "when", "where", "how"]):
        return ParsedQuery(query, QueryIntent.QUESTION_ANSWER)
    if any(kw in q for kw in ["write a story", "poem", "creative", "imagine"]):
        return ParsedQuery(query, QueryIntent.CREATIVE_WRITING)
    return ParsedQuery(query, QueryIntent.UNKNOWN)


def dispatch(query: str) -> dict:
    parsed = parse_intent(query)

    if parsed.intent == QueryIntent.CODE_GENERATION:
        response = call_code_model(query)
        pipeline = "code-model"
    elif parsed.intent == QueryIntent.CODE_EXPLANATION:
        response = call_code_explanation_model(query)
        pipeline = "code-explanation"
    elif parsed.intent == QueryIntent.SUMMARIZATION:
        response = call_summarization_model(query)
        pipeline = "summarization"
    elif parsed.intent == QueryIntent.TRANSLATION:
        response = call_translation_model(query, parsed.target_language)
        pipeline = f"translation-{parsed.target_language}"
    elif parsed.intent == QueryIntent.QUESTION_ANSWER:
        response = call_rag_pipeline(query)
        pipeline = "rag"
    elif parsed.intent == QueryIntent.CREATIVE_WRITING:
        response = call_creative_model(query)
        pipeline = "creative"
    else:
        response = call_general_model(query)
        pipeline = "general"

    return {"response": response, "pipeline": pipeline, "intent": parsed.intent.name}


def call_code_model(q):             return f"[CodeModel] {q[:30]}"
def call_code_explanation_model(q): return f"[CodeExplainer] {q[:30]}"
def call_summarization_model(q):    return f"[Summarizer] {q[:30]}"
def call_translation_model(q, lang):return f"[Translator→{lang}] {q[:30]}"
def call_rag_pipeline(q):           return f"[RAG] {q[:30]}"
def call_creative_model(q):         return f"[Creative] {q[:30]}"
def call_general_model(q):          return f"[General] {q[:30]}"
```

---

## Pattern 7 — Router with Fallback Chain

<div class="diagram">
<div class="diagram-title">Router with Cascading Fallback</div>
<div class="flow">
  <div class="flow-node blue wide">Query</div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-node green wide">Primary Route<br/><small>specialist model</small></div>
  <div class="flow-arrow accent">↓ fail / low confidence</div>
  <div class="flow-node orange wide">Secondary Route<br/><small>general-purpose model</small></div>
  <div class="flow-arrow accent">↓ fail</div>
  <div class="flow-node teal wide">Default Route<br/><small>lightweight fallback</small></div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-node blue wide">Response</div>
</div>
</div>

```python
from typing import Optional


class FallbackRouter:
    """
    Tries each handler in the chain; advances on exception or low confidence.
    """

    def __init__(
        self,
        chain: list[tuple[str, Callable]],
        min_confidence: float = 0.7,
    ):
        self.chain = chain  # list of (name, handler)
        self.min_confidence = min_confidence

    def route(self, query: str) -> dict:
        last_error: Optional[Exception] = None
        for name, handler in self.chain:
            try:
                result = handler(query)
                confidence = result.get("confidence", 1.0) if isinstance(result, dict) else 1.0
                if confidence >= self.min_confidence:
                    return {"response": result, "handler": name, "confidence": confidence}
                logger.info("Handler %s confidence %.2f below threshold — advancing", name, confidence)
            except Exception as exc:
                last_error = exc
                logger.warning("Handler %s failed: %s — trying next", name, exc)

        # All handlers exhausted
        return {
            "response": None,
            "handler": "none",
            "error": str(last_error) if last_error else "All handlers below threshold",
        }
```

---

## Pattern 8 — A/B Routing for Model Comparison

```python
import random
import time
import logging
from dataclasses import dataclass, field
from typing import Callable
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ABVariant:
    name: str
    handler: Callable
    traffic_fraction: float  # 0.0 to 1.0


@dataclass
class ABResult:
    variant: str
    response: str
    latency_ms: float


class ABRouter:
    """
    Splits traffic between model variants for online evaluation.

    Args:
        variants: list of ABVariant objects; fractions should sum to 1.0
        log_results: if True, records all results for analysis
    """

    def __init__(self, variants: list[ABVariant], log_results: bool = True):
        total = sum(v.traffic_fraction for v in variants)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Traffic fractions must sum to 1.0, got {total:.2f}")
        self.variants = variants
        self.log_results = log_results
        self._results: list[ABResult] = []
        self._counts: dict[str, int] = defaultdict(int)

    def route(self, query: str) -> ABResult:
        variant = self._select_variant()
        self._counts[variant.name] += 1

        start = time.monotonic()
        response = variant.handler(query)
        latency_ms = (time.monotonic() - start) * 1000

        result = ABResult(variant.name, response, latency_ms)
        if self.log_results:
            self._results.append(result)

        logger.debug("A/B: variant=%s latency=%.1fms", variant.name, latency_ms)
        return result

    def _select_variant(self) -> ABVariant:
        r = random.random()
        cumulative = 0.0
        for v in self.variants:
            cumulative += v.traffic_fraction
            if r < cumulative:
                return v
        return self.variants[-1]

    def stats(self) -> dict:
        return {
            "total_requests": sum(self._counts.values()),
            "per_variant": dict(self._counts),
            "avg_latency_ms": {
                name: sum(r.latency_ms for r in self._results if r.variant == name)
                     / max(self._counts[name], 1)
                for name in self._counts
            },
        }


ab_router = ABRouter([
    ABVariant("gpt-4o",    handler=call_general_model, traffic_fraction=0.1),
    ABVariant("gpt-4o-mini", handler=call_general_model, traffic_fraction=0.9),
])
```

---

## Comparison Table

<table class="compare-table">
<thead>
<tr>
  <th>Router Type</th>
  <th>Mechanism</th>
  <th>Speed</th>
  <th>Cost</th>
  <th>Accuracy</th>
  <th>Setup Effort</th>
  <th>Best For</th>
</tr>
</thead>
<tbody>
<tr>
  <td><strong>Rule-Based</strong></td>
  <td>Regex / keyword match</td>
  <td>⚡ Instant</td>
  <td>$0</td>
  <td>Medium</td>
  <td>Low</td>
  <td>Well-defined intents with distinct keywords</td>
</tr>
<tr>
  <td><strong>Embedding (Semantic)</strong></td>
  <td>Cosine similarity to prototypes</td>
  <td>Fast (10–50ms)</td>
  <td>Low</td>
  <td>High</td>
  <td>Medium</td>
  <td>Domain routing with fuzzy boundaries</td>
</tr>
<tr>
  <td><strong>LLM Router</strong></td>
  <td>Small LLM classification</td>
  <td>Medium (200–500ms)</td>
  <td>Medium</td>
  <td>Very High</td>
  <td>Low</td>
  <td>Complex intent with nuanced distinctions</td>
</tr>
<tr>
  <td><strong>Cost-Aware</strong></td>
  <td>Complexity estimation + escalation</td>
  <td>Varies</td>
  <td>Optimized</td>
  <td>Adapts</td>
  <td>Medium</td>
  <td>Cost-sensitive production workloads</td>
</tr>
<tr>
  <td><strong>MoE (Soft)</strong></td>
  <td>Learned gating network</td>
  <td>Fast (learned)</td>
  <td>Training needed</td>
  <td>Highest</td>
  <td>High</td>
  <td>Fine-grained blending of specialist models</td>
</tr>
</tbody>
</table>

---

<div class="callout tip">
<span class="callout-icon">💡</span>
<div class="callout-body">
Start simple: a rule-based router with 5–10 patterns handles 80% of traffic in most products. Add semantic routing for ambiguous intents. Only reach for an LLM router when rule-based patterns become unmaintainable.
</div>
</div>

<div class="callout warn">
<span class="callout-icon">⚠️</span>
<div class="callout-body">
Avoid making the router a <strong>god object</strong>. It should classify and dispatch — not pre-process inputs, manage sessions, cache results, or apply business logic. Those concerns belong in the handlers.
</div>
</div>

---

## Summary

<div class="diagram">
<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-year">Rule-Based</div>
    <div class="timeline-title">Start here</div>
    <div class="timeline-desc">Fast, free, interpretable. Handles clear intents with deterministic keywords.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Semantic</div>
    <div class="timeline-title">Fuzzy matching</div>
    <div class="timeline-desc">Embedding cosine similarity generalizes beyond exact keywords to semantically similar queries.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">LLM Router</div>
    <div class="timeline-title">Flexible classification</div>
    <div class="timeline-desc">Use a small, cheap model (GPT-4o-mini) to classify complex or ambiguous intents.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Cost-Aware</div>
    <div class="timeline-title">Spend wisely</div>
    <div class="timeline-desc">Try cheap models first; escalate to expensive ones only when confidence is low.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">A/B Routing</div>
    <div class="timeline-title">Measure before committing</div>
    <div class="timeline-desc">Split live traffic to compare model quality before full rollout.</div>
  </div>
</div>
</div>

<span class="badge agentic">Agentic</span> <span class="badge mlops">MLOps</span>

*Last updated: May 2026*
