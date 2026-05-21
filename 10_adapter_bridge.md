---
title: "Chapter 10 — Adapter & Bridge"
---
[← Back to Table of Contents](./README.md)

# Chapter 10 — Adapter & Bridge

> *"The Adapter makes incompatible things speak the same language. The Bridge separates what a thing is from how it's done."*

---

## 10.1 The Adapter Pattern

The **Adapter** (also called Wrapper) converts the interface of a class into another interface that clients expect. It lets classes work together that otherwise couldn't because of incompatible interfaces.

Think of it as a power-plug adapter: the appliance (client) expects a round socket (target interface), the wall provides a flat socket (adaptee). The adapter physically bridges the gap without modifying either end.

<div class="diagram">
  <div class="diagram-title">Adapter Pattern — Interface Translation</div>
  <div class="flow flow-h">
    <div class="flow-node accent wide">Client<br><small>expects: Target interface</small></div>
    <div class="flow-arrow accent">→ calls</div>
    <div class="flow-node blue wide">Adapter<br><small>implements Target<br>wraps Adaptee</small></div>
    <div class="flow-arrow blue">→ delegates</div>
    <div class="flow-node green wide">Adaptee<br><small>has incompatible interface<br>e.g. legacy API</small></div>
  </div>
</div>

---

## 10.2 Object Adapter vs Class Adapter

Python supports both forms. The **Object Adapter** (preferred) wraps an instance via composition. The **Class Adapter** inherits from both Target and Adaptee simultaneously using multiple inheritance.

```python
# ── Object Adapter (preferred) ────────────────────────────────────────────────
class ObjectAdapter:
    """Wraps an adaptee instance. Preferred — loose coupling."""
    def __init__(self, adaptee):
        self._adaptee = adaptee          # composition

    def target_method(self, *args):
        return self._adaptee.legacy_method(*args)   # translation


# ── Class Adapter (use sparingly) ─────────────────────────────────────────────
class ClassAdapter(Target, Adaptee):
    """Inherits from both. Python MRO handles dispatch.
    Fragile — changes in Adaptee break the adapter."""
    def target_method(self, *args):
        return self.legacy_method(*args)
```

| | Object Adapter | Class Adapter |
|---|---|---|
| **Coupling** | Loose (composition) | Tight (inheritance) |
| **Adaptee** | Can adapt any instance | Fixed to one Adaptee class |
| **Overriding** | Can wrap multiple adaptees | Can override Adaptee internals |
| **Python idiom** | ✅ Preferred | Use only when inheritance benefit is clear |

---

## 10.3 UML Diagram

<div class="uml-row">
  <div class="uml-box">
    <div class="uml-title">&lt;&lt;interface&gt;&gt; Target</div>
    <div class="uml-section">Methods</div>
    <div class="uml-item">+ request() → Result</div>
    <div class="uml-item">+ configure(**kwargs) → None</div>
  </div>

  <div class="uml-box">
    <div class="uml-title">Adapter</div>
    <div class="uml-section">Attributes</div>
    <div class="uml-item">- _adaptee: Adaptee</div>
    <div class="uml-section">Methods</div>
    <div class="uml-item">+ request() → Result</div>
    <div class="uml-item">  ↳ _adaptee.old_request()</div>
    <div class="uml-item">+ configure(**kwargs)</div>
    <div class="uml-item">  ↳ _adaptee.set_params()</div>
  </div>

  <div class="uml-box">
    <div class="uml-title">Adaptee</div>
    <div class="uml-section">Methods</div>
    <div class="uml-item">+ old_request() → OldResult</div>
    <div class="uml-item">+ set_params(**legacy_kw)</div>
    <div class="uml-item">  (incompatible interface)</div>
  </div>
</div>

---

## 10.4 ML Adapter Examples

### HuggingFace Model → PyTorch `nn.Module` Adapter

HuggingFace `PreTrainedModel` returns a `ModelOutput` dataclass. Our training loop expects a raw `Tensor`. An adapter bridges the gap:

```python
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Any


class HuggingFaceModelAdapter(nn.Module):
    """
    Target interface: standard nn.Module with forward(x) → Tensor.
    Adaptee: HuggingFace PreTrainedModel with complex ModelOutput.

    Clients call this exactly like any nn.Module — they never
    see the HuggingFace API surface.
    """

    def __init__(
        self,
        model_name: str,
        pooling: str = "cls",     # "cls" | "mean" | "max"
        output_hidden: bool = False,
    ):
        super().__init__()
        self._hf_model = AutoModel.from_pretrained(model_name)
        self._pooling = pooling
        self._output_hidden = output_hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Adaptee returns ModelOutput (not a plain Tensor)
        outputs = self._hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state   # (B, L, H)

        # Translate to the target interface's expected shape
        if self._pooling == "cls":
            return hidden[:, 0, :]           # [CLS] token
        elif self._pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float() if attention_mask is not None \
                   else torch.ones_like(hidden[..., :1])
            return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        elif self._pooling == "max":
            return hidden.max(dim=1).values
        raise ValueError(f"Unknown pooling: {self._pooling}")

    @property
    def hidden_size(self) -> int:
        return self._hf_model.config.hidden_size

    def freeze_encoder(self) -> None:
        for p in self._hf_model.parameters():
            p.requires_grad_(False)


# Usage — client code sees a standard nn.Module
# encoder = HuggingFaceModelAdapter("bert-base-uncased", pooling="cls")
# encoder.freeze_encoder()
# classifier = nn.Linear(encoder.hidden_size, 5)
# model = nn.Sequential(encoder, classifier)
```

### Dataset Format Adapter: HuggingFace Dataset → PyTorch Dataset

```python
import torch
from torch.utils.data import Dataset
from typing import Callable, Any


class HuggingFaceDatasetAdapter(Dataset):
    """
    Target: torch.utils.data.Dataset with __getitem__ and __len__.
    Adaptee: HuggingFace datasets.Dataset (arrow-backed, dict-based).

    Bridges the interface mismatch without copying data.
    """

    def __init__(
        self,
        hf_dataset,                         # datasets.Dataset
        text_column: str = "text",
        label_column: str = "label",
        tokenizer: Callable | None = None,
        max_length: int = 128,
    ):
        self._ds = hf_dataset
        self._text_col = text_column
        self._label_col = label_column
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._ds)               # HF Dataset supports len()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._ds[idx]                 # returns a dict
        text  = row[self._text_col]
        label = row[self._label_col]

        if self._tokenizer is not None:
            encoding = self._tokenizer(
                text,
                max_length=self._max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids":      encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "label":          torch.tensor(label, dtype=torch.long),
            }

        return {"text": text, "label": torch.tensor(label, dtype=torch.long)}


# ── Usage ─────────────────────────────────────────────────────────────────────
# from datasets import load_dataset
# from transformers import AutoTokenizer
#
# raw = load_dataset("imdb", split="train")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# adapted_ds = HuggingFaceDatasetAdapter(raw, tokenizer=tokenizer)
# loader = DataLoader(adapted_ds, batch_size=32, num_workers=4)
```

### OpenAI API → Internal LLM Interface Adapter

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    text: str
    tokens_used: int
    model: str
    finish_reason: str


class ILanguageModel(ABC):
    """Target interface — our internal contract for all LLMs."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

    @property
    @abstractmethod
    def max_context_length(self) -> int: ...


class OpenAIAdapter(ILanguageModel):
    """
    Adapts the openai Python client to our ILanguageModel interface.
    Client code calls generate() / embed() — never openai-specific methods.
    """

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        import openai
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._embed_model = "text-embedding-3-small"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        choice = resp.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            tokens_used=resp.usage.total_tokens,
            model=self._model,
            finish_reason=choice.finish_reason,
        )

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(
            model=self._embed_model, input=text
        )
        return resp.data[0].embedding

    @property
    def max_context_length(self) -> int:
        context_lengths = {
            "gpt-4o": 128_000,
            "gpt-4-turbo": 128_000,
            "gpt-3.5-turbo": 16_385,
        }
        return context_lengths.get(self._model, 4_096)


class AnthropicAdapter(ILanguageModel):
    """Same ILanguageModel interface, different backend."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        import anthropic
        self._client = anthropic.Anthropic()
        self._model = model

    def generate(self, prompt, max_tokens=256, temperature=0.7, stop=None) -> LLMResponse:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            stop_sequences=stop,
        )
        return LLMResponse(
            text=resp.content[0].text,
            tokens_used=resp.usage.input_tokens + resp.usage.output_tokens,
            model=self._model,
            finish_reason=resp.stop_reason,
        )

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Anthropic does not provide an embedding API")

    @property
    def max_context_length(self) -> int:
        return 200_000


# Client code is fully decoupled from provider
def summarise(llm: ILanguageModel, document: str) -> str:
    response = llm.generate(
        f"Summarise this in 3 sentences:\n\n{document}",
        max_tokens=200,
        temperature=0.3,
    )
    return response.text
```

---

## 10.5 The Bridge Pattern

The **Bridge** pattern decouples an abstraction from its implementation so that both can vary independently. Unlike the Adapter (which makes existing things work together), Bridge is a *design-time* decision: you intentionally split a class hierarchy in two.

> Adapter solves a problem with existing code.
> Bridge solves a problem before you write the code.

<div class="diagram">
  <div class="diagram-title">Bridge Pattern — Two Independent Dimensions</div>
  <div class="flow flow-h">
    <div class="flow-node accent wide">Abstraction<br><small>high-level logic</small></div>
    <div class="flow-arrow accent">has-a →</div>
    <div class="flow-node blue wide">Implementor<br><small>low-level interface</small></div>
  </div>
  <div class="flow flow-h">
    <div class="flow-node green wide">RefinedAbstraction A<br><small>extends abstraction</small></div>
    <div class="flow-arrow green">↕ varies</div>
    <div class="flow-node purple wide">ConcreteImplementor B<br><small>extends implementor</small></div>
  </div>
</div>

**Key insight**: Without Bridge you get N×M classes (every abstraction × every implementation). With Bridge you get N + M classes.

---

## 10.6 UML Diagram

<div class="uml-row">
  <div class="uml-box">
    <div class="uml-title">Abstraction</div>
    <div class="uml-section">Attributes</div>
    <div class="uml-item"># impl: Implementor</div>
    <div class="uml-section">Methods</div>
    <div class="uml-item">+ operation() → Result</div>
    <div class="uml-item">  ↳ delegates to impl</div>
  </div>

  <div class="uml-box">
    <div class="uml-title">RefinedAbstraction</div>
    <div class="uml-section">Extends Abstraction</div>
    <div class="uml-item">+ refined_operation()</div>
    <div class="uml-item">  ↳ adds logic, still</div>
    <div class="uml-item">     delegates to impl</div>
  </div>

  <div class="uml-box">
    <div class="uml-title">&lt;&lt;interface&gt;&gt; Implementor</div>
    <div class="uml-section">Methods</div>
    <div class="uml-item">+ execute(data) → Result</div>
  </div>

  <div class="uml-box">
    <div class="uml-title">ConcreteImplementor</div>
    <div class="uml-section">E.g. CPUImpl / CUDAImpl</div>
    <div class="uml-item">+ execute(data) → Result</div>
    <div class="uml-item">  ↳ device-specific logic</div>
  </div>
</div>

---

## 10.7 Bridge in ML: Training vs Inference Bridged to Backends

Consider a model abstraction that bridges to different compute backends — CPU, CUDA, and Apple MPS:

```python
from __future__ import annotations
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Implementor side — computation backends
# ══════════════════════════════════════════════════════════════════════════════

class ComputeBackend(ABC):
    """Implementor — defines the interface for all compute backends."""

    @abstractmethod
    def allocate(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor: ...

    @abstractmethod
    def transfer(self, tensor: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def synchronize(self) -> None: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @abstractmethod
    def optimise_model(self, model: nn.Module) -> nn.Module: ...


class CPUBackend(ComputeBackend):
    def allocate(self, shape, dtype=torch.float32): return torch.empty(shape, dtype=dtype)
    def transfer(self, tensor): return tensor.cpu()
    def synchronize(self): pass     # no-op on CPU
    @property
    def device(self): return torch.device("cpu")
    def optimise_model(self, model): return torch.jit.script(model)


class CUDABackend(ComputeBackend):
    def __init__(self, device_id: int = 0):
        self._device = torch.device(f"cuda:{device_id}")

    def allocate(self, shape, dtype=torch.float32):
        return torch.empty(shape, dtype=dtype, device=self._device)

    def transfer(self, tensor): return tensor.to(self._device)
    def synchronize(self): torch.cuda.synchronize(self._device)

    @property
    def device(self): return self._device

    def optimise_model(self, model: nn.Module) -> nn.Module:
        return torch.compile(model, backend="inductor", mode="max-autotune")


class MPSBackend(ComputeBackend):
    """Apple Silicon Metal Performance Shaders backend."""
    def allocate(self, shape, dtype=torch.float32):
        return torch.empty(shape, dtype=dtype, device="mps")

    def transfer(self, tensor): return tensor.to("mps")
    def synchronize(self): torch.mps.synchronize()

    @property
    def device(self): return torch.device("mps")

    def optimise_model(self, model: nn.Module) -> nn.Module:
        return model   # torch.compile MPS support is partial as of 2025


# ══════════════════════════════════════════════════════════════════════════════
# Abstraction side — model usage modes
# ══════════════════════════════════════════════════════════════════════════════

class ModelAbstraction(ABC):
    """
    Abstraction — knows WHAT to do (train/infer) but not HOW (CPU/CUDA/MPS).
    The HOW is delegated to the ComputeBackend implementor.
    """

    def __init__(self, model: nn.Module, backend: ComputeBackend):
        self._model   = backend.optimise_model(model.to(backend.device))
        self._backend = backend

    @abstractmethod
    def run(self, x: torch.Tensor, **kwargs) -> Any: ...

    def to_backend(self, x: torch.Tensor) -> torch.Tensor:
        return self._backend.transfer(x)


class TrainingMode(ModelAbstraction):
    """
    RefinedAbstraction — adds training-specific logic:
    gradient computation, loss backward, optimizer step.
    """

    def __init__(
        self,
        model: nn.Module,
        backend: ComputeBackend,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ):
        super().__init__(model, backend)
        self._optimizer  = optimizer
        self._criterion  = criterion

    def run(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x, y = self.to_backend(x), self.to_backend(y)
        self._model.train()
        self._optimizer.zero_grad()
        loss = self._criterion(self._model(x), y)
        loss.backward()
        self._optimizer.step()
        self._backend.synchronize()
        return loss.item()


class InferenceMode(ModelAbstraction):
    """
    RefinedAbstraction — adds inference-specific logic:
    no grad, batch aggregation, optional half-precision.
    """

    def __init__(
        self,
        model: nn.Module,
        backend: ComputeBackend,
        use_amp: bool = False,
    ):
        super().__init__(model, backend)
        self._use_amp = use_amp

    def run(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_backend(x)
        self._model.eval()
        ctx = torch.autocast(str(self._backend.device).split(":")[0]) \
              if self._use_amp else torch.no_grad()
        with ctx:
            out = self._model(x)
        self._backend.synchronize()
        return out.cpu()


# ── Usage: swap backends without changing abstraction ─────────────────────────
model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

if torch.cuda.is_available():
    backend = CUDABackend(device_id=0)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    backend = MPSBackend()
else:
    backend = CPUBackend()

trainer   = TrainingMode(model, backend, opt, nn.CrossEntropyLoss())
predictor = InferenceMode(model, backend, use_amp=torch.cuda.is_available())

x_batch = torch.randn(32, 784)
y_batch = torch.randint(0, 10, (32,))

loss = trainer.run(x_batch, y_batch)
preds = predictor.run(x_batch)
```

---

## 10.8 Bridge: Loss Function Bridge

Decouple the logical loss structure from framework-specific implementations:

```python
from __future__ import annotations
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


# ── Implementor ───────────────────────────────────────────────────────────────

class LossImplementor(ABC):
    @abstractmethod
    def compute(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor: ...


class PyTorchCEImpl(LossImplementor):
    def compute(self, logits, targets):
        return nn.functional.cross_entropy(logits, targets)


class PyTorchFocalImpl(LossImplementor):
    def __init__(self, gamma: float = 2.0):
        self.gamma = gamma

    def compute(self, logits, targets):
        ce   = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt   = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class NumpyCEImpl(LossImplementor):
    """CPU-only numpy implementation for debugging / portability."""
    def compute(self, logits, targets):
        import numpy as np
        logits_np  = logits.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        exp = np.exp(logits_np - logits_np.max(axis=-1, keepdims=True))
        softmax = exp / exp.sum(axis=-1, keepdims=True)
        ce = -np.log(softmax[np.arange(len(targets_np)), targets_np] + 1e-9)
        return torch.tensor(ce.mean(), dtype=torch.float32)


# ── Abstraction ───────────────────────────────────────────────────────────────

class LossFunction(ABC):
    """Abstraction — knows WHAT loss behaviour to apply, not HOW to compute it."""

    def __init__(self, impl: LossImplementor):
        self._impl = impl

    @abstractmethod
    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor: ...


class ClassificationLoss(LossFunction):
    """Standard classification loss."""
    def __call__(self, logits, targets):
        return self._impl.compute(logits, targets)


class WeightedClassificationLoss(LossFunction):
    """Applies a scalar weight to the base loss."""
    def __init__(self, impl: LossImplementor, weight: float = 1.0):
        super().__init__(impl)
        self._weight = weight

    def __call__(self, logits, targets):
        return self._weight * self._impl.compute(logits, targets)


class MultiTaskLoss(LossFunction):
    """Combines multiple task losses — abstraction over N tasks."""
    def __init__(self, impls: dict[str, LossImplementor], weights: dict[str, float] | None = None):
        self._impls   = impls
        self._weights = weights or {k: 1.0 for k in impls}

    def __call__(self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]):
        total = torch.tensor(0.0)
        for task, impl in self._impls.items():
            total = total + self._weights[task] * impl.compute(outputs[task], targets[task])
        return total


# ── Bridge: swap impl at runtime ─────────────────────────────────────────────
debug_loss = ClassificationLoss(NumpyCEImpl())
production_loss = ClassificationLoss(PyTorchCEImpl())
focal_loss = WeightedClassificationLoss(PyTorchFocalImpl(gamma=2.5), weight=0.7)

# Logits and targets
logits  = torch.randn(8, 10)
targets = torch.randint(0, 10, (8,))

print(debug_loss(logits, targets))
print(production_loss(logits, targets))
print(focal_loss(logits, targets))
```

---

## 10.9 Comparison Table: Adapter vs Bridge vs Decorator

<div class="compare-table">

| Aspect | Adapter | Bridge | Decorator |
|---|---|---|---|
| **Intent** | Make incompatible interfaces work together | Decouple abstraction from implementation | Add behaviour without subclassing |
| **When applied** | After the fact (existing code) | Before writing new code | Before or after |
| **Structure** | Wraps one incompatible class | Has-a reference to an implementor | Wraps same interface recursively |
| **Hierarchy** | Does not change hierarchy | Splits into two independent hierarchies | Extends existing hierarchy |
| **Interface change** | Yes — exposes a different interface | No — same interface to clients | No — same interface |
| **Multiplied classes** | 1 adapter per pair | N + M instead of N×M | N + K decorators |
| **Python example** | `HuggingFaceModelAdapter` | `TrainingMode(model, CUDABackend())` | `nn.Dropout` wrapping any module |
| **ML use case** | Cross-framework compatibility | Multi-backend inference | Regularisation, logging wrappers |

</div>

---

## 10.10 Flow Diagram: Adapter

<div class="diagram">
  <div class="diagram-title">Adapter: Old Interface → Adapter → New Interface</div>
  <div class="flow">
    <div class="flow-node accent wide">Client Code<br><small>expects: nn.Module.forward(input_ids, attention_mask)</small></div>
    <div class="flow-arrow accent">↓ calls</div>
    <div class="flow-node blue wide">HuggingFaceModelAdapter<br><small>implements nn.Module<br>wraps HF PreTrainedModel</small></div>
    <div class="flow-arrow blue">↓ translates call to</div>
    <div class="flow-node green wide">HuggingFace PreTrainedModel<br><small>returns ModelOutput(last_hidden_state, ...)</small></div>
    <div class="flow-arrow green">↓ result translated back</div>
    <div class="flow-node purple wide">Plain torch.Tensor returned<br><small>client never sees ModelOutput</small></div>
  </div>
</div>

---

## 10.11 Flow Diagram: Bridge

<div class="diagram">
  <div class="diagram-title">Bridge: Abstraction ↔ Implementor — Both Can Vary Independently</div>
  <div class="flow flow-h">
    <div class="flow-node accent wide">Abstraction axis<br><small>TrainingMode<br>InferenceMode<br>ExportMode</small></div>
    <div class="flow-arrow accent">bridged via has-a →</div>
    <div class="flow-node green wide">Implementor axis<br><small>CPUBackend<br>CUDABackend<br>MPSBackend</small></div>
  </div>
  <div class="flow flow-h">
    <div class="flow-node blue">3 abstractions</div>
    <div class="flow-label">✕</div>
    <div class="flow-node purple">3 backends</div>
    <div class="flow-label">= 3+3=6 classes (not 9!)</div>
  </div>
</div>

---

## 10.12 Real-World ML Adapters

<div class="compare-table">

| Framework Pair | What Gets Adapted | Adapter Type |
|---|---|---|
| HuggingFace → PyTorch | `ModelOutput` → `Tensor` | Object Adapter |
| HuggingFace Dataset → PyTorch | `Dataset` dict API → `__getitem__` | Object Adapter |
| OpenAI → internal LLM | Chat completions API → `generate()` | Object Adapter |
| Anthropic → internal LLM | Messages API → `generate()` | Object Adapter |
| sklearn → PyTorch | `predict_proba` → `forward()` | Object Adapter |
| TensorFlow SavedModel → TorchScript | `tf.function` → `torch.jit` | Class Adapter |
| ONNX Runtime → PyTorch | `InferenceSession.run()` → `forward()` | Object Adapter |
| LangChain → internal | `chain.invoke()` → `generate()` | Object Adapter |
| Gymnasium → custom RL env | `step()` / `reset()` → typed API | Object Adapter |
| MLflow → internal logger | `log_metric()` → `ILogger.log()` | Object Adapter |
| W&B → internal logger | `wandb.log()` → `ILogger.log()` | Object Adapter |
| Triton Inference → gRPC | Protobuf requests → Tensor I/O | Object Adapter |

</div>

---

## 10.13 Advanced Pattern: Composing Adapter and Bridge

In large ML systems, Adapter and Bridge are often used together:

```python
from __future__ import annotations
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


# Bridge implementor — inference engine
class InferenceEngine(ABC):
    @abstractmethod
    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...


class TorchInferenceEngine(InferenceEngine):
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self._model = model.to(device).eval()
        self._device = device

    def infer(self, inputs):
        with torch.no_grad():
            x = inputs["input"].to(self._device)
            return {"output": self._model(x).cpu()}


class ONNXInferenceEngine(InferenceEngine):
    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        self._session = ort.InferenceSession(onnx_path)

    def infer(self, inputs):
        np_inputs = {k: v.numpy() for k, v in inputs.items()}
        results = self._session.run(None, np_inputs)
        return {"output": torch.from_numpy(results[0])}


# Bridge abstraction — prediction API
class Predictor(ABC):
    def __init__(self, engine: InferenceEngine):
        self._engine = engine

    @abstractmethod
    def predict(self, raw_input: Any) -> Any: ...


class ClassificationPredictor(Predictor):
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        result = self._engine.infer({"input": x})
        return result["output"].softmax(dim=-1)


# Adapter — makes the Bridge-based Predictor look like an sklearn estimator
class SklearnPredictor:
    """
    Adapter: wraps Predictor (Bridge) to expose sklearn's predict_proba interface.
    Combines Bridge (engine-agnostic prediction) and Adapter (sklearn compat).
    """

    def __init__(self, predictor: Predictor):
        self._predictor = predictor

    def predict_proba(self, X):
        """sklearn-compatible interface: numpy in, numpy out."""
        import numpy as np
        tensor = torch.from_numpy(X.astype("float32"))
        probs = self._predictor.predict(tensor)
        return probs.numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
```

---

## Summary

<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-icon">🔌</div>
    <div class="card-title">Adapter</div>
    <div class="card-desc">Converts one interface to another. Applied to existing, incompatible code. The go-to pattern for cross-framework ML interop: HuggingFace ↔ PyTorch, OpenAI ↔ internal APIs.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🌉</div>
    <div class="card-title">Bridge</div>
    <div class="card-desc">Separates WHAT from HOW at design time. Keeps N + M classes instead of N×M. Ideal for multi-backend systems: training/inference abstractions over CPU/CUDA/MPS.</div>
  </div>
</div>

<div class="callout tip">
<strong>💡 Choosing Between Adapter and Bridge</strong><br>
Ask: <em>Am I solving a legacy compatibility problem?</em> → <strong>Adapter</strong>.<br>
Ask: <em>Am I designing for multiple independently varying dimensions?</em> → <strong>Bridge</strong>.<br>
They are complementary — Adapter fixes the past, Bridge designs the future.
</div>

<div class="callout info">
<strong>ℹ️ Pattern Interaction</strong><br>
In production ML systems, Adapter and Bridge frequently appear together:
<ul>
<li>An <strong>Adapter</strong> wraps a third-party model (HuggingFace) to match your <em>Target</em> interface.</li>
<li>A <strong>Bridge</strong> then decouples the model's training mode from its compute backend.</li>
<li>A <strong>Decorator</strong> might wrap either one to add logging or profiling.</li>
</ul>
Understanding when to apply each pattern — and how they compose — is the hallmark of mature ML system design.
</div>

---

*Last updated: May 2026*
