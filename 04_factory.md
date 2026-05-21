---
title: "Chapter 4 — Factory & Abstract Factory"
---

[← Back to Table of Contents](./README.md)

# Chapter 4 — Factory & Abstract Factory

> *"The Factory pattern decouples object creation from the code that uses the object. The creator does not need to know the specific class it is creating."*
> — Erich Gamma et al., *Design Patterns*, 1994

---

## The Factory Pattern: Intent and Motivation

At its core, the Factory pattern answers one question: *how do we create objects without specifying their exact class?*

In ML code, you constantly face this problem. You want to write a training loop that works with any model architecture — ResNet, ViT, BERT, a custom graph network. You want a configuration file to drive which model gets created, without the training loop importing each architecture explicitly. You want to add new architectures without changing the training loop at all.

The naive solution — a long `if/elif` chain in the calling code — works for two or three options, but collapses under the weight of a real model zoo. The Factory pattern solves this with a clean separation: calling code asks for an object by name; factory code creates the right concrete class.

### The Problem in ML Code

```python
# ── PROBLEM: if/elif explosion in calling code ────────────────────────────────

import torch.nn as nn


def create_model_naive(config: dict) -> nn.Module:
    """This approach breaks down quickly."""
    arch = config["architecture"]
    if arch == "resnet18":
        from torchvision.models import resnet18
        return resnet18(num_classes=config["num_classes"])
    elif arch == "resnet50":
        from torchvision.models import resnet50
        return resnet50(num_classes=config["num_classes"])
    elif arch == "vit_base":
        from torchvision.models import vit_b_16
        return vit_b_16(num_classes=config["num_classes"])
    # ... 20 more elifs ...
    # Adding a new architecture means editing THIS function
    else:
        raise ValueError(f"Unknown architecture: {arch}")
```

This pattern has three serious problems:
1. **Violates OCP**: every new architecture requires modifying this function.
2. **Tight coupling**: the calling code imports and knows about every concrete class.
3. **Not extensible by third parties**: users cannot register their own architectures.

---

## Simple Factory vs. Factory Method vs. Abstract Factory

<div class="diagram">
<div class="diagram-title">Factory Variants</div>
<div class="flow">
  <div class="flow-node accent wide">The Factory Family</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-h">
    <div class="flow-node blue narrow">
      <strong>Simple Factory</strong><br/>
      <small>Not a GoF pattern.<br/>A static method or<br/>function that creates<br/>objects. No subclassing.</small>
    </div>
    <div class="flow-node green narrow">
      <strong>Factory Method</strong><br/>
      <small>GoF Creational.<br/>Subclasses override<br/>a factory method to<br/>decide which class<br/>to instantiate.</small>
    </div>
    <div class="flow-node purple narrow">
      <strong>Abstract Factory</strong><br/>
      <small>GoF Creational.<br/>Creates families of<br/>related objects via<br/>an abstract interface.<br/>Multiple methods.</small>
    </div>
  </div>
</div>
</div>

| | Simple Factory | Factory Method | Abstract Factory |
|-|----------------|----------------|-----------------|
| **GoF Pattern?** | No | Yes | Yes |
| **Mechanism** | Function or static method | Virtual method in subclass | Abstract class with multiple factory methods |
| **What varies** | Which concrete class | Which factory method to call | Which factory (whole family of objects) |
| **Extension** | Edit the factory function | Create a new subclass | Create a new concrete factory |
| **Best for** | Small, stable sets of types | Frameworks that let plugins define products | Related families of objects that must be consistent |

---

## Factory Method: UML Structure

<div class="diagram">
<div class="diagram-title">Factory Method — Structure</div>
<div class="uml-row">
  <div class="uml-box">
    <div class="uml-title">«abstract» Creator</div>
    <div class="uml-section">
      <div class="uml-item">+ create_product() → Product</div>
      <div class="uml-item">+ some_operation() → None</div>
    </div>
    <div class="uml-section">
      <div class="uml-item">▲ create_product() is abstract</div>
      <div class="uml-item">▲ some_operation calls create_product()</div>
    </div>
  </div>
  <div class="uml-box">
    <div class="uml-title">ConcreteCreatorA</div>
    <div class="uml-section">
      <div class="uml-item">+ create_product() → ProductA</div>
    </div>
  </div>
  <div class="uml-box">
    <div class="uml-title">ConcreteCreatorB</div>
    <div class="uml-section">
      <div class="uml-item">+ create_product() → ProductB</div>
    </div>
  </div>
</div>
<div class="uml-row">
  <div class="uml-box">
    <div class="uml-title">«interface» Product</div>
    <div class="uml-section">
      <div class="uml-item">+ operation() → Tensor</div>
    </div>
  </div>
  <div class="uml-box">
    <div class="uml-title">ConcreteProductA</div>
    <div class="uml-section">
      <div class="uml-item">+ operation() → Tensor</div>
    </div>
  </div>
  <div class="uml-box">
    <div class="uml-title">ConcreteProductB</div>
    <div class="uml-section">
      <div class="uml-item">+ operation() → Tensor</div>
    </div>
  </div>
</div>
</div>

---

## Factory Method in ML: Model Factory

The Factory Method pattern in ML most commonly appears as a **model factory** — a framework class that defines how to create a model, but defers the choice of architecture to a subclass or a registry.

```python
from __future__ import annotations

import abc
from typing import Any

import torch
import torch.nn as nn


# ── Abstract Product ──────────────────────────────────────────────────────────

class BaseModel(nn.Module, abc.ABC):
    """Abstract product: any model that the factory produces must be an nn.Module."""

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def get_feature_dim(self) -> int:
        """Return the dimensionality of the penultimate feature representation."""
        ...

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Concrete Products ─────────────────────────────────────────────────────────

class MLPModel(BaseModel):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self._feature_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_feature_dim(self) -> int:
        return self._feature_dim


class CNNModel(BaseModel):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))

    def get_feature_dim(self) -> int:
        return 128 * 4 * 4


class TransformerEncoderModel(BaseModel):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, num_classes: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        self._d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        enc = self.encoder(emb)
        return self.classifier(enc[:, 0, :])  # CLS token

    def get_feature_dim(self) -> int:
        return self._d_model


# ── Abstract Creator (Factory Method) ────────────────────────────────────────

class ModelFactory(abc.ABC):
    """
    Abstract Creator.
    Subclasses override `create_model` to instantiate the correct architecture.
    """

    @abc.abstractmethod
    def create_model(self, **kwargs: Any) -> BaseModel:
        """Factory method: create and return a BaseModel."""
        ...

    def build_for_training(self, **kwargs: Any) -> BaseModel:
        """
        Template method using the factory method.
        Creates a model and applies common initialisation.
        """
        model = self.create_model(**kwargs)
        self._init_weights(model)
        print(f"Created {model.__class__.__name__} with {model.num_parameters():,} params")
        return model

    @staticmethod
    def _init_weights(model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


# ── Concrete Creators ─────────────────────────────────────────────────────────

class MLPFactory(ModelFactory):
    def create_model(self, **kwargs: Any) -> BaseModel:
        return MLPModel(**kwargs)


class CNNFactory(ModelFactory):
    def create_model(self, **kwargs: Any) -> BaseModel:
        return CNNModel(**kwargs)


class TransformerFactory(ModelFactory):
    def create_model(self, **kwargs: Any) -> BaseModel:
        return TransformerEncoderModel(**kwargs)


# ── Usage ─────────────────────────────────────────────────────────────────────

factory_map: dict[str, ModelFactory] = {
    "mlp": MLPFactory(),
    "cnn": CNNFactory(),
    "transformer": TransformerFactory(),
}

configs = [
    ("mlp", {"input_dim": 784, "hidden_dim": 256, "output_dim": 10}),
    ("cnn", {"in_channels": 3, "num_classes": 10}),
    ("transformer", {"vocab_size": 30522, "d_model": 256, "nhead": 8, "num_layers": 4, "num_classes": 2}),
]

for arch, kwargs in configs:
    model = factory_map[arch].build_for_training(**kwargs)
```

---

## The Model Registry Pattern: Factory + Registry

In practice, the most powerful ML factory pattern combines Factory with a **Registry**: a dictionary mapping string names to constructors, populated via a `@register` decorator. This is the pattern used by Detectron2, MMDetection, Hugging Face Transformers, and PyTorch Image Models (timm).

```python
from __future__ import annotations

import inspect
from typing import Any, Callable, TypeVar

import torch.nn as nn

T = TypeVar("T", bound=type)


class ModelRegistry:
    """
    A registry that maps string names to model classes.
    Supports registration via a decorator and creation via a factory method.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: dict[str, type[nn.Module]] = {}

    def register(self, name: str | None = None) -> Callable[[T], T]:
        """
        Decorator to register a model class under an optional name.
        If no name is given, the class name (lowercased) is used.

        Usage::

            @registry.register("resnet18")
            class ResNet18(nn.Module): ...

            @registry.register()
            class MyCustomModel(nn.Module): ...
        """
        def decorator(cls: T) -> T:
            key = name if name is not None else cls.__name__.lower()
            if key in self._registry:
                raise KeyError(
                    f"Model '{key}' already registered in {self.name} registry"
                )
            self._registry[key] = cls
            return cls

        return decorator

    def create(self, name: str, **kwargs: Any) -> nn.Module:
        """
        Instantiate a registered model by name.

        Args:
            name:     Registry key.
            **kwargs: Forwarded to the model constructor.

        Raises:
            KeyError: When the name is not registered.
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry))
            raise KeyError(
                f"'{name}' not found in {self.name} registry. "
                f"Available: {available}"
            )
        return self._registry[name](**kwargs)

    def list_models(self) -> list[str]:
        return sorted(self._registry)

    def get_class(self, name: str) -> type[nn.Module]:
        if name not in self._registry:
            raise KeyError(f"'{name}' not found in {self.name} registry")
        return self._registry[name]

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        return f"ModelRegistry(name={self.name!r}, models={self.list_models()})"


# ── Instantiate a global registry ────────────────────────────────────────────

MODEL_REGISTRY = ModelRegistry("models")


# ── Register models using the decorator ──────────────────────────────────────

@MODEL_REGISTRY.register("mlp")
class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


@MODEL_REGISTRY.register("wide_mlp")
class WideMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# Third-party / user-defined models register themselves the same way
@MODEL_REGISTRY.register("custom_net")
class CustomNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


# ── Usage: configuration-driven model creation ────────────────────────────────

def build_model_from_config(config: dict) -> nn.Module:
    """Create a model entirely from a config dict."""
    arch = config.pop("architecture")
    return MODEL_REGISTRY.create(arch, **config)


print(MODEL_REGISTRY)
model = build_model_from_config({
    "architecture": "mlp",
    "input_dim": 784,
    "hidden_dim": 512,
    "output_dim": 10,
})
```

---

## Abstract Factory: Intent and UML

The **Abstract Factory** pattern provides an interface for creating *families* of related or dependent objects without specifying their concrete classes. Where Factory Method creates a single product, Abstract Factory creates a *suite* of products that are designed to work together.

In ML, this applies whenever you need to create multiple coordinated components — a dataset *and* its transforms *and* its collate function *and* its data loader must all be consistent with each other. An abstract factory ensures they are.

<div class="diagram">
<div class="diagram-title">Abstract Factory — Structure</div>
<div class="uml-row">
  <div class="uml-box">
    <div class="uml-title">«abstract» DataPipelineFactory</div>
    <div class="uml-section">
      <div class="uml-item">+ create_dataset() → Dataset</div>
      <div class="uml-item">+ create_transform() → Transform</div>
      <div class="uml-item">+ create_dataloader() → DataLoader</div>
    </div>
  </div>
</div>
<div class="uml-row">
  <div class="uml-box">
    <div class="uml-title">ImageNetPipelineFactory</div>
    <div class="uml-section">
      <div class="uml-item">+ create_dataset() → ImageNetDataset</div>
      <div class="uml-item">+ create_transform() → ImageNetTransform</div>
      <div class="uml-item">+ create_dataloader() → ImageNetDataLoader</div>
    </div>
  </div>
  <div class="uml-box">
    <div class="uml-title">CIFAR10PipelineFactory</div>
    <div class="uml-section">
      <div class="uml-item">+ create_dataset() → CIFAR10Dataset</div>
      <div class="uml-item">+ create_transform() → CIFAR10Transform</div>
      <div class="uml-item">+ create_dataloader() → CIFAR10DataLoader</div>
    </div>
  </div>
</div>
</div>

---

## Abstract Factory in ML: Data Pipeline Factory

```python
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# ── Abstract Products ─────────────────────────────────────────────────────────

class AugmentTransform(abc.ABC):
    @abc.abstractmethod
    def train_transform(self) -> Any:
        ...

    @abc.abstractmethod
    def eval_transform(self) -> Any:
        ...


# ── Abstract Factory ──────────────────────────────────────────────────────────

class DataPipelineFactory(abc.ABC):
    """
    Abstract Factory: creates a consistent family of data pipeline components.
    All products produced by the same factory are designed to work together.
    """

    @abc.abstractmethod
    def create_train_dataset(self, root: str) -> Dataset:
        ...

    @abc.abstractmethod
    def create_val_dataset(self, root: str) -> Dataset:
        ...

    @abc.abstractmethod
    def create_transform(self) -> AugmentTransform:
        ...

    def create_train_loader(
        self,
        root: str,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> DataLoader:
        """Template method: uses the abstract factory methods."""
        transform = self.create_transform()
        dataset = self.create_train_dataset(root)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def create_val_loader(
        self,
        root: str,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> DataLoader:
        transform = self.create_transform()
        dataset = self.create_val_dataset(root)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )


# ── Concrete Factories ────────────────────────────────────────────────────────

class CIFAR10Transform(AugmentTransform):
    def train_transform(self):
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def eval_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


class CIFAR10PipelineFactory(DataPipelineFactory):
    """Concrete factory that produces a consistent CIFAR-10 data pipeline."""

    def create_transform(self) -> AugmentTransform:
        return CIFAR10Transform()

    def create_train_dataset(self, root: str) -> Dataset:
        t = self.create_transform().train_transform()
        return datasets.CIFAR10(root, train=True, transform=t, download=True)

    def create_val_dataset(self, root: str) -> Dataset:
        t = self.create_transform().eval_transform()
        return datasets.CIFAR10(root, train=False, transform=t, download=True)


class ImageNetTransform(AugmentTransform):
    def train_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def eval_transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


class ImageNetPipelineFactory(DataPipelineFactory):
    """Concrete factory for the ImageNet data pipeline."""

    def create_transform(self) -> AugmentTransform:
        return ImageNetTransform()

    def create_train_dataset(self, root: str) -> Dataset:
        t = self.create_transform().train_transform()
        return datasets.ImageFolder(f"{root}/train", transform=t)

    def create_val_dataset(self, root: str) -> Dataset:
        t = self.create_transform().eval_transform()
        return datasets.ImageFolder(f"{root}/val", transform=t)


# ── Client code — depends only on the abstract factory ───────────────────────

def run_experiment(
    factory: DataPipelineFactory,
    data_root: str,
    batch_size: int = 32,
) -> None:
    """
    The client depends only on the abstract DataPipelineFactory.
    Swap CIFAR10PipelineFactory for ImageNetPipelineFactory — nothing else changes.
    """
    train_loader = factory.create_train_loader(data_root, batch_size)
    val_loader = factory.create_val_loader(data_root, batch_size)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")


# Select factory from config
PIPELINE_FACTORIES = {
    "cifar10": CIFAR10PipelineFactory,
    "imagenet": ImageNetPipelineFactory,
}

factory_cls = PIPELINE_FACTORIES["cifar10"]
factory = factory_cls()
# run_experiment(factory, data_root="./data", batch_size=64)
```

---

## Loss Factory: Creating Loss Functions from Config

The loss function is another prime candidate for a factory. Experiments swap losses frequently — cross-entropy to focal loss to label-smoothed cross-entropy. A loss factory makes this configuration-driven.

```python
from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Custom loss implementations ───────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss for class-imbalanced classification.
    Lin et al. (2017), "Focal Loss for Dense Object Detection".
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing regularisation."""

    def __init__(self, smoothing: float = 0.1, num_classes: int = 10) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=-1)
        # Hard targets
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        # Smooth targets
        smooth = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll + self.smoothing * smooth
        return loss.mean()


class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(inputs)
        targets_f = targets.float()
        intersection = (probs * targets_f).sum()
        return 1 - (2 * intersection + self.eps) / (probs.sum() + targets_f.sum() + self.eps)


# ── Loss registry and factory ─────────────────────────────────────────────────

_LOSS_REGISTRY: dict[str, type[nn.Module]] = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCEWithLogitsLoss,
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "huber": nn.HuberLoss,
    "focal": FocalLoss,
    "label_smoothing": LabelSmoothingCrossEntropy,
    "dice": DiceLoss,
}


def create_loss(loss_type: str, **kwargs: Any) -> nn.Module:
    """
    Factory function: create a loss module from a string name and kwargs.

    Example::

        loss = create_loss("focal", alpha=0.5, gamma=2.0)
        loss = create_loss("label_smoothing", smoothing=0.05, num_classes=1000)
        loss = create_loss("cross_entropy")
    """
    if loss_type not in _LOSS_REGISTRY:
        available = ", ".join(sorted(_LOSS_REGISTRY))
        raise ValueError(f"Unknown loss '{loss_type}'. Available: {available}")
    return _LOSS_REGISTRY[loss_type](**kwargs)


def register_loss(name: str) -> Callable[[type], type]:
    """Decorator to register a new loss class."""
    def decorator(cls: type) -> type:
        _LOSS_REGISTRY[name] = cls
        return cls
    return decorator


# Register a custom loss without touching the registry code
@register_loss("combined_focal_dice")
class CombinedFocalDiceLoss(nn.Module):
    def __init__(self, focal_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.fw = focal_weight
        self.dw = dice_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.fw * self.focal(inputs, targets) + self.dw * self.dice(inputs, targets)


# ── Usage ─────────────────────────────────────────────────────────────────────

loss_configs = [
    {"type": "cross_entropy"},
    {"type": "focal", "alpha": 0.25, "gamma": 2.0},
    {"type": "label_smoothing", "smoothing": 0.1, "num_classes": 10},
    {"type": "combined_focal_dice", "focal_weight": 0.6, "dice_weight": 0.4},
]

for cfg in loss_configs:
    loss_type = cfg.pop("type")
    loss_fn = create_loss(loss_type, **cfg)
    print(f"  {loss_type}: {loss_fn}")
```

---

## Flow: Config to Concrete Object

<div class="diagram">
<div class="diagram-title">Configuration-Driven Object Creation</div>
<div class="flow">
  <div class="flow-node accent wide">YAML / JSON / Python Dict Config</div>
  <div class="flow-arrow">↓ parse</div>
  <div class="flow-node blue wide">Config Object (dataclass / OmegaConf)</div>
  <div class="flow-arrow">↓ extract "type" key</div>
  <div class="flow-node green wide">Factory / Registry Lookup</div>
  <div class="flow-arrow">↓ instantiate</div>
  <div class="flow-h">
    <div class="flow-node purple narrow">ResNet50<br/><small>nn.Module</small></div>
    <div class="flow-node purple narrow">ViT-Base<br/><small>nn.Module</small></div>
    <div class="flow-node purple narrow">BERT<br/><small>nn.Module</small></div>
    <div class="flow-node purple narrow">CustomNet<br/><small>nn.Module</small></div>
  </div>
  <div class="flow-arrow">↓ return</div>
  <div class="flow-node teal wide">Concrete Object (opaque to calling code)</div>
</div>
</div>

---

## Configuration-Driven Instantiation: Hydra + Factory

[Hydra](https://hydra.cc) by Facebook Research is the standard configuration management framework for ML. It uses `_target_` keys to specify the class to instantiate — essentially a first-class Factory built into the config system.

```python
from __future__ import annotations

# ── config/model/mlp.yaml ─────────────────────────────────────────────────────
# _target_: mypackage.models.SimpleMLP
# input_dim: 784
# hidden_dim: 256
# output_dim: 10

# ── config/model/cnn.yaml ─────────────────────────────────────────────────────
# _target_: mypackage.models.CNNModel
# in_channels: 3
# num_classes: 10

# ── Using Hydra's instantiate as a factory ────────────────────────────────────

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def build_from_hydra_config(cfg: DictConfig) -> object:
    """
    Hydra's instantiate reads '_target_' and creates the object.
    This is the standard Factory pattern expressed in YAML.
    """
    return instantiate(cfg.model)


# ── Manual equivalent without Hydra ──────────────────────────────────────────

import importlib
from typing import Any


def hydra_style_instantiate(config: dict[str, Any]) -> Any:
    """
    Pure-Python equivalent of Hydra's instantiate.
    Reads '_target_' as 'module.ClassName', imports it, and creates an instance.
    """
    config = dict(config)
    target = config.pop("_target_")
    module_path, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**config)


# ── Example usage ─────────────────────────────────────────────────────────────

model_configs = [
    {
        "_target_": "torch.nn.Linear",
        "in_features": 784,
        "out_features": 10,
    },
    {
        "_target_": "torch.nn.TransformerEncoderLayer",
        "d_model": 512,
        "nhead": 8,
        "batch_first": True,
    },
]

for cfg in model_configs:
    obj = hydra_style_instantiate(cfg)
    print(f"Created: {obj.__class__.__name__}")
```

---

## When to Use (and Not Use) the Factory

<div class="callout tip">
<strong>✅ Use Factory When:</strong><br/><br/>
• You don't know at design time exactly which type of object to create (it depends on runtime config).<br/>
• You want to centralise creation logic and remove if/elif chains from calling code.<br/>
• You want calling code to be closed for modification when new types are added.<br/>
• Different environments (dev, staging, prod) require different object implementations.<br/>
• You're building a framework or library where users will extend the type system with their own classes.
</div>

<div class="callout warn">
<strong>⚠️ Don't Use Factory When:</strong><br/><br/>
• There is only one concrete type and the code will never need to support more — direct instantiation is cleaner.<br/>
• The construction logic is trivial (one or two lines) — a factory is unnecessary indirection.<br/>
• The type is truly fixed at compile/import time and will never change — factories add complexity for no benefit.<br/>
• You're early in a prototype — add the factory when you observe the need, not speculatively.
</div>

---

**Next: [Chapter 5 — Builder Pattern →](./05_builder.md)**

*Last updated: May 2026*
