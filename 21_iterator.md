---
title: "Chapter 21 — Iterator, Generator & Lazy Datasets"
---

[← Back to Table of Contents](./README.md)

# Chapter 21 — Iterator, Generator & Lazy Datasets

> *"Don't iterate over data — let data flow through your pipeline."*

<span class="badge mlops">MLOps</span> <span class="badge pytorch">PyTorch</span>

---

## 21.1 Intent

The **Iterator** pattern provides a way to access elements of a collection sequentially without exposing the underlying representation. In ML pipelines this maps directly onto how we feed data to models: we want uniform access to batches regardless of whether data lives on disk, in memory, behind an API, or in a distributed object store.

The pattern separates two concerns:

- **What** the collection contains (dataset logic)
- **How** to traverse it (iterator logic)

<div class="diagram">
  <div class="diagram-title">Iterator Pattern — Core Structure</div>
  <div class="flow">
    <div class="flow-node accent">Client</div>
    <div class="flow-arrow">uses</div>
    <div class="flow-node blue wide">«interface» Iterator<br/><small>__iter__ / __next__</small></div>
    <div class="flow-arrow">←implements</div>
    <div class="flow-node green wide">ConcreteIterator<br/><small>DataLoader / Generator</small></div>
  </div>
  <div class="flow">
    <div class="flow-node purple wide">«interface» Iterable<br/><small>__iter__ → Iterator</small></div>
    <div class="flow-arrow">←implements</div>
    <div class="flow-node orange wide">ConcreteIterable<br/><small>Dataset / WebDataset</small></div>
  </div>
</div>

---

## 21.2 Python's Iterator Protocol

Python formalises iteration with two dunder methods:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `__iter__` | `self → Iterator` | Return the iterator object (often `self`) |
| `__next__` | `self → T` | Return next element or raise `StopIteration` |

```python
class CountUp:
    """Simple iterator that counts from start to stop."""

    def __init__(self, start: int, stop: int) -> None:
        self.current = start
        self.stop = stop

    def __iter__(self) -> "CountUp":
        return self

    def __next__(self) -> int:
        if self.current >= self.stop:
            raise StopIteration
        value = self.current
        self.current += 1
        return value


# Usage
for n in CountUp(0, 5):
    print(n)  # 0 1 2 3 4

# Equivalent manual iteration
it = CountUp(0, 3)
print(next(it))  # 0
print(next(it))  # 1
print(next(it))  # 2
# next(it) → StopIteration
```

An **iterable** implements only `__iter__` (returning a fresh iterator each time). An **iterator** implements both — it is stateful and can only be traversed once.

```python
class NumberRange:
    """Iterable (not iterator) — produces a fresh iterator each call."""

    def __init__(self, start: int, stop: int) -> None:
        self.start = start
        self.stop = stop

    def __iter__(self) -> CountUp:
        return CountUp(self.start, self.stop)


r = NumberRange(0, 5)
print(list(r))  # [0, 1, 2, 3, 4]
print(list(r))  # [0, 1, 2, 3, 4]  — reusable!
```

---

## 21.3 Generators as Iterators

Python generators are the most ergonomic way to implement the iterator protocol. A function containing `yield` becomes a generator function; calling it returns a generator object that implements `__iter__` and `__next__` automatically.

```python
from typing import Iterator
import random

def epoch_shuffler(data: list, seed: int = 42) -> Iterator:
    """Yield items in a shuffled order — new shuffle each call."""
    indices = list(range(len(data)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    for i in indices:
        yield data[i]


samples = ["img_001.jpg", "img_002.jpg", "img_003.jpg", "img_004.jpg"]
for item in epoch_shuffler(samples):
    print(item)
```

### `yield from` — Delegating to Sub-Iterators

```python
def multi_epoch(data: list, epochs: int) -> Iterator:
    """Stream multiple epochs without materialising all data."""
    for epoch in range(epochs):
        yield from epoch_shuffler(data, seed=epoch)
```

### Generator Expressions

```python
import torch

# Lazy tensor creation — nothing allocated until consumed
pixel_tensors = (
    torch.tensor(pixels, dtype=torch.float32) / 255.0
    for pixels in raw_pixel_batches
)

# Only the current batch lives in memory
for batch in pixel_tensors:
    model(batch)
```

---

## 21.4 DataLoader Internals as Iterator

PyTorch's `DataLoader` is the canonical example of the Iterator pattern in ML. Understanding its internals helps you customise and debug it.

<div class="diagram">
  <div class="diagram-title">DataLoader Iteration Pipeline</div>
  <div class="flow">
    <div class="flow-node accent wide">DataLoader<br/><small>__iter__</small></div>
    <div class="flow-arrow accent">→</div>
    <div class="flow-node blue">Sampler<br/><small>indices</small></div>
    <div class="flow-arrow accent">→</div>
    <div class="flow-node green wide">Dataset<br/><small>__getitem__</small></div>
    <div class="flow-arrow accent">→</div>
    <div class="flow-node purple">Collate<br/><small>fn</small></div>
    <div class="flow-arrow accent">→</div>
    <div class="flow-node orange">Prefetch<br/><small>queue</small></div>
    <div class="flow-arrow accent">→</div>
    <div class="flow-node teal">Batch<br/><small>Tensor</small></div>
  </div>
</div>

```python
from torch.utils.data import DataLoader, Dataset, Sampler
from typing import Iterator, List
import torch

class SimpleDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.data = torch.randn(size, 3, 224, 224)
        self.labels = torch.randint(0, 1000, (size,))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]
```

### Simplified DataLoader Re-implementation

```python
from torch.utils.data import Dataset, Sampler, RandomSampler, default_collate
from typing import Callable, Iterator, Optional
import torch

class MinimalDataLoader:
    """Stripped-down DataLoader showing the iterator pattern clearly."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = False,
        collate_fn: Optional[Callable] = None,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn or default_collate
        self.drop_last = drop_last

    def __iter__(self) -> Iterator:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            import random
            random.shuffle(indices)

        batch: list = []
        for idx in indices:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []

        if batch and not self.drop_last:
            yield self.collate_fn(batch)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


# Usage
ds = SimpleDataset(1000)
loader = MinimalDataLoader(ds, batch_size=32, shuffle=True)
for images, labels in loader:
    print(images.shape, labels.shape)
    break  # torch.Size([32, 3, 224, 224]) torch.Size([32])
```

---

## 21.5 IterableDataset vs Dataset

| Criterion | `Dataset` (map-style) | `IterableDataset` | `WebDataset` |
|-----------|----------------------|-------------------|--------------|
| Indexing | Random access `__getitem__` | Sequential only | Sequential only |
| Shuffling | Full shuffle via sampler | Limited / approx | Shard-level shuffle |
| Multi-process | Simple — split indices | Requires worker splitting | Built-in shard split |
| Streaming | No — needs full materialisation | Yes | Yes |
| Best for | Local datasets that fit on disk | Streaming / infinite data | Large sharded cloud data |
| Length known? | Yes — `__len__` required | Optional | No |

```python
from torch.utils.data import IterableDataset
import math

class ChunkedCSVDataset(IterableDataset):
    """Stream rows from a large CSV without loading it all into memory."""

    def __init__(self, filepath: str, chunk_size: int = 1000) -> None:
        self.filepath = filepath
        self.chunk_size = chunk_size

    def __iter__(self):
        import pandas as pd
        worker_info = torch.utils.data.get_worker_info()
        reader = pd.read_csv(self.filepath, chunksize=self.chunk_size)

        for chunk_idx, chunk in enumerate(reader):
            if worker_info is not None:
                # Distribute chunks across workers
                if chunk_idx % worker_info.num_workers != worker_info.id:
                    continue
            for _, row in chunk.iterrows():
                yield torch.tensor(row.values, dtype=torch.float32)
```

---

## 21.6 Streaming Dataset Iterator — Cloud Storage

```python
import io
from typing import Iterator
import boto3
import torch
from torch.utils.data import IterableDataset

class S3StreamingDataset(IterableDataset):
    """Lazily stream samples from S3 without downloading entire files."""

    def __init__(
        self,
        bucket: str,
        prefix: str,
        transform=None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix
        self.transform = transform
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = boto3.client("s3")
        return self._client

    def _list_keys(self) -> list[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def __iter__(self) -> Iterator:
        for key in self._list_keys():
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            data = torch.load(io.BytesIO(response["Body"].read()))
            if self.transform:
                data = self.transform(data)
            yield data
```

---

## 21.7 WebDataset — Sharded TAR Streaming

```python
# pip install webdataset
import webdataset as wds
import torchvision.transforms as T

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Shards can be local paths or gs:// / s3:// URIs
dataset = (
    wds.WebDataset("gs://my-bucket/imagenet/train-{000000..001281}.tar")
    .shuffle(1000)           # buffer-based approximate shuffle
    .decode("pil")           # decode JPEG → PIL Image
    .to_tuple("jpg", "cls")  # extract fields
    .map_tuple(transform, int)
    .batched(32)
)

loader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=None)
for images, labels in loader:
    print(images.shape)  # [32, 3, 224, 224]
    break
```

---

## 21.8 Infinite Data Iterators

```python
import itertools
from typing import Iterator, TypeVar
T = TypeVar("T")

def infinite_cycle(dataset, shuffle_each_epoch: bool = True) -> Iterator:
    """Cycle through a dataset infinitely, optionally reshuffling."""
    import random
    indices = list(range(len(dataset)))
    while True:
        if shuffle_each_epoch:
            random.shuffle(indices)
        for idx in indices:
            yield dataset[idx]


def take(iterator: Iterator[T], n: int) -> list[T]:
    return list(itertools.islice(iterator, n))


# Training loop that doesn't care about epoch boundaries
ds = SimpleDataset(10000)
stream = infinite_cycle(ds, shuffle_each_epoch=True)

for step in range(100_000):
    sample = next(stream)
    # train on sample ...
```

---

## 21.9 Pipeline Iterator Composition

Composing iterators into processing pipelines keeps each transformation isolated and testable.

```python
from typing import Callable, Iterator, TypeVar, Generic
T = TypeVar("T")
U = TypeVar("U")

class MapIterator(Generic[T, U]):
    """Apply a transformation to each element."""

    def __init__(self, source: Iterator[T], fn: Callable[[T], U]) -> None:
        self.source = source
        self.fn = fn

    def __iter__(self) -> "MapIterator":
        return self

    def __next__(self) -> U:
        return self.fn(next(self.source))


class FilterIterator(Generic[T]):
    """Skip elements that don't satisfy a predicate."""

    def __init__(self, source: Iterator[T], predicate: Callable[[T], bool]) -> None:
        self.source = source
        self.predicate = predicate

    def __iter__(self) -> "FilterIterator":
        return self

    def __next__(self) -> T:
        while True:
            item = next(self.source)  # propagates StopIteration
            if self.predicate(item):
                return item


class BatchIterator(Generic[T]):
    """Group elements into batches."""

    def __init__(self, source: Iterator[T], batch_size: int, drop_last: bool = False) -> None:
        self.source = source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self._exhausted = False

    def __iter__(self) -> "BatchIterator":
        return self

    def __next__(self) -> list[T]:
        if self._exhausted:
            raise StopIteration
        batch = []
        try:
            while len(batch) < self.batch_size:
                batch.append(next(self.source))
        except StopIteration:
            self._exhausted = True
        if not batch or (self.drop_last and len(batch) < self.batch_size):
            raise StopIteration
        return batch


class PrefetchIterator(Generic[T]):
    """Prefetch items in a background thread."""

    def __init__(self, source: Iterator[T], buffer_size: int = 8) -> None:
        import queue
        import threading
        self.queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._sentinel = object()

        def _worker():
            for item in source:
                self.queue.put(item)
            self.queue.put(self._sentinel)

        threading.Thread(target=_worker, daemon=True).start()

    def __iter__(self) -> "PrefetchIterator":
        return self

    def __next__(self) -> T:
        item = self.queue.get()
        if item is self._sentinel:
            raise StopIteration
        return item


# Composing a full pipeline
import torch

raw_stream = iter(range(10000))
pipeline = (
    MapIterator(raw_stream, lambda x: torch.tensor(x, dtype=torch.float32))
)
filtered = FilterIterator(pipeline, lambda t: t % 2 == 0)
batched = BatchIterator(filtered, batch_size=16)
prefetched = PrefetchIterator(batched, buffer_size=4)

for batch in prefetched:
    stacked = torch.stack(batch)
    print(stacked.shape)  # torch.Size([16])
    break
```

---

## 21.10 `itertools` Patterns for ML

```python
import itertools
from typing import Iterator
import torch

# --- Chain multiple datasets ---
def chain_datasets(*datasets) -> Iterator:
    """Iterate through multiple datasets sequentially."""
    return itertools.chain.from_iterable(datasets)


# --- Cross-product for hyperparameter search ---
def hyperparam_grid(**param_grid) -> Iterator[dict]:
    """Yield all combinations of hyperparameters."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


for params in hyperparam_grid(lr=[1e-3, 1e-4], batch_size=[16, 32], dropout=[0.1, 0.3]):
    print(params)
# {'lr': 0.001, 'batch_size': 16, 'dropout': 0.1}
# ... 7 more combinations


# --- islice: take first N batches for smoke-testing ---
ds = SimpleDataset(10000)
loader = torch.utils.data.DataLoader(ds, batch_size=32)
for images, labels in itertools.islice(loader, 5):
    print(f"Smoke test batch: {images.shape}")


# --- cycle: repeat small dataset for curriculum learning ---
tiny_ds = SimpleDataset(100)
tiny_loader = torch.utils.data.DataLoader(tiny_ds, batch_size=8)
curriculum = itertools.cycle(tiny_loader)
for step, (imgs, lbls) in enumerate(itertools.islice(curriculum, 50)):
    print(f"Step {step}: {imgs.shape}")
```

---

## 21.11 Lazy Evaluation for Large-Scale Data

### Memory-Mapped Arrays

```python
import numpy as np
import torch
from pathlib import Path

class MemmapDataset(torch.utils.data.Dataset):
    """Access huge arrays on disk without loading them entirely into RAM."""

    def __init__(self, data_path: str, label_path: str) -> None:
        meta = np.load(data_path.replace(".npy", "_meta.npz"))
        self.shape = tuple(meta["shape"])
        self.dtype = np.dtype(str(meta["dtype"]))

        # Memory-map: OS pages data from disk on demand
        self.data = np.memmap(data_path, dtype=self.dtype, mode="r", shape=self.shape)
        self.labels = np.memmap(label_path, dtype=np.int64, mode="r",
                                shape=(self.shape[0],))

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx: int):
        # Only the requested pages are loaded from disk
        x = torch.from_numpy(self.data[idx].copy()).float()
        y = int(self.labels[idx])
        return x, y


def create_memmap_dataset(data: np.ndarray, labels: np.ndarray, path: str) -> None:
    """Save arrays in memory-mappable format."""
    fp = np.memmap(path, dtype=data.dtype, mode="w+", shape=data.shape)
    fp[:] = data[:]
    np.savez(path.replace(".npy", "_meta.npz"),
             shape=data.shape, dtype=str(data.dtype))
```

### Arrow / Parquet Lazy Streaming

```python
# pip install pyarrow datasets
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset

class ParquetStreamDataset(IterableDataset):
    """Stream rows from a Parquet file using Arrow's batch reader."""

    def __init__(self, parquet_path: str, batch_size: int = 256,
                 feature_cols: list[str] = None) -> None:
        self.path = parquet_path
        self.batch_size = batch_size
        self.feature_cols = feature_cols

    def __iter__(self):
        pf = pq.ParquetFile(self.path)
        cols = self.feature_cols

        for batch in pf.iter_batches(batch_size=self.batch_size, columns=cols):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                yield torch.tensor(row.values, dtype=torch.float32)
```

---

## 21.12 Comparison Table

| Feature | `Dataset` (map) | `IterableDataset` | `WebDataset` | HuggingFace `datasets` |
|---------|----------------|-------------------|--------------|------------------------|
| Random access | ✅ O(1) | ❌ | ❌ | ✅ (with index) |
| True streaming | ❌ | ✅ | ✅ | ✅ |
| Distributed shuffle | Via sampler | Manual | Shard shuffle | Built-in |
| Cloud storage | Manual | Manual | Native (gs/s3) | Native |
| Multi-worker | Simple | Worker split needed | Auto-split | Auto-split |
| Memory usage | Dataset-size | O(buffer) | O(buffer) | O(buffer) |
| Caching | Manual | Manual | Manual | Automatic |
| Best scale | < 1 TB local | Any | > 1 TB sharded | Research datasets |

<div class="callout tip">
<strong>Tip:</strong> For datasets that exceed RAM, start with <code>IterableDataset</code> and a local shuffle buffer. Move to WebDataset only when you need cross-machine shard distribution or want the tar-based ecosystem tools.
</div>

---

## Summary

<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">🔄</div>
    <div class="card-title">Protocol</div>
    <div class="card-desc"><code>__iter__</code> + <code>__next__</code> — the minimal contract. Generators implement it for free via <code>yield</code>.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">📦</div>
    <div class="card-title">DataLoader</div>
    <div class="card-desc">Sampler → Dataset → Collate → Prefetch. Each stage is a composable iterator.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">☁️</div>
    <div class="card-title">Lazy & Streaming</div>
    <div class="card-desc">memmap, Parquet, WebDataset — never load more than a buffer. Scale to terabytes.</div>
  </div>
</div>

*Last updated: May 2026*
