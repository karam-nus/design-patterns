---
title: "Chapter 28 — Data Pipeline & DAG Patterns"
---

[← Back to Table of Contents](./README.md)

# Chapter 28 — Data Pipeline & DAG Patterns

> *"A data pipeline without a DAG is a script waiting to become a mystery."*

<span class="badge mlops">MLOps</span>

---

## DAGs as the Fundamental Pipeline Pattern

A **Directed Acyclic Graph (DAG)** is the right abstraction for data pipelines because:

- **Explicit dependencies** — every node declares its inputs and outputs; no implicit state
- **Parallelism** — independent nodes can run concurrently
- **Reproducibility** — re-executing a node with the same inputs always gives the same output
- **Partial re-execution** — only invalidated nodes need to re-run when inputs change
- **Auditing** — the graph *is* the documentation of your data lineage

Sequential scripts have none of these properties. They are linear, stateful, and brittle.

<div class="diagram">
  <div class="diagram-title">DAG Data Pipeline</div>
  <div class="flow">
    <div class="flow-node teal extra-wide">Source<br><small>raw files, database, API</small></div>
    <div class="flow-arrow accent">↓</div>
    <div class="flow-h">
      <div class="flow-node blue wide">Ingest<br><small>download · deduplicate · hash</small></div>
      <div class="flow-arrow green">→</div>
      <div class="flow-node orange wide">Validate<br><small>schema · range checks · nulls</small></div>
    </div>
    <div class="flow-arrow accent">↓</div>
    <div class="flow-node purple extra-wide">Transform<br><small>tokenize · normalize · augment · featurize</small></div>
    <div class="flow-arrow accent">↓</div>
    <div class="flow-h">
      <div class="flow-node green wide">Train Split<br><small>80 % of data</small></div>
      <div class="flow-arrow accent">→</div>
      <div class="flow-node teal wide">Val Split<br><small>10 % of data</small></div>
      <div class="flow-arrow accent">→</div>
      <div class="flow-node blue narrow">Test Split<br><small>10 %</small></div>
    </div>
    <div class="flow-arrow accent">↓</div>
    <div class="flow-node accent extra-wide">Model Training</div>
  </div>
</div>

---

## DVC Pipelines

[DVC](https://dvc.org) (Data Version Control) expresses pipelines as a `dvc.yaml` file. Each stage declares its command, dependencies (`deps`), and outputs (`outs`). DVC tracks content hashes, so re-running only occurs when inputs change.

### `dvc.yaml`

```yaml
stages:

  ingest:
    cmd: python src/ingest.py --output data/raw
    deps:
      - src/ingest.py
      - config/sources.yaml
    outs:
      - data/raw

  validate:
    cmd: python src/validate.py --input data/raw --output data/validated
    deps:
      - src/validate.py
      - data/raw
    outs:
      - data/validated
    metrics:
      - reports/validation_report.json:
          cache: false

  transform:
    cmd: python src/transform.py --input data/validated --output data/features
    deps:
      - src/transform.py
      - data/validated
      - config/transform.yaml
    outs:
      - data/features

  split:
    cmd: python src/split.py --input data/features --output data/splits --seed 42
    deps:
      - src/split.py
      - data/features
    outs:
      - data/splits/train.parquet
      - data/splits/val.parquet
      - data/splits/test.parquet

  train:
    cmd: python src/train.py --data data/splits --output models/
    deps:
      - src/train.py
      - data/splits/train.parquet
      - data/splits/val.parquet
      - config/model.yaml
    outs:
      - models/best.safetensors
    metrics:
      - reports/metrics.json:
          cache: false
    plots:
      - reports/loss_curve.csv:
          cache: false
```

```python
# src/transform.py — a DVC stage implementation
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def normalize_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    for col in config.get("numeric_cols", []):
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std() + 1e-8
            df[col] = (df[col] - mean) / std
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config("config/transform.yaml")
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    records = []
    for fpath in sorted(input_path.glob("*.jsonl")):
        df = pd.read_json(fpath, lines=True)
        df = normalize_features(df, config)
        records.append(df)

    combined = pd.concat(records, ignore_index=True)
    combined.to_parquet(output_path / "features.parquet", index=False)
    print(f"Transformed {len(combined)} rows → {output_path / 'features.parquet'}")


if __name__ == "__main__":
    main()
```

**Running the pipeline:**

```bash
# Run entire pipeline (only executes changed stages)
dvc repro

# Force re-run a specific stage
dvc repro --force transform

# Visualize the DAG
dvc dag

# Push data to remote (S3, GCS, Azure)
dvc push
```

<div class="callout tip">
  <span class="callout-icon">💡</span>
  <div class="callout-body">Start with DVC for new ML projects. It integrates with Git, requires no external services, handles data versioning, and its pipeline syntax is the easiest to learn. Add a heavier orchestrator only when you need scheduling, retry logic, or multi-team coordination.</div>
</div>

---

## Kedro

[Kedro](https://kedro.org) is an opinionated framework that separates **nodes** (pure functions), **pipelines** (DAGs of nodes), and **data catalogs** (I/O declarations). This hard separation makes pipelines fully testable.

### `catalog.yml`

```yaml
# conf/base/catalog.yml

raw_data:
  type: pandas.CSVDataset
  filepath: data/01_raw/data.csv

validated_data:
  type: pandas.ParquetDataset
  filepath: data/02_intermediate/validated.parquet

features:
  type: pandas.ParquetDataset
  filepath: data/03_primary/features.parquet

train_set:
  type: pandas.ParquetDataset
  filepath: data/04_feature/train.parquet

val_set:
  type: pandas.ParquetDataset
  filepath: data/04_feature/val.parquet

model_metrics:
  type: tracking.MetricsDataset
  filepath: data/09_tracking/metrics.json
```

```python
# src/my_project/pipelines/data_processing/nodes.py
from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)


def validate_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, drop nulls, assert ranges."""
    required_cols = {"label", "feature_a", "feature_b", "feature_c"}
    missing = required_cols - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    before = len(raw)
    raw = raw.dropna(subset=list(required_cols))
    after = len(raw)
    logger.info(f"Dropped {before - after} rows with nulls ({after} remain)")

    assert raw["label"].between(0, 9).all(), "Labels out of range [0, 9]"
    return raw


def build_features(validated: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Normalize numeric features, add engineered features."""
    feature_cols = params["feature_cols"]
    for col in feature_cols:
        mu = validated[col].mean()
        sigma = validated[col].std() + 1e-8
        validated[f"{col}_norm"] = (validated[col] - mu) / sigma

    # Polynomial interaction
    validated["interaction_ab"] = validated["feature_a"] * validated["feature_b"]
    return validated


def split_data(
    features: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/val split."""
    from sklearn.model_selection import train_test_split

    train, val = train_test_split(
        features,
        test_size=params["val_fraction"],
        stratify=features["label"],
        random_state=params["random_seed"],
    )
    logger.info(f"Split: train={len(train)}, val={len(val)}")
    return train, val
```

```python
# src/my_project/pipelines/data_processing/pipeline.py
from kedro.pipeline import Pipeline, node, pipeline


def create_pipeline() -> Pipeline:
    return pipeline([
        node(
            func=validate_data,
            inputs="raw_data",
            outputs="validated_data",
            name="validate_node",
        ),
        node(
            func=build_features,
            inputs=["validated_data", "params:feature_engineering"],
            outputs="features",
            name="feature_node",
        ),
        node(
            func=split_data,
            inputs=["features", "params:split"],
            outputs=["train_set", "val_set"],
            name="split_node",
        ),
    ])
```

```bash
# Run the full pipeline
kedro run

# Run only the feature node
kedro run --nodes feature_node

# Visualize in browser
kedro viz
```

<div class="callout info">
  <span class="callout-icon">ℹ️</span>
  <div class="callout-body">Kedro's DataCatalog decouples your code from I/O. Swapping from local CSV to S3 Parquet only requires changing <code>catalog.yml</code> — no Python changes. This is the <strong>Adapter</strong> pattern applied to data sources.</div>
</div>

---

## Prefect

[Prefect](https://www.prefect.io) uses `@task` and `@flow` decorators. It supports dynamic task mapping, conditional branching, retries, and a cloud dashboard.

```python
# pip install prefect
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta


@task(
    retries=3,
    retry_delay_seconds=30,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
)
def download_shard(url: str, output_dir: str) -> str:
    """Download one data shard; cached by URL hash."""
    import urllib.request
    logger = get_run_logger()
    fname = hashlib.md5(url.encode()).hexdigest()[:8] + ".jsonl"
    dest = Path(output_dir) / fname
    if not dest.exists():
        logger.info(f"Downloading {url} → {dest}")
        urllib.request.urlretrieve(url, dest)
    return str(dest)


@task
def validate_shard(path: str) -> dict[str, Any]:
    """Validate one shard; return quality report."""
    logger = get_run_logger()
    df = pd.read_json(path, lines=True)
    report = {
        "path": path,
        "rows": len(df),
        "null_rate": df.isnull().mean().mean(),
        "passed": True,
    }
    if report["null_rate"] > 0.05:
        logger.warning(f"High null rate in {path}: {report['null_rate']:.2%}")
        report["passed"] = False
    return report


@task
def transform_shard(path: str, config: dict[str, Any]) -> str:
    """Transform one shard; write parquet alongside source."""
    df = pd.read_json(path, lines=True)
    for col in config.get("numeric_cols", []):
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
    out = path.replace(".jsonl", ".parquet")
    df.to_parquet(out, index=False)
    return out


@task
def merge_shards(shard_paths: list[str], output_path: str) -> str:
    """Merge transformed shards into a single dataset."""
    dfs = [pd.read_parquet(p) for p in shard_paths]
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_parquet(output_path, index=False)
    return output_path


@flow(name="ml-data-pipeline", log_prints=True)
def run_pipeline(
    shard_urls: list[str],
    raw_dir: str = "data/raw",
    output_path: str = "data/processed/dataset.parquet",
    transform_config: dict[str, Any] | None = None,
) -> str:
    transform_config = transform_config or {"numeric_cols": ["feature_a", "feature_b"]}
    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    # Download all shards concurrently (Prefect maps tasks in parallel)
    downloaded = download_shard.map(shard_urls, unmapped(raw_dir))

    # Validate each shard
    reports = validate_shard.map(downloaded)

    # Filter to only passed shards (conditional logic)
    good_paths = [
        path for path, report in zip(downloaded, reports)
        if report["passed"]
    ]
    print(f"{len(good_paths)}/{len(shard_urls)} shards passed validation")

    # Transform only the good shards
    transformed = transform_shard.map(good_paths, unmapped(transform_config))

    # Merge into final dataset
    final = merge_shards(transformed, output_path)
    return final


if __name__ == "__main__":
    from prefect import unmapped
    urls = [f"https://example.com/data/shard_{i:04d}.jsonl" for i in range(10)]
    run_pipeline(shard_urls=urls)
```

---

## Metaflow

[Metaflow](https://metaflow.org) (Netflix) models pipelines as Python classes with `@step` methods. The `self.next()` call declares the DAG topology. `@batch` runs steps on AWS Batch or Kubernetes.

```python
# pip install metaflow
from metaflow import FlowSpec, step, batch, Parameter, current
import pandas as pd
from pathlib import Path


class MLDataPipeline(FlowSpec):

    data_path = Parameter("data_path", help="Input data directory", default="data/raw")
    val_fraction = Parameter("val_fraction", help="Validation split fraction", default=0.1)
    random_seed = Parameter("random_seed", help="Random seed for splits", default=42)

    @step
    def start(self):
        """Entry point — list data files."""
        self.files = sorted(Path(self.data_path).glob("*.jsonl"))
        print(f"Found {len(self.files)} shards")
        self.next(self.ingest)

    @batch(cpu=4, memory=8192)            # run on cloud compute
    @step
    def ingest(self):
        """Load and deduplicate all shards."""
        dfs = [pd.read_json(str(f), lines=True) for f in self.files]
        self.raw_df = pd.concat(dfs, ignore_index=True).drop_duplicates()
        print(f"Ingested {len(self.raw_df)} unique rows")
        self.next(self.validate)

    @step
    def validate(self):
        """Validate schema and data quality."""
        df = self.raw_df
        required = {"label", "feature_a", "feature_b"}
        assert required.issubset(df.columns), f"Missing cols: {required - set(df.columns)}"
        null_mask = df[list(required)].isnull().any(axis=1)
        self.validated_df = df[~null_mask]
        self.validation_report = {
            "total_rows": len(df),
            "valid_rows": len(self.validated_df),
            "dropped_rows": int(null_mask.sum()),
        }
        print(self.validation_report)
        self.next(self.transform)

    @batch(cpu=8, memory=16384)
    @step
    def transform(self):
        """Normalize numeric features."""
        df = self.validated_df.copy()
        for col in ["feature_a", "feature_b"]:
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        self.transformed_df = df
        self.next(self.split)

    @step
    def split(self):
        """Stratified train/val split."""
        from sklearn.model_selection import train_test_split
        self.train_df, self.val_df = train_test_split(
            self.transformed_df,
            test_size=self.val_fraction,
            stratify=self.transformed_df["label"],
            random_state=self.random_seed,
        )
        print(f"Train: {len(self.train_df)}, Val: {len(self.val_df)}")
        self.next(self.end)

    @step
    def end(self):
        """Save splits and summarize."""
        Path("data/splits").mkdir(parents=True, exist_ok=True)
        self.train_df.to_parquet("data/splits/train.parquet", index=False)
        self.val_df.to_parquet("data/splits/val.parquet", index=False)
        print(f"Run ID: {current.run_id} — pipeline complete")


if __name__ == "__main__":
    MLDataPipeline()
```

```bash
# Run locally
python pipeline.py run --data_path data/raw

# Run on AWS Batch
python pipeline.py run --with batch

# Inspect past runs
python pipeline.py show
python pipeline.py list runs
```

---

## Airflow DAG

[Apache Airflow](https://airflow.apache.org) is the industry-standard scheduler for production data engineering. It provides a rich UI, retry policies, SLA monitoring, and connection management.

```python
# dags/ml_data_pipeline.py
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

import pandas as pd


def _ingest(**context) -> None:
    import urllib.request
    urls = context["params"]["shard_urls"]
    raw_dir = Path("/opt/airflow/data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    for i, url in enumerate(urls):
        urllib.request.urlretrieve(url, raw_dir / f"shard_{i:04d}.jsonl")
    context["ti"].xcom_push("num_shards", len(urls))


def _validate(**context) -> None:
    raw_dir = Path("/opt/airflow/data/raw")
    dfs = [pd.read_json(f, lines=True) for f in sorted(raw_dir.glob("*.jsonl"))]
    df = pd.concat(dfs, ignore_index=True)
    null_rate = df.isnull().mean().mean()
    context["ti"].xcom_push("null_rate", null_rate)
    context["ti"].xcom_push("num_rows", len(df))


def _branch_on_quality(**context) -> str:
    null_rate = context["ti"].xcom_pull("null_rate", task_ids="validate")
    return "transform" if null_rate < 0.1 else "quarantine"


def _quarantine(**context) -> None:
    null_rate = context["ti"].xcom_pull("null_rate", task_ids="validate")
    raise ValueError(f"Data quality check failed: null_rate={null_rate:.2%} > 10%")


def _transform(**context) -> None:
    raw_dir = Path("/opt/airflow/data/raw")
    out_dir = Path("/opt/airflow/data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    dfs = [pd.read_json(f, lines=True) for f in sorted(raw_dir.glob("*.jsonl"))]
    df = pd.concat(dfs, ignore_index=True).dropna()
    for col in ["feature_a", "feature_b"]:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
    df.to_parquet(out_dir / "features.parquet", index=False)


def _split(**context) -> None:
    from sklearn.model_selection import train_test_split
    df = pd.read_parquet("/opt/airflow/data/processed/features.parquet")
    train, val = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label"])
    split_dir = Path("/opt/airflow/data/splits")
    split_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(split_dir / "train.parquet", index=False)
    val.to_parquet(split_dir / "val.parquet", index=False)


default_args = {
    "owner": "ml-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email": ["ml-alerts@example.com"],
}

with DAG(
    dag_id="ml_data_pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    params={"shard_urls": []},
    tags=["ml", "data-pipeline"],
) as dag:

    ingest = PythonOperator(task_id="ingest", python_callable=_ingest)
    validate = PythonOperator(task_id="validate", python_callable=_validate)
    branch = BranchPythonOperator(task_id="quality_branch", python_callable=_branch_on_quality)
    quarantine = PythonOperator(task_id="quarantine", python_callable=_quarantine)
    transform = PythonOperator(task_id="transform", python_callable=_transform)
    split = PythonOperator(task_id="split", python_callable=_split)
    done = EmptyOperator(task_id="done", trigger_rule="none_failed_min_one_success")

    ingest >> validate >> branch >> [transform, quarantine]
    transform >> split >> done
    quarantine >> done
```

<div class="callout warn">
  <span class="callout-icon">⚠️</span>
  <div class="callout-body">Avoid Airflow for research and exploration pipelines. Its operational overhead (scheduler, workers, database, web server) is high. XCom-based data passing is fragile for large DataFrames. Use DVC or Prefect for ML experiments, and reserve Airflow for production ETL with strict scheduling requirements.</div>
</div>

---

## Incremental Processing

Processing only new or changed data is critical at scale — re-running the full pipeline every night is wasteful and slow.

```python
import hashlib
import json
from pathlib import Path
from datetime import datetime


class IncrementalProcessor:
    """Process only files that are new or have changed since last run."""

    def __init__(self, state_file: str = ".pipeline_state.json") -> None:
        self.state_file = Path(state_file)
        self._state: dict[str, str] = {}   # filepath → sha256 hash
        if self.state_file.exists():
            self._state = json.loads(self.state_file.read_text())

    def _file_hash(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def changed_files(self, input_dir: str, pattern: str = "*.jsonl") -> list[Path]:
        """Return files that are new or content-changed since last run."""
        changed = []
        for f in sorted(Path(input_dir).rglob(pattern)):
            current_hash = self._file_hash(f)
            if self._state.get(str(f)) != current_hash:
                changed.append(f)
        return changed

    def mark_processed(self, files: list[Path]) -> None:
        """Record the hash of each processed file."""
        for f in files:
            self._state[str(f)] = self._file_hash(f)
        self._state["_last_run"] = datetime.utcnow().isoformat()
        self.state_file.write_text(json.dumps(self._state, indent=2))

    def run(self, input_dir: str, output_dir: str) -> int:
        """Process only changed files, append to output."""
        changed = self.changed_files(input_dir)
        if not changed:
            print("No new or changed files — skipping run")
            return 0

        import pandas as pd
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        records = []
        for f in changed:
            df = pd.read_json(f, lines=True)
            df["source_file"] = f.name
            df["processed_at"] = datetime.utcnow().isoformat()
            records.append(df)

        batch_df = pd.concat(records, ignore_index=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = Path(output_dir) / f"batch_{ts}.parquet"
        batch_df.to_parquet(out_path, index=False)

        self.mark_processed(changed)
        print(f"Processed {len(changed)} files → {out_path}")
        return len(changed)


# Usage
processor = IncrementalProcessor(".pipeline_state.json")
n = processor.run("data/raw", "data/processed")
```

---

## Data Lineage Tracking

Lineage answers: *which data produced which model?* Recording it enables auditing, compliance, and debugging model regressions.

```python
import json
import hashlib
import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict


@dataclass
class DataLineageRecord:
    model_id: str
    model_path: str
    train_data_hash: str
    val_data_hash: str
    pipeline_config_hash: str
    git_hash: str
    timestamp: str
    num_train_rows: int
    num_val_rows: int
    metrics: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


def compute_file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_lineage(
    model_id: str,
    model_path: str,
    train_path: str,
    val_path: str,
    pipeline_config: dict,
    metrics: dict[str, float],
    lineage_dir: str = "lineage",
) -> DataLineageRecord:
    import pandas as pd
    import subprocess

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    config_hash = hashlib.sha256(
        json.dumps(pipeline_config, sort_keys=True).encode()
    ).hexdigest()[:12]

    record = DataLineageRecord(
        model_id=model_id,
        model_path=str(Path(model_path).resolve()),
        train_data_hash=compute_file_hash(train_path),
        val_data_hash=compute_file_hash(val_path),
        pipeline_config_hash=config_hash,
        git_hash=git_hash,
        timestamp=datetime.datetime.utcnow().isoformat(),
        num_train_rows=len(train_df),
        num_val_rows=len(val_df),
        metrics=metrics,
    )

    Path(lineage_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(lineage_dir) / f"{model_id}.json"
    out_path.write_text(json.dumps(asdict(record), indent=2))
    print(f"Lineage recorded → {out_path}")
    return record


# Usage after training
lineage = record_lineage(
    model_id="run_2024_0312_001",
    model_path="models/best.safetensors",
    train_path="data/splits/train.parquet",
    val_path="data/splits/val.parquet",
    pipeline_config={"tokenizer": "bert-base", "max_length": 512},
    metrics={"val_accuracy": 0.923, "val_loss": 0.198},
)
```

---

## Pipeline Comparison Table

<table class="compare-table">
  <thead>
    <tr>
      <th>Criterion</th>
      <th>DVC</th>
      <th>Kedro</th>
      <th>Prefect</th>
      <th>Metaflow</th>
      <th>Airflow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Setup complexity</strong></td>
      <td>⭐ Minimal</td>
      <td>⭐⭐ Low</td>
      <td>⭐⭐ Low</td>
      <td>⭐⭐ Low</td>
      <td>⭐⭐⭐⭐ High</td>
    </tr>
    <tr>
      <td><strong>Data versioning</strong></td>
      <td>✅ Native</td>
      <td>⚠️ Via catalog</td>
      <td>❌ External</td>
      <td>✅ Built-in</td>
      <td>❌ External</td>
    </tr>
    <tr>
      <td><strong>Scheduling</strong></td>
      <td>❌ Manual/CI</td>
      <td>❌ External</td>
      <td>✅ Built-in</td>
      <td>⚠️ AWS Events</td>
      <td>✅ Cron-based</td>
    </tr>
    <tr>
      <td><strong>Cloud compute</strong></td>
      <td>⚠️ Manual</td>
      <td>⚠️ Plugins</td>
      <td>✅ Workers</td>
      <td>✅ @batch/@kubernetes</td>
      <td>✅ Operators</td>
    </tr>
    <tr>
      <td><strong>Code testability</strong></td>
      <td>⭐⭐⭐ Good</td>
      <td>⭐⭐⭐⭐ Excellent</td>
      <td>⭐⭐⭐ Good</td>
      <td>⭐⭐ Moderate</td>
      <td>⭐ Poor (DAG coupling)</td>
    </tr>
    <tr>
      <td><strong>Best for</strong></td>
      <td>ML experiments, research</td>
      <td>Team ML projects</td>
      <td>Dynamic/event-driven</td>
      <td>Data-science at scale</td>
      <td>Production ETL / BI</td>
    </tr>
  </tbody>
</table>

---

## Pipeline Testing

Untested pipelines are time bombs. Test individual nodes in isolation, then test the full pipeline end-to-end with a small synthetic dataset.

```python
# tests/test_pipeline_nodes.py
import pytest
import pandas as pd
import numpy as np


# ── Unit tests for individual nodes ───────────────────────────────────

def make_raw_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "label": rng.integers(0, 10, n),
        "feature_a": rng.normal(0, 1, n),
        "feature_b": rng.normal(5, 2, n),
        "feature_c": rng.uniform(-1, 1, n),
    })


class TestValidateData:
    def test_passes_clean_data(self):
        df = make_raw_df()
        result = validate_data(df)
        assert len(result) == len(df)
        assert result["label"].between(0, 9).all()

    def test_drops_nulls(self):
        df = make_raw_df()
        df.loc[0:4, "feature_a"] = np.nan
        result = validate_data(df)
        assert len(result) == len(df) - 5
        assert result.isnull().sum().sum() == 0

    def test_raises_on_missing_column(self):
        df = make_raw_df().drop(columns=["label"])
        with pytest.raises(ValueError, match="Missing columns"):
            validate_data(df)


class TestBuildFeatures:
    def test_normalization(self):
        df = make_raw_df()
        params = {"feature_cols": ["feature_a", "feature_b"]}
        result = build_features(df.copy(), params)
        # Normalized columns should be ~N(0,1)
        assert abs(result["feature_a_norm"].mean()) < 0.1
        assert abs(result["feature_a_norm"].std() - 1.0) < 0.1

    def test_interaction_feature_exists(self):
        df = make_raw_df()
        params = {"feature_cols": []}
        result = build_features(df.copy(), params)
        assert "interaction_ab" in result.columns


class TestSplitData:
    def test_split_sizes(self):
        df = make_raw_df(1000)
        params = {"val_fraction": 0.1, "random_seed": 42}
        train, val = split_data(df, params)
        assert len(train) + len(val) == len(df)
        assert abs(len(val) / len(df) - 0.1) < 0.02

    def test_no_overlap(self):
        df = make_raw_df(1000)
        df["id"] = range(len(df))
        params = {"val_fraction": 0.2, "random_seed": 0}
        train, val = split_data(df, params)
        train_ids = set(train["id"])
        val_ids = set(val["id"])
        assert train_ids.isdisjoint(val_ids)


# ── Integration test for the full pipeline ────────────────────────────

def test_full_pipeline_integration(tmp_path):
    """Run the entire pipeline on synthetic data; assert outputs exist and are valid."""
    # Create raw data
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    df = make_raw_df(500)
    df.to_json(raw_dir / "shard_0000.jsonl", orient="records", lines=True)
    df.to_json(raw_dir / "shard_0001.jsonl", orient="records", lines=True)

    out_dir = tmp_path / "processed"
    splits_dir = tmp_path / "splits"

    # Stage 1: validate
    dfs = [pd.read_json(f, lines=True) for f in sorted(raw_dir.glob("*.jsonl"))]
    combined = pd.concat(dfs, ignore_index=True)
    validated = validate_data(combined)
    assert len(validated) > 0

    # Stage 2: features
    params = {"feature_cols": ["feature_a", "feature_b"]}
    features = build_features(validated.copy(), params)
    assert "feature_a_norm" in features.columns

    # Stage 3: split
    split_params = {"val_fraction": 0.1, "random_seed": 42}
    train, val = split_data(features, split_params)

    # Save + verify
    splits_dir.mkdir()
    train.to_parquet(splits_dir / "train.parquet")
    val.to_parquet(splits_dir / "val.parquet")
    assert (splits_dir / "train.parquet").exists()
    assert (splits_dir / "val.parquet").exists()

    reloaded_train = pd.read_parquet(splits_dir / "train.parquet")
    assert len(reloaded_train) == len(train)
    assert "feature_a_norm" in reloaded_train.columns
```

```bash
# Run all pipeline tests
pytest tests/test_pipeline_nodes.py -v

# Run with coverage
pytest tests/test_pipeline_nodes.py --cov=src --cov-report=term-missing
```

---

## Summary

<div class="diagram">
  <div class="diagram-title">Pipeline Tool Decision Tree</div>
  <div class="timeline">
    <div class="timeline-item">
      <div class="timeline-year">Solo / Research</div>
      <div class="timeline-title">Start with DVC</div>
      <div class="timeline-desc">Minimal setup. Git-native. Data versioning included. Run with <code>dvc repro</code>.</div>
    </div>
    <div class="timeline-item">
      <div class="timeline-year">Small Team</div>
      <div class="timeline-title">Add Kedro</div>
      <div class="timeline-desc">Enforce structure. Pure-function nodes are easy to test and reason about. Great for data scientists learning software engineering.</div>
    </div>
    <div class="timeline-item">
      <div class="timeline-year">Production / Event-Driven</div>
      <div class="timeline-title">Prefect or Metaflow</div>
      <div class="timeline-desc">Prefect for dynamic pipelines and retries. Metaflow for heavy cloud compute steps with <code>@batch</code>.</div>
    </div>
    <div class="timeline-item">
      <div class="timeline-year">Enterprise ETL</div>
      <div class="timeline-title">Airflow</div>
      <div class="timeline-desc">Complex scheduling, SLA alerts, extensive operator ecosystem. High operational burden — justify before adopting.</div>
    </div>
  </div>
</div>

<div class="diagram">
  <div class="diagram-grid cols-2">
    <div class="diagram-card teal">
      <div class="card-icon">✅</div>
      <div class="card-title">Pipeline Best Practices</div>
      <div class="card-desc">Declare deps explicitly · Test nodes in isolation · Version data alongside code · Track lineage · Use incremental processing · Write idempotent stages</div>
    </div>
    <div class="diagram-card red">
      <div class="card-icon">❌</div>
      <div class="card-title">Common Pitfalls</div>
      <div class="card-desc">Hardcoded paths · Implicit global state · No unit tests · Re-processing unchanged data · XCom-passing large DataFrames (Airflow) · No input validation</div>
    </div>
  </div>
</div>

*Last updated: May 2026*
