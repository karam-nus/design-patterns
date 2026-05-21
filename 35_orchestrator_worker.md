---
title: "Chapter 35 — Orchestrator-Worker Pattern"
---

[← Back to Table of Contents](./README.md)

# Chapter 35 — Orchestrator-Worker Pattern

> *"Divide the problem, conquer in parallel, synthesize the whole — this is the orchestrator's contract with its workers."*

The Orchestrator-Worker pattern is one of the most important structural patterns for large-scale ML systems and agentic AI. A central coordinator (the orchestrator) decomposes a task, delegates sub-tasks to specialized workers, and aggregates their results into a coherent final output. Workers are stateless, focused, and replaceable. The orchestrator is stateful, strategic, and responsible for the overall plan.

<div class="callout info">
<span class="callout-icon">ℹ️</span>
<div class="callout-body">
The Orchestrator-Worker pattern enables <strong>parallelism</strong> (multiple workers running concurrently), <strong>specialization</strong> (each worker optimized for one task type), and <strong>fault isolation</strong> (one worker's failure doesn't kill the orchestration).
</div>
</div>

---

## Pattern Overview

<div class="diagram">
<div class="diagram-title">Orchestrator-Worker — Core Structure</div>
<div class="flow">
  <div class="flow-node blue wide">Task / Goal</div>
  <div class="flow-arrow accent">↓ decompose</div>
  <div class="flow-node purple extra-wide">Orchestrator<br/><small>plan · dispatch · aggregate · synthesize</small></div>
</div>
<div class="flow">
  <div class="flow-arrow accent">↙</div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-arrow accent">↘</div>
</div>
<div class="flow">
  <div class="flow-node green">Worker 1<br/><small>specialist A</small></div>
  <div class="flow-node orange">Worker 2<br/><small>specialist B</small></div>
  <div class="flow-node teal">Worker 3<br/><small>specialist C</small></div>
</div>
<div class="flow">
  <div class="flow-arrow green">↓ result</div>
  <div class="flow-arrow accent">↓ result</div>
  <div class="flow-arrow accent">↓ result</div>
</div>
<div class="flow">
  <div class="flow-node purple extra-wide">Aggregator<br/><small>merge · validate · synthesize</small></div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-node blue wide">Final Output</div>
</div>
</div>

---

## Comparison with Similar Patterns

<table class="compare-table">
<thead>
<tr>
  <th>Pattern</th>
  <th>Structure</th>
  <th>Task Decomposition</th>
  <th>Worker State</th>
  <th>Aggregation</th>
  <th>ML Use Case</th>
</tr>
</thead>
<tbody>
<tr>
  <td><strong>Orchestrator-Worker</strong></td>
  <td>Central coordinator + workers</td>
  <td>Dynamic, adaptive</td>
  <td>Stateless</td>
  <td>Custom synthesis</td>
  <td>Agentic tasks, eval pipelines</td>
</tr>
<tr>
  <td><strong>Map-Reduce</strong></td>
  <td>Map phase + reduce phase</td>
  <td>Fixed: partition data uniformly</td>
  <td>Stateless</td>
  <td>Commutative reduce function</td>
  <td>Batch scoring, distributed training</td>
</tr>
<tr>
  <td><strong>Pipeline</strong></td>
  <td>Linear sequence of stages</td>
  <td>Sequential, ordered</td>
  <td>Stage-local</td>
  <td>None (pass-through)</td>
  <td>Feature engineering → train → evaluate</td>
</tr>
<tr>
  <td><strong>Blackboard</strong></td>
  <td>Shared knowledge store + agents</td>
  <td>Opportunistic, event-driven</td>
  <td>Reads/writes shared state</td>
  <td>Emergent via shared blackboard</td>
  <td>Multi-agent collaborative reasoning</td>
</tr>
</tbody>
</table>

---

## ML Use Case — Parallel Model Evaluation

An orchestrator spawns one worker per dataset, runs evaluation in parallel, and collects metrics.

```python
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class WorkerTask:
    task_id: str
    payload: Any
    worker_type: str = "default"


@dataclass
class WorkerResult:
    task_id: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class EvalConfig:
    model_name: str
    dataset_path: str
    metrics: list[str] = field(default_factory=lambda: ["accuracy", "f1", "latency_p99"])


@dataclass
class EvalResult:
    dataset: str
    metrics: dict[str, float]
    num_examples: int
    error: Optional[str] = None


def evaluate_on_dataset(config: EvalConfig) -> EvalResult:
    """Worker function: evaluate a model on a single dataset."""
    logger.info("Evaluating %s on %s", config.model_name, config.dataset_path)
    time.sleep(0.1)  # simulate IO-bound eval
    # In production: load dataset, run inference, compute metrics
    return EvalResult(
        dataset=config.dataset_path,
        metrics={"accuracy": 0.91, "f1": 0.89, "latency_p99": 43.2},
        num_examples=1000,
    )


class EvalOrchestrator:
    """
    Orchestrates parallel model evaluation across multiple datasets.

    Args:
        max_workers: maximum concurrent eval workers
        timeout_per_dataset: per-dataset timeout in seconds
    """

    def __init__(self, max_workers: int = 4, timeout_per_dataset: float = 300.0):
        self.max_workers = max_workers
        self.timeout = timeout_per_dataset

    def run(
        self,
        model_name: str,
        datasets: list[str],
        metrics: list[str] = None,
    ) -> dict:
        metrics = metrics or ["accuracy", "f1", "latency_p99"]
        configs = [
            EvalConfig(model_name, ds, metrics)
            for ds in datasets
        ]

        logger.info(
            "Orchestrating evaluation of %s across %d datasets (max %d workers)",
            model_name, len(configs), self.max_workers,
        )

        results = self._dispatch_parallel(configs)
        return self._aggregate(model_name, results)

    def _dispatch_parallel(self, configs: list[EvalConfig]) -> list[EvalResult]:
        import concurrent.futures
        results = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_config = {
                executor.submit(evaluate_on_dataset, cfg): cfg
                for cfg in configs
            }
            for future in concurrent.futures.as_completed(
                future_to_config, timeout=self.timeout * len(configs)
            ):
                cfg = future_to_config[future]
                try:
                    result = future.result(timeout=self.timeout)
                    results.append(result)
                    logger.info("✓ %s completed", cfg.dataset_path)
                except concurrent.futures.TimeoutError:
                    logger.error("✗ %s timed out", cfg.dataset_path)
                    results.append(EvalResult(cfg.dataset_path, {}, 0, error="timeout"))
                except Exception as exc:
                    logger.error("✗ %s failed: %s", cfg.dataset_path, exc)
                    results.append(EvalResult(cfg.dataset_path, {}, 0, error=str(exc)))
        return results

    def _aggregate(self, model_name: str, results: list[EvalResult]) -> dict:
        successful = [r for r in results if r.error is None]
        failed     = [r for r in results if r.error is not None]

        if not successful:
            return {"model": model_name, "status": "all_failed", "results": []}

        # Average metrics across datasets
        all_metric_keys = {k for r in successful for k in r.metrics}
        avg_metrics = {}
        for key in all_metric_keys:
            values = [r.metrics[key] for r in successful if key in r.metrics]
            avg_metrics[key] = sum(values) / len(values)

        return {
            "model": model_name,
            "status": "partial" if failed else "complete",
            "datasets_evaluated": len(successful),
            "datasets_failed": len(failed),
            "total_examples": sum(r.num_examples for r in successful),
            "aggregate_metrics": avg_metrics,
            "per_dataset": [
                {"dataset": r.dataset, "metrics": r.metrics, "n": r.num_examples}
                for r in successful
            ],
            "failures": [{"dataset": r.dataset, "error": r.error} for r in failed],
        }


# --- Run it ---
orchestrator = EvalOrchestrator(max_workers=4)
report = orchestrator.run(
    model_name="my-classifier-v2",
    datasets=[
        "data/eval/en_test.jsonl",
        "data/eval/de_test.jsonl",
        "data/eval/fr_test.jsonl",
        "data/eval/es_test.jsonl",
        "data/eval/zh_test.jsonl",
    ],
)
```

---

## Supervisor-Worker for Agentic Tasks

The orchestrator decomposes a complex goal into sub-tasks, delegates each to a specialized agent-worker, then synthesizes their outputs.

```python
import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)
aclient = AsyncOpenAI()


@dataclass
class AgentTask:
    task_id: str
    description: str
    worker_type: str
    context: dict = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)


@dataclass
class AgentTaskResult:
    task_id: str
    worker_type: str
    output: str
    success: bool
    error: Optional[str] = None


SUPERVISOR_DECOMPOSE_PROMPT = """\
You are a task supervisor. Break down the following goal into concrete sub-tasks.
Each sub-task should be assigned to one of these worker types:
- researcher: gathers information and facts
- coder: writes or reviews code
- critic: evaluates quality and identifies issues
- writer: drafts prose, reports, or documentation

Respond with a JSON array of tasks:
[{{"task_id": "t1", "description": "...", "worker_type": "...", "dependencies": []}}]

Goal: {goal}
"""

WORKER_SYSTEM_PROMPTS = {
    "researcher": "You are a research specialist. Gather comprehensive, accurate information.",
    "coder":      "You are a software engineer. Write clean, efficient, well-documented code.",
    "critic":     "You are a quality reviewer. Identify flaws, gaps, and improvements.",
    "writer":     "You are a technical writer. Produce clear, well-structured prose.",
}


class SupervisorOrchestrator:
    """
    Agentic orchestrator: decomposes a goal, dispatches to typed workers,
    aggregates and synthesizes results.
    """

    def __init__(self, model: str = "gpt-4o", max_concurrent: int = 3):
        self.model = model
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def run(self, goal: str) -> dict:
        logger.info("Decomposing goal: %s", goal[:60])
        tasks = await self._decompose(goal)
        logger.info("Created %d sub-tasks", len(tasks))

        results = await self._execute_dag(tasks)

        synthesis = await self._synthesize(goal, results)
        return {
            "goal": goal,
            "num_tasks": len(tasks),
            "completed": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "synthesis": synthesis,
            "task_outputs": [
                {"id": r.task_id, "type": r.worker_type, "output": r.output[:200]}
                for r in results
            ],
        }

    async def _decompose(self, goal: str) -> list[AgentTask]:
        response = await aclient.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": SUPERVISOR_DECOMPOSE_PROMPT.format(goal=goal),
            }],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = json.loads(response.choices[0].message.content)
        tasks_data = raw if isinstance(raw, list) else raw.get("tasks", [])
        return [
            AgentTask(
                task_id=t["task_id"],
                description=t["description"],
                worker_type=t["worker_type"],
                dependencies=t.get("dependencies", []),
            )
            for t in tasks_data
        ]

    async def _execute_dag(self, tasks: list[AgentTask]) -> list[AgentTaskResult]:
        """Execute tasks respecting dependency order."""
        results: dict[str, AgentTaskResult] = {}
        pending = list(tasks)

        while pending:
            # Find tasks whose dependencies are all satisfied
            ready = [
                t for t in pending
                if all(dep in results for dep in t.dependencies)
            ]
            if not ready:
                logger.error("Dependency deadlock detected among: %s",
                             [t.task_id for t in pending])
                break

            # Execute ready tasks concurrently
            batch_results = await asyncio.gather(
                *[self._execute_worker(t, results) for t in ready],
                return_exceptions=True,
            )
            for task, result in zip(ready, batch_results):
                if isinstance(result, Exception):
                    results[task.task_id] = AgentTaskResult(
                        task.task_id, task.worker_type, "", False, str(result)
                    )
                else:
                    results[task.task_id] = result
                pending.remove(task)

        return list(results.values())

    async def _execute_worker(
        self,
        task: AgentTask,
        prior_results: dict[str, AgentTaskResult],
    ) -> AgentTaskResult:
        async with self._semaphore:
            system_prompt = WORKER_SYSTEM_PROMPTS.get(task.worker_type, "You are a helpful assistant.")
            context_str = ""
            if task.dependencies:
                context_str = "\n\nRelevant prior work:\n" + "\n".join(
                    f"[{dep}]: {prior_results[dep].output[:300]}"
                    for dep in task.dependencies
                    if dep in prior_results
                )
            try:
                response = await aclient.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": task.description + context_str},
                    ],
                    temperature=0.0,
                )
                output = response.choices[0].message.content
                logger.info("Worker %s (%s) completed", task.task_id, task.worker_type)
                return AgentTaskResult(task.task_id, task.worker_type, output, True)
            except Exception as exc:
                logger.error("Worker %s failed: %s", task.task_id, exc)
                return AgentTaskResult(task.task_id, task.worker_type, "", False, str(exc))

    async def _synthesize(self, goal: str, results: list[AgentTaskResult]) -> str:
        successful = [r for r in results if r.success]
        if not successful:
            return "No successful sub-tasks to synthesize."
        combined = "\n\n".join(
            f"## {r.worker_type.title()} ({r.task_id})\n{r.output}"
            for r in successful
        )
        response = await aclient.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": (
                    f"Goal: {goal}\n\n"
                    f"Sub-task outputs:\n{combined}\n\n"
                    f"Synthesize a cohesive final response addressing the original goal."
                ),
            }],
            temperature=0.0,
        )
        return response.choices[0].message.content
```

---

## Worker Specialization

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class WorkerInput:
    task_id: str
    query: str
    context: dict


@dataclass
class WorkerOutput:
    task_id: str
    result: Any
    metadata: dict = field(default_factory=dict)


class BaseWorker(ABC):
    """Abstract base for all workers."""

    @property
    @abstractmethod
    def worker_type(self) -> str: ...

    @abstractmethod
    def execute(self, inp: WorkerInput) -> WorkerOutput: ...

    def can_handle(self, task_type: str) -> bool:
        return task_type == self.worker_type


class SearchWorker(BaseWorker):
    """Retrieves information from external sources."""

    worker_type = "search"

    def __init__(self, search_fn: Callable[[str], str]):
        self._search = search_fn

    def execute(self, inp: WorkerInput) -> WorkerOutput:
        results = self._search(inp.query)
        return WorkerOutput(inp.task_id, results, {"source": "web"})


class CodeWorker(BaseWorker):
    """Generates or analyzes code."""

    worker_type = "code"

    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def execute(self, inp: WorkerInput) -> WorkerOutput:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert software engineer."},
                {"role": "user",   "content": inp.query},
            ],
            temperature=0.0,
        )
        code = response.choices[0].message.content
        return WorkerOutput(inp.task_id, code, {"model": self.model})


class MathWorker(BaseWorker):
    """Solves mathematical problems."""

    worker_type = "math"

    def execute(self, inp: WorkerInput) -> WorkerOutput:
        try:
            allowed = set("0123456789+-*/.() ")
            expr = inp.query.strip()
            if all(c in allowed for c in expr):
                result = eval(expr)  # noqa: S307 — guarded above
            else:
                result = f"Cannot evaluate: {expr}"
        except Exception as exc:
            result = f"Math error: {exc}"
        return WorkerOutput(inp.task_id, result)


class WorkerRegistry:
    """Maps task types to worker instances."""

    def __init__(self):
        self._workers: dict[str, BaseWorker] = {}

    def register(self, worker: BaseWorker):
        self._workers[worker.worker_type] = worker

    def get(self, worker_type: str) -> Optional[BaseWorker]:
        return self._workers.get(worker_type)

    def dispatch(self, task_type: str, inp: WorkerInput) -> WorkerOutput:
        worker = self.get(task_type)
        if worker is None:
            raise ValueError(f"No worker registered for type: {task_type}")
        return worker.execute(inp)
```

---

## Async Orchestration

```python
import asyncio
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


class AsyncOrchestrator:
    """
    asyncio-based parallel orchestration with streaming results.
    """

    def __init__(self, max_concurrent: int = 5):
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def run_batch(
        self,
        tasks: list[WorkerTask],
        worker_fn: Callable,
    ) -> list[WorkerResult]:
        return await asyncio.gather(
            *[self._run_single(t, worker_fn) for t in tasks]
        )

    async def _run_single(
        self,
        task: WorkerTask,
        worker_fn: Callable,
    ) -> WorkerResult:
        import time
        async with self._semaphore:
            start = time.monotonic()
            try:
                if asyncio.iscoroutinefunction(worker_fn):
                    data = await worker_fn(task)
                else:
                    data = await asyncio.get_event_loop().run_in_executor(
                        None, worker_fn, task
                    )
                return WorkerResult(
                    task_id=task.task_id,
                    success=True,
                    data=data,
                    latency_ms=(time.monotonic() - start) * 1000,
                )
            except Exception as exc:
                logger.error("Task %s failed: %s", task.task_id, exc)
                return WorkerResult(
                    task_id=task.task_id,
                    success=False,
                    error=str(exc),
                    latency_ms=(time.monotonic() - start) * 1000,
                )

    async def stream_results(
        self,
        tasks: list[WorkerTask],
        worker_fn: Callable,
    ) -> AsyncGenerator[WorkerResult, None]:
        """Yield results as they complete (not in submission order)."""
        queue: asyncio.Queue = asyncio.Queue()

        async def worker_with_notify(t):
            result = await self._run_single(t, worker_fn)
            await queue.put(result)

        runners = [asyncio.create_task(worker_with_notify(t)) for t in tasks]
        for _ in range(len(tasks)):
            yield await queue.get()
        await asyncio.gather(*runners)


# Usage example
async def main():
    orchestrator = AsyncOrchestrator(max_concurrent=4)

    async def mock_worker(task: WorkerTask) -> str:
        await asyncio.sleep(0.1)
        return f"result for {task.task_id}"

    tasks = [WorkerTask(f"t{i}", {"query": f"Query {i}"}) for i in range(10)]
    results = await orchestrator.run_batch(tasks, mock_worker)
    successes = [r for r in results if r.success]
    print(f"Completed {len(successes)}/{len(tasks)} tasks")
```

---

## Error Handling — Partial Results and Timeouts

```python
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RobustOrchestrator:
    """
    Handles partial failures gracefully: collects whatever workers succeed,
    retries transiently failed tasks, and returns partial results on timeout.
    """

    def __init__(
        self,
        max_retries: int = 2,
        per_task_timeout: float = 30.0,
        min_success_fraction: float = 0.5,
    ):
        self.max_retries = max_retries
        self.per_task_timeout = per_task_timeout
        self.min_success_fraction = min_success_fraction

    async def run(
        self,
        tasks: list[WorkerTask],
        worker_fn: Callable,
    ) -> dict:
        all_results: dict[str, WorkerResult] = {}
        remaining = list(tasks)

        for attempt in range(self.max_retries + 1):
            if not remaining:
                break

            futures = {
                asyncio.create_task(
                    asyncio.wait_for(
                        self._execute(t, worker_fn),
                        timeout=self.per_task_timeout,
                    )
                ): t
                for t in remaining
            }

            done, _ = await asyncio.wait(futures.keys())
            next_remaining = []

            for fut in done:
                task = futures[fut]
                try:
                    result: WorkerResult = fut.result()
                    all_results[task.task_id] = result
                    if not result.success and attempt < self.max_retries:
                        next_remaining.append(task)
                except asyncio.TimeoutError:
                    all_results[task.task_id] = WorkerResult(
                        task.task_id, False, error="timeout"
                    )
                    if attempt < self.max_retries:
                        next_remaining.append(task)

            remaining = next_remaining
            if remaining and attempt < self.max_retries:
                logger.warning(
                    "Retrying %d failed tasks (attempt %d/%d)",
                    len(remaining), attempt + 1, self.max_retries,
                )

        succeeded = [r for r in all_results.values() if r.success]
        success_rate = len(succeeded) / max(len(tasks), 1)

        return {
            "status": "ok" if success_rate >= self.min_success_fraction else "degraded",
            "success_rate": success_rate,
            "results": list(all_results.values()),
            "succeeded": len(succeeded),
            "failed": len(tasks) - len(succeeded),
        }

    async def _execute(self, task: WorkerTask, worker_fn: Callable) -> WorkerResult:
        import time
        start = time.monotonic()
        try:
            if asyncio.iscoroutinefunction(worker_fn):
                data = await worker_fn(task)
            else:
                data = await asyncio.get_event_loop().run_in_executor(None, worker_fn, task)
            return WorkerResult(task.task_id, True, data,
                                latency_ms=(time.monotonic() - start) * 1000)
        except Exception as exc:
            return WorkerResult(task.task_id, False, error=str(exc),
                                latency_ms=(time.monotonic() - start) * 1000)
```

---

## Dynamic Worker Spawning

Spawn more workers for complex tasks, fewer for simple ones.

```python
import math
import logging
from typing import Any

logger = logging.getLogger(__name__)


class DynamicOrchestrator:
    """
    Adjusts the number of workers based on estimated task complexity.
    """

    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 8,
        complexity_fn: Optional[Callable] = None,
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self._complexity_fn = complexity_fn or self._default_complexity

    def _default_complexity(self, task: Any) -> float:
        """Estimate complexity from task payload size."""
        payload_size = len(str(task))
        return min(payload_size / 1000, 1.0)

    def _workers_for_complexity(self, complexity: float) -> int:
        workers = math.ceil(complexity * self.max_workers)
        return max(self.min_workers, min(workers, self.max_workers))

    def plan(self, task: Any) -> dict:
        complexity = self._complexity_fn(task)
        workers = self._workers_for_complexity(complexity)
        logger.info(
            "Dynamic plan: complexity=%.2f → %d workers", complexity, workers
        )
        return {
            "complexity": complexity,
            "num_workers": workers,
            "strategy": "parallel" if workers > 1 else "sequential",
        }

    def decompose(self, task: Any, num_workers: int) -> list:
        """Split task into num_workers sub-tasks."""
        if isinstance(task, list):
            chunk_size = max(1, math.ceil(len(task) / num_workers))
            return [task[i:i + chunk_size] for i in range(0, len(task), chunk_size)]
        return [task] * num_workers
```

---

## LangGraph, CrewAI, and AutoGen Overview

Modern agentic frameworks implement the Orchestrator-Worker pattern out of the box.

<div class="diagram">
<div class="diagram-grid cols-3">
  <div class="diagram-card blue">
    <div class="card-icon">🔷</div>
    <div class="card-title">LangGraph</div>
    <div class="card-desc">Graph-based orchestration. Nodes = agents/tools. Edges = conditional routing. Supports cycles, human-in-the-loop, and streaming. Best for: stateful multi-agent workflows.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🤝</div>
    <div class="card-title">CrewAI</div>
    <div class="card-desc">Role-based crews with explicit agent personas. Orchestrator assigns tasks; workers have defined roles (researcher, writer, critic). Best for: document-generation pipelines.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">🤖</div>
    <div class="card-title">AutoGen</div>
    <div class="card-desc">Conversation-driven multi-agent. GroupChat manager orchestrates; specialist agents respond. Built-in code execution, human proxy. Best for: coding and debugging tasks.</div>
  </div>
</div>
</div>

```python
# LangGraph orchestrator-worker sketch
# from langgraph.graph import StateGraph, END
# from typing import TypedDict, Annotated
# import operator
#
# class OrchestratorState(TypedDict):
#     goal: str
#     tasks: list[dict]
#     results: Annotated[list, operator.add]
#     final_answer: str
#
# def planner_node(state: OrchestratorState) -> OrchestratorState:
#     tasks = decompose_goal(state["goal"])
#     return {"tasks": tasks}
#
# def worker_node(state: OrchestratorState) -> OrchestratorState:
#     results = [execute_task(t) for t in state["tasks"]]
#     return {"results": results}
#
# def synthesizer_node(state: OrchestratorState) -> OrchestratorState:
#     answer = synthesize(state["goal"], state["results"])
#     return {"final_answer": answer}
#
# graph = StateGraph(OrchestratorState)
# graph.add_node("planner",     planner_node)
# graph.add_node("worker",      worker_node)
# graph.add_node("synthesizer", synthesizer_node)
# graph.add_edge("planner", "worker")
# graph.add_edge("worker",  "synthesizer")
# graph.add_edge("synthesizer", END)
# app = graph.compile()

# CrewAI sketch
# from crewai import Agent, Task, Crew
#
# researcher = Agent(role="Researcher", goal="Find accurate information", ...)
# writer     = Agent(role="Writer",     goal="Write clear reports",       ...)
# critic     = Agent(role="Critic",     goal="Identify errors and gaps",  ...)
#
# crew = Crew(
#     agents=[researcher, writer, critic],
#     tasks=[research_task, write_task, review_task],
#     process=Process.sequential,
# )
# result = crew.kickoff(inputs={"topic": "..."})
```

---

## Hierarchical Orchestration

An orchestrator of orchestrators — each sub-orchestrator manages a domain.

<div class="diagram">
<div class="diagram-title">Hierarchical Orchestrator Structure</div>
<div class="flow">
  <div class="flow-node blue extra-wide">Top-Level Orchestrator<br/><small>decomposes goal into domain sub-goals</small></div>
</div>
<div class="flow">
  <div class="flow-arrow accent">↙</div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-arrow accent">↘</div>
</div>
<div class="flow">
  <div class="flow-node purple">Research<br/>Orchestrator</div>
  <div class="flow-node orange">Engineering<br/>Orchestrator</div>
  <div class="flow-node teal">QA<br/>Orchestrator</div>
</div>
<div class="flow">
  <div class="flow-arrow purple">↓</div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-arrow accent">↓</div>
</div>
<div class="flow">
  <div class="flow-node green narrow">Web<br/>Worker</div>
  <div class="flow-node green narrow">PDF<br/>Worker</div>
  <div class="flow-node orange narrow">Code<br/>Worker</div>
  <div class="flow-node orange narrow">Test<br/>Worker</div>
  <div class="flow-node teal narrow">Review<br/>Worker</div>
  <div class="flow-node teal narrow">Report<br/>Worker</div>
</div>
<div class="flow">
  <div class="flow-arrow accent">↓ aggregate all domain results</div>
</div>
<div class="flow">
  <div class="flow-node blue extra-wide">Synthesized Final Output</div>
</div>
</div>

```python
class HierarchicalOrchestrator:
    """
    Two-level orchestration: top-level decomposes by domain,
    sub-orchestrators handle domain-specific parallelism.
    """

    def __init__(self, sub_orchestrators: dict[str, "EvalOrchestrator"]):
        self.sub_orchestrators = sub_orchestrators

    def run(self, goal: str, domain_tasks: dict[str, list]) -> dict:
        import concurrent.futures

        domain_results = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.sub_orchestrators)
        ) as executor:
            futures = {
                domain: executor.submit(
                    self.sub_orchestrators[domain].run,
                    goal,
                    tasks,
                )
                for domain, tasks in domain_tasks.items()
                if domain in self.sub_orchestrators
            }
            for domain, future in futures.items():
                try:
                    domain_results[domain] = future.result(timeout=600)
                except Exception as exc:
                    domain_results[domain] = {"error": str(exc)}

        return self._synthesize_domains(goal, domain_results)

    def _synthesize_domains(self, goal: str, domain_results: dict) -> dict:
        return {
            "goal": goal,
            "domains_completed": len(domain_results),
            "domain_results": domain_results,
        }
```

---

## Full Agentic Task Flow

<div class="diagram">
<div class="diagram-title">Agentic Task: Planner → Specialists → Synthesizer</div>
<div class="flow">
  <div class="flow-node blue wide">User Goal:<br/><small>"Produce a technical report on LLM scaling laws"</small></div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-node purple wide">Planner Agent<br/><small>decomposes into sub-tasks; assigns worker types</small></div>
</div>
<div class="flow">
  <div class="flow-arrow accent">↙</div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-arrow accent">↘</div>
</div>
<div class="flow">
  <div class="flow-node green">Researcher<br/><small>gather papers, facts, citations</small></div>
  <div class="flow-node orange">Coder<br/><small>reproduce scaling law plots</small></div>
  <div class="flow-node teal">Critic<br/><small>identify gaps, add counterpoints</small></div>
</div>
<div class="flow">
  <div class="flow-arrow green">↓ findings</div>
  <div class="flow-arrow accent">↓ figures</div>
  <div class="flow-arrow accent">↓ critique</div>
</div>
<div class="flow">
  <div class="flow-node blue wide">Synthesizer Agent<br/><small>weave all outputs into coherent report</small></div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-node blue wide">Final Report</div>
</div>
</div>

---

## Comparison Table

<table class="compare-table">
<thead>
<tr>
  <th>Pattern</th>
  <th>Coordination</th>
  <th>Parallelism</th>
  <th>Worker State</th>
  <th>Failure Handling</th>
  <th>ML Use Case</th>
</tr>
</thead>
<tbody>
<tr>
  <td><strong>Orchestrator-Worker</strong></td>
  <td>Central, adaptive</td>
  <td>✅ High</td>
  <td>Stateless</td>
  <td>Retry / partial results</td>
  <td>Agentic pipelines, eval harnesses</td>
</tr>
<tr>
  <td><strong>Pipeline</strong></td>
  <td>Sequential stages</td>
  <td>❌ Sequential</td>
  <td>Stage-local</td>
  <td>Abort or checkpoint</td>
  <td>Preprocessing → train → eval</td>
</tr>
<tr>
  <td><strong>Map-Reduce</strong></td>
  <td>Fixed 2-phase</td>
  <td>✅ Map phase</td>
  <td>Stateless mapper</td>
  <td>Re-run failed partitions</td>
  <td>Distributed scoring, feature agg</td>
</tr>
<tr>
  <td><strong>Blackboard</strong></td>
  <td>Shared memory, event-driven</td>
  <td>✅ Medium</td>
  <td>Reads/writes shared state</td>
  <td>Competing writes → conflicts</td>
  <td>Multi-agent collaborative reasoning</td>
</tr>
</tbody>
</table>

---

<div class="callout tip">
<span class="callout-icon">💡</span>
<div class="callout-body">
Keep workers truly stateless. Any state that needs to persist between calls belongs in the orchestrator or an external store (e.g., Redis, a database). Stateless workers are trivially restartable after failures.
</div>
</div>

<div class="callout warn">
<span class="callout-icon">⚠️</span>
<div class="callout-body">
Watch for <strong>orchestrator bottleneck</strong>: if the orchestrator itself is a single Python process with a slow synthesis step, it can become the throughput ceiling. Offload heavy synthesis to a worker too.
</div>
</div>

---

## Summary

<div class="diagram">
<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-year">Decompose</div>
    <div class="timeline-title">Break the goal into atomic sub-tasks</div>
    <div class="timeline-desc">Orchestrator plans; each sub-task should be independently executable and testable.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Dispatch</div>
    <div class="timeline-title">Route to specialized workers</div>
    <div class="timeline-desc">Match task type to worker capability. Use the WorkerRegistry to avoid hard-coding dispatch logic.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Execute</div>
    <div class="timeline-title">Run in parallel with timeouts</div>
    <div class="timeline-desc">Use asyncio or ThreadPoolExecutor. Respect per-task deadlines; collect partial results on timeout.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Aggregate</div>
    <div class="timeline-title">Collect and validate results</div>
    <div class="timeline-desc">Handle partial failures gracefully. Decide minimum success fraction before synthesizing.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Synthesize</div>
    <div class="timeline-title">Merge into coherent output</div>
    <div class="timeline-desc">A synthesizer step (often another LLM call) weaves worker outputs into the final response.</div>
  </div>
</div>
</div>

<span class="badge agentic">Agentic</span> <span class="badge mlops">MLOps</span>

*Last updated: May 2026*
