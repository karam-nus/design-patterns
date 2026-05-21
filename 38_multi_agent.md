---
title: "Chapter 38 — Multi-Agent Communication"
---

[← Back to Table of Contents](./README.md)

# Chapter 38 — Multi-Agent Communication

> *"No single agent can know everything; the collective must communicate to act intelligently." — Multi-Agent Systems, 2024*

<span class="badge agentic">Agentic</span>

---

## Overview

Multi-agent systems decompose complex tasks across specialised agents that collaborate by passing information. This chapter covers the core communication patterns — message passing, blackboard, pub-sub, shared state — and shows how they manifest in modern LLM frameworks like LangGraph and AutoGen.

---

## 38.1 Why Multi-Agent?

<div class="diagram-grid cols-3">
  <div class="diagram-card blue">
    <div class="card-icon">🔀</div>
    <div class="card-title">Task Decomposition</div>
    <div class="card-desc">Break a monolithic task into sub-tasks each agent handles independently, reducing per-agent context load.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🎯</div>
    <div class="card-title">Specialisation</div>
    <div class="card-desc">Each agent uses a model or prompt optimised for its role: researcher, coder, critic, summariser.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">⚡</div>
    <div class="card-title">Parallelism</div>
    <div class="card-desc">Independent sub-tasks run concurrently, dramatically reducing wall-clock time for complex workflows.</div>
  </div>
</div>

---

## 38.2 Message Passing Pattern

Agents communicate by sending typed messages to each other. A central message bus or direct peer-to-peer delivery routes messages.

```python
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import Callable, Any
from enum import Enum

class MessageType(str, Enum):
    TASK = "task"
    RESULT = "result"
    CRITIQUE = "critique"
    REQUEST = "request"
    ERROR = "error"

@dataclass
class Message:
    sender: str
    recipient: str
    type: MessageType
    content: Any
    message_id: str = field(default_factory=lambda: __import__('uuid').uuid4().hex[:8])
    reply_to: str | None = None

class MessageBus:
    """Simple async message bus for agent-to-agent communication."""

    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = {}
        self._history: list[Message] = []

    def register(self, agent_id: str) -> None:
        self._queues[agent_id] = asyncio.Queue()

    async def send(self, message: Message) -> None:
        self._history.append(message)
        if message.recipient not in self._queues:
            raise KeyError(f"Agent '{message.recipient}' not registered")
        await self._queues[message.recipient].put(message)

    async def receive(self, agent_id: str, timeout: float = 30.0) -> Message:
        return await asyncio.wait_for(self._queues[agent_id].get(), timeout=timeout)

    def get_history(self) -> list[Message]:
        return list(self._history)

@dataclass
class BaseAgent:
    agent_id: str
    bus: MessageBus
    process_fn: Callable[[Message], Message | None]

    def __post_init__(self):
        self.bus.register(self.agent_id)

    async def run(self, max_messages: int = 100) -> None:
        for _ in range(max_messages):
            try:
                msg = await self.bus.receive(self.agent_id, timeout=5.0)
                response = self.process_fn(msg)
                if response:
                    await self.bus.send(response)
            except asyncio.TimeoutError:
                break

# ---- Example: Research + Summary pipeline -----------------------------------

async def message_passing_demo():
    bus = MessageBus()

    def researcher(msg: Message) -> Message | None:
        if msg.type == MessageType.TASK:
            result = f"Research findings on: {msg.content}"
            return Message(
                sender="researcher",
                recipient="summariser",
                type=MessageType.RESULT,
                content=result,
                reply_to=msg.message_id,
            )
        return None

    def summariser(msg: Message) -> Message | None:
        if msg.type == MessageType.RESULT:
            summary = f"Summary: {msg.content[:50]}..."
            return Message(
                sender="summariser",
                recipient="coordinator",
                type=MessageType.RESULT,
                content=summary,
            )
        return None

    researcher_agent = BaseAgent("researcher", bus, researcher)
    summariser_agent = BaseAgent("summariser", bus, summariser)
    bus.register("coordinator")

    await bus.send(Message(
        sender="coordinator",
        recipient="researcher",
        type=MessageType.TASK,
        content="quantum computing breakthroughs 2024",
    ))

    await asyncio.gather(
        researcher_agent.run(max_messages=1),
        summariser_agent.run(max_messages=1),
    )
    return bus.get_history()
```

---

## 38.3 Blackboard Pattern

All agents read from and write to a shared **blackboard** (knowledge store). Agents are triggered when new information appears that matches their expertise.

```python
from threading import Lock
from typing import Any
import time

class Blackboard:
    """
    Thread-safe shared workspace for multi-agent collaboration.
    Agents post partial results; others read and extend them.
    """

    def __init__(self):
        self._data: dict[str, Any] = {}
        self._lock = Lock()
        self._watchers: dict[str, list[Callable]] = {}

    def write(self, key: str, value: Any, author: str = "unknown") -> None:
        with self._lock:
            self._data[key] = {"value": value, "author": author, "timestamp": time.time()}
        # Notify watchers
        for pattern, callbacks in self._watchers.items():
            if key.startswith(pattern):
                for cb in callbacks:
                    cb(key, value)

    def read(self, key: str) -> Any:
        with self._lock:
            entry = self._data.get(key)
            return entry["value"] if entry else None

    def watch(self, key_prefix: str, callback: Callable) -> None:
        self._watchers.setdefault(key_prefix, []).append(callback)

    def snapshot(self) -> dict:
        with self._lock:
            return {k: v["value"] for k, v in self._data.items()}


class BlackboardAgent:
    def __init__(self, name: str, board: Blackboard, triggers: list[str]):
        self.name = name
        self.board = board
        for trigger in triggers:
            board.watch(trigger, self._on_update)

    def _on_update(self, key: str, value: Any) -> None:
        result = self.process(key, value)
        if result is not None:
            out_key, out_val = result
            self.board.write(out_key, out_val, author=self.name)

    def process(self, key: str, value: Any) -> tuple[str, Any] | None:
        raise NotImplementedError
```

---

## 38.4 Publish-Subscribe Pattern

An event bus decouples agents: publishers emit events by *type*, subscribers register interest in event types. No agent needs to know who receives its output.

```python
from collections import defaultdict
from typing import Any, Callable
import asyncio

class EventBus:
    """Async publish-subscribe event bus for agent coordination."""

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable) -> None:
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        self._subscribers[event_type].remove(handler)

    async def publish(self, event_type: str, payload: Any) -> None:
        for handler in self._subscribers.get(event_type, []):
            if asyncio.iscoroutinefunction(handler):
                await handler(payload)
            else:
                handler(payload)

    async def publish_all(self, events: list[tuple[str, Any]]) -> None:
        await asyncio.gather(*[self.publish(t, p) for t, p in events])

# Usage sketch
async def pubsub_demo():
    bus = EventBus()
    results = []

    async def on_query(payload):
        results.append(f"research: {payload['query']}")
        await bus.publish("research.complete", {"findings": "result A"})

    async def on_research(payload):
        results.append(f"writing: {payload['findings']}")
        await bus.publish("draft.complete", {"draft": "final draft"})

    bus.subscribe("query.received", on_query)
    bus.subscribe("research.complete", on_research)

    await bus.publish("query.received", {"query": "AI safety 2025"})
    return results
```

---

## 38.5 Shared State with Optimistic Locking

When agents concurrently update a shared object, optimistic locking (CAS — Compare-and-Swap) prevents lost updates.

```python
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

@dataclass
class VersionedState:
    """CAS-style shared state for concurrent agent access."""

    _state: dict = field(default_factory=dict)
    _version: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def read(self) -> tuple[dict, int]:
        with self._lock:
            return dict(self._state), self._version

    def compare_and_swap(self, expected_version: int, new_state: dict) -> bool:
        """Return True if swap succeeded, False if stale (another agent wrote first)."""
        with self._lock:
            if self._version != expected_version:
                return False  # optimistic lock failed
            self._state = new_state
            self._version += 1
            return True

    def update(self, key: str, value: Any, max_retries: int = 5) -> bool:
        """Retry loop for safe field updates."""
        for _ in range(max_retries):
            current, version = self.read()
            current[key] = value
            if self.compare_and_swap(version, current):
                return True
        return False
```

---

## 38.6 Agent Handoff Pattern

One agent transfers responsibility to another, passing accumulated context so the receiving agent can continue seamlessly.

```python
@dataclass
class HandoffContext:
    task: str
    history: list[dict]
    artifacts: dict = field(default_factory=dict)
    next_agent: str | None = None
    notes: str = ""

@dataclass
class HandoffOrchestrator:
    """Routes tasks between agents based on handoff instructions."""

    agents: dict[str, Callable[[HandoffContext], HandoffContext]]
    max_hops: int = 10

    def run(self, initial_context: HandoffContext) -> HandoffContext:
        ctx = initial_context
        for hop in range(self.max_hops):
            if ctx.next_agent is None:
                break
            agent_fn = self.agents.get(ctx.next_agent)
            if agent_fn is None:
                raise KeyError(f"Unknown agent: {ctx.next_agent}")
            ctx.history.append({"hop": hop, "agent": ctx.next_agent, "task": ctx.task})
            ctx = agent_fn(ctx)
        return ctx
```

---

## 38.7 Consensus Pattern

Agents vote on a shared decision. Useful for safety-critical outputs or when multiple agents produce conflicting results.

```python
from collections import Counter
from typing import Any

def majority_vote(votes: list[Any]) -> tuple[Any, float]:
    """Return (winner, confidence) where confidence = winning_fraction."""
    if not votes:
        raise ValueError("No votes provided")
    counts = Counter(votes)
    winner, count = counts.most_common(1)[0]
    return winner, count / len(votes)

@dataclass
class ConsensusAgent:
    """
    Collects votes from a panel of agents and returns majority decision.
    """

    panel: list[Callable[[str], Any]]
    quorum_fraction: float = 0.6   # minimum fraction needed to accept

    def decide(self, question: str) -> dict:
        votes = [agent(question) for agent in self.panel]
        winner, confidence = majority_vote(votes)
        accepted = confidence >= self.quorum_fraction
        return {
            "decision": winner if accepted else None,
            "accepted": accepted,
            "confidence": confidence,
            "votes": votes,
            "tally": dict(Counter(votes)),
        }
```

---

## 38.8 Communication Topology

<div class="diagram">
  <div class="diagram-title">Agent Communication Topologies</div>
  <div class="diagram-grid cols-4">
    <div class="diagram-card blue">
      <div class="card-icon">⭐</div>
      <div class="card-title">Star</div>
      <div class="card-desc">Central coordinator routes all messages. Simple, single point of failure.</div>
    </div>
    <div class="diagram-card green">
      <div class="card-icon">🔁</div>
      <div class="card-title">Ring</div>
      <div class="card-desc">Agents pass output to the next in sequence. Good for pipelines.</div>
    </div>
    <div class="diagram-card purple">
      <div class="card-icon">🕸️</div>
      <div class="card-title">Mesh</div>
      <div class="card-desc">Any agent can message any other. Flexible; complex routing needed.</div>
    </div>
    <div class="diagram-card orange">
      <div class="card-icon">🌲</div>
      <div class="card-title">Hierarchical</div>
      <div class="card-desc">Supervisor → sub-agents → sub-sub-agents. Scales to complex tasks.</div>
    </div>
  </div>
</div>

---

## 38.9 LangGraph Multi-Agent

LangGraph models agents as nodes in a directed graph with conditional edges.

```python
# Requires: pip install langgraph langchain-openai
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    task: str
    research: str
    draft: str
    critique: str
    revision_count: int
    messages: Annotated[list, operator.add]

def research_node(state: AgentState) -> AgentState:
    # In practice: call an LLM with research tools
    state["research"] = f"Research on: {state['task']}"
    state["messages"].append({"role": "research", "content": state["research"]})
    return state

def writer_node(state: AgentState) -> AgentState:
    state["draft"] = f"Draft using: {state['research']}"
    state["messages"].append({"role": "writer", "content": state["draft"]})
    return state

def critic_node(state: AgentState) -> AgentState:
    state["critique"] = "Needs more depth in section 2."
    state["revision_count"] = state.get("revision_count", 0) + 1
    return state

def should_revise(state: AgentState) -> str:
    if state.get("revision_count", 0) < 2 and "needs more" in state.get("critique", "").lower():
        return "writer"
    return END

builder = StateGraph(AgentState)
builder.add_node("research", research_node)
builder.add_node("writer", writer_node)
builder.add_node("critic", critic_node)
builder.set_entry_point("research")
builder.add_edge("research", "writer")
builder.add_edge("writer", "critic")
builder.add_conditional_edges("critic", should_revise, {"writer": "writer", END: END})
graph = builder.compile()
```

---

## 38.10 AutoGen GroupChat

AutoGen's `GroupChat` allows multiple agents to take turns in a round-table format.

```python
# Requires: pip install pyautogen
import autogen

llm_config = {"model": "gpt-4o-mini", "api_key": "YOUR_KEY"}

researcher = autogen.AssistantAgent(
    name="Researcher",
    system_message="You are an expert researcher. Find and summarise relevant information.",
    llm_config=llm_config,
)

coder = autogen.AssistantAgent(
    name="Coder",
    system_message="You are an expert Python programmer. Write clean, tested code.",
    llm_config=llm_config,
)

critic = autogen.AssistantAgent(
    name="Critic",
    system_message="You review outputs for quality, accuracy, and safety. Be constructive.",
    llm_config=llm_config,
)

user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    code_execution_config={"work_dir": "coding", "use_docker": False},
)

groupchat = autogen.GroupChat(
    agents=[user_proxy, researcher, coder, critic],
    messages=[],
    max_round=8,
    speaker_selection_method="auto",
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
```

---

## 38.11 Trust & Verification

Agents should not blindly trust each other's outputs.

```python
@dataclass
class VerifyingAgent:
    """
    Wraps an agent with an independent verifier that checks outputs
    before they are passed downstream.
    """

    inner_agent: Callable[[str], str]
    verifier: Callable[[str, str], tuple[bool, str]]  # (input, output) -> (ok, reason)
    fallback_agent: Callable[[str], str] | None = None

    def __call__(self, input_text: str) -> str:
        output = self.inner_agent(input_text)
        ok, reason = self.verifier(input_text, output)
        if ok:
            return output
        if self.fallback_agent:
            return self.fallback_agent(input_text)
        raise RuntimeError(f"Agent output failed verification: {reason}")
```

---

## 38.12 Coordinator Flow Diagram

<div class="diagram">
  <div class="diagram-title">Hierarchical Multi-Agent Flow</div>
  <div class="flow">
    <div class="flow-node accent wide">User Task</div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node blue wide">Coordinator Agent<br/><small>decomposes task, assigns sub-tasks</small></div>
    <div class="flow-h">
      <div class="flow-node green">Research Agent<br/><small>retrieves facts</small></div>
      <div class="flow-node purple">Code Agent<br/><small>writes &amp; tests code</small></div>
      <div class="flow-node orange">Critic Agent<br/><small>reviews outputs</small></div>
    </div>
    <div class="flow-arrow accent">↓ results ↓</div>
    <div class="flow-node teal wide">Synthesiser Agent<br/><small>merges sub-results into final response</small></div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node accent">Final Output</div>
  </div>
</div>

---

## 38.13 Comparison Table

<table class="compare-table">
  <thead>
    <tr>
      <th>Pattern</th>
      <th>Coupling</th>
      <th>Scalability</th>
      <th>Observability</th>
      <th>Best For</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Message Passing</strong></td>
      <td>Loose (typed msgs)</td>
      <td>High (async queues)</td>
      <td>High (message log)</td>
      <td>Sequential pipelines</td>
    </tr>
    <tr>
      <td><strong>Blackboard</strong></td>
      <td>Very loose</td>
      <td>Medium</td>
      <td>High (shared log)</td>
      <td>Opportunistic collaboration</td>
    </tr>
    <tr>
      <td><strong>Pub-Sub</strong></td>
      <td>Minimal</td>
      <td>Very high</td>
      <td>Medium (event stream)</td>
      <td>Event-driven pipelines</td>
    </tr>
    <tr>
      <td><strong>Shared State</strong></td>
      <td>Tight (locks)</td>
      <td>Low (contention)</td>
      <td>Low</td>
      <td>Concurrent state updates</td>
    </tr>
  </tbody>
</table>

---

*Last updated: May 2026*
