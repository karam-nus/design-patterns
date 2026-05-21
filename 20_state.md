---
title: "Chapter 20 — State Machine & Lifecycle"
---

[← Back to Table of Contents](./README.md)

# Chapter 20 — State Machine & Lifecycle

> *"An object's behaviour should be a function of its state. Make the states explicit, and the transitions become the design."*

ML systems are inherently stateful. A model is untrained, then training, then evaluating, then serving. A training loop cycles through warmup, training, validation, and checkpointing. An LLM request passes through prefill and decode phases. Keeping this state implicit — scattered through boolean flags and nested `if/elif` chains — creates bugs and makes adding new states expensive. The **State** pattern and its cousin, the **Finite State Machine**, make lifecycle transitions first-class citizens.

<span class="badge behavioral">Behavioral</span>

---

## Intent

Allow an object to **alter its behaviour when its internal state changes**. The object will appear to change its class.

Instead of one class with many `if state == X` branches, each state becomes its own class (or enum value) with its own behaviour. The context delegates to the current state object.

---

## UML Structure

<div class="diagram">
  <div class="diagram-title">State Pattern — Class Structure</div>
  <div class="uml-row">
    <div class="uml-box">
      <div class="uml-title">Context</div>
      <div class="uml-section">
        <div class="uml-item">– _state: State</div>
      </div>
      <div class="uml-section">
        <div class="uml-item">+ transition_to(state: State)</div>
        <div class="uml-item">+ request()</div>
      </div>
    </div>
    <div class="uml-box">
      <div class="uml-title">«abstract» State</div>
      <div class="uml-section">
        <div class="uml-item">– context: Context</div>
      </div>
      <div class="uml-section">
        <div class="uml-item">+ handle(context)</div>
        <div class="uml-item">+ on_enter(context)</div>
        <div class="uml-item">+ on_exit(context)</div>
      </div>
    </div>
  </div>
  <div class="uml-row">
    <div class="uml-box">
      <div class="uml-title">ConcreteStateA</div>
      <div class="uml-section">
        <div class="uml-item">+ handle(ctx) → transition to B</div>
      </div>
    </div>
    <div class="uml-box">
      <div class="uml-title">ConcreteStateB</div>
      <div class="uml-section">
        <div class="uml-item">+ handle(ctx) → stay / transition</div>
      </div>
    </div>
  </div>
</div>

---

## Finite State Machine Fundamentals

A **Finite State Machine (FSM)** is a formal model of an object's lifecycle defined by:

| Concept | Description | ML example |
|---|---|---|
| **State** | Discrete mode of operation | TRAINING, EVALUATING |
| **Event / Trigger** | Something that causes a transition | `epoch_end`, `eval_complete` |
| **Transition** | (state, event) → new state | TRAINING + `eval_trigger` → EVALUATING |
| **Guard** | Condition that must hold for transition | `val_loss improved` |
| **Action** | Side effect executed on transition | `save_checkpoint()` |
| **Entry action** | Runs on entering a state | `enable_gradient_computation()` |
| **Exit action** | Runs on leaving a state | `log_epoch_metrics()` |

### State Transition Table

<table class="compare-table">
  <thead>
    <tr>
      <th>From State</th>
      <th>Event</th>
      <th>Guard</th>
      <th>To State</th>
      <th>Action</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>UNINITIALIZED</td>
      <td>load()</td>
      <td>weights file exists</td>
      <td>LOADED</td>
      <td>load_weights()</td>
    </tr>
    <tr>
      <td>LOADED</td>
      <td>start_training()</td>
      <td>—</td>
      <td>TRAINING</td>
      <td>setup_optimizer()</td>
    </tr>
    <tr>
      <td>TRAINING</td>
      <td>eval_trigger()</td>
      <td>—</td>
      <td>EVALUATING</td>
      <td>disable_dropout()</td>
    </tr>
    <tr>
      <td>EVALUATING</td>
      <td>eval_done(improved)</td>
      <td>improved = True</td>
      <td>TRAINING</td>
      <td>save_checkpoint()</td>
    </tr>
    <tr>
      <td>EVALUATING</td>
      <td>eval_done(improved)</td>
      <td>improved = False</td>
      <td>TRAINING</td>
      <td>increment_patience()</td>
    </tr>
    <tr>
      <td>TRAINING</td>
      <td>training_complete()</td>
      <td>—</td>
      <td>SERVING</td>
      <td>load_best_checkpoint()</td>
    </tr>
    <tr>
      <td>SERVING</td>
      <td>deprecate()</td>
      <td>newer model deployed</td>
      <td>DEPRECATED</td>
      <td>log_deprecation()</td>
    </tr>
  </tbody>
</table>

---

## Model Lifecycle FSM

```python
from __future__ import annotations
import abc
import time
from enum import Enum, auto
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class ModelState(Enum):
    UNINITIALIZED = auto()
    LOADED        = auto()
    TRAINING      = auto()
    EVALUATING    = auto()
    SERVING       = auto()
    DEPRECATED    = auto()


# ---------------------------------------------------------------------------
# Transition guard + action types
# ---------------------------------------------------------------------------

Guard  = Callable[["ModelLifecycle"], bool]
Action = Callable[["ModelLifecycle"], None]


# ---------------------------------------------------------------------------
# Transition descriptor
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field as dc_field

@dataclass
class Transition:
    event: str
    from_state: ModelState
    to_state: ModelState
    guard: Guard | None = None
    action: Action | None = None


# ---------------------------------------------------------------------------
# Context — Model Lifecycle
# ---------------------------------------------------------------------------

class ModelLifecycle:
    """
    Context + FSM for an ML model's full lifecycle.

    States:
        UNINITIALIZED → LOADED → TRAINING ↔ EVALUATING → SERVING → DEPRECATED
    """

    _transitions: list[Transition] = []  # class-level registry

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.state: ModelState = ModelState.UNINITIALIZED
        self.checkpoint_path: str | None = None
        self.epoch: int = 0
        self.best_val_loss: float = float("inf")
        self.patience_counter: int = 0
        self._history: list[tuple[float, ModelState, str]] = []
        self._register_transitions()

    # ------------------------------------------------------------------
    # FSM core
    # ------------------------------------------------------------------

    def _register_transitions(self) -> None:
        self._transitions = [
            Transition("load",             ModelState.UNINITIALIZED, ModelState.LOADED,
                       action=self._on_load),
            Transition("start_training",   ModelState.LOADED,        ModelState.TRAINING,
                       action=self._on_start_training),
            Transition("eval_trigger",     ModelState.TRAINING,      ModelState.EVALUATING,
                       action=self._on_eval_trigger),
            Transition("eval_done",        ModelState.EVALUATING,    ModelState.TRAINING,
                       action=self._on_eval_done),
            Transition("training_complete",ModelState.TRAINING,      ModelState.SERVING,
                       action=self._on_training_complete),
            Transition("training_complete",ModelState.EVALUATING,    ModelState.SERVING,
                       action=self._on_training_complete),
            Transition("deprecate",        ModelState.SERVING,       ModelState.DEPRECATED,
                       action=self._on_deprecate),
        ]

    def fire(self, event: str, **kwargs) -> None:
        """Fire an event, triggering a state transition if valid."""
        for t in self._transitions:
            if t.event == event and t.from_state == self.state:
                if t.guard is not None and not t.guard(self):
                    raise RuntimeError(
                        f"Guard failed for event '{event}' in state {self.state.name}"
                    )
                old = self.state
                self.state = t.to_state
                self._history.append((time.time(), self.state, event))
                print(
                    f"[{self.model_id}] {old.name} --{event}--> {self.state.name}"
                )
                if t.action:
                    t.action(self, **kwargs)
                return
        raise ValueError(
            f"No valid transition for event '{event}' in state {self.state.name}. "
            f"Allowed: {self._valid_events()}"
        )

    def _valid_events(self) -> list[str]:
        return [t.event for t in self._transitions if t.from_state == self.state]

    # ------------------------------------------------------------------
    # State-dependent behaviour
    # ------------------------------------------------------------------

    def predict(self, x):
        if self.state != ModelState.SERVING:
            raise RuntimeError(
                f"predict() requires SERVING state, got {self.state.name}"
            )
        return f"prediction for {x}"

    def train_step(self, batch):
        if self.state != ModelState.TRAINING:
            raise RuntimeError(
                f"train_step() requires TRAINING state, got {self.state.name}"
            )
        return {"loss": 0.42}

    def get_metrics(self) -> dict:
        return {
            "model_id": self.model_id,
            "state": self.state.name,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
        }

    # ------------------------------------------------------------------
    # Transition actions
    # ------------------------------------------------------------------

    def _on_load(self, path: str = "weights.pt", **_) -> None:
        self.checkpoint_path = path
        print(f"  → Loaded weights from {path}")

    def _on_start_training(self, **_) -> None:
        print(f"  → Optimizer initialised, dropout enabled")

    def _on_eval_trigger(self, **_) -> None:
        print(f"  → Dropout disabled for evaluation")

    def _on_eval_done(self, val_loss: float = 0.0, **_) -> None:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            print(f"  → New best! val_loss={val_loss:.4f}. Saving checkpoint.")
        else:
            self.patience_counter += 1
            print(
                f"  → No improvement. patience={self.patience_counter}, "
                f"best={self.best_val_loss:.4f}"
            )
        self.epoch += 1

    def _on_training_complete(self, **_) -> None:
        print(f"  → Loading best checkpoint for serving")

    def _on_deprecate(self, reason: str = "", **_) -> None:
        print(f"  → Deprecated. Reason: {reason or 'newer model deployed'}")

    # ------------------------------------------------------------------
    # Lifecycle history
    # ------------------------------------------------------------------

    def history(self) -> list[tuple[str, str, str]]:
        return [
            (
                time.strftime("%H:%M:%S", time.localtime(ts)),
                state.name,
                event,
            )
            for ts, state, event in self._history
        ]


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

model = ModelLifecycle("bert-finetuned-sst2")
model.fire("load", path="pretrained/bert.pt")
model.fire("start_training")

for epoch in range(5):
    # Simulate a training epoch
    model.fire("eval_trigger")
    val_loss = 0.50 - epoch * 0.03   # simulated improvement
    model.fire("eval_done", val_loss=val_loss)

model.fire("training_complete")
print(model.predict("great movie!"))
model.fire("deprecate", reason="bert-v2 deployed")
```

---

## Training Phase FSM

A more granular FSM governing the phases *within* a single training run.

```python
class TrainingPhase(Enum):
    WARMUP     = auto()
    TRAINING   = auto()
    VALIDATION = auto()
    CHECKPOINT = auto()
    EARLY_STOP = auto()


class TrainingFSM:
    """FSM governing transitions between training phases."""

    def __init__(
        self,
        warmup_epochs: int = 3,
        patience: int = 5,
        eval_every: int = 1,
    ) -> None:
        self.phase = TrainingPhase.WARMUP
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.eval_every = eval_every
        self.epoch = 0
        self.patience_counter = 0
        self.best_loss = float("inf")

    def on_epoch_end(self, train_loss: float, val_loss: float | None = None) -> TrainingPhase:
        """Drive FSM transitions at the end of each epoch."""
        self.epoch += 1

        if self.phase == TrainingPhase.WARMUP:
            if self.epoch >= self.warmup_epochs:
                self._transition_to(TrainingPhase.TRAINING)
            return self.phase

        if self.phase == TrainingPhase.TRAINING:
            if val_loss is not None and self.epoch % self.eval_every == 0:
                self._transition_to(TrainingPhase.VALIDATION)
                return self._handle_validation(val_loss)
            return self.phase

        return self.phase

    def _handle_validation(self, val_loss: float) -> TrainingPhase:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            self._transition_to(TrainingPhase.CHECKPOINT)
            # After saving checkpoint, return to TRAINING
            self._transition_to(TrainingPhase.TRAINING)
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self._transition_to(TrainingPhase.EARLY_STOP)
            else:
                self._transition_to(TrainingPhase.TRAINING)
        return self.phase

    def _transition_to(self, new_phase: TrainingPhase) -> None:
        print(f"  Phase: {self.phase.name} → {new_phase.name}")
        self.phase = new_phase

    @property
    def is_done(self) -> bool:
        return self.phase == TrainingPhase.EARLY_STOP


# Run the training FSM
fsm = TrainingFSM(warmup_epochs=2, patience=3)
val_losses = [0.9, 0.8, 0.7, 0.65, 0.64, 0.64, 0.64, 0.64]

for i, val_loss in enumerate(val_losses):
    phase = fsm.on_epoch_end(train_loss=0.5, val_loss=val_loss)
    print(f"Epoch {fsm.epoch}: phase={phase.name}")
    if fsm.is_done:
        print("Early stopping triggered.")
        break
```

---

## LLM Request Lifecycle FSM

```python
class LLMRequestState(Enum):
    IDLE    = auto()
    PREFILL = auto()
    DECODE  = auto()
    DONE    = auto()
    ERROR   = auto()


class LLMRequestFSM:
    """Models the lifecycle of a single LLM inference request."""

    _valid_transitions: dict[LLMRequestState, set[LLMRequestState]] = {
        LLMRequestState.IDLE:    {LLMRequestState.PREFILL},
        LLMRequestState.PREFILL: {LLMRequestState.DECODE, LLMRequestState.ERROR},
        LLMRequestState.DECODE:  {LLMRequestState.DONE,   LLMRequestState.ERROR},
        LLMRequestState.DONE:    set(),
        LLMRequestState.ERROR:   set(),
    }

    def __init__(self, request_id: str, prompt: str) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.state = LLMRequestState.IDLE
        self.prompt_tokens: list[int] = []
        self.generated_tokens: list[int] = []
        self.error_message: str | None = None

    def _transition(self, new_state: LLMRequestState) -> None:
        allowed = self._valid_transitions[self.state]
        if new_state not in allowed:
            raise ValueError(
                f"Invalid transition {self.state.name} → {new_state.name}"
            )
        print(f"[{self.request_id}] {self.state.name} → {new_state.name}")
        self.state = new_state

    def start_prefill(self, tokenized_input: list[int]) -> None:
        """Begin KV-cache computation from the prompt."""
        self._transition(LLMRequestState.PREFILL)
        self.prompt_tokens = tokenized_input
        print(f"  Prefilling {len(tokenized_input)} tokens…")

    def start_decode(self) -> None:
        """Begin autoregressive generation."""
        self._transition(LLMRequestState.DECODE)

    def append_token(self, token_id: int, is_eos: bool = False) -> bool:
        """Append a generated token.  Returns True if generation complete."""
        if self.state != LLMRequestState.DECODE:
            raise RuntimeError("Not in DECODE state.")
        self.generated_tokens.append(token_id)
        if is_eos:
            self._transition(LLMRequestState.DONE)
            return True
        return False

    def fail(self, message: str) -> None:
        """Transition to ERROR state from any non-terminal state."""
        self.error_message = message
        self._transition(LLMRequestState.ERROR)

    @property
    def is_terminal(self) -> bool:
        return self.state in {LLMRequestState.DONE, LLMRequestState.ERROR}


# Simulate a request
req = LLMRequestFSM("req-abc123", "Translate to French: Hello world")
req.start_prefill([1, 234, 456, 789, 101])
req.start_decode()
for i, (tok, is_eos) in enumerate([(512, False), (513, False), (2, True)]):
    done = req.append_token(tok, is_eos=is_eos)
    if done:
        print(f"  Generated {len(req.generated_tokens)} tokens. Request complete.")
        break
```

---

## Python `enum.Enum` for States

Using Python's `enum.Enum` for states is idiomatic and provides several advantages: type safety, exhaustive matching, set membership tests, and clean `repr`.

```python
from enum import Enum, auto, Flag

# Simple state enum
class TrafficLight(Enum):
    RED    = auto()
    YELLOW = auto()
    GREEN  = auto()

# Flag enum: states can be combined (useful for capability flags)
class ModelCapabilities(Flag):
    NONE       = 0
    TRAIN      = auto()
    EVALUATE   = auto()
    SERVE      = auto()
    FINE_TUNE  = auto()
    FULL = TRAIN | EVALUATE | SERVE | FINE_TUNE

# State-specific configuration via Enum methods
class TrainingStateConfig(Enum):
    WARMUP     = {"lr_scale": 0.1, "dropout": 0.1}
    TRAINING   = {"lr_scale": 1.0, "dropout": 0.1}
    VALIDATION = {"lr_scale": 0.0, "dropout": 0.0}
    CHECKPOINT = {"lr_scale": 0.0, "dropout": 0.0}

    @property
    def lr_scale(self) -> float:
        return self.value["lr_scale"]

    @property
    def dropout_rate(self) -> float:
        return self.value["dropout"]

# Usage
state = TrainingStateConfig.VALIDATION
print(state.lr_scale)    # 0.0 — no gradient updates during eval
print(state.dropout_rate)  # 0.0 — no dropout during eval
```

---

## State Pattern vs Switch Statements

### Before — Imperative flags (anti-pattern)

```python
class ModelBefore:
    def __init__(self):
        self.is_loaded     = False
        self.is_training   = False
        self.is_evaluating = False
        self.is_serving    = False

    def predict(self, x):
        if not self.is_serving:
            if self.is_training:
                raise RuntimeError("Still training!")
            elif not self.is_loaded:
                raise RuntimeError("Not loaded!")
            elif self.is_evaluating:
                raise RuntimeError("Evaluating!")
        return "prediction"  # only reached if is_serving

    def start_eval(self):
        if not self.is_training:
            raise RuntimeError("Must be training to evaluate")
        self.is_training   = False
        self.is_evaluating = True
        # What about is_loaded? is_serving? Easy to forget to update one.
```

### After — State pattern (clean)

```python
class ModelAfter:
    def __init__(self):
        self.state = ModelState.UNINITIALIZED

    def predict(self, x):
        if self.state != ModelState.SERVING:
            raise RuntimeError(
                f"Cannot predict in state {self.state.name}. "
                f"Call fire('training_complete') first."
            )
        return "prediction"

    def start_eval(self):
        if self.state != ModelState.TRAINING:
            raise RuntimeError(f"Cannot start eval from {self.state.name}")
        self.state = ModelState.EVALUATING
        # Single, explicit transition — no forgotten flags
```

The State pattern replaces a tangle of boolean flags with a single `state` field and explicit, documented transitions. Illegal states become impossible because the enum simply doesn't have a value for them.

---

## `pytransitions` — Declarative FSMs

```python
from transitions import Machine


class TrainingProcess:
    """FSM defined declaratively using pytransitions."""

    states = [
        {"name": "warmup",     "on_enter": "log_warmup_start"},
        {"name": "training",   "on_enter": "enable_gradients"},
        {"name": "validation", "on_enter": "disable_dropout"},
        {"name": "checkpoint", "on_enter": "save_checkpoint"},
        {"name": "early_stop", "on_enter": "notify_early_stop"},
    ]

    transitions = [
        {"trigger": "warmup_done",    "source": "warmup",     "dest": "training"},
        {"trigger": "epoch_end",      "source": "training",   "dest": "validation"},
        {"trigger": "improved",       "source": "validation", "dest": "checkpoint"},
        {"trigger": "not_improved",   "source": "validation", "dest": "training",
         "conditions": "patience_ok"},
        {"trigger": "not_improved",   "source": "validation", "dest": "early_stop",
         "unless": "patience_ok"},
        {"trigger": "checkpoint_saved","source": "checkpoint","dest": "training"},
    ]

    def __init__(self, patience: int = 5) -> None:
        self.patience = patience
        self.patience_counter = 0
        self.machine = Machine(
            model=self,
            states=self.states,
            transitions=self.transitions,
            initial="warmup",
            auto_transitions=False,
        )

    def patience_ok(self) -> bool:
        return self.patience_counter < self.patience

    def log_warmup_start(self) -> None:
        print("[FSM] Warmup phase started")

    def enable_gradients(self) -> None:
        print("[FSM] Gradients enabled (training phase)")

    def disable_dropout(self) -> None:
        print("[FSM] Dropout disabled (validation phase)")

    def save_checkpoint(self) -> None:
        print("[FSM] Best checkpoint saved")
        self.patience_counter = 0

    def notify_early_stop(self) -> None:
        print("[FSM] Early stopping triggered")
```

---

## Agent State Management FSM

Agentic AI systems — ReAct agents, tool-calling agents, multi-agent frameworks — exhibit their own rich lifecycle.

```python
class AgentState(Enum):
    IDLE       = auto()
    PERCEIVING = auto()
    REASONING  = auto()
    ACTING     = auto()
    REFLECTING = auto()
    ERROR      = auto()


class AgentFSM:
    """FSM for a ReAct-style LLM agent."""

    _transitions: dict[AgentState, dict[str, AgentState]] = {
        AgentState.IDLE:       {"perceive": AgentState.PERCEIVING},
        AgentState.PERCEIVING: {"think":    AgentState.REASONING,
                                "fail":     AgentState.ERROR},
        AgentState.REASONING:  {"act":      AgentState.ACTING,
                                "finish":   AgentState.IDLE,
                                "fail":     AgentState.ERROR},
        AgentState.ACTING:     {"observe":  AgentState.PERCEIVING,
                                "reflect":  AgentState.REFLECTING,
                                "fail":     AgentState.ERROR},
        AgentState.REFLECTING: {"think":    AgentState.REASONING,
                                "finish":   AgentState.IDLE},
        AgentState.ERROR:      {"reset":    AgentState.IDLE},
    }

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.state = AgentState.IDLE
        self.step_count = 0
        self.max_steps = 20
        self.scratchpad: list[dict] = []

    def fire(self, event: str, payload: dict | None = None) -> AgentState:
        allowed = self._transitions.get(self.state, {})
        if event not in allowed:
            raise ValueError(
                f"Event '{event}' not valid in state {self.state.name}. "
                f"Allowed: {list(allowed.keys())}"
            )
        new_state = allowed[event]
        print(f"[{self.agent_id}] {self.state.name} --{event}--> {new_state.name}")
        self.state = new_state
        self.step_count += 1
        if payload:
            self.scratchpad.append({"step": self.step_count, "event": event, **payload})
        if self.step_count >= self.max_steps:
            print(f"[{self.agent_id}] Max steps reached. Forcing IDLE.")
            self.state = AgentState.IDLE
        return self.state

    def run_step(self, observation: str, thought: str, action: str) -> None:
        """Simulate one ReAct loop iteration."""
        self.fire("perceive",  {"observation": observation})
        self.fire("think",     {"thought": thought})
        self.fire("act",       {"action": action})


# Simulate a 3-step agent loop
agent = AgentFSM("research-agent-01")
agent.run_step(
    "User asks: What is the capital of France?",
    "I need to look this up in my knowledge base.",
    "search(query='capital of France')",
)
agent.fire("reflect", {"reflection": "Answer is Paris."})
agent.fire("finish")
print(f"Agent final state: {agent.state.name}")
```

---

## Flow Diagram — Model Lifecycle Transitions

<div class="diagram">
  <div class="diagram-title">Model Lifecycle State Transitions</div>
  <div class="flow">
    <div class="flow-node accent wide">UNINITIALIZED</div>
    <div class="flow-arrow accent">▼ load() → load_weights()</div>
    <div class="flow-node green wide">LOADED</div>
    <div class="flow-arrow green">▼ start_training() → setup_optimizer()</div>
    <div class="flow-node blue wide">TRAINING</div>
    <div class="flow-h">
      <div class="flow-arrow">▼ eval_trigger()</div>
      <div class="flow-node purple">EVALUATING</div>
      <div class="flow-arrow purple">▼ eval_done()</div>
    </div>
    <div class="flow-node blue wide">TRAINING  ← (loop back)</div>
    <div class="flow-arrow accent">▼ training_complete() → load_best_ckpt()</div>
    <div class="flow-node teal wide">SERVING</div>
    <div class="flow-arrow">▼ deprecate(reason)</div>
    <div class="flow-node red wide">DEPRECATED</div>
  </div>
</div>

<div class="diagram">
  <div class="diagram-title">Agent ReAct Loop State Transitions</div>
  <div class="flow">
    <div class="flow-node accent wide">IDLE</div>
    <div class="flow-arrow accent">▼ perceive</div>
    <div class="flow-node green wide">PERCEIVING</div>
    <div class="flow-arrow green">▼ think</div>
    <div class="flow-node blue wide">REASONING</div>
    <div class="flow-h">
      <div class="flow-arrow">▼ act</div>
      <div class="flow-node orange">ACTING</div>
      <div class="flow-arrow orange">▼ observe (loop) / reflect</div>
    </div>
    <div class="flow-node purple wide">REFLECTING</div>
    <div class="flow-arrow purple">▼ think (loop) / finish</div>
    <div class="flow-node accent wide">IDLE  (reset)</div>
  </div>
</div>

---

## Comparison Table

<table class="compare-table">
  <thead>
    <tr>
      <th>Dimension</th>
      <th>State Pattern (GoF)</th>
      <th>Strategy Pattern</th>
      <th>FSM Library (pytransitions)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>State representation</td>
      <td>Concrete state objects</td>
      <td>Algorithm objects</td>
      <td>Strings / enums</td>
    </tr>
    <tr>
      <td>Transition logic</td>
      <td>In state classes</td>
      <td>N/A — no transitions</td>
      <td>Declarative transition table</td>
    </tr>
    <tr>
      <td>Guard conditions</td>
      <td>Manual in state methods</td>
      <td>N/A</td>
      <td>Declarative `conditions`</td>
    </tr>
    <tr>
      <td>Entry/exit actions</td>
      <td>on_enter / on_exit methods</td>
      <td>N/A</td>
      <td>Declarative `on_enter` / `on_exit`</td>
    </tr>
    <tr>
      <td>Boilerplate</td>
      <td>High (one class per state)</td>
      <td>Medium</td>
      <td>Low (config-driven)</td>
    </tr>
    <tr>
      <td>Best for</td>
      <td>Complex per-state behaviour</td>
      <td>Single swappable algorithm</td>
      <td>Many states, simple behaviour</td>
    </tr>
    <tr>
      <td>ML use case</td>
      <td>Model lifecycle, agent FSM</td>
      <td>Loss function selection</td>
      <td>Training phase FSM, LLM lifecycle</td>
    </tr>
  </tbody>
</table>

---

## Key Takeaways

<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">🗺️</div>
    <div class="card-title">Make States Explicit</div>
    <div class="card-desc">Replace boolean flags (<code>is_training</code>, <code>is_loaded</code>) with a single <code>state: ModelState</code> enum. Illegal states become impossible to represent.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🚦</div>
    <div class="card-title">Guard Transitions</div>
    <div class="card-desc">Transitions with guards encode business rules declaratively — "only save checkpoint when val_loss improves" — rather than burying them in conditional logic.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🤖</div>
    <div class="card-title">Agent Lifecycles</div>
    <div class="card-desc">Agentic AI systems have rich, non-linear lifecycles. An explicit FSM prevents agents from acting before perceiving or reasoning without an observation.</div>
  </div>
</div>

---

*Last updated: May 2026*
