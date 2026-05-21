---
title: "Chapter 33 — ReAct & Chain-of-Thought"
---

[← Back to Table of Contents](./README.md)

# Chapter 33 — ReAct & Chain-of-Thought

> *"Language models become dramatically more capable when they are allowed to think out loud before they act."*

The ReAct framework (Reason + Act, Yao et al. 2022) changed how we build LLM agents. Rather than asking a model to produce a final answer in one shot, we give it a scratchpad: it reasons, takes an action, observes the result, and reasons again. Combined with Chain-of-Thought prompting, this simple loop unlocks tool use, multi-step planning, and self-correction in a single unified architecture.

<div class="callout info">
<span class="callout-icon">ℹ️</span>
<div class="callout-body">
ReAct = <strong>Re</strong>asoning + <strong>Act</strong>ing. The core insight: interleaving language reasoning traces with discrete actions outperforms either reasoning alone (CoT) or acting alone (tool-use chains) on complex tasks.
</div>
</div>

---

## The Core ReAct Loop

<div class="diagram">
<div class="diagram-title">ReAct Thought / Action / Observation Cycle</div>
<div class="flow">
  <div class="flow-node blue wide">User Query</div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-node purple wide">Thought<br/><small>CoT scratchpad reasoning</small></div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-node orange wide">Action<br/><small>tool call: search / execute / retrieve</small></div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-node teal wide">Observation<br/><small>tool result injected into context</small></div>
  <div class="flow-arrow accent">↓ more reasoning needed?</div>
  <div class="flow-node green wide">Final Answer<br/><small>when reasoning is complete</small></div>
</div>
</div>

The loop repeats until the model emits a `Final Answer` action or a step budget is exhausted. Each cycle adds a new `Thought / Action / Observation` triple to the context, giving the model a growing reasoning trace.

---

## Chain-of-Thought Prompting

CoT is the foundation that makes the Thought step useful. Instead of answering directly, the model narrates its reasoning.

### Zero-Shot CoT

Appending **"Let's think step by step."** to a prompt reliably elicits intermediate reasoning without any examples.

```python
from openai import OpenAI

client = OpenAI()


def zero_shot_cot(question: str, model: str = "gpt-4o") -> str:
    """Zero-shot chain-of-thought: append trigger phrase."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"{question}\n\nLet's think step by step.",
            }
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


# Example
answer = zero_shot_cot(
    "A train leaves Chicago at 9 AM travelling at 80 mph toward New York, "
    "460 miles away. When does it arrive?"
)
print(answer)
```

### Few-Shot CoT

Provide worked examples in the prompt to prime the reasoning style.

```python
FEW_SHOT_COT_PROMPT = """\
Solve each problem by thinking step by step, then state the answer on a final line starting with "Answer:".

Problem: Roger has 5 tennis balls. He buys 2 cans of tennis balls. Each can has 3 balls. How many does he have?
Thought: Roger starts with 5. He buys 2 × 3 = 6 more. Total = 5 + 6 = 11.
Answer: 11

Problem: The cafeteria had 23 apples. They used 20 for lunch. They bought 6 more. How many apples do they have?
Thought: Start with 23. Remove 20 → 3 remaining. Add 6 → 9 total.
Answer: 9

Problem: {problem}
Thought:"""


def few_shot_cot(problem: str, model: str = "gpt-4o") -> str:
    prompt = FEW_SHOT_COT_PROMPT.format(problem=problem)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        stop=["Problem:"],
    )
    return response.choices[0].message.content
```

---

## ReAct Implementation

A full ReAct agent loop with thought parsing, tool dispatch, and observation injection.

```python
import re
import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)
client = OpenAI()


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]

    def __call__(self, input_str: str) -> str:
        return self.func(input_str)


@dataclass
class ReActStep:
    thought: str
    action: str
    action_input: str
    observation: str = ""


@dataclass
class ReActTrace:
    query: str
    steps: list[ReActStep] = field(default_factory=list)
    final_answer: str = ""


REACT_SYSTEM_PROMPT = """\
You are a helpful assistant that solves problems by reasoning and using tools.

Format your response strictly as:
Thought: <your reasoning about what to do next>
Action: <tool name>
Action Input: <input to the tool>

When you have enough information to answer, use:
Thought: I now know the final answer.
Action: Final Answer
Action Input: <your complete answer>

Available tools:
{tools}
"""

REACT_OBSERVATION_TEMPLATE = "Observation: {observation}\n"


class ReActAgent:
    """
    Implements the ReAct (Reason + Act) loop.

    Args:
        tools: list of Tool objects the agent may call
        model: LLM to use
        max_steps: safety budget — max thought/action/observation cycles
        temperature: generation temperature (0 for deterministic)
    """

    def __init__(
        self,
        tools: list[Tool],
        model: str = "gpt-4o",
        max_steps: int = 10,
        temperature: float = 0.0,
    ):
        self.tools = {t.name: t for t in tools}
        self.model = model
        self.max_steps = max_steps
        self.temperature = temperature

    def _build_system_prompt(self) -> str:
        tool_descriptions = "\n".join(
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        )
        return REACT_SYSTEM_PROMPT.format(tools=tool_descriptions)

    def _parse_llm_output(self, text: str) -> tuple[str, str, str]:
        """Extract Thought, Action, Action Input from model output."""
        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", text, re.DOTALL)
        action_match  = re.search(r"Action:\s*(.+?)(?=\nAction Input:|\Z)", text, re.DOTALL)
        input_match   = re.search(r"Action Input:\s*(.+?)(?=\nObservation:|\Z)", text, re.DOTALL)

        thought      = thought_match.group(1).strip() if thought_match else ""
        action       = action_match.group(1).strip()  if action_match  else ""
        action_input = input_match.group(1).strip()   if input_match   else ""
        return thought, action, action_input

    def run(self, query: str) -> ReActTrace:
        trace = ReActTrace(query=query)
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user",   "content": query},
        ]

        for step_num in range(self.max_steps):
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            llm_output = response.choices[0].message.content
            thought, action, action_input = self._parse_llm_output(llm_output)

            logger.debug("Step %d | Action: %s | Input: %s", step_num + 1, action, action_input)

            if action == "Final Answer":
                trace.final_answer = action_input
                messages.append({"role": "assistant", "content": llm_output})
                break

            # Dispatch tool call
            tool = self.tools.get(action)
            if tool is None:
                observation = f"Error: unknown tool '{action}'. Available: {list(self.tools)}"
            else:
                try:
                    observation = tool(action_input)
                except Exception as exc:
                    observation = f"Tool error: {exc}"

            step = ReActStep(thought, action, action_input, observation)
            trace.steps.append(step)

            obs_text = REACT_OBSERVATION_TEMPLATE.format(observation=observation)
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({"role": "user",      "content": obs_text})
        else:
            trace.final_answer = "Maximum steps reached without a final answer."

        return trace
```

### Tool Definitions

```python
import requests


def web_search(query: str) -> str:
    """Mock web search — replace with real search API."""
    return f"[Search results for '{query}']: Python was created by Guido van Rossum in 1991."


def calculator(expression: str) -> str:
    """Safe arithmetic evaluator."""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: unsafe expression"
        return str(eval(expression))  # noqa: S307 — guarded above
    except Exception as exc:
        return f"Calculation error: {exc}"


def python_repl(code: str) -> str:
    """Execute simple Python and return stdout."""
    import io, contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": {}})  # noqa: S102
        return buf.getvalue() or "(no output)"
    except Exception as exc:
        return f"Error: {exc}"


tools = [
    Tool("Search",     "Search the web for factual information", web_search),
    Tool("Calculator", "Evaluate a math expression",             calculator),
    Tool("Python",     "Run a Python code snippet",              python_repl),
]

agent = ReActAgent(tools=tools, model="gpt-4o", max_steps=8)
trace = agent.run("What is the square root of the year Python was first released?")
print(trace.final_answer)
```

---

## Tool Integration — Parsing and Dispatching

Robust tool dispatch requires graceful handling of malformed model output.

```python
import json
from typing import Any


class RobustToolDispatcher:
    """
    Handles JSON, plain-text, and partial tool-call formats from LLM output.
    """

    def __init__(self, tools: dict[str, Tool]):
        self.tools = tools

    def dispatch(self, action: str, raw_input: str) -> str:
        tool = self.tools.get(action)
        if tool is None:
            return f"Error: tool '{action}' not found"

        # Try to parse as JSON for structured inputs
        parsed_input = self._parse_input(raw_input)
        try:
            if isinstance(parsed_input, dict):
                return tool(**parsed_input)
            return tool(str(parsed_input))
        except TypeError as exc:
            return f"Tool input error: {exc}"

    def _parse_input(self, raw: str) -> Any:
        raw = raw.strip()
        if raw.startswith("{") or raw.startswith("["):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
        return raw
```

---

## Self-Correction Pattern

The model detects an error in its own reasoning and regenerates.

```python
SELF_CORRECTION_PROMPT = """\
Review your previous response and check for errors.

Previous response:
{previous_response}

Question: Is this response correct and complete? If not, provide a corrected response.
If it IS correct, reply with exactly: CONFIRMED: {previous_response}
"""


def self_correct(
    original_query: str,
    original_response: str,
    model: str = "gpt-4o",
    max_corrections: int = 2,
) -> str:
    current = original_response
    for _ in range(max_corrections):
        check_prompt = SELF_CORRECTION_PROMPT.format(previous_response=current)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": original_query},
                {"role": "assistant", "content": current},
                {"role": "user", "content": check_prompt},
            ],
            temperature=0.0,
        )
        reply = response.choices[0].message.content.strip()
        if reply.startswith("CONFIRMED:"):
            break
        current = reply
    return current
```

---

## Tree of Thought (ToT)

ToT extends CoT by exploring multiple reasoning branches and selecting the most promising one.

<div class="diagram">
<div class="diagram-title">Tree of Thought — Branching Reasoning</div>
<div class="flow">
  <div class="flow-node blue wide">Problem</div>
  <div class="flow-arrow accent">↓ generate N candidate thoughts</div>
</div>
<div class="flow">
  <div class="flow-node green">Thought A<br/><small>score: 0.9</small></div>
  <div class="flow-node orange">Thought B<br/><small>score: 0.6</small></div>
  <div class="flow-node purple">Thought C<br/><small>score: 0.4</small></div>
</div>
<div class="flow">
  <div class="flow-arrow green">↓ expand best</div>
  <div class="flow-arrow accent">↓ prune</div>
  <div class="flow-arrow accent">↓ prune</div>
</div>
<div class="flow">
  <div class="flow-node green">A.1 <small>0.95</small></div>
  <div class="flow-node teal">A.2 <small>0.80</small></div>
</div>
<div class="flow">
  <div class="flow-arrow green">↓ select best path</div>
</div>
<div class="flow">
  <div class="flow-node blue wide">Final Answer</div>
</div>
</div>

```python
from dataclasses import dataclass, field
import heapq


@dataclass
class ThoughtNode:
    thought: str
    score: float
    depth: int
    parent: Optional["ThoughtNode"] = None
    children: list["ThoughtNode"] = field(default_factory=list)

    def __lt__(self, other):
        return self.score > other.score  # max-heap semantics


class TreeOfThought:
    """
    Breadth-first tree search over reasoning steps.

    Args:
        breadth: candidate thoughts generated per node
        depth: maximum reasoning depth
        beam_width: top-k nodes to expand at each level
    """

    def __init__(
        self,
        breadth: int = 3,
        depth: int = 4,
        beam_width: int = 2,
        model: str = "gpt-4o",
    ):
        self.breadth = breadth
        self.depth = depth
        self.beam_width = beam_width
        self.model = model

    def solve(self, problem: str) -> str:
        root = ThoughtNode(thought=problem, score=1.0, depth=0)
        frontier = [root]

        for level in range(self.depth):
            next_frontier = []
            for node in frontier:
                candidates = self._generate_thoughts(node.thought, problem)
                scored = [(self._score_thought(t, problem), t) for t in candidates]
                for score, thought in scored:
                    child = ThoughtNode(thought, score, level + 1, parent=node)
                    node.children.append(child)
                    next_frontier.append(child)

            # Keep top beam_width nodes
            frontier = heapq.nlargest(self.beam_width, next_frontier)

            # Check if any node has reached a final answer
            for node in frontier:
                if self._is_terminal(node.thought):
                    return self._extract_answer(node.thought)

        # Return best leaf node's thought
        best = max(frontier, key=lambda n: n.score)
        return self._extract_answer(best.thought)

    def _generate_thoughts(self, current: str, problem: str) -> list[str]:
        response = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": (
                    f"Problem: {problem}\nCurrent reasoning: {current}\n\n"
                    f"Generate {self.breadth} distinct next reasoning steps. "
                    f"Number each step 1. 2. 3."
                ),
            }],
            temperature=0.7,
        )
        text = response.choices[0].message.content
        steps = re.findall(r"\d+\.\s*(.+?)(?=\n\d+\.|\Z)", text, re.DOTALL)
        return [s.strip() for s in steps[:self.breadth]]

    def _score_thought(self, thought: str, problem: str) -> float:
        response = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": (
                    f"Rate this reasoning step for solving: '{problem}'\n"
                    f"Step: {thought}\n\n"
                    f"Rate 0.0 (poor) to 1.0 (excellent). Reply with just the number."
                ),
            }],
            temperature=0.0,
        )
        try:
            return float(response.choices[0].message.content.strip())
        except ValueError:
            return 0.5

    def _is_terminal(self, thought: str) -> bool:
        return "final answer" in thought.lower() or "therefore" in thought.lower()

    def _extract_answer(self, thought: str) -> str:
        return thought
```

---

## Scratchpad Reasoning Pattern

Maintain intermediate state across multiple model calls without full history.

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Scratchpad:
    """Persistent intermediate reasoning state for multi-step problems."""
    facts: dict[str, Any] = field(default_factory=dict)
    steps: list[str]      = field(default_factory=list)
    errors: list[str]     = field(default_factory=list)

    def note(self, key: str, value: Any):
        self.facts[key] = value
        self.steps.append(f"Learned: {key} = {value}")

    def add_step(self, description: str):
        self.steps.append(description)

    def add_error(self, error: str):
        self.errors.append(error)
        self.steps.append(f"Error encountered: {error}")

    def to_context(self) -> str:
        lines = ["## Scratchpad"]
        if self.facts:
            lines.append("Known facts:")
            for k, v in self.facts.items():
                lines.append(f"  - {k}: {v}")
        if self.steps:
            lines.append("Steps taken:")
            for step in self.steps[-5:]:  # last 5 only
                lines.append(f"  - {step}")
        return "\n".join(lines)


class ScratchpadAgent:
    def __init__(self, tools: dict[str, Tool], model: str = "gpt-4o"):
        self.tools = tools
        self.model = model

    def run(self, query: str) -> str:
        pad = Scratchpad()
        for _ in range(10):
            context = pad.to_context()
            prompt = f"{context}\n\nQuery: {query}\n\nNext step:"
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            step_text = response.choices[0].message.content.strip()

            if step_text.upper().startswith("FINAL:"):
                return step_text[6:].strip()

            # Parse and execute any tool calls embedded in the step
            tool_match = re.search(r"TOOL\((\w+)\):\s*(.+)", step_text)
            if tool_match:
                tool_name, tool_input = tool_match.groups()
                tool = self.tools.get(tool_name)
                if tool:
                    result = tool(tool_input)
                    pad.note(f"{tool_name}({tool_input[:30]})", result[:100])
                else:
                    pad.add_error(f"Unknown tool: {tool_name}")
            else:
                pad.add_step(step_text[:100])

        return "Max steps reached"
```

---

## Step-by-Step Prompting Templates

```python
from string import Template


STEP_BY_STEP_TEMPLATES = {
    "analysis": Template("""\
You are an analytical assistant. Break down the following problem into clear steps,
solve each step, and combine the results.

Problem: $problem

Step 1: Identify key components
Step 2: Analyze each component
Step 3: Synthesize findings
Step 4: Draw conclusions

Begin:"""),

    "debugging": Template("""\
Debug the following error step by step.

Error: $error
Code context: $code

Step 1: Understand what the error means
Step 2: Identify the root cause
Step 3: Propose a fix
Step 4: Verify the fix is correct

Begin:"""),

    "research": Template("""\
Research the following question step by step.

Question: $question

Step 1: Break down the question into sub-questions
Step 2: For each sub-question, state what you know
Step 3: Identify gaps in knowledge (use Search tool for these)
Step 4: Synthesize into a comprehensive answer

Begin:"""),
}


def step_by_step_prompt(
    template_name: str,
    model: str = "gpt-4o",
    **kwargs,
) -> str:
    template = STEP_BY_STEP_TEMPLATES[template_name]
    prompt = template.substitute(**kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content
```

---

## LangChain ReAct Agent

LangChain provides a battle-tested ReAct implementation with many built-in tools.

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool as LCTool
from langchain_openai import ChatOpenAI
from langchain import hub


def build_langchain_react_agent(model_name: str = "gpt-4o") -> AgentExecutor:
    llm = ChatOpenAI(model=model_name, temperature=0)

    # Pull the standard ReAct prompt from LangChain Hub
    prompt = hub.pull("hwchase17/react")

    tools = [
        LCTool(
            name="Calculator",
            description="Useful for math calculations. Input: arithmetic expression.",
            func=calculator,
        ),
        LCTool(
            name="Search",
            description="Search the web for current information.",
            func=web_search,
        ),
    ]

    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
    )


# LlamaIndex ReAct agent (overview)
# from llama_index.core.agent import ReActAgent
# from llama_index.core.tools import FunctionTool
#
# agent = ReActAgent.from_tools(
#     tools=[FunctionTool.from_defaults(fn=calculator)],
#     llm=llm,
#     verbose=True,
# )
# response = agent.chat("What is sqrt(144) + 10?")
```

---

## ReAct Prompt Templates

```python
REACT_SYSTEM_TEMPLATE = """\
You are a ReAct agent. You solve problems by interleaving reasoning and actions.

## Format
Always respond in this exact format:
Thought: <your reasoning about the current state and next step>
Action: <one of: {tool_names}, Final Answer>
Action Input: <input to the action>

## Rules
- Never skip the Thought step
- One action per response
- Use "Final Answer" as the action when you have enough information
- Be concise in your thoughts; be precise in your action inputs

## Tools
{tool_descriptions}

## Examples
{examples}
"""

REACT_FEW_SHOT_EXAMPLES = """\
Query: What is 15% of 240?
Thought: I need to calculate 15% of 240. I'll use the calculator.
Action: Calculator
Action Input: 240 * 0.15
Observation: 36.0
Thought: 15% of 240 is 36.
Action: Final Answer
Action Input: 15% of 240 is 36.

---

Query: Who invented the telephone and in what year?
Thought: I need to search for information about the telephone's invention.
Action: Search
Action Input: who invented telephone year
Observation: Alexander Graham Bell is credited with inventing the telephone in 1876.
Thought: I have the answer.
Action: Final Answer
Action Input: Alexander Graham Bell invented the telephone in 1876.
"""


def build_react_prompt(tools: list[Tool]) -> str:
    return REACT_SYSTEM_TEMPLATE.format(
        tool_names=", ".join(t.name for t in tools) + ", Final Answer",
        tool_descriptions="\n".join(
            f"- {t.name}: {t.description}" for t in tools
        ),
        examples=REACT_FEW_SHOT_EXAMPLES,
    )
```

---

## Full ReAct Loop with Tool Calls

<div class="diagram">
<div class="diagram-title">ReAct — Full Query Resolution Flow</div>
<div class="flow">
  <div class="flow-node blue wide">User Query: "What is the population of the capital of France squared?"</div>
  <div class="flow-arrow accent">↓ inject into context</div>
</div>
<div class="flow">
  <div class="flow-node purple wide">Thought 1: I need to find the capital of France first.</div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-node orange wide">Action: Search("capital of France")</div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-node teal wide">Observation: "Paris is the capital of France."</div>
  <div class="flow-arrow accent">↓</div>
</div>
<div class="flow">
  <div class="flow-node purple wide">Thought 2: Now I need the population of Paris.</div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-node orange wide">Action: Search("population of Paris")</div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-node teal wide">Observation: "Paris has a population of ~2.1 million."</div>
  <div class="flow-arrow accent">↓</div>
</div>
<div class="flow">
  <div class="flow-node purple wide">Thought 3: I need to square 2,100,000.</div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-node orange wide">Action: Calculator("2100000 ** 2")</div>
  <div class="flow-arrow accent">↓</div>
  <div class="flow-node teal wide">Observation: "4410000000000"</div>
  <div class="flow-arrow accent">↓</div>
</div>
<div class="flow">
  <div class="flow-node green wide">Final Answer: 4,410,000,000,000</div>
</div>
</div>

---

## Comparison Table

<table class="compare-table">
<thead>
<tr>
  <th>Approach</th>
  <th>Reasoning Style</th>
  <th>Tool Use</th>
  <th>Multi-Step</th>
  <th>Best For</th>
  <th>Weakness</th>
</tr>
</thead>
<tbody>
<tr>
  <td><strong>Direct Prompting</strong></td>
  <td>None (single shot)</td>
  <td>❌</td>
  <td>❌</td>
  <td>Simple Q&A</td>
  <td>Fails on complex reasoning</td>
</tr>
<tr>
  <td><strong>CoT</strong></td>
  <td>Narrated steps</td>
  <td>❌</td>
  <td>Partial</td>
  <td>Math, logic</td>
  <td>No grounding in external data</td>
</tr>
<tr>
  <td><strong>ReAct</strong></td>
  <td>CoT + actions</td>
  <td>✅</td>
  <td>✅</td>
  <td>Research, multi-step tasks</td>
  <td>Verbose context; latency per step</td>
</tr>
<tr>
  <td><strong>ToT</strong></td>
  <td>Tree search</td>
  <td>Optional</td>
  <td>✅</td>
  <td>Planning, game-solving</td>
  <td>High token cost; slow</td>
</tr>
<tr>
  <td><strong>RAG</strong></td>
  <td>Retrieval-augmented</td>
  <td>Retrieval only</td>
  <td>Partial</td>
  <td>Knowledge-grounded Q&A</td>
  <td>Single retrieval step; no iteration</td>
</tr>
</tbody>
</table>

---

## Summary

<div class="diagram">
<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-year">Zero-Shot CoT</div>
    <div class="timeline-title">"Let's think step by step"</div>
    <div class="timeline-desc">Append a trigger phrase. Dramatically improves multi-step arithmetic and logic with no examples.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Few-Shot CoT</div>
    <div class="timeline-title">Worked examples in context</div>
    <div class="timeline-desc">Show the model how to reason with 2–8 solved examples in the prompt.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">ReAct</div>
    <div class="timeline-title">Reason → Act → Observe loop</div>
    <div class="timeline-desc">Interleave reasoning traces with tool calls. The workhorse of modern LLM agents.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Self-Correction</div>
    <div class="timeline-title">Verify and revise</div>
    <div class="timeline-desc">Ask the model to critique its own output and regenerate if errors are found.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Tree of Thought</div>
    <div class="timeline-title">Explore and prune</div>
    <div class="timeline-desc">Branch multiple reasoning paths; score and select the best. High cost, high quality.</div>
  </div>
</div>
</div>

<div class="callout tip">
<span class="callout-icon">💡</span>
<div class="callout-body">
Start with zero-shot CoT for reasoning-heavy tasks. Upgrade to ReAct when you need tool use. Reserve ToT for planning problems where exploring alternatives is worth the token cost.
</div>
</div>

<span class="badge agentic">Agentic</span> <span class="badge mlops">MLOps</span>

*Last updated: May 2026*
