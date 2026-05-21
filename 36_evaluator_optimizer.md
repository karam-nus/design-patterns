---
title: "Chapter 36 — Evaluator-Optimizer Pattern"
---

[← Back to Table of Contents](./README.md)

# Chapter 36 — Evaluator-Optimizer Pattern

> *"The first draft of anything is garbage." — Ernest Hemingway (adapted for LLMs)*

<span class="badge agentic">Agentic</span>

---

## Overview

The **Evaluator-Optimizer** pattern structures generation as a closed feedback loop: a *generator* produces a candidate output, an *evaluator* scores it against a quality rubric, and an *optimizer* (often the same LLM with a different prompt) refines the candidate based on that feedback. The loop repeats until the score exceeds a threshold or a maximum iteration count is reached.

This pattern underpins Constitutional AI, Process Reward Models, code-generation-with-tests, and Best-of-N sampling — all explored in this chapter.

---

## 36.1 The Self-Reflection Loop

Self-reflection asks the same LLM to critique its own output and then revise it. The pattern was popularised by *Reflexion* (Shinn et al., 2023) and *Self-Refine* (Madaan et al., 2023).

<div class="diagram">
  <div class="diagram-title">Self-Reflection Loop</div>
  <div class="flow">
    <div class="flow-node accent">User Prompt</div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node blue">Generator LLM<br/><small>produces draft</small></div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node purple">Evaluator LLM<br/><small>scores + critiques</small></div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node orange wide">Score ≥ threshold?</div>
    <div class="flow-h">
      <div class="flow-node green narrow">Yes → Accept</div>
      <div class="flow-arrow accent">↙&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↘</div>
      <div class="flow-node pink narrow">No → Refine</div>
    </div>
    <div class="flow-arrow accent">↑ feedback loop ↑</div>
  </div>
</div>

```python
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class ReflectionLoop:
    """Self-reflection loop: generate → evaluate → refine."""

    generate: Callable[[str], str]
    evaluate: Callable[[str, str], tuple[float, str]]  # (score, critique)
    refine: Callable[[str, str, str], str]             # (prompt, draft, critique) -> new draft
    threshold: float = 0.8
    max_iterations: int = 5

    def run(self, prompt: str) -> dict:
        draft = self.generate(prompt)
        history = []
        for i in range(self.max_iterations):
            score, critique = self.evaluate(prompt, draft)
            history.append({"iteration": i + 1, "score": score, "critique": critique})
            if score >= self.threshold:
                return {"output": draft, "score": score, "iterations": i + 1, "history": history}
            draft = self.refine(prompt, draft, critique)
        # return best effort after max iterations
        return {"output": draft, "score": score, "iterations": self.max_iterations, "history": history}


# ---- Example with OpenAI (sketch) ----------------------------------------

def make_openai_loop(client, model: str = "gpt-4o-mini") -> ReflectionLoop:
    system_generate = "You are a helpful assistant. Answer the user's question clearly."
    system_evaluate = (
        "You are a strict evaluator. Given the original question and a draft answer, "
        "return a JSON object with keys 'score' (float 0-1) and 'critique' (string)."
    )
    system_refine = (
        "You are a skilled editor. Given the original question, a draft answer, and a critique, "
        "produce an improved answer that addresses all points in the critique."
    )

    def generate(prompt: str) -> str:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_generate},
                      {"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content

    def evaluate(prompt: str, draft: str) -> tuple[float, str]:
        import json
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_evaluate},
                      {"role": "user", "content": f"Question: {prompt}\n\nDraft: {draft}"}],
            response_format={"type": "json_object"},
        )
        data = json.loads(r.choices[0].message.content)
        return float(data["score"]), data["critique"]

    def refine(prompt: str, draft: str, critique: str) -> str:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_refine},
                      {"role": "user", "content":
                       f"Question: {prompt}\n\nDraft: {draft}\n\nCritique: {critique}"}],
        )
        return r.choices[0].message.content

    return ReflectionLoop(generate=generate, evaluate=evaluate, refine=refine)
```

---

## 36.2 Constitutional AI — Critique + Revision

Constitutional AI (Bai et al., 2022) uses an explicit *constitution* — a list of principles — to guide the evaluator. The model critiques its own output against each principle and then revises accordingly.

```python
from dataclasses import dataclass
from typing import Sequence

ANTHROPIC_CONSTITUTION = [
    "The response must not assist with any illegal activity.",
    "The response must be honest and avoid deception.",
    "The response must be helpful and clear.",
    "The response must respect the autonomy of the user.",
    "The response must not include harmful or offensive content.",
]

@dataclass
class ConstitutionalReviser:
    """
    Implements Constitutional AI critique-revision.
    For each principle, the model critiques its draft, then revises.
    """

    llm: Callable[[list[dict]], str]
    constitution: list[str] = field(default_factory=lambda: ANTHROPIC_CONSTITUTION)

    def _critique(self, draft: str, principle: str) -> str:
        messages = [
            {"role": "user", "content":
             f"Draft response:\n{draft}\n\n"
             f"Principle: {principle}\n\n"
             "Does the draft violate this principle? "
             "If so, explain precisely how. If not, say 'No violation.'"}
        ]
        return self.llm(messages)

    def _revise(self, draft: str, principle: str, critique: str) -> str:
        if "no violation" in critique.lower():
            return draft
        messages = [
            {"role": "user", "content":
             f"Draft response:\n{draft}\n\n"
             f"Principle violated: {principle}\n"
             f"Critique: {critique}\n\n"
             "Rewrite the response to comply with the principle "
             "while remaining helpful. Output only the revised response."}
        ]
        return self.llm(messages)

    def apply(self, initial_response: str) -> dict:
        draft = initial_response
        revisions = []
        for principle in self.constitution:
            critique = self._critique(draft, principle)
            revised = self._revise(draft, principle, critique)
            revisions.append({
                "principle": principle,
                "critique": critique,
                "changed": revised != draft,
            })
            draft = revised
        return {"final": draft, "revisions": revisions}


# ---- Streaming variant -------------------------------------------------------

def constitutional_stream(llm_stream, prompt: str, constitution: list[str]):
    """Yield revised drafts after each principle is applied."""
    draft = llm_stream(prompt)
    yield {"stage": "initial", "draft": draft}
    for i, principle in enumerate(constitution):
        reviser = ConstitutionalReviser(llm=llm_stream, constitution=[principle])
        result = reviser.apply(draft)
        draft = result["final"]
        yield {"stage": f"principle_{i+1}", "principle": principle, "draft": draft}
```

---

## 36.3 Process Reward Model (PRM)

A **Process Reward Model** scores *each reasoning step* rather than just the final answer. This is critical for chain-of-thought tasks where a correct final answer can be reached via flawed reasoning.

<div class="diagram">
  <div class="diagram-title">PRM vs Outcome Reward Model</div>
  <div class="diagram-grid cols-2">
    <div class="diagram-card blue">
      <div class="card-icon">🎯</div>
      <div class="card-title">Outcome Reward Model (ORM)</div>
      <div class="card-desc">Scores only the final answer. Cannot distinguish lucky-correct from correct-by-valid-reasoning.</div>
    </div>
    <div class="diagram-card green">
      <div class="card-icon">🔬</div>
      <div class="card-title">Process Reward Model (PRM)</div>
      <div class="card-desc">Scores each intermediate step. Encourages valid reasoning chains. Used in OpenAI PRM800K.</div>
    </div>
  </div>
</div>

```python
from dataclasses import dataclass
from typing import Sequence

@dataclass
class PRMEvaluator:
    """
    Process Reward Model evaluator.
    Scores each step of a chain-of-thought response.
    """

    step_scorer: Callable[[str, str, int], float]
    # step_scorer(question, step_text, step_index) -> score in [0, 1]
    aggregation: str = "min"  # "min" | "mean" | "product"

    def score_steps(self, question: str, steps: list[str]) -> list[float]:
        return [self.step_scorer(question, step, i) for i, step in enumerate(steps)]

    def aggregate(self, step_scores: list[float]) -> float:
        if not step_scores:
            return 0.0
        if self.aggregation == "min":
            return min(step_scores)
        if self.aggregation == "mean":
            return sum(step_scores) / len(step_scores)
        if self.aggregation == "product":
            result = 1.0
            for s in step_scores:
                result *= s
            return result
        raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def evaluate(self, question: str, cot_response: str) -> dict:
        steps = [s.strip() for s in cot_response.split("\n") if s.strip()]
        step_scores = self.score_steps(question, steps)
        overall = self.aggregate(step_scores)
        return {
            "steps": steps,
            "step_scores": step_scores,
            "overall_score": overall,
            "weakest_step": step_scores.index(min(step_scores)) if step_scores else -1,
        }


def llm_prm_scorer(llm_call: Callable) -> Callable:
    """Wrap an LLM call as a step scorer returning a float."""
    import json

    def scorer(question: str, step: str, step_idx: int) -> float:
        prompt = (
            f"Question: {question}\n"
            f"Reasoning step {step_idx + 1}: {step}\n\n"
            "Is this reasoning step logically valid and helpful toward the correct answer? "
            "Return JSON: {\"valid\": true/false, \"score\": 0.0-1.0, \"reason\": \"...\"}"
        )
        raw = llm_call([{"role": "user", "content": prompt}])
        try:
            data = json.loads(raw)
            return float(data.get("score", 0.5))
        except Exception:
            return 0.5

    return scorer
```

---

## 36.4 Quality Scoring — Rubric-Based LLM Judge

Design explicit rubrics to make the evaluator consistent and auditable.

```python
from dataclasses import dataclass, field
from typing import NamedTuple
import json

class RubricCriterion(NamedTuple):
    name: str
    weight: float
    description: str
    scale: tuple[int, int] = (1, 5)

DEFAULT_RUBRIC = [
    RubricCriterion("accuracy",    0.35, "Factual correctness of claims"),
    RubricCriterion("completeness",0.25, "All parts of the question addressed"),
    RubricCriterion("clarity",     0.20, "Clear, well-structured prose"),
    RubricCriterion("conciseness", 0.10, "No unnecessary repetition or padding"),
    RubricCriterion("safety",      0.10, "No harmful or offensive content"),
]

@dataclass
class RubricJudge:
    llm: Callable[[list[dict]], str]
    rubric: list[RubricCriterion] = field(default_factory=lambda: DEFAULT_RUBRIC)

    def _build_prompt(self, question: str, answer: str) -> str:
        criteria_text = "\n".join(
            f"- {c.name} (weight {c.weight}): {c.description} "
            f"[score {c.scale[0]}-{c.scale[1]}]"
            for c in self.rubric
        )
        return (
            f"You are an expert evaluator. Score the following answer.\n\n"
            f"Question: {question}\n\nAnswer: {answer}\n\n"
            f"Criteria:\n{criteria_text}\n\n"
            f"Return JSON with key per criterion name mapping to "
            f"{{\"score\": int, \"rationale\": str}}."
        )

    def judge(self, question: str, answer: str) -> dict:
        prompt = self._build_prompt(question, answer)
        raw = self.llm([{"role": "user", "content": prompt}])
        try:
            scores = json.loads(raw)
        except json.JSONDecodeError:
            scores = {}

        weighted_total = 0.0
        details = {}
        for c in self.rubric:
            entry = scores.get(c.name, {})
            raw_score = entry.get("score", (c.scale[0] + c.scale[1]) / 2)
            normalised = (raw_score - c.scale[0]) / (c.scale[1] - c.scale[0])
            weighted_total += normalised * c.weight
            details[c.name] = {
                "raw": raw_score,
                "normalised": round(normalised, 3),
                "weight": c.weight,
                "rationale": entry.get("rationale", ""),
            }

        return {"weighted_score": round(weighted_total, 4), "details": details}
```

---

## 36.5 Auto-Refinement Loop

The full auto-refinement loop ties together generator, judge, and refiner with configurable stopping criteria.

```python
from __future__ import annotations
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class AutoRefinementLoop:
    """
    Full auto-refinement loop with budget management.

    Attributes:
        generator:     prompt -> initial response
        judge:         (question, response) -> score float in [0, 1]
        refiner:       (question, response, feedback) -> improved response
        threshold:     accept if score >= threshold
        max_iters:     hard cap on refinement rounds
        min_delta:     stop if improvement between rounds < min_delta
    """

    generator: Callable[[str], str]
    judge: Callable[[str, str], float]
    refiner: Callable[[str, str, str], str]
    threshold: float = 0.85
    max_iters: int = 4
    min_delta: float = 0.02

    def run(self, prompt: str, verbose: bool = False) -> dict:
        response = self.generator(prompt)
        score = self.judge(prompt, response)
        history = [{"iteration": 0, "score": score, "response": response}]

        for i in range(1, self.max_iters + 1):
            if score >= self.threshold:
                logger.info("Accepted at iteration %d with score %.3f", i - 1, score)
                break

            feedback = f"Current score: {score:.2f}. Please improve the response."
            response = self.refiner(prompt, response, feedback)
            new_score = self.judge(prompt, response)
            delta = new_score - score

            if verbose:
                print(f"Iter {i}: score {score:.3f} → {new_score:.3f} (Δ={delta:+.3f})")

            history.append({"iteration": i, "score": new_score, "response": response})
            score = new_score

            if delta < self.min_delta:
                logger.info("Stopping: improvement %.4f < min_delta %.4f", delta, self.min_delta)
                break

        return {
            "final_response": response,
            "final_score": score,
            "iterations_used": len(history) - 1,
            "accepted": score >= self.threshold,
            "history": history,
        }
```

<div class="callout tip">
  <div class="callout-icon">💡</div>
  <div class="callout-body">
    Set <code>min_delta</code> to catch cases where the model is stuck in a local optimum and refinement is not helping. This avoids wasting API tokens on unhelpful iterations.
  </div>
</div>

---

## 36.6 Evaluator-Optimizer for Code Generation

Code generation is a natural fit: the evaluator *runs* the code against tests.

```python
import subprocess
import textwrap
import tempfile
import os
from dataclasses import dataclass

@dataclass
class CodeGenLoop:
    """
    Generate → run tests → fix loop for code generation tasks.
    """

    code_generator: Callable[[str], str]
    code_fixer: Callable[[str, str, str], str]   # (spec, code, error) -> fixed code
    max_attempts: int = 5
    timeout_seconds: int = 10

    def _extract_code(self, llm_output: str) -> str:
        """Strip markdown fences if present."""
        lines = llm_output.strip().splitlines()
        in_block = False
        code_lines = []
        for line in lines:
            if line.startswith("```"):
                in_block = not in_block
                continue
            if in_block or not any(llm_output.strip().startswith("```")):
                code_lines.append(line)
        return "\n".join(code_lines) if code_lines else llm_output

    def _run_tests(self, code: str, test_code: str) -> tuple[bool, str]:
        """Execute code + tests in a subprocess, return (passed, output)."""
        full_source = code + "\n\n" + test_code
        # Write to a file in the current working directory (no /tmp)
        test_file = "._codegen_test_runner.py"
        try:
            with open(test_file, "w") as f:
                f.write(full_source)
            result = subprocess.run(
                ["python", test_file],
                capture_output=True, text=True,
                timeout=self.timeout_seconds,
            )
            passed = result.returncode == 0
            output = result.stdout + result.stderr
            return passed, output
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    def run(self, spec: str, test_code: str) -> dict:
        code = self._extract_code(self.code_generator(spec))
        for attempt in range(1, self.max_attempts + 1):
            passed, output = self._run_tests(code, test_code)
            if passed:
                return {"code": code, "passed": True, "attempts": attempt}
            # Ask LLM to fix based on the error output
            code = self._extract_code(self.code_fixer(spec, code, output))
        passed, output = self._run_tests(code, test_code)
        return {"code": code, "passed": passed, "attempts": self.max_attempts, "last_output": output}
```

---

## 36.7 Comparison with Reinforcement Learning

The Evaluator-Optimizer pattern is a form of **policy optimisation** with a sparse reward signal.

<div class="diagram">
  <div class="diagram-title">RL vs Evaluator-Optimizer Analogy</div>
  <div class="diagram-grid cols-2">
    <div class="diagram-card blue">
      <div class="card-icon">🤖</div>
      <div class="card-title">Reinforcement Learning</div>
      <div class="card-desc">
        <strong>Policy</strong> = LLM<br/>
        <strong>Action</strong> = token sequence<br/>
        <strong>Reward model</strong> = evaluator<br/>
        <strong>Policy update</strong> = gradient step (PPO/GRPO)<br/>
        <strong>Episode</strong> = full generation
      </div>
    </div>
    <div class="diagram-card purple">
      <div class="card-icon">🔄</div>
      <div class="card-title">Evaluator-Optimizer</div>
      <div class="card-desc">
        <strong>Policy</strong> = generator prompt<br/>
        <strong>Action</strong> = LLM output<br/>
        <strong>Reward model</strong> = evaluator LLM<br/>
        <strong>Policy update</strong> = revised prompt / in-context feedback<br/>
        <strong>Episode</strong> = one refinement round
      </div>
    </div>
  </div>
</div>

The key difference: RL updates **weights**; Evaluator-Optimizer updates **prompts or context** — making it training-free at inference time.

---

## 36.8 Best-of-N Sampling

The simplest Evaluator-Optimizer: generate N independent candidates and pick the best-scoring one.

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

@dataclass
class BestOfN:
    """
    Best-of-N sampling: generate N candidates, score each, return best.

    Args:
        generator:  prompt -> response
        scorer:     (prompt, response) -> float
        n:          number of candidates
        parallel:   whether to generate candidates in parallel
    """

    generator: Callable[[str], str]
    scorer: Callable[[str, str], float]
    n: int = 8
    parallel: bool = True

    def run(self, prompt: str) -> dict:
        if self.parallel:
            candidates = self._generate_parallel(prompt)
        else:
            candidates = [self.generator(prompt) for _ in range(self.n)]

        scored = [(c, self.scorer(prompt, c)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_response, best_score = scored[0]
        return {
            "best": best_response,
            "best_score": best_score,
            "all_scores": [s for _, s in scored],
            "n": self.n,
        }

    def _generate_parallel(self, prompt: str) -> list[str]:
        results = []
        with ThreadPoolExecutor(max_workers=min(self.n, 16)) as pool:
            futures = [pool.submit(self.generator, prompt) for _ in range(self.n)]
            for f in as_completed(futures):
                try:
                    results.append(f.result())
                except Exception as e:
                    results.append(f"[generation failed: {e}]")
        return results
```

<div class="callout info">
  <div class="callout-icon">ℹ️</div>
  <div class="callout-body">
    Best-of-N is embarrassingly parallel and requires no iterative feedback. It trades compute (N × generation cost) for quality. Useful when latency is not critical and the scorer is cheap (e.g., a small verifier model).
  </div>
</div>

---

## 36.9 Full Flow Diagram

<div class="diagram">
  <div class="diagram-title">Evaluator-Optimizer — Complete Flow</div>
  <div class="flow">
    <div class="flow-node accent wide">Input Prompt / Task Specification</div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node blue">Generator<br/><small>Draft response (iteration i)</small></div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node purple wide">Evaluator<br/><small>Rubric judge / PRM / test runner / Best-of-N scorer</small></div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node orange wide">Decision Gate<br/><small>score ≥ θ OR i = max_iters?</small></div>
    <div class="flow-h">
      <div class="flow-node green">✅ Accept<br/><small>Return output</small></div>
      <div class="flow-node pink">🔁 Refine<br/><small>Critique → revised prompt → Generator</small></div>
    </div>
  </div>
</div>

---

## 36.10 Comparison Table

<table class="compare-table">
  <thead>
    <tr>
      <th>Variant</th>
      <th>Evaluator</th>
      <th>Optimizer</th>
      <th>Iterations</th>
      <th>Strength</th>
      <th>Weakness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Self-Reflection</strong></td>
      <td>Same LLM</td>
      <td>Same LLM</td>
      <td>3-5</td>
      <td>No extra model</td>
      <td>Blind to own biases</td>
    </tr>
    <tr>
      <td><strong>Constitutional AI</strong></td>
      <td>Principle list</td>
      <td>Same LLM</td>
      <td>= num principles</td>
      <td>Explicit safety</td>
      <td>Principle ordering matters</td>
    </tr>
    <tr>
      <td><strong>PRM</strong></td>
      <td>Step-level RM</td>
      <td>Beam search / sampling</td>
      <td>1 pass</td>
      <td>Catches flawed reasoning</td>
      <td>Requires step-annotated data</td>
    </tr>
    <tr>
      <td><strong>Best-of-N</strong></td>
      <td>Scorer model</td>
      <td>Re-ranking</td>
      <td>1 (parallel)</td>
      <td>Simple, parallelisable</td>
      <td>N× inference cost</td>
    </tr>
    <tr>
      <td><strong>Code Gen Loop</strong></td>
      <td>Test runner</td>
      <td>Fix prompt</td>
      <td>3-5</td>
      <td>Ground truth signal</td>
      <td>Needs test suite</td>
    </tr>
  </tbody>
</table>

---

<div class="callout tip">
  <div class="callout-icon">💡</div>
  <div class="callout-body">
    <strong>Design recommendation:</strong> prefer a <em>stronger evaluator</em> over more iterations. An evaluator that is only marginally better than the generator will produce noisy feedback and slow convergence. Use a larger model for evaluation even if a smaller one generates.
  </div>
</div>

---

*Last updated: May 2026*
