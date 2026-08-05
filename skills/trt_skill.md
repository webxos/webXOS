---
name: trt-loop
description: Equip any agent or LLM to solve hard problems via the Three-Stage TRT (Test-time Recursive Thinking) loop. Use when the task requires multi-round iterative self-improvement without external feedback or ground-truth, especially for math, coding, reasoning, or open-ended problems where one-shot answers fail. Triggers include TRT, recursive thinking, self-improve at test time, generate-select-reflect, iterative refinement without reward model.
---

# Test-time Recursive Thinking (TRT) Loop

Instead of a single one-shot response, run an adaptive iterative loop across multiple sequential rounds. Each round has three phases: Generate → Select → Reflect. Accumulate compact knowledge from failures so later rounds avoid prior mistakes and explore complementary strategies.

## Core Algorithm

Initialize:
- Knowledge list K ← empty (distilled negative constraints / failure modes)
- Solution pool S ← empty
- Round counter t ← 1
- Max rounds T (default 8–64 depending on difficulty; start small and grow)
- Rollouts per round K (default 2–4; depth matters more than breadth)

For each round t = 1 … T (or until confident convergence):

### Phase 1 — Generate
Produce K distinct candidate solutions (rollouts). Condition each rollout on:
1. The original problem P
2. The full accumulated knowledge list K_t
3. A unique, rollout-specific strategy s_k that actively avoids known failure modes while encouraging diversity

Strategies are generated dynamically (not from a fixed pool). Example strategy prompts: “Solve via contradiction and check edge cases”, “Use dynamic programming while avoiding off-by-one”, “Generate unit tests first then implement”.

### Phase 2 — Select
Evaluate all K rollouts of the current round with domain-specific self-assessment (no ground-truth required). Choose the single best r*.

Selection mechanisms:
- Math (unique numeric answer): exploit mutual exclusivity / convergence. Track previously rejected answers; prefer answers that consistent reasoners converge on.
- Code: generate unit tests covering typical, edge, and boundary cases from the problem statement; execute candidates against self-generated tests; rank by pass rate + code quality.
- General reasoning: internal consistency checks, logical-error detection, self-critique, pairwise preference, or confidence scoring.

Add the selected r* (and optionally strong runners-up) to the solution pool S.

### Phase 3 — Reflect
Perform pairwise comparisons of r* against each non-selected rollout. Extract concise failure insights:
- Why the weaker solution failed
- What made r* superior
- Concrete “don’t” constraints (e.g., “avoid off-by-one in indexing”, “handle empty-input edge case”, “do not assume sorted input”)

Append the distilled insights to K, forming K_{t+1}. Synthesize 1–2 new complementary strategies that avoid the newly discovered failure modes for the next Generate phase.

Keep the knowledge list compact (negative constraints only). Aim for <1–2 % of context window.

## Termination

Stop early when:
- Selection margin is negligible across recent rounds
- Same answer/strategy repeatedly selected with high internal confidence
- Max rounds T reached
- Domain-specific signal (e.g., all self-generated tests pass and no new failure modes appear)

Return the final selected solution from the last Select phase (or majority / best from the pool S).

## Domain Adaptations

**Mathematics (AIME-style unique answers)**  
Knowledge tracks rejected answers and common algebraic/combinatorial pitfalls. Convergence of independent rollouts is a strong correctness signal.

**Code generation**  
Primary verification = self-generated unit tests + execution. Knowledge focuses on bug patterns, missing edge cases, and API misuse.

**Open-ended / multi-answer domains**  
Replace mutual exclusivity with multi-criteria self-scoring (completeness, consistency, novelty, adherence to constraints). Prefer strategies that explore complementary technique clusters.

## Practical Guidelines for Agents

- Prefer iterative depth (more rounds) over extreme breadth (many rollouts per round). Ablations show K=2 is often sufficient; T drives gains.
- Knowledge must stay distilled: store only actionable “don’ts” and short failure explanations, never full traces.
- Always condition generation on both K and a fresh strategy to prevent mode collapse.
- When tools are available (code interpreter, calculator, search), use them inside Generate and Select for stronger self-verification signals.
- Log the knowledge list growth and strategy switches; adaptive strategy switching after failures correlates with higher solve rates.
- Computational cost scales roughly linearly with T × K. Start with T=4–8 for medium difficulty; escalate only when needed.

## When to Activate

Activate TRT whenever a one-shot or short chain-of-thought answer is likely insufficient: competition math, hard coding problems, multi-constraint planning, scientific reasoning, or any task where self-correction without external reward is valuable. Deactivate for trivial lookups or when latency is critical.

## Output Format

When using TRT, structure intermediate thinking (visible to the agent, optionally summarized for the user) as:

```
[TRT Round t]
Generate: strategy s1 → candidate1
          strategy s2 → candidate2
Select: r* = …
Reflect: insights added to K = […]; next strategies = […]
```

Final answer appears only after the loop terminates.
