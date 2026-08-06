---
name: graph-task
description: Split complex tasks into a dependency graph of nodes before execution. Use when a task has three or more steps, requires parallel work, branching, or fallback paths. Triggers include graph plan, task graph, execution graph, dependency graph, structured JSON task plan.
---

# Graph Task

Decompose multi-step work into an explicit execution graph before running any code or tools. Output the graph first. Execute only after the graph is defined.

## When to Activate

Activate when the task has three or more steps, needs parallel work, conditional branches, retries, or fallback paths. Skip for single-step or trivial two-step tasks.

## Node Schema

Every node must include these fields:

- id: unique string identifier (example: fetch_data, validate_1)
- goal: one clear sentence describing the purpose
- inputs: list of required data or outputs from prior nodes
- outputs: list of data or artifacts this node produces
- type: one of action, decision, parallel, fallback
- status: start as pending

## Edge Rules

- An edge from A to B means B starts only after A succeeds.
- Parallel nodes share the same predecessor and have no edges between them.
- Fallback edges use condition on_error and point from a primary node to its recovery node.

## Required Output Format

Always emit the graph as structured JSON before any execution:

```json
{
  "nodes": [
    {
      "id": "step_1",
      "goal": "Clear one-sentence purpose",
      "inputs": [],
      "outputs": ["data_x"],
      "type": "action",
      "status": "pending"
    }
  ],
  "edges": [
    {
      "from": "step_1",
      "to": "step_2",
      "condition": "success"
    }
  ],
  "entry": "step_1",
  "exit": "final_step"
}
```

For parallel starts, entry may be an array of node ids.

## Execution Protocol

1. Emit the complete JSON graph.
2. Wait for explicit confirmation unless the original request authorized automatic execution.
3. Run nodes in topological order. Update status from pending to running, then to success or failed.
4. On failure, follow any on_error edge. If none exists, halt and report.
5. Do not invent new nodes during execution. If the plan must change, revise the full graph and re-emit it.

## Minimal Example

Task: Scrape three sites, merge results, write a report.

```json
{
  "nodes": [
    {
      "id": "scrape_a",
      "goal": "Fetch site A",
      "inputs": [],
      "outputs": ["raw_a"],
      "type": "action",
      "status": "pending"
    },
    {
      "id": "scrape_b",
      "goal": "Fetch site B",
      "inputs": [],
      "outputs": ["raw_b"],
      "type": "action",
      "status": "pending"
    },
    {
      "id": "scrape_c",
      "goal": "Fetch site C",
      "inputs": [],
      "outputs": ["raw_c"],
      "type": "action",
      "status": "pending"
    },
    {
      "id": "merge",
      "goal": "Combine raw results",
      "inputs": ["raw_a", "raw_b", "raw_c"],
      "outputs": ["merged"],
      "type": "action",
      "status": "pending"
    },
    {
      "id": "report",
      "goal": "Write final report",
      "inputs": ["merged"],
      "outputs": ["report.md"],
      "type": "action",
      "status": "pending"
    }
  ],
  "edges": [
    {"from": "scrape_a", "to": "merge", "condition": "success"},
    {"from": "scrape_b", "to": "merge", "condition": "success"},
    {"from": "scrape_c", "to": "merge", "condition": "success"},
    {"from": "merge", "to": "report", "condition": "success"}
  ],
  "entry": ["scrape_a", "scrape_b", "scrape_c"],
  "exit": "report"
}
```

Keep graphs minimal. One node per atomic goal. Add fallback or decision nodes only when the user requested them.
```

Validated and ready. Pure text — no rendering issues on GitHub or anywhere else.
