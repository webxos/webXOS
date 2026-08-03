# 05 · Agentic Use Cases

MSPYB is not only a convenience for human + LLM collaboration. It is a natural substrate for **autonomous agents** that create, extend, diagnose, and heal entire software systems.

---

## Why Agents Love Megaliths

An agent that must operate on a conventional multi-file repository faces several hard problems:

- Discovering the real architecture
- Keeping multiple files consistent after a change
- Knowing which documentation is still accurate
- Recovering from partial or failed edits

A well-formed MSPYB file removes most of those problems:

| Agent challenge | How MSPYB helps |
|-----------------|-----------------|
| Understanding the system | One file contains architecture, tree, and instructions |
| Making a consistent change | Edit one string → regenerate everything |
| Documenting the change | Update the header’s known-issues / architecture section in the same edit |
| Recovering from failure | Re-run the bootstrap; the tree is always clean |
| Passing state to the next agent | The megalith *is* the state |

---

## Core Agentic Patterns

### 1. Scaffolding Agent
**Goal:** Turn a natural-language product description into a runnable project.

```
Input:  “Build a small multi-tenant SaaS with auth, billing, and a React dashboard”
Agent:  Generates a complete MSPYB bootstrap.py
Human:  python bootstrap.py && docker-compose up
```

Because the agent emits one file, the human can review the entire design before any files are written to disk.

### 2. Extension Agent
**Goal:** Add a new capability while preserving coherence.

The agent is given the current megalith and a high-level request:

```
“Add an audit-log service that every other service can call.
Update routes, docker-compose, shared client, and the header.”
```

It edits only the bootstrap script, then the human (or another agent) regenerates. The new service is guaranteed to match the existing style and conventions.

### 3. Healing / Self-Repair Agent
**Goal:** Diagnose a runtime failure and produce a fixed megalith.

Typical loop:

1. Agent observes logs or failing healthchecks.
2. Agent opens the current bootstrap.py.
3. Agent documents the symptom, location, root cause, and exact fix in the header.
4. Agent applies the fix inside the relevant `write_file` string.
5. Agent bumps the version.
6. Regeneration produces a corrected project.

Because the fix is applied to the source of truth, the same bug cannot reappear on the next generation.

### 4. Multi-Agent Pipeline
Multiple specialized agents can operate on the same megalith in sequence:

```
Architect Agent  →  writes initial architecture & LLM-INSTRUCTION block
Implementer Agent →  fills in the write_file bodies
Security Agent    →  audits for secrets, tightens CORS / rate limits
Docs Agent        →  expands the README and usage section
Validator Agent   →  runs the bootstrap and basic smoke tests
```

Each agent receives the complete current state and returns an improved complete state. There is no partial-file hand-off problem.

### 5. Golden-Master / Regression Agent
The megalith is treated as a golden definition. An agent can:

- regenerate the project,
- run the test suite,
- compare against expected behavior,
- and, if a regression appears, propose a patch back into the megalith.

This turns the bootstrap script into both the specification and the regression oracle.

---

## Safety Properties for Agents

- **Idempotence** – re-running the bootstrap never leaves the tree in a half-written state.
- **Auditability** – every change is a diff against a single file; Git history is complete.
- **Reversibility** – any previous version of the megalith can be checked out and regenerated.
- **No hidden state** – there are no “generated but not committed” files that an agent must discover.

---

## Example Agent Prompt Template

```
You are an MSPYB maintenance agent.

Current megalith is provided below.
Your task: <high-level goal>

Rules:
1. Edit only the bootstrap script.
2. Keep the LLM-INSTRUCTION block accurate.
3. Document any bug you fix under “Known issues resolved”.
4. Bump the version if the change is significant.
5. Preserve all existing section banners and style.
6. Return the complete updated bootstrap.py.
```

---

## Limits and Guardrails

Agents should still be constrained:

- Never hard-code real secrets into the megalith.
- Prefer environment variables and `.env.example`.
- For extremely large systems, the megalith may need to be split into composable sections (see Advanced Patterns).
- Human review of the megalith before regeneration remains best practice for production systems.

---

Next: the concrete day-to-day process that both humans and agents follow in [Recommended Workflow](06-workflow.md).
