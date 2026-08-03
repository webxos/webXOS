# 04 · LLM Vibe Coding

“Vibe coding” is the practice of describing intent in natural language and letting a language model produce working software. MSPYB was designed from the ground up to make this loop dramatically more reliable.

---

## The Context Window Problem

When a project lives in dozens of files, an LLM session must either:

- receive a curated subset (and invent the missing parts), or
- receive a huge dump of files (and still lose cross-file relationships).

Both approaches degrade as the system grows. MSPYB collapses the entire system into **one coherent artifact**. The model sees the complete state at once.

---

## Six Reasons MSPYB Supercharges Vibe Coding

### 1. Single context window
The model reads one file and therefore holds the whole architecture, every config, and every source string simultaneously. There is no “I forgot the Docker healthcheck” moment.

### 2. Explicit permanent instructions
`LLM-INSTRUCTION` blocks act as durable system prompts. They survive across sessions and across different models. A new agent that opens the megalith immediately knows the intended architecture, the known pitfalls, and the preferred extension patterns.

### 3. Diff-friendly edits
Changing one multi-line string updates exactly one generated file. The model (or the human) can surgically edit a single `write_file` call without hunting through a directory tree.

### 4. Self-documenting
The header *is* the architecture document. The model never has to guess which README is authoritative; the docstring at the top of the bootstrap is the single source of truth for design intent.

### 5. Error-resilient regeneration
Fix a bug once inside the megalith, re-run `python bootstrap.py`, and the fix is applied everywhere. There is no partial-update risk.

### 6. Versionable artifact
The bootstrap script itself is the versioned object. Git history of the megalith is the complete history of the system definition.

---

## Typical Vibe-Coding Loop with MSPYB

```
Human / Agent
    │
    │  “Add rate limiting to the gateway and document it”
    ▼
LLM edits the single bootstrap.py
    │
    │  (updates write_file for gateway/app/main.py,
    │   updates docker-compose if needed,
    │   updates the LLM-INSTRUCTION header)
    ▼
Human / Agent runs
    python bootstrap.py
    │
    ▼
Fresh project/ tree appears with the change applied
```

Because the model always works on the complete definition, the probability of producing a consistent change is far higher than when it must juggle many independent files.

---

## Prompt Patterns That Work Especially Well

**Extension prompt**
```
Add a new billing service following the existing MSPYB pattern.
Update the header, docker-compose.yml, the route table,
and create the new service files. Keep the same style of
LLM-INSTRUCTION comments.
```

**Repair prompt**
```
The healthcheck in the gateway Dockerfile is failing.
Document the symptom, root cause, and fix in the header
under “Known issues resolved”, then apply the exact fix
inside the corresponding write_file string. Bump the version.
```

**Audit prompt**
```
Read the entire megalith and list every place where a secret
could accidentally be hard-coded. Suggest .env.example entries
and the corresponding os.getenv calls.
```

---

## Vibe Coding vs. Traditional Pair Programming

| Aspect | Traditional multi-file | MSPYB megalith |
|--------|------------------------|----------------|
| Context the model receives | Partial or noisy | Complete and structured |
| Risk of inconsistent change | High | Low |
| Effort to keep docs in sync | Continuous | Automatic (docs live in header) |
| Ability to regenerate cleanly | Manual | One command |
| Suitability for long agent sessions | Degrades | Stable |

---

## Practical Tip

Keep the `LLM-INSTRUCTION` block at the top of the file *more up-to-date than the code itself*. Future agents will trust that block more than any other comment. Treat it as the permanent system prompt for every subsequent collaboration.

---

Next: see how the same properties enable fully agentic workflows in [Agentic Use Cases](05-agentic-use-cases.md).
