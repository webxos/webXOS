# 02 · Core Philosophy

MSPYB rests on six deliberate principles. Together they define what makes a file a true megalith rather than just a large script that happens to write files.

---

## The Six Principles

| Principle | Meaning |
|-----------|---------|
| **Singular source of truth** | One file holds the full project definition. There is no second place where the “real” structure lives. |
| **Self-bootstrapping** | `python bootstrap.py` creates the entire directory tree, configs, and source. No external scaffolding tools are required. |
| **LLM-native** | `LLM-INSTRUCTION` comments act as permanent system prompts. Future agents read them and know how to extend or fix the system. |
| **Megalithic** | The file is intentionally large and complete rather than fragmented. Completeness is a feature, not a bug. |
| **Executable documentation** | The script is simultaneously the human-readable specification *and* the machine that generates the project. |
| **Idempotent & reproducible** | Re-running the script always regenerates a clean project. Side effects are deterministic. |

---

## Singular Source of Truth

In a conventional repository the “truth” is distributed across many files. A change to a route may require coordinated edits in a gateway, a service, a YAML config, a Dockerfile, and a README. Drift is inevitable.

In MSPYB the truth lives in one place. When you change a string inside a `write_file(...)` call, you have changed the definition of that file for every future generation of the project.

---

## Self-Bootstrapping

The only external dependency is a Python interpreter (3.8+). No code generators, no Yeoman, no custom CLI. The script itself contains everything needed to materialize the tree:

```python
def write_file(path: str, content: str):
    full_path = os.path.join(PROJECT_ROOT, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
```

This makes distribution trivial: send one file. The recipient runs it and obtains a complete project.

---

## LLM-Native by Design

Every important section of a well-written megalith contains guidance written for future language models:

```python
"""
LLM-INSTRUCTION:
Architecture Overview:
- Gateway terminates TLS and routes by path prefix
- Auth service issues JWTs; all other services verify them
- Shared module provides structured logging and circuit breakers

Known issues resolved in this version:
- Healthcheck used wrong SSL import (fixed)
- Route config omitted /billing prefix (fixed)

How to extend:
- Add a new service by copying the gateway pattern
- Update docker-compose.yml and the route table in the same commit
"""
```

These comments survive regeneration and become the permanent “system prompt” for any agent that later opens the file.

---

## Megalithic Completeness

A fragmented project forces an LLM to reconstruct missing context. A megalithic project hands the model the complete state. The trade-off is file size; the benefit is coherence. For the systems MSPYB targets (green-field platforms, agents, teaching demos, golden definitions) the coherence wins.

---

## Executable Documentation

The header of the bootstrap script *is* the architecture document. The `write_file` calls *are* the file inventory. The final `print` statements *are* the getting-started guide. There is no separate Markdown that can fall out of date relative to the code that actually produces the project.

---

## Idempotence

Because every file is written from a constant string, re-running the bootstrap always yields the same tree (assuming the same `PROJECT_ROOT`). This makes the megalith safe to re-execute after every edit and turns regeneration into the normal way of applying fixes.

---

Next: see the exact shape of a valid MSPYB file in [Canonical Structure](03-canonical-structure.md).
