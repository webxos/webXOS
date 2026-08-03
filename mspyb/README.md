# MSPYB Guide
*webXOS 2026*

**Megalithic Singular Python Bootstrap Guide**

> Pack an entire multi-file software project into a single self-executing Python script that regenerates the full project on demand.

This is the official multi-page guide for the MSPYB format. It is designed for developers, LLM users, and agent builders who want a **singular source of truth** that both humans and language models can reason about end-to-end.


## MSPYB

MSPYB's bootstrap algorithm uses a single Python script with a strict canonical structure: a shebang, rich docstring with LLM instructions, a write_file helper that creates directories, and grouped write_file calls embedding all project files as triple-quoted strings. Execution simply runs the script, which regenerates the full multi-file project directory from the embedded definitions, overwriting existing files for clean iteration while maintaining a singular source of truth. The workflow emphasizes editing the megalith script (with versioned headers and LLM guidance), bootstrapping, configuring, running, and iterating—enabling seamless human/LLM/agent collaboration for project evolution without scattered file management.


---

## Pages

| # | Page | Description |
|---|------|-------------|
| 01 | [Introduction](01-introduction.md) | What MSPYB is and why it exists |
| 02 | [Core Philosophy](02-core-philosophy.md) | The six principles behind the format |
| 03 | [Canonical Structure](03-canonical-structure.md) | Exact layout of a valid MSPYB file |
| 04 | [LLM Vibe Coding](04-llm-vibe-coding.md) | How MSPYB supercharges prompt-driven development |
| 05 | [Agentic Use Cases](05-agentic-use-cases.md) | Autonomous agents that edit, extend, and heal projects |
| 06 | [Recommended Workflow](06-workflow.md) | From idea to running system in six steps |
| 07 | [Best Practices](07-best-practices.md) | Conventions that keep megaliths healthy |
| 08 | [Real-World Use Cases](08-use-cases.md) | Concrete scenarios where MSPYB shines |
| 09 | [Advanced Patterns](09-advanced-patterns.md) | Scaling, versioning, and production techniques |
| 10 | [Checklist & Summary](10-checklist-and-summary.md) | Production-ready checklist and closing thoughts |

---

## LICENSE

MIT 

## AUTHOR

webXOS (github.com/webxos)
