# MSPYB v3 — Megalithic Singular Python Bootstrap

## Philosophy

One self-executing Python file defines, generates, and launches an entire multi-file project.

| Principle              | Meaning |
|------------------------|---------|
| Singular source of truth | One file holds the full project definition |
| Self-bootstrapping     | `python3 bootstrap.py` creates the tree after installing deps |
| LLM-native             | `LLM-INSTRUCTION` comments guide future agents |
| Megalithic             | Intentionally complete rather than fragmented |
| Executable documentation | The script is both the spec and the generator |
| Idempotent             | Re-running regenerates a clean project |
| Full-circle (v3)       | Install → Generate → Auto-Launch |

## Five Required Capabilities

1. **Error Logging** — Bootstrap logger before INSTALLS; shared `logging_setup.py`.
2. **Templating** — Single `TEMPLATE_VARS` dict; `string.Template.safe_substitute`.
3. **Versioning** — `MSPYB_VERSION` + generated `VERSION` file; header documents known fixes.
4. **INSTALLS** — OS/arch/Python/Docker detection; pinned packages; `--skip-installs`.
5. **Auto-Start** — After generation, spawn the app respecting `AUTO_LAUNCH` and `--no-launch`.

## Canonical Invocation

```bash
python3 bootstrap.py                 # full circle
python3 bootstrap.py --skip-installs # generation + launch
python3 bootstrap.py --no-launch     # pure generation
```

## Auto-Start Decision Flow

```
parse flags → run INSTALLS → generate tree →
  AUTO_LAUNCH == "true" AND not --no-launch ?
    yes → spawn LAUNCH_COMMAND in LAUNCH_CWD
    no  → log "Auto-launch disabled"
→ final status prints
```

## Why This Is Dynamic Programming

- The bootstrap is the **policy**.
- Each generated file is a **state**.
- The install → generate → launch → observe loop is a **Bellman backup**.
- An LLM agent improves the bootstrap (the policy) rather than hand-editing every leaf file.

Critical implementation note: use triple-single-quotes `'''` for multi-line file contents inside `write_file` so that inner `"""` docstrings do not terminate the outer string.
