# 09 · Advanced Patterns

Once the basic MSPYB structure is solid, several patterns appear repeatedly in production megaliths.

---

## 1. Comprehensive Error-Log + Fix Guide

Place a detailed error log at the top of the docstring (or immediately after the architecture overview). Each entry records:

- Symptom
- Location
- Root cause
- Exact fix
- Impact
- Version in which the fix landed

This turns the megalith into a living post-mortem document. Future agents can read the history of failures and avoid repeating them.

---

## 2. Visible Evolution Across Versions

Keep multiple successive versions of the same bootstrap in the repository (or as clearly marked sections):

```
v1 – known bugs still present
v2 – all fixes applied
```

The evolution stays visible. An agent (or a human) can see exactly what changed between versions without digging through Git history of many files.

---

## 3. Synchronized Route & Permission Configs

Store route tables and permission matrices as YAML (or JSON) strings inside the megalith. The gateway’s path-capturing logic is generated from the same data, or at least kept in lock-step by living in adjacent `write_file` calls.

```python
ROUTE_CONFIG = """
routes:
  - path: /auth
    service: auth
    auth_required: false
  - path: /billing
    service: billing
    auth_required: true
"""

write_file("gateway/config/routes.yaml", ROUTE_CONFIG)
# gateway main.py also embeds or loads the same structure
```

---

## 4. Working Healthchecks as One-Liners

Healthcheck commands must be valid inside the container image that is actually built. Common pitfalls (missing modules, wrong paths, TLS verification) are documented and fixed once in the megalith so every future generation inherits the working command.

```dockerfile
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

---

## 5. Consistent `sys.path` / Import Hygiene

In monorepo-style layouts every service’s `main.py` receives the same small preamble that adds the project root to `sys.path`. Because the preamble lives in the megalith, it is impossible for one service to drift.

---

## 6. Reusable Shared Modules

Common capabilities are extracted into the `shared/` package and generated once:

- Structured JSON logging with trace IDs
- Circuit-breaker wrapper
- Redis client factory
- OAuth / JWT helpers
- Environment-variable-driven configuration objects

All services import from the same generated package, guaranteeing identical behavior.

---

## 7. Environment-Variable-Driven Runtime Config

Model names, intervals, feature flags, and paths are read once at import time from environment variables. The generated `.env.example` documents every variable. Changing behavior later requires zero source edits—only a new `.env`.

This pattern is used heavily in the DREAM PET reference build.

---

## 8. Helper Functions for Large Sections

When a logical area grows large, factor the content into a helper that returns a dictionary of path → content:

```python
def billing_service():
    return {
        "billing/Dockerfile": """...""",
        "billing/app/main.py": """...""",
        "billing/app/models.py": """...""",
    }

for path, content in billing_service().items():
    write_file(path, content)
```

The main body of the megalith stays readable; each helper can still carry its own `LLM-INSTRUCTION` comments.

---

## 9. Composition of Multiple Megaliths

For platforms that are genuinely too large for a single file, keep several focused MSPYB files (e.g., `bootstrap-core.py`, `bootstrap-billing.py`) and a thin orchestrator that runs them in order into the same `PROJECT_ROOT`. The ideal remains one megalith per coherent platform; composition is a pragmatic escape hatch.

---

## 10. Self-Signed Certificates for Local HTTPS

Dockerfiles can generate a short-lived self-signed certificate at build or start time so that local demos run under HTTPS without external certificate management. The generation logic lives inside the megalith and is therefore reproducible.

---

These patterns are not mandatory, but they appear repeatedly in megaliths that have been maintained across many agent sessions and human iterations.

---

Next: the final production checklist and closing thoughts in [Checklist & Summary](10-checklist-and-summary.md).
