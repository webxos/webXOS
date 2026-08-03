# 06 · Recommended Workflow

The MSPYB lifecycle is intentionally short and repeatable. Both humans and agents follow the same six steps.

---

## The Six Steps

### 1. Author / Evolve the Megalith

Edit `bootstrap.py` (or ask an LLM / agent to edit it).  
Keep the header and every `LLM-INSTRUCTION` comment accurate.

This is the only place where design decisions are recorded.

### 2. Bootstrap

```bash
python bootstrap.py
```

A clean `project/` directory appears containing every file defined in the megalith.

### 3. Configure

```bash
cd project
cp .env.example .env
# edit secrets, model names, ports, etc.
```

Never commit real secrets. The megalith only ever contains the example file.

### 4. Run

For a containerized platform:

```bash
docker-compose up --build
```

For a simple CLI tool (the DREAM PET style):

```bash
python app.py
```

### 5. Iterate

When you discover a bug or missing feature:

1. Open the megalith.
2. Update the matching `write_file(...)` string.
3. Document the change in the header (symptom → root cause → fix).
4. Bump the version if the change is significant.
5. Re-run `python bootstrap.py` (optionally delete `project/` first for a completely clean slate).

The regenerated tree now contains the fix everywhere it is needed.

### 6. Extend

Ask an LLM or agent:

> “Add a new billing service following the existing MSPYB pattern.  
> Update the header, docker-compose, route config, and create the new service files.”

The agent returns an updated megalith. You regenerate and the new service appears, already wired into the rest of the system.

---

## Clean-Slate vs. In-Place Regeneration

| Approach | Command | When to use |
|----------|---------|-------------|
| Clean slate | `rm -rf project && python bootstrap.py` | After structural changes, or when you want to guarantee no leftover files |
| In-place | `python bootstrap.py` | Everyday iteration; existing files are overwritten with the latest definitions |

Because every file is fully specified inside the megalith, both approaches are safe.

---

## Version Discipline

Treat the bootstrap script as the single versioned artifact:

```
bootstrap.py          ← this is what lives in Git
project/              ← generated, usually gitignored (or committed only for release snapshots)
```

Meaningful changes always produce a new version number in the docstring and a clear entry under “Known issues resolved” or “Architecture changes”.

---

## Collaboration Pattern

When multiple people (or agents) work on the same system:

1. The megalith is the contract.
2. Pull requests modify only `bootstrap.py`.
3. CI can run the bootstrap and execute smoke tests against the generated tree.
4. Reviewers examine one file and immediately see the global impact of the change.

---

## Daily Rhythm Example

```
09:00  Agent proposes a new feature as a patch to bootstrap.py
09:15  Human reviews the single-file diff
09:20  python bootstrap.py
09:25  docker-compose up --build
09:40  Runtime issue discovered
09:45  Human (or healing agent) documents + fixes inside the megalith
09:50  Re-bootstrap → issue gone
```

The loop stays short because there is never a “which files do I need to touch?” question.

---

Next: the conventions that keep megaliths healthy over time in [Best Practices](07-best-practices.md).
