# 03 · Canonical Structure

A valid MSPYB file follows a strict, recognizable layout. This consistency is what lets both humans and agents navigate any megalith without learning a new convention each time.

---

## Skeleton

```python
#!/usr/bin/env python3
"""
MSPYB – Megalithic Singular Python Bootstrap
Version: X.Y.Z

LLM-INSTRUCTION:
  High-level architecture overview,
  known fixes, usage instructions,
  generated file tree, and guidance
  for future LLM agents.

Architecture Overview:
- ...
- ...

Usage:
    python bootstrap.py
    cd project
    ...
"""

import os

PROJECT_ROOT = "project"   # or any desired output directory

def write_file(path: str, content: str):
    """Create parent directories and write the file."""
    full_path = os.path.join(PROJECT_ROOT, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)

# ------------------------------------------------------------------
# 1. Root files (.env.example, docker-compose.yml, README.md, ...)
# ------------------------------------------------------------------
write_file(".env.example", """...""")
write_file("docker-compose.yml", """...""")
write_file("README.md", """...""")

# ------------------------------------------------------------------
# 2. Shared utilities
# ------------------------------------------------------------------
write_file("shared/__init__.py", "")
write_file("shared/logging.py", """...""")

# ------------------------------------------------------------------
# 3. Gateway / Services / ...
# ------------------------------------------------------------------
write_file("gateway/Dockerfile", """...""")
write_file("gateway/app/main.py", """...""")

print(f"✅ MSPYB bootstrap complete – project generated in '{PROJECT_ROOT}'")
print("Next steps: ...")
```

---

## Required Elements

1. **Shebang + rich module docstring**  
   Must contain version, an `LLM-INSTRUCTION` block, architecture notes, known issues, usage commands, and ideally the generated file tree.

2. **`write_file` helper**  
   Creates parent directories automatically and always writes UTF-8.

3. **Logical sections**  
   Grouped by responsibility (root → shared → services → …) and separated by clear comment banners:

   ```python
   # ------------------------------------------------------------------
   # 2. Shared utilities
   # ------------------------------------------------------------------
   ```

4. **Embedded content as multi-line strings**  
   Every real file lives inside a triple-quoted string passed to `write_file`. There are no external templates.

5. **Final print statements**  
   Confirm success and print the exact next steps a human or agent should take.

---

## Minimal Working Example

```python
#!/usr/bin/env python3
"""
MSPYB – Megalithic Singular Python Bootstrap
Version: 0.1.0

LLM-INSTRUCTION: Simple FastAPI demo.
Usage:
    python bootstrap.py
    cd project
    pip install -r requirements.txt
    uvicorn app.main:app --reload
"""

import os

PROJECT_ROOT = "project"

def write_file(path, content):
    full_path = os.path.join(PROJECT_ROOT, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)

# Root
write_file("README.md", """# My Project\n""")
write_file(".env.example", """SECRET=change_me\n""")
write_file("requirements.txt", """fastapi\nuvicorn\n""")

# Shared
write_file("shared/__init__.py", "")
write_file("shared/utils.py", """# LLM-INSTRUCTION: Shared helpers live here.\n""")

# Application
write_file("app/main.py", """# LLM-INSTRUCTION: Entry point.
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}
""")

print(f"✅ MSPYB bootstrap complete – project generated in '{PROJECT_ROOT}'")
```

---

## Naming Conventions

| Item | Recommended name |
|------|------------------|
| Script | `bootstrap.py`, `mspyb.py`, or `megalith.py` |
| Output directory | `project/` (or a constant `PROJECT_ROOT`) |
| Section banners | Full-width `# ------------------------------------------------------------------` |

---

## What Is *Not* Required

- External dependencies beyond the Python standard library (for the bootstrap itself)
- A particular web framework or container technology
- Git, Docker, or any other tool at generation time

The generated project may of course use any stack; the bootstrap only needs to write the files.

---

Next: see how this structure turns LLM “vibe coding” into a reliable, high-bandwidth collaboration in [LLM Vibe Coding](04-llm-vibe-coding.md).
