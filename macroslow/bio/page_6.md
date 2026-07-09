# MAML-Enhanced bio.md: Building the Perfect Living Agent Memory System with Markdown as Medium Language (.maml.md)

**Universal, Offline-First Guide for Persistent, Self-Evolving Agentic Intelligence**  
**Prepared for Local Ollama Harnesses, Skill Developers, and Bioinformatics-Inspired AI Builders**  
**Focus: Transforming Traditional bio.md into Executable, DP-Optimized .maml.md Workflows**  
**Compatibility: Hermes Memory Curation, OpenClaw Gateways, Pure Local Ollama (No Internet)**  
**Version: 2.1 (Advanced MAML + Dynamic Programming Integration, July 2026)**  
**Page 6 of 10**

## 6. Skill Definition Templates, Auto-Generation Patterns, and Reusable MAML Skill Modules for bio.maml.md

The **## SKILLS** section is one of the most powerful parts of bio.maml.md. This page provides complete templates, auto-generation strategies, and reusable skill modules that the harness (or the agent itself via reflection) can use to populate and evolve capabilities.

Skills in MAML format are self-describing, executable, versioned, and DP-scored — making them first-class citizens in the living agent system.

### Core Skill Template (MAML-Enhanced)

```markdown
## SKILLS

### Incremental File Editor (Core Built-in Skill)
**DP Binding Affinity:** 0.97  
**Usage Count:** 142  
**Last Used:** 2026-07-09T19:15:00Z

**Intent**: Perform safe, targeted edits to bio.maml.md and other files without full rewrites.

**Code_Blocks**:
```python
from typing import List
import re

def incremental_edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> dict:
    """MAML-aware incremental replace with validation."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    if replace_all:
        new_content = content.replace(old_string, new_string)
    else:
        new_content = content.replace(old_string, new_string, 1)
    
    # Safety checks: validate MAML structure preserved
    if '---' not in new_content or '## SOUL' not in new_content:
        return {"status": "failed", "reason": "Structure broken"}
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    return {"status": "success", "lines_changed": new_content.count('\n') - content.count('\n')}
```

**Input_Schema**:
```json
{
  "type": "object",
  "properties": {
    "file_path": {"type": "string"},
    "old_string": {"type": "string"},
    "new_string": {"type": "string"}
  },
  "required": ["file_path", "old_string", "new_string"]
}
```

**History**:
- 2026-07-01: [CREATE] Initial skill with DP affinity scoring.
```

### Auto-Generation Pattern (Reflection Skill)

Embed this in bio.maml.md to let the agent improve itself:

```python
def generate_new_skill(task_description: str, current_skills_summary: str) -> str:
    """Agent proposes new MAML skill."""
    prompt = f"""
    You are evolving your own bio.maml.md.
    Current skills: {current_skills_summary}
    New task: {task_description}
    
    Output a complete, ready-to-insert MAML skill definition (including Code_Blocks, schemas, etc.)
    """
    response = ollama.chat(model="qwen2.5:1.5b", messages=[{"role": "user", "content": prompt}])
    proposed_skill = response['message']['content']
    
    # Validate with DP affinity before insertion
    affinity = BioDPEngine().compute_binding_affinity(proposed_skill, soul_vector)
    if affinity > 0.88:
        # Use incremental_edit to insert into ## SKILLS
        print(f"New skill approved (affinity: {affinity})")
    return proposed_skill
```

### Reusable Skill Modules Library (data/skills/)

1. **Memory Consolidation Skill** — Uses DP + Ollama summarization (hourly cron).
2. **Session Branch Manager** — Forks bio.maml.md for parallel explorations.
3. **Context Snapshotter** — Saves compressed versions with DP folding.
4. **User Preference Learner** — Analyzes heartbeats for SOUL updates.
5. **Self-Reflection Loop** — Analyzes recent History and proposes bio.maml.md improvements.

### Auto-Registration Flow
1. Agent generates new skill via reflection.
2. Harness validates schema + DP affinity.
3. Inserts into main bio.maml.md ## SKILLS section using `incremental_edit`.
4. Updates `dp_metadata.binding_affinities`.
5. Logs in History.

### Best Practices for Skill Development in bio.maml.md
- Keep skills modular and idempotent.
- Always include Input/Output schemas.
- Embed DP scoring where relevant.
- Version skills internally.
- Test in isolation before integration.
- Use the agent's own reflection to evolve the skill library continuously.

This system allows bio.maml.md to grow its own "organs" (skills) organically, creating a truly self-improving agent that gets better with every heartbeat.

**(End of Page 6. Continued on Page 7: Security, Validation, Permissions, and Best Practices for Safe Self-Modification in bio.maml.md.)**
