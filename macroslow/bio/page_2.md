# MAML-Enhanced bio.md: Building the Perfect Living Agent Memory System with Markdown as Medium Language (.maml.md)

**Universal, Offline-First Guide for Persistent, Self-Evolving Agentic Intelligence**  
**Prepared for Local Ollama Harnesses, Skill Developers, and Bioinformatics-Inspired AI Builders**  
**Focus: Transforming Traditional bio.md into Executable, DP-Optimized .maml.md Workflows**  
**Compatibility: Hermes Memory Curation, OpenClaw Gateways, Pure Local Ollama (No Internet)**  
**Version: 2.1 (Advanced MAML + Dynamic Programming Integration, July 2026)**  
**Page 2 of 10**

## 2. MAML Schema Specifications and Front Matter Reference for bio.maml.md

The power of the MAML-enhanced bio.md lies in its rigorously defined, machine-parsable schema that perfectly balances human readability with automated harness processing, validation, and bioinformatics optimizations. This page provides the **complete reference** for the YAML front matter, required/optional fields, Markdown section dictionary, and adaptation guidelines for pure local Ollama environments (no MCP dependency required, but fully compatible).

All elements are designed to be processed by lightweight local harnesses using standard tools like PyYAML, Pydantic for validation, and custom DP modules in Python or Rust. This ensures the bio.maml.md remains efficient even on resource-constrained hardware while supporting advanced self-evolution.

### YAML Front Matter: Comprehensive Field Reference (Expanded)

The front matter appears at the very top of `bio.maml.md`, delimited by `---`. It is the first thing the harness parses on every bootstrap, heartbeat, or update. It drives routing, security, DP computations, and state management.

| Field                    | Type              | Required | Description                                                                 | Example Value |
|--------------------------|-------------------|----------|-----------------------------------------------------------------------------|---------------|
| `maml_version`          | String           | Yes     | MAML specification version for compatibility. Use "2.1" for full DP/bio features. | "2.1" |
| `id`                    | String (URN)     | Yes     | Unique persistent identifier for the agent instance.                        | "urn:uuid:perfect-bio-agent-20260709" |
| `type`                  | String           | Yes     | File purpose. For main bio file: "living_agent_memory". Others: "skill", "session_branch", "memory_snapshot". | "living_agent_memory" |
| `origin`                | String (URI)     | Yes     | Creator/harness identifier.                                                 | "harness://local-ollama-v2.1" |
| `requires`              | Object           | No      | Dependencies for execution and DP. Focus on local/offline libs.             | `{ "libs": ["numpy", "ollama"], "tools": ["cron", "inotify"] }` |
| `permissions`           | Object           | Yes     | Granular controls for read/execute/write to prevent unauthorized changes.   | `{ "read": ["*"], "execute": ["self", "ollama"], "write": ["bio.maml.md"] }` |
| `created_at`            | String (ISO)     | Yes     | Creation timestamp.                                                         | "2026-07-09T18:00:00Z" |
| `last_updated`          | String (ISO)     | Yes     | Last modification (auto-updated on heartbeats).                             | "2026-07-09T18:30:00Z" |
| `dp_metadata`           | Object           | Recommended | Bioinformatics DP state (scores, lengths, affinities). Critical for optimization. | `{ "global_alignment_score": 0.942, "compression_ratio": 0.68, ... }` |
| `agent_role`            | String           | Recommended | High-level role for SOUL injection.                                         | "Concise, Reflective, Self-Improving Local Assistant" |
| `timeout_seconds`       | Integer          | No      | Max time for heartbeat processing/DP runs.                                  | 120 |
| `pruning_strategy`      | String/Object    | No      | Rules for context compression (e.g., "dp_folding", age-based).              | "dp_folding" |
| `loop_compatible`       | Boolean          | Yes     | Indicates support for iterative heartbeats and reflection.                  | true |

**Additional Advanced Fields for Depth**:
- `sequence_references`: Array of pointers to key MEMORY sequences for fast DP lookups.
- `binding_affinity_matrix`: Summary scores between skills, preferences, and new heartbeats.
- `version_history`: Lightweight Git-like changelog within the file.

### Content Body Sections (Markdown Headers) – Full Dictionary

Use `##` level headers for consistent parsing by harnesses. Agents/Ollama can extract specific sections dynamically.

- **`## SOUL`** (Required, semi-immutable): Core personality, values, ethical constraints, and behavioral rules. Include personality vector and hard constraints. Can be updated slowly via high-affinity reflections.
- **`## SKILLS`** (Required): Catalog of capabilities. Each skill can include description, DP binding affinity, usage stats, and embedded `Code_Blocks` or links to sub-.maml.md files.
- **`## MEMORY`** (Required): Long-term curated knowledge. Subsections for User Preferences, Learned Facts, Summarized Interactions. Backed by DP-aligned sequences.
- **`## CONTEXT`** (Required): Dynamic runtime state. Includes OS info, working dir, recent file changes (`git status`), hardware metrics, and live DP snapshots.
- **`## SESSION TREE`** (Recommended): Branchable conversation history. Includes pointers to session logs, optimal paths computed via DP (like phylogenetic reconstruction), and active branches.
- **`## BIOINFORMATICS`** (Recommended): JSON block with full DP metadata, alignment matrices summaries, compression stats, and evolutionary notes.
- **`## Code_Blocks`** (Highly Recommended in SKILLS/MEMORY): Executable implementations for heartbeats, DP aligners, summarizers, etc.
- **`## Input_Schema`** / **`## Output_Schema`** (Recommended for skills): JSON Schema for validation of heartbeats or tool calls.
- **`## History`** (Auto-managed, Append-only): Immutable log of every heartbeat, update, reflection, and DP computation. Format: `- TIMESTAMP: [TYPE] Description (score: X)`.

### Non-Quantum / Local Ollama Adaptations and Best Practices

- **Pure Offline Focus**: Remove any cloud/MCP dependencies. Use local Ollama for summarization and scoring.
- **DP Integration**: Front matter `dp_metadata` is updated by native code (Python with NumPy for matrix ops) or small-model calls.
- **Incremental Updates**: Never rewrite the entire file. Use targeted string replacements or section-aware parsers to maintain performance.
- **Validation Workflow**:
  1. Parse front matter with PyYAML + Pydantic.
  2. Validate permissions against current harness identity.
  3. Run DP alignment on incoming heartbeat.
  4. Execute Code_Blocks in sandbox (subprocess with whitelists).
  5. Validate output schema.
  6. Append to History and update `last_updated`.
- **Context Window Management**: Use `pruning_strategy` to fold long sections before injection.

**Extended Front Matter Example for a Production bio.maml.md**:

```yaml
---
maml_version: "2.1"
id: "urn:uuid:perfect-bio-agent-20260709"
type: "living_agent_memory"
origin: "harness://local-ollama-v2.1"
requires:
  libs: ["ollama", "numpy>=1.0", "pydantic"]
  tools: ["filesystem", "cron", "inotify-tools"]
permissions:
  read: ["ollama://*", "user://*"]
  execute: ["self", "skills://*", "dp_engine"]
  write: ["bio.maml.md", "data/logs/*"]
created_at: "2026-07-09T18:00:00Z"
last_updated: "2026-07-09T18:30:00Z"
dp_metadata:
  global_alignment_score: 0.942
  compression_ratio: 0.68
  total_heartbeats: 47
  sequence_length: 2856
  binding_affinities:
    skills: 0.97
    user_prefs: 0.91
timeout_seconds: 90
pruning_strategy: "dp_folding"
loop_compatible: true
---
```

### Common Pitfalls to Avoid in bio.maml.md
- Overly large History without pruning.
- Inconsistent DP metadata leading to poor alignment.
- Full file rewrites instead of incremental edits.
- Missing schemas for skill Code_Blocks.
- Neglecting permissions, allowing unsafe self-modification.

By following this schema, your bio.maml.md becomes a robust, evolvable foundation that supports complex agent behaviors while remaining simple to inspect and maintain.

**(End of Page 2. Continued on Page 3: Code Block Examples, Heartbeat Processors, DP Alignment Implementations, and Multi-Language Support for bio.maml.md Workflows.)**
