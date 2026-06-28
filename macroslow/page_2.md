# MAML (Markdown as Medium Language): A Practical Communication Medium for Modern MCP-Based Agentic Harnesses

**Report for Skill.md Integration**  
**Prepared for Agentic AI Developers**  
**Focus: Non-Quantum PCs, MCP Compatibility, Hermes & OpenClaw Harnesses**  
**Version: 1.0 (Adapted from Webxos Concepts, June 2026)**  
**Page 2 of 10**

## 2. MAML Schema Specifications and Front Matter Reference

The strength of MAML lies in its rigorously defined schema, which balances machine parsability with flexibility for agentic workflows. This page details the complete front matter dictionary, content section requirements, and adaptation guidelines for non-quantum MCP environments. All elements are designed to be processed by standard MCP servers running on classical hardware, using libraries such as PyYAML, Pydantic, and JSON Schema validators.

### YAML Front Matter: Comprehensive Field Reference

The front matter is a YAML object enclosed by `---` lines at the top of the `.maml.md` file. MCP gateways and agent harnesses (Hermes, OpenClaw) parse this section first to determine routing, dependency resolution, permission enforcement, and execution eligibility.

| Field                  | Type          | Required | Description                                                                 | Example Value |
|------------------------|---------------|----------|-----------------------------------------------------------------------------|---------------|
| maml_version          | String       | Yes     | MAML specification version. Use "1.0.0" or "lite-1.0.0" for non-quantum.   | "1.0.0" |
| id                    | String (URN) | Yes     | Unique identifier, preferably a UUID URN for traceability across sessions. | "urn:uuid:123e4567-e89b-12d3-a456-426614174000" |
| type                  | String       | Yes     | Primary purpose of the file. Common values: skill, workflow, prompt, dataset, agent_blueprint. | "skill" |
| origin                | String (URI) | Yes     | Creator identifier (agent, harness, or user).                               | "agent://hermes-skill-author" |
| requires              | Object       | No      | Dependencies for execution. Focus on classical libraries.                   | { "libs": ["pydantic>=2.0", "pandas"], "apis": ["mcp://data-validator"] } |
| permissions           | Object       | Yes     | Access control lists for read, write, and execute operations.               | { "read": ["agent://*"], "execute": ["gateway://local", "agent://openclaw"] } |
| created_at            | String (ISO) | Yes     | Timestamp of creation in ISO 8601 format.                                   | "2026-06-28T02:23:00Z" |
| verification          | Object       | No      | Optional lightweight validation settings (non-quantum).                     | { "method": "pydantic", "level": "strict" } |

**Additional Recommended Fields for Agentic Harnesses**:
- `timeout_seconds`: Integer specifying maximum execution time (e.g., 300 for 5 minutes).
- `memory_hint`: String or object providing context cues for Hermes-style memory plugins.
- `loop_compatible`: Boolean indicating support for iterative execution with History updates.

### Content Body Sections (Markdown Headers)

All major sections use `##` level headers for consistent parsing. Agents and MCP servers extract these programmatically.

- **`## Intent`** (Required): Concise natural language description of the file's goal. This serves as the primary discovery and summarization text for agents in Hermes or OpenClaw.
- **`## Context`** (Required): Supporting information including variables, dataset descriptions, prior results, or harness-specific instructions. Use key-value pairs or embedded JSON for structured data.
- **`## Code_Blocks`** (Highly Recommended): One or more fenced code blocks with language identifiers (e.g., `python`, `javascript`). Multiple blocks are supported for hybrid Python + JS workflows.
- **`## Input_Schema`** (Recommended for skills/workflows): JSON Schema defining expected inputs. Enables automatic validation in MCP tool calls.
- **`## Output_Schema`** (Recommended): JSON Schema for expected outputs, facilitating result parsing and chaining.
- **`## History`** (Auto-managed): Append-only log of events. Agents append entries after execution (e.g., `[EXECUTE]`, `[VALIDATE]`, `[ERROR]`). Critical for maintaining state in long-running agent loops.

### Non-Quantum Adaptations and Best Practices

For standard PCs and MCP servers:
- Avoid any quantum-specific libraries or flags (remove `qiskit`, `quantum_security_flag`, etc.).
- Prioritize `pydantic` for runtime validation within Python code blocks.
- Use Docker or virtual environments for sandboxing code execution to maintain security across Hermes/OpenClaw deployments.
- Ensure all `requires.libs` are installable via `pip` without specialized hardware.
- For JavaScript blocks in OpenClaw gateways: Use Node.js-compatible syntax with `fetch` or Axios for MCP interactions.

**Example Extended Front Matter for a Skill in Hermes/OpenClaw**:

```yaml
---
maml_version: "1.0.0"
id: "urn:uuid:a1b2c3d4-e5f6-7890-abcd-ef1234567890"
type: "skill"
origin: "agent://openclaw-gateway"
requires:
  libs: ["pydantic>=2.0", "pandas>=2.0", "requests"]
  apis: ["mcp://internal-data"]
permissions:
  read: ["agent://hermes-memory", "agent://*"]
  execute: ["gateway://localhost", "harness://openclaw"]
  write: ["agent://hermes-history"]
created_at: "2026-06-28T02:23:00Z"
timeout_seconds: 180
memory_hint: "Focus on data validation patterns for tabular inputs"
loop_compatible: true
---
```

### Schema Validation Workflow in MCP Harnesses

1. MCP server receives `.maml.md` file.
2. Parse front matter and validate against core MAML schema.
3. Check permissions against current agent/harness identity.
4. Resolve dependencies and prepare sandbox.
5. Validate input against `Input_Schema` using Pydantic or JSON Schema library.
6. Execute relevant `Code_Blocks`.
7. Validate output and append to `History`.
8. Return enriched MAML file or extracted results to the calling agent.

This workflow ensures reliability in agentic loops, where a Hermes agent might invoke the same MAML skill multiple times with updated context, while OpenClaw handles gateway-level orchestration and routing.

**Common Pitfalls to Avoid**:
- Omitting required fields (especially `permissions` and schemas).
- Using hardware-specific code in blocks.
- Large file sizes (keep under 1MB for efficient MCP transmission).
- Inconsistent History formatting, which breaks auditability.

By adhering to these specifications, MAML becomes a robust backbone for skill definitions, enabling seamless, versioned communication between agents in modern MCP ecosystems.

(End of Page 2. Continued on Page 3: Code Block Examples and Multi-Language Support for Agentic Workflows.)
