# MAML (Markdown as Medium Language): A Practical Communication Medium for Modern MCP-Based Agentic Harnesses

**Report for Skill.md Integration**  
**Prepared for Agentic AI Developers**  
**Focus: Non-Quantum PCs, MCP Compatibility, Hermes & OpenClaw Harnesses**  
**Version: 1.0 (Adapted from Webxos Concepts, June 2026)**  
**Page 4 of 10**

## 4. Integration Patterns with Hermes and OpenClaw Harnesses

This page explores practical integration strategies for using MAML files within Hermes and OpenClaw agentic harnesses. Both systems leverage MCP as the underlying communication layer, making MAML an excellent standardized medium for skill sharing, workflow delegation, and stateful agent loops on non-quantum hardware.

### Hermes Integration: Memory-Centric Agent Loops

Hermes emphasizes persistent memory, self-improvement, and reflective reasoning. MAML complements this by providing structured, versioned artifacts that Hermes agents can ingest, update, and reference across sessions.

**Key Patterns**:
- **Skill Ingestion**: Hermes agents load `.maml.md` files as skill definitions. The `## Intent` and `## Context` sections feed directly into memory curation.
- **Loop Execution**: An agent invokes a MAML skill, executes code blocks, then appends results and reflections to the `## History` section before saving the updated file.
- **Memory Plugin Synergy**: Use `memory_hint` in front matter and structured output to populate Hermes memory stores.

**Example Workflow in Hermes**:
1. Agent discovers a MAML skill via MCP.
2. Loads context and validates against schemas.
3. Executes Python block(s).
4. Appends: `- [REFLECT] Analysis complete. Key insight: ...`
5. Persists updated MAML for future sessions.

**Sample MAML Snippet for Hermes**:

```yaml
---
maml_version: "1.0.0"
id: "urn:uuid:hermes-memory-skill-001"
type: "skill"
origin: "agent://hermes-core"
requires:
  libs: ["pydantic", "pandas"]
permissions:
  execute: ["agent://hermes-memory"]
memory_hint: "Store validation metrics for trend analysis across iterations"
---
## Intent
Perform iterative data cleaning with memory-aware adjustments.

## Context
Previous iterations showed high error rates on categorical fields. Adjust rules dynamically.

## Code_Blocks
```python
# Code that reads prior History if available and adapts
def adaptive_clean(data, history_context):
    # Adaptation logic based on past errors
    ...
```

### OpenClaw Integration: Gateway and Multi-Agent Orchestration

OpenClaw focuses on broad connectivity, sub-agent dispatching, and gateway functionality. MAML serves as a portable payload for routing tasks across channels and coordinating multiple agents.

**Key Patterns**:
- **Gateway Routing**: OpenClaw parses MAML front matter to route code blocks to appropriate runtimes or sub-harnesses.
- **Multi-Agent Delegation**: One MAML file can trigger sub-MAML workflows for specialized agents.
- **Tool Chaining**: Use MCP to pass MAML files between OpenClaw and external services.

**Example OpenClaw Gateway Flow**:
1. User/harness submits MAML via MCP endpoint.
2. Gateway validates permissions and dependencies.
3. Dispatches Python/JS blocks to sandboxes.
4. Collects outputs and enriches History.
5. Returns updated MAML or forwards to Hermes for memory persistence.

**Hybrid Hermes + OpenClaw Setup** (Recommended):
- OpenClaw handles execution and external integrations.
- Hermes manages long-term memory and reflection using MAML History.

### Comparative Integration Table

| Aspect                  | Hermes Integration                          | OpenClaw Integration                        | Combined Benefit |
|-------------------------|---------------------------------------------|---------------------------------------------|------------------|
| Primary Strength       | Persistent memory and reflection            | Gateway routing and multi-agent coordination | Full agentic loop coverage |
| MAML Usage             | Skill loading + History updates             | Payload routing + sub-workflow triggering   | Standardized communication |
| MCP Role               | Context and tool result ingestion           | Server-side execution and API bridging      | Seamless interoperability |
| Typical Code Focus     | Python with reflective logic                | JavaScript for orchestration + Python compute | Hybrid efficiency |
| State Management       | Memory plugins + MAML History               | Session-based with MAML persistence         | Robust auditability |

### Implementation Steps for Developers

1. **Create Base MAML Skill**: Follow schema from Page 2 and code examples from Page 3.
2. **Test Locally**: Use a simple MCP server (e.g., Dockerized Python runner) to execute the file.
3. **Hermes Deployment**:
   - Place `.maml.md` in skill directory.
   - Configure memory plugin to monitor History updates.
4. **OpenClaw Deployment**:
   - Register MAML handler in gateway config.
   - Expose via MCP endpoint for external calls.
5. **Loop Testing**: Simulate multi-turn interactions where agents iteratively refine a MAML file.
6. **Monitoring**: Track execution metrics via appended History entries.

### Security Considerations in Harnesses

- Enforce `permissions` strictly in both harnesses.
- Run all code blocks in isolated containers (Docker with resource limits).
- Validate all inputs/outputs against schemas before and after execution.
- Use signed History entries where harness supports lightweight signatures.

These patterns enable MAML to function as the connective tissue between execution (OpenClaw) and intelligence (Hermes), creating powerful, maintainable agent systems on standard hardware.

(End of Page 4. Continued on Page 5: MAML-Lite Setup, Execution, and Tooling for Non-Quantum Environments.)
```
