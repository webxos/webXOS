*THE MACROSLOW INTRODUCTION TO:* 
```
▗▖  ▗▖ ▗▄▖ ▗▖  ▗▖▗▖   
▐▛▚▞▜▌▐▌ ▐▌▐▛▚▞▜▌▐▌   
▐▌  ▐▌▐▛▀▜▌▐▌  ▐▌▐▌   
▐▌  ▐▌▐▌ ▐▌▐▌  ▐▌▐▙▄▄▖
```

# MAML (Markdown as Medium Language): A Practical Communication Syntax for Modern MCP-Based Agentic Harnesses

**Report for Skill.md Integration**  
**Prepared for Agentic AI Developers**  
**Focus: MCP Compatibility, Hermes & OpenClaw Harnesses**  
**Version: 1.0 (Adapted from Webxos Concepts, June 2026)**  
**Page 1 of 10**

## Executive Summary

MAML, or Markdown as Medium Language, represents a structured evolution of plain Markdown syntax into a dynamic, executable, and context-rich protocol designed specifically for agentic AI systems. By using the `.maml.md` file extension, MAML files serve as self-contained containers that bundle metadata, declarative intent, supporting context, executable code blocks, input/output schemas, and an immutable execution history. This design positions MAML as a powerful communication medium that bridges human-readable documentation with machine-executable workflows, making it particularly well-suited for integration with the Model Context Protocol (MCP).

MCP, developed by Anthropic and widely adopted across the AI ecosystem, provides a standardized interface for AI agents to interact with external tools, data sources, and services. MAML extends MCP by offering a richer, file-based payload format that supports persistent state, versioning, and auditability—capabilities that are essential for robust agentic loops in production environments. Unlike ad-hoc tool definitions or scattered prompt files, MAML encapsulates complete workflows in a single, portable artifact that agents can discover, validate, execute, and update collaboratively.

The protocol shines in **agentic harnesses** such as Hermes (from Nous Research, with strong emphasis on persistent memory curation and self-improvement) and OpenClaw (focused on flexible gateway integrations, multi-agent coordination, and broad tool orchestration). In these systems, MAML files act as standardized messages or skill definitions: agents can pass `.maml.md` files via MCP servers to invoke tools, maintain conversation state across sessions, log outcomes in the History section, and enable reflective loops (observe → plan → act → reflect). This reduces integration friction, improves reproducibility, and supports complex, long-running agent behaviors without relying on fragile in-memory state alone.

### Primary Use Cases in Skill.md Contexts
- **Reusable Skills**: Embed complete, validated skill implementations (with schemas and test logic) directly in `.maml.md` files for import across Hermes or OpenClaw instances.
- **Workflow Orchestration**: Define multi-step agentic processes that span data ingestion, processing, validation, and output—ideal for MCP tool calling chains.
- **Memory and Provenance**: Leverage the History section for audit trails and Hermes-style memory plugins, allowing agents to reference past executions.
- **Inter-Harness Communication**: Share MAML files between OpenClaw gateways and Hermes memory layers for hybrid setups.
- **Development Productivity**: Skill.md authors can treat MAML as both documentation and executable code, streamlining testing and deployment via MCP endpoints.

### Strategic Benefits
- **Interoperability**: Works with any MCP-compliant client or server; files are version-controlled via Git and human-inspectable.
- **Security and Permissions**: Granular controls in the front matter prevent unauthorized execution while supporting agent-to-agent delegation.
- **Lightweight Execution**: Code blocks run in isolated environments (e.g., Docker containers with Python 3.8+), making it suitable for local development and scaled harness deployments.
- **Extensibility**: Easily extended with custom sections or libraries while maintaining backward compatibility for non-quantum setups.

This 10-page report synthesizes core MAML specifications, practical implementation patterns, examples tailored to Hermes and OpenClaw, integration with skill.md files, best practices for agentic loops, and a forward-looking roadmap. It draws from established MAML language guides while adapting them explicitly for classical computing environments and contemporary MCP workflows as of mid-2026.

Subsequent pages will cover: detailed schema specifications (Page 2), code block examples and language support (Page 3), integration patterns with Hermes and OpenClaw (Page 4), MAML-Lite setup and execution (Page 5), skill definition templates (Page 6), security and validation best practices (Page 7), real-world use cases and performance considerations (Page 8), limitations and troubleshooting (Page 9), and conclusion with future directions (Page 10).

## 1. MAML Fundamentals Adapted for MCP and Non-Quantum Environments

### The Core File Anatomy
Every MAML file follows a predictable, dual-layer structure optimized for both human collaboration and automated agent processing:

1. **YAML Front Matter**: Delimited by triple dashes (`---`), this section contains all machine-parsable configuration. It is the entry point for MCP gateways to validate permissions, resolve dependencies, and route execution.
2. **Markdown Body**: Uses conventional Markdown headers (primarily `##`) to organize semantic sections. This ensures readability in any Markdown viewer while allowing structured parsing by agents or harnesses.

A representative MAML skill file begins as follows:

```yaml
---
maml_version: "1.0.0"
id: "urn:uuid:123e4567-e89b-12d3-a456-426614174000"
type: "skill"  # Options: skill, workflow, prompt, agent_blueprint
origin: "agent://skill-author-hermes"
requires:
  libs: ["pydantic>=2.0", "pandas", "requests"]
  apis: ["mcp://local-tool-server"]
permissions:
  read: ["agent://*", "harness://openclaw"]
  execute: ["gateway://localhost", "agent://hermes-memory"]
created_at: "2026-06-28T02:17:00Z"
```

---
## Intent
Provide a reusable, schema-validated data processing skill that can be invoked via MCP from Hermes or OpenClaw harnesses for cleaning and validating tabular datasets prior to analysis.

## Context
This skill targets CSV files from internal data pipelines. Expected columns include identifiers, numeric metrics, and categorical labels. Agents should maintain state across multiple invocations using the History section for progressive refinement.

## Code_Blocks
```python
from pydantic import BaseModel, field_validator, ValidationError
import pandas as pd
import json

class DatasetRecord(BaseModel):
    record_id: int
    metric_value: float
    category: str

    @field_validator('metric_value')
    def validate_metric(cls, v):
        if v < 0 or v > 1000:
            raise ValueError('Metric value must be in range [0, 1000]')
        return round(v, 4)

def process_dataset(input_path: str, output_path: str = None):
    try:
        df = pd.read_csv(input_path)
        records = []
        errors = []
        for _, row in df.iterrows():
            try:
                record = DatasetRecord(**row.to_dict())
                records.append(record.model_dump())
            except ValidationError as e:
                errors.append({"row": row.to_dict(), "error": str(e)})
        result = {
            "processed_records": len(records),
            "validation_errors": len(errors),
            "sample_output": records[:5] if records else None
        }
        if output_path:
            pd.DataFrame(records).to_csv(output_path, index=False)
        return result
    except Exception as e:
        return {"status": "failed", "error": str(e)}
```

## Input_Schema
{
  "type": "object",
  "properties": {
    "input_path": {"type": "string", "description": "Path to input CSV file"},
    "output_path": {"type": "string", "description": "Optional path for cleaned output"}
  },
  "required": ["input_path"]
}

## Output_Schema
{
  "type": "object",
  "properties": {
    "processed_records": {"type": "integer"},
    "validation_errors": {"type": "integer"},
    "sample_output": {"type": "array"},
    "status": {"type": "string"}
  }
}

## History
- 2026-06-28T02:17:00Z: [CREATE] Initial skill definition authored for MCP integration testing.

This format ensures that MAML files are immediately actionable: an MCP server or harness can parse the front matter, validate schemas, execute code blocks in a sandbox, append results to History, and return enriched context to the calling agent.

(End of Page 1. Continued on Page 2: In-Depth Schema Specifications and Front Matter Reference.)
