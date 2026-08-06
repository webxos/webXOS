*THE MACROSLOW INTRODUCTION TO:* 
```
▗▖  ▗▖ ▗▄▖ ▗▖  ▗▖▗▖   
▐▛▚▞▜▌▐▌ ▐▌▐▛▚▞▜▌▐▌   
▐▌  ▐▌▐▛▀▜▌▐▌  ▐▌▐▌   
▐▌  ▐▌▐▌ ▐▌▐▌  ▐▌▐▙▄▄▖
```

by webXOS 2026

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

**MAML (Markdown as Medium Language)**  
**README / Full Overview for Modern Binary MCP Harnesses (Pi, OpenClaw, and similar)**

### What is MAML?

MAML (Markdown as Medium Language) is a structured, executable extension of Markdown designed as a communication and workflow medium for agentic AI systems that use the **Model Context Protocol (MCP)**.

Instead of scattering prompts, tool definitions, schemas, state, and logs across multiple files or ephemeral messages, a single `.maml.md` file acts as a self-contained, portable, versionable, human-readable *and* machine-executable artifact.

It is particularly well-suited for modern binary/native MCP harnesses such as:

- **OpenClaw** — multi-channel agent gateway with strong MCP client support, tool policy, and session management
- **Pi** (pi-coding-agent / pi SDK and related packages) — embedded agent runtime used by OpenClaw and other systems for high-control agent sessions
- Similar production-oriented harnesses that prefer efficient binary or native runtimes over pure interpreted scripting layers

MAML sits *on top of* MCP rather than replacing it. MCP handles tool discovery, calling, and transport (stdio, Streamable HTTP, SSE, etc.). MAML provides the rich, persistent, auditable *payload and skill definition* format that agents exchange or load.

### Core Design Goals

- Human-readable documentation that is simultaneously executable
- Self-describing workflows with explicit permissions, dependencies, and schemas
- Immutable or append-only execution history for auditability and reflection
- Portable skills/workflows that work across compatible MCP harnesses
- Support for multi-language code blocks (Python/CPython, OCaml, shell, SQL, Qiskit, etc.)
- Security-conscious design (permissions, signed tickets, validation) suitable for production agent loops

## Output Schema

Expected results and side effects.

## History

Append-only log of previous executions (timestamps, agent IDs, outcomes, error logs, Signed Execution Tickets if used).

### How It Works with MCP Harnesses (OpenClaw, Pi, etc.)

In modern binary MCP harnesses:

1. An agent or gateway discovers or receives a `.maml.md` file (via filesystem MCP server, memory store, message payload, or skill registry).
2. The harness (or a thin MAML-aware adapter/gateway) parses the YAML front matter for permissions, dependencies, and type.
3. The body is treated as structured context + executable sections.
4. Code blocks or declared tool calls are executed through the harness’s normal MCP tool interface or native execution environment.
5. Results, errors, and reflections are appended to the History section, turning the file into a living, auditable artifact.
6. The updated MAML can be passed to another agent, stored, or versioned in git.

### Key Benefits for Binary/Modern Harnesses

- **Token efficiency + structure** — Markdown is already the native language of most agent systems; MAML adds just enough structure without the verbosity of pure JSON/YAML.
- **Auditability & reproducibility** — Execution history lives with the skill.
- **Portability** — One file works across harnesses that understand the convention.
- **Security surface** — Explicit permissions, origin, and (in full implementations) signed tickets and validation.
- **Hybrid execution** — Mix natural language intent with real code and MCP tool calls.
- **Version control friendly** — Git-diffable skills and workflows.

### Typical Sections in a Production MAML

| Section          | Purpose                                      |
|------------------|----------------------------------------------|
| Intent           | High-level goal and triggering conditions    |
| Context          | Supporting knowledge and state               |
| Requires / Permissions | Dependencies and access control         |
| Input / Output Schema | Structured contracts                     |
| Code / Tools     | Executable blocks or MCP tool mappings       |
| Workflow Steps   | Ordered plan (for multi-step skills)         |
| Error Handling   | Expected failures and recovery               |
| History          | Immutable or append-only execution log       |

### Implementation Notes

- Full production implementations (as described in the WebXOS / MACROSLOW ecosystem) may include a MAML Gateway for validation, routing, Signed Execution Tickets, dual-mode AES encryption, and quantum-resistant features.
- For lightweight use in OpenClaw or Pi today, treat MAML as a strong convention: parse the front matter yourself or via a small helper, feed the body into the agent context, and execute declared tools/code through the existing MCP or native tool layers.
- Compatible with existing MCP servers (filesystem, memory, custom tools, etc.).

### Example Use Cases

- Reusable skills for OpenClaw agents (research, coding, ops, multi-channel actions)
- Shared workflows between Pi-embedded sessions and other agents
- Persistent agent memory or handoff packets
- Auditable tool-using pipelines
- Hybrid quantum/classical or multi-language workflows (where supported by the runtime)

### Status & Ecosystem

MAML originated in the open-source MACROSLOW / WebXOS work as a practical communication syntax for MCP-based agentic harnesses, with explicit design attention to systems like OpenClaw and similar modern runtimes. It remains a living convention rather than a frozen formal standard — adopt the core structure (YAML front matter + semantic Markdown body + history) and extend as needed for your harness.

## Intent
Provide a reusable, schema-validated data processing skill that can be invoked via MCP from Hermes or OpenClaw harnesses for cleaning and validating tabular datasets prior to analysis.

## Context
This skill targets CSV files from internal data pipelines. Expected columns include identifiers, numeric metrics, and categorical labels. Agents should maintain state across multiple invocations using the History section for progressive refinement.


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
