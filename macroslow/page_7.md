# MAML (Markdown as Medium Language): A Practical Communication Medium for Modern MCP-Based Agentic Harnesses

**Report for Skill.md Integration**  
**Prepared for Agentic AI Developers**  
**Focus: Non-Quantum PCs, MCP Compatibility, Hermes & OpenClaw Harnesses**  
**Version: 1.0 (Adapted from Webxos Concepts, June 2026)**  
**Page 7 of 10**

## 7. Security, Validation, and Best Practices

Robust security and validation are critical when using MAML in production agentic environments. This page outlines comprehensive guidelines to protect against risks while maintaining the flexibility needed for Hermes and OpenClaw harnesses.

### Permission and Access Control Model

The `permissions` object in the front matter enforces least-privilege execution:

```yaml
permissions:
  read: ["agent://*", "harness://openclaw"]      # Who can read the file
  write: ["agent://hermes-core"]                 # Who can modify History
  execute: ["gateway://local", "agent://trusted-subagent"]  # Execution rights
```

**Best Practices**:
- Never use wildcards (`*`) in production `execute` lists unless the skill is purely read-only.
- Validate the calling agent's identity against the list at MCP gateway level.
- Use origin tracking to prevent spoofing.

### Input/Output Schema Validation

Always define and enforce schemas using Pydantic (Python) or equivalent validators:

- Reject malformed inputs early.
- Sanitize outputs before returning to agents.
- Log validation failures to History with details (without exposing sensitive data).

**Implementation Tip**:
Wrap code blocks with try/except blocks that return structured error objects compliant with `Output_Schema`.

### Sandboxing and Execution Security

**Recommended Approaches**:
1. **Docker Containers**: Isolate each MAML execution with resource limits (CPU, memory, network).
2. **Read-Only Filesystems**: Mount only necessary volumes.
3. **Network Controls**: Allowlist MCP endpoints and external APIs.
4. **Dependency Pinning**: Use exact versions in `requires.libs` to prevent supply-chain attacks.

**Hermes-Specific**: Run skills in memory-isolated contexts to prevent cross-contamination between agent sessions.

**OpenClaw-Specific**: Gateway-level sandboxing before dispatching to sub-agents.

### History Integrity and Auditing

- Make History append-only; never allow deletion of past entries.
- Include timestamps, actor identity, and action type in every entry.
- Consider lightweight checksums or harness-level signing for critical skills.
- Use History for debugging loops and compliance auditing.

### Common Security Pitfalls and Mitigations

| Risk                        | Mitigation Strategy |
|-----------------------------|---------------------|
| Code Injection              | Sandbox execution; avoid `eval`/`exec` |
| Path Traversal              | Validate and sanitize all file paths |
| Resource Exhaustion         | Enforce `timeout_seconds` and Docker limits |
| Unauthorized Execution      | Strict permission checks at gateway |
| Sensitive Data Leakage      | Redact outputs; use context variables instead of hard-coding secrets |
| Schema Bypass               | Dual validation (pre- and post-execution) |

### Validation Checklist for Skill Authors

Before publishing a MAML skill:
- [ ] All required front matter fields present and valid.
- [ ] Schemas are comprehensive and tested with edge cases.
- [ ] Code blocks are deterministic and handle errors gracefully.
- [ ] Permissions follow least privilege.
- [ ] Tested in both Hermes (memory) and OpenClaw (gateway) environments.
- [ ] History template entries are included.
- [ ] File size < 500KB for efficient MCP transmission.
- [ ] No hard-coded credentials or secrets.

### Monitoring and Observability

- Instrument code blocks with structured logging.
- Have harnesses aggregate History entries from multiple MAML executions.
- Set up alerts for repeated validation failures or permission violations.

Adhering to these practices ensures MAML skills are secure, reliable, and maintainable, enabling confident deployment in production agentic systems powered by MCP, Hermes, and OpenClaw.

(End of Page 7. Continued on Page 8: Real-World Use Cases and Performance Considerations.)
