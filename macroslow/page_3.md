# MAML (Markdown as Medium Language): A Practical Communication Medium for Modern MCP-Based Agentic Harnesses

**Report for Skill.md Integration**  
**Prepared for Agentic AI Developers**  
**Focus: Non-Quantum PCs, MCP Compatibility, Hermes & OpenClaw Harnesses**  
**Version: 1.0 (Adapted from Webxos Concepts, June 2026)**  
**Page 3 of 10**

## 3. Code Block Examples and Multi-Language Support for Agentic Workflows

The `## Code_Blocks` section is the executable heart of a MAML file. It allows developers to embed runnable code directly within the document, making MAML files both documentation and implementation. For non-quantum environments, the focus is on Python (primary), JavaScript/Node.js (for web/gateway integrations), and lightweight shell scripts. This page provides complete, production-ready examples optimized for MCP tool invocation within Hermes and OpenClaw harnesses.

### Supported Languages and MCP Routing

MCP servers and agent harnesses detect the language tag in fenced code blocks (e.g., ````python`) and route execution accordingly. Recommended priorities for classical hardware:

- **python**: Default for data processing, validation, and core logic. Use Python 3.8+ with common scientific libraries.
- **javascript**: Ideal for OpenClaw gateway interactions, API orchestration, and browser-adjacent tasks.
- **bash**: For environment setup, file operations, or simple scripting (use sparingly for security).
- **Other**: Extendable via custom runners in harness configurations.

Code blocks can appear multiple times in one file for hybrid workflows (e.g., Python analysis followed by JavaScript reporting).

### Example 1: Python Data Validation Skill (Recommended for Skill.md)

## Code_Blocks
```python
from pydantic import BaseModel, field_validator, ValidationError
import pandas as pd
import json
from typing import Dict, Any

class ValidatedRecord(BaseModel):
    record_id: int
    metric: float
    category: str

    @field_validator('metric')
    def metric_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1000.0:
            raise ValueError('Metric must be between 0.0 and 1000.0')
        return round(v, 4)

def execute_skill(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for MCP invocation.
    input_data expected from Input_Schema.
    """
    try:
        input_path = input_data['input_path']
        df = pd.read_csv(input_path)
        
        valid_records = []
        errors = []
        
        for index, row in df.iterrows():
            try:
                record = ValidatedRecord(**row.to_dict())
                valid_records.append(record.model_dump())
            except ValidationError as e:
                errors.append({
                    "row_index": int(index),
                    "data": row.to_dict(),
                    "errors": e.errors()
                })
        
        result = {
            "status": "success",
            "processed": len(valid_records),
            "errors": len(errors),
            "valid_sample": valid_records[:3] if valid_records else [],
            "error_summary": errors[:5] if errors else []
        }
        
        # Optional: persist results for History
        with open("/tmp/maml_result.json", "w") as f:
            json.dump(result, f, indent=2)
        
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}
```

### Example 2: JavaScript for OpenClaw Gateway Orchestration
```markdown
```javascript
// JavaScript block for MCP response formatting or external API calls
async function orchestrateMCP(data) {
  const response = await fetch('http://localhost:8000/mcp/execute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      maml_id: data.maml_id,
      payload: data.payload
    })
  });
  
  const result = await response.json();
  console.log('MCP Execution Result:', result);
  
  // Return structured data for Hermes memory integration
  return {
    timestamp: new Date().toISOString(),
    action: 'orchestrated',
    payload: result
  };
}

// Example invocation (called from harness)
orchestrateMCP({ maml_id: 'urn:uuid:example', payload: { input_path: './data.csv' } })
  .then(console.log)
  .catch(console.error);
```

### Example 3: Hybrid Workflow with History Update

Multiple blocks can be chained:

```markdown
```python
# Python preprocessing block
import pandas as pd
df = pd.read_csv("input.csv")
print("Preprocessed rows:", len(df))
```

```javascript
// JavaScript post-processing for reporting
const fs = require('fs');
const data = JSON.parse(fs.readFileSync('/tmp/maml_result.json', 'utf8'));
console.log(`Final Report: ${data.processed} records processed.`);
```

### Integration Patterns with Hermes and OpenClaw

- **Hermes (Memory-Focused)**: Code blocks should read/write to memory plugins via context variables. Use the History section to log reflective notes (e.g., "Refined validation rule based on previous errors").
- **OpenClaw (Gateway-Focused)**: JavaScript blocks handle routing to sub-agents or external MCP servers. Python blocks perform heavy computation.
- **Agentic Loops**: Design code to be idempotent and state-aware. After execution, the harness appends to History:
  
  ```markdown
  ## History
  - 2026-06-28T02:24:00Z: [EXECUTE] Python validation completed with 95% success rate.
  - 2026-06-28T02:25:00Z: [REFLECT] Adjusted threshold based on error distribution.
  ```

### Best Practices for Code Blocks in Skill.md

1. Always include a clear main function (e.g., `execute_skill(input_data)`) for MCP entry points.
2. Handle exceptions gracefully and return structured error objects.
3. Respect `Input_Schema` and `Output_Schema` strictly.
4. Keep blocks modular and under 200 lines for maintainability.
5. Test locally with sample data before deploying to harnesses.
6. Use comments liberally to explain MCP/harness interactions.
7. Avoid hard-coded paths; use context variables or environment variables.

By leveraging these code block patterns, MAML files become powerful, self-documenting skills that agents in Hermes and OpenClaw can discover, invoke, and evolve over time through iterative History updates.

(End of Page 3. Continued on Page 4: Integration Patterns with Hermes and OpenClaw Harnesses.)
