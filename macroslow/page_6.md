# MAML (Markdown as Medium Language): A Practical Communication Medium for Modern MCP-Based Agentic Harnesses

**Report for Skill.md Integration**  
**Prepared for Agentic AI Developers**  
**Focus: Non-Quantum PCs, MCP Compatibility, Hermes & OpenClaw Harnesses**  
**Version: 1.0 (Adapted from Webxos Concepts, June 2026)**  
**Page 6 of 10**

## 6. Skill Definition Templates and Examples for .maml.md Files

This page provides ready-to-use templates and concrete examples for defining skills in MAML format. These templates are designed for direct inclusion in skill.md repositories and seamless use within Hermes and OpenClaw harnesses via MCP.

### Template 1: Basic Data Processing Skill

```markdown
---
maml_version: "1.0.0"
id: "urn:uuid:replace-with-generated-uuid"
type: "skill"
origin: "agent://skill-repository"
requires:
  libs: ["pydantic>=2.0", "pandas>=2.0"]
permissions:
  read: ["agent://*"]
  execute: ["gateway://local", "agent://hermes", "agent://openclaw"]
created_at: "2026-06-28T00:00:00Z"
timeout_seconds: 120
---
## Intent
Clean and validate tabular data from CSV files, returning structured statistics and cleaned output.

## Context
Input files contain mixed numeric and categorical data. Apply standard cleaning rules. Store summary statistics for memory integration in Hermes.

## Code_Blocks
```python
from pydantic import BaseModel, field_validator
import pandas as pd
from typing import Dict, Any

class Record(BaseModel):
    id: int
    value: float
    label: str

    @field_validator('value')
    def check_value(cls, v):
        if v < 0:
            raise ValueError("Value cannot be negative")
        return v

def run_skill(input_path: str, output_path: str = None) -> Dict[str, Any]:
    df = pd.read_csv(input_path)
    valid = []
    invalid = []
    for _, row in df.iterrows():
        try:
            rec = Record(**row.to_dict())
            valid.append(rec.model_dump())
        except Exception as e:
            invalid.append({"row": row.to_dict(), "error": str(e)})
    result = {
        "total_rows": len(df),
        "valid_rows": len(valid),
        "invalid_rows": len(invalid),
        "summary": df.describe().to_dict() if not df.empty else {}
    }
    if output_path and valid:
        pd.DataFrame(valid).to_csv(output_path, index=False)
    return result
```

## Input_Schema
{
  "type": "object",
  "properties": {
    "input_path": {"type": "string"},
    "output_path": {"type": "string"}
  },
  "required": ["input_path"]
}

## Output_Schema
{
  "type": "object",
  "properties": {
    "total_rows": {"type": "integer"},
    "valid_rows": {"type": "integer"},
    "invalid_rows": {"type": "integer"},
    "summary": {"type": "object"}
  }
}

## History
- 2026-06-28T00:00:00Z: [CREATE] Template instantiated.

### Template 2: API Integration Skill (JavaScript + Python Hybrid)

```markdown
---
maml_version: "1.0.0"
id: "urn:uuid:api-skill-002"
type: "skill"
origin: "agent://openclaw-gateway"
requires:
  libs: ["requests"]
  apis: ["external-api"]
permissions:
  execute: ["agent://openclaw"]
---
## Intent
Fetch data from external API and process with local validation.

## Code_Blocks
```python
import requests
def fetch_and_validate(url):
    response = requests.get(url)
    data = response.json()
    # Validation logic
    return {"fetched": len(data), "status": "ok"}
```

```javascript
// Post-processing in gateway context
async function enrichResult(result) {
  console.log("Enriched for OpenClaw:", result);
  return result;
}
```

## History

### Advanced Template Features for Agentic Loops

Include adaptive logic that reads previous History entries for continuous improvement. Use `loop_compatible: true` in front matter.

**Best Practices for Skill Repositories**:
- One skill per `.maml.md` file.
- Generate UUIDs using `uuidgen` or online tools.
- Maintain a catalog Markdown index of all skills with summaries.
- Version skills by updating `maml_version` and History.
- Include sample input data in a companion `/tests/` directory.
- Document harness-specific notes in `## Context`.

These templates allow developers to rapidly build a library of reusable MAML skills that integrate natively with Hermes memory systems and OpenClaw orchestration layers, accelerating agent development while maintaining standardization through MCP.

(End of Page 6. Continued on Page 7: Security, Validation, and Best Practices.)
