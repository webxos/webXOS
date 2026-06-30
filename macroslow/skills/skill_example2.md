**This is a fully complete `skill.md`**

```markdown
# MAML Skill: External API Orchestration

**Skill Name:** `external-api-orchestrator`  
**Version:** 1.0.0  
**Type:** `skill`  
**Target Harnesses:** Local Harness or Agent
**Purpose:** Fetch live data from external APIs, validate it, process locally, and return clean structured results.

---

## Intent

Provide a reliable, reusable skill that fetches data from external REST APIs and performs local validation and processing before returning results to calling agents.

---

## Context

This skill is optimized for OpenClaw gateway scenarios where agents need to pull real-time data. It supports API key authentication via environment variables and returns validated records using Pydantic. Ideal for data ingestion pipelines, monitoring agents, and research workflows.

---

## Code_Blocks

```javascript
// JavaScript block for OpenClaw gateway orchestration and API fetching
async function fetchExternalData(endpoint) {
  try {
    const headers = {};
    if (process.env.API_KEY) {
      headers.Authorization = `Bearer ${process.env.API_KEY}`;
    }
    const response = await fetch(endpoint, { 
      method: 'GET',
      headers: headers 
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error("API fetch failed:", error);
    throw error;
  }
}
```

```python
# Python block for validation and processing
from pydantic import BaseModel, field_validator
import pandas as pd
from typing import Dict, Any, List

class ApiRecord(BaseModel):
    id: int
    value: float
    category: str | None = None

    @field_validator('value')
    def value_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 10000.0:
            raise ValueError('Value must be between 0.0 and 10000.0')
        return round(v, 4)

def process_api_data(raw_data: List[Dict]) -> Dict[str, Any]:
    """Process and validate API response data."""
    try:
        df = pd.DataFrame(raw_data)
        validated_records = []
        errors = []
        
        for item in df.to_dict('records'):
            try:
                record = ApiRecord(**item)
                validated_records.append(record.model_dump())
            except Exception as e:
                errors.append({"item": item, "error": str(e)})
        
        return {
            "status": "success",
            "processed_records": len(validated_records),
            "error_count": len(errors),
            "sample_validated": validated_records[:5],
            "total_fetched": len(raw_data)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
```

---

## Input_Schema

```json
{
  "type": "object",
  "properties": {
    "endpoint": {
      "type": "string",
      "description": "Full URL of the external API endpoint"
    },
    "api_key_env": {
      "type": "string",
      "default": "API_KEY"
    }
  },
  "required": ["endpoint"]
}
```

## Output_Schema

```json
{
  "type": "object",
  "properties": {
    "status": { "type": "string" },
    "processed_records": { "type": "integer" },
    "error_count": { "type": "integer" },
    "sample_validated": { "type": "array" },
    "total_fetched": { "type": "integer" }
  }
}
```

---

## History

- 2026-06-29
