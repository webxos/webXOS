**Here is the complete `skill.md`**

```markdown
# MAML Skill: Schema Evolution Manager

**Skill Name:** `schema-evolution-manager`  
**Version:** 1.0.0  
**Type:** `skill`  
**Target Harnesses:** Local Harness or Agent 
**Purpose:** Track and manage changes to Input/Output schemas over time, detect drift, and recommend updates for long-term skill maintenance.

---

## Intent

Monitor schema evolution across multiple executions, detect breaking changes or drift, and provide actionable recommendations to keep skills compatible.

---

## Context

Critical for maintaining reusable skills over months. Uses History to compare current schemas against past versions. Helps prevent silent failures in evolving agent ecosystems.

---

## Code_Blocks

```python
from typing import Dict, Any, List
import json

def compare_schemas(current_input: Dict, current_output: Dict, history: List[Dict]) -> Dict[str, Any]:
    """
    Compare current schemas with historical ones and detect drift.
    """
    previous_schemas = extract_previous_schemas(history)
    
    input_drift = detect_drift(current_input, previous_schemas.get("input", {}))
    output_drift = detect_drift(current_output, previous_schemas.get("output", {}))
    
    recommendations = generate_evolution_plan(input_drift, output_drift)
    
    return {
        "status": "analyzed",
        "input_drift_detected": bool(input_drift),
        "output_drift_detected": bool(output_drift),
        "input_changes": input_drift,
        "output_changes": output_drift,
        "recommendations": recommendations,
        "history_entries_analyzed": len(history)
    }

def extract_previous_schemas(history: List[Dict]) -> Dict[str, Dict]:
    """Extract the most recent schemas from History"""
    for entry in reversed(history):
        if "schemas" in entry:
            return entry["schemas"]
    return {}

def detect_drift(current: Dict, previous: Dict) -> List[str]:
    """Simple schema drift detection"""
    changes = []
    if current.get("properties") and previous.get("properties"):
        current_keys = set(current["properties"].keys())
        prev_keys = set(previous["properties"].keys())
        if current_keys != prev_keys:
            changes.append(f"Property changes: {current_keys.symmetric_difference(prev_keys)}")
    return changes

def generate_evolution_plan(input_drift: List, output_drift: List) -> List[str]:
    """Generate recommendations"""
    plan = []
    if input_drift:
        plan.append("Update Input_Schema to accommodate new fields or types.")
    if output_drift:
        plan.append("Review and version Output_Schema for breaking changes.")
    if not plan:
        plan.append("No significant schema drift detected. Skill remains compatible.")
    return plan
```

---

## Input_Schema

```json
{
  "type": "object",
  "properties": {
    "current_input_schema": { "type": "object" },
    "current_output_schema": { "type": "object" },
    "history": { "type": "array", "description": "Recent History entries" }
  },
  "required": ["current_input_schema", "current_output_schema"]
}
```

## Output_Schema

```json
{
  "type": "object",
  "properties": {
    "status": { "type": "string" },
    "input_drift_detected": { "type": "boolean" },
    "output_drift_detected": { "type": "boolean" },
    "recommendations": { "type": "array" }
  }
}
```

---

## History

- 2026-06-29
