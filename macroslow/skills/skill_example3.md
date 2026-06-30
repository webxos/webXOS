**Here is the full complete `skill.md`**

```markdown
# MAML Skill: Multi-Step Research Workflow

**Skill Name:** `multi-step-research-workflow`  
**Version:** 1.0.0  
**Type:** `workflow`  
**Target Harnesses:** Hermes (memory & reflection) + OpenClaw (orchestration & delegation)  
**Purpose:** Execute a complete end-to-end research pipeline: query → literature search → data analysis → final report generation.

---

## Intent

Orchestrate a multi-step research workflow that chains search, analysis, and reporting stages, producing a structured final output suitable for agent review or human consumption.

---

## Context

This workflow is ideal for research agents. It can call sub-MAML skills via MCP and maintains state through the History section. Hermes can use past research runs to improve future queries; OpenClaw can delegate individual steps to specialized sub-agents.

---

## Code_Blocks

```python
# Main orchestration logic
def orchestrate_research(query: str, max_steps: int = 3):
    """
    Full research workflow:
    1. Search for relevant information
    2. Analyze collected data
    3. Generate final report
    """
    # Step 1: Literature / Data Search (can call external MCP tool)
    literature = perform_search(query)
    
    # Step 2: Analysis
    insights = analyze_findings(literature)
    
    # Step 3: Report Generation
    report = generate_final_report(query, insights)
    
    return {
        "query": query,
        "steps_completed": max_steps,
        "insights": insights,
        "final_report": report,
        "status": "completed"
    }

def perform_search(query: str):
    """Stub for search step - replace with real MCP tool call or API"""
    # Example: Call external search MAML or API
    return [
        {"title": "Paper 1 on " + query, "summary": "Key findings..."},
        {"title": "Paper 2 on " + query, "summary": "Additional insights..."}
    ]

def analyze_findings(literature):
    """Analyze collected data"""
    # Simple analysis - can be expanded with ML models
    total_papers = len(literature)
    key_themes = ["innovation", "challenges", "future directions"]
    return {
        "total_sources": total_papers,
        "key_themes": key_themes,
        "summary": f"Analyzed {total_papers} sources on {literature[0]['title'] if literature else 'topic'}"
    }

def generate_final_report(query: str, insights: dict):
    """Generate markdown-style final report"""
    return f"""
# Research Report: {query}

## Summary
{insights['summary']}

## Key Themes
{', '.join(insights['key_themes'])}

## Recommendations
- Continue monitoring latest publications
- Consider experimental validation
"""
```

---

## Input_Schema

```json
{
  "type": "object",
  "properties": {
    "query": { "type": "string", "description": "Research query or topic" },
    "max_steps": { "type": "integer", "default": 3 }
  },
  "required": ["query"]
}
```

## Output_Schema

```json
{
  "type": "object",
  "properties": {
    "query": { "type": "string" },
    "steps_completed": { "type": "integer" },
    "insights": { "type": "object" },
    "final_report": { "type": "string" },
    "status": { "type": "string" }
  }
}
```

---

## History

- 2026-06-29
