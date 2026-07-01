**Here is a complete `skill.md file`.**

```markdown
# MAML Skill: Sub-Agent Delegation

**Skill Name:** `sub-agent-delegation`  
**Version:** 1.0.0  
**Type:** `skill`  
**Target Harnesses:** Local harness or Agent
**Purpose:** Coordinate and delegate subtasks to specialized sub-agents via MCP, then aggregate results.

---

## Intent

Enable intelligent task decomposition and delegation to multiple specialized agents, with orchestration and result aggregation.

---

## Context

Ideal for OpenClaw gateway scenarios where a main agent needs to split complex tasks across specialized sub-agents (e.g., analyzer, summarizer, validator). Returns consolidated results for the calling agent.

---

## Code_Blocks

```javascript
// JavaScript orchestration layer (perfect for OpenClaw gateway)
async function delegateSubtasks(mainTask) {
  const subAgents = ['analyzer', 'summarizer', 'validator'];
  
  console.log(`Delegating task "${mainTask.title}" to ${subAgents.length} sub-agents...`);
  
  try {
    const results = await Promise.all(
      subAgents.map(async (agent) => {
        const subResult = await callMCPSubAgent(agent, mainTask);
        return { agent, result: subResult, timestamp: new Date().toISOString() };
      })
    );
    
    const aggregated = aggregateResults(results);
    return {
      status: "success",
      task: mainTask.title,
      subAgentsUsed: subAgents,
      results: aggregated
    };
  } catch (error) {
    console.error("Delegation failed:", error);
    return { status: "error", message: error.message };
  }
}

async function callMCPSubAgent(agentName, task) {
  // Call via MCP to sub-agent
  const response = await fetch(`http://localhost:8001/mcp/delegate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      targetAgent: agentName,
      task: task
    })
  });
  return await response.json();
}

function aggregateResults(subResults) {
  return {
    totalSubTasks: subResults.length,
    successful: subResults.filter(r => r.result.status === "success").length,
    combinedOutput: subResults.map(r => ({
      agent: r.agent,
      output: r.result
    }))
  };
}
```

```python
# Python fallback / post-processing
def finalize_delegation(aggregated_data):
    """Additional Python processing if needed"""
    summary = {
        "overall_status": "success" if aggregated_data["successful"] == aggregated_data["totalSubTasks"] else "partial",
        "completion_rate": round(aggregated_data["successful"] / aggregated_data["totalSubTasks"] * 100, 2)
    }
    return summary
```

---

## Input_Schema

```json
{
  "type": "object",
  "properties": {
    "title": { "type": "string", "description": "Main task title" },
    "description": { "type": "string" },
    "subAgents": { 
      "type": "array", 
      "items": { "type": "string" },
      "default": ["analyzer", "summarizer", "validator"]
    }
  },
  "required": ["title"]
}
```

## Output_Schema

```json
{
  "type": "object",
  "properties": {
    "status": { "type": "string" },
    "task": { "type": "string" },
    "subAgentsUsed": { "type": "array" },
    "results": { "type": "object" }
  }
}
```

---
*Use at your own risk*

## History

- 2026-06-29
