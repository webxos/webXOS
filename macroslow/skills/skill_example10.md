**Here is the complete `skill.md`**

```markdown
# MAML Skill: Hybrid Analysis + Reporting

**Skill Name:** `hybrid-analysis-reporting`  
**Version:** 1.0.0  
**Type:** `skill`  
**Target Harnesses:** Local Harness or Agent
**Purpose:** Perform Python-based data analysis and prepare results for visualization/reporting using JavaScript.

---

## Intent

Combine computational analysis in Python with JavaScript-based visualization preparation to produce ready-to-use insights and dashboard data.

---

## Context

Hybrid skill for analytics pipelines. Python handles heavy computation; JavaScript prepares outputs for web dashboards or agent UIs. Results can be consumed by reporting agents.

---

## Code_Blocks

```python
# Python analysis block
import pandas as pd
from typing import Dict, Any

def run_analysis(dataset_path: str) -> Dict[str, Any]:
    """Core statistical analysis"""
    df = pd.read_csv(dataset_path)
    
    stats = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "mean_values": df.mean(numeric_only=True).to_dict(),
        "summary": df.describe().to_dict()
    }
    
    return stats
```

```javascript
// JavaScript visualization & reporting prep (OpenClaw / UI friendly)
function prepareVisualization(stats) {
  const chartData = transformForChart(stats);
  
  return {
    reportTitle: "Analysis Summary Report",
    summaryStats: stats,
    chartData: chartData,
    recommendations: generateInsights(stats),
    generatedAt: new Date().toISOString()
  };
}

function transformForChart(stats) {
  // Prepare data for charts (bar, line, etc.)
  return {
    labels: Object.keys(stats.mean_values || {}),
    values: Object.values(stats.mean_values || {}),
    type: "bar"
  };
}

function generateInsights(stats) {
  const insights = [];
  if (stats.row_count > 1000) {
    insights.push("Large dataset detected - consider sampling for faster visualization.");
  }
  if (Object.keys(stats.mean_values || {}).length > 5) {
    insights.push("Multiple metrics available - recommend multi-chart dashboard.");
  }
  return insights;
}
```

---

## Input_Schema

```json
{
  "type": "object",
  "properties": {
    "dataset_path": { "type": "string", "description": "Path to dataset CSV" }
  },
  "required": ["dataset_path"]
}
```

## Output_Schema

```json
{
  "type": "object",
  "properties": {
    "reportTitle": { "type": "string" },
    "summaryStats": { "type": "object" },
    "chartData": { "type": "object" },
    "recommendations": { "type": "array" }
  }
}
```

---

## History

- 2026-06-29
