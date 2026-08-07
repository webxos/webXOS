# Dynamic Programming with LLMs (2026)

**A Practical Guide to Megalithic Orchestration for the Agentic Frontier**

This package contains a complete 10-page arXiv-style technical report and supporting materials that reframe classic Bellman-style dynamic programming as a design principle for large, self-contained “megalith” scripts and documents. These megaliths — single-file HTML (MHTML-style), Python bootstraps (MSPYB), Bash agents (UNIXAI-style), and PowerShell modules — serve as living specifications that an LLM or autonomous agent can generate, mutate, and execute end-to-end.

## Contents

| Path | Description |
|------|-------------|
| `Dynamic_Programming_with_LLMs_2026.pdf` | Full arXiv-format technical report (~10 pages) |
| `README.md` | This overview |
| `guides/MHTML_Orchestration.md` | Self-contained HTML prototype skill summary |
| `guides/MSPYB_Python_Bootstrap.md` | Singular Python bootstrap pattern (v3 Auto-Start) |
| `guides/UNIXAI_Bash_Megalith.md` | Anatomy of a production Bash AI agent |
| `guides/PowerShell_Megalith_Guide.md` | Equivalent patterns for Windows / cross-platform PS1 |
| `examples/` | Minimal working skeletons for each format |
| `src/bootstrap_example.py` | Tiny MSPYB-style generator (illustrative) |

## Core Thesis

Treat every large deliverable as a **single source of truth** that:

1. Declares its own environment and dependencies,
2. Generates or contains the complete application tree,
3. Can be launched (or re-launched) by the same file,
4. Exposes clear extension points and recovery hints for an agent,
5. Remains idempotent and offline-capable.

This is dynamic programming applied to software construction: optimal substructure appears as reusable templates and shared logging; overlapping subproblems appear as the repeated generation–launch–observe cycle that an LLM performs.

## Citation

```
@techreport{dp-llms-2026,
  title   = {Dynamic Programming with LLMs: Megalithic Orchestration for the Agentic Frontier},
  author  = {Grok (xAI)},
  year    = {2026},
  month   = {August},
  note    = {Technical report and companion materials},
  url     = {local package}
}
```

## License

Materials are released for research and educational use. Adapt freely; attribution appreciated.
