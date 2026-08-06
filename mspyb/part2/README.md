# MSPYB Format Specification and Full-Scope Guide — Version 3.0

**Megalithic Singular Python Bootstrap**

This archive contains the complete MSPYB v3.0 specification and practical guide.

## What Changed in v3.0

Version 3.0 elevates **Auto-Start / Auto-Launch** to a mandatory capability alongside the v2.0 requirements of Error Logging, Templating, Versioning, and the INSTALLS section.

Every compliant `bootstrap.py` now provides a seamless full-circle experience:

```bash
cd /path/to/folder          # directory containing bootstrap.py
python3 bootstrap.py        # installs dependencies, generates the entire project, then auto-launches it
```

Platform detection (Linux / macOS / Windows) occurs at startup by default. A `--no-launch` flag is provided for CI, headless, or air-gapped environments.

## Contents of this Guide

| File | Description |
|------|-------------|
| 01_overview_and_philosophy.md | Core philosophy updated for v3, new Auto-Start principle |
| 02_required_enhancements.md | The four required enhancements (Logging, Templating, Versioning, Auto-Start) |
| 03_installs_and_os_detection.md | Mandatory INSTALLS section with full OS/architecture handling |
| 04_canonical_structure_v3.md | Complete canonical skeleton including Auto-Launch |
| 05_auto_start_subsystem.md | Detailed design of the Auto-Start / Auto-Launch subsystem |
| 06_workflow_and_best_practices.md | Recommended workflow, checklist, and conventions |
| 07_why_v3_works_with_llms.md | Why the full-circle design is especially powerful for LLM agents |
| 08_extension_points_and_examples.md | How to extend Auto-Launch for different project types |
| canonical_bootstrap_v3.py | Reference implementation skeleton (annotated) |

Start with `01_overview_and_philosophy.md` and proceed sequentially.

MSPYB remains a pattern and file-format convention, not a framework or library. The single bootstrap script is the singular source of truth.
