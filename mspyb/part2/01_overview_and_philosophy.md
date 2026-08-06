# MSPYB v3.0 — Overview and Core Philosophy

## Introduction

MSPYB (Megalithic Singular Python Bootstrap) is a format specification for packing an entire multi-file software project — directory tree, configuration files, source code, documentation, and instructions — into a single self-executing Python script that regenerates the full project on demand.

Version 3.0 builds directly on the foundations established in v2.0 (Error Logging, Templating, Versioning, and the mandatory INSTALLS section) and adds one decisive capability:

**Auto-Start / Auto-Launch**

Every MSPYB script is now expected to deliver a complete, seamless boot cycle:

1. Detect the host operating system and architecture.
2. Install or verify all required libraries at pinned versions (idempotent).
3. Materialize the entire project directory tree from the embedded definitions.
4. Automatically launch the generated application or service so the developer sees a running result immediately.

The canonical invocation remains deliberately simple:

```bash
cd /folder/containing/the/script
python3 bootstrap.py
```

No additional flags are required for the common case. The script detects Linux, macOS, or Windows at startup and chooses the appropriate launch method. A `--no-launch` flag (and the existing `--skip-installs` flag) are provided for continuous-integration, headless, or offline environments.

## Core Philosophy (v3.0)

MSPYB is not a framework or a library. It is a pattern and a file-format convention designed for delivering software both to humans and through large language models.

| Principle | Meaning |
|-----------|---------|
| Singular source of truth | One file holds the full project definition. |
| Self-bootstrapping | `python bootstrap.py` creates the entire tree after installing dependencies. |
| LLM-native | `LLM-INSTRUCTION` comments tell future agents how to extend or fix the system. |
| Megalithic | Intentionally large and complete rather than fragmented. |
| Executable documentation | The script is both the specification and the generator. |
| Idempotent and reproducible | Re-running the script regenerates a clean project. |
| Error Logging (v2) | Structured, persistent error logs with severity, context, and recovery hints. |
| Templating (v2) | `string.Template` expansion for configs, code, and docs at bootstrap time. |
| Versioning (v2) | Semantic version of the megalith + pinned dependencies + generated VERSION file. |
| INSTALLS (v2) | First-boot OS/device detection + automatic installation of correct libraries. |
| **Auto-Start / Auto-Launch (v3)** | **After generation, the script automatically starts the resulting application on the detected platform. Full-circle from single command.** |

The addition of Auto-Start closes the last remaining gap between “the project has been generated” and “the project is running.” In v2 the developer still had to remember the correct `cd`, `docker-compose up`, or `python app.py` command. In v3 that final step is performed by the bootstrap itself, using platform-aware launch logic.

## Design Goals for the Full-Circle Experience

- **Zero-friction first run.** A developer or an LLM agent who receives only the single `.py` file can produce a live, running system with one command.
- **Platform neutrality.** The same script behaves correctly on Linux, macOS, and Windows without requiring the user to supply OS-specific flags.
- **Safe defaults.** Auto-launch is on by default for interactive use, but can be disabled cleanly for automation.
- **Observable.** Every decision (install, generate, launch, or skip) is recorded in the structured log with recovery hints.
- **Reversible.** Re-running the script remains idempotent; a second launch does not create conflicting processes unless the generated application itself does so.

## Relationship to Previous Versions

- **v1** established the basic “one file generates many files” idea.
- **v2** made Error Logging, Templating, Versioning, and INSTALLS mandatory infrastructure.
- **v3** makes the bootstrap a complete self-contained runtime cycle: install → generate → launch.

All v2 requirements remain in force. A script that lacks Auto-Start is not a compliant MSPYB v3 megalith.

## What You Will Find in the Rest of This Guide

- Detailed treatment of the four required enhancements, with Auto-Start given equal weight to the original three.
- Expanded INSTALLS semantics that feed platform information directly into the launch subsystem.
- A complete canonical skeleton that includes the Auto-Launch block.
- Practical guidance on launching different kinds of generated projects (CLI tools, web services, Docker Compose stacks, native binaries, desktop applications).
- Updated checklists, best practices, and LLM-oriented workflow recommendations.

The single bootstrap script remains the living specification of the system. Everything else is generated from it — and, in v3, started by it.
