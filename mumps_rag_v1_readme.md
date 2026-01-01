# MUMPS RAG Terminal (mumps_rag_v1.py)

A Lightweight Python terminal simulating basic MUMPS global storage with in-memory database and simple RAG retrieval over `^DOC` nodes.

*MUMPS (Massachusetts General Hospital Utility Multi-Programming System), also known as M, uses a
concise, whitespace-aware syntax designed for high-throughput data processing and a built-in hierarchical database.*

## Overview

- Emulates MUMPS-style globals (`^NAME(sub1,sub2)`)
- Flat internal storage using `|` separator
- Preloaded sample medical documentation
- Basic TF-based ranking for RAG queries
- Interactive REPL interface

## Requirements

Python 3.6+  
No external dependencies (uses only standard library)

## Quick Start

```bash
python3 mumps_rag_v1.py
```

```
*** MUMPS RAG TERMINAL (by webXOS 2025) ***
Type HELP
```

## Basic MUMPS-Style Commands

| Command                  | Syntax Example                              | Description                                      |
|--------------------------|---------------------------------------------|--------------------------------------------------|
| **SET**                  | `SET ^PATIENT(123)="John Doe"`              | Create or update global node                     |
|                          | `SET ^LAB(123,"GLU")=105`                   | Supports subscripts                              |
| **GET**                  | `GET ^PATIENT(123)`                         | Retrieve value of node                           |
| **LIST**                 | `LIST ^PATIENT`                             | List all nodes under root (exact or subtree)     |
|                          | `LIST ^LAB`                                 | Shows all subnodes                               |
| **DOC SHOW**             | `DOC SHOW`                                  | Display all documentation nodes (^DOC)           |
| **DOC ADD**              | `DOC ADD ^DOC(6)="New medical note text"`   | Add new documentation entry                      |
| **RAG QUERY**            | `RAG QUERY chest pain assessment`           | Search ^DOC nodes, returns top 5 ranked matches  |
| **HELP**                 | `HELP`                                      | Show command summary                             |
| **EXIT** / **QUIT**      | `EXIT`                                      | Quit terminal                                    |

## Key Syntax Notes

- Keys always start with `^`
- Subscripts in parentheses: `^GLOBAL("sub1",sub2,"sub 3")`
- Quotes optional if subscript has no spaces/commas
- Internal flattening: `^DOC("chest pain")` → `DOC|chest pain`

## Preloaded Sample ^DOC Nodes

1. `^DOC(1)` – SOAP framework explanation  
2. `^DOC(2)` – ABCDE emergency assessment  
3. `^DOC(3)` – Chest pain assessment protocol  
4. `^DOC(4)` – Emergency department billing levels  
5. `^DOC(5)` – Diabetes follow-up visit checklist  

## Example Session

```
USER> DOC SHOW
  ^DOC(1) = SOAP framework: Subjective, Objective, Assessment, Plan...
  ^DOC(2) = ABCDE framework: Airway, Breathing, Circulation...
  ...

USER> RAG QUERY diabetes follow up
RAG: diabetes follow up
  #1 [^DOC(5)] 0.812
      Diabetes follow-up visit: check A1C, medications...

USER> DOC ADD ^DOC(6)="Hypertension management: BP targets, lifestyle, meds"
DOC ADDED

USER> RAG QUERY high blood pressure
  #1 [^DOC(6)] 0.745
      Hypertension management: BP targets...
```

## Extending

Add any global data via `SET` or more `^DOC` entries via `DOC ADD`. All data persists only in memory during session.

Free to modify and extend. Public domain / Unlicense.
