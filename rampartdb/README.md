# RAMPARTDB

**© MIT webXOS 2026 · v3.0.7 · Anchor-Guarded English · Self-Contained · Boot Lint · 14 fixes applied**

> A fully on-device, anchor-guarded OSINT knowledge database that runs entirely inside your browser. One HTML file. No build step. No cloud. No CDN at startup. Every word you type is classified as either an **anchor** (function word) or a **payload noun** (content word). Your PII is redacted on-device before anything else happens. The network is only reached when you explicitly ask for it with `/osint`.

---

## What It Is

RAMPARTDB is a single-file, browser-native knowledge database. It is not a cloud service, not a chat wrapper, and not a frontend to a remote model. It is a local reasoning sandbox with:

- **Anchor-Guarded English** — every sentence is decomposed into *anchors* (articles, prepositions, pronouns, auxiliaries, conjunctions, quantifiers, negations, degree words, particles) and *payload nouns* (content words with dictionary definitions).
- **On-device PII redaction** — emails, IPv4/IPv6 addresses, phone numbers, SSNs, credit cards and URLs are replaced with numbered placeholders before anything else touches your data.
- **Local-first retrieval** — every reply starts by searching your local RAMPART DB via a 384-dimension hash vector store.
- **OSINT-gated fan-out** — the network is only reached when you explicitly type `/osint <topic>`.
- **Event-sourced** — every insert, update, delete, tag, pin, verify, trash, restore, merge, import and clear is written to a JSON Event log (JEV).
- **Live knowledge graph** — every turn adds nodes and edges to a 2D canvas graph.
- **Self-diagnosing** — a boot lint runs 25+ subsystem checks at startup and prints the results into the chat.

---

## Quick Start

1. Save the file as `rampartdb.html`.
2. Open it in a modern browser.
3. Type something:
   ```
   hello
   what is a cylinder
   how does gravity work
   /help
   ```
4. Optional — live research (requires http(s), see [Deployment](#deployment)):
   ```
   /osint solar physics
   ```

---

## Deployment

**Local file (`file://`)** — double-click. Everything works except `/osint`, which requires an http(s) origin for CORS.

**Local HTTP server (recommended for OSINT)**
```bash
python3 -m http.server 8000
# or
npx serve .
# or
php -S localhost:8000
```
Then open `http://localhost:8000/rampartdb.html`.

**Production** — when served over HTTP(S), send this response header (the meta CSP cannot enforce `frame-ancestors`):
```
Content-Security-Policy: frame-ancestors 'none'
```

---

## First Run

You will see:

1. A **boot lint** block listing every subsystem check.
2. A **welcome card** with `[ LOAD DEMO PACK ]`, `[ SEARCH LIBRARY ]`, `[ RUN OSINT ]`, `[ HELP ]`.
3. A **first-run checklist** (add 3 entries, search DB, run one OSINT pass, export DB).

Click `[ LOAD DEMO PACK ]` to insert five demo entries (Cylinder, Photosynthesis, Gravity, Climate change, Log4Shell).

---

## Interface

```
+---------------------------------------------------+------------------+
|  RAMPARTDB      [ws] [status] [OSINT] [⚙] [⛶]     |  Graph | Library |
+---------------------------------------------------+  -------+-------+
|                                                   |                  |
|  chat messages  (user / agent / system / anchors  |   2D canvas      |
|  / pii / osint / db / diag / sources / boot)      |   or Library     |
|                                                   |   or Integrity   |
+---------------------------------------------------+                  |
|  [UPLOAD] [CONVERT] [EXPORT]  [input]  [Send]     |------------------|
+---------------------------------------------------+  status badges   |
|  © webXOS 2026 footer                             |  proto ws ...    |
+---------------------------------------------------+------------------+
```

**Left column** — workspace selector, OSINT toggle, settings, fullscreen; chain banner; messages; input area with UPLOAD, CONVERT, EXPORT.

**Right column** — three tabs: **Graph** (2D canvas), **Library** (search/filter entries), **Integrity** (subsystem health).

**Status bar** — live badges: `proto ws rampart pii dict anchors payload tag api conf NLP mode vec rdb jev graph focus trash watch chain osint guard net ~kb latency`.

---

## Core Concepts

### Anchors vs. Payload Nouns

Every token is classified into one of three buckets:

1. **Anchors** — function words (~1,000 in the rail), grouped by role: `interrogative`, `article`, `demonstrative`, `determiner`, `quantifier`, `negation`, `pronoun`, `auxiliary`, `preposition`, `conjunction`, `adverb-time`, `adverb-place`, `adverb-manner`, `adverb-reason`, `degree`, `particle`.
2. **Payload nouns** — content words looked up in the dictionary, POS-variant map, and stemmer.
3. **Numbers** — matched by `^\d+(\.\d+)?$`.

The **signature** of a turn describes its anchor composition, e.g. `interrogativex1 · articlex1 · prepositionx2 · payload · cylinder`. **Intent** is derived from the interrogative anchor: `compare`, `howto`, `explain`, `define`, `who`, `when`, `where`, `which`, `hypothetical`, `other`.

### The Rampart PII Engine

Runs on every input before anything else. Patterns applied in order:

| Order | Label | Notes |
|---|---|---|
| 1 | `EMAIL` | — |
| 2 | `IPV6` | — |
| 3 | `IPV4` | octet-verified 0–255 |
| 4 | `URL` | — |
| 5 | `SSN` | keyword-gated: requires `ssn`, `social security`, `tax id`, or `taxpayer` within ±80 chars |
| 6 | `PHONE` | requires a separator or `+` prefix; 7–15 digits |
| 7 | `CREDIT_CARD` | separator form, Luhn-validated |
| 8 | `CREDIT_CARD` | bare 16-digit, Luhn-validated |

Each match becomes `[LABEL_N]`. Real values live in a per-tab, per-workspace map in `sessionStorage` (cap 4096), cleared on tab close, `/clear`, or `/reset`.

### The Dictionary RAG

- **`SEED_DICTIONARY`** — core English nouns with short definitions.
- **`SYNONYMS` / `ANTONYMS`** — bidirectional.
- **`HYPERNYMS`** — category map (fly → animal, physics → science, cylinder → shape).
- **`EXAMPLES`** — example sentences.
- **`POS_VARIANTS`** — morphological variants (run/running/ran, child/children, man/men).
- **`MULTIWORD_PHRASES`** — 50+ phrases detected before tokenization.
- **`FREQUENCY`** — frequency band 1/2/3.
- **`STATIC_RELATIONS`** — 40+ canonical edges.

Installed into the vector store as `role: 'seed'` docs at boot and versioned.

### OSINT Mode & The Guard Rail

Two independent gates on network access:

1. **Guard Rail (`Guard.on`)** — when on (default), fan-out only if `OsintMode.active`.
2. **OSINT Mode (`OsintMode.active`)** — entered via `/osint <topic>` or the `[ OSINT ]` header button. Has a session ID, topic, and optional case name.

Even when both gates are open: `file://` skips OSINT; `networkOffline` (kill switch) blocks it; the target host must be on the ~60-host allowlist.

### The JEV Event Log

**JSON Event** log. Every mutation writes an event with `schema`, `id`, `ts`, `kind`, `target`, `source`, `actor`, `anchors`, `payload`, `delta`, `prev`, `next`, `checksum`.

Kinds: `insert`, `update`, `delete`, `freeze`, `tag`, `untag`, `repair`, `import`, `clear`, `verify`, `pin`, `unpin`, `trash`, `restore`, `merge`, `case`, `purge`.

Events form a linked list. Capped at 20 000. Query with `/jev`.

### The Vector Store

`CADKernel` — 384-dimension hash embedding. Roles: `seed`, `meta`, `anchor-seed`, `user-noun`, `agent`, `doc`. Cap 3000 docs. Persisted in IndexedDB.

### The Knowledge Graph

`GraphStore` — max 200 nodes / 1200 edges. Node kinds: `anchor` (white), `token` (grey), `user` (bright white), `agent` (light grey), `pii` (red), `doc` (green). Anchor positions computed once via a Fibonacci sphere. Physics: repulsion, spring, gravity, damping. Persisted in IndexedDB.

### Chain Context

Keeps a topic loaded so subsequent turns inherit its nouns. Activated by `[ EXPAND ]` on any agent reply. Cleared by `[ MINIMIZE ]`, `Esc`, or `[ END CHAIN ]`.

### Workspaces

A named scope. Each workspace has its own IndexedDB databases, rehydrate map, watchlist, graph, vectors, RDB entries, and JEV events. Switch via the header select or `#ws=<name>` in the URL hash.

---

## Command Reference

Every command starts with `/`.

### Core
| Command | Description |
|---|---|
| `/help` | Full command reference |
| `/lint` | Render the boot lint report |
| `/lint json` | Same, as JSON |
| `/settings` | Open settings |
| `/library` | Switch to Library tab |
| `/integrity` | Switch to Integrity tab |

### Write
| Command | Description |
|---|---|
| `/add <text or JSON>` | Insert one entry |
| `/import <RAMPART-DB-JSON>` | Strict envelope import |
| `/convert <json>` | Universal JSON walker |
| `/forget <id or last>` | Soft-delete |
| `/undo` | Undo last write |

### Read
| Command | Description |
|---|---|
| `/jev [q\|log\|where\|by\|validate\|repair\|schema]` | Query JEV |
| `/find <natural language>` | Semantic search |
| `/db` | DB stats |
| `/db search <q>` | Text search |
| `/db clear` | Wipe the RDB |
| `/export rampart\|full\|md\|csv` | Export |

### Library
| Command | Description |
|---|---|
| `/verify <id> <status>` | Set verification status |
| `/pin <id>` / `/unpin <id>` | Pin / unpin |
| `/entities` | Top entities + tags |

### OSINT
| Command | Description |
|---|---|
| `/osint <topic>` | Live research pass |
| `/osint off` / `purge` / `commit` | Session control |
| `/case <name>` | Set active case |
| `/case list` / `clear` | Case control |

### Network
| Command | Description |
|---|---|
| `/guard on\|off` | Toggle the OSINT-only guard rail |
| `/remote on\|off` | Remote assistant opt-in |
| `/network` | Allowlist size + last 10 requests |
| `/offline on\|off` | Kill switch |

### Settings
| Command | Description |
|---|---|
| `/theme dark\|light` | Theme |
| `/density compact\|comfortable` | Density |
| `/mode prefer-local\|prefer-remote\|local-only` | Model mode |

### Session
| Command | Description |
|---|---|
| `/clear` | Wipe every store |
| `/reset` | Wipe and re-seed |
| `/sources` | List OSINT endpoints |
| `/rampart` | Rampart engine status |
| `/pii` | Rehydrate map report (masked) |
| `/anchors` | Anchor rail stats |
| `/dictionary` | Dictionary stats |
| `/schema` | RAMPART-DB-JSON skeleton |
| `/nouns` | Last turn's anchors + payload |
| `/watchlist [list\|add\|remove\|run]` | Watchlist |

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `Ctrl+K` / `Cmd+K` | Command palette |
| `Ctrl+,` / `Cmd+,` | Settings |
| `Esc` | Close modal → palette → settings → clear chain |
| `/` (outside input) | Focus chat input |
| `Enter` (in input) | Send |
| `Shift+Enter` (in input) | Newline |
| `ArrowUp` (empty input) | Recall last message |
| `Tab` (in modal/palette/settings) | Cycle focusables |
| `ArrowUp`/`ArrowDown` (in palette) | Move active row |
| `Enter` (in palette) | Run active command |

---

## Data Formats

### RAMPART-DB-JSON 1.1

Media type `application/vnd.webxos.rampartdb+json`. Structure:

```json
{
  "$schema": "https://webxos.dev/schemas/rampartdb/1.1.json",
  "rampartdb": { "version": "1.1", "generator": "...", "workspace": "...", "counts": { ... } },
  "entries": [ { "schema": "rampartdb/entry/1.0", "id": "...", "source": "user", "title": "...", "content": "...", "checksum": "sha256:...", "vec": null, "meta": { ... } } ],
  "events": [],
  "graph": { "nodes": [], "edges": [] }
}
```

Validation: envelope must have `rampartdb` or `entries`; version must start with `1.`; each entry must have `id`, `source`, `ts`, `schema`; checksum is recomputed from `title + '\n' + content`.

### Universal JSON

Any other JSON goes through `universalJSONToEntries`, a depth-6 walker that looks for title keys (`title`, `name`, `heading`, `subject`, `label`, `topic`, `key`, `slug`, `id`, `type`, `kind`) and content keys (`content`, `text`, `body`, `description`, `summary`, `abstract`, `value`, `fact`, `note`, `message`, `data`, `payload`, `answer`, `explanation`), falling back to leaf flattening. Cap 5000 entries.

### CSV / TSV

Header row detected. Columns `title`/`name`, `content`/`text`/`body`/`description`, `tags`. Tags split on `;`, `,`, `|`.

### Markdown

Split on ATX headings (`# ...`). Each heading starts a new entry.

### Bookmarks HTML

Extracts `<a href="...">title</a>` pairs.

---

## OSINT Sources

Only these hosts can ever be reached.

**Core knowledge** — `en.wikipedia.org`, `www.wikidata.org`, `api.duckduckgo.com`

**Dictionary** — `api.dictionaryapi.dev`

**Academic** — `export.arxiv.org`, `api.openalex.org`, `api.semanticscholar.org`, `eutils.ncbi.nlm.nih.gov`

**Developer** — `api.stackexchange.com`, `api.github.com`, `hn.algolia.com`

**Reference** — `openlibrary.org`

**Data** — `restcountries.com`, `api.open-meteo.com` + `geocoding-api.open-meteo.com`, `api.frankfurter.app`

**CORS proxies** — `api.allorigins.win`, `corsproxy.io`

Scoring blends token overlap (0.55), source weight (0.20), text depth (0.15), trust (0.10), type bonus (0.12), hypernym bonus (0.08). Items below `minScore` (0.28) are dropped.

---

## Security & Privacy

**What leaves the browser:** nothing — unless you run `/osint <topic>`. When you do: your topic is PII-redacted, sent to the top-N allowlisted sources, responses are PII-redacted before storage, then summarized and inserted into your local RDB.

**What stays local:** original inputs (in the sessionStorage rehydrate map), RDB (IndexedDB), JEV (IndexedDB), graph (IndexedDB), vectors (IndexedDB), config (localStorage), workspaces (localStorage), watchlist (localStorage).

**Kill switch:** `/offline on` blocks every network call.

**Guard rail:** `/guard on` (default) blocks fan-out from non-OSINT turns.

**CSP:** meta CSP restricts `default-src`, `script-src`, `style-src`, `connect-src`, `object-src`, `base-uri`, `form-action`. No CDN at startup. For HTTP(S) deployments, also send `Content-Security-Policy: frame-ancestors 'none'` as a response header.

---

## Boot Lint

Runs 25+ subsystem checks at startup and prints them into the chat. Levels: `fail` (must pass) and `warn` (expected to fail in some environments). Re-render with `/lint`, get JSON with `/lint json`.

---

## The 14 Applied Fixes

- **E01** — IPv4 pattern runs before phone
- **E02** — Phone requires a separator or `+` prefix
- **E03** — Credit-card regex requires Luhn validation
- **E04** — LOC gazetteer duplicates removed
- **E05** — `STOPWORDS` etc. declared before use
- **E06** — Jaccard guards `0/0`
- **E07** — Storage badge labelled `~kb`
- **E08** — No `eval`; direct function map
- **E09** — `frame-ancestors` documented as response-header-only
- **E10** — OSINT early-returns on `file://`
- **E11** — Physics skipped when `document.hidden`
- **E12** — Quiet path uses `try/finally`
- **E13** — SSN requires a nearby keyword
- **E14** — Wikipedia summary URL uses underscores

---

## Credits & License

MIT OPENSOURCED BY: **webXOS claims no responsibility for how this app is used, this app is intended for private and local data only. Not intended for unauthorized use for any DB**

**© webXOS 2026 — RAMPARTDB v3.0.7**
