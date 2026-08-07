#!/usr/bin/env python3
"""Generate the arXiv-style PDF report: Dynamic Programming with LLMs (2026). ~10 pages."""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Preformatted, HRFlowable
)

OUTPUT = Path("/home/workdir/artifacts/dp_llms_2026/Dynamic_Programming_with_LLMs_2026.pdf")

ARXIV_BLUE = HexColor("#1a3a5c")
MED_GRAY = HexColor("#666666")
CODE_BG = HexColor("#f0f0f0")
LIGHT_BLUE = HexColor("#e8eef5")

def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleMain", fontName="Times-Bold", fontSize=15,
        leading=19, alignment=TA_CENTER, spaceAfter=6, textColor=ARXIV_BLUE))
    styles.add(ParagraphStyle(name="Subtitle", fontName="Times-Roman", fontSize=10,
        leading=13, alignment=TA_CENTER, spaceAfter=3, textColor=MED_GRAY))
    styles.add(ParagraphStyle(name="Author", fontName="Times-Roman", fontSize=10,
        leading=13, alignment=TA_CENTER, spaceAfter=1))
    styles.add(ParagraphStyle(name="AbstractHead", fontName="Times-Bold", fontSize=10,
        leading=13, alignment=TA_CENTER, spaceBefore=10, spaceAfter=4))
    styles.add(ParagraphStyle(name="AbstractBody", fontName="Times-Roman", fontSize=9,
        leading=12, alignment=TA_JUSTIFY, leftIndent=18, rightIndent=18, spaceAfter=8))
    styles.add(ParagraphStyle(name="Section", fontName="Times-Bold", fontSize=11,
        leading=14, spaceBefore=12, spaceAfter=5, textColor=ARXIV_BLUE))
    styles.add(ParagraphStyle(name="Subsection", fontName="Times-Bold", fontSize=10,
        leading=13, spaceBefore=8, spaceAfter=3))
    styles.add(ParagraphStyle(name="BodyJust", fontName="Times-Roman", fontSize=9.5,
        leading=12.5, alignment=TA_JUSTIFY, spaceAfter=5))
    styles.add(ParagraphStyle(name="CodeBlock", fontName="Courier", fontSize=7.5,
        leading=9.5, leftIndent=8, rightIndent=8, spaceBefore=3, spaceAfter=3, backColor=CODE_BG))
    styles.add(ParagraphStyle(name="BulletItem", fontName="Times-Roman", fontSize=9.5,
        leading=12.5, leftIndent=12, spaceAfter=2))
    styles.add(ParagraphStyle(name="Footer", fontName="Times-Roman", fontSize=8,
        leading=10, alignment=TA_CENTER, textColor=MED_GRAY))
    styles.add(ParagraphStyle(name="Caption", fontName="Times-Italic", fontSize=8,
        leading=10, alignment=TA_CENTER, spaceBefore=2, spaceAfter=6))
    styles.add(ParagraphStyle(name="TableCell", fontName="Times-Roman", fontSize=8, leading=10.5))
    styles.add(ParagraphStyle(name="TableHeader", fontName="Times-Bold", fontSize=8, leading=10.5))
    return styles


def add_page_number(canvas, doc):
    canvas.saveState()
    page_num = canvas.getPageNumber()
    canvas.setFont("Times-Roman", 8)
    canvas.setFillColor(MED_GRAY)
    canvas.drawCentredString(letter[0] / 2, 0.5 * inch, f"— {page_num} —")
    if page_num > 1:
        canvas.setStrokeColor(HexColor("#cccccc"))
        canvas.setLineWidth(0.4)
        canvas.line(0.85 * inch, letter[1] - 0.55 * inch, letter[0] - 0.85 * inch, letter[1] - 0.55 * inch)
        canvas.setFont("Times-Roman", 7.5)
        canvas.drawString(0.85 * inch, letter[1] - 0.5 * inch, "Dynamic Programming with LLMs (2026)")
        canvas.drawRightString(letter[0] - 0.85 * inch, letter[1] - 0.5 * inch, "arXiv-style technical report")
    canvas.restoreState()


def P(text, style):
    return Paragraph(text, style)


def build():
    styles = make_styles()
    S = styles
    story = []

    # TITLE
    story.append(Spacer(1, 0.45 * inch))
    story.append(P("Dynamic Programming with LLMs:<br/>Megalithic Orchestration for the Agentic Frontier", S["TitleMain"]))
    story.append(P("A Practical Guide to Self-Contained HTML, Python Bootstraps,<br/>Bash Agents, and PowerShell Scaffolds", S["Subtitle"]))
    story.append(Spacer(1, 6))
    story.append(P("Grok (xAI)", S["Author"]))
    story.append(P("August 2026", S["Author"]))
    story.append(P("Technical Report — Companion materials included in the distribution package", S["Subtitle"]))

    # ABSTRACT
    story.append(P("Abstract", S["AbstractHead"]))
    story.append(P(
        "We reframe classic Bellman-style dynamic programming as a design principle for large, "
        "self-contained “megalith” artifacts that large language models (LLMs) and autonomous "
        "agents can generate, mutate, and execute end-to-end. Four concrete orchestration formats "
        "are examined in depth: (1) self-contained HTML documents optimized for rapid interactive "
        "prototypes, (2) singular Python bootstrap scripts that install dependencies, generate "
        "multi-file projects, and auto-launch them (MSPYB v3), (3) monolithic Bash agents that "
        "embed chat, reflection, safety firewalls, hardware automation, and background daemons "
        "(UNIXAI-style), and (4) equivalent PowerShell patterns for Windows and cross-platform "
        "environments. Each format is treated as a living value function whose optimal substructure "
        "and overlapping subproblems map directly onto the generation–launch–observe loop performed "
        "by an agent. The result is a practical methodology for turning any idea into an immediately "
        "executable, offline-capable, and LLM-editable scaffold. We supply a complete companion "
        "package containing the present report, detailed format guides, and minimal working examples.",
        S["AbstractBody"]))
    story.append(P(
        "<b>Keywords:</b> dynamic programming, large language models, agentic systems, megalithic "
        "scripts, self-bootstrapping, HTML prototypes, Bash agents, PowerShell, software scaffolding, "
        "Bellman optimality, template-driven generation",
        S["BodyJust"]))

    # 1. INTRODUCTION
    story.append(P("1. Introduction", S["Section"]))
    story.append(P(
        "The dominant workflow for LLM-assisted software construction still treats the model as a "
        "sophisticated autocomplete engine: the human or agent requests a file, receives a fragment, "
        "pastes it into a larger tree, and repeats. This produces brittle, partially-specified systems "
        "whose coherence is maintained only by the operator’s working memory and by ad-hoc test suites. "
        "In contrast, the systems that have proved most robust under autonomous iteration—large "
        "self-contained Bash agents, single-file HTML prototypes, and Python “bootstrap” generators—"
        "share a common architectural property: they are <i>megalithic</i>. A single source of truth "
        "declares the entire system, generates any necessary secondary files, and can launch itself.",
        S["BodyJust"]))
    story.append(P(
        "We argue that this property is not merely engineering convenience; it is the practical "
        "expression of the Bellman principle of optimality applied to software construction. Optimal "
        "substructure appears as reusable templates, shared logging modules, and namespaced capability "
        "sections. Overlapping subproblems appear as the repeated cycle of environment detection, "
        "generation, launch, and observation that an agent performs. By making the megalith itself the "
        "object that the LLM edits, the agent improves a <i>policy</i> rather than a collection of "
        "disconnected leaves.",
        S["BodyJust"]))
    story.append(P(
        "The remainder of this report is organized as follows. Section 2 recalls the relevant ideas "
        "from dynamic programming and maps them onto agentic workflows. Section 3 presents four "
        "concrete orchestration formats with design rules and implementation notes drawn from production "
        "reference implementations. Section 4 discusses the design invariants that make a megalith "
        "agent-friendly. Section 5 offers a comparative evaluation, composition patterns, and practical "
        "recommendations. Section 6 outlines open challenges and future directions. Section 7 concludes. "
        "Two short appendices describe the companion package layout and minimal invocation examples.",
        S["BodyJust"]))

    # 2. DP FOR AGENTS
    story.append(P("2. Dynamic Programming for Agents", S["Section"]))
    story.append(P("2.1 Bellman Recurrence in the Software Domain", S["Subsection"]))
    story.append(P(
        "Recall the Bellman equation for a deterministic finite-horizon problem:",
        S["BodyJust"]))
    story.append(P(
        "<i>V</i>(s) = max<sub>a</sub> [ <i>R</i>(s, a) + <i>V</i>(s′(s, a)) ]",
        S["BodyJust"]))
    story.append(P(
        "In the agentic setting we interpret the state <i>s</i> as the current specification of a "
        "software system (the megalith text plus any generated artifacts on disk), the action <i>a</i> "
        "as an edit performed by the LLM or a human operator, the immediate reward <i>R</i> as a "
        "scalar or structured measure of successful launch, test passage, or user feedback, and the "
        "successor state <i>s′</i> as the project after regeneration. Because the megalith is a complete, "
        "executable specification, the value of any partial improvement can be evaluated by simply "
        "re-running the single file—exactly the “backup” operation of classical dynamic programming.",
        S["BodyJust"]))
    story.append(P(
        "This mapping is more than metaphorical. When an agent proposes a change to a template variable, "
        "a firewall rule, or a launch command, the subsequent execution of the megalith produces a new "
        "observable state whose quality can be scored. The agent then selects the edit that maximises "
        "expected future value. In practice the “max” is performed by the LLM’s own ranking of candidate "
        "edits, guided by the recovery hints and structured logs that the megalith emits.",
        S["BodyJust"]))

    story.append(P("2.2 Optimal Substructure and Overlapping Subproblems", S["Subsection"]))
    story.append(P(
        "A well-designed megalith exhibits optimal substructure: the best way to solve the “install "
        "dependencies” subproblem is independent of the particular application being generated, yet the "
        "result is reused by every later stage. Likewise, logging, templating, environment detection, "
        "and auto-launch are solved once and composed. Overlapping subproblems arise whenever an agent "
        "iterates: each new generation re-solves the same environment-detection and logging-setup tasks; "
        "memoization (caching the detected environment, idempotent package installation) prevents "
        "redundant work and keeps the generation loop fast enough for interactive use.",
        S["BodyJust"]))
    story.append(P(
        "The same principle appears inside long-running Bash or PowerShell agents. Message-history "
        "trimming, model-list caching, and firewall-pattern matching are classic overlapping "
        "subproblems; solving them once and storing the results in process state yields the efficiency "
        "gains that make a multi-module agent practical inside a single process.",
        S["BodyJust"]))

    story.append(P("2.3 Why Fragmented Codebases Defeat Agents", S["Subsection"]))
    story.append(P(
        "When a system is spread across dozens of small files, an agent must maintain a mental model "
        "of inter-file consistency—import graphs, configuration drift, version skew, and hidden side "
        "effects. Context-window limits, missing imports, and divergent configuration quickly produce "
        "incoherent states that are expensive to diagnose. A megalith collapses that surface area: the "
        "entire policy lives in one context window, and regeneration restores global consistency by "
        "construction. The agent’s job therefore shrinks from “keep the whole tree coherent” to "
        "“improve the single specification.”",
        S["BodyJust"]))

    story.append(P("2.4 The Generation–Launch–Observe Loop as Value Iteration", S["Subsection"]))
    story.append(P(
        "Value iteration in classical DP repeatedly applies the Bellman backup until the value "
        "function converges. The agentic analogue is the loop:",
        S["BodyJust"]))
    story.append(Preformatted(
        "while not satisfied:\n"
        "    edit megalith          # improve policy\n"
        "    run megalith           # generate + launch\n"
        "    observe logs / tests   # estimate value\n"
        "    decide next edit",
        S["CodeBlock"]))
    story.append(P(
        "Because each iteration is cheap (a single process start) and because the megalith emits "
        "structured recovery hints, the loop can be driven by an LLM with minimal human intervention. "
        "This is the practical realisation of dynamic programming for software construction in 2026.",
        S["BodyJust"]))

    # 3. FOUR FORMATS
    story.append(P("3. Four Orchestration Formats", S["Section"]))
    story.append(P(
        "We now examine four concrete formats that embody the principles above. Each has been "
        "stress-tested in production-scale reference implementations and distilled into a set of "
        "non-negotiable rules that an agent (or a human author) can follow.",
        S["BodyJust"]))

    story.append(P("3.1 Self-Contained HTML (MHTML-Style Prototypes)", S["Subsection"]))
    story.append(P(
        "The HTML megalith is a single file containing every style, script, and asset required for a "
        "complete interactive experience. All CSS lives inside <font face='Courier'>&lt;style&gt;</font> "
        "tags; all JavaScript inside <font face='Courier'>&lt;script&gt;</font> tags; images, icons, and "
        "fonts are embedded as base64 data URIs. The file opens offline in any modern browser and "
        "requires no build step, no package manager, and no network after the initial save.",
        S["BodyJust"]))
    story.append(P(
        "From the dynamic-programming viewpoint the HTML document is the value function of a "
        "user-interface idea. An agent can emit the entire document in one generation step, then refine "
        "individual components (CSS custom properties for theming, JavaScript state machines, content "
        "blocks) while the global layout remains coherent. Because there is no build step, the evaluation "
        "of any change is instantaneous: reopen the file or refresh the browser tab.",
        S["BodyJust"]))
    story.append(P(
        "Quality standards for “megalithic” generation include realistic content, multiple interaction "
        "states (loading, empty, error, success), responsive design (mobile-first Flexbox/Grid), dark/light "
        "theming via CSS custom properties, and subtle micro-interactions. The goal is a production-ready "
        "prototype that a stakeholder can click through, not a wireframe. Semantic HTML and accessibility "
        "attributes are expected; the resulting document should be usable with keyboard navigation and "
        "screen readers.",
        S["BodyJust"]))
    story.append(P(
        "Typical agent response structure: a one-sentence summary of what was built, followed by the "
        "complete HTML inside a single markdown code fence, followed by a short “how to use” note "
        "(save as .html / open in browser). The code itself must be immediately usable without further "
        "assembly.",
        S["BodyJust"]))

    story.append(P("3.2 Singular Python Bootstraps (MSPYB v3)", S["Subsection"]))
    story.append(P(
        "An MSPYB bootstrap is a single Python file that (1) detects the host environment (OS family, "
        "architecture, Python version, presence of Docker), (2) installs pinned dependencies "
        "idempotently, (3) generates a complete multi-file project via templated "
        "<font face='Courier'>write_file</font> calls that use <font face='Courier'>string.Template"
        ".safe_substitute</font>, and (4) optionally launches the resulting application. The five "
        "required capabilities are:",
        S["BodyJust"]))
    story.append(P("• <b>Error Logging</b> — Bootstrap logger available before any INSTALLS; a shared "
        "<font face='Courier'>logging_setup.py</font> is generated for every service.", S["BulletItem"]))
    story.append(P("• <b>Templating</b> — Single <font face='Courier'>TEMPLATE_VARS</font> dictionary "
        "near the top; all file content may contain <font face='Courier'>$VAR</font> placeholders.", S["BulletItem"]))
    story.append(P("• <b>Versioning</b> — <font face='Courier'>MSPYB_VERSION</font> constant plus a "
        "generated <font face='Courier'>VERSION</font> file; header documents known fixes.", S["BulletItem"]))
    story.append(P("• <b>INSTALLS</b> — OS-aware, pinned packages, honour <font face='Courier'>--skip-installs</font>.", S["BulletItem"]))
    story.append(P("• <b>Auto-Start</b> — After generation, spawn the app respecting "
        "<font face='Courier'>AUTO_LAUNCH</font> and <font face='Courier'>--no-launch</font>; log "
        "recovery hints on failure.", S["BulletItem"]))

    header = [P("<b>Capability</b>", S["TableHeader"]), P("<b>DP role</b>", S["TableHeader"]),
              P("<b>Agent benefit</b>", S["TableHeader"])]
    rows = [header,
        [P("Logging", S["TableCell"]), P("Immediate reward signal", S["TableCell"]),
         P("Diagnose launch failures", S["TableCell"])],
        [P("Templating", S["TableCell"]), P("Shared substructure", S["TableCell"]),
         P("One change updates all files", S["TableCell"])],
        [P("Versioning", S["TableCell"]), P("State identification", S["TableCell"]),
         P("Track policy iterations", S["TableCell"])],
        [P("INSTALLS", S["TableCell"]), P("Environment transition", S["TableCell"]),
         P("Idempotent, recoverable", S["TableCell"])],
        [P("Auto-Start", S["TableCell"]), P("Terminal evaluation", S["TableCell"]),
         P("Full-circle feedback", S["TableCell"])],
    ]
    t = Table(rows, colWidths=[1.2*inch, 2.0*inch, 2.6*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BLUE),
        ("GRID", (0, 0), (-1, -1), 0.35, HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(Spacer(1, 4))
    story.append(t)
    story.append(P("Table 1: MSPYB capabilities mapped onto dynamic-programming concepts.", S["Caption"]))

    story.append(P(
        "The bootstrap is both the policy and the value function. Changing a port number or a launch "
        "command in <font face='Courier'>TEMPLATE_VARS</font> propagates to every generated file and "
        "to the launch behaviour. An agent therefore improves the system by editing only the bootstrap; "
        "re-execution reconstitutes a clean, consistent tree. The v3 Auto-Start contract closes the last "
        "gap between “generated” and “running.” A critical implementation detail: multi-line file contents "
        "inside <font face='Courier'>write_file</font> should be delimited with triple single-quotes so "
        "that inner double-quoted docstrings do not terminate the outer string prematurely.",
        S["BodyJust"]))

    story.append(P("3.3 Monolithic Bash Agents (UNIXAI-Style)", S["Subsection"]))
    story.append(P(
        "A UNIXAI-style megalith is a single Bash script that embeds a complete conversational agent "
        "together with safety, automation, and research modules. The reference implementation (UNIXAI "
        "Full Suite) integrates chat and reflection loops, agent command extraction via structured "
        "<font face='Courier'>&lt;cmd&gt;...&lt;/cmd&gt;</font> tags, a dangerous-command firewall that "
        "consults the LLM for side-effect analysis, hardware automators (CPU governor / fan control), "
        "a compiler-fix loop, an interactive config mutator, a Whisper transcription watchdog, a "
        "web-scraping research agent, a named-pipe IPC listener, and a log-triage daemon. All modules "
        "share the same process state and are dispatched by a central command parser.",
        S["BodyJust"]))
    story.append(P(
        "Design principles that make such a script tractable for an agent include: self-containment "
        "with graceful fallbacks (jq → python3 → awk for JSON), namespaced modules with clear variable "
        "prefixes, rigorous trap and cleanup hygiene, bounded message history with character budget, "
        "tool tags as the sole action interface, daemon orchestration via an associative array of PIDs, "
        "and a safety-first firewall that intercepts risky patterns before execution and demands "
        "interactive confirmation.",
        S["BodyJust"]))
    story.append(P(
        "The script itself is the value function of the agentic session. Sub-problems (firewall rule "
        "matching, scrape turns, compiler iterations) are solved optimally inside the running process; "
        "their results are memoized in associative arrays and temporary files. An LLM can improve any "
        "module by editing the corresponding section of the single file; the next execution reconstitutes "
        "the whole system. This is optimal substructure at the scale of a multi-thousand-line interactive "
        "agent.",
        S["BodyJust"]))

    story.append(P("3.4 PowerShell Megaliths", S["Subsection"]))
    story.append(P(
        "The same principles translate directly to PowerShell 7+ (<font face='Courier'>pwsh</font>). "
        "A single <font face='Courier'>.ps1</font> file detects the platform (Windows / Linux / macOS), "
        "installs modules and packages idempotently (Install-Module, pip, winget, apt, brew according "
        "to the detected environment), generates a project tree via templated file writes, and "
        "auto-launches the result. Interactive agents mirror the Bash pattern: a "
        "<font face='Courier'>switch -Regex</font> dispatcher for slash commands, bounded conversation "
        "history stored as an ArrayList of hashtables, structured tool tags parsed from model replies, "
        "and background jobs via Start-Job or Start-Process for long-running daemons.",
        S["BodyJust"]))
    story.append(P(
        "Cross-platform considerations—executable bits on Unix, package-manager differences, Docker "
        "availability—are encoded once in the environment-detection block and reused by every subsequent "
        "stage. Secrets never appear in the megalith; only a generated "
        "<font face='Courier'>.env.example</font> is written, which the operator copies once. Structured "
        "JSON-line logging plus coloured console output give the agent the same recovery-hint surface "
        "that MSPYB and UNIXAI provide.",
        S["BodyJust"]))
    story.append(P(
        "From the dynamic-programming viewpoint the single <font face='Courier'>.ps1</font> is again "
        "both policy and value function. An LLM edits the bootstrap; re-execution produces a fresh, "
        "consistent project state. Overlapping subproblems (environment detection, logging setup, "
        "launch recovery) are solved once and reused across every generation.",
        S["BodyJust"]))

    # 4. INVARIANTS
    story.append(P("4. Design Invariants for Agent-Friendly Megaliths", S["Section"]))
    story.append(P(
        "Across all four formats a small set of invariants makes the artifact tractable for an "
        "autonomous agent. Violating any of them re-introduces the coherence problems that megaliths "
        "were invented to solve.",
        S["BodyJust"]))
    story.append(P("• <b>Singular source of truth.</b> All policy lives in one file. Secondary artifacts "
        "are generated, never hand-edited. The agent’s context window therefore contains the complete "
        "specification.", S["BulletItem"]))
    story.append(P("• <b>Idempotence.</b> Re-running the megalith produces a clean, consistent state; "
        "partial previous runs do not accumulate corruption. This is the software analogue of a pure "
        "Bellman backup.", S["BulletItem"]))
    story.append(P("• <b>Offline capability.</b> After the initial dependency installation the system "
        "functions without network access. This is critical for reproducible evaluation and for "
        "air-gapped or latency-sensitive environments.", S["BulletItem"]))
    story.append(P("• <b>Structured recovery hints.</b> Launch or generation failures emit "
        "machine-readable messages (and, ideally, a one-line recovery suggestion) that an agent can "
        "parse and act upon without human translation.", S["BulletItem"]))
    story.append(P("• <b>Explicit extension points.</b> Template variables, namespaced modules, and "
        "clearly delimited sections allow surgical improvement. An agent should never have to rewrite "
        "an entire module to change a port number.", S["BulletItem"]))
    story.append(P("• <b>Versioned policy.</b> A version constant and a generated VERSION file let the "
        "agent (and later human auditors) track which policy produced which state.", S["BulletItem"]))
    story.append(P("• <b>No secrets in the megalith.</b> Credentials live only in generated "
        "<font face='Courier'>.env.example</font> files that the operator copies once. The megalith "
        "itself remains safe to commit, share, and regenerate.", S["BulletItem"]))
    story.append(P(
        "These invariants turn the megalith into a reliable Bellman backup operator: the agent can "
        "always evaluate the current policy by executing a single command, observe the outcome, and "
        "improve the policy in place. The loop is closed, measurable, and amenable to automated "
        "search.",
        S["BodyJust"]))

    # 5. COMPARATIVE
    story.append(P("5. Comparative Evaluation and Recommendations", S["Section"]))
    story.append(P("5.1 When to Choose Which Format", S["Subsection"]))
    story.append(P(
        "HTML megaliths excel at interactive front-ends and rapid visual feedback; they are the "
        "natural choice when the deliverable is a dashboard, a multi-step form, an interactive "
        "documentation site, or any prototype UI that must be clicked through by stakeholders. "
        "Python bootstraps are optimal when the target is a multi-file service, library, or data "
        "pipeline that must be launched, tested, and potentially containerised. Bash megaliths "
        "dominate when the environment is a Unix shell and the agent must interleave conversation "
        "with system administration, log analysis, or hardware control. PowerShell fills the "
        "analogous role on Windows and, via PowerShell Core, provides a clean cross-platform "
        "alternative that shares the same design language.",
        S["BodyJust"]))

    h2 = [P("<b>Format</b>", S["TableHeader"]), P("<b>Primary strength</b>", S["TableHeader"]),
          P("<b>Typical deliverable</b>", S["TableHeader"]), P("<b>Eval cost</b>", S["TableHeader"])]
    r2 = [h2,
        [P("HTML / MHTML", S["TableCell"]), P("Instant visual feedback", S["TableCell"]),
         P("UI prototype, dashboard", S["TableCell"]), P("Browser refresh", S["TableCell"])],
        [P("Python MSPYB", S["TableCell"]), P("Multi-file generation + launch", S["TableCell"]),
         P("API, CLI, data pipeline", S["TableCell"]), P("Process start", S["TableCell"])],
        [P("Bash UNIXAI", S["TableCell"]), P("Interactive system agent", S["TableCell"]),
         P("Shell assistant, daemons", S["TableCell"]), P("In-process", S["TableCell"])],
        [P("PowerShell", S["TableCell"]), P("Cross-platform admin + gen", S["TableCell"]),
         P("Windows services, hybrid", S["TableCell"]), P("Process start", S["TableCell"])],
    ]
    t2 = Table(r2, colWidths=[1.15*inch, 1.7*inch, 1.7*inch, 1.25*inch])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BLUE),
        ("GRID", (0, 0), (-1, -1), 0.35, HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(Spacer(1, 4))
    story.append(t2)
    story.append(P("Table 2: Format selection guide.", S["Caption"]))

    story.append(P("5.2 Composition Patterns", S["Subsection"]))
    story.append(P(
        "In practice an agentic workflow often composes several formats. A common pattern is:",
        S["BodyJust"]))
    story.append(P("1. A Python MSPYB bootstrap generates a FastAPI backend, a Docker Compose file, "
        "and a self-contained HTML front-end that talks to the API.", S["BulletItem"]))
    story.append(P("2. The same bootstrap’s Auto-Start launches the backend; the HTML file is opened "
        "in a browser for visual verification.", S["BulletItem"]))
    story.append(P("3. A Bash (or PowerShell) agent later monitors the running services, performs log "
        "triage, and can mutate configuration through the config-mutator module.", S["BulletItem"]))
    story.append(P(
        "Because all three artifacts share the same design language—template variables, structured "
        "logging, auto-launch, recovery hints—the composition remains coherent. An improvement to the "
        "shared logging format, for example, propagates automatically the next time the bootstrap is "
        "re-run.",
        S["BodyJust"]))

    story.append(P("5.3 Practical Recommendations", S["Subsection"]))
    story.append(P(
        "We recommend that any new agent-facing deliverable begin as a megalith of the appropriate "
        "format. Only after the single-file version is stable, fully instrumented, and has survived "
        "several generation–launch–observe cycles should the system be optionally expanded into a "
        "conventional multi-file repository—itself generated by the megalith. Premature fragmentation "
        "re-introduces the coherence tax that the megalith was designed to eliminate.",
        S["BodyJust"]))
    story.append(P(
        "When an existing multi-file codebase must be brought under agent control, the reverse "
        "transformation is often valuable: write a bootstrap that can regenerate the tree from a "
        "single specification, then treat the bootstrap as the new source of truth. Subsequent "
        "improvements are made to the bootstrap; the leaf files become disposable build products.",
        S["BodyJust"]))

    # 6. FUTURE
    story.append(P("6. Open Challenges and Future Directions", S["Section"]))
    story.append(P(
        "Several research and engineering challenges remain. First, the design of reward functions "
        "that an autonomous agent can optimise remains under-specified; current practice relies on "
        "human inspection of logs and manual acceptance of generated UIs. Automated visual regression, "
        "API contract testing, and differential performance measurement would close the loop more "
        "tightly. Second, the tension between megalith size and context-window limits will grow as "
        "agents tackle larger systems; hierarchical megaliths (a top-level bootstrap that generates "
        "secondary bootstraps) are a promising direction. Third, formal verification of the invariants "
        "listed in Section 4—especially idempotence and secret-freedom—could be performed by static "
        "analysis before any generation step. Finally, the mapping onto classical DP suggests that "
        "techniques such as prioritized sweeping, eligibility traces, or model-based rollouts may "
        "have direct analogues in agentic software construction; exploring those analogues is left "
        "as future work.",
        S["BodyJust"]))

    # 7. CONCLUSION
    story.append(P("7. Conclusion", S["Section"]))
    story.append(P(
        "Dynamic programming supplies more than a metaphor for agentic software construction; it "
        "supplies a concrete design discipline. By insisting that every large idea be expressed as a "
        "self-contained, self-bootstrapping, self-launching megalith, we give the LLM a well-defined "
        "policy object, a reliable evaluation procedure, and a clear path for iterative improvement. "
        "The four formats examined here—self-contained HTML, Python bootstraps (MSPYB), Bash agents "
        "(UNIXAI-style), and PowerShell modules—demonstrate that the discipline is practical across "
        "the major platforms and languages of 2026. The companion package distributed with this report "
        "contains detailed guides and minimal working examples so that practitioners can adopt the "
        "pattern immediately.",
        S["BodyJust"]))
    story.append(P(
        "We close with a concise maxim: <i>edit the policy, regenerate the world</i>. When that maxim "
        "is followed, the generation–launch–observe loop becomes a practical value-iteration procedure, "
        "and the agent’s task shrinks to the one problem it is best equipped to solve—improving a "
        "single, coherent specification.",
        S["BodyJust"]))

    # APPENDIX A
    story.append(P("Appendix A — Companion Package Layout", S["Section"]))
    story.append(P("The distribution archive contains the following files and directories:", S["BodyJust"]))
    story.append(P("• <font face='Courier'>Dynamic_Programming_with_LLMs_2026.pdf</font> — this report "
        "(arXiv-style technical report, approximately ten pages).", S["BulletItem"]))
    story.append(P("• <font face='Courier'>README.md</font> — package overview, quick-start instructions, "
        "and citation entry.", S["BulletItem"]))
    story.append(P("• <font face='Courier'>guides/MHTML_Orchestration.md</font> — rules and quality "
        "standards for self-contained HTML prototypes.", S["BulletItem"]))
    story.append(P("• <font face='Courier'>guides/MSPYB_Python_Bootstrap.md</font> — MSPYB v3 philosophy, "
        "required capabilities, and Auto-Start contract.", S["BulletItem"]))
    story.append(P("• <font face='Courier'>guides/UNIXAI_Bash_Megalith.md</font> — anatomy of a production "
        "Bash AI agent and modular design principles.", S["BulletItem"]))
    story.append(P("• <font face='Courier'>guides/PowerShell_Megalith_Guide.md</font> — equivalent patterns "
        "for Windows and cross-platform PowerShell.", S["BulletItem"]))
    story.append(P("• <font face='Courier'>examples/minimal_megalith.sh</font> — minimal UNIXAI-style "
        "skeleton demonstrating the modular command pattern.", S["BulletItem"]))
    story.append(P("• <font face='Courier'>src/bootstrap_example.py</font> — illustrative MSPYB-style "
        "generator that creates a tiny HTTP server project.", S["BulletItem"]))
    story.append(P("• <font face='Courier'>src/generate_report.py</font> — the script that produced the "
        "present PDF (included for reproducibility).", S["BulletItem"]))

    # APPENDIX B
    story.append(P("Appendix B — Minimal Invocation Examples", S["Section"]))
    story.append(P("Python bootstrap (generation only):", S["BodyJust"]))
    story.append(Preformatted("python3 bootstrap.py --no-launch", S["CodeBlock"]))
    story.append(P("Python bootstrap (full circle: install + generate + launch):", S["BodyJust"]))
    story.append(Preformatted("python3 bootstrap.py", S["CodeBlock"]))
    story.append(P("Bash agent (after making executable):", S["BodyJust"]))
    story.append(Preformatted("chmod +x unixai-full.sh && ./unixai-full.sh", S["CodeBlock"]))
    story.append(P("PowerShell bootstrap:", S["BodyJust"]))
    story.append(Preformatted("pwsh -File bootstrap.ps1 -NoLaunch", S["CodeBlock"]))
    story.append(P("Self-contained HTML: save the generated file with a .html extension and open it "
        "in any modern browser (Chrome, Firefox, Safari, Edge). No server is required.", S["BodyJust"]))
    story.append(P(
        "In all cases the first successful run establishes a baseline value; subsequent edits to the "
        "megalith, followed by re-execution, constitute the practical value-iteration loop described "
        "in Section 2.",
        S["BodyJust"]))

    # APPENDIX C
    story.append(P("Appendix C — Selected Implementation Notes", S["Section"]))
    story.append(P(
        "JSON handling in Bash agents should prefer jq when available, fall back to a short Python "
        "one-liner, and only as a last resort use a carefully written awk parser. The same layered "
        "fallback pattern applies to HTML-to-text conversion (lynx → pup → sed). Environment detection "
        "should expose a single dictionary or associative array that later stages consume; never "
        "re-detect the same property in multiple modules. Launch failures must never be treated as "
        "generation failures: the project tree is still valuable even if the process could not start. "
        "Finally, every background daemon should register its PID in a central table and expose "
        "start / stop / status commands so that the agent (or a human) can manage the process graph "
        "without hunting through ps output.",
        S["BodyJust"]))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.4, color=HexColor("#cccccc")))
    story.append(P(
        "End of report. Materials prepared August 2026 for research and educational use. "
        "Adapt freely; attribution appreciated.",
        S["Footer"]))

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.7 * inch,
        title="Dynamic Programming with LLMs: Megalithic Orchestration for the Agentic Frontier",
        author="Grok (xAI)",
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"Wrote {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    build()
