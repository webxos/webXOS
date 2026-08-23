# WEBXOS — Chatbot (version 3)

**A single-file, zero-dependency, Markdown-only personal knowledge base chatbot**  
Train an AI assistant directly in your browser using your own `.md` files. No backend, no API keys, fully offline.

![[CHATBOT BANNER](https://github.com/webxos/webXOS/blob/main/assets/chatbot.jpeg)](https://github.com/webxos/webXOS/blob/main/assets/chatbot.jpeg)  

# UNDER DEVELOPMENT — UPDATED DECEMBER 1, 2025

**Local, private, offline-first AI chatbot with full markdown knowledge-base + sandboxed JavaScript/Python script execution**

*webxos.netlify.app/chatbot*

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
![Pyodide](https://img.shields.io/badge/Powered_by-Pyodide-39FF14)  
![No backend](https://img.shields.io/badge/No_backend-100%25_local-blue)

# WEBXOS Terminal Comprehensive User Guide

## Overview
WEBXOS Terminal v3.0 is a standalone, single-file HTML application simulating a command-line interface (CLI) in the browser. It features a neon-green cyberpunk aesthetic with black background, green text for general output, and yellow for scripts and calculator elements. The app uses pure HTML, CSS, and JavaScript with no external dependencies. Data persistence is handled via browser localStorage for contexts, scripts, and command history. It includes dummy AI responses for non-command inputs, file management for Markdown contexts and executable scripts, a built-in calculator, import/export functionality, and modal-based editors. Designed for local use only; no server or network required. Updated as of December 15, 2025. No Three.js integration; remains 2D canvas-free; editor mindset focuses on potential 3D expansions but current version is flat HTML.

## System Requirements and Setup
- **Browser Compatibility**: Modern browsers like Chrome, Firefox, Edge, or Safari (version 15+ recommended for full CSS support).
- **File Size**: Approximately 50-100 KB (single HTML file).
- **Storage**: Uses localStorage; browser limits apply (typically 5-10 MB per domain). Clear browser data to reset.
- **Installation**:
  1. Download or copy the HTML code to a file named `webxos-terminal.html`.
  2. Open the file directly in your browser (file:// protocol works).
  3. The loading screen will appear with an ASCII art logo, progress bar, and phased status messages (e.g., "LOADING CORE SYSTEMS", "INITIALIZING SCRIPTS").
  4. Once loaded, the terminal displays ASCII art, welcome message, command list, and system status (e.g., loaded files count).
- **Reset**: Clear localStorage via browser dev tools (Application > Storage > Local Storage) or use `/clear` for output only.

## User Interface Elements
- **Loading Screen**:
  - Centered ASCII logo in neon green.
  - Progress bar with scanning animation.
  - Status text with animated dots.
  - Subtext: "SYSTEM BOOT: WEBXOS v3.0".
  - Fades out after simulated phases (total ~3 seconds).
- **Terminal Layout**:
  - **Output Area**: Scrollable container for messages, ASCII art, file lists, and responses. Auto-scrolls on new content; manual scroll shows "↓" button to jump to bottom.
  - **Input Line**: Fixed at bottom with "$" prompt (changes to "calc>" in yellow during calculator mode). Blinking cursor, real-time text display.
  - **Messages**: Styled by type – green for commands/responses, yellow for scripts/calc, red for errors, dim green for info/system.
  - **ASCII Art**: Displays on load (large "WEBXOS" and smaller "CHATBOT") and in some responses.
  - **Scroll Behavior**: Smooth scrolling; button appears if not at bottom.
  - **Footer**: "WEBXOS v3.0" in dim green at input right.
- **Modals**:
  - **Context Editor**: Green-bordered modal for .md files. Header with title/stats, textarea for Markdown, buttons: Save, Read, Test Read, Delete.
  - **Script Editor**: Yellow-bordered modal for .js/.json. Similar to context but with Run, Test buttons; char count stat.
  - **Upload Modal**: For importing files; drag/drop or click to select .md/.json; shows file list, Process/Cancel buttons.
  - All modals: Centered, 85% viewport, close with ✕ or Esc; focus shifts to textarea.
- **File Displays**:
  - List views: Card-like items with title, preview/description, stats (chars), actions (Read/Edit/Run).
  - Empty states: Icons and messages like "No context files yet."
  - New buttons: Dashed borders to create new files.
- **Animations**:
  - Cursor blink.
  - Loading scan effect.
  - Typewriter for reading contexts (5ms/char delay).
- **Accessibility**: Keyboard-focused; tab auto-complete; arrow history navigation.

## Input and Navigation
- **Basic Input**: Type in hidden input field (visible as styled text with cursor). Enter submits.
- **History**: Up/down arrows cycle previous commands (up to unlimited, persisted).
- **Auto-Complete**: Tab suggests/matches commands (e.g., "/con" → "/context").
- **Modes**:
  - Normal: Green "$" prompt.
  - Calculator: Yellow "calc>" prompt; handles math only.
- **Focus**: Click anywhere refocuses input.
- **Shortcuts**: Esc closes modals; no other globals.

## Available Commands: Detailed Usage
All commands start with "/". Non-command text sends to dummy AI (e.g., "hello" → greeting with ASCII).
- **/help**: Displays "Available commands:" followed by pre-formatted list (one per line). No args.
- **/context**: Shows interactive list of .md files. Each card: Index, name, content preview (60 chars), char count, buttons (Read/Edit). Includes "+ CREATE NEW" button. Use "list" arg for simple text list (e.g., "/context list" → numbered names with chars).
- **/scripts**: Similar to /context but yellow-themed for .js/.json. Cards: Index, title, description, char/type, buttons (Run/Edit). "+ CREATE NEW SCRIPT" button. Args: "list" for text list; "run [name]" to execute specific script.
- **/import**: Opens upload modal. Accepts multiple .md (to contexts) or .json (to scripts). Shows selected files with sizes/types; Process adds them (parses JSON for title/desc/run if valid). Outputs success messages per file.
- **/export**: Downloads "webxos-export-[timestamp].md" with sections: Export Info (date, version, counts), Context Files (full Markdown content in code blocks), Scripts (title, desc, code in JSON blocks). No args; warns if no data.
- **/clear**: Clears output area, redisplays initial ASCII and welcome. Preserves storage.
- **/stats**: Outputs "SYSTEM STATISTICS:" with: Context files/count/chars, Scripts/count/chars, Storage used (bytes/KB/MB), History entries.
- **/calc**: Enters calculator mode (yellow theme). Initial help: Expressions like "2+2", "history" for last 5 calcs. Uses built-in script for eval (sanitized; no vars). "/exit" returns to normal.
- **/list**: Combines /stats, /context list, /scripts list in one output.
- **/run [name]**: Executes script by exact title (case-insensitive). Outputs "[Running: name]" then result/error. For calculator script: Enters calc mode.
- **/newscript**: Opens script editor with template examples (JSON with run, pure JS func, text, object).
- **/read [file/index]**: Locates .md by name (partial match, ignores .md) or index (1-based). Outputs "Reading: name" then typewriter-animates full content, ends with "[End of file]". Errors if not found or typing in progress.

## Context Files: Management and Usage
- **Purpose**: Store knowledge as Markdown (headings, lists, code blocks).
- **Creation**: Via /context > New button or upload.
- **Editor Modal**:
  - Title editable (default "Untitled.md").
  - Textarea with placeholder Markdown example.
  - Save: Updates/creates; persists to storage.
  - Read: Closes modal, runs /read on current.
  - Test Read: Previews 200 chars in terminal.
  - Delete: Removes file; refreshes list.
- **Reading**: Typewriter effect for immersion; supports long files.
- **Storage**: Array of objects {id, name, content, type: 'manual/imported', timestamp}.
- **Limits**: Char count displayed; browser storage caps total size.

## Scripts: Management and Execution
- **Purpose**: Executable code in JS/JSON/text.
- **Types**:
  - JSON with "run": Function string; parsed and run.
  - Pure JS function: Eval as func().
  - Text/Invalid JSON: Outputs as string.
- **Built-in**: "calculator" (JSON with run for math eval, history).
- **Creation**: /newscript or /scripts > New; template with examples.
- **Editor Modal**:
  - Title editable (default "Untitled.js").
  - Textarea with examples; live char count.
  - Save: Parses type/desc; persists.
  - Run: Executes code, outputs in terminal, closes modal.
  - Test: Executes without closing; shows "[TEST] result".
  - Delete: For non-builtins only.
- **Execution**: Safe in Function() sandbox; errors caught and displayed. Outputs "[name] result" in yellow.
- **Storage**: Array of {id, title, desc, type, code, builtin: bool}.
- **Limits**: No external calls; browser sandbox.

## Calculator Mode: In-Depth
- **Activation**: /calc or /run calculator.
- **Input**: Math expressions (operators: + - * / (); sanitized to numbers/ops only).
- **Features**:
  - Eval via new Function().
  - History: Up to 50 entries {expr, result, timestamp}; "history" shows last 5.
  - Errors: "Error: message" in red.
  - Exit: "/exit" resets prompt.
- **Output**: "= result" in yellow.
- **Persistence**: Session-only history (not stored).

## Import and Export: Data Handling
- **Import**:
  - Modal: Click/Drag for files; filters .md/.json.
  - Process: Adds .md as contexts (full text); .json as scripts (parses for title/desc/run; falls back to text if invalid).
  - Outputs: "Added [type]: name" per file; total summary.
- **Export**:
  - Markdown structure: Headers for info, contexts (```markdown:disable-run
  - Auto-downloads; includes all data.
- **Backup**: Use export for manual backups; localStorage is browser-specific.

## Dummy AI Chat Functionality
- **Trigger**: Any non-/ input.
- **Responses**: Basic keyword-based (e.g., "hello" → greeting + ASCII; "script" → count info; fallback: self-description).
- **Simulation**: No real AI; static logic. Outputs in green "ai-response" style.
- **Expansion Potential**: Could integrate real API, but current is dummy.

## Advanced Tips and Customization
- **Debug**: Browser console shows errors; inspect elements for styles.
- **Customization**: Edit HTML/CSS/JS directly (e.g., change --neon color; add commands in processCommand()).
- **Performance**: Handles 100s of files; slow on very large contexts (typewriter lags).
- **Security**: Local only; scripts can't access network/DOM beyond eval.
- **Three.js Mindset**: For future 3D editor expansion – imagine modal with canvas for 3D script previews (e.g., visualize calc graphs); current: No 3D, but editor code structured for potential WebGL integration (e.g., add <canvas> in modals).
- **Real-Time Data**: None; all local/static. No mock code here.

## Troubleshooting
- **Loading Stuck**: Refresh; check console for JS errors.
- **No Save**: Storage full? Clear data.
- **Script Errors**: Check code syntax; no installs.
- **Modal Issues**: Esc or ✕ to close; click outside doesn't.
- **Browser Quirks**: Safari may limit localStorage; mobile touch input works but no hardware keyboard shortcuts.

## Limitations and Future Ideas
- **Limits**: No network, multiplayer, or plugins; dummy AI only; 2D no graphics.
- **Ideas**: Add real AI via service worker; Three.js for 3D script viz (e.g., 3D calc plots); more builtins (e.g., graph scripts).
- **License**: Open-source MIT; modify freely.

Updated for v3.0 as of December 15, 2025.
