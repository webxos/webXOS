# MicroSeek OS by webXOS 2025

*webxos.netlify.app/microseek*

A minimal, browser-based text operating system agent inspired by terminal interfaces. Built as a single HTML file for portability, it provides file management, editing, and computation tools in a retro green-screen aesthetic. No server required—runs entirely client-side with local storage via IndexedDB.

## Features

- **Terminal Interface**: Command-line driven with history, suggestions, and auto-complete navigation.
- **File System**: Branch-based organization (like Git branches) for grouping files. Supports Markdown files (.md) with auto-sanitization (spaces to underscores).
- **Commands**:
  - `HELP`: Display command list.
  - `TREE`: Show file structure.
  - `BROWSER`: Open graphical file manager with drag-and-drop between branches.
  - `WRITE [FILE]`: Create or edit a file in built-in editor.
  - `READ FILE.MD`: View file in typewriter mode (animated character-by-character reading with speed control).
  - `EDIT FILE.MD`: Open file in editor.
  - `RENAME FILE.MD`: Prompt to rename a file.
  - `DELETE FILE`: Delete a file with confirmation.
  - `BRANCH [NAME]`: Switch or create branches.
  - `CALC`: Open calculator popup.
  - `IMPORT`: Upload files.
  - `EXPORT ALL`: Download entire system as unified Markdown.
  - `EXPORT FILE.MD`: Download single file.
  - `CLEAR`: Clear terminal.
- **Typewriter Mode**: Animated reading with adjustable speed (50-400 CPS), stop via `READ STOP`.
- **File Browser**: Windows-style popup with sidebar, drag-drop, new file/branch creation, rename/delete.
- **Calculator**: Basic arithmetic with parentheses and error handling.
- **Persistence**: Files stored in browser's IndexedDB—data survives page reloads.
- **Mobile Optimized**: Responsive design with touch support.
- **Extras**: Simulated loading boot sequence, RAG embeddings (basic TF.js), GPU.js integration (unused in core).

## How It Works

MicroSeek OS is a self-contained HTML file with inline CSS and JavaScript. Here's the breakdown:

1. **Boot Process**:
   - Simulates loading with progress bar (BOOTING KERNEL → READY).
   - Initializes TensorFlow.js for embeddings, Math.js for calculations, and GPU.js.
   - Sets up IndexedDB for file storage (database: `microseek_os`, store: `files`).
   - Loads existing files or creates a default `README.md`.

2. **Terminal**:
   - Input commands in the prompt (`/tree/main>`).
   - Output displays in a scrollable area with syntax highlighting (e.g., links for files/branches).
   - Suggestions appear below for quick actions.

3. **File Management**:
   - Files are Markdown-only, stored with metadata (name, branch, content, size, modified, accessed).
   - Branches organize files; switch with `BRANCH name`.
   - Import: Upload text files, select branch.
   - Export: Unified MD with metadata or single files.

4. **Editor Popup**:
   - Simple textarea for Markdown editing.
   - Save updates IndexedDB; delete with confirmation.

5. **Typewriter Reading**:
   - Animates content reveal with blinking cursor.
   - Speed slider for real-time adjustment.
   - Stops on end or command.

6. **Calculator Popup**:
   - Grid-based buttons for digits/operators.
   - Evaluates expressions using Math.js.

7. **File Browser Popup**:
   - Sidebar for branches/all files.
   - Grid view with icons, sizes, rename/delete buttons.
   - Drag files to branches for moving.

8. **Technical Notes**:
   - **Storage**: IndexedDB persists data locally (no cloud).
   - **Sanitization**: Filenames auto-convert spaces to underscores, remove invalid chars.
   - **Embeddings**: Basic TF.js for future RAG/search (currently simulated).
   - **No Internet**: All offline except CDN scripts (TF.js, Math.js).
   - **Compatibility**: Modern browsers (Chrome, Firefox, Safari). Mobile-friendly but best on desktop.

To run: Open `microseek.html` in a browser. No installation needed.

## Installation

1. Download `microseek.html`.
2. Open in any modern browser.
3. (Optional) Host on a static server for sharing.

No dependencies beyond browser APIs. Scripts loaded from CDNs:
- TensorFlow.js: For embeddings.
- Math.js: For calculations.

## Usage

1. Open the file—wait for boot.
2. Type `HELP` for commands.
3. Create a file: `WRITE notes.md`.
4. Read: `READ notes.md` (adjust speed slider).
5. Organize: `BRANCH ideas`, then `WRITE idea1.md`.
6. Browse: `BROWSER` for visual management.
7. Export: `EXPORT ALL` for backup.

Data is local—clear browser storage to reset.

## Use Cases

- **Personal Note-Taking**: Quick Markdown notes in branches (e.g., `daily`, `projects`). Typewriter mode for focused reading.
- **Offline Writing Tool**: Draft articles/emails without distractions. Export to MD for GitHub/blogs.
- **Learning/Prototyping**: Teach CLI basics; simulate OS in education. Calculator for quick math.
- **Idea Organization**: Branch per topic (e.g., `recipes`, `todo`). Drag-drop to reorganize.
- **Portable Knowledge Base**: Carry in a single file; import/export for backups. Useful for travelers/offline workers.
- **Minimalist Computing**: Retro interface for focus; extend with JS for custom commands.

Built by webxos  
x.com/webxos | webxos.netlify.app | github.com/webxos
