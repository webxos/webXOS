# AUTO-CAD PRO KERNEL v14.0.5 (BETA TEST ONLY)
**by webXOS · 2026**

[![Status](https://img.shields.io/badge/status-beta-orange)](https://github.com/webXOS)
[![Single File](https://img.shields.io/badge/architecture-single--file-blue)](https://github.com/webXOS)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> A self-contained, zero-dependency 3D mechanical assembly + PCB design editor that runs entirely in the browser.  
> One HTML file. No build step. No frameworks. No backend.

This is an **active beta** development repository. The entire application lives in a single `.html` file. The goal is to keep it lightweight, portable, and fully editable while pushing the limits of what a pure software-rendered CAD tool can do in modern browsers.

---

## Quick Start

1. Download or clone the repository.
2. Open `index.html` (or whatever the single file is named) in a modern browser (Chrome, Firefox, Edge, Safari 16+).
3. That’s it. No install, no server required.

Works offline after the first load (fonts are loaded from Google Fonts; the rest is self-contained).

---

## Core Features

### 3D Assembly Workspace
- Custom software 3D renderer (no WebGL)
- Drag & drop mechanical parts from the library
- Move / Rotate tools with Edit mode snapping (5 mm / 15°)
- Real-time properties panel (position, rotation, volume, mass)
- Gauntlet stress test (drop physics simulation)
- Floating IoT node animation + camera presets
- Export viewport as PNG or full project HTML report

### PCB Design Workspace
- Interactive 2-layer board editor
- Component footprints with pin nets
- Manual routing (waypoints, L-bends, net merging)
- Free pads & vias
- Design Rule Check (DRC) + Netlist panel
- Grid A* autorouter (two layers + automatic via insertion)
- PCB Wizard – guided board generation (place + route)
- Live 3D preview of the board and components

### System
- Full undo / redo (up to 60 steps)
- Browser localStorage snapshots + JSON import/export
- Keyboard-driven workflow
- Responsive panels (collapsible on smaller screens)
- Units: mm / in / mil

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1` `2` `3` | Select / Move / Rotate (3D) or Select / Pad (PCB) |
| `W` | Route tool (PCB) |
| `4` | Via tool (PCB) |
| `R` | Rotate selected PCB component 90° |
| `F` | Focus selected 3D part |
| `Del` / `Backspace` | Delete selection |
| `Ctrl+Z` / `Ctrl+Y` | Undo / Redo |
| `Enter` | Finish unterminated trace (PCB) or add focused library part |
| `Esc` | Cancel drag / cancel route / close dialogs / deselect |
| `L` | Toggle kernel log |
| `F1` | Help |

---

## Architecture Notes (Single-File Constraints)

- Everything lives in one HTML file (~15–20 kLOC of JS + CSS).
- Software rasterizer for 3D (painters algorithm + backface culling).
- PCB analysis and autorouter are pure JavaScript.
- No external libraries except Google Fonts.
- State is serialized to plain JSON for history and export.
- Performance is intentionally traded for zero dependencies and extreme portability.

---

## Known Issues & Bug Bounties

This is a beta. The following issues are known and open for community fixes.  
Feel free to open a PR or issue. High-impact fixes will be credited in the changelog.

### High Priority

| ID | Description | Severity | Notes / How to Reproduce |
|----|-------------|----------|--------------------------|
| **B001** | Software 3D renderer becomes very slow (>30 parts or complex meshes + many traces) | High | Run Gauntlet with 30–40 parts while the PCB has many traces. FPS drops hard. |
| **B002** | Autorouter can leave unroutable nets as ratsnest even when a path exists | High | Complex boards with dense components or power nets. Grid resolution is adaptive but still limited. |
| **B003** | Undo/redo of large PCB states (hundreds of traces/vias) can cause brief freezes | Medium-High | Create a board via Wizard with max components, then undo repeatedly. |
| **B004** | LocalStorage quota can be exceeded on long sessions → silent save failure | Medium | Save many large projects. Browser private mode also fails. |
| **B005** | Touch / multi-touch pinch-zoom on both canvases still has edge-case pointer identity bugs | Medium | Fast two-finger gestures while a drag is starting. |

### Medium Priority

| ID | Description | Severity | Notes |
|----|-------------|----------|-------|
| **B006** | DRC misses some real-world rules (annular rings, exact solder-mask clearance, copper-to-edge on complex outlines) | Medium | Current DRC is intentionally lightweight. |
| **B007** | Floating-point drift on long drag sequences or repeated 90° rotations | Low-Medium | Rare visual misalignment after many operations. |
| **B008** | Seed board DRC self-check can report false positives after certain undo sequences | Low | Boot-time check is usually clean. |
| **B009** | IoT satellite spheres and gauntlet physics do not participate in proper collision detection | Low | Visual only. |
| **B010** | HTML report export embeds a large base64 PNG → file size balloons | Low | Expected for single-file design. |

### Wishlist / Not Bugs (but welcome)

- Gerber / Excellon / BOM export
- More than 2 copper layers
- Schematic capture
- Proper mechanical ↔ PCB assembly constraints
- WebGL fallback renderer
- Better mobile layout

---

## Bug Bounty Style Contributions

We treat this as an open beta development playground.  
Useful pull requests that fix any of the issues above (or discover new ones) will be:

- Credited in the next version banner and changelog
- Marked as “Community Fix” in the commit history
- Prioritized for merge if they keep the **single-file** constraint

**Preferred contribution format:**
1. Fork → edit the single HTML file
2. Add a clear comment near the fix: `// BUGFIX B00X – short description`
3. Open a PR with steps to reproduce the original issue and how you verified the fix

No build system. No package.json. Just pure improvements to the single file.

---

## Development Philosophy

- Stay single-file as long as humanly possible.
- Prefer readable, well-commented code over cleverness.
- Performance optimizations are welcome but must not introduce dependencies.
- Accessibility and keyboard-first workflows are first-class.
- The 3D renderer and autorouter are intentionally educational / experimental.

---

## License

MIT License  
Copyright (c) 2026 webXOS

You are free to use, modify, and redistribute this single-file application, including commercial use, as long as the original copyright notice is retained.

---

## Credits

- Concept, architecture & implementation: **webXOS**

---

**Current Version:** v14.0.5  
**Last Updated:** 2026  

Open the file. Break it. Fix it. Make it better.  
That’s the point of a single-file beta CAD kernel.
