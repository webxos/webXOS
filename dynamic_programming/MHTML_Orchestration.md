# MHTML / Self-Contained HTML Orchestration

## Purpose

Produce a single, fully self-contained HTML document that can be saved as `.html` (or strict `.mhtml`) and opened offline. Ideal for rapid UI/UX prototypes, dashboards, interactive demos, and agent-generated front-ends.

## Non-Negotiable Rules

- All CSS inside `<style>` tags.
- All JavaScript inside `<script>` tags (no external CDNs unless absolutely critical and noted).
- All images, icons, fonts, media embedded as base64 data URIs.
- No external dependencies that break offline use.
- File must open and function when double-clicked or dropped into a browser with no network.

## Quality Standards (“Megalithic” Generation)

- Build the fullest reasonable version of the idea, not a minimal stub.
- Include realistic content, multiple states/views, interactions, responsive design, dark/light mode if appropriate, loading/empty states, and polish.
- Use modern, clean, accessible HTML5 + CSS (Flexbox/Grid) + vanilla JS.
- Prefer a single cohesive page or SPA-like experience inside one file.
- Add subtle animations and micro-interactions where they enhance the prototype.

## Response Structure for an Agent

1. Short title + one-sentence summary.
2. Complete code inside a single markdown code block labeled `html`.
3. Optional short “How to use” note.

## Technical Preferences

- Mobile-first responsive design.
- Semantic HTML.
- CSS custom properties for theming.
- Clean, readable code with light comments only where helpful.
- No build steps, no frameworks that require compilation.

## Why This Fits Dynamic Programming

The single HTML file is the **value function** of the UI idea. An agent can:

- Generate the entire file in one shot (optimal substructure).
- Mutate sections (CSS variables, JS state machines, content blocks) without breaking the whole.
- Re-render / re-evaluate by simply reopening the file (overlapping subproblems of layout and interaction).

This is the frontend analogue of a Bellman update: each improvement to a component improves the global prototype.
