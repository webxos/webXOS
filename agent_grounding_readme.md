# AGENT GROUNDING – User & Agent Guide

**AGENT GROUNDING** is a lightweight, self‑contained web application that acts as a coordination and utility hub for autonomous AI agents.  
It runs entirely in your browser (or as an installable PWA) with **no backend** – all data stays on your device using IndexedDB.

Agents communicate with it by sending JSON messages via HTTP `fetch` requests.  
Humans get a simple debug interface to monitor activity and test phases manually.

---

## 📌 How to Get It

You have two options:

1. **Use a hosted version** – deploy the single `index.html` file to any static host (GitHub Pages, Vercel, Netlify, etc.) and point your agents to that URL.  
2. **Run locally** – just open the HTML file in a modern browser. The service worker will register and the app works offline.

Once opened, the page displays a minimal console showing recent requests and a form for manual testing.

---

## 👤 Human Interface

When you load the page, you’ll see:

- **Last requests** – the five most recent interactions (method, path, request/response JSON) are listed for debugging.
- **Manual test panel** – select a phase (1–10), enter JSON data, and click:
  - **Call handler directly** – invokes the phase logic inside the page (no network).
  - **Send via fetch** – makes an actual `fetch` request to `/api` (useful for testing the service worker).

Use this panel to experiment with phases without writing any code.

---

## 🤖 Agent Interaction

Agents interact with AGENT GROUNDING by sending HTTP requests to the `/api` endpoint.  
Both `GET` and `POST` methods are supported.

### Request Format

- **POST** (recommended): send a JSON body with `phase` (number 1–10) and `data` (object).
  ```json
  {
    "phase": 1,
    "data": {
      "agent_id": "agent-123",
      "capabilities": ["memory"],
      "version": "1.0"
    }
  }