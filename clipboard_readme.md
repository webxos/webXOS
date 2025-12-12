# webXOS prompt Clipboard

A retro-futuristic terminal interface for managing AI/LLM prompts. Built with HTML, CSS, and JavaScript, featuring prompt creation, editing, reviewing, and exporting in a command-line style environment. Inspired by old-school terminals with green-on-black aesthetics, integrated with modern features like token counting and TensorFlow.js simulation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Commands Reference](#commands-reference)
- [Architecture and Technical Details](#architecture-and-technical-details)
- [Prompt Management Workflow](#prompt-management-workflow)
- [Customization and Extension](#customization-and-extension)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

webXOS Clipboard is a web-based application designed to serve as a specialized clipboard for handling prompts intended for large language models (LLMs) or AI systems. It emulates a classic terminal interface, complete with a blinking cursor, typewriter-effect responses, and a monochromatic green glow reminiscent of vintage computing. This tool is particularly useful for developers, AI enthusiasts, and prompt engineers who need to craft, refine, store, and analyze prompts efficiently.

The core philosophy behind webXOS Clipboard is to blend nostalgia with functionality. Users interact via a command-line interface (CLI) where they can type commands to create new prompts, edit existing ones, review token counts for cost estimation, format text for better readability, and even upload or export prompts in markdown format. It uses local storage for persistence, ensuring prompts are saved across sessions without needing a backend server.

Under the hood, it incorporates TensorFlow.js for simulated AI responses (though currently mocked for demonstration), providing suggestions and reviews to optimize prompts. The application is fully client-side, making it easy to deploy as a static web page. With its modular JavaScript structure, it's extensible for adding more AI integrations or features.

This project was built to address common pain points in prompt engineering: tracking token usage to manage API costs, organizing multiple prompts, and quickly iterating on ideas without leaving the browser. Whether you're preparing inputs for models like GPT or experimenting with custom AI workflows, webXOS Clipboard streamlines the process in an engaging, hacker-friendly environment.

## Features

- **Terminal-Style Interface**: Full-screen terminal with input prompt, output scrolling, and a blinking cursor for an immersive CLI experience.
- **Prompt Creation and Editing**: Quickly create new prompts via direct input or commands, with a built-in modal editor featuring real-time stats (tokens, words, cost estimation).
- **Review and Suggestions**: Analyze prompts for token count, word count, and simulated AI feedback, including optimization suggestions like adding line breaks or context.
- **Storage and Management**: Save prompts to local storage, list them with previews, load/delete by ID, and attach files (e.g., markdown uploads).
- **Export and Import**: Export all prompts as a formatted markdown file for sharing or backups; upload markdown files to import as new prompts.
- **Formatting Tools**: Auto-format prompts for clarity, capitalization, and punctuation.
- **Stats and Monitoring**: System-wide stats on total prompts, tokens, and estimated costs; individual prompt timestamps and sizes.
- **Modal Windows**: Dedicated modals for editing, exporting, and uploading, with drag-and-drop support and preview areas.
- **Typewriter Effect**: Responses appear character-by-character for a dynamic feel.
- **TensorFlow.js Integration**: Loaded for potential ML-based prompt analysis (currently simulated).
- **Responsive Design**: Works on desktop browsers; mobile support limited due to terminal nature.
- **Customizable Aesthetics**: Easy to tweak CSS for colors, fonts, or animations.

These features make webXOS Clipboard more than just a text editor—it's a dedicated workspace for prompt lifecycle management, from ideation to deployment in AI applications.

4. **Dependencies**:
   - TensorFlow.js is loaded via CDN: `<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>`.
   - No other external dependencies; all CSS and JS are inline.

If deploying to GitHub Pages or a web host, just upload the HTML file. Ensure local storage is enabled in the browser for prompt persistence.

## Usage

Launch the application by opening `clipboard.html`. You'll see a loading screen with the webXOS ASCII art, followed by the terminal interface.

- **Basic Interaction**:
  - Type commands starting with `/` (e.g., `/help`) or enter plain text to create a new prompt.
  - Use the arrow keys for history navigation.
  - Click the send button (↵) or press Enter to submit.

- **Example Workflow**:
  1. Type `/new Hello, world!` to create a prompt.
  2. It auto-reviews with stats and suggestions.
  3. Edit via the [EDIT] link or `/edit <id>`.
  4. Save with `/save`.
  5. List with `/list`.
  6. Export all with `/export all`.

The interface auto-scrolls to new output, and modals appear for advanced actions. Focus is always on the hidden input for seamless typing.

For detailed command usage, see the next section. Remember, all data is stored in browser local storage—clearing cache will reset prompts.

## Commands Reference

webXOS Clipboard uses a slash-command system for control. Here's a comprehensive breakdown:

- **/help**: Displays a full list of commands with descriptions. This is your go-to for quick reference, showing everything from basic creation to advanced management.
  
- **/new [text]**: Creates a new prompt with the provided text. If no text is given, it prompts an error. The new prompt becomes the "current" one, displayed with its ID and an [EDIT] link. Automatically triggers a review for immediate feedback.

- **/list**: Lists all saved prompts, showing ID, preview (truncated to 40 chars), token count, and links for editing or viewing attached files. Useful for browsing your prompt library without loading each one.

- **/review [text]**: Analyzes the given text (or current prompt if omitted). Outputs detailed stats like tokens, words, lines, and cost estimation (based on $0.002/1000 tokens). Simulates a response and provides suggestions for improvement, such as splitting long text or adding punctuation.

- **/save**: Saves the current prompt to local storage. If already saved, updates it. Prevents data loss after edits.

- **/load [id]**: Loads a saved prompt by ID, setting it as current and displaying its content with metadata.

- **/delete [id]**: Removes a prompt by ID from storage. Irreversible, so use with caution.

- **/clear**: Clears the terminal output, resetting to the welcome screen. Doesn't affect saved data.

- **/stats**: Shows system statistics, including total prompts, aggregate tokens, average tokens per prompt, estimated total cost, and storage usage in bytes.

- **/format [text]**: Formats the given text (or current prompt) by trimming whitespace, capitalizing the first letter, adding ending punctuation, and replacing multiple spaces. Updates the current prompt if applicable.

- **/upload**: Opens the upload modal for importing a markdown (.md, .txt) file as a new prompt. Supports drag-and-drop, displays file size, and auto-reviews after upload.

- **/export all**: Prepares all prompts for export in a markdown format, opening a modal with preview. Includes headers, dates, tokens, and fenced code blocks for each prompt. Options to copy or download.

- **/edit [id]**: Opens the editor modal for the specified prompt (or current if omitted). Features a textarea with live stats updates on keypress.

Non-command input is treated as a new prompt via `/new`. All commands are case-insensitive and support basic argument parsing.

## Architecture and Technical Details

The application is structured as a single HTML file with embedded CSS and JavaScript for portability. Key components:

- **HTML Structure**: Loading screen, terminal container (output and input areas), and modals for editor/export/upload. Uses semantic classes for styling.

- **CSS Styling**: Retro theme with #000 background, #0f0 text, and animations (blink, load bar). Scrollbars customized, modals with borders and shadows. Responsive but optimized for desktop.

- **JavaScript Logic**:
  - **State Management**: A `state` object tracks prompts, history, input, and typing queue.
  - **Input Handling**: Hidden input for real typing, mirrored to visible text with cursor positioning via offset calculation.
  - **Typewriter Effect**: Queues responses for character-by-character output, enhancing immersion.
  - **Modals and Events**: Click handlers for links/buttons, keydown for Escape/Enter, file readers for uploads.
  - **Storage**: JSON-serialized prompts in localStorage, with load/save functions.
  - **Token Counting**: Simple heuristic (length / 4) for estimation; can be replaced with actual tokenizer.
  - **Simulations**: Mock responses and suggestions; TensorFlow.js is loaded but not actively used—future hook for real ML analysis.

Performance is lightweight, with no heavy computations. Browser compatibility: Modern browsers (ES6+). Potential extensions: Integrate real AI APIs or add search functionality.

## Prompt Management Workflow

Managing prompts in webXOS Clipboard follows a structured yet flexible workflow, designed to mimic a developer's iterative process:

1. **Ideation and Creation**: Start by typing a prompt directly or using `/new`. The system generates an ID, displays it with edit options, and auto-reviews for immediate insights. This step ensures quick capture of ideas without overhead.

2. **Review and Optimization**: Use `/review` to get metrics. Token count helps estimate API costs (e.g., for OpenAI), while suggestions guide refinements—like breaking monolithic text into sections for better LLM comprehension. The simulated response previews potential output quality.

3. **Editing and Formatting**: Click [EDIT] or use `/edit` to open the modal. Edit in a resizable textarea, watching live stats. Format with one click to standardize structure, improving prompt effectiveness (e.g., clear instructions lead to better AI responses).

4. **Storage and Organization**: Save via `/save`, list with `/list` for overviews. Attach files during upload for context-rich prompts (e.g., code snippets). Delete unused ones to keep things tidy.

5. **Collaboration and Archiving**: Export all to markdown for GitHub repos or sharing. Import from files to collaborate or restore backups. This closes the loop, allowing prompts to live beyond the browser.

This workflow reduces friction in prompt engineering, saving time on revisions and cost tracking. For advanced users, integrate with external tools by copying formatted prompts.

## Customization and Extension

webXOS Clipboard is highly modular and customizable:

- **Theming**: Edit CSS variables for colors (e.g., change #0f0 to blue). Add fonts or animations.
- **Extending Commands**: Add cases to the `process(cmd)` function for new features, like `/search` for prompt querying.
- **AI Integration**: Replace `simulateResponse` with actual TensorFlow.js models for sentiment analysis or prompt scoring.
- **Storage Alternatives**: Swap localStorage for IndexedDB for larger datasets.
- **Mobile Optimization**: Add media queries for touch inputs and smaller screens.

## License

MIT License. See [LICENSE](LICENSE) for details. Free to use, modify, distribute.
