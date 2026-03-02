# Shadowclaw v1.1 (Testing)


![[SHADOWCLAW](https://github.com/webxos/webXOS/blob/main/assets/shadowclaw.jpeg)](https://github.com/webxos/webXOS/blob/main/assets/shadowclaw.jpeg)

<div style="max-width: 100%; overflow-x: auto; background: #f6f8fa; padding: 16px; border-radius: 8px;">
  <pre style="font-family: 'Courier New', monospace; font-size: clamp(12px, 2vw, 18px); line-height: 1.2; margin: 0;">
  ____  _               _                   _
 / ___|| |__   __ _  __| | _____      _____| | __ ___      __
 \___ \| '_ \ / _` |/ _` |/ _ \ \ /\ / / __| |/ _` \ \ /\ / /
  ___) | | | | (_| | (_| | (_) \ V  V / (__| | (_| |\ V  V /
 |____/|_| |_|\__,_|\__,_|\___/ \_/\_/ \___|_|\__,_| \_/\_/
  </pre>
</div>
                                                  
**Shadowclaw** is a minimal, single‑binary personal AI agent written in C. It follows the *OpenClaw* philosophy: self‑hosted, tool‑using, persistent memory, and minimal dependencies. The core memory management uses **Tsoding's "shadow header" trick** (like `stb_ds` but for a growable arena). All data (conversation history, tool definitions, results) lives inside a single `realloc`‑ed memory block with a hidden header. The agent communicates with a local LLM (Ollama) via curl, can execute shell commands, read/write files, perform HTTP GET, and evaluate simple math expressions. State is automatically saved to disk after every interaction.

**Niche edge use cases:**

RPi Zero/IoT: offline sensor scripts (shell + persistent shadow.bin)

Air-gapped systems: USB-stick local LLM agent (file/HTTP/math)

Embedded routers: 100-200KB network automation (low-mem Linux)

Low-power edge nodes: self-hosted persistent AI, no cloud.

---

## Repository Structure

```
shadowclaw/
├── shadowclaw.c      # main program: arena, tools, LLM glue, event loop
├── Makefile          # simple build file
├── cJSON.c           # lightweigt JSON parser (from cJSON library)
└── cJSON.h           # cJSON header
```

---

## Features

- **Growable shadow arena** – all objects stored in one contiguous block with hidden headers.
- **Persistent memory** – saves and loads full arena to/from `shadowclaw.bin`.
- **Tool use** – shell, file I/O, HTTP GET, math (via `bc`).
- **Ollama integration** – talk to any model running locally.
- **Blob storage** – system prompts, user messages, assistant replies, tool calls, results.
- **Tiny footprint** – stripped binary ≈ 100–200 KiB.

---

## Requirements

- Linux (developed on Debian, should work on most Unix‑likes)
- `gcc`, `make`, `libcurl4-openssl-dev`, `bc`
- [Ollama](https://ollama.com/) running locally (default `http://localhost:11434`) with a model pulled (e.g. `llama3.2`)

Install build dependencies on Debian/Ubuntu:

```bash
sudo apt update
sudo apt install build-essential libcurl4-openssl-dev bc
```

---

## Build

Just run `make`:

```bash
make
```
Optionally strip the binary to make it even smaller:

```bash
make strip
```

The executable `shadowclaw` will appear in the current directory.

---

## Usage

1. Make sure Ollama is running and you have a model (change `ollama_model` in `shadowclaw.c` if needed).
   
2. Run the agent:
   ```bash
   ./shadowclaw
   ```
3. Type your input and press Enter. The agent will:
   - Build a prompt from recent conversation and system message.
   - Call Ollama.
   - If the LLM outputs a tool call in the format ` ```tool {"tool":"name","args":"..."} ``` `, the tool is executed and the result is fed back in the next turn.
   - The conversation and state are saved to `shadowclaw.bin` after each turn.

Example tool invocation (the LLM must produce this):

````
I need to list the current directory.
```tool
{"tool":"shell","args":"ls -la"}
```
````

To exit, press **Ctrl+D**.

---

Optional:

- **Change Ollama endpoint/model**: edit `ollama_endpoint` and `ollama_model` in `shadowclaw.c`.

Find this line (474) in the shadowclaw.c file:

  ```bash
  // --------------------------------------------------------------------
//  Main
// --------------------------------------------------------------------
int main(int argc, char **argv) {
    (void)argc; (void)argv;

    const char *state_file = "shadowclaw.bin";
    const char *ollama_endpoint = "http://localhost:11434";
    const char *ollama_model = getenv("OLLAMA_MODEL");
    if (!ollama_model) {
        ollama_model = "qwen2.5:0.5b";
    }
  ```

Adjust the model endpoint in "ollama_model = "qwen2.5:0.5b";" to meet your model.

- **Add new tools**: extend the `tools` array in `shadowclaw.c` with a name and a function that takes a `const char*` argument and returns a `char*` (must be `malloc`ed, the caller will `free` it).

---

## How It Works

### Shadow Arena

All dynamic data lives in a single `realloc`‑ed block.  
A `ShadowHeader` is stored **immediately before** the user‑visible pointer.  
This header holds capacity, used length, a magic number, and a dirty flag.

```
+-------------------+---------------------+
|  ShadowHeader     |  payload (blobs)    |
+-------------------+---------------------+
^
`- pointer returned to user
```

### Blob Format

Each item (system prompt, user message, tool result, etc.) is stored as:

```
+------------+------+------+-----------------+
| BlobHeader | kind |  id  | payload (bytes) |
+------------+------+------+-----------------+
```

- `BlobHeader` contains the payload size, kind, and a 64‑bit ID.
- The payload is arbitrary data (null‑terminated strings for most kinds).

### Persistence

The whole arena (header + payload) is written to `shadowclaw.bin` with `fwrite`.  
On startup, if the file exists and has a valid magic number, it is loaded back.

### Tool Calling

The LLM is instructed to output a tool call inside a fenced block:

```tool
{"tool":"name","args":"arguments"}
```

The agent parses the block, executes the tool, and appends the result as a new blob (kind 5). The result is then visible in the next prompt.

---

# Updated 3/2/2026: Shadowclaw v1.1

This version of Shadowclaw builds upon the original minimal AI agent, adding a set of convenient built‑in commands and an extra native tool. All changes are contained in `shadowclaw.c` – no external dependencies or scripts required.

### Built‑in Slash Commands

Type any of these commands directly at the `>` prompt – they are handled without invoking the LLM.

| Command   | Description |
|-----------|-------------|
| `/help`   | Show this help message. |
| `/tools`  | List all available tools (shell, file I/O, HTTP, math, `list_dir`). |
| `/state`  | Display current arena memory statistics (capacity, used bytes, dirty flag). |
| `/clear`  | Clear the conversation history while retaining the system prompt. |
| `/chat`   | Remind you that you are already in chat mode (the default behaviour). |
| `/exit`   | Quit Shadowclaw. |

### New Tool: `list_dir`

A lightweight native tool that lists the contents of a directory **without spawning a shell**.  
Example usage (the LLM can produce this, or you can type it manually):

````
```tool
{"tool":"list_dir","args":"."}
```
````

### Persistent Memory

All conversations and tool results are automatically saved to `shadowclaw.bin` and reloaded on restart.

## Requirements

- Linux with `gcc`, `make`, `libcurl`, and `bc`.
- [Ollama](https://ollama.com/) (optional, for LLM features) with a model like `qwen2.5:0.5b`.

## Build

```bash
make
```

## Run

```bash
./shadowclaw
```

Use the slash commands above to get started – even without Ollama running, you can explore the built‑in features.

## Notes
- The original LLM/tool functionality remains unchanged.
- All warnings during compilation (from `cJSON.c`) are harmless and can be ignored.
- Locally you can still use Shadowclaw's built‑in commands and manually invoke tools via the ```tool syntax – those work without Ollama. The LLM features require a working Ollama connection.

## Credits

- **Tsoding** – for the “insane shadow data trick” (the header‑before‑data arena idea).
- **cJSON** – Dave Gamble and contributors – the minimal JSON parser used.
- **curl** – HTTP requests to Ollama.
- **Vibe Code** - xAI Grok 4.20 and Deepseek AI
- **Openclaw**
---

## License

This project is released under the open sourced MIT License.  
cJSON is also MIT licensed.
