# Shadowclaw v1.3 (Testing) 

*(v3.2 is out now: https://github.com/webxos/shadowclaw/)*


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
                                                  
**Shadowclaw** is a minimal, single‑binary agent harness written in C. It follows the *OpenClaw* philosophy: self‑hosted, tool‑using, persistent memory, and minimal dependencies. The core memory management uses **Tsoding's "shadow header" trick** (like `stb_ds` but for a growable arena). All data (conversation history, tool definitions, results) lives inside a single `realloc`‑ed memory block with a hidden header. The agent communicates with a local LLM (Ollama) via curl, can execute shell commands, read/write files, perform HTTP GET, and evaluate simple math expressions. State is automatically saved to disk after every interaction.

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

## Define your model:

- **Change Ollama endpoint/model**: edit `ollama_endpoint` and `ollama_model` in `shadowclaw.c`.

Find this line (507) in the shadowclaw.c file:

  ```bash
// --------------------------------------------------------------------
//  Main (with slash commands from v1.2.2)
// --------------------------------------------------------------------
int main(int argc, char **argv) {
    const char *state_file = "shadowclaw.bin";
    const char *ollama_endpoint = "http://localhost:11434";
    const char *ollama_model = "qwen2.5:0.5b";  // change as needed
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

-The whole arena (header + payload) is written to `shadowclaw.bin` with `fwrite`. 

-On startup, if the file exists and has a valid magic number, it is loaded back.

-All conversations and tool results are automatically saved to `shadowclaw.bin` and reloaded on restart.

---

# Updated 3/3/2026: Shadowclaw v1.3

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

## Build

```bash
cd /Home/User/Shadowclaw        # The folder you have the files located.
```

```bash
make clean && make
```

## Run

```bash
./shadowclaw
```

Use the slash commands above to get started – even without Ollama running, you can explore the built‑in features.

When your agent is running you should be able to use /help and see (example):

```bash
┌──(kali㉿user)-[~/shadowclaw]
└─$ ./shadowclaw
ShadowClaw ready. Type your message (Ctrl-D to exit)
/help
Shadowclaw commands:
  /help       Show this help
  /tools      List available tools
  /state      Show arena memory stats
  /clear      Clear conversation history (keeps system prompt)
  /chat       Remind you that chat mode is active
  /exit       Exit Shadowclaw
```

## Notes

-If Ollama is not running, you’ll see LLM call failed.

-Tool arguments can be a JSON array – they will be joined with spaces (useful if your model outputs "args":["arg1","arg2"]).

-All conversations and tool results are saved in shadowclaw.bin and reloaded on restart.

---

## How Tool Arguments Work

Shadowclaw processes tool calls by parsing JSON, where the args value is passed as a string to the function, executing it, and appending the result as a kind 5 blob for the next prompt. It supports a fallback where the model outputs args as an array (e.g., [arg1, arg2]), allowing for flexible input handling when standard JSON encoding fails. 

-Tool Call Flow: The agent parses the JSON block, executes the tool, and appends the result.

-Result Visibility: The tool output is visible to the model in the next prompt.

-Fallback Mechanism: If JSON parsing fails, Shadowclaw interprets args as an array.

-Alternative Tooling: In some scenarios, tool calls can be handled as a single string to avoid parsing errors. 

This system helps in handling malformed JSON, which occasionally occurs with certain LLM providers, ensuring the tool call succeeds.

```json
{"tool":"write_file","args":["notes.txt","Hello world"]}
```

Shadowclaw automatically **joins the array elements with spaces**, so the tool receives a single string `"notes.txt Hello world"`. This makes the agent robust to different model behaviours to ensure:

- **Simplicity** – most tools only need a single string argument (a command, a filename, a URL).  
- **Flexibility** – if a tool needs multiple pieces of information (like `write_file` needing a filename **and** content), we use a simple delimiter (newline). The LLM can learn this pattern from the system prompt.  
- **Lightweight Design** – no complex argument schemas, no extra JSON nesting. The entire tool call is tiny.
- **Args are always a single string** to keep things simple.  
- **Multiple values are encoded with delimiters** (like newline) – the tool and LLM agree on the format.  
- **Shadowclaw’s niche** is minimalism: a single binary with persistent memory, running anywhere, with just enough tools to be genuinely useful.  

You can easily add new tools by editing the `tools` array – each new function opens up more automation possibilities for your unique environment.

---

## Niche Use Cases

Shadowclaw is designed for **minimal, self‑contained AI agents** running on resource‑constrained or air‑gapped systems.  
Its toolset is deliberately small but powerful enough to automate many tasks without spawning heavy processes.

| Tool | What it does | Example args | Unique Niche Use |
|------|--------------|--------------|------------------|
| `shell` | Executes any shell command | `"ls -la /home"` | Automate system maintenance, run scripts, control services on a headless Raspberry Pi or embedded Linux device. |
| `read_file` | Reads a file from disk | `"/etc/passwd"` | Inspect configuration files, read logs, retrieve data for the LLM to analyse – all without a web interface. |
| `write_file` | Writes content to a file (args: `filename\ncontent`) | `"notes.txt\nBuy milk"` | Create notes, write configuration files, save results – perfect for offline data logging. |
| `http_get` | Fetches a URL (via libcurl) | `"https://api.example.com/data"` | Retrieve weather, fetch RSS feeds, call local REST APIs – even on devices with no browser, just a network stack. |
| `math` | Evaluates an expression using `bc` | `"2 + 2 * 5"` | Let the LLM do arithmetic, unit conversions, or simple calculations without relying on external tools. |
| `list_dir` | Lists directory contents | `"/home/user"` | Explore filesystem, find documents, check available storage – all natively, without forking a shell. |

---

## Niche Use Case Examples:

### 1. Raspberry Pi Zero / IoT Sensor Node
- **Tool used:** `shell` + `write_file`  
- **Flow:** LLM asks to read a temperature sensor via a shell script, then logs the value to a file.  
- **Why Shadowclaw?** Runs in <200KB RAM, no Python, no bloat. Persistent memory keeps the last readings.

### 2. Air‑Gapped Engineering Workstation
- **Tool used:** `http_get` + `math` + `read_file`  
- **Flow:** Engineer asks for a component value calculation; LLM fetches a local datasheet via HTTP, reads a config file, does the math, and writes a report.  
- **Why Shadowclaw?** No cloud, no internet required. All data stays on the machine.

### 3. Embedded Router Automation
- **Tool used:** `shell` + `list_dir`  
- **Flow:** Network admin asks for a list of active interfaces; LLM runs `ifconfig` and parses the output, then suggests a config change.  
- **Why Shadowclaw?** Tiny binary fits in router storage (often just a few MB). Uses curl for local API calls.

### 4. Low‑Power Field Logger
- **Tool used:** `write_file` + `math`  
- **Flow:** A solar‑powered device collects environmental data; Shadowclaw can process and store it locally, then answer queries about trends.  
- **Why Shadowclaw?** No database needed – just a binary and a single file (`shadowclaw.bin`) for persistence.

---

## Credits

- **Tsoding** – for the “insane shadow data trick” (the header‑before‑data arena idea).
- **cJSON** – Dave Gamble and contributors – the minimal JSON parser used.
- **Ollama** – HTTP requests via curl.
- **Vibe Code** - xAI Grok 4.20 and Deepseek AI
- **Openclaw**
- **webXOS**
---

## License

This project is released under the open sourced MIT License.  
cJSON is also MIT licensed.
