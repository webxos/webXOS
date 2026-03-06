# Pencilclaw v1.0 (Testing) ✏️
---
```
██████╗ ███████╗███╗   ██╗ ██████╗██╗██╗     ██████╗██╗      █████╗ ██╗    ██╗
██╔══██╗██╔════╝████╗  ██║██╔════╝██║██║    ██╔════╝██║     ██╔══██╗██║    ██║
██████╔╝█████╗  ██╔██╗ ██║██║     ██║██║    ██║     ██║     ███████║██║ █╗ ██║
██╔═══╝ ██╔══╝  ██║╚██╗██║██║     ██║██║    ██║     ██║     ██╔══██║██║███╗██║
██║     ███████╗██║ ╚████║╚██████╗██║███████╗╚██████╗██████╗██║  ██║╚███╔███╔╝
╚═╝     ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝╚══════╝ ╚═════╝╚═════╝╚═╝  ╚═╝ ╚══╝╚══╝ 
```

**PENCILCLAW** is a C++ command-line agent harness that turns your local [Ollama](https://ollama.com/) instance into a creative writing partner with the ability to execute generated C++ code. It follows a simple ADA-style command interface - perfect for writers, tinkerers, and AI enthusiasts who want to keep their data private and their workflows offline.

---

## Features

- **Story & Poem Generation** - Use `/STORY` or `/POEM` with a title/subject to get creative text from your local LLM.
- **Book Continuation** - The `/BOOK` command appends new chapters to a running `book.txt`, maintaining context from previous content.
- **Code Execution** - If the AI responds with a C++ code block (triple backticks), `/EXECUTE` compiles and runs it - ideal for prototyping or exploring AI-generated algorithms.
- **Session Logging** - All interactions are saved in `pencil_data/session.log` for later reference.
- **Workspace Isolation** - Everything lives in the `./pencil_data/` folder; temporary files are cleaned up after execution.
- **Security Awareness** - Includes filename sanitisation and a confirmation prompt before running any AI-generated code.

---

## Project Structure

All necessary files for PENCILCLAW are contained within the `/home/kali/pencilclaw/` directory. Below is the complete tree:

```
/home/kali/pencilclaw/
├── pencilclaw.cpp          # Main program source
├── pencil_utils.hpp        # Workspace and template helpers
├── pencilclaw              # Compiled executable (after build)
└── pencil_data/            # **Created automatically on first run**
    ├── session.log         # Full interaction log
    ├── book.txt            # Accumulated book chapters
    ├── temp_code.cpp       # Temporary source file (deleted after execution)
    ├── temp_code           # Temporary executable (deleted after execution)
    └── [story/poem files]  # Individual .txt files for each /STORY or /POEM
```

**The `pencil_data` directory is created automatically when you run the program. All generated content and logs reside there.**

---

## Requirements

- **libcurl** development libraries
- **cJSON** library
- **Ollama** installed and running
- A model pulled in Ollama (default: `qwen2.5:0.5b` - change in source if desired)

---

## Installation

### 1. Install System Dependencies
```bash
sudo apt update
sudo apt install -y build-essential libcurl4-openssl-dev
```

### 2. Install cJSON
If your distribution does not provide a package, build from source:
```bash
git clone https://github.com/DaveGamble/cJSON.git
cd cJSON
mkdir build && cd build
cmake ..
make
sudo make install
sudo ldconfig
cd ../..
```

### 3. Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &   # start the service
ollama pull qwen2.5:0.5b   # or another model of your choice
```

## Custom Models

Edit line 36 of the pencilclaw.cpp file:

```
    // Model name – change this to match your installed model (e.g., "llama3", "qwen2.5", "mistral")
    const std::string MODEL_NAME = "qwen2.5:0.5b";
```

### 4. Compile PENCILCLAW
Place the source files in the same directory and compile:
```bash
g++ -std=c++17 -o pencilclaw pencilclaw.cpp -lcurl -lcjson
```
If cJSON headers are in a non-standard location (e.g., `/usr/local/include/cjson`), add the appropriate `-I` flag:
```bash
g++ -std=c++17 -o pencilclaw pencilclaw.cpp -lcurl -lcjson -I/usr/local/include/cjson
```

---

## Usage

Start the program:
```bash
./pencilclaw
```

You will see the `>` prompt. Commands are case-sensitive and start with `/`.

### Available Commands
| Command           | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `/HELP`           | Show this help message.                                                     |
| `/STORY <title>`  | Generate a short story with the given title. Saved as `<title>.txt`.       |
| `/POEM <subject>` | Compose a poem about the subject. Saved as `<subject>.txt`.                 |
| `/BOOK <chapter>` | Append a new chapter to `book.txt` (creates file if it doesn't exist).     |
| `/EXECUTE`        | Compile and run the first C++ code block from the last AI response.        |
| `/DEBUG`          | Toggle verbose debug output (shows JSON requests/responses).               |
| `/EXIT`           | Quit the program.                                                           |

Any line not starting with `/` is sent directly to Ollama as a free prompt; the response is displayed and logged.

---

## Security Notes

- **Code execution is a powerful feature.** PENCILCLAW asks for confirmation before running any AI-generated code. Always review the code if you are unsure.
- **Filename sanitisation** prevents path traversal attacks (e.g., `../../etc/passwd` becomes `____etc_passwd`).
- All operations are confined to the `pencil_data` subdirectory; no system-wide changes are made.

---

## Customisation

- **Model**: Change the `MODEL_NAME` constant in `pencilclaw.cpp` to use a different Ollama model.
- **Prompts**: Edit the templates in `pencil_utils.hpp` (`get_template` function) to adjust the AI's behaviour.
- **Timeout**: The default HTTP timeout is 60 seconds. Adjust `CURLOPT_TIMEOUT` in the source if needed.

---

## Troubleshooting

| Problem                          | Solution                                                       |
|----------------------------------|----------------------------------------------------------------|
| `cJSON.h: No such file or directory` | Install cJSON or add the correct `-I` flag during compilation. |
| `curl failed: Timeout was reached`   | Ensure Ollama is running (`ollama serve`) and the model is pulled. |
| Model not found                  | Run `ollama pull <model_name>` (e.g., `qwen2.5:0.5b`).         |
| Compilation errors (C++17)       | Use a compiler that supports `-std=c++17` (g++ 7+ or clang 5+).|

---

## License

This project is released under the MIT License. Built with C++ and Ollama.
