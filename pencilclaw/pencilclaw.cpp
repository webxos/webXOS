// pencilclaw.cpp – Command & Control loop (fixed version)
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <memory>
#include <cstring>
#include <curl/curl.h>
#include <filesystem>   // for temporary file cleanup

#include "pencil_utils.hpp"
#include "cJSON.h"

// Global debug flag (can be toggled at runtime)
static bool debug_enabled = false;

// ----------------------------------------------------------------------
// Helper: libcurl write callback
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *output) {
    size_t total = size * nmemb;
    output->append((char*)contents, total);
    return total;
}

// ----------------------------------------------------------------------
// Send prompt to Ollama, return the generated text
std::string ask_ollama(const std::string &prompt) {
    CURL *curl = curl_easy_init();
    if (!curl) {
        return "[Error] Failed to initialize curl.";
    }

    // Model name – change this to match your installed model (e.g., "llama3", "qwen2.5", "mistral")
    const std::string MODEL_NAME = "qwen2.5:0.5b";

    // Build JSON request
    cJSON *root = cJSON_CreateObject();
    if (!root) {
        curl_easy_cleanup(curl);
        return "[Error] Failed to create JSON object (out of memory).";
    }
    cJSON_AddStringToObject(root, "model", MODEL_NAME.c_str());
    cJSON_AddStringToObject(root, "prompt", prompt.c_str());
    cJSON_AddBoolToObject(root, "stream", false);
    char *json_str = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);

    if (!json_str) {
        curl_easy_cleanup(curl);
        return "[Error] Failed to format JSON request.";
    }

    if (debug_enabled) {
        std::cerr << "\n[DEBUG] Request JSON: " << json_str << std::endl;
    }

    std::string response_string;
    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:11434/api/generate");
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_str);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L); // increased timeout to 60 seconds

    // Perform request
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::string err = "[Error] curl failed: ";
        err += curl_easy_strerror(res);
        free(json_str);
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return err;
    }

    // Check HTTP response code
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    if (http_code != 200) {
        std::string err = "[Error] HTTP " + std::to_string(http_code) + " response from Ollama";
        if (http_code == 404) {
            err += ".\n       Make sure Ollama is running and the model '" + MODEL_NAME + "' is installed (try 'ollama pull " + MODEL_NAME + "').";
        }
        free(json_str);
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return err;
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    free(json_str);

    if (debug_enabled) {
        std::cerr << "[DEBUG] Raw response: " << response_string << std::endl;
    }

    // Parse JSON response
    cJSON *json = cJSON_Parse(response_string.c_str());
    if (!json) {
        return "[Error] Failed to parse Ollama JSON (invalid JSON).";
    }
    cJSON *resp = cJSON_GetObjectItem(json, "response");
    std::string result;
    if (resp && resp->valuestring) {
        result = resp->valuestring;
    } else {
        // Check for error field
        cJSON *error = cJSON_GetObjectItem(json, "error");
        if (error && error->valuestring) {
            result = "[Error from Ollama] " + std::string(error->valuestring);
        } else {
            result = "[Error] No 'response' field in Ollama output.";
        }
    }
    cJSON_Delete(json);
    return result;
}

// ----------------------------------------------------------------------
// Extract code blocks marked with ```lang ... ```
std::vector<std::string> extract_code_blocks(const std::string &text) {
    std::vector<std::string> blocks;
    size_t pos = 0;
    while (true) {
        size_t start = text.find("```", pos);
        if (start == std::string::npos) break;
        size_t end = text.find("```", start + 3);
        if (end == std::string::npos) break;
        
        // Determine content start: skip the language specifier line if present
        size_t content_start = text.find('\n', start) + 1;
        if (content_start == std::string::npos || content_start > end)
            content_start = start + 3;
        else {
            // content_start now points after the newline; but if the line after ``` is empty, we may need to skip more?
            // Simpler: just take everything after the first newline until the closing ```
            // Already correct.
        }
        std::string block = text.substr(content_start, end - content_start);
        blocks.push_back(block);
        pos = end + 3;
    }
    return blocks;
}

// ----------------------------------------------------------------------
// Execute a code block: save to temp file, compile, run
bool execute_code(const std::string &code) {
    // Save to a temporary file
    std::string tmp_cpp = pencil::PENCIL_DIR + "temp_code.cpp";
    std::string tmp_exe = pencil::PENCIL_DIR + "temp_code";
    std::ofstream out(tmp_cpp);
    if (!out) {
        std::cerr << "Failed to create temporary file: " << tmp_cpp << std::endl;
        return false;
    }
    out << code;
    out.close();
    if (out.fail()) {
        std::cerr << "Failed to write code to temporary file." << std::endl;
        return false;
    }

    // Compile with g++
    std::string compile_cmd = "g++ " + tmp_cpp + " -o " + tmp_exe + " 2>&1";
    FILE *pipe = popen(compile_cmd.c_str(), "r");
    if (!pipe) {
        std::cerr << "Failed to run compiler." << std::endl;
        return false;
    }
    char buffer[128];
    std::string compile_out;
    while (fgets(buffer, sizeof buffer, pipe) != nullptr) {
        compile_out += buffer;
    }
    int status = pclose(pipe);
    if (status != 0) {
        std::cerr << "Compilation failed:\n" << compile_out << std::endl;
        // Clean up source file even on failure
        std::filesystem::remove(tmp_cpp);
        return false;
    }

    // Run the compiled program
    std::string run_cmd = tmp_exe;
    int ret = system(run_cmd.c_str());
    std::cout << "\n[Program exited with code " << ret << "]" << std::endl;

    // Clean up temporary files
    std::filesystem::remove(tmp_cpp);
    std::filesystem::remove(tmp_exe);
    return true;
}

// ----------------------------------------------------------------------
// Sanitize user input for use as a filename: remove path separators and ".."
std::string sanitize_filename(const std::string &input) {
    std::string safe;
    for (char c : input) {
        // Allow alphanumerics, dot, dash, underscore; replace others with underscore
        if (isalnum(c) || c == '.' || c == '-' || c == '_')
            safe += c;
        else
            safe += '_';
    }
    // Prevent empty or dot-only names
    if (safe.empty() || safe == "." || safe == "..")
        safe = "unnamed";
    return safe;
}

// ----------------------------------------------------------------------
int main() {
    // Prepare workspace
    if (!pencil::init_workspace()) {
        std::cerr << "Fatal error: cannot create workspace directory." << std::endl;
        return 1;
    }

    std::cout << "PENCILCLAW v1.1 – ADA‑style writing agent (local Ollama)\n";
    std::cout << "Type /HELP for commands.\n";

    std::string last_response;   // store last LLM output for /EXECUTE

    while (true) {
        std::cout << "\n> ";
        std::string line;
        std::getline(std::cin, line);
        if (line.empty()) continue;

        if (line[0] == '/') {
            // ADA command
            std::string cmd;
            std::string arg;
            size_t sp = line.find(' ');
            if (sp == std::string::npos) {
                cmd = line;
            } else {
                cmd = line.substr(0, sp);
                arg = line.substr(sp + 1);
            }

            if (cmd == "/EXIT") {
                break;
            }
            else if (cmd == "/HELP") {
                std::cout << "Available commands:\n";
                std::cout << "  /HELP                 – this help\n";
                std::cout << "  /STORY   <title>      – write a story\n";
                std::cout << "  /POEM    <subject>    – compose a poem\n";
                std::cout << "  /BOOK    <chapter>    – write a chapter (appends to book.txt)\n";
                std::cout << "  /EXECUTE              – compile & run code from last response\n";
                std::cout << "  /DEBUG                – toggle debug output\n";
                std::cout << "  /EXIT                 – quit\n";
            }
            else if (cmd == "/DEBUG") {
                debug_enabled = !debug_enabled;
                std::cout << "Debug mode " << (debug_enabled ? "enabled" : "disabled") << ".\n";
            }
            else if (cmd == "/STORY" || cmd == "/POEM") {
                if (arg.empty()) {
                    std::cout << "Please provide a " << (cmd == "/STORY" ? "title" : "subject") << ".\n";
                    continue;
                }
                // Build prompt with template
                std::string prompt = pencil::get_template(cmd) + "\n\n" + arg;
                std::cout << "Asking Ollama...\n";
                last_response = ask_ollama(prompt);
                std::cout << last_response << "\n";

                // Save to a file using sanitized filename
                std::string safe_arg = sanitize_filename(arg);
                std::string filename = pencil::PENCIL_DIR + safe_arg + ".txt";
                if (!pencil::save_text(filename, last_response))
                    std::cerr << "Warning: could not save file " << filename << std::endl;
                pencil::append_to_session("User: " + line);
                pencil::append_to_session("Assistant: " + last_response);
            }
            else if (cmd == "/BOOK") {
                if (arg.empty()) {
                    std::cout << "Please provide a chapter name.\n";
                    continue;
                }
                // Build context: previous book content
                std::string book_content = pencil::read_file(pencil::PENCIL_DIR + pencil::BOOK_FILE);
                std::string prompt = pencil::get_template("/BOOK") + "\n\nExisting book content:\n" +
                                     book_content + "\n\nWrite the next chapter: " + arg;
                std::cout << "Asking Ollama...\n";
                last_response = ask_ollama(prompt);
                std::cout << last_response << "\n";

                // Append chapter to book.txt
                if (!pencil::append_to_book("\n--- " + arg + " ---\n" + last_response))
                    std::cerr << "Warning: could not append to book file." << std::endl;
                pencil::append_to_session("User: " + line);
                pencil::append_to_session("Assistant: " + last_response);
            }
            else if (cmd == "/EXECUTE") {
                if (last_response.empty()) {
                    std::cout << "No previous response to execute from.\n";
                    continue;
                }
                auto blocks = extract_code_blocks(last_response);
                if (blocks.empty()) {
                    std::cout << "No code blocks found in last response.\n";
                    continue;
                }
                // Optional security confirmation
                std::cout << "WARNING: You are about to execute code generated by an AI.\n";
                std::cout << "Only proceed if you trust the source. Continue? (y/n): ";
                std::string confirm;
                std::getline(std::cin, confirm);
                if (confirm != "y" && confirm != "Y") {
                    std::cout << "Execution cancelled.\n";
                    continue;
                }
                // Execute the first block
                std::cout << "Executing code block...\n";
                if (execute_code(blocks[0])) {
                    std::cout << "Execution finished.\n";
                } else {
                    std::cout << "Execution failed.\n";
                }
            }
            else {
                std::cout << "Unknown command. Type /HELP for list.\n";
            }
        } else {
            // Non‑command: treat as free prompt
            std::cout << "Sending to Ollama...\n";
            last_response = ask_ollama(line);
            std::cout << last_response << "\n";
            pencil::append_to_session("User: " + line);
            pencil::append_to_session("Assistant: " + last_response);
        }
    }

    return 0;
}
