// pencilclaw.cpp – C++ coding agent with autonomous task mode and Git integration
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
#include <filesystem>
#include <chrono>
#include <thread>
#include <ctime>
#include <iomanip>
#include <cstdio>
#include <algorithm>
#include <cctype>
#include <optional>
#include <unistd.h>
#include <sys/wait.h>
#include <nlohmann/json.hpp>

#include "pencil_utils.hpp"

using json = nlohmann::json;

// Global debug flag
static bool debug_enabled = false;

// ----------------------------------------------------------------------
// Keep‑alive and heartbeat timing
static time_t last_ollama_time = 0;
const int KEEP_ALIVE_INTERVAL = 120;
const int HEARTBEAT_INTERVAL = 120;

// ----------------------------------------------------------------------
// Last AI output (for saving, executing, etc.)
static std::string last_ai_output;
static std::string last_ai_type;   // "code", "task_iteration", "free"

// ----------------------------------------------------------------------
// RAII wrapper for libcurl (with move semantics)
class CurlRequest {
    CURL* curl;
    struct curl_slist* headers;
    std::string response;
    void cleanup() {
        if (headers) curl_slist_free_all(headers);
        if (curl) curl_easy_cleanup(curl);
    }
public:
    CurlRequest() : curl(curl_easy_init()), headers(nullptr) {}
    ~CurlRequest() { cleanup(); }
    CurlRequest(const CurlRequest&) = delete;
    CurlRequest& operator=(const CurlRequest&) = delete;
    CurlRequest(CurlRequest&& other) noexcept
        : curl(std::exchange(other.curl, nullptr)),
          headers(std::exchange(other.headers, nullptr)),
          response(std::move(other.response)) {}
    CurlRequest& operator=(CurlRequest&& other) noexcept {
        if (this != &other) {
            cleanup();
            curl = std::exchange(other.curl, nullptr);
            headers = std::exchange(other.headers, nullptr);
            response = std::move(other.response);
        }
        return *this;
    }

    bool perform(const std::string& url, const std::string& postdata) {
        if (!curl) return false;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postdata.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            response = "[Error] curl failed: " + std::string(curl_easy_strerror(res));
            return false;
        }
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code != 200) {
            response = "[Error] HTTP " + std::to_string(http_code);
            return false;
        }
        return true;
    }
    const std::string& get_response() const { return response; }
    static size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *output) {
        size_t total = size * nmemb;
        output->append((char*)contents, total);
        return total;
    }
};

// ----------------------------------------------------------------------
// Forward declarations
std::string ask_ollama(const std::string &prompt);
std::string ask_ollama_with_retry(const std::string& prompt, int max_retries = 3);
void check_and_keep_alive(time_t now);
void warm_up_ollama();

// ----------------------------------------------------------------------
// Get model name from environment or default
std::string get_model_name() {
    const char* env = std::getenv("OLLAMA_MODEL");
    return env ? env : "qwen2.5:0.5b";
}

// ----------------------------------------------------------------------
// Send prompt to Ollama, return the generated text or an error string.
std::string ask_ollama(const std::string &prompt) {
    json request = {
        {"model", get_model_name()},
        {"prompt", prompt},
        {"stream", false}
    };
    std::string request_str = request.dump();

    if (debug_enabled) {
        std::cerr << "\n[DEBUG] Request JSON: " << request_str << std::endl;
    }

    CurlRequest req;
    if (!req.perform("http://localhost:11434/api/generate", request_str)) {
        return req.get_response(); // contains error message
    }

    std::string response_string = req.get_response();
    if (debug_enabled) {
        std::cerr << "[DEBUG] Raw response: " << response_string << std::endl;
    }

    try {
        auto response = json::parse(response_string);
        if (response.contains("response")) {
            return response["response"].get<std::string>();
        } else if (response.contains("error")) {
            return "[Error from Ollama] " + response["error"].get<std::string>();
        } else {
            return "[Error] No 'response' field in Ollama output.";
        }
    } catch (const json::parse_error& e) {
        return "[Error] Failed to parse Ollama JSON: " + std::string(e.what());
    }
}

// ----------------------------------------------------------------------
// Wrapper with retry logic for timeouts
std::string ask_ollama_with_retry(const std::string& prompt, int max_retries) {
    int attempt = 0;
    int base_delay = 2;
    while (attempt < max_retries) {
        std::string result = ask_ollama(prompt);
        if (result.compare(0, 9, "[Timeout]") == 0) {
            attempt++;
            if (attempt < max_retries) {
                int delay = base_delay * (1 << (attempt - 1));
                std::cerr << "Timeout, retrying in " << delay << " seconds...\n";
                std::this_thread::sleep_for(std::chrono::seconds(delay));
                continue;
            } else {
                return "[Error] Maximum retries reached, giving up.";
            }
        }
        return result;
    }
    return "[Error] Maximum retries reached, giving up.";
}

// ----------------------------------------------------------------------
// Keep‑alive
void check_and_keep_alive(time_t now) {
    if (now - last_ollama_time > KEEP_ALIVE_INTERVAL) {
        if (debug_enabled) std::cout << "[Keep alive] Sending ping to Ollama.\n";
        ask_ollama("Hello");
        last_ollama_time = now;
    }
}

// ----------------------------------------------------------------------
// Warm up
void warm_up_ollama() {
    std::cout << "Warming up Ollama model..." << std::endl;
    std::string result = ask_ollama("Hello");
    if (result.compare(0, 7, "[Error]") == 0 || result.compare(0, 9, "[Timeout]") == 0) {
        std::cerr << "Warning: Warm-up failed: " << result << std::endl;
        std::cerr << "Check that Ollama is running and the model is available.\n";
    } else {
        std::cout << "Model ready.\n";
    }
}

// ----------------------------------------------------------------------
// Safe command execution (no shell) – returns stdout + status
struct CommandResult {
    std::string output;
    int exit_status;
};
CommandResult run_command(const std::vector<std::string>& args) {
    if (args.empty()) return {"", -1};
    std::vector<char*> argv;
    for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);

    int pipefd[2];
    if (pipe(pipefd) == -1) return {"pipe() failed", -1};

    pid_t pid = fork();
    if (pid == -1) {
        close(pipefd[0]);
        close(pipefd[1]);
        return {"fork() failed", -1};
    }

    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        execvp(argv[0], argv.data());
        perror("execvp");
        _exit(127);
    }

    close(pipefd[1]);
    std::string output;
    char buffer[4096];
    ssize_t n;
    while ((n = read(pipefd[0], buffer, sizeof(buffer)-1)) > 0) {
        buffer[n] = '\0';
        output += buffer;
    }
    close(pipefd[0]);

    int status;
    waitpid(pid, &status, 0);
    int exit_status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    return {output, exit_status};
}

// ----------------------------------------------------------------------
// Git helper functions (safe, no shell)
bool is_git_repo() {
    return std::filesystem::exists(pencil::get_pencil_dir() + ".git");
}

bool init_git_repo() {
    if (is_git_repo()) return true;
    auto res = run_command({"git", "-C", pencil::get_pencil_dir(), "init"});
    if (res.exit_status != 0) return false;
    // Set local identity so commits don't fail
    run_command({"git", "-C", pencil::get_pencil_dir(), "config", "user.email", "pencilclaw@local"});
    run_command({"git", "-C", pencil::get_pencil_dir(), "config", "user.name", "PencilClaw"});
    return true;
}

// Run a git command with arguments, return output and status
CommandResult git_command(const std::vector<std::string>& args) {
    std::vector<std::string> cmd = {"git", "-C", pencil::get_pencil_dir()};
    cmd.insert(cmd.end(), args.begin(), args.end());
    return run_command(cmd);
}

bool git_commit_file(const std::string& file_path, const std::string& commit_message) {
    std::filesystem::path full_path(file_path);
    std::string rel_path = std::filesystem::relative(full_path, pencil::get_pencil_dir()).string();

    // git add
    auto add_res = git_command({"add", rel_path});
    if (add_res.exit_status != 0) {
        std::cerr << "Git add failed: " << add_res.output << std::endl;
        return false;
    }

    // git commit
    auto commit_res = git_command({"commit", "-m", commit_message});
    if (commit_res.exit_status != 0) {
        // It's okay if "nothing to commit" – check output
        if (commit_res.output.find("nothing to commit") == std::string::npos &&
            commit_res.output.find("no changes added") == std::string::npos) {
            std::cerr << "Git commit failed: " << commit_res.output << std::endl;
            return false;
        }
    }
    if (debug_enabled) std::cerr << "[Git] " << commit_res.output << std::endl;
    return true;
}

// ----------------------------------------------------------------------
// Extract code blocks
std::vector<std::string> extract_code_blocks(const std::string &text) {
    std::vector<std::string> blocks;
    size_t pos = 0;
    while (true) {
        size_t start = text.find("```", pos);
        if (start == std::string::npos) break;
        size_t end = text.find("```", start + 3);
        if (end == std::string::npos) break;

        size_t nl = text.find('\n', start);
        size_t content_start;
        if (nl != std::string::npos && nl < end) {
            content_start = nl + 1;
        } else {
            content_start = start + 3;
        }

        std::string block = text.substr(content_start, end - content_start);
        blocks.push_back(block);
        pos = end + 3;
    }
    return blocks;
}

// ----------------------------------------------------------------------
// Execute code (compiles and runs C++)
bool execute_code(const std::string &code) {
    std::string tmp_cpp = pencil::get_pencil_dir() + "temp_code.cpp";
    std::string tmp_exe = pencil::get_pencil_dir() + "temp_code";

    if (!pencil::save_text(tmp_cpp, code)) {
        std::cerr << "Failed to write code to temporary file." << std::endl;
        return false;
    }

    auto compile_res = run_command({"g++", "-o", tmp_exe, tmp_cpp});
    if (compile_res.exit_status != 0) {
        std::cerr << "Compilation failed:\n" << compile_res.output << std::endl;
        std::filesystem::remove(tmp_cpp);
        return false;
    }

    auto run_res = run_command({tmp_exe});
    std::cout << "\n[Program exited with code " << run_res.exit_status << "]\n";
    std::cout << run_res.output << std::endl;

    std::filesystem::remove(tmp_cpp);
    std::filesystem::remove(tmp_exe);
    return true;
}

// ----------------------------------------------------------------------
// Secure filename sanitization (using canonical)
std::string sanitize_and_secure_path(const std::string &input, const std::string &subdir = "") {
    std::error_code ec;
    std::filesystem::path base = std::filesystem::canonical(pencil::get_pencil_dir(), ec);
    if (ec) {
        std::cerr << "Error: Cannot resolve base directory.\n";
        return "";
    }
    if (!subdir.empty()) base /= subdir;

    // Construct a safe filename: keep only alphanumeric, dot, dash, underscore
    std::string safe_name;
    for (char c : input) {
        if (isalnum(c) || c == '.' || c == '-' || c == '_')
            safe_name += c;
        else
            safe_name += '_';
    }
    if (safe_name.empty() || safe_name == "." || safe_name == "..")
        safe_name = "unnamed";

    std::filesystem::path full = base / safe_name;
    std::filesystem::path resolved = std::filesystem::canonical(full, ec);
    if (ec) {
        // Path may not exist yet; use absolute and check prefix manually
        std::string abs_full = std::filesystem::absolute(full).string();
        std::string base_str = base.string();
        if (abs_full.compare(0, base_str.size(), base_str) != 0 ||
            (abs_full.size() > base_str.size() && abs_full[base_str.size()] != '/')) {
            return "";
        }
        return abs_full;
    }

    std::string resolved_str = resolved.string();
    std::string base_str = base.string();
    if (resolved_str.compare(0, base_str.size(), base_str) != 0 ||
        (resolved_str.size() > base_str.size() && resolved_str[base_str.size()] != '/')) {
        return "";
    }
    return resolved_str;
}

// ----------------------------------------------------------------------
// Save content with verification and Git commit
bool save_content_to_file(const std::string& content, const std::string& filename, const std::string& description) {
    std::string safe_path = sanitize_and_secure_path(filename);
    if (safe_path.empty()) {
        std::cerr << "Error: Invalid or insecure filename." << std::endl;
        return false;
    }

    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(safe_path).parent_path(), ec);
    if (ec) {
        std::cerr << "Error creating directory: " << ec.message() << std::endl;
        return false;
    }

    if (!pencil::save_text(safe_path, content)) {
        std::cerr << "Error: Failed to write file " << safe_path << std::endl;
        return false;
    }

    if (!std::filesystem::exists(safe_path)) {
        std::cerr << "Error: File " << safe_path << " does not exist after save." << std::endl;
        return false;
    }
    auto size = std::filesystem::file_size(safe_path);
    if (size == 0) {
        std::cerr << "Error: File " << safe_path << " is empty." << std::endl;
        return false;
    }

    std::cout << "✅ Saved " << description << " to: " << safe_path << " (" << size << " bytes)" << std::endl;

    // Git commit if repository is active
    if (is_git_repo()) {
        std::string commit_msg = description;
        if (commit_msg.length() > 100) commit_msg = commit_msg.substr(0, 100) + "...";
        if (!git_commit_file(safe_path, commit_msg)) {
            std::cerr << "Warning: Git commit failed (check your Git configuration).\n";
        }
    }
    return true;
}

// ----------------------------------------------------------------------
// Task management
std::string get_active_task_folder() {
    std::ifstream f(pencil::get_active_task_file());
    std::string folder;
    std::getline(f, folder);
    if (folder.empty()) return "";

    std::error_code ec;
    std::filesystem::path p = std::filesystem::weakly_canonical(folder, ec);
    if (ec) return "";

    std::string tasks_dir_canon = std::filesystem::weakly_canonical(pencil::get_tasks_dir()).string();
    std::string p_str = p.string();
    if (p_str.compare(0, tasks_dir_canon.size(), tasks_dir_canon) != 0 ||
        (p_str.size() > tasks_dir_canon.size() && p_str[tasks_dir_canon.size()] != '/')) {
        return "";
    }
    return p_str;
}

bool set_active_task_folder(const std::string& folder) {
    std::ofstream f(pencil::get_active_task_file());
    if (!f) return false;
    f << folder;
    return !f.fail();
}

void clear_active_task() {
    std::filesystem::remove(pencil::get_active_task_file());
}

bool start_new_task(const std::string& description) {
    // Create a folder with timestamp and sanitized description prefix
    std::string safe_desc;
    for (char c : description) {
        if (isalnum(c) || c == ' ' || c == '-') safe_desc += c;
        else safe_desc += '_';
    }
    if (safe_desc.length() > 30) safe_desc = safe_desc.substr(0, 30);
    std::string folder_name = pencil::timestamp() + "_" + safe_desc;
    std::string task_folder = pencil::get_tasks_dir() + folder_name + "/";

    std::error_code ec;
    if (!std::filesystem::create_directories(task_folder, ec) && ec) {
        std::cerr << "Failed to create task folder: " << ec.message() << std::endl;
        return false;
    }

    // Save description
    if (!pencil::save_text(task_folder + "description.txt", description)) {
        std::cerr << "Failed to save task description.\n";
        return false;
    }

    // Create log file with initial entry
    std::string log_entry = "Task started at " + pencil::timestamp() + "\nDescription: " + description + "\n\n";
    if (!pencil::save_text(task_folder + "log.txt", log_entry)) {
        std::cerr << "Failed to create log file.\n";
        return false;
    }

    if (!set_active_task_folder(task_folder)) {
        std::cerr << "Warning: Could not set active task.\n";
    } else {
        std::cout << "✅ New task started: \"" << description << "\"\n";
        std::cout << "Task folder: " << task_folder << "\n";
    }

    pencil::append_to_session("Started new task: " + description);
    return true;
}

bool continue_task(const std::string& task_folder) {
    // Read description and log
    auto desc_opt = pencil::read_file(task_folder + "description.txt");
    if (!desc_opt.has_value()) {
        std::cerr << "Task description missing.\n";
        return false;
    }
    std::string description = desc_opt.value();

    auto log_opt = pencil::read_file(task_folder + "log.txt");
    std::string log = log_opt.value_or("");

    // Determine iteration number: count occurrences of "Iteration" in log
    int iteration = 1;
    size_t pos = 0;
    while ((pos = log.find("Iteration", pos)) != std::string::npos) {
        iteration++;
        pos += 9;
    }

    // Build prompt for next step
    std::string prompt = "You are a C++ coding agent working on the following task:\n\n" +
                         description + "\n\n" +
                         "Previous work log:\n" + log + "\n\n" +
                         "Generate the next iteration of code or progress. If the task is not yet complete, "
                         "produce a new C++ code snippet that advances the work. If the task is complete, "
                         "output a message indicating completion and include no code.\n\n"
                         "Provide your response with optional explanation, but include any code inside ```cpp ... ``` blocks.";

    std::cout << "Continuing task (iteration " << iteration << ")...\n";
    std::string response = ask_ollama_with_retry(prompt);
    if (response.compare(0, 7, "[Error]") == 0) {
        std::cerr << "Failed to generate continuation: " << response << std::endl;
        return false;
    }

    // Save this iteration
    std::string iter_file = task_folder + "iteration_" + std::to_string(iteration) + ".txt";
    if (!pencil::save_text(iter_file, response)) {
        std::cerr << "Failed to save iteration.\n";
        return false;
    }

    // Append to log
    std::ofstream log_file(task_folder + "log.txt", std::ios::app);
    if (log_file) {
        log_file << "\n--- Iteration " << iteration << " (" << pencil::timestamp() << ") ---\n";
        log_file << response << "\n";
    }

    std::cout << "✅ Iteration " << iteration << " saved to: " << iter_file << "\n";
    last_ai_output = response;
    last_ai_type = "task_iteration";
    pencil::append_to_session("Task continued: iteration " + std::to_string(iteration));

    // Git commit if repository is active
    if (is_git_repo()) {
        std::string commit_msg = "Task iteration " + std::to_string(iteration) + ": " + description;
        if (commit_msg.length() > 100) commit_msg = commit_msg.substr(0, 100) + "...";
        if (!git_commit_file(iter_file, commit_msg)) {
            std::cerr << "Warning: Git commit failed.\n";
        }
    }
    return true;
}

// ----------------------------------------------------------------------
// Heartbeat
void run_heartbeat(time_t now) {
    check_and_keep_alive(now);
    std::string active_task = get_active_task_folder();
    if (!active_task.empty()) {
        if (debug_enabled) std::cout << "[Heartbeat] Continuing active task.\n";
        continue_task(active_task);
    }
}

// ----------------------------------------------------------------------
// Natural language helpers
std::string to_lowercase(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
    return s;
}

bool contains_phrase(const std::string& text, const std::string& phrase) {
    std::string lower = to_lowercase(text);
    std::string lower_phrase = to_lowercase(phrase);
    size_t pos = lower.find(lower_phrase);
    while (pos != std::string::npos) {
        if ((pos == 0 || !isalnum(lower[pos-1])) &&
            (pos + lower_phrase.length() == lower.length() || !isalnum(lower[pos + lower_phrase.length()]))) {
            return true;
        }
        pos = lower.find(lower_phrase, pos + 1);
    }
    return false;
}

std::string extract_after(const std::string& input, const std::string& phrase) {
    std::string lower_input = to_lowercase(input);
    std::string lower_phrase = to_lowercase(phrase);
    size_t pos = lower_input.find(lower_phrase);
    if (pos == std::string::npos) return "";
    if (pos > 0 && isalnum(lower_input[pos-1])) return "";
    size_t after = pos + phrase.length();
    if (after < lower_input.length() && isalnum(lower_input[after])) return "";
    size_t start = after;
    while (start < input.length() && isspace(input[start])) ++start;
    std::string result = input.substr(start);
    while (!result.empty() && isspace(result.back())) result.pop_back();
    return result;
}

std::string extract_quoted(const std::string& input) {
    size_t start = input.find('"');
    if (start == std::string::npos) start = input.find('\'');
    if (start == std::string::npos) return "";
    size_t end = input.find(input[start], start + 1);
    if (end == std::string::npos) return "";
    return input.substr(start + 1, end - start - 1);
}

std::string extract_filename(const std::string& line) {
    std::string quoted = extract_quoted(line);
    if (!quoted.empty()) return quoted;

    std::string lower = to_lowercase(line);
    size_t as_pos = lower.find(" as ");
    if (as_pos != std::string::npos) {
        std::string after = line.substr(as_pos + 4);
        size_t start = after.find_first_not_of(" \t");
        if (start != std::string::npos) {
            after = after.substr(start);
            size_t end = after.find_first_of(" \t\n\r,;");
            if (end != std::string::npos) after = after.substr(0, end);
            return after;
        }
    }
    return "";
}

// ----------------------------------------------------------------------
// Code generation handler
void handle_code(const std::string& idea) {
    std::string prompt = "Write C++ code to accomplish the following task. Provide only the code without explanations unless requested. Include necessary headers and a main function if appropriate.\n\n" + idea;
    std::cout << "Asking Ollama...\n";
    std::string response = ask_ollama_with_retry(prompt);
    std::cout << "\n" << response << "\n";

    last_ai_output = response;
    last_ai_type = "code";

    std::string base = idea;
    if (base.length() > 50) base = base.substr(0, 50);
    // Pass raw filename; save_content_to_file will sanitize it.
    save_content_to_file(response, base + ".txt", "code for \"" + idea + "\"");
    pencil::append_to_session("User asked for code: " + idea);
    pencil::append_to_session("Assistant: " + response);
}

// ----------------------------------------------------------------------
// NLU dispatcher
bool handle_natural_language(const std::string& line) {
    // Save requests
    if (contains_phrase(line, "save it") || contains_phrase(line, "save the code") ||
        contains_phrase(line, "write it to a file") || contains_phrase(line, "save as")) {

        if (debug_enabled) std::cout << "[NLU] Matched save request.\n";

        if (last_ai_output.empty()) {
            std::cout << "I don't have any recent code to save.\n";
            return true;
        }

        std::string default_name = "code.txt";
        std::string filename = extract_filename(line);
        if (filename.empty()) {
            std::cout << "What filename would you like to save it as? (default: " << default_name << ")\n> ";
            std::getline(std::cin, filename);
            if (filename.empty()) filename = default_name;
        }

        if (filename.find('.') == std::string::npos) filename += ".txt";

        save_content_to_file(last_ai_output, filename, "code");
        return true;
    }

    // Code triggers
    std::vector<std::pair<std::string, std::string>> code_triggers = {
        {"write code for", "for"},
        {"write a program that", "that"},
        {"generate code for", "for"},
        {"generate a program that", "that"},
        {"create code for", "for"},
        {"create a program that", "that"},
        {"write a function that", "that"},
        {"code for", "for"}
    };
    for (const auto& [trigger, _] : code_triggers) {
        if (contains_phrase(line, trigger)) {
            if (debug_enabled) std::cout << "[NLU] Matched code trigger: " << trigger << "\n";
            std::string idea = extract_after(line, trigger);
            if (idea.empty()) idea = extract_quoted(line);
            if (idea.empty()) {
                std::cout << "What should the code do?\n> ";
                std::getline(std::cin, idea);
            }
            if (!idea.empty()) handle_code(idea);
            return true;
        }
    }
    // Generic code
    std::vector<std::string> generic_code = {
        "write code", "generate code", "create code", "write a program", "generate a program"
    };
    for (const auto& trigger : generic_code) {
        if (contains_phrase(line, trigger)) {
            if (debug_enabled) std::cout << "[NLU] Matched generic code trigger: " << trigger << "\n";
            std::cout << "What should the code do?\n> ";
            std::string idea;
            std::getline(std::cin, idea);
            if (!idea.empty()) handle_code(idea);
            return true;
        }
    }

    // Task triggers
    std::vector<std::pair<std::string, std::string>> task_triggers = {
        {"start a task to", "to"},
        {"begin a task to", "to"},
        {"create a task to", "to"},
        {"start a task that", "that"},
        {"begin a task that", "that"},
        {"create a task that", "that"}
    };
    for (const auto& [trigger, _] : task_triggers) {
        if (contains_phrase(line, trigger)) {
            if (debug_enabled) std::cout << "[NLU] Matched task trigger: " << trigger << "\n";
            std::string desc = extract_after(line, trigger);
            if (desc.empty()) desc = extract_quoted(line);
            if (desc.empty()) {
                std::cout << "Describe the task:\n> ";
                std::getline(std::cin, desc);
            }
            if (!desc.empty()) start_new_task(desc);
            return true;
        }
    }
    // Generic task
    std::vector<std::string> generic_task = {
        "start a task", "begin a task", "create a task", "new task"
    };
    for (const auto& trigger : generic_task) {
        if (contains_phrase(line, trigger)) {
            if (debug_enabled) std::cout << "[NLU] Matched generic task trigger: " << trigger << "\n";
            std::cout << "Describe the task:\n> ";
            std::string desc;
            std::getline(std::cin, desc);
            if (!desc.empty()) start_new_task(desc);
            return true;
        }
    }

    return false;
}

// ----------------------------------------------------------------------
// List files
void list_files() {
    std::cout << "\n📁 Files in " << std::filesystem::absolute(pencil::get_pencil_dir()).string() << ":\n";
    try {
        for (const auto& entry : std::filesystem::directory_iterator(pencil::get_pencil_dir())) {
            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                std::cout << "  " << entry.path().filename().string()
                          << " (" << entry.file_size() << " bytes)\n";
            }
        }
        if (std::filesystem::exists(pencil::get_tasks_dir())) {
            std::cout << "\n📂 Tasks:\n";
            for (const auto& entry : std::filesystem::directory_iterator(pencil::get_tasks_dir())) {
                if (entry.is_directory()) {
                    std::cout << "  " << entry.path().filename().string() << "/\n";
                    // Optionally list iteration files
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error listing files: " << e.what() << std::endl;
    }
}

// ----------------------------------------------------------------------
int main() {
    if (!pencil::init_workspace()) {
        std::cerr << "Fatal error: cannot create workspace directory." << std::endl;
        return 1;
    }

    std::cout << "📁 Workspace: " << std::filesystem::absolute(pencil::get_pencil_dir()).string() << "\n";

    // Initialize Git repository if possible
    if (!init_git_repo()) {
        std::cerr << "Warning: Could not initialise Git repository. Git features disabled.\n";
    } else {
        std::cout << "Git repository initialised (or already exists).\n";
    }

    warm_up_ollama();
    if (last_ollama_time == 0) last_ollama_time = time(nullptr);

    std::cout << "PENCILCLAW – C++ Coding Agent with Git integration\n";
    std::cout << "Heartbeat interval: " << HEARTBEAT_INTERVAL << " seconds\n";
    std::cout << "Type /HELP for commands.\n";

    std::string last_response;
    time_t last_heartbeat_run = time(nullptr);

    while (true) {
        time_t now = time(nullptr);
        check_and_keep_alive(now);

        std::cout << "\n> ";
        std::string line;
        std::getline(std::cin, line);
        if (line.empty()) continue;

        if (line[0] != '/') {
            if (handle_natural_language(line)) {
                if (now - last_heartbeat_run >= HEARTBEAT_INTERVAL) {
                    run_heartbeat(now);
                    last_heartbeat_run = now;
                }
                continue;
            }
        }

        if (line[0] == '/') {
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
                std::cout << "  /HELP                – this help\n";
                std::cout << "  /CODE <idea>         – generate C++ code for a task\n";
                std::cout << "  /TASK <description>  – start a new autonomous coding task\n";
                std::cout << "  /TASK_STATUS         – show current active task\n";
                std::cout << "  /STOP_TASK           – clear active task\n";
                std::cout << "  /EXECUTE             – compile & run code from last output\n";
                std::cout << "  /FILES               – list all saved files and tasks\n";
                std::cout << "  /DEBUG               – toggle debug output\n";
                std::cout << "  /EXIT                – quit\n";
                std::cout << "\nNatural language examples:\n";
                std::cout << "  'write code for a fibonacci function'\n";
                std::cout << "  'start a task to build a calculator'\n";
                std::cout << "  'save it as mycode.txt' (after code generation)\n";
            }
            else if (cmd == "/DEBUG") {
                debug_enabled = !debug_enabled;
                std::cout << "Debug mode " << (debug_enabled ? "enabled" : "disabled") << ".\n";
            }
            else if (cmd == "/CODE") {
                if (arg.empty()) {
                    std::cout << "Please provide a description of the code.\n";
                    continue;
                }
                handle_code(arg);
            }
            else if (cmd == "/TASK") {
                if (arg.empty()) {
                    std::cout << "Please provide a task description.\n";
                    continue;
                }
                start_new_task(arg);
            }
            else if (cmd == "/TASK_STATUS") {
                std::string folder = get_active_task_folder();
                if (folder.empty()) {
                    std::cout << "No active task.\n";
                } else {
                    auto desc_opt = pencil::read_file(folder + "description.txt");
                    std::string desc = desc_opt.value_or("unknown");
                    std::cout << "Active task: " << desc << "\n";
                    std::cout << "Folder: " << folder << "\n";
                    // Count iterations
                    int count = 0;
                    for (const auto& entry : std::filesystem::directory_iterator(folder)) {
                        if (entry.path().filename().string().rfind("iteration_", 0) == 0)
                            count++;
                    }
                    std::cout << "Iterations so far: " << count << "\n";
                }
            }
            else if (cmd == "/STOP_TASK") {
                clear_active_task();
                std::cout << "Active task cleared.\n";
            }
            else if (cmd == "/FILES") {
                list_files();
            }
            else if (cmd == "/EXECUTE") {
                if (last_ai_output.empty()) {
                    std::cout << "No previous AI output to execute from.\n";
                    continue;
                }
                auto blocks = extract_code_blocks(last_ai_output);
                if (blocks.empty()) {
                    std::cout << "No code blocks found in last output.\n";
                    continue;
                }
                std::cout << "--- Code to execute ---\n";
                std::cout << blocks[0] << "\n";
                std::cout << "------------------------\n";
                std::cout << "WARNING: This code was generated by an AI and may be unsafe.\n";
                std::cout << "Type 'yes' to confirm execution (any other input cancels): ";
                std::string confirm;
                std::getline(std::cin, confirm);
                if (confirm != "yes") {
                    std::cout << "Execution cancelled.\n";
                    continue;
                }
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
            // Free prompt (not handled by NLU)
            std::cout << "Sending to Ollama...\n";
            last_response = ask_ollama_with_retry(line);
            last_ai_output = last_response;
            last_ai_type = "free";
            std::cout << last_response << "\n";
            pencil::append_to_session("User: " + line);
            pencil::append_to_session("Assistant: " + last_response);
        }

        time_t now2 = time(nullptr);
        if (now2 - last_heartbeat_run >= HEARTBEAT_INTERVAL) {
            run_heartbeat(now2);
            last_heartbeat_run = now2;
        }
    }

    return 0;
}
