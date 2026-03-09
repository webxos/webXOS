// pencil_utils.hpp – Core file utilities for PencilClaw coding agent
#ifndef PENCIL_UTILS_HPP
#define PENCIL_UTILS_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <optional>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace pencil {
    // Base directory – can be overridden by environment variable PENCIL_DATA
    inline std::string get_pencil_dir() {
        const char* env = std::getenv("PENCIL_DATA");
        return env ? env : "./pencil_data/";
    }

    inline std::string get_session_log() {
        return get_pencil_dir() + "session.log";
    }

    inline std::string get_tasks_dir() {
        return get_pencil_dir() + "tasks/";
    }

    inline std::string get_active_task_file() {
        return get_pencil_dir() + "active_task.txt";
    }

    // Ensure the working directory exists. Returns true on success or if already exists.
    inline bool init_workspace() {
        std::error_code ec;
        bool created = std::filesystem::create_directory(get_pencil_dir(), ec);
        if (ec) {
            std::cerr << "Error creating directory " << get_pencil_dir() << ": " << ec.message() << std::endl;
            return false;
        }
        // Also create tasks directory
        std::filesystem::create_directory(get_tasks_dir(), ec);
        return true;
    }

    // Append a line to the session log. Returns true on success.
    inline bool append_to_session(const std::string& text) {
        std::ofstream log(get_session_log(), std::ios::app);
        if (!log) return false;
        log << text << std::endl;
        return !log.fail();
    }

    // Read entire file content. Returns std::nullopt if file cannot be opened.
    inline std::optional<std::string> read_file(const std::string& path) {
        std::ifstream f(path);
        if (!f) {
            std::cerr << "Warning: Could not open file: " << path << std::endl;
            return std::nullopt;
        }
        std::string content((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
        return content;
    }

    // Save text to a file (overwrite). Returns true on success.
    inline bool save_text(const std::string& path, const std::string& text) {
        std::ofstream f(path);
        if (!f) return false;
        f << text;
        return !f.fail();
    }

    // Get a timestamp string for folder/file names.
    inline std::string timestamp() {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
        return ss.str();
    }
}

#endif // PENCIL_UTILS_HPP
