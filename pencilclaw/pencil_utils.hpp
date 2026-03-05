// pencil_utils.hpp – Notepad logic: directories, templates, session log
#ifndef PENCIL_UTILS_HPP
#define PENCIL_UTILS_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <map>

namespace pencil {
    const std::string PENCIL_DIR = "./pencil_data/";
    const std::string SESSION_LOG = "session.log";
    const std::string BOOK_FILE = "book.txt";

    // Ensure the working directory exists. Returns true on success or if already exists.
    inline bool init_workspace() {
        std::error_code ec;
        bool created = std::filesystem::create_directory(PENCIL_DIR, ec);
        if (ec) {
            std::cerr << "Error creating directory " << PENCIL_DIR << ": " << ec.message() << std::endl;
            return false;
        }
        return true; // either created or already existed (create_directory returns false if existed, but no error)
    }

    // Append a line to the session log. Returns true on success.
    inline bool append_to_session(const std::string& text) {
        std::ofstream log(PENCIL_DIR + SESSION_LOG, std::ios::app);
        if (!log) return false;
        log << text << std::endl;
        return !log.fail();
    }

    // Read entire file content
    inline std::string read_file(const std::string& path) {
        std::ifstream f(path);
        if (!f) return "";
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

    // Append to the book file. Returns true on success.
    inline bool append_to_book(const std::string& text) {
        std::ofstream book(PENCIL_DIR + BOOK_FILE, std::ios::app);
        if (!book) return false;
        book << text << std::endl;
        return !book.fail();
    }

    // Return a prompt template for each ADA command
    inline std::string get_template(const std::string& cmd) {
        static std::map<std::string, std::string> templates = {
            {"/STORY", "Write a creative story with the following title. Use vivid descriptions and a clear narrative."},
            {"/POEM", "Compose a poem about the given subject. Use rhythm and imagery."},
            {"/BOOK", "You are a novelist. Continue the existing book by writing a new chapter. Maintain style and characters."}
        };
        auto it = templates.find(cmd);
        if (it != templates.end()) return it->second;
        return "";
    }
}

#endif // PENCIL_UTILS_HPP
