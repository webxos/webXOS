#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <curl/curl.h>
#include <sys/stat.h>
#include <stdint.h>
#include <dirent.h>
#include "cJSON.h"

// ------------------------------------------------------------------
// Shadow arena: all persistent data lives in a single realloc'd block.
// A header is stored immediately before the user pointer.
// ------------------------------------------------------------------
typedef struct ShadowHeader {
    size_t capacity;
    size_t used;
    unsigned int magic;
    int dirty;
} ShadowHeader;

#define SHADOW_MAGIC 0xDEADBEEF
#define HEADER_SIZE (sizeof(ShadowHeader))

static void* shadow_alloc(void* ptr, size_t old_size, size_t new_size) {
    (void)old_size;
    return realloc(ptr, new_size);
}

static void* shadow_malloc(size_t size) {
    return malloc(size);
}

static void shadow_free(void* ptr) {
    free(ptr);
}

// Growable arena functions
static void* arena_grow(void* arena, size_t new_cap) {
    if (!arena) {
        size_t total = HEADER_SIZE + new_cap;
        ShadowHeader* hdr = (ShadowHeader*)shadow_malloc(total);
        if (!hdr) return NULL;
        hdr->capacity = new_cap;
        hdr->used = 0;
        hdr->magic = SHADOW_MAGIC;
        hdr->dirty = 0;
        return (char*)hdr + HEADER_SIZE;
    }
    ShadowHeader* hdr = (ShadowHeader*)((char*)arena - HEADER_SIZE);
    if (hdr->magic != SHADOW_MAGIC) return NULL;
    size_t total = HEADER_SIZE + new_cap;
    ShadowHeader* new_hdr = (ShadowHeader*)shadow_alloc(hdr, HEADER_SIZE + hdr->capacity, total);
    if (!new_hdr) return NULL;
    new_hdr->capacity = new_cap;
    return (char*)new_hdr + HEADER_SIZE;
}

static void* arena_alloc(void* arena, size_t size) {
    if (!arena) return NULL;
    ShadowHeader* hdr = (ShadowHeader*)((char*)arena - HEADER_SIZE);
    if (hdr->used + size > hdr->capacity) return NULL;
    void* ptr = (char*)arena + hdr->used;
    hdr->used += size;
    return ptr;
}

// ------------------------------------------------------------------
// Blob kinds (stored in the arena)
// ------------------------------------------------------------------
typedef enum {
    BLOB_KIND_SYSTEM = 0,
    BLOB_KIND_USER,
    BLOB_KIND_ASSISTANT,
    BLOB_KIND_TOOL_CALL,
    BLOB_KIND_TOOL_RESULT,
    BLOB_KIND_COUNT
} BlobKind;

typedef struct BlobHeader {
    size_t size;
    BlobKind kind;
    uint64_t id;
} BlobHeader;

#define BLOB_HEADER_SIZE (sizeof(BlobHeader))

static uint64_t next_id = 1;

static void* blob_append(void** arena, BlobKind kind, const char* data) {
    if (!arena || !*arena || !data) return NULL;
    ShadowHeader* hdr = (ShadowHeader*)((char*)*arena - HEADER_SIZE);
    size_t len = strlen(data) + 1;  // include null terminator
    size_t total = BLOB_HEADER_SIZE + len;

    if (hdr->used + total > hdr->capacity) {
        size_t new_cap = hdr->capacity * 2;
        if (new_cap < hdr->used + total) new_cap = hdr->used + total;
        void* new_arena = arena_grow(*arena, new_cap);
        if (!new_arena) return NULL;
        *arena = new_arena;
        hdr = (ShadowHeader*)((char*)*arena - HEADER_SIZE);
    }

    void* ptr = (char*)*arena + hdr->used;
    BlobHeader* bh = (BlobHeader*)ptr;
    bh->size = len;
    bh->kind = kind;
    bh->id = next_id++;
    char* blob_data = (char*)ptr + BLOB_HEADER_SIZE;
    memcpy(blob_data, data, len);
    hdr->used += total;
    return bh;
}

static BlobHeader* blob_iterate(void* arena, BlobHeader* prev) {
    if (!arena) return NULL;
    ShadowHeader* shadow = (ShadowHeader*)((char*)arena - HEADER_SIZE);
    if (!prev) {
        // first blob
        if (shadow->used < BLOB_HEADER_SIZE) return NULL;
        return (BlobHeader*)arena;
    }
    char* end = (char*)arena + shadow->used;
    char* next = (char*)prev + BLOB_HEADER_SIZE + prev->size;
    if (next + BLOB_HEADER_SIZE > end) return NULL;
    return (BlobHeader*)next;
}

// ------------------------------------------------------------------
// Persistence: save/load arena to/from file
// ------------------------------------------------------------------
static int save_state(void* arena, const char* filename) {
    if (!arena) return -1;
    ShadowHeader* hdr = (ShadowHeader*)((char*)arena - HEADER_SIZE);
    FILE* f = fopen(filename, "wb");
    if (!f) return -1;
    size_t total = HEADER_SIZE + hdr->used;
    if (fwrite(hdr, 1, total, f) != total) {
        fclose(f);
        return -1;
    }
    fclose(f);
    return 0;
}

static void* load_state(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long total = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (total < (long)HEADER_SIZE) {
        fclose(f);
        return NULL;
    }
    ShadowHeader* hdr = (ShadowHeader*)shadow_malloc(total);
    if (!hdr) {
        fclose(f);
        return NULL;
    }
    if (fread(hdr, 1, total, f) != (size_t)total) {
        shadow_free(hdr);
        fclose(f);
        return NULL;
    }
    fclose(f);
    if (hdr->magic != SHADOW_MAGIC || hdr->used > hdr->capacity) {
        shadow_free(hdr);
        return NULL;
    }
    return (char*)hdr + HEADER_SIZE;
}

// ------------------------------------------------------------------
// Tool system: functions the LLM can call
// ------------------------------------------------------------------
typedef struct Tool {
    const char* name;
    char* (*fn)(const char*);
} Tool;

// Write callback for curl (used by both http_get and call_llm)
static size_t write_callback(void* ptr, size_t size, size_t nmemb, void* userdata) {
    size_t total = size * nmemb;
    char** response_ptr = (char**)userdata;
    size_t current_len = *response_ptr ? strlen(*response_ptr) : 0;
    char* new = realloc(*response_ptr, current_len + total + 1);
    if (!new) return 0;
    *response_ptr = new;
    memcpy(*response_ptr + current_len, ptr, total);
    (*response_ptr)[current_len + total] = '\0';
    return total;
}

// Tool implementations
static char* tool_shell(const char* args) {
    FILE* fp = popen(args, "r");
    if (!fp) return strdup("Failed to execute command");
    char* result = NULL;
    size_t total = 0;
    char buf[256];
    while (fgets(buf, sizeof(buf), fp)) {
        size_t len = strlen(buf);
        char* new_result = realloc(result, total + len + 1);
        if (!new_result) {
            free(result);
            pclose(fp);
            return strdup("Memory error");
        }
        result = new_result;
        memcpy(result + total, buf, len + 1);
        total += len;
    }
    pclose(fp);
    if (!result) result = strdup("");
    return result;
}

static char* tool_file_read(const char* args) {
    FILE* f = fopen(args, "rb");
    if (!f) return strdup("File not found");
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* content = malloc(size + 1);
    if (!content) {
        fclose(f);
        return strdup("Memory error");
    }
    fread(content, 1, size, f);
    content[size] = '\0';
    fclose(f);
    return content;
}

static char* tool_file_write(const char* args) {
    // simplistic: args = "filename content"
    char* copy = strdup(args);
    char* space = strchr(copy, ' ');
    if (!space) {
        free(copy);
        return strdup("Invalid format: need 'filename content'");
    }
    *space = '\0';
    const char* filename = copy;
    const char* content = space + 1;
    FILE* f = fopen(filename, "w");
    if (!f) {
        free(copy);
        return strdup("Cannot write file");
    }
    fprintf(f, "%s", content);
    fclose(f);
    free(copy);
    return strdup("File written");
}

static char* tool_http_get(const char* args) {
    CURL* curl = curl_easy_init();
    if (!curl) return strdup("curl init failed");
    char* response = NULL;
    curl_easy_setopt(curl, CURLOPT_URL, args);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    if (res != CURLE_OK) {
        free(response);
        return strdup("HTTP request failed");
    }
    return response ? response : strdup("");
}

static char* tool_math(const char* args) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "echo '%s' | bc", args);
    FILE* fp = popen(cmd, "r");
    if (!fp) return strdup("bc failed");
    char buf[256];
    if (fgets(buf, sizeof(buf), fp) == NULL) {
        pclose(fp);
        return strdup("");
    }
    pclose(fp);
    return strdup(buf);
}

static char* tool_list_dir(const char* args) {
    DIR* d = opendir(args);
    if (!d) return strdup("Error: cannot open directory");
    char* result = NULL;
    size_t len = 0;
    FILE* out = open_memstream(&result, &len);
    if (!out) {
        closedir(d);
        return strdup("Memory error");
    }
    struct dirent* entry;
    while ((entry = readdir(d)) != NULL) {
        fprintf(out, "%s\n", entry->d_name);
    }
    closedir(d);
    fclose(out);
    return result ? result : strdup("");
}

static Tool tools[] = {
    {"shell", tool_shell},
    {"file_read", tool_file_read},
    {"file_write", tool_file_write},
    {"http_get", tool_http_get},
    {"math", tool_math},
    {"list_dir", tool_list_dir},
    {NULL, NULL}
};

// ------------------------------------------------------------------
// LLM communication (Ollama)
// ------------------------------------------------------------------
static const char* ollama_endpoint = "http://localhost:11434/api/generate";
static const char* ollama_model = "qwen2.5:0.5b";  // change to your model

static char* call_llm(const char* prompt) {
    CURL* curl = curl_easy_init();
    if (!curl) return NULL;

    cJSON* root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "model", ollama_model);
    cJSON_AddStringToObject(root, "prompt", prompt);
    cJSON_AddBoolToObject(root, "stream", 0);
    char* json_str = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);

    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    char* response = NULL;
    curl_easy_setopt(curl, CURLOPT_URL, ollama_endpoint);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_str);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);
    free(json_str);

    if (res != CURLE_OK) {
        free(response);
        return NULL;
    }

    // Parse JSON response
    cJSON* resp = cJSON_Parse(response);
    free(response);
    if (!resp) return NULL;
    cJSON* text = cJSON_GetObjectItem(resp, "response");
    char* result = text && cJSON_IsString(text) ? strdup(text->valuestring) : NULL;
    cJSON_Delete(resp);
    return result;
}

// ------------------------------------------------------------------
// Helper: build a conversation prompt from all blobs
// ------------------------------------------------------------------
static char* build_prompt(void* shadow) {
    // Start with a large buffer (dynamically grow if needed)
    size_t cap = 8192;
    char* prompt = malloc(cap);
    if (!prompt) return NULL;
    prompt[0] = '\0';
    size_t used = 0;

    BlobHeader* blob = NULL;
    while ((blob = blob_iterate(shadow, blob)) != NULL) {
        const char* data = (const char*)blob + BLOB_HEADER_SIZE;
        const char* role = NULL;
        switch (blob->kind) {
            case BLOB_KIND_SYSTEM:    role = "System: "; break;
            case BLOB_KIND_USER:       role = "User: "; break;
            case BLOB_KIND_ASSISTANT:  role = "Assistant: "; break;
            case BLOB_KIND_TOOL_RESULT:role = "Tool result: "; break;
            default: continue;  // skip tool calls (they are not part of conversation directly)
        }
        size_t needed = strlen(role) + blob->size;  // blob->size includes null terminator
        if (used + needed + 1 > cap) {
            cap = (used + needed + 1) * 2;
            char* new_prompt = realloc(prompt, cap);
            if (!new_prompt) {
                free(prompt);
                return NULL;
            }
            prompt = new_prompt;
        }
        strcat(prompt, role);
        strcat(prompt, data);
        strcat(prompt, "\n");
        used = strlen(prompt);
    }
    // Add final "Assistant: " to prompt the model
    strcat(prompt, "Assistant: ");
    return prompt;
}

// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------
int main(int argc, char** argv) {
    // Check for no-LLM mode flag
    int no_llm_mode = 0;
    if (argc > 1 && strcmp(argv[1], "--no-llm") == 0) {
        no_llm_mode = 1;
    }

    // Load persistent state
    void* shadow = load_state("shadowclaw.bin");
    if (!shadow) {
        // Create a fresh arena (initial capacity 64KB)
        shadow = arena_grow(NULL, 64 * 1024);
        if (!shadow) {
            fprintf(stderr, "Failed to allocate arena\n");
            return 1;
        }
        // Add system prompt (IMPROVED VERSION)
        const char* system_prompt = 
            "You are Shadowclaw, a helpful AI assistant with access to tools. "
            "When you need to use a tool, you MUST output exactly:\n"
            "```tool\n{\"tool\":\"name\",\"args\":\"arguments\"}\n```\n"
            "For example, to fetch a webpage:\n"
            "```tool\n{\"tool\":\"http_get\",\"args\":\"https://example.com\"}\n```\n"
            "Then wait for the result before giving your final answer.\n"
            "Available tools: shell, file_read, file_write, http_get, math, list_dir.";
        blob_append(&shadow, BLOB_KIND_SYSTEM, system_prompt);
    }

    char buf[4096];
    while (1) {
        printf("> ");
        if (!fgets(buf, sizeof(buf), stdin)) break;
        buf[strcspn(buf, "\n")] = 0;  // remove trailing newline

        // ----- Slash commands (always available) -----
        if (buf[0] == '/') {
            if (strcmp(buf, "/help") == 0) {
                printf("Shadowclaw commands:\n"
                       "  /help       Show this help\n"
                       "  /tools      List available tools\n"
                       "  /state      Show arena memory stats\n"
                       "  /clear      Clear conversation history (keeps system prompt)\n"
                       "  /chat       Remind you that chat mode is active\n"
                       "  /exit       Exit Shadowclaw\n");
            } else if (strcmp(buf, "/tools") == 0) {
                printf("Available tools:\n");
                for (Tool* t = tools; t->name; t++) {
                    printf("  %s\n", t->name);
                }
            } else if (strcmp(buf, "/state") == 0) {
                ShadowHeader* hdr = (ShadowHeader*)((char*)shadow - HEADER_SIZE);
                printf("Arena capacity: %zu bytes\n", hdr->capacity);
                printf("Arena used: %zu bytes\n", hdr->used);
                printf("Dirty flag: %d\n", hdr->dirty);
            } else if (strcmp(buf, "/clear") == 0) {
                ShadowHeader* hdr = (ShadowHeader*)((char*)shadow - HEADER_SIZE);
                hdr->used = 0;
                next_id = 1; // reset ID counter
                // Re-add system prompt
                const char* system_prompt = 
                    "You are Shadowclaw, a helpful AI assistant with access to tools. "
                    "When you need to use a tool, you MUST output exactly:\n"
                    "```tool\n{\"tool\":\"name\",\"args\":\"arguments\"}\n```\n"
                    "For example, to fetch a webpage:\n"
                    "```tool\n{\"tool\":\"http_get\",\"args\":\"https://example.com\"}\n```\n"
                    "Then wait for the result before giving your final answer.\n"
                    "Available tools: shell, file_read, file_write, http_get, math, list_dir.";
                blob_append(&shadow, BLOB_KIND_SYSTEM, system_prompt);
                printf("Conversation cleared.\n");
            } else if (strcmp(buf, "/chat") == 0) {
                printf("You are already in chat mode. Type your message.\n");
            } else if (strcmp(buf, "/exit") == 0) {
                break;
            } else {
                printf("Unknown command. Try /help\n");
            }
            continue;
        }

        // ----- no‑LLM mode using tiny interpreter (just echo) -----
        if (no_llm_mode) {
            printf("(no‑llm) %s\n", buf);
            continue;
        }

        // ----- Normal LLM mode -----
        // Append user input
        blob_append(&shadow, BLOB_KIND_USER, buf);

        // Inner loop to handle tool calls
        int tool_loop = 1;
        int max_turns = 5;  // prevent infinite loops
        while (tool_loop && max_turns-- > 0) {
            char* prompt = build_prompt(shadow);
            if (!prompt) {
                fprintf(stderr, "Failed to build prompt\n");
                break;
            }

            char* llm_response = call_llm(prompt);
            free(prompt);
            if (!llm_response) {
                fprintf(stderr, "LLM call failed\n");
                break;
            }

            // Check for tool call in response
            char* tool_start = strstr(llm_response, "```tool");
            if (tool_start) {
                // Extract the JSON part (between newline after ```tool and closing ```)
                char* json_start = tool_start + strlen("```tool");
                while (*json_start == ' ' || *json_start == '\n') json_start++;
                char* json_end = strstr(json_start, "```");
                if (!json_end) {
                    // malformed, treat as normal response
                    printf("Assistant: %s\n", llm_response);
                    blob_append(&shadow, BLOB_KIND_ASSISTANT, llm_response);
                    free(llm_response);
                    break;
                }
                *json_end = '\0';  // temporarily terminate

                // Parse JSON
                cJSON* tool_json = cJSON_Parse(json_start);
                *json_end = '`';  // restore (not strictly needed)
                if (!tool_json) {
                    fprintf(stderr, "Failed to parse tool JSON\n");
                    printf("Assistant: %s\n", llm_response);
                    blob_append(&shadow, BLOB_KIND_ASSISTANT, llm_response);
                    free(llm_response);
                    break;
                }

                cJSON* tool_name = cJSON_GetObjectItem(tool_json, "tool");
                cJSON* tool_args = cJSON_GetObjectItem(tool_json, "args");
                if (!cJSON_IsString(tool_name) || !cJSON_IsString(tool_args)) {
                    cJSON_Delete(tool_json);
                    fprintf(stderr, "Tool JSON missing 'tool' or 'args' string\n");
                    printf("Assistant: %s\n", llm_response);
                    blob_append(&shadow, BLOB_KIND_ASSISTANT, llm_response);
                    free(llm_response);
                    break;
                }

                // Find and execute tool
                char* result = NULL;
                for (Tool* t = tools; t->name; t++) {
                    if (strcmp(t->name, tool_name->valuestring) == 0) {
                        result = t->fn(tool_args->valuestring);
                        break;
                    }
                }
                if (!result) {
                    result = strdup("Unknown tool");
                }

                // Store tool call and result
                blob_append(&shadow, BLOB_KIND_TOOL_CALL, llm_response);
                blob_append(&shadow, BLOB_KIND_TOOL_RESULT, result);
                free(result);
                cJSON_Delete(tool_json);
                free(llm_response);

                // Loop again to let LLM respond with final answer
                // (tool_loop remains 1)
            } else {
                // Normal response, no tool call
                printf("Assistant: %s\n", llm_response);
                blob_append(&shadow, BLOB_KIND_ASSISTANT, llm_response);
                free(llm_response);
                tool_loop = 0;  // exit inner loop
            }
        }

        // Save state after each user turn (including tool interactions)
        save_state(shadow, "shadowclaw.bin");
    }

    // Final save on exit
    save_state(shadow, "shadowclaw.bin");
    return 0;
}
