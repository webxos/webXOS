#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <time.h>
#include <curl/curl.h>
#include "cJSON.h"

// --------------------------------------------------------------------
//  Shadow Header + Arena (Tsoding/stb_ds style)
// --------------------------------------------------------------------
typedef struct {
    size_t   capacity;     // bytes available AFTER header
    size_t   length;       // bytes used AFTER header
    uint64_t tag;          // magic: 0x534841444F57434C = "SHADOWCL"
    uint32_t version;
    uint32_t flags;        // bit 0 = dirty
} ShadowHeader;

#define SHADOW_MAGIC 0x534841444F57434CULL
#define SHADOW_SIZE (sizeof(ShadowHeader))
#define ALIGN_UP(x, a) (((x) + (a)-1) & ~((a)-1))

static inline ShadowHeader* shadow_header(void *data) {
    return (ShadowHeader*)((char*)data - SHADOW_SIZE);
}

typedef struct {
    void    *data;       // user‑facing payload pointer
    size_t   reserved;   // total malloc size (header + capacity)
} ShadowArena;

ShadowArena shadow_arena_create(size_t initial_capacity) {
    ShadowArena a = {0};
    size_t total = SHADOW_SIZE + initial_capacity;
    total = ALIGN_UP(total, 64);

    ShadowHeader *h = malloc(total);
    if (!h) abort();

    *h = (ShadowHeader){
        .capacity = initial_capacity,
        .length   = 0,
        .tag      = SHADOW_MAGIC,
        .version  = 1,
        .flags    = 0
    };
    a.data = (char*)h + SHADOW_SIZE;
    a.reserved = total;
    return a;
}

void shadow_arena_destroy(ShadowArena *a) {
    if (a->data) {
        free(shadow_header(a->data));
        a->data = NULL;
    }
}

void* shadow_arena_push(ShadowArena *a, const void *src, size_t bytes) {
    ShadowHeader *h = shadow_header(a->data);

    if (h->length + bytes > h->capacity) {
        size_t new_cap = h->capacity ? h->capacity * 2 : 4096;
        while (new_cap < h->length + bytes) new_cap *= 2;
        size_t new_total = SHADOW_SIZE + new_cap;
        new_total = ALIGN_UP(new_total, 64);

        ShadowHeader *new_h = realloc(h, new_total);
        if (!new_h) abort();

        new_h->capacity = new_cap;
        a->data = (char*)new_h + SHADOW_SIZE;
        a->reserved = new_total;
        h = new_h;
    }

    char *dst = (char*)a->data + h->length;
    if (src) memcpy(dst, src, bytes);
    else     memset(dst, 0, bytes);

    h->length += bytes;
    h->flags |= 1;  // dirty
    return dst;
}

size_t shadow_arena_len(const ShadowArena *a) {
    return shadow_header(a->data)->length;
}

void shadow_arena_clear(ShadowArena *a) {
    ShadowHeader *h = shadow_header(a->data);
    h->length = 0;
    h->flags &= ~1;
}

// --------------------------------------------------------------------
//  Blob Format (tagged, length‑prefixed items)
// --------------------------------------------------------------------
typedef struct {
    uint32_t size;       // payload size (excluding this header)
    uint32_t kind;       // 1=system,2=user,3=assistant,4=tool_call,5=tool_result,6=memory
    uint64_t id;         // unique id (timestamp or counter)
} BlobHeader;

// Append a typed blob – returns offset from arena->data start
ptrdiff_t blob_append(ShadowArena *a, uint32_t kind, uint64_t id,
                      const void *payload, size_t payload_bytes)
{
    size_t total = sizeof(BlobHeader) + payload_bytes;
    char *p = shadow_arena_push(a, NULL, total);

    BlobHeader bh = {
        .size = (uint32_t)payload_bytes,
        .kind = kind,
        .id   = id
    };
    memcpy(p, &bh, sizeof(bh));
    if (payload_bytes) memcpy(p + sizeof(bh), payload, payload_bytes);
    return p - (char*)a->data;
}

// Iterate over all blobs: calls `f(blob_header, payload, userdata)`
void blob_foreach(ShadowArena *a,
                  void (*f)(const BlobHeader*, const char*, void*),
                  void *userdata)
{
    char *start = a->data;
    size_t len = shadow_header(a->data)->length;
    char *end = start + len;
    char *p = start;
    while (p < end) {
        BlobHeader *bh = (BlobHeader*)p;
        char *payload = p + sizeof(BlobHeader);
        f(bh, payload, userdata);
        p += sizeof(BlobHeader) + bh->size;
    }
}

// --------------------------------------------------------------------
//  Persistence: save / load the whole arena to a file
// --------------------------------------------------------------------
bool arena_save(const ShadowArena *a, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) return false;
    // write the whole malloc block (header + payload)
    size_t n = fwrite(shadow_header(a->data), 1, a->reserved, f);
    fclose(f);
    return n == a->reserved;
}

bool arena_load(ShadowArena *a, const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) return false;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size < (long)SHADOW_SIZE) {
        fclose(f);
        return false;
    }

    void *block = malloc(size);
    if (!block) { fclose(f); return false; }
    if (fread(block, 1, size, f) != (size_t)size) {
        free(block);
        fclose(f);
        return false;
    }
    fclose(f);

    // validate magic
    ShadowHeader *h = (ShadowHeader*)block;
    if (h->tag != SHADOW_MAGIC || h->version != 1) {
        free(block);
        return false;
    }

    // destroy old arena and replace
    shadow_arena_destroy(a);
    a->data = (char*)block + SHADOW_SIZE;
    a->reserved = size;
    return true;
}

// --------------------------------------------------------------------
//  Tools
// --------------------------------------------------------------------
typedef struct {
    char *name;
    char *(*func)(const char *args);   // returns newly allocated string
} Tool;

// tool: shell command execution
char* tool_shell(const char *args) {
    FILE *fp = popen(args, "r");
    if (!fp) return strdup("error: popen failed");
    char *result = NULL;
    size_t len = 0;
    FILE *out = open_memstream(&result, &len);
    char buf[256];
    while (fgets(buf, sizeof(buf), fp)) fputs(buf, out);
    pclose(fp);
    fclose(out);
    return result ? result : strdup("");
}

// tool: read file
char* tool_read_file(const char *args) {
    FILE *fp = fopen(args, "rb");
    if (!fp) return strdup("error: cannot open file");
    char *content = NULL;
    size_t len = 0;
    FILE *out = open_memstream(&content, &len);
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), fp)) > 0)
        fwrite(buf, 1, n, out);
    fclose(fp);
    fclose(out);
    return content ? content : strdup("");
}

// tool: write file (args: "filename\ncontent")
char* tool_write_file(const char *args) {
    char *filename = strdup(args);
    char *newline = strchr(filename, '\n');
    if (!newline) {
        free(filename);
        return strdup("error: missing newline separator");
    }
    *newline = '\0';
    char *content = newline + 1;
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        free(filename);
        return strdup("error: cannot write file");
    }
    fwrite(content, 1, strlen(content), fp);
    fclose(fp);
    free(filename);
    return strdup("ok");
}

// tool: HTTP GET (args = URL)
size_t write_cb(void *ptr, size_t size, size_t nmemb, void *stream) {
    size_t total = size * nmemb;
    fwrite(ptr, 1, total, (FILE*)stream);
    return total;
}
char* tool_http_get(const char *args) {
    CURL *curl = curl_easy_init();
    if (!curl) return strdup("error: curl init failed");

    char *response = NULL;
    size_t len = 0;
    FILE *out = open_memstream(&response, &len);

    curl_easy_setopt(curl, CURLOPT_URL, args);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, out);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        fclose(out);
        curl_easy_cleanup(curl);
        return strdup("error: curl failed");
    }
    fclose(out);
    curl_easy_cleanup(curl);
    return response ? response : strdup("");
}

// tool: math expression (using bc)
char* tool_math(const char *args) {
    char cmd[4096];
    snprintf(cmd, sizeof(cmd), "echo '%s' | bc 2>/dev/null", args);
    FILE *fp = popen(cmd, "r");
    if (!fp) return strdup("error: bc failed");
    char *result = NULL;
    size_t len = 0;
    FILE *out = open_memstream(&result, &len);
    char buf[256];
    while (fgets(buf, sizeof(buf), fp)) fputs(buf, out);
    pclose(fp);
    fclose(out);
    return result ? result : strdup("");
}

// tool registry
Tool tools[] = {
    {"shell", tool_shell},
    {"read_file", tool_read_file},
    {"write_file", tool_write_file},
    {"http_get", tool_http_get},
    {"math", tool_math},
    {NULL, NULL}
};

char* execute_tool(const char *name, const char *args) {
    for (Tool *t = tools; t->name; t++) {
        if (strcmp(t->name, name) == 0) {
            return t->func(args);
        }
    }
    return strdup("error: unknown tool");
}

// --------------------------------------------------------------------
//  LLM interaction (Ollama)
// --------------------------------------------------------------------
typedef struct {
    char *data;
    size_t len;
} ResponseBuffer;

size_t write_response(void *ptr, size_t size, size_t nmemb, void *stream) {
    ResponseBuffer *buf = (ResponseBuffer*)stream;
    size_t total = size * nmemb;
    buf->data = realloc(buf->data, buf->len + total + 1);
    if (!buf->data) return 0;
    memcpy(buf->data + buf->len, ptr, total);
    buf->len += total;
    buf->data[buf->len] = '\0';
    return total;
}

// call Ollama generate endpoint, return JSON string (malloced)
char* ollama_generate(const char *prompt, const char *model, const char *endpoint) {
    CURL *curl = curl_easy_init();
    if (!curl) return NULL;

    char url[256];
    snprintf(url, sizeof(url), "%s/api/generate", endpoint);

    cJSON *req_json = cJSON_CreateObject();
    cJSON_AddStringToObject(req_json, "model", model);
    cJSON_AddStringToObject(req_json, "prompt", prompt);
    cJSON_AddBoolToObject(req_json, "stream", false);
    char *req_str = cJSON_PrintUnformatted(req_json);
    cJSON_Delete(req_json);

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    ResponseBuffer resp = {0};
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, req_str);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_response);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);

    CURLcode res = curl_easy_perform(curl);
    free(req_str);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        free(resp.data);
        return NULL;
    }
    return resp.data; // caller must free
}

// --------------------------------------------------------------------
//  Prompt builder (very simple: system + last 5 user/assistant messages)
// --------------------------------------------------------------------
typedef struct {
    char *text;
    size_t cap;
    size_t len;
} StringBuilder;

void sb_append(StringBuilder *sb, const char *s) {
    size_t add = strlen(s);
    if (sb->len + add + 1 > sb->cap) {
        sb->cap = sb->cap ? sb->cap * 2 : 1024;
        while (sb->len + add + 1 > sb->cap) sb->cap *= 2;
        sb->text = realloc(sb->text, sb->cap);
    }
    memcpy(sb->text + sb->len, s, add);
    sb->len += add;
    sb->text[sb->len] = '\0';
}

void collect_blob(const BlobHeader *bh, const char *payload, void *user) {
    StringBuilder *sb = (StringBuilder*)user;
    static int count = 0;
    // keep only system (kind 1) and last 10 user/assistant/tool messages
    if (bh->kind == 1) {
        sb_append(sb, "[System]\n");
        sb_append(sb, payload);
        sb_append(sb, "\n\n");
    } else if (bh->kind == 2 || bh->kind == 3 || bh->kind == 5) {
        if (count < 10) {
            const char *role = bh->kind==2 ? "User" : (bh->kind==3 ? "Assistant" : "Tool");
            sb_append(sb, "[");
            sb_append(sb, role);
            sb_append(sb, "]\n");
            sb_append(sb, payload);
            sb_append(sb, "\n\n");
            count++;
        }
    }
}

char* build_prompt(ShadowArena *arena) {
    StringBuilder sb = {0};
    // include system prompt and recent conversation
    blob_foreach(arena, collect_blob, &sb);
    // add instructions for tool use
    sb_append(&sb,
        "[Instruction]\n"
        "You are ShadowClaw, a tiny AI agent. You can use tools by outputting a JSON block like:\n"
        "```tool\n{\"tool\":\"name\",\"args\":\"arguments\"}\n```\n"
        "Available tools: shell, read_file, write_file, http_get, math.\n"
        "After using a tool, you'll see its result. Continue the conversation.\n\n"
        "[User]\n");
    return sb.text; // caller must free
}

// --------------------------------------------------------------------
//  Parse tool call from assistant text (look for ```tool ... ```)
// --------------------------------------------------------------------
char* parse_tool_call(const char *text, char **tool_name, char **tool_args) {
    const char *start = strstr(text, "```tool");
    if (!start) return NULL;
    start += 7; // skip ```tool
    while (*start == ' ' || *start == '\n') start++;
    const char *end = strstr(start, "```");
    if (!end) return NULL;

    size_t len = end - start;
    char *json_str = malloc(len + 1);
    memcpy(json_str, start, len);
    json_str[len] = '\0';

    cJSON *root = cJSON_Parse(json_str);
    free(json_str);
    if (!root) return NULL;

    cJSON *name = cJSON_GetObjectItem(root, "tool");
    cJSON *args = cJSON_GetObjectItem(root, "args");
    if (!cJSON_IsString(name) || !cJSON_IsString(args)) {
        cJSON_Delete(root);
        return NULL;
    }

    *tool_name = strdup(name->valuestring);
    *tool_args = strdup(args->valuestring);
    cJSON_Delete(root);
    return (char*)end + 3; // pointer after the closing ```
}

// --------------------------------------------------------------------
//  Main
// --------------------------------------------------------------------
int main(int argc, char **argv) {
    const char *state_file = "shadowclaw.bin";
    const char *ollama_endpoint = "http://localhost:11434";
    const char *ollama_model = "llama3.2";  // change as needed

    ShadowArena arena = shadow_arena_create(128 * 1024); // 128KB start

    // load previous state if exists
    if (access(state_file, F_OK) == 0) {
        if (arena_load(&arena, state_file)) {
            printf("[ShadowClaw] loaded state from %s\n", state_file);
        } else {
            printf("[ShadowClaw] failed to load %s, starting fresh\n", state_file);
        }
    } else {
        // bootstrap system prompt
        const char *sys = "You are ShadowClaw – tiny, shadowy, Unix‑punk AI agent. Use tools when helpful. Stay minimal.";
        blob_append(&arena, 1, 1, sys, strlen(sys)+1);
    }

    uint64_t msg_id = time(NULL); // simple id

    printf("ShadowClaw ready. Type your message (Ctrl-D to exit)\n");
    char line[4096];
    while (fgets(line, sizeof(line), stdin)) {
        // remove trailing newline
        line[strcspn(line, "\n")] = 0;
        if (strlen(line) == 0) continue;

        // store user message
        blob_append(&arena, 2, msg_id++, line, strlen(line)+1);

        // build prompt from arena
        char *prompt = build_prompt(&arena);
        if (!prompt) {
            fprintf(stderr, "error building prompt\n");
            break;
        }

        // append current user input (already in prompt builder? we added it separately)
        // but we already appended to arena, and builder collected last 10 messages,
        // so the new user message will be included. No need to add again.

        // call ollama
        char *response_json = ollama_generate(prompt, ollama_model, ollama_endpoint);
        free(prompt);
        if (!response_json) {
            fprintf(stderr, "LLM call failed\n");
            break;
        }

        // parse response
        cJSON *root = cJSON_Parse(response_json);
        free(response_json);
        if (!root) {
            fprintf(stderr, "JSON parse error\n");
            break;
        }
        cJSON *resp_text = cJSON_GetObjectItem(root, "response");
        if (!cJSON_IsString(resp_text)) {
            cJSON_Delete(root);
            fprintf(stderr, "no 'response' field\n");
            break;
        }
        const char *assistant_msg = resp_text->valuestring;

        // check for tool call
        char *tool_name = NULL, *tool_args = NULL;
        char *after_tool = parse_tool_call(assistant_msg, &tool_name, &tool_args);
        if (tool_name && tool_args) {
            // execute tool
            char *tool_result = execute_tool(tool_name, tool_args);
            // store tool call and result
            blob_append(&arena, 4, msg_id++, assistant_msg, after_tool - assistant_msg);
            blob_append(&arena, 5, msg_id++, tool_result, strlen(tool_result)+1);
            // print result and continue (LLM will see it in next round)
            printf("\n[Tool %s] → %s\n", tool_name, tool_result);
            free(tool_result);
            free(tool_name);
            free(tool_args);
        } else {
            // normal assistant response
            printf("\n[ShadowClaw] %s\n", assistant_msg);
            blob_append(&arena, 3, msg_id++, assistant_msg, strlen(assistant_msg)+1);
        }

        cJSON_Delete(root);

        // save arena after each interaction (dirty flag is set)
        if (!arena_save(&arena, state_file)) {
            fprintf(stderr, "warning: could not save state\n");
        }
    }

    shadow_arena_destroy(&arena);
    return 0;
}