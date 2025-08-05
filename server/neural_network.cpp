#include <emscripten.h>

extern "C" {
    EMSCRIPTEN_KEEPALIVE
    void _create_network() {
        // Initialize neural network logic
    }

    EMSCRIPTEN_KEEPALIVE
    void _destroy_network() {
        // Cleanup logic
    }

    EMSCRIPTEN_KEEPALIVE
    float _predict(float input) {
        return input * 2; // Example prediction
    }

    EMSCRIPTEN_KEEPALIVE
    void _free_result(float* ptr) {
        if (ptr) {
            free(ptr);
        }
    }
}
