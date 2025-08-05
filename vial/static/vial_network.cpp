#include <emscripten.h>
#include <vector>
#include <random>

std::vector<std::vector<double>> vial_network;

extern "C" {
    EMSCRIPTEN_KEEPALIVE
    void create_vial_network() {
        vial_network.clear();
        std::vector<double> initial_weights(5, 0.0);
        vial_network.push_back(initial_weights);
    }

    EMSCRIPTEN_KEEPALIVE
    double test_vial(double input) {
        if (vial_network.empty()) return 0.0;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        return input * dis(gen);
    }
}
