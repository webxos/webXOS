#include <emscripten.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Neural Network class for simple feedforward inference
class NeuralNetwork {
private:
    std::vector<std::vector<float>> weights_input_hidden;
    std::vector<std::vector<float>> weights_hidden_output;
    std::vector<float> bias_hidden;
    std::vector<float> bias_output;
    int input_size, hidden_size, output_size;

    // Sigmoid activation function
    float sigmoid(float x) {
        return 1.0f / (1.0f + expf(-x));
    }

    // Matrix multiplication for layer computation
    std::vector<float> matmul(const std::vector<float>& input, const std::vector<std::vector<float>>& weights, const std::vector<float>& bias) {
        std::vector<float> result(weights.size(), 0.0f);
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < input.size(); ++j) {
                result[i] += input[j] * weights[i][j];
            }
            result[i] += bias[i];
            result[i] = sigmoid(result[i]);
        }
        return result;
    }

public:
    NeuralNetwork(int input, int hidden, int output) : input_size(input), hidden_size(hidden), output_size(output) {
        // Initialize weights and biases with random values
        srand(static_cast<unsigned>(time(0)));
        weights_input_hidden.resize(hidden);
        for (auto& row : weights_input_hidden) {
            row.resize(input);
            for (auto& w : row) w = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        }
        weights_hidden_output.resize(output);
        for (auto& row : weights_hidden_output) {
            row.resize(hidden);
            for (auto& w : row) w = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        }
        bias_hidden.resize(hidden, 0.0f);
        bias_output.resize(output, 0.0f);
        for (auto& b : bias_hidden) b = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        for (auto& b : bias_output) b = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    // Inference function
    std::vector<float> predict(const std::vector<float>& input) {
        // Input to hidden layer
        std::vector<float> hidden = matmul(input, weights_input_hidden, bias_hidden);
        // Hidden to output layer
        return matmul(hidden, weights_hidden_output, bias_output);
    }
};

// Export functions to JavaScript
extern "C" {
    EMSCRIPTEN_KEEPALIVE
    NeuralNetwork* create_network(int input_size, int hidden_size, int output_size) {
        return new NeuralNetwork(input_size, hidden_size, output_size);
    }

    EMSCRIPTEN_KEEPALIVE
    void destroy_network(NeuralNetwork* nn) {
        delete nn;
    }

    EMSCRIPTEN_KEEPALIVE
    float* predict(NeuralNetwork* nn, float* input, int input_size, int* output_size) {
        std::vector<float> input_vec(input, input + input_size);
        std::vector<float> output = nn->predict(input_vec);
        *output_size = output.size();
        float* result = new float[output.size()];
        for (size_t i = 0; i < output.size(); ++i) {
            result[i] = output[i];
        }
        return result;
    }

    EMSCRIPTEN_KEEPALIVE
    void free_result(float* result) {
        delete[] result;
    }
}
