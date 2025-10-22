# LAUNCH SEQUENCE Training Guide

## Overview
LAUNCH SEQUENCE by webXOS 2025 is a web-based application for training and testing drone AI models using a quantum neural network (QNN) simulation. This guide focuses on **Training Mode**, which enables users to develop and optimize drone AI models in a 3D environment with real-time metrics and antifragility controls.

## Getting Started
1. **Open the Application**: Load `launch.html` in a modern web browser with WebGL 2.0 support.
2. **Select Training Mode**: Click the **TRAIN** tab in the header to enter Training Mode.
3. **Choose Drone Type**: Select **FPV** or **Long Range** from the left panel's drone type selector.

## Training Mode
Training Mode is designed to develop drone AI models by adjusting parameters, monitoring progress, and visualizing QNN activity. The interface includes controls for learning rate, creativity, memory, and antifragility, with real-time metrics and a 3D simulation.

### Left Panel: Training Controls
- **Drone Type Selector**:
  - **FPV**: Optimized for agile, short-range navigation.
  - **Long Range**: Suited for extended distances and endurance.
  - Click to switch; the active type is highlighted in green.
- **Sliders**:
  - **Learning Rate (0.01–1.0)**: Controls how quickly the model learns. Lower values ensure stability; higher values speed up learning but may cause instability.
  - **Creativity (0.0–1.0)**: Balances exploration (new strategies) vs. exploitation (refining known strategies). Higher values encourage innovative paths.
  - **Training Memory (1–4096 MB)**: Sets the memory limit for training data. Higher limits allow more data points but increase resource usage.
  - **X-Antifragility (Stability, 0.0–1.0)**: Enhances model stability, reducing errors like hallucinations. Higher values improve robustness.
  - **Y-Antifragility (Adaptability, 0.0–1.0)**: Improves adaptability to stressors (e.g., complex environments). Higher values enhance flexibility.
- **Training Actions**:
  - **Start Endless Training**: Initiates continuous training until paused.
  - **Pause**: Temporarily stops training, preserving progress.
  - **Reset**: Clears all training data and metrics.
  - **Export Training Log**: Downloads training data and logs as a JSON file.

### Right Panel: Training Status
- **Training Stats**:
  - **Training Rounds**: Total number of training sessions (including cycles).
  - **Total Time**: Cumulative training duration (hours and minutes).
  - **Best Score**: Highest accuracy achieved during training.
- **Progress Bar**:
  - Displays training progress as a percentage (0–100%).
- **Metrics Grid**:
  - **Accuracy**: Current model accuracy (0–95%).
  - **Active Nodes**: Number of active QNN nodes (20–80).
- **Training Data**:
  - **Model Name**: Input field to name the model for export.
  - **Epoch**: Number of training iterations completed.
  - **Loss**: Current model error rate (lower is better).
  - **Complexity**: Model complexity level (Low, Medium).
  - **Response Time**: Model response speed in milliseconds.
  - **Robustness**: Antifragility performance (0–100%).
  - **Stress Response**: System stress level (0.0–1.0; lower indicates better handling).
  - **Training Cycles**: Number of training iterations.
  - **Data Points**: Accumulated training data points.
  - **Memory Used**: Current memory usage in MB.

### QNN Grid (Top-Right Overlay)
- Visualizes neural network activity in a 10x10 grid:
  - **Inactive (Black)**: Dormant nodes.
  - **Active (Green)**: Currently active nodes.
  - **Training (Yellow)**: Nodes undergoing training.
  - **Learned (Dark Green)**: Successfully trained nodes.
- Updates dynamically based on learning rate, creativity, and training progress.

### Antifragility Grid (Top-Left Overlay)
- Displays a 5x5 grid representing the model’s antifragility:
  - **Stable (Green)**: Cells indicating robust performance.
  - **Stressed (Red)**: Cells under stress, indicating potential instability.
  - **Green Dot**: Represents the model’s current X-Antifragility (Stability) and Y-Antifragility (Adaptability) position, moving within the grid based on slider values.

### Console Commands
- Access the console at the bottom of the center panel to issue commands and monitor progress.
- **Available Commands**:
  - `/help`: Lists all available commands.
  - `/status`: Displays current system status (mode, drone type, training state).
  - `/train [start|stop|reset]`: Controls training (start, pause, or reset).
  - `/mode [train|test]`: Switches between Train and Test modes.
  - `/drone [fpv|longrange]`: Changes drone type.
  - `/antifragility [x] [y]`: Sets X-Antifragility and Y-Antifragility (0–1).
  - `/export`: Opens the export modal to save training data.
  - `/copylog`: Copies console log to clipboard.
- **Usage**: Type commands in the input field and press **SEND** or Enter.

## Usage Tips
- **Learning Rate**: Start with 0.1 for stability. Increase gradually (e.g., 0.3–0.5) if loss decreases consistently, but monitor for instability.
- **Creativity**: Use 0.5–0.7 for balanced exploration. Higher values (0.8–1.0) are useful for complex environments but may slow convergence.
- **Memory Limit**: Set to 128–256 MB for initial training. Increase for larger datasets if accuracy plateaus.
- **Antifragility**:
  - Increase **X-Antifragility** (>0.7) to reduce errors in stable environments.
  - Increase **Y-Antifragility** (>0.7) for better adaptation to dynamic or challenging conditions.
- **Monitoring**: Check the console for training logs (e.g., epoch updates, antifragility metrics) and use `/status` to review settings.
- **Saving Progress**: Use **Export Training Log** or `/export` regularly to save training data to a JSON file, stored in localStorage between sessions.
- **QNN Grid**: Watch for increased **Active** and **Learned** nodes as indicators of training progress.
- **Antifragility Grid**: Adjust sliders to keep the green dot in a balanced position (center of the grid) for optimal robustness and adaptability.

## System Requirements
- Modern web browser (Chrome, Firefox, Edge).
- WebGL 2.0 support for 3D visualization.
- Internet connection for loading Three.js library.
- Minimum 4GB RAM for smooth performance.

## Notes
- Training data persists in localStorage between sessions unless reset.
- Antifragility controls enhance model resilience, with **Robustness** and **Stress Response** metrics reflecting performance under stress.
- The 3D scene visualizes the drone in a simulated environment, updated dynamically during training.
- Use the console to track detailed training progress and issue commands for fine-tuned control.