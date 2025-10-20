# LAUNCH SEQUENCE Guide

## Overview
LAUNCH SEQUENCE by webXOS 2025 is a web-based application for training and testing drone AI models using a quantum neural network (QNN) simulation. It features a 3D environment for visualizing drone navigation, training controls, and real-time metrics.

## Features
- **Modes**: Train and Test modes for QNN model development and evaluation.
- **Drone Types**: Tiny Whoop, Long Range, FPV.
- **3D Visualization**: Interactive 3D scene with drone, obstacles, and gates.
- **Training Controls**: Adjust learning rate, creativity, and memory limit.
- **Testing Controls**: Manual or 24/7 Auto-Pilot testing with customizable duration and obstacle density.
- **Console**: Command-line interface for system control and monitoring.
- **QNN Grid**: Visual representation of neural network activity.
- **Export/Import**: Save and load training models and logs.

## Getting Started
1. **Open the Application**: Load `launch.html` in a modern web browser.
2. **Select Mode**:
   - Click **TRAIN** or **TEST** tab in the header.
   - TRAIN mode for model training; TEST mode for model evaluation.
3. **Choose Drone Type**: Select Tiny Whoop, Long Range, or FPV from the left panel.

## Training Mode
### Controls
- **Drone Type**: Choose drone type (Tiny Whoop, Long Range, FPV).
- **Sliders**:
  - **Learning Rate** (0.01–1.0): Controls model learning speed.
  - **Creativity** (0.0–1.0): Adjusts exploration vs. exploitation.
  - **Training Memory** (64–1024 MB): Sets memory limit for training data.
- **Actions**:
  - **Start Endless Training**: Begins continuous training.
  - **Pause**: Stops training temporarily.
  - **Reset**: Clears all training data.
  - **Export Training Log**: Downloads training data as JSON.

### Metrics (Right Panel)
- **Model Name**: Input field for naming the model.
- **Epoch**: Training iterations completed.
- **Loss**: Model error rate.
- **Complexity**: Model complexity level (Low, Medium, High, Very High).
- **Response Time**: Model response speed.
- **Training Cycles**: Number of training iterations.
- **Data Points**: Accumulated training data.
- **Memory Used**: Current memory usage.
- **Training Rounds**: Total training sessions.
- **Total Time**: Training duration.
- **Best Score**: Highest accuracy achieved.
- **Accuracy**: Current model accuracy.
- **Active Nodes**: Number of active QNN nodes.
- **Training Progress**: Percentage of training completion.

### QNN Grid
- Visualizes neural network activity:
  - **Inactive** (black): Dormant nodes.
  - **Active** (green): Active nodes.
  - **Training** (yellow): Nodes in training.
  - **Learned** (dark green): Trained nodes.

## Test Mode
### Controls
- **Actions**:
  - **Import QNN**: Load a previously exported model.
  - **Manual**: Control drone manually using keyboard.
  - **24/7 Auto-Pilot**: Run continuous autonomous testing.
  - **Stop Auto-Pilot**: Halt autonomous testing.
- **Sliders**:
  - **Test Duration** (0–60 min): Set test duration (0 for continuous).
  - **Obstacle Density** (Very Low to Very High): Adjust obstacle count.

### Metrics (Right Panel)
- **Accuracy**: Model accuracy during testing.
- **Speed**: Drone speed (km/h).
- **Battery**: Remaining battery percentage.
- **Lap Progress**: Percentage of current lap completed.
- **Laps Completed**: Number of completed laps.
- **Success Rate**: Percentage of gates passed successfully.
- **Learning Rate**: Current learning rate from training.
- **Test Score**: Overall test performance score.
- **Environment**: Current testing environment (A, B, C, D).
- **Gates Passed**: Number of gates passed.
- **Flight Time**: Total flight duration.
- **Distance**: Total distance traveled.
- **Collisions**: Number of obstacle collisions.

### Manual Controls
- **W/S**: Move forward/backward.
- **A/D**: Move left/right.
- **R/F**: Move up/down.

## 3D Scene Controls
- **Mouse Drag**: Rotate camera around the scene.
- **Mouse Wheel**: Zoom in/out (5–50 units distance).

## Console Commands
- **/help**: List available commands.
- **/status**: Display system status (mode, training, auto-pilot, etc.).
- **/export**: Export training log (TRAIN mode only).
- **/import**: Import model (TEST mode only).
- **/train [start|stop|reset]**: Control training.
- **/test [manual|autopilot]**: Set test mode.
- **/autopilot [start|stop]**: Control auto-pilot.
- **/memory [size]**: Set memory limit (64–1024 MB).
- **/log**: Show recent training log entries.
- **/clear**: Clear console output.

## Usage Tips
- **Training**: Start with a low learning rate (0.1) and adjust based on loss/accuracy metrics.
- **Testing**: Import a trained model before using Auto-Pilot. Adjust obstacle density for difficulty.
- **Monitoring**: Use the console to track progress and issue commands.
- **Saving Progress**: Export training logs regularly to avoid data loss.
- **Performance**: Ensure a modern browser with WebGL support for optimal 3D rendering.

## System Requirements
- Modern web browser (Chrome, Firefox, Edge).
- WebGL 2.0 support.
- Internet connection for loading Three.js library.
- Minimum 4GB RAM for smooth performance.

## Notes
- Training data is saved to localStorage and persists between sessions.
- Auto-Pilot mode requires a loaded model to function.
- The 3D scene includes four environments (A, B, C, D) with varying obstacle layouts.
- The application simulates a QNN for drone navigation training and testing.