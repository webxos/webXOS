# WEBXOS 2025 DESTROYER FPV DRONE TEST ENVIRONMENT v1.0 Beta Test

## Overview
WEBXOS 2025 DESTROYER is a first-person view (FPV) drone simulation game designed for an immersive, fast-paced experience. Players navigate a drone through dynamic tunnels and vertical tubes, engaging enemies to score points while managing health, shields, and drone lives. The game supports a variety of input methods, including Xbox, PS5, USB controllers, joysticks, mouse/keyboard, mobile screen tilt, and popular drone controllers (e.g., DJI) via emulators. This beta version (v1.0) introduces enhanced tunnel patterns, a lightweight minimal menu, and a modernized user interface (UI).

## Features
- **Dynamic Environments**: Navigate procedurally generated tunnels and vertical tubes with increased pattern variation for a challenging and engaging flight experience.
- **Multi-Platform Controls**: Supports Xbox, PS5, USB controllers, joysticks, mouse/keyboard, mobile screen tilt, and drone controllers (e.g., DJI) via emulators.
- **Minimalist UI**: Streamlined interface with critical game stats (Wave, Enemies, Score, Health, Shield, Drones) and a lightweight menu optimized for drone controllers.
- **Game Mechanics**:
  - **Waves**: Progress through increasingly difficult waves of enemies.
  - **Scoring**: Earn points by destroying enemies.
  - **Health & Shield**: Manage your drone’s health (100) and shield (100) to survive.
  - **Drone Lives**: Start with 3 drones; lose one when health or shield reaches zero.
  - **Special Abilities**: Use "Lightspeed" for a speed boost and "Destroy All" to clear enemies on-screen.
- **Responsive Design**: Optimized for desktop and mobile browsers, with tilt-based controls for mobile devices.

## How to Play
1. **Start the Game**:
   - Click the "CLICK TO START" button on the main screen, or press any supported controller input to begin.
2. **Controls**:
   - **Mouse/Keyboard**:
     - **WASD/Arrows**: Move the drone (up, down, left, right).
     - **Left Click**: Fire weapon.
     - **Right Click**: Activate Lightspeed (speed boost).
   - **Gamepad (Xbox, PS5, USB)**:
     - **Left Stick**: Move the drone.
     - **Right Trigger**: Fire weapon.
     - **Left Trigger**: Activate Lightspeed.
   - **Mobile**:
     - **Screen Tilt**: Tilt device to move the drone.
     - **On-Screen Buttons**: Tap "Fire" or "Lightspeed" to activate.
   - **Drone Controllers (e.g., DJI)**:
     - Use emulator software to map controls (e.g., left stick for movement, trigger for fire).
     - Lightweight menu supports quick access to Fire, Lightspeed, Destroy All, and End Game.
3. **Gameplay**:
   - Navigate through procedurally generated tunnels and vertical tubes with varied patterns.
   - Destroy enemies to increase your score and progress through waves.
   - Monitor health and shield levels; losing all health or shield consumes a drone life.
   - Use "Lightspeed" to evade obstacles or enemies and "Destroy All" to clear the screen of threats.
   - The game ends when all drones are lost or the player selects "End Game."
4. **Controller Detection**:
   - The UI displays "CONTROLLER: DETECTED" when a gamepad or drone controller is connected; otherwise, it shows "CONTROLLER: NOT DETECTED."

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/webxos-destroyer-2025.git
   ```
2. **Run Locally**:
   - Open `destroyer.html` in a modern web browser (Chrome, Firefox, or Edge recommended).
   - Ensure an internet connection for any external dependencies (e.g., controller libraries).
3. **Dependencies**:
   - No external installations are required for basic gameplay.
   - For drone controllers, install an emulator compatible with your device (e.g., DJI-compatible software).
   - Ensure your browser supports WebGL and JavaScript for rendering and controller input.

## Controller Setup
- **Xbox/PS5/USB Controllers**:
  - Connect via USB or Bluetooth; the game automatically detects standard gamepads.
  - Map controls as follows:
    - Left Stick: Movement.
    - Right Trigger: Fire.
    - Left Trigger: Lightspeed.
- **Joystick/Drone Controllers**:
  - Use an emulator to map inputs to game controls.
  - Recommended emulators: QGroundControl or custom DJI-compatible software.
  - Map movement to the left stick and assign fire/lightspeed to triggers or buttons.
- **Mobile**:
  - Enable device orientation in browser settings for tilt controls.
  - Use on-screen buttons for Fire and Lightspeed.
- **Mouse/Keyboard**:
  - No setup required; uses standard WASD/Arrows and mouse clicks.

## Development Ideas
- **Enhanced Environments**: Add more complex tunnel patterns, such as spiraling paths or branching routes, with dynamic obstacles.
- **Multiplayer Mode**: Introduce co-op or competitive modes for online play.
- **Customizable Drones**: Allow players to upgrade drones with different weapons, shields, or speed boosts.
- **Leaderboards**: Implement a global leaderboard for high scores, stored via a backend API.
- **VR Support**: Add compatibility with VR headsets for a fully immersive FPV experience.
- **Controller Calibration**: Include an in-game menu for custom mapping of drone controller inputs.
- **Haptic Feedback**: Add rumble support for gamepads and vibration for mobile devices to enhance immersion.

## Known Issues
- **Controller Detection**: Some USB controllers may not be detected in certain browsers; ensure your browser supports the Gamepad API.
- **Mobile Tilt Sensitivity**: Tilt controls may require calibration for optimal responsiveness on some devices.
- **Drone Controller Compatibility**: Emulator performance varies; test with your specific controller setup.
- **Performance**: High tunnel complexity may impact performance on low-end devices; consider lowering graphics settings in future updates.

## Contributing
1. Fork the repository and create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
2. Make changes and test locally.
3. Submit a pull request with a detailed description of your changes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For issues, feature requests, or feedback, open an issue on the GitHub repository or contact the developer at [your-email@example.com].