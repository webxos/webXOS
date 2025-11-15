# webXOS IoT RADAR HUD Template Guide

This guide provides an overview of the webXOS IoT RADAR HUD Template, a Three.js-based interface for visualizing IoT objects in a 3D radar-like environment with a retro CRT aesthetic.

## Overview
The template creates an interactive HUD with:
- A 3D scene displaying a central face mesh and orbiting objects/entities.
- A 2D radar panel showing object positions.
- A terminal for object selection and management.
- A status panel for real-time metrics.
- Controls for mode switching, scene reset, and object scanning.
- A tag editor for labeling objects.

## Features
- **3D Visualization**: Displays a neon-green face mesh with orbiting friendly (green) and hostile (red) objects, plus red cube entities.
- **Radar Panel**: Shows objects and entities relative to the face's direction, with a rotating pulse.
- **Terminal Panel**: Lists objects/entities for selection and interaction.
- **Status Panel**: Shows FPS, object count, face direction, mode, and selected object.
- **Controls**:
  - Toggle between MIRROR and SCANNER modes.
  - Reset the scene.
  - Scan for new objects.
- **Tag Editor**: Edit labels for selected objects.
- **Responsive Design**: Optimized for desktop and mobile.

## Setup
1. **Include Dependencies**:
   - The template uses Three.js (v0.158.0) via CDN:
     ```html
     <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.min.js"></script>
     ```
2. **Host the File**:
   - Save the HTML file and serve it via a web server (local or remote) to avoid CORS issues with Three.js.
3. **Browser Compatibility**:
   - Works on modern browsers supporting WebGL (Chrome, Firefox, Safari, Edge).

## Usage
1. **Open the Interface**:
   - Load the HTML file in a browser to start the application.
2. **Interact with Controls**:
   - **RADAR 1 (Mode Toggle)**: Switches between MIRROR (default) and SCANNER modes, regenerating objects.
   - **RADAR 2 (Reset)**: Resets face direction, regenerates objects/entities, and clears selection.
   - **ADD OBJECTS (Scan)**: Adds 2-4 new objects with random positions and types.
3. **Select Objects**:
   - Click an object or entity in the terminal to select it.
   - Selected objects turn blue (objects) or lighter red (entities).
   - The tag editor opens to edit the object’s label.
4. **Edit Tags**:
   - In the tag editor, enter a new label and click **SAVE TAG** to update, or **CANCEL** to close.
5. **Monitor Status**:
   - **FRAMES**: Displays FPS (updated every second).
   - **OBJECTS**: Shows total objects in the scene.
   - **FACING**: Indicates face direction (-30° to 30°).
   - **MODE**: Shows current mode (MIRROR or SCANNER).
   - **SELECTED**: Displays the selected object’s label or “NONE”.
6. **Radar Interaction**:
   - Objects within 10 units and entities within 15 units appear on the radar.
   - Green dots represent friendly objects, red dots represent hostile objects/entities.
   - A yellow line indicates face direction.
7. **Mouse Control**:
   - Move the mouse horizontally to adjust the face’s rotation (-30° to 30°).

## Customization
1. **Styling**:
   - Modify the CSS in the `<style>` section to change colors, fonts, or layout.
   - Example: Change the grid color by editing `.matrix-grid` background.
2. **Object Generation**:
   - Adjust `generateObjects()` and `generateEntities()` in the `<script>` section to change:
     - Number of objects (`numObjects`).
     - Object types (box, sphere, cone, cylinder).
     - Distance ranges or colors.
3. **Radar Behavior**:
   - Modify `updateRadar()` to adjust radar range, pulse speed (`radarPulseAngle`), or dot sizes.
4. **Face Geometry**:
   - Edit `faceGeometry` vertices and indices to change the central face mesh.
5. **Performance**:
   - Reduce `numObjects` or simplify geometries in `generateObjects()` for better performance on low-end devices.
   - Adjust `renderer.setPixelRatio(1)` to balance quality and performance.

## Code Structure
- **HTML**:
  - Containers: `#appContainer`, `#visualizationContainer`, `#ideInterface`.
  - UI Elements: `.header`, `.status-panel`, `.terminal-panel`, `.radar-panel`, `.main-controls`, `.tag-editor`.
- **CSS**:
  - Retro CRT aesthetic with neon-green accents.
  - Responsive design for mobile (`@media (max-width: 767px)`).
  - Animations: Glitch effect (`.header`), scanline (`.crt-overlay`).
- **JavaScript**:
  - **Three.js Setup**: Initializes scene, camera, renderer, and face mesh (`init3D`, `createFaceMesh`).
  - **Object Management**: Generates and animates objects/entities (`generateObjects`, `generateEntities`, `updateObjects`).
  - **Radar**: Draws radar circles, pulse, and object dots (`updateRadar`).
  - **UI Updates**: Handles terminal (`updateTerminal`), status, and tag editor (`selectObject`).
  - **Event Listeners**: Manages button clicks, mouse movement, and window resize (`setupEventListeners`).
  - **Animation Loop**: Updates scene, radar, and FPS (`animate`, `calculateFPS`).
  - **Cleanup**: Disposes resources on unload (`cleanup`).

## Troubleshooting
- **Blank Canvas**:
  - Ensure Three.js CDN is accessible.
  - Check browser console for WebGL errors.
- **Low FPS**:
  - Reduce `numObjects` or `numEntities`.
  - Set `antialias: false` in renderer or lower `setPixelRatio`.
- **Non-Responsive UI**:
  - Verify CSS media queries for mobile.
  - Check `pointer-events` on UI elements.
- **Objects Not Appearing**:
  - Ensure `generateObjects()` is called (triggered on init, mode toggle, reset, or scan).
  - Check radar range in `updateRadar()`.

## Notes
- The template is performance-optimized with low-poly geometries and minimal lighting.
- The face direction is simulated via mouse movement and a gentle sway (`updateFaceDirection`).
- Cleanup on page unload prevents memory leaks (`cleanup`).

For further customization or integration with real IoT data, modify the `generateObjects()` function to pull from an API or sensor data.
