# MIRROR IDE BETA TEST - BTMESH GIMBAL DRONE Train Guide

MIRROR IDE is a web-based platform for simulating and controlling First-Person View (FPV) drones using Bluetooth Mesh networking. It provides a 3D visualization environment, terminal interface, webcam integration for face detection, accelerometer support, and an ESP32 SDK for programming drones with JavaScript.

## Features
- **Test Mode**: Simulate drone behavior in a 3D environment with friendly (green) and hostile (red) objects.
- **Live Mode**: Connect to a Bluetooth Mesh network for real-time drone control.
- **Terminal Interface**: Issue commands for full drone control and debugging.
- **Webcam Sync**: Utilize face detection and accelerometer data for user tracking and motion input.
- **ESP32 SDK**: Program drones using JavaScript in a secure execution environment.
- **Controls**: Virtual joysticks for throttle/yaw (left) and pitch/roll (right), plus buttons for mode switching, arming, and emergency actions.
- **Scene Management**: Click/tap to move objects, import/export scenes as JSON.
- **Accessibility**: ARIA labels and focus styles for improved usability.

## Use Cases
1. **Drone Training**:
   - Practice FPV drone piloting in Test Mode without risking physical hardware.
   - Simulate scenarios with friendly and hostile objects to develop navigation skills.
2. **Bluetooth Mesh Development**:
   - Test Bluetooth Mesh networking for drone swarms or IoT devices in Live Mode.
   - Monitor node connections, signal strength, and data rates.
3. **Robotics Programming**:
   - Use the ESP32 SDK to write and test JavaScript code for drone behavior.
   - Experiment with automation scripts for tasks like obstacle avoidance or coordinated flight.
4. **Research and Prototyping**:
   - Import/export JSON scenes to replicate specific environments for testing.
   - Leverage webcam and accelerometer data for human-drone interaction studies.
5. **Education**:
   - Teach concepts of drone control, 3D visualization, and networked systems.
   - Use the terminal to introduce command-line interfaces and scripting.

## Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/mirror-ide.git
   ```
2. **Serve the Application**:
   - Use a local server: `python -m http.server 8000`.
   - Open `http://localhost:8000/mirror2.html` in a modern browser (Chrome/Edge recommended).
3. **Dependencies**:
   - No local installation needed; dependencies (Three.js, MediaPipe) are loaded via CDNs.
   - Requires a webcam and accelerometer-compatible device (e.g., smartphone or laptop) for full functionality.
   - Ensure HTTPS or localhost for webcam and service worker support.
4. **Browser Permissions**:
   - Grant camera access for webcam features.
   - Allow motion sensor access on mobile devices (iOS requires explicit permission).

## In-Depth Guide
### Interface Overview
- **Matrix Grid**: A pixelated green grid overlay for a retro aesthetic.
- **CRT Overlay**: Simulates a scanline effect for visual style.
- **Visualization Container**: Displays the 3D scene in Test Mode (canvas-based).
- **Live Scene**: Shows a simplified grid and status in Live Mode.
- **IDE Interface**:
  - **Beta Badge**: Indicates MIRROR IDE v3.0.
  - **View Controls** (top-left):
    - `FS`: Toggle fullscreen.
    - `FPV/3PV`: Switch between First-Person View and Third-Person View.
    - `GIMBAL`: Enable filming mode (drone focuses on user).
  - **Top Controls** (center):
    - `GUIDE`: Open usage instructions.
    - `TERM`: Access terminal for commands.
    - `STAT`: View system status and manage scenes.
    - `SDK`: Configure ESP32 JavaScript programming.
  - **Mode Controls** (bottom-center):
    - `TEST`: Simulation mode (default).
    - `LIVE`: Bluetooth Mesh mode.
  - **Main Controls** (bottom):
    - `WEBCAM`: Toggle camera for face detection.
    - `ARM`: Arm/disarm drone for flight.
    - `MESH`: Manage Bluetooth Mesh network.
    - `EMERGENCY`: Trigger Return-to-Launch (RTL) and landing (disabled if not armed).
  - **Zoom Controls** (top-right):
    - `+`: Zoom in.
    - `-`: Zoom out.
    - `R`: Reset view and zoom.
  - **Joysticks** (bottom):
    - Left: Throttle (up/down) and yaw (left/right).
    - Right: Pitch (forward/back) and roll (left/right).
  - **HUD Warning**: Shows "USER MISSING" if face detection fails.
  - **Copyright**: Displays © 2025 webXOS.

### Popups
- **Guide**:
  - Lists key controls and their functions (e.g., TEST/LIVE, MESH, EMERGENCY).
  - Color coding: Green = friendly, Red = hostile.
- **Terminal**:
  - Displays logs (success, error, warning, prompt, info).
  - Input field for commands; press Enter to execute, Escape to close.
- **Status**:
  - Shows mode, altitude, battery, arm status, view, mesh status, and drone count.
  - Buttons to export/import JSON scenes.
- **Webcam**:
  - Displays live feed and calibration status (e.g., "LIVE - WEBCAM ACTIVE").
  - `STOP WEBCAM` button to disable.
- **Bluetooth Mesh**:
  - Shows nodes, connected devices, signal strength, and data rate.
  - Buttons: `SCAN DEVICES`, `CONNECT MESH`, `DISCONNECT`.
- **SDK Config**:
  - Displays ESP32 connection status and JavaScript input area.
  - Buttons: `CONNECT ESP32`, `RUN CODE`.

### Key Actions
1. **Switching Modes**:
   - **Test Mode**: Click `TEST` for simulation. The 3D scene shows the drone, user (cyan cube), friendly (green spheres), and hostile (red cubes) objects.
   - **Live Mode**: Click `LIVE` to open the Mesh popup. Connect to a Bluetooth Mesh network for real-time control.
2. **Arming the Drone**:
   - Click `ARM` to enable flight controls. The button highlights when active.
   - Disarm to disable movement; `EMERGENCY` is disabled when not armed.
3. **Flying the Drone**:
   - Use the left joystick for throttle (vertical) and yaw (horizontal).
   - Use the right joystick for pitch (forward/back) and roll (left/right).
   - In Test Mode, the drone moves within a 60x60x60 bounding box; altitude is 0.5–25m.
   - In Live Mode, movement is sent to the mesh network (simulated).
4. **Moving Objects**:
   - In Test Mode, click/tap the canvas to move the drone, user, or objects to the clicked position.
   - Useful for repositioning friendly/hostile objects during simulation.
5. **Emergency Stop**:
   - Click `EMERGENCY` (when armed) to trigger RTL and landing.
   - The drone follows a safe path to the user’s position, avoiding objects, then lands.
6. **Webcam and Face Detection**:
   - Click `WEBCAM` to start the camera. The popup shows the feed and status.
   - Face detection tracks the user; "USER MISSING" appears if no face is detected.
   - Stop via the `STOP WEBCAM` button or close the popup.
7. **Bluetooth Mesh**:
   - Click `MESH` to open the Mesh popup.
   - `SCAN DEVICES`: Simulates finding drones, sensors, and user devices.
   - `CONNECT MESH`: Establishes a connection (simulated after 2 seconds).
   - `DISCONNECT`: Returns to Test Mode.
8. **ESP32 SDK**:
   - Click `SDK` to open the popup.
   - `CONNECT ESP32`: Simulates connecting to an ESP32 (1.5-second delay).
   - Enter JavaScript in the textarea and click `RUN CODE`. Code is validated to block dangerous operations (e.g., `eval`, `fetch`).
   - Example code:
     ```javascript
     drone.x += 2; // Move drone 2 units right
     add("Moved right", "success");
     ```
9. **Scene Management**:
   - In the Status popup, click `EXPORT JSON` to download the scene (friendly/hostile object positions).
   - Click `IMPORT JSON` to load a `.json` file, replacing the current scene.
10. **Terminal Commands**:
    - Open via `TERM` or type commands in the terminal popup.
    - Key commands:
      - `test`, `live`: Switch modes.
      - `arm`, `disarm`: Toggle drone state.
      - `takeoff`: Increase altitude by 5m.
      - `land`: Return to user and land.
      - `flyto x y z`: Move to coordinates (e.g., `flyto 10 5 -10`).
      - `emergency`: Trigger RTL and landing.
      - `adddrone`: Add a new drone.
      - `switchdrone [n]`: Switch to drone `n` (e.g., `switchdrone 2`).
      - `listdrones`: Show all drones and positions.
      - `mesh`, `scan`, `connect`, `disconnect`: Mesh controls.
      - `webcam`, `stopwebcam`: Webcam toggle.
      - `sdk`, `connectsdk`, `runcode`: SDK controls.
      - `status`: Display system info.
      - `clear`: Clear terminal logs.
      - `help`: List all commands.
    - Example: `flyto 5 3 0` moves the drone to (5, 3, 0).

### Tips
- **Performance**: Use a modern browser and device for smooth 3D rendering. Lower pixel ratio (`ren.setPixelRatio(Math.min(window.devicePixelRatio, 2))`) ensures performance on high-DPI screens.
- **Webcam**: Ensure good lighting for reliable face detection.
- **Terminal**: Use `clear` to manage logs; `help` for quick reference.
- **Joystick Sensitivity**: Adjust movement by tweaking `deltaTime` multipliers in the `animate` function (e.g., `pitch * deltaTime * 7`).
- **Scene Design**: Export a scene, edit the JSON, and import to create custom layouts.

## Limitations
- **Simulation-Based**: Live Mode and Mesh connections are simulated; real Bluetooth Mesh requires hardware integration.
- **Browser Compatibility**: Webcam and accelerometer may not work on older browsers or devices.
- **Code Security**: SDK restricts certain JavaScript operations for safety (e.g., no `eval` or network requests).
- **Single Drone Focus**: Only one drone is controlled at a time, though multiple can be added.

You can use an Android device for both sending high-level commands and receiving telemetry data to command a drone or maintain a strict gimbal path. 

#Communication between the Android GUI and the ESP32 drone

#Android GUI via Bluetooth mesh

 Using Android Telemetry for Gimbal Path Control
This more advanced method involves using the Android device's internal sensors (gyroscope, accelerometer, GPS) as telemetry data to guide the drone or a specific gimbal path.
Implementation Details:

    Sensor Data Acquisition: The custom Android app (built with MIT App Inventor or Android Studio) needs to access the phone's sensors and format this data into a packet (e.g., a comma-separated string like "roll:1.2,pitch:5.4").
    Bluetooth Transmission: This sensor data is streamed from the Android phone to the ESP32
    via Bluetooth.
    ESP32
    Processing: The ESP32
    receives and parses this real-time data. It uses the phone's orientation data as a reference for the drone's or gimbal's target orientation. The ESP32
    's flight/gimbal control logic (likely using a PID loop with its own MPU6050 sensor data) works to match the physical device's orientation to the phone's orientation.
    Gimbal Path: For a "strict gimbal path," the Android device's movements become the control input, allowing intuitive "point-and-shoot" or "follow-me" type control. 

### Error Testing and Fixes for `deepseek_html_20251114_a38644.html`

The provided HTML file (`deepseek_html_20251114_a38644.html`) is a modified version of the MIRROR IDE, but it contains several issues that could affect functionality, particularly for Android integration and HC-SR04 support. Below is a concise error analysis and necessary fixes, followed by instructions for Android integration.

#### Error Analysis
1. **Top Banner Overlap**:
   - The `.top-banner` (new addition) is positioned at `top: 0` with `height: 40px`, overlapping existing controls (`top-controls`, `zoom-controls`, `view-controls`) also positioned at `top: 8px`. This causes visual and interaction issues.
   - **Fix**: Adjust control positions to account for the banner.

2. **Missing HC-SR04 and Radar Support**:
   - The file lacks the HC-SR04 integration, radar visualization, and gimbal agent logic from `mirror3.html`. These are critical for the requested functionality.
   - **Fix**: Port the HC-SR04, radar, and service worker logic from `mirror3.html`.

3. **Webcam Mesh Styles Incomplete**:
   - The `.webcam-mesh-container` and related styles are truncated, potentially breaking webcam visualization.
   - **Fix**: Ensure complete styles are included.

4. **Service Worker Limitations**:
   - The `swScript` lacks the gimbal agent logic for processing HC-SR04, face detection, and orientation data.
   - **Fix**: Update `swScript` to include gimbal calculations.

5. **HTTPS Requirement**:
   - Web Bluetooth and webcam features require HTTPS, but the manifest’s `start_url` is set to `./`, which may fail on non-localhost deployments.
   - **Fix**: Update `start_url` for HTTPS.

6. **Android Compatibility**:
   - No explicit handling for Android’s `DeviceOrientationEvent` permission or BLE connectivity, which are critical for telemetry integration.
   - **Fix**: Add orientation permission logic and BLE support.

#### Fixes
Below are the specific modifications to address these errors, incorporating HC-SR04 support, radar, gimbal agent, and Android integration.

---

### Step 1: Fix Control Positioning
Adjust control containers to avoid overlap with the top banner.

**Location**: `<style>` tag, update `.top-controls`, `.zoom-controls`, `.view-controls`.

**Code**:
```css
.top-controls { top: 45px; }
.zoom-controls { top: 45px; }
.view-controls { top: 45px; }
```

---

### Step 2: Add HC-SR04 UI and Radar
Add UI elements for HC-SR04 data, BLE connection, calibration, and radar visualization in the SDK popup.

**Location**: `<div class="popup" id="sdkPopup">`, after `<textarea id="codeInput" ...>`.

**Code**:
```html
<div class="popup-item">SENSOR: <span id="sensorData">DISCONNECTED</span></div>
<button class="popup-btn" id="connectBLE" aria-label="Connect to BLE Sensor">CONNECT BLE</button>
<button class="popup-btn" id="calibrateSensor" aria-label="Calibrate Sensor with Webcam">CALIBRATE</button>
<canvas id="radarCanvas" style="width:100%;height:100px;border:2px solid #0f0;background:#000;"></canvas>
```

**CSS Update** (in `<style>`, after `.popup-btn`):
```css
#radarCanvas {
    display: block;
    margin-top: 8px;
}
.radar-container {
    position: relative;
    width: 100%;
    height: 100px;
}
```

---

### Step 3: Complete Webcam Mesh Styles
Ensure the `.webcam-mesh-container` styles are fully defined to prevent rendering issues.

**Location**: `<style>`, replace truncated `.webcam-mesh-container` styles.

**Code**:
```css
.webcam-mesh-container {
    width: 100%;
    height: 200px;
    background: #000;
    border: 2px solid #0f0;
    position: relative;
    overflow: hidden;
}
.webcam-grid {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(rgba(0,255,0,0.1) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0,255,0,0.1) 1px, transparent 1px);
    background-size: 20px 20px;
    z-index: 1;
}
.face-direction-arrow {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 40px;
    height: 40px;
    transform-origin: center;
    z-index: 3;
    transition: transform 0.1s ease;
}
.face-direction-arrow::before {
    content: '';
    position: absolute;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 0;
    height: 0;
    border-left: 8px solid transparent;
    border-right: 8px solid transparent;
    border-bottom: 20px solid #f00;
}
.face-direction-arrow::after {
    content: '';
    position: absolute;
    top: 20px;
    left: 50%;
    transform: translateX(-50%);
    width: 4px;
    height: 20px;
    background: #f00;
}
```

---

### Step 4: Update Service Worker for Gimbal Agent
Replace the service worker script to include HC-SR04, face detection, and Android orientation processing for gimbal control.

**Location**: `<script>`, replace `swScript` constant.

**Code**:
```javascript
const swScript = `
    const CACHE_NAME = 'mirror-ide-cache-v3';
    const MAP_DATA = { friendly: [], hostile: [] };
    let faceDirection = 0;
    let faceDetected = false;
    let sensorDistance = 0;
    let deviceOrientation = 0;
    let gimbalAngle = 0;

    self.addEventListener('install', (event) => {
        self.skipWaiting();
        console.log('Service Worker: Installed');
    });

    self.addEventListener('activate', (event) => {
        self.clients.claim();
        console.log('Service Worker: Activated');
    });

    self.addEventListener('message', (event) => {
        if (!event.data || !event.data.type) return;
        try {
            switch (event.data.type) {
                case 'init':
                    MAP_DATA.friendly = event.data.map?.friendly || [];
                    MAP_DATA.hostile = event.data.map?.hostile || [];
                    break;
                case 'face_data':
                    const detections = event.data.detections;
                    faceDetected = detections && detections.length > 0;
                    if (faceDetected) {
                        const bbox = detections[0].boundingBox;
                        const centerX = (bbox.xMin + bbox.xMax) / 2;
                        faceDirection = Math.round((centerX * 360) % 360);
                    } else {
                        faceDirection = 0;
                    }
                    updateGimbal();
                    break;
                case 'sensor_data':
                    sensorDistance = event.data.distance || 0;
                    updateGimbal();
                    break;
                case 'orientation':
                    deviceOrientation = event.data.angle || 0;
                    updateGimbal();
                    break;
                case 'update':
                    const drone = event.data.drone;
                    const user = event.data.user;
                    const objects = event.data.objects || [];
                    if (drone) {
                        const newPos = {
                            x: drone.x + (Math.random() * 0.1 - 0.05),
                            y: drone.y,
                            z: drone.z + (Math.random() * 0.1 - 0.05)
                        };
                        objects.forEach(obj => {
                            const dist = Math.hypot(newPos.x - obj.x, newPos.z - obj.z);
                            if (dist < 3) newPos.y += 1;
                        });
                        self.clients.matchAll().then(clients => {
                            clients.forEach(client => {
                                client.postMessage({
                                    type: 'position',
                                    drone: newPos,
                                    timestamp: Date.now()
                                });
                            });
                        });
                    }
                    break;
                case 'accel':
                    const accel = event.data.data;
                    if (accel && typeof accel.x === 'number') {
                        self.clients.matchAll().then(clients => {
                            clients.forEach(client => {
                                client.postMessage({
                                    type: 'accel_update',
                                    data: accel
                                });
                            });
                        });
                    }
                    break;
            }
        } catch (error) {
            console.error('Service Worker Error:', error);
        }
    });

    function updateGimbal() {
        const weightFace = faceDetected ? 0.5 : 0;
        const weightSensor = sensorDistance > 0 ? 0.3 : 0;
        const weightOrientation = 0.2;
        const totalWeight = weightFace + weightSensor + weightOrientation || 1;
        let angle = 0;
        if (faceDetected) angle += faceDirection * weightFace;
        if (sensorDistance > 0) {
            const sensorAngle = Math.min(Math.max((sensorDistance - 200) * 0.45, -90), 90);
            angle += sensorAngle * weightSensor;
        }
        angle += deviceOrientation * weightOrientation;
        gimbalAngle = angle / totalWeight;
        self.clients.matchAll().then(clients => {
            clients.forEach(client => {
                client.postMessage({
                    type: 'gimbal_update',
                    angle: gimbalAngle,
                    faceDirection,
                    sensorDistance,
                    deviceOrientation,
                    timestamp: Date.now()
                });
            });
        });
    }

    self.addEventListener('fetch', (event) => {
        event.respondWith(
            caches.match(event.request).then(response => response || fetch(event.request))
        );
    });
`;
```

---

### Step 5: Add HC-SR04, Radar, and Android Integration JavaScript
Integrate Web Bluetooth for HC-SR04, a Three.js radar, calibration, and Android orientation handling.

**Location**: `<script>`, after `setupServiceWorker`.

**Code**:
```javascript
// HC-SR04 and Radar
let radarScene, radarCam, radarRen, radarPoints = [];
const SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b";
const CHARACTERISTIC_UUID = "beb5483e-36c1-4688-b7f5-ea07361b26a8";
let myCharacteristic = null;
let gimbalAngle = 0;
let sensorDistance = 0;

function initRadar() {
    radarScene = new THREE.Scene();
    radarCam = new THREE.PerspectiveCamera(75, 4 / 3, 0.1, 100);
    radarCam.position.set(0, 5, 0);
    radarCam.lookAt(0, 0, 0);
    radarRen = new THREE.WebGLRenderer({ canvas: $('#radarCanvas'), alpha: true });
    radarRen.setSize(200, 150);
    const grid = new THREE.GridHelper(10, 10, 0x00ff00, 0x00ff00);
    grid.rotation.x = Math.PI / 2;
    radarScene.add(grid);
    const droneGeo = new THREE.SphereGeometry(0.2, 8, 8);
    const droneMat = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    const droneMarker = new THREE.Mesh(droneGeo, droneMat);
    radarScene.add(droneMarker);
}

function updateRadar(distance) {
    radarPoints.forEach(point => radarScene.remove(point));
    radarPoints = [];
    if (distance > 0) {
        const rad = gimbalAngle * Math.PI / 180;
        const x = (distance / 40) * Math.cos(rad);
        const z = (distance / 40) * Math.sin(rad);
        const pointGeo = new THREE.SphereGeometry(0.1, 8, 8);
        const pointMat = new THREE.MeshBasicMaterial({ color: 0xff0000 });
        const point = new THREE.Mesh(pointGeo, pointMat);
        point.position.set(x, 0, z);
        radarScene.add(point);
        radarPoints.push(point);
    }
    radarRen.render(radarScene, radarCam);
}

function connectBLE() {
    navigator.bluetooth.requestDevice({ filters: [{ services: [SERVICE_UUID] }] })
        .then(device => device.gatt.connect())
        .then(server => server.getPrimaryService(SERVICE_UUID))
        .then(service => service.getCharacteristic(CHARACTERISTIC_UUID))
        .then(characteristic => {
            myCharacteristic = characteristic;
            return characteristic.startNotifications();
        })
        .then(() => {
            $('#sensorData').textContent = 'Connected. Receiving data...';
            myCharacteristic.addEventListener('characteristicvaluechanged', handleSensorData);
            add("BLE SENSOR CONNECTED", "success");
        })
        .catch(error => {
            $('#sensorData').textContent = `Error: ${error.message}`;
            add(`BLE CONNECTION FAILED: ${error.message}`, "error");
        });
}

function handleSensorData(event) {
    const value = new TextDecoder().decode(event.target.value);
    sensorDistance = parseFloat(value.match(/\d+/)[0]) || 0;
    $('#sensorData').textContent = value;
    add(`SENSOR DATA: ${value}`, "info");
    updateRadar(sensorDistance);
    if (navigator.serviceWorker?.controller) {
        navigator.serviceWorker.controller.postMessage({
            type: 'sensor_data',
            distance: sensorDistance
        });
    }
}

function calibrateSensor() {
    if (!testMode) {
        add("CALIBRATION ONLY IN TEST MODE", "error");
        return;
    }
    if (!stream || !userDetected) {
        add("WEBCAM AND USER DETECTION REQUIRED", "error");
        return;
    }
    if (!myCharacteristic) {
        add("BLE SENSOR NOT CONNECTED", "error");
        return;
    }
    add("CALIBRATION STARTED: MOVE DEVICE TO TEST RANGE", "prompt");
    let samples = 0, totalDistance = 0;
    const calibrationInterval = setInterval(() => {
        if (samples >= 10) {
            clearInterval(calibrationInterval);
            const avgDistance = totalDistance / samples;
            add(`CALIBRATION COMPLETE: AVG DISTANCE ${avgDistance.toFixed(1)}cm`, "success");
            return;
        }
        if (sensorDistance > 0) {
            totalDistance += sensorDistance;
            samples++;
            add(`SAMPLE ${samples}: ${sensorDistance}cm`, "info");
        }
    }, 500);
}

// Android Orientation
function setupOrientation() {
    if (window.DeviceOrientationEvent) {
        if (typeof DeviceOrientationEvent.requestPermission === 'function') {
            DeviceOrientationEvent.requestPermission()
                .then(permission => {
                    if (permission === 'granted') {
                        window.addEventListener('deviceorientation', handleOrientation);
                        add("ORIENTATION ACCESS GRANTED", "success");
                    } else {
                        add("ORIENTATION ACCESS DENIED", "warning");
                    }
                })
                .catch(error => add(`ORIENTATION ERROR: ${error.message}`, "error"));
        } else {
            window.addEventListener('deviceorientation', handleOrientation);
        }
    } else {
        add("DEVICE ORIENTATION NOT SUPPORTED", "warning");
    }
}

function handleOrientation(event) {
    const angle = event.alpha || 0;
    if (btMeshActive && navigator.serviceWorker?.controller) {
        navigator.serviceWorker.controller.postMessage({
            type: 'orientation',
            angle
        });
    }
}
```

---

### Step 6: Update Event Listeners
Add handlers for BLE and calibration buttons, and initialize radar/orientation.

**Location**: 
1. `setupPopups`, update `buttonHandlers`.
2. `init`, after `setupServiceWorker()`.

**Code**:
1. **In `setupPopups`**:
   ```javascript
   buttonHandlers: {
       // ... existing handlers ...
       'connectBLE': () => connectBLE(),
       'calibrateSensor': () => calibrateSensor()
   }
   ```

2. **In `init`**:
   ```javascript
   initRadar();
   setupOrientation();
   ```

---

### Step 7: Update Service Worker Listener
Handle gimbal updates for LIVE mode.

**Location**: `setupServiceWorker`, in `navigator.serviceWorker.addEventListener('message', ...)`.

**Code**:
Add to `switch`:
```javascript
case 'gimbal_update':
    gimbalAngle = event.data.angle;
    if (cur === State.LIVE && isGimbal && armed) {
        droneObj.rotation.y = -gimbalAngle * Math.PI / 180;
        cam.rotation.y = -gimbalAngle * Math.PI / 180;
        add(`GIMBAL ADJUSTED: ${gimbalAngle.toFixed(1)}°`, "info");
    }
    updateRadar(event.data.sensorDistance);
    break;
```

---

### Step 8: Update `animate` for Gimbal Sync
Adjust gimbal tracking in LIVE mode.

**Location**: `animate`, modify gimbal block.

**Code**:
```javascript
if (isGimbal && armed) {
    cam.position.copy(drone);
    cam.position.y += 0.2;
    if (cur === State.LIVE) {
        cam.rotation.y = -gimbalAngle * Math.PI / 180;
    } else {
        cam.lookAt(user);
    }
} else if (isFPV) {
    // ... existing code ...
```

---

### Step 9: Update Terminal Commands
Add sensor commands.

**Location**: `processCommand`, update `commands`.

**Code**:
```javascript
connectble: () => connectBLE(),
calibrate: () => calibrateSensor()
```
Update `help`:
```javascript
add("connectble, calibrate - Sensor control", "info");
```

---

### Step 10: Update Manifest for HTTPS
Ensure Web Bluetooth works.

**Location**: `<link rel="manifest">`, update `start_url`.

**Code**:
```html
<link rel="manifest" href="data:application/manifest+json,{
    \"name\": \"MIRROR IDE v3.0\",
    \"short_name\": \"MIRROR IDE\",
    \"description\": \"BT MESH FPV Drone Training Platform\",
    \"start_url\": \"https://your-domain-or-localhost/\",
    \"display\": \"fullscreen\",
    \"background_color\": \"#000000\",
    \"theme_color\": \"#000000\",
    \"icons\": [
        {
            \"src\": \"data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTkyIiBoZWlnaHQ9IjE5MiIgdmlld0JveD0iMCAwIDE5MiAxOTIiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIxOTIiIGhlaWdodD0iMTkyIiBmaWxsPSIjMDAwMDAwIi8+CjxwYXRoIGQ9Ik05NiAxMjhMMTI4IDk2SDY0TDEyOCA2NEw5NiAzMkw2NCA2NEgxMjhMNjQgOTZIMTI4LDk2IDEyOFoiIGZpbGw9IiMwMGZmMDAiLz4KPC9zdmc+Cg==\",
            \"sizes\": \"192x192\",
            \"type\": \"image/svg+xml\"
        }
    ]
}">
```

---

### Android Integration Instructions

Per the `mirror_readme.md`, Android devices can send telemetry data (gyroscope, accelerometer, GPS) via Bluetooth to control the drone or gimbal. Below are instructions to integrate this with the modified HTML.

#### Hardware Setup
- **ESP32 with HC-SR04**:
  - Wire as specified: VCC to 5V, GND to GND, TRIG to GPIO 19, ECHO to GPIO 18 via voltage divider (2kΩ to ECHO, 1kΩ to GND).
  - Flash the ESP32 with the Arduino firmware from the original context:
    ```cpp
    #include <BLEDevice.h>
    #include <BLEServer.h>
    #include <BLEUtils.h>
    #include <BLE2902.h>
    #define TRIG_PIN 19
    #define ECHO_PIN 18
    #define SERVICE_UUID "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
    #define CHARACTERISTIC_UUID "beb5483e-36c1-4688-b7f5-ea07361b26a8"
    BLECharacteristic *pCharacteristic;
    bool deviceConnected = false;
    class MyServerCallbacks: public BLEServerCallbacks {
        void onConnect(BLEServer* pServer) { deviceConnected = true; };
        void onDisconnect(BLEServer* pServer) { deviceConnected = false; }
    };
    void setup() {
        Serial.begin(115200);
        pinMode(TRIG_PIN, OUTPUT);
        pinMode(ECHO_PIN, INPUT);
        BLEDevice::init("ESP32_DroneSensor");
        BLEServer *pServer = BLEDevice::createServer();
        pServer->setCallbacks(new MyServerCallbacks());
        BLEService *pService = pServer->createService(SERVICE_UUID);
        pCharacteristic = pService->createCharacteristic(
            CHARACTERISTIC_UUID,
            BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
        );
        pCharacteristic->addDescriptor(new BLE2902());
        pService->start();
        BLEAdvertising *pAdvertising = BLEDevice::pAdvertising;
        pAdvertising->addServiceUUID(SERVICE_UUID);
        pAdvertising->setScanResponse(true);
        pAdvertising->setMinPreferred(0x06);
        pAdvertising->start();
    }
    void loop() {
        digitalWrite(TRIG_PIN, LOW);
        delayMicroseconds(2);
        digitalWrite(TRIG_PIN, HIGH);
        delayMicroseconds(10);
        digitalWrite(TRIG_PIN, LOW);
        long duration = pulseIn(ECHO_PIN, HIGH);
        long distance = duration * 0.034 / 2;
        if (deviceConnected) {
            char txValueString[16];
            sprintf(txValueString, "D:%ldcm", distance);
            pCharacteristic->setValue(txValueString);
            pCharacteristic->notify();
            delay(100);
        }
        delay(100);
    }
    ```

#### Android App Setup
1. **Develop App**:
   - Use **MIT App Inventor** or **Android Studio** to create an app that:
     - Accesses sensors (gyroscope, accelerometer, GPS).
     - Formats data as a string (e.g., `roll:1.2,pitch:5.4,yaw:0.0`).
     - Sends data via Bluetooth to the ESP32.
   - Example MIT App Inventor blocks:
     - Use `OrientationSensor` for roll/pitch/yaw.
     - Use `BluetoothClient` to connect to ESP32’s MAC address.
     - Send data every 100ms.

2. **BLE Service**:
   - Create a second BLE characteristic for Android telemetry:
     ```cpp
     #define TELEMETRY_UUID "beb5483e-36c1-4688-b7f5-ea07361b26a9"
     BLECharacteristic *pTelemetryCharacteristic;
     // In setup():
     pTelemetryCharacteristic = pService->createCharacteristic(
         TELEMETRY_UUID,
         BLECharacteristic::PROPERTY_WRITE
     );
     pTelemetryCharacteristic->setCallbacks(new MyCharacteristicCallbacks());
     // Add class:
     class MyCharacteristicCallbacks: public BLECharacteristicCallbacks {
         void onWrite(BLECharacteristic *pCharacteristic) {
             std::string value = pCharacteristic->getValue();
             Serial.println(value.c_str()); // Process telemetry
         }
     };
     ```

3. **ESP32 Processing**:
   - Parse telemetry (e.g., `roll:1.2,pitch:5.4`) and forward to the HTML app via the existing BLE characteristic or a new one.
   - Example:
     ```cpp
     void onWrite(BLECharacteristic *pCharacteristic) {
         std::string value = pCharacteristic->getValue();
         pCharacteristic->setValue(value); // Forward to HTML
         pCharacteristic->notify();
     }
     ```

#### HTML Integration
1. **Add Telemetry Handling**:
   - Modify `handleSensorData` to parse Android telemetry if received:
     ```javascript
     function handleSensorData(event) {
         const value = new TextDecoder().decode(event.target.value);
         if (value.startsWith("D:")) {
             sensorDistance = parseFloat(value.match(/\d+/)[0]) || 0;
             $('#sensorData').textContent = value;
             add(`SENSOR DATA: ${value}`, "info");
             updateRadar(sensorDistance);
             navigator.serviceWorker?.controller?.postMessage({
                 type: 'sensor_data',
                 distance: sensorDistance
             });
         } else if (value.includes("roll:")) {
             const [roll, pitch, yaw] = value.match(/[-]?\d+\.\d+/g).map(parseFloat);
             navigator.serviceWorker?.controller?.postMessage({
                 type: 'orientation',
                 angle: yaw
             });
             add(`TELEMETRY: roll=${roll}, pitch=${pitch}, yaw=${yaw}`, "info");
         }
     }
     ```

2. **Test Integration**:
   - Open the HTML app on an Android browser (Chrome).
   - Connect to ESP32 via SDK popup (“CONNECT BLE”).
   - Start the Android app to send telemetry.
   - Verify telemetry appears in the terminal (e.g., `TELEMETRY: roll=1.2, pitch=5.4, yaw=0.0`).
   - In LIVE mode, ensure gimbal tracks based on yaw from telemetry, face detection, and HC-SR04 distance.

#### Testing
1. **Hardware**:
   - Verify HC-SR04 wiring and ESP32 firmware.
   - Ensure Android device pairs with ESP32 via BLE.
2. **Web App**:
   - Serve over HTTPS (`ngrok http 8000` or `npx http-server -S`).
   - Load in Chrome on Android.
   - TEST Mode: Calibrate HC-SR04, verify radar.
   - LIVE Mode: Arm drone, enable gimbal, confirm tracking via telemetry, webcam, and sensor.
3. **Fallback**:
   - If BLE fails, use Web Serial (requires ESP32 firmware changes).

---

### Notes
- **Radar**: Shows HC-SR04 distance (red point) relative to drone (green center) on a 10-unit grid (400cm max).
- **Gimbal**: Uses 50% face direction, 30% sensor distance, 20% Android orientation.
- **Performance**: Radar updates are optimized for 60fps.
- **UUIDs**: Use unique UUIDs in production (`uuidgen`).
- **Range**: BLE range is 10-30m; test in open space


## License
© 2025 webXOS. Licensed under [MIT License](LICENSE).

