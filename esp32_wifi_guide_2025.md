# ESP32 WIFI Technical Breakdown

*by: webXOS 2025* 
*webxos.netlify.app*
*x.com/webxos*

To implement a Wi-Fi sensing system using an ESP32 for data collection and a laptop for processing, you will need to set up both devices to communicate over a local network. This guide uses the Arduino IDE for the ESP32 and Python for the laptop's processing software.
Prerequisites

    -An ESP32 development board.
    -A laptop with Wi-Fi capability.
    -The latest Arduino IDE installed.
    -Python 3 installed on your laptop.
    -Basic knowledge of Arduino programming and Python.

#Basic knowledge of Arduino programming and Python. a guide for esp32 bluetooth and wifi networking:

ESP32 Wi-Fi and Bluetooth networking using the Arduino IDE, you will configure the ESP32 to run in dual mode (Wi-Fi Station and Bluetooth Classic server), allowing for network connectivity for data streaming and a Bluetooth serial connection for control or debugging. The ESP32's single radio manages both, so actual simultaneous transmission is time-multiplexed by the hardware, which generally works seamlessly for many applications.
Prerequisites

#Arduino IDE installed on your laptop. You can download it from the Arduino website.

ESP32 board support installed in the Arduino IDE. Add https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json to the "Additional Boards Manager URLs" in File > Preferences, then search for and install "esp32 by Espressif Systems" in the Tools > Board > Boards Manager.

    A laptop connected to the same Wi-Fi network as the ESP32 will be.

    Basic knowledge of Arduino programming and Python scripting.

#Step 1: Configure the ESP32 with Arduino IDE (Wi-Fi and Bluetooth)

Upload the following sketch to your ESP32 board. This code initializes both Wi-Fi in station mode and Bluetooth Classic as a serial device.

-Open the Arduino IDE.
-Go to Tools > Board and select your specific ESP32 board (e.g., ESP32 Dev Module).
-Connect your ESP32 to your laptop via a USB cable. Go to Tools > Port and select the appropriate COM port.
-Copy and paste the code below into a new sketch:

arduino
```
#include "BluetoothSerial.h"
#include <WiFi.h>

// Replace with your network credentials
const char* ssid = "YOUR_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

BluetoothSerial SerialBT;

void setup() {
  Serial.begin(115200);
  Serial.println("\nStarting ESP32 Dual Mode (WiFi & Bluetooth)");

  // Start Bluetooth serial with a device name
  SerialBT.begin("ESP32_CSI_Sensor"); // Bluetooth device name
  Serial.println("Bluetooth started. You can pair with the device named ESP32_CSI_Sensor");

  // Connect to Wi-Fi
  WiFi.mode(WIFI_STA); // Set the ESP32 to Station mode
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi ..");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print('.');
    delay(1000);
  }
  Serial.println("\nConnected to WiFi network!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP()); // Print the assigned IP address
}

void loop() {
  // Handle Bluetooth communication
  if (SerialBT.available()) {
    // Read from Bluetooth and potentially process commands or data
    Serial.write(SerialBT.read());
  }

  // Handle WiFi communication (your CSI data streaming logic would go here)
  // This is where you would implement the code to collect CSI data 
  // and send it to your laptop's IP address over the network.
}
```

Use code with caution.

*Update YOUR_SSID and YOUR_WIFI_PASSWORD with your actual network credentials.*

Upload the code to your ESP32. If the board gets stuck at the "Connecting..." screen, press the on-board EN (reset) or BOOT button for a second after it starts connecting.

Open the Serial Monitor at a baud rate of 115200 to monitor the connection status and the assigned IP address.

#Step 2: Set Up Your Laptop for Data Processing

Your laptop needs software to receive the data from the ESP32. For Wi-Fi-based data streaming, a simple Python script can act as a server to listen for incoming connections or UDP packets.

Install Python: If not already installed, download it from the official Python website.

Create a Python script to receive data (e.g., data_receiver.py). This simple script listens for UDP packets.

python
```
import socket

# Use the IP address printed in the Arduino Serial Monitor

UDP_IP = "YOUR_ESP32_IP_ADDRESS" 
UDP_PORT = 12345 # Must match the port used in your Arduino code for sending data

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for UDP data on {UDP_IP}:{UDP_PORT}...")

while True:
    data, addr = sock.recvfrom(1024) # Buffer size is 1024 bytes
    print(f"Received message from {addr}: {data.decode()}")
```

Use code with caution.

*Update YOUR_ESP32_IP_ADDRESS with the IP address shown in the Arduino Serial Monitor.*

#Step 3: Test Networking

Wi-Fi: You would add code to the Arduino sketch's loop() function to send data to the laptop's IP address (not shown here, as it requires specific CSI libraries). Run the Python script on your laptop, and it should display the received data.

Bluetooth: On your laptop, you can use a Bluetooth serial terminal application (or a Python script using a library like pyserial) to connect to the "ESP32_CSI_Sensor" device. Any data sent from the laptop via the Bluetooth terminal will appear in the Arduino Serial Monitor.

# This guide provides the dual-mode framework; you will need to incorporate the specific code from the CSI or ESPectre project to collect and stream the Channel State Information data over the established Wi-Fi network.

ESP32 Connection (Station Mode): The ESP32 can be configured in "Station mode" (STA) to connect to an existing Wi-Fi network, such as the one provided by your home router or a mobile hotspot created by your laptop. This allows the ESP32 to send data to, and receive requests from, other devices on the same local network, including your laptop.

Web Page Interface: You can host an HTML web page directly on the ESP32 itself. Your laptop's web browser can then access this page by navigating to the ESP32's assigned IP address on the local network. This page can display the data, provide controls, or serve as a user interface for the presence detection system.

# CSI Data

# The ESP32 is capable of capturing CSI data.

-The project usually involves the ESP32 gathering this data and sending it to a processing unit (your laptop, or typically a Home Assistant instance or a dedicated server) over the network.

-The laptop needs the appropriate software (likely Python scripts provided by the ESPectre project) to receive, process, and interpret the raw CSI data stream from the ESP32 to determine presence.

-Network Setup: The simplest setup involves both the ESP32 and the laptop connecting to the same standard Wi-Fi router. The ESP32 sends the CSI data packets to the laptop's IP address on this network. 

#Summary

The single ESP32 and laptop setup is feasible. The ESP32 acts as a sensor and a web server client/server, while the laptop acts as the data processing and visualization center. The standard Wi-Fi network is the communication backbone. You will need to program the ESP32 to connect to your Wi-Fi network and stream the CSI data, and set up your laptop to receive and process this data, potentially through a web interface or a dedicated application. 

Creating a web interface to control a Wi-Fi drone involves an HTML front-end for the user interface and JavaScript to handle connectivity and send commands, typically using WebSockets for real-time communication. The drone or an attached companion computer (like an ESP32 or Raspberry Pi) needs to run a server to receive these commands.

#Connection and Architecture Overview

-Connect to the Drone's Network: First, your controlling device (computer, phone, tablet) must connect to the Wi-Fi network hosted by the drone or its controller board (e.g., an ESP32 as an Access Point). The network name (SSID) is often provided in the drone's documentation (e.g., TELLO-xxxx or quadcopter).

-Web Server: The drone's companion computer runs a web server that hosts the HTML page and opens a WebSocket connection for real-time data exchange.

-Communication Protocol: WebSockets provide a persistent, bidirectional communication channel between your browser and the drone, which is essential for low-latency control.

#Guide to the HTML Page and Code

Below is a basic guide and an HTML structure using JavaScript for connecting and sending simple commands.

#1. Basic HTML Structure

This single HTML file will contain the user interface and all the necessary JavaScript code.
html

```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wi-Fi Drone Controller</title>
    <style>
        body { font-family: sans-serif; text-align: center; }
        .controls button { margin: 10px; padding: 10px 20px; font-size: 16px; }
        #status { margin-top: 20px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Drone Control Interface</h1>
    <div id="status">Status: Disconnected</div>

    <button onclick="connectWebSocket()">Connect to Drone</button>

    <div class="controls">
        <h2>Flight Controls</h2>
        <button onclick="sendCommand('takeoff')">Take Off</button>
        <button onclick="sendCommand('land')">Land</button>
        <button onclick="sendCommand('up 20')">Up 20cm</button>
        <button onclick="sendCommand('down 20')">Down 20cm</button>
        <button onclick="sendCommand('forward 50')">Forward 50cm</button>
    </div>

    <script src="drone_control.js"></script> <!-- Link to the JavaScript file -->
</body>
</html>
```

Use code with caution.

#2. JavaScript (drone_control.js)

You'll need a separate JavaScript file (drone_control.js) to handle the WebSocket connection and command sending logic.
javascript

```
let socket;
const statusDiv = document.getElementById('status');
// Change this to the IP address of your drone or ESP32 board
// Common default IPs include 192.168.4.1 or 192.168.2.1
const DRONE_IP = 'ws://192.168.4.1/ws'; // Use ws:// for WebSocket protocol

function connectWebSocket() {
    statusDiv.textContent = 'Status: Connecting...';
    socket = new WebSocket(DRONE_IP);

    socket.onopen = function(event) {
        statusDiv.textContent = 'Status: Connected';
        console.log('WebSocket connection opened:', event);
    };

    socket.onmessage = function(event) {
        console.log('Message from drone:', event.data);
        // Handle telemetry data or confirmation messages here
    };

    socket.onclose = function(event) {
        statusDiv.textContent = 'Status: Disconnected';
        console.log('WebSocket connection closed:', event);
    };

    socket.onerror = function(error) {
        statusDiv.textContent = 'Status: Error';
        console.error('WebSocket error:', error);
    };
}

function sendCommand(command) {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(command);
        console.log('Sent command:', command);
    } else {
        alert('Not connected to the drone. Click "Connect to Drone" first.');
    }
}
```

Use code with caution.

#Streaming

Channel State Information (CSI) data from an ESP32 to a laptop over Wi-Fi requires a specialized ESP32 firmware/toolkit, such as the ESP32-CSI-Tool or the ESP32-CSI-Collection-and-Display tool. These projects provide the necessary code for both the ESP32 (typically using ESP-IDF, but wrappers exist for Arduino) and the laptop (Python scripts for processing). The data is commonly sent via UDP packets due to the high volume of real-time information. 

Below is an outline of the process, using the principles from those tools.

#Key Components

    ESP32 Firmware: Configured to capture CSI data and stream it over Wi-Fi using the UDP protocol.
    Laptop Receiver: A Python script running on your laptop to listen for UDP packets, parse the raw data, and perform analysis.
    Network: Both the ESP32 and laptop must be on the same Wi-Fi network. 

#Step 1: Set up the ESP32 Firmware (Conceptual Example)
 
You'll need to use specific libraries to access the raw CSI from the ESP32's Wi-Fi chip. A full, ready-to-flash Arduino sketch is complex and depends heavily on these external libraries, but the logic would look like this: 
cpp

```
#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_wifi_types.h>
#include <esp_csi_internal.h> // Header for internal CSI functions

// ... (your Wi-Fi credentials and UDP setup code) ...
WiFiUDP Udp;
IPAddress laptopIP(192, 168, 1, 10); // !! REPLACE with your laptop's actual IP !!
const unsigned int localPort = 12345;
const unsigned int remotePort = 12345;

void setup() {
  // ... (Serial and Wi-Fi connection setup from the previous guide) ...

  // Configure CSI here (requires specific library functions)
  // esp_wifi_set_csi(true);
  // esp_wifi_set_csi_cb(on_csi_receive, NULL); // A callback function to handle received CSI

  Serial.println("CSI collection enabled and streaming to laptop...");
}

// A placeholder for the actual CSI callback function
void on_csi_receive(void *ctx, wifi_csi_info_t *data) {
  if (data != NULL) {
    // Process the raw 'data->buf' (amplitude, phase, etc.)
    // Package it into a string or byte array
    // Udp.beginPacket(laptopIP, remotePort);
    // Udp.write(packagedData);
    // Udp.endPacket();
  }
}

void loop() {
  // Main loop remains empty as data is handled in the callback
}
```

Use code with caution.

Note: The actual implementation for collecting CSI is non-trivial within the standard Arduino framework and generally requires using specific open-source tools' libraries or the ESP-IDF framework. It involves low-level Wi-Fi stack access. 

#Step 2: Set Up the Laptop Receiver

This Python script will listen for the UDP packets sent by the ESP32 and print the raw data to the console. 
python

```
import socket

UDP_IP = "0.0.0.0"  # Listen on all available interfaces
UDP_PORT = 12345    # Must match the port used in your Arduino code

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for raw CSI data on UDP port {UDP_PORT}...")

while True:
    # Receive data packet
    data, addr = sock.recvfrom(2048) # Buffer size large enough for a CSI packet
    # The 'data' variable now holds the raw byte stream from the ESP32
    print(f"Received packet from {addr}, size: {len(data)} bytes")
    # You would use a data parsing library (often specific to the toolkit used)
    # to convert the raw bytes into meaningful CSI amplitudes/phases.
```

Use code with caution.

#Step 3: Run the System

    Flash your ESP32 with the correct CSI-enabled firmware.
    Connect your laptop and ESP32 to the same Wi-Fi network.
    Ensure you know the IP addresses of both devices and configure them in their respective code/scripts.
    Run the Python script on your laptop first.
    Power up the ESP32. 

You should start seeing received data packets printed in your laptop's Python console in real time. The crucial next step (data parsing and processing) depends entirely on the specific format of the raw data stream defined by the ESP32 firmware you choose to use. 



