# AudioDrive 56K Modem Transcoder

AudioDrive is a web-based application that transforms text into authentic 56K modem audio WAV files, mimicking the sounds of dial-up modems from the 1990s. It provides a retro terminal interface for encoding, importing, playing, managing, and exporting audio files. Built with HTML, CSS, and JavaScript, it runs entirely in the browser using Web Audio API and IndexedDB for storage. No backend server is required.

## Features

- **Text-to-Modem Audio Encoding**: Convert any text input into a 56K modem-style WAV file using frequency-shift keying (FSK) modulation (1200Hz for '1', 2400Hz for '0').
- **WAV File Import**: Upload external .wav files for integration into the app's library.
- **Audio Playback**: Play encoded or imported audio with controls for play, pause, stop, and volume adjustment in a popup player.
- **File Management**: List, delete, and export individual or all files as combined WAVs.
- **Queue Processing**: Handles multiple encoding tasks in a queue with progress indicators.
- **Retro Interface**: Terminal-style UI with typewriter effects, ASCII art, and green-on-black aesthetics for an immersive experience.
- **Local Storage**: Uses IndexedDB to persist audio files across sessions.
- **Commands**: Supports terminal commands like `/help`, `/clear`, `/import`, `/export`, `/list`, and direct text input for encoding.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/audiodrive.git
   ```
2. Open `audiodrive.html` in a modern web browser (Chrome, Firefox, or Edge recommended for full Web Audio API support).
3. No dependencies or build steps required—it's a single HTML file.

## Usage

1. **Launch the App**: Open the HTML file. After a loading screen with ASCII art, you'll see the terminal interface.
2. **Encoding Text**: Type any text (e.g., "Hello World") and press Enter or click "SEND". The app encodes it into a 56K modem WAV, adds it to the library, and displays it with controls.
3. **Importing WAV**: Click "IMPORT" to upload a .wav file. It will be added to the library.
4. **Playing Audio**: Click "PLAY" on any file entry to open the popup player. Use play/pause/stop and volume slider.
5. **Exporting**: Click "EXPORT" on a file for single export, or "EXPORT ALL" for a combined WAV of all files.
6. **Commands**:
   - `/help`: Displays the help menu.
   - `/clear`: Clears the terminal and deletes all stored files (with confirmation).
   - `/list`: Shows all audio files with controls.
   - `/import`: Triggers file upload.
   - `/export`: Exports all files as one WAV.
7. **Navigation**: Use ESC to close popups. The terminal auto-scrolls but respects manual scrolling.

Audio files are stored locally in IndexedDB. Clear browser data to reset.

## Functionality Details

AudioDrive simulates 56K modem transmission:
- **Encoding Process**: Text is converted to bytes, then modulated with start/stop bits, handshake tones (2100Hz answer, 1800Hz carrier), and end sequences. Uses sine waves for realistic dial-up sounds.
- **Playback**: Leverages Web Audio API for buffer sources, gain nodes, and real-time controls.
- **Export**: Converts AudioBuffers to WAV format (PCM 16-bit) for download, with gaps between combined files.
- **Limitations**: Browser-based, so large files may impact performance. No decoding back to text—focus is on encoding and playback.

---

AudioDrive is a nostalgic tool for generating and managing 56K modem audio, blending retro computing aesthetics with modern web technologies. Originally inspired by the era of dial-up internet, it serves as both an educational demo of audio modulation and a practical utility for niche applications. Below is a comprehensive overview, expanding on its core mechanics, implementation, historical context, and forward-looking use cases.

### Core Mechanics and Implementation

#### Encoding and Modulation
At its heart, AudioDrive encodes text using frequency-shift keying (FSK), a technique common in V.22bis modems (the standard for 2400 baud rates, scalable to 56K equivalents). Here's how it works:
- **Input Handling**: Any text entered is UTF-8 encoded into bytes.
- **Modulation Scheme**:
  - Mark (binary 1): 1200 Hz sine wave.
  - Space (binary 0): 2400 Hz sine wave.
  - Baud Rate: 2400 bits/second, with each bit lasting ~1/2400 seconds (adjusted for sample rate, default 44.1 kHz).
  - Frame Structure: Start bit (0), 8 data bits, stop bit (1), plus short pauses.
- **Handshake and Tones**: Prepends a 1-second answer tone (2100 Hz) and 0.5-second carrier (1800 Hz) for authenticity. Ends with a fading low-frequency sequence (400-600 Hz).
- **Output**: Generates an AudioBuffer via Web Audio API, stored as a float array in IndexedDB for persistence.
- **Queue System**: Multiple inputs are queued to prevent overload, with visual feedback (e.g., "Queue: X items remaining").

This results in WAV files that sound like screeching modem negotiations, accurately emulating data transmission over phone lines.

#### Audio Management and Storage
- **IndexedDB Integration**: Files are saved with metadata (ID, timestamp, duration, audio data). On load, files are reconstructed into playable buffers.
- **Import/Export**:
  - Import decodes uploaded WAVs into AudioBuffers.
  - Export combines buffers with 0.5-second silence gaps, outputting as downloadable WAVs.
- **Player Popup**: A modal with play/pause/stop, volume (0-100%), and time display (MM:SS). Uses GainNode for volume and BufferSource for playback.

#### UI and UX Elements
- **Terminal Style**: Green text on black, with typewriter animation for outputs (5ms per character). Scrollable with custom scrollbar and auto-scroll logic (disables if user scrolls up >50px).
- **Progress Bar**: Horizontal top bar for encoding/import/export (updates in 25-100% increments with delays for visual effect).
- **Error Handling**: Red text for issues (e.g., invalid files, empty input).
- **Accessibility**: Keyboard-friendly (Enter to send, ESC to close), but primarily visual—future enhancements could add ARIA labels.

The app is self-contained in one HTML file (~15KB minified), making it portable.

### Historical Context and Legacy Integrations

AudioDrive pays homage to the dial-up era (1990s-2000s), when modems converted digital data to analog tones for transmission over POTS (Plain Old Telephone Service) lines. 56K modems (V.90 standard) achieved ~56 kbps downstream by leveraging digital PSTN backends.

#### Use Cases for Legacy Integrations
- **Educational Demos**: Teach signal processing, modulation, and telecommunications history. Integrate into CS curricula to simulate how data was sent before broadband—e.g., pair with tools like Minitel emulators or BBS software.
- **Retro Computing Projects**: 
  - Connect to vintage hardware via audio jacks. For example, play encoded WAVs through a sound card to "transmit" data to a real modem or acoustic coupler.
  - BBS Revival: Encode messages for playback over phone lines in hobbyist setups, bridging web apps to legacy telnet/BBS systems like WWIV or Renegade.
- **Data Archiving**: Store text as audio for analog media (e.g., cassette tapes). Useful for cold storage in scenarios mimicking pre-digital eras, like museum exhibits on internet history.
- **Art and Sound Design**: Generate modem noises for chiptune music or soundscapes. Integrate with DAWs (e.g., export to Audacity) for layering in retro games or films (e.g., recreating scenes from *WarGames* or *The Matrix*).
- **Compatibility Testing**: Test audio interfaces with legacy devices, such as old fax machines or telemetry systems that use FSK. Niche developers can adapt the modulation code for custom baud rates.

These integrations leverage AudioDrive's output as a bridge between digital text and analog audio, preserving functionality in environments without modern networking.

### 2025 Use Cases for Niche Developers

In 2025, with IoT, edge computing, and retro-futurism trends, AudioDrive finds renewed relevance among niche developers in audio engineering, security, and creative tech.

#### Modern Niche Applications
- **Acoustic Data Transmission**: 
  - **Air-Gapped Systems**: Encode sensitive data (e.g., keys or configs) as audio for transfer between isolated devices via speakers/microphones. Useful in secure environments like air-gapped networks—developers can extend the code for error correction (e.g., adding Reed-Solomon codes) to improve reliability over noisy channels.
  - **IoT Prototyping**: Transmit commands via ultrasound/infrasound variations (modify frequencies) for device control without Wi-Fi/Bluetooth. E.g., niche home automation where audio acts as a low-bandwidth fallback.
- **Steganography and Security Tools**:
  - Hide text in audio streams for covert communication. Developers can integrate with libraries like WebCrypto for encrypted payloads, then modulate them—ideal for privacy-focused apps or CTF challenges.
  - Forensic Analysis: Simulate modem traffic for network security testing, or decode similar audio in investigations (though AudioDrive is encode-only; add a decoder module for bidirectional use).
- **Creative and Artistic Development**:
  - **Generative Art**: Use as a module in p5.js or Tone.js projects to sonify data (e.g., encode real-time sensor readings into modem sounds for installations).
  - **Game Development**: Add authentic dial-up SFX to retro-style games (e.g., in Unity WebGL exports). Niche devs can fork for procedural audio generation, like modulating based on game events.
- **Accessibility and Assistive Tech**:
  - Convert text to audio for low-vision users in constrained environments (e.g., encode Braille alternatives). Extend for haptic feedback via vibration APIs synced to tones.
- **Edge Cases in 2025 Tech Landscape**:
  - **Satellite/Remote Comms**: In areas with poor internet (e.g., rural or space tech), encode data for transmission over radio/VHF—adapt frequencies for ham radio integrations.
  - **Blockchain and Web3**: Encode transaction hashes or NFTs metadata as audio NFTs. Niche Web3 devs can build dApps where "minting" creates playable modem art.
  - **AI Integration**: Pipe outputs to AI audio models (e.g., via WebSockets to external services) for hybrid sonification—e.g., encode AI-generated text for experimental music.

#### Development Extensions
- **Customization**: Modify `generateModemAudio` function for different baud rates or tones (e.g., 300 baud for slower, more audible results).
- **API Exposure**: Wrap as a Web Component or NPM module for embedding in larger apps.
- **Performance Tips**: For large texts, chunk processing to avoid browser freezes. Test on mobile for Web Audio compatibility.
- **Limitations and Improvements**: No real-time decoding; add via FFT analysis. Enhance with Web Workers for background encoding.

AudioDrive exemplifies how legacy tech can inspire modern innovation, offering a playground for developers exploring audio as data.

### Contributing
Fork the repo, make changes, and submit PRs. Focus on bug fixes, new modulations, or UI enhancements. Licensed under MIT—free for any use.

### Credits
Built by webxos (2025). Inspired by modem history and Web Audio experiments.