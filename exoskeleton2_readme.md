# EXOSKELETON 2 – Neural Audio-Visual Encryption System  

**Technical & Conceptual User Guide**  

December 22, 2025

## Introduction – Aesthetic Cryptography in the Late 2020s

EXOSKELETON 2 is a single-page, browser-based PWA/UX project that deliberately blends three eras of computing:

- the glowing green phosphor of 1980s–1990s terminal interfaces 
- the handshakes of 56k dial-up audio
- contemporary neural-network visualization

Rather than pursuing standard cryptographic security, the project performs a kind of **visual and audio cryptography** — it takes ordinary text, subjects it to multiple layers of deterministic transformation, and presents the result as both a visualized neural hologram and an audio artifact.

## Core

Every piece of text you enter is simultaneously:

1. **Encrypted** — using a reproducible but very weak XOR-based scheme  
2. **Sonified** — turned into frequency-modulated audio that imitates a 56k modem negotiation mixed with data transmission  
3. **Spatialized** — visualized as a three-dimensional-ish constellation of floating squares, pulsing nodes, and connecting synapses  

The app triangulates the encryption with three dramatically different sensory channels at once:

- linguistic (the original plaintext)  
- auditory (modem-like screeching tones)  
- visual-kinesthetic (cybernetic neural architecture)

## Use:

When you type or paste text and press ENCRYPT, the following sequence occurs behind the terminal interface:

1. **Text → Hash Seed Generation**  
   A long SHA-512 digest is computed from the text concatenated with a fixed application salt. The first portion of this hash becomes a deterministic seed that controls almost every visual parameter.

2. **Procedural Neural Architecture**  
   Using golden-angle distribution, character codes, and hash-derived pseudo-random values, the system places:
   - 20–80 floating grid squares  
   - three concentric layers of neural nodes (9 → 6 → 3)  
   - probabilistic synaptic connections between layers

   All positions, rotations, sizes, opacities, and hues are derived from the hash — the same text always reproduces the same visualization.

3. **Reproducible Encryption**  
   The plaintext bytes are XORed against a repeating key stream generated from SHA-256(text + fixed salt).  

4. **Audio Synthesis**  
   The XORed byte stream is turned into audio using frequency-shift keying reminiscent of early modem protocols:
   - 1200–1300 Hz range ≈ binary 1 (mark)  
   - 2400–2500 Hz range ≈ binary 0 (space)  
   - Answer tone (2100 Hz) and carrier simulation  
   - Deliberate light noise injection  
   - Start/stop bits and short inter-byte silences

5. **Persistent Storage**  
   The complete package (original text, hash, neural parameters, full float32 audio samples) is stored in IndexedDB.

6. **Export/Import Ritual**  
   Everything is bundled into a ZIP archive with the extension .zip containing:
   - manifest.json  
   - plaintext (for convenience)  
   - neural architecture JSON  
   - base64-encoded raw PCM audio  
   - signature (for theatrical integrity checking)


## Technical Stack – 2025 Browser-native Edition

The entire application runs as a **single HTML file** containing:

- **CryptoJS** → SHA-256/512, general-purpose cryptography primitives  
- **pako** → DEFLATE compression for ZIP export  
- **JSZip** → ZIP archive creation & reading in browser  
- **Web Audio API** → real-time frequency-modulated audio synthesis  
- **IndexedDB** → long-term storage of encrypted entries  
- **Pure CSS 3D transforms** → faux-holographic grid & node positioning  
- **DOM animation** → pulsing, rotating, breathing effects

Notably absent:

- any backend server  
- any npm dependencies outside CDNs  
- any build step  
- any framework

The project embraces the 2020s “single-file maximalism” trend taken to an extreme.

## Limitations & Disclosure

EXOSKELETON 2 should **never** be used to protect information that actually matters. It's conceptual and designed to test encryption:

The “encryption” is reproducible from the plaintext alone.  
The audio can (in theory) be demodulated with enough signal-processing knowledge.  
The ZIP contains the plaintext in clear text for user convenience.

-it performs all the secrecy while containing almost none of the mathematical protections.

## Final Thoughts:

In an age where real end-to-end encryption has become boringly ubiquitous and invisible, EXOSKELETON 2 chooses the opposite direction:

It reminds us that for most of human history, sending a private message was a dedicated process — involving wax seals, trusted couriers, code books, and hours (or days) of waiting.

EXOSKELETON 2 tries to resurrect that craft using the materials available in 2025: a browser tab, some green glow css, and the encryption sounds of 56k modems that once connected the early internet.

Enjoy the app.
webxos.netlify.app
