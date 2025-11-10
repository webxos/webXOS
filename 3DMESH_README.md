# 3DMESH by webXOS 2025

A retro-styled wireframe 3D modeling IDE built with Three.js, designed to run efficiently on embedded systems like Raspberry Pi and Arduino-based devices.

![3DMESH Screenshot](https://webxos.netlify.app/3dmesh-preview.png)

## Features

- **Wireframe 3D Modeling**: Create and manipulate 3D objects in pure wireframe style
- **Virtual 3D Ruler**: Measurement tool that stays flush with the XY grid
- **Template Library**: Pre-built models for games (RPG items, FPS weapons, buildings)
- **Code Generation**: Export your scenes as ready-to-use Three.js code
- **Retro Interface**: CRT-style display with matrix grid background
- **Cross-Platform**: Runs in any modern browser, optimized for low-power devices

## Quick Start

1. Open `3dmesh.html` in any modern web browser
2. Use the shape buttons to add 3D primitives
3. Manipulate objects with Select/Move/Rotate/Scale tools
4. Export your scene as Three.js code or JSON

## Controls

- **Basic Shapes**: Cube, Sphere, Cylinder, Cone, Torus, Plane
- **Manipulation Tools**: Select, Move, Rotate, Scale
- **View Controls**: Orbit/FPV camera, Wireframe toggle
- **Templates**: RPG items, FPS weapons, Buildings, Vehicles

## Template Categories

- **Basic Shapes**: Standard 3D primitives
- **Buildings**: Houses, towers, castles, bridges
- **RPG Items**: Swords, shields, treasure chests, potions
- **FPS Templates**: Weapons, HUD elements, ammo crates
- **Vehicles**: Simple car, tank, spaceship models
- **Characters**: Basic humanoid and creature models

## Export Options

- **Three.js Code**: Generates ready-to-use JavaScript code
- **JSON Format**: Save and load your scenes
- **Copy to Clipboard**: Quick code copying

## System Requirements

- Any modern web browser with WebGL support
- Recommended: 512MB RAM, WebGL-capable GPU
- Optimized for Raspberry Pi 3/4, Arduino-based systems

## Deployment

Deploy to webxos.netlify.app or any static hosting service:

```bash
# Clone repository
git clone https://github.com/webxos/3dmesh.git

# Deploy to Netlify
netlify deploy --dir=. --prod