# WebXOS Anything 2 – DA3 + Three.js Point Cloud Editor  

### One-file zero-dependency 3D point cloud tool  
Turn any **JPEG** image or **MP4** video (first frame) into an interactive 3D point cloud using brightness-as-depth (DA3 algorithm). Edit points live, export JSON.

## Features
- Drag & drop JPEG / MP4 onto canvas  
- Instant brightness-based depth → 3D point cloud (max ~8000 points)  
- Click to select individual points  
- Change color of single point or all points  
- Delete selected point  
- Adjustable point size, depth scale, grid, auto-rotate  
- Manual camera position + zoom sliders  
- Export full point cloud as JSON  
- Real-time FPS & point counter  
- Fully offline – single HTML file

## Quick Start
1. Download or copy `anything2.html`  
2. Open in any modern browser (Chrome/Edge/Firefox)  
3. Drag a JPEG photo or short MP4 onto the 3D view  
   → point cloud appears instantly  
4. Orbit with mouse, click points to edit

## Controls

### Left Panel Tabs
- **Point Cloud** – upload, recompute, reset, export  
- **Point Editor** – select point → change color / delete  
- **View Controls** – point size, depth scale, grid, auto-rotate

### Right Overlay (DA3 Point Cloud Controls)
- Camera Zoom / X / Y / Z sliders for precise framing

### Keyboard / Mouse
- Left click + drag → orbit  
- Right click + drag → pan  
- Scroll → zoom  
- Click any point → selects it (info shown in Point Editor tab)

## Export Format (point_cloud.json)
```json
{
  "points": [
    { "x": 1.23, "y": -0.45, "z": 3.67, "r": 0.9, "g": 0.1, "b": 0.2 },
    ...
  ],
  "count": 5421,
  "pointSize": "0.5",
  "depthScale": "12"
}
```

## Tips for Best Results
- Use high-contrast photos (portraits, objects, landscapes work great  
- Bright areas = closer, dark areas = farther  
- Keep source < 4K (automatically downscaled)  
- Recompute Depth after changing Depth Scale slider

## Tech
- Three.js r128 (CDN)  
- OrbitControls  
- No backend, no build step, no npm

License: MIT – fork, remix, use commercially.  
Made with neon love in 2025. Enjoy!
