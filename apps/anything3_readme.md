# webXOS | ANYTHING3 — Point Cloud Video Converter

*webxos.netlify.app/anything3*

Real-time browser-based video → 3D point cloud converter built with Three.js (r128).  
Matrix-style UI · Optimized for low-end machines · No server required.

Live Demo: Drop this `index.html` in any browser.

## Features

- Drag & drop video upload
- Adjustable resolution (200–1200 px)
- Point density & depth intensity controls
- Auto-brightness correction per frame
- Live Three.js point cloud preview
- Play/stop animation (8 FPS capped for performance)
- Orbit controls + reset view
- Matrix rain background
- Terminal-style status log

## How to Use

1. Open `index.html` in a modern browser
2. Drop a video file (mp4, webm, etc.) or click "Upload Video"
3. Adjust:
   - **Resolution** – higher = more detail (slower)
   - **Point Density** – more/fewer points
   - **Depth Intensity** – stronger 3D pop
4. Click **Convert to Point Cloud**
5. Wait for conversion (modal progress)
6. Press **Play** to watch the point cloud animation
7. Orbit with mouse, **Reset View** or **Clear Scene** as needed

## Performance Notes

- Max 30 frames converted (keeps RAM < 300 MB)
- Fixed 8 FPS playback
- Point size 0.05 (attenuated)
- Tested on 8 GB RAM laptops

## Dependencies (CDN)

html:
script src = https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js
script src = https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.min.js
