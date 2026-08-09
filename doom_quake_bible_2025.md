*I generated this guide in 2025: The idea was to give this guide to LLMs, with attached plans to code in three js to create doom/quake fps clone games as close as possible*

### THE DOOM AND QUAKE BIBLE FOR THREE.JS 2025

**Edition: November 25, 2025** 
**Author: webXOS software - x.com/webxos - webxos.netlify.app** 

    The "Doom Bible" and "Quake Bible" were internal design documents used by id Software, outlining the games' backstories, characters, and high-level mechanics, rather than technical guides for 3D rendering. The actual rendering techniques used in the games involved two very different approaches and can be implemented using modern tools like the Three.js JavaScript library. 

## 3D Design & Rendering Principles 

The Doom engine did not render a "true" 3D world in the modern sense. It was a 2.5D engine that used a highly optimized software renderer for speed on contemporary hardware. 

    World Structure: The environment was based on a 2D map with a fixed height for walls, floors, and ceilings. It used sectors (areas with a defined floor and ceiling height) and was organized using a Binary Space Partitioning (BSP) tree for efficient visibility determination.

    Rendering Technique: The engine used a form of raycasting combined with a column-drawing approach (known as the "Doom engine" renderer). It determined which vertical columns of pixels on the screen corresponded to walls and drew them from front to back, using clipping to avoid overdraw.

    Objects: Enemies and items were 2D sprites (billboards) that always faced the player camera. 


## iD Software 3D Design & Rendering Principles 

Quake was id Software's first "true" 3D engine, utilizing polygons (triangles) and a full 3D space. 

    World Structure: Quake used a fully 3D world made of polygonal models (meshes) for both the environment and objects. Like Doom, it used a BSP tree for optimization, but in full 3D space.

    Rendering Technique: Quake implemented hardware-accelerated rendering (via OpenGL) as an option, which processed 3D vertices and projected them to the screen using the graphics card, a significant departure from Doom's software-only approach.

    Objects: All objects, including player models and enemies, were 3D polygonal models that could be viewed from any angle.
    Lighting: Quake introduced lightmaps and dynamic lighting effects, adding significant realism compared to Doom's simpler distance-based shading. 

## Modern Rendering in HTML with Three.js 

Modern 3D web development with Three.js inherently uses the "true" 3D, polygon-based approach similar to the Quake engine, leveraging WebGL for hardware acceleration in the browser. 

To recreate the style of these games, you would focus on specific implementation aspects within a standard Three.js setup: 

    Scene: Create a basic Three.js Scene with a PerspectiveCamera.
    Geometry:

        For a Doom-style renderer, you might build custom geometries in Three.js or even write custom shaders to simulate the 2.5D effect and use THREE.Sprite for enemies, but it is much simpler to just use 3D geometry and enforce Doom's visual constraints (e.g., restricted vertical camera movement).

        For a Quake-style renderer, use standard THREE.Mesh objects with BufferGeometry or load models in formats like glTF (modern equivalent of Quake's MD2/MDL formats).

    Materials & Textures: Use THREE.MeshBasicMaterial for unlit/flat-shaded effects to mimic early software renderers, or THREE.MeshStandardMaterial with lightmaps or simple point lights to emulate Quake's lighting model.

    Workflow: Instead of building a custom software renderer like id Software did for Doom, Three.js handles the complex rasterization and projection, allowing you to focus on game logic and visual style. You can find extensive tutorials and documentation on the Three.js official website and MDN Web Docs. 

## FROM ID TECH TO WEBGL

Doom Bible (Tom Hall, 1992): Story-heavy sci-fi horror on Tei Tenga moon base. 4 playable marines (BJ Blazkowicz, etc.), cinematics, hubs, Unmaker weapon. Scrapped for pure action by Carmack/Romero.

Quake Bible: No formal doc like Doom's. Romero's 10-page 1996 email + chaotic dev (D&D-inspired, Cthulhu, shifting from RPG to FPS). True 3D polygons, MD2 models, lightmaps.

This Bible: Recreates both in Three.js. Doom: Simulate 2.5D raycast/BSP via custom shaders/sprites. Quake: Native meshes/lightmaps. 2025 updates: WebGPU path-traced lighting, glTF models, physics via Cannon/Ammo.

**Requirements:**  
- Three.js r181 CDN: `<script src="https://cdn.jsdelivr.net/npm/three@0.181.0/build/three.min.js"></script>`  
- Editor: https://threejs.org/editor/ (import scenes)  
- Tools: glTF loaders, dat.GUI for debug.  

**Core Loop (all examples):**  
```js
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

function animate() {
  requestAnimationFrame(animate);
  // Update logic here
  renderer.render(scene, camera);
}
animate();
```

### DOOM ENGINE

## World Structure: Sectors + BSP Tree

Doom: 2D map → sectors (floor/ceiling heights). BSP precomputes visibility (front-to-back draw, no Z-buffer).

Three.js: Use planes/meshes for walls. Custom BSP via THREE.BufferGeometry + shader sim. Load WAD via parser.

**BSP Simulator Code (Doom-style visibility):**  
```js
// Simplified BSP node (from DoomWiki impl)
class BSPNode {
  constructor(splitX, splitY, front, back) {
    this.splitX = splitX; this.splitY = splitY;
    this.front = front; this.back = back; // Child nodes or leaf
  }
  traverse(playerX, playerY, callback) {
    if (this.isLeaf()) { callback(this.sector); return; }
    const side = (playerX - this.splitX) * this.dirY - (playerY - this.splitY) * this.dirX > 0 ? 1 : -1;
    if (side > 0) { this.front.traverse(playerX, playerY, callback); this.back.traverse(playerX, playerY, callback); }
    else { this.back.traverse(playerX, playerY, callback); this.front.traverse(playerX, playerY, callback); }
  }
}

// Build scene from BSP
function buildDoomScene(bspRoot, textures) {
  const walls = new THREE.Group();
  bspRoot.traverse(camera.position.x, camera.position.z, (sector) => {
    // Gen wall meshes from linedefs
    sector.linedefs.forEach(ld => {
      const geometry = new THREE.PlaneGeometry(1, sector.ceilingH - sector.floorH);
      const material = new THREE.MeshBasicMaterial({ map: textures[ld.texture] });
      const wall = new THREE.Mesh(geometry, material);
      wall.position.set(ld.x, (sector.ceilingH + sector.floorH)/2, ld.y);
      walls.add(wall);
    });
  });
  scene.add(walls);
}
```
Import WAD parser: Use https://github.com/emericg/OpenDoomJS (adapt for Three.js).

## Rendering

Doom: Raycast per screen column → draw vertical spans (walls/floors). Y-shear for "look up/down".

Three.js: Custom fragment shader mimics column texel fetch. No true raycast (WebGL rasterizes).

**Raycast Shader (Doom-style):**
```glsl
// Vertex shader: Pass UV/Y for column sim
varying vec2 vUv;
varying float vDepth;

void main() {
  vUv = uv;
  vDepth = (modelViewMatrix * vec4(position, 1.0)).z;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

// Fragment: Fake raymarch columns
uniform sampler2D wallTex;
uniform float raySteps;

void main() {
  float column = vUv.x * raySteps; // Screen column
  float dist = vDepth; // Ray distance
  float texY = vUv.y * (1.0 / dist); // Perspective scale
  vec4 color = texture2D(wallTex, vec2(column / 256.0, texY));
  gl_FragColor = color * (1.0 - dist * 0.1); // Distance fog
}
```
Attach to wall planes. Full demo: https://github.com/smol/doom (Three.js port).

**Floors/Ceilings: Floor-caster Sim**
```js
const floorGeo = new THREE.PlaneGeometry(1000, 1000);
const floorMat = new THREE.MeshBasicMaterial({ 
  vertexShader: `...`, // Project floor tex based on ray dist
  fragmentShader: `texture2D(floorTex, gl_FragCoord.xy / dist)`
});
```

## Objects

Enemies/items: 2D sprites always face camera.

```js
class DoomSprite {
  constructor(tex, pos) {
    this.sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex }));
    this.sprite.position.copy(pos);
    scene.add(this.sprite);
  }
  update() {
    this.sprite.lookAt(camera.position); // Billboard
  }
}

// Imp example
const impTex = new THREE.TextureLoader().load('imp.png');
const imp = new DoomSprite(impTex, new THREE.Vector3(5, 0, 5));
```

## Player Controls

From ourcade/threejs-getting-started.
```js
const keys = {}; // WASD
document.addEventListener('keydown', e => keys[e.code] = true);
document.addEventListener('keyup', e => keys[e.code] = false);

function updatePlayer(dt) {
  const speed = 5 * dt;
  const rotSpeed = 2 * dt;
  if (keys['KeyA']) camera.rotation.y += rotSpeed;
  if (keys['KeyD']) camera.rotation.y -= rotSpeed;
  const dir = new THREE.Vector3(0,0,-1).applyQuaternion(camera.quaternion);
  if (keys['KeyW']) camera.position.addScaledVector(dir, speed);
  if (keys['KeyS']) camera.position.addScaledVector(dir, -speed);
  if (keys['KeyQ']) camera.position.add(new THREE.Vector3(1,0,0).applyQuaternion(camera.quaternion).multiplyScalar(speed)); // Strafe
}
```

## Lighting

No dynamics. Shader-based.
```js
material.uniforms.fogDensity = { value: 0.1 };
```

## Demoscene
Export to Three.js editor JSON. Load E1M1 WAD → BSP → render.

### QUAKE ENGINE 

### World Structure

Quake: Full 3D BSP (nodes split space). Brushes → convex polys. PVS for vis.

Three.js: Load BSP29 via parser → BufferGeometry.

```js
// Parse .bsp (Quake1 format)
function loadQuakeMap(url) {
  const loader = new THREE.FileLoader();
  loader.load(url, (mapText) => {
    const bsp = parseBSP(mapText); // Custom parser: nodes, faces, verts
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(bsp.verts, 3));
    geo.setIndex(bsp.indices);
    const mat = new THREE.MeshLambertMaterial({ map: bsp.tex });
    const mesh = new THREE.Mesh(geo, mat);
    scene.add(mesh);
  });
}
```
Parser impl: https://dev.to/mcharytoniuk/loading-quake-engine-maps-in-three-js-part-1-parsing-55mp

## Rendering

Quake: Software → GLQuake (vertex proj, rasterize).

Three.js: Native. Multisample for AA.
```js
renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.shadowMap.enabled = true; // Dynamic shadows
```

Lightmaps: Dual textures (diffuse + lightmap).
```js
const mat = new THREE.MeshLambertMaterial({ 
  map: diffuseTex,
  lightMap: lightmapTex,
  lightMapIntensity: 1.0
});
```

## Objects

MD2 anim loader → glTF (2025 std).
```js
const loader = new THREE.GLTFLoader();
loader.load('ogre.gltf', (gltf) => {
  const model = gltf.scene;
  model.animations = gltf.animations; // Idle, walk, attack
  model.mixer = new THREE.AnimationMixer(model);
  scene.add(model);
});
```

## Lighting

Static: Prebake (Quake .bsp). Dynamic: PointLights.
```js
const light = new THREE.PointLight(0xffaa00, 1, 100);
light.position.set(10, 10, 10);
light.castShadow = true;
scene.add(light);
scene.add(new THREE.AmbientLight(0x404040));
```

## Player Controls
```js
let mouseX = 0, mouseY = 0;
document.addEventListener('mousemove', e => {
  mouseX += e.movementX * 0.002;
  mouseY += e.movementY * 0.002;
  mouseY = Math.max(-Math.PI/2, Math.min(Math.PI/2, mouseY));
  camera.rotation.order = 'YXZ';
  camera.rotation.y = -mouseX;
  camera.rotation.x = -mouseY;
});
```

### ADVANCED FEATURES

## Physics: Cannon.js/Ammo for Both
```js
import * as CANNON from 'cannon-es';

const world = new CANNON.World({ gravity: -9.82 });
const playerBody = new CANNON.Body({ mass: 1 });
world.addBody(playerBody);
```

Doom: Grounded movement. Quake: Jump/slide.

## Post-Processing
```js
const composer = new POSTPROCESSING.EffectComposer(renderer);
composer.addPass(new POSTPROCESSING.RenderPass(scene, camera));
composer.addPass(new POSTPROCESSING.ScanlinesEffect());
```

## WebGPU Path Tracing
Use three-gpu-pathtracer: https://github.com/gkjohnson/three-gpu-pathtracer

**Extended Guide:**

Similar to Q1K3 by Dominic Szablewski (phoboslab), a masterful 13KB JS homage to 1996 Quake. It needs to use extreme optimization: procedural textures, axis-aligned block maps, compact model data, custom tiny audio, and heavy compression.

The Three.js frames per second can benefit from these ideas: for better performance, smaller bundle size (great for web sharing), retro Quake aesthetics, and smoother gameplay. Focus on **efficiency** while leveraging Three.js strengths (WebGL rendering, built-in materials, etc.).

*Start by implementing procedural textures and AABB-based levels/collision in your existing project — you'll see immediate gains in size and retro authenticity. This hybrid approach (Three.js rendering + Q1K3-style data) gives great visuals without bloat.*

### 1. Core Architecture & Rendering Optimizations
- **Use a Custom or Simplified Renderer for Retro Feel**: Q1K3 uses a software-style rasterizer in JS/WebGL. In Three.js, stick with `WebGLRenderer` but simplify:
  - Low-poly geometry (axis-aligned where possible).
  - Disable unnecessary features: turn off anti-aliasing for performance, limit shadow maps or use baked lighting.
  - Target 60 FPS: Use `requestAnimationFrame`, cap updates, and profile with Three.js stats or browser dev tools.
- **Instancing & Batching**: Group similar objects (walls, enemies) into `InstancedMesh` to reduce draw calls dramatically.
- **LOD & Culling**: Implement simple frustum culling or distance-based LOD for enemies/pickups.
- **Resolution Scaling**: Add a settings option to render at lower internal resolution and upscale (like retro games).

**Three.js Tip**: Set `renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))` and experiment with lower values for performance.

### 2. Procedural Textures (Biggest Win for Size & Style)
Q1K3 generates 31 textures procedurally at runtime using a tiny library (~1.3KB zipped), avoiding large image files.

**How to Implement in Three.js**:
- Use HTML5 Canvas to generate textures dynamically.
- Create a simple "Tiny Texture Generator" inspired by Q1K3's TTT (embossed rects, noise layers, grids).
- Example skeleton:

```js
function createQuakeTexture(width = 64, height = 64) {
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d');
  
  // Base color
  ctx.fillStyle = '#444444';
  ctx.fillRect(0, 0, width, height);
  
  // Embossed panels, rivets, noise, etc.
  // Add Perlin/simplex noise (include a tiny noise lib or implement simple one)
  // ctx.putImageData for pixel-level noise
  
  const texture = new THREE.CanvasTexture(canvas);
  texture.magFilter = THREE.NearestFilter; // Retro pixel look
  texture.minFilter = THREE.NearestFilter;
  return texture;
}
```

- Apply to `MeshStandardMaterial` or `MeshPhongMaterial` (Phong for more retro).
- Reuse textures heavily. Generate once at load and store in a map.
- For models: Project textures frontally (simple UVs) — accept some stretching for tiny size.

This keeps your bundle tiny and gives authentic 90s Quake vibes (metallic panels, stone, etc.).

### 3. Maps & Level Data (Axis-Aligned Blocks)
Q1K3 uses TrenchBroom → custom packer to axis-aligned blocks (position + size + texture ID). Collision is trivial.

**For Your Game**:
- Build levels in TrenchBroom (free Quake editor) or Blender, then export to a compact JSON/binary format (only AABBs).
- Or procedurally generate simple levels.
- In Three.js: Create `BoxGeometry` instances for each block. Merge geometries where possible (same material) using `BufferGeometryUtils.mergeGeometries` for fewer draw calls.
- Store levels compactly: arrays of `[x, y, z, w, h, d, texIndex]`.

**Collision**:
- For player/enemies: Use simple AABB intersection tests (very fast). No need for full physics engine unless you want slopes.
- Three.js `Raycaster` for bullets/projectiles, but limit checks or use a spatial grid/octree for larger maps.

Add doors as scaled boxes with animation.

### 4. Models & Entities
- **Compact Format**: Store as vertex lists + indices (bytes). Reuse meshes heavily (one humanoid scaled/textured differently for enemy types).
- In Three.js: Use `BufferGeometry` with low vertex counts. Animate via morph targets or simple frame swapping (like Q1K3's 2-6 frames).
- Gibs, pickups, projectiles: Reuse box/cylinder geometries.
- Weapons: Viewmodel in screen space or attached to camera (simple meshes).

### 5. Gameplay Settings for Proper Quake Feel
Configure these for authentic fast-paced FPS:

- **Movement**:
  - Air acceleration, bunny hopping support (Quake-style).
  - Gravity ~800-1000 units, jump velocity strong.
  - Max speed ~300-400 units/sec.
  - Use velocity damping on ground, less in air.

- **Weapons**:
  - Hitscan for shotgun/nailgun + projectile for rockets/grenades.
  - Muzzle flash, recoil (camera kick), sound.
  - 3 weapons like Q1K3: Shotgun, Nailgun, Rocket/Plasma.

- **Enemies**:
  - Simple AI: Line-of-sight raycast + state machine (idle, chase, attack).
  - No complex pathfinding initially (steer towards player, avoid walls via multiple rays).
  - Health, gib on death with particles.

- **Physics/Collision**:
  - Player capsule or AABB with step-up for small ledges.
  - Fast projectile collision.

- **Lighting**:
  - Dynamic point lights for muzzle flashes/explosions.
  - Ambient + directional for base.
  - Optional lightmaps for static areas.

### 6. Audio (Tiny Footprint)
- Use a modified tiny synth like Sonant-X (as in Q1K3) or ZzFX for sfx.
- Generate sounds procedurally.
- Spatial audio: `PositionalAudio` in Three.js with distance/rolloff.
- Music: Short looped tracker-style track.

### 7. Build & Compression Pipeline (Aim for Small Size)
- Minify JS (Terser/Uglify).
- Use Roadroller (advanced JS compressor used in js13k).
- Inline assets (base64 if tiny, or procedural).
- ZIP the final HTML if distributing as single file.
- Tools: `build.sh` style script with custom packers (see Q1K3 repo).

**Example Pipeline**:
1. Develop uncompressed (easy debugging).
2. Bundle with Vite/Rollup.
3. Run compressors.
4. Test final zipped size.

### 8. Performance & Polish Tips
- Profile often: Watch draw calls (<200 ideal), texture binds, JS GC.
- Particles: Simple CPU particles or GPU with Points/BufferGeometry.
- UI/HUD: DOM overlay or CanvasTexture for retro feel.
- Controls: Pointer lock, WASD + mouse, jump, weapon switch.
- Testing: Mobile? Add touch fallback but prioritize desktop for FPS.

### Resources to Study
- Q1K3 full source: https://github.com/phoboslab/q1k3 (highly recommended — study `source/`, pack scripts, map packer).
- Making-of: https://phoboslab.org/log/2021/09/q1k3-making-of (goldmine of techniques).
- Three.js examples: FPS controls, raycasting, instancing.
- TrenchBroom for maps.


