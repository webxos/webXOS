# Asset Generation Skill (asset_gen_skill.md)
This skill processes a single HTML/CSS file or Three.js web application to extract, generate, and package high-quality game assets into a downloadable `.zip` archive. 
## 🛠️ System Capabilities
The AI engine analyzes the visual DOM elements, canvas rendering contexts, CSS styles, or Three.js scene graphs to output production-ready game assets. 
### 1. Extraction Types* **Three.js Scenes**: Captures 3D mesh projections, procedural textures, and material shaders.* **HTML/CSS Canvas**: Extracts dynamically drawn 2D pixel art, UI frames, and vector elements.* **CSS Sprites**: Segments stylized bounding boxes, buttons, and typography sheets.
### 2. Export Specifications* **Format**: Standard ZIP compilation.
* **Image Assets**: Transparent `.png` sheets, optimized `.svg` vectors.
* **3D Assets**: Validated `.gltf` or `.obj` formats with embedded textures.
* **Organization**: Structured subdirectories (`/sprites`, `/ui`, `/textures`).
---## 🚀 Execution Workflow

[Input: Code File] ➡️ [Analyze Visual/Scene Tree] ➡️ [Render Elements] ➡️ [Slice & Pack] ➡️ [Output: .zip]


### Step 1: Input Analysis
Parse the target file to determine the source technology stack:
* **Three.js**: Locate the WebGL renderer, active scene, camera layout, and geometry definitions.
* **Canvas API**: Trace `2d` rendering context draw calls (`fillRect`, `drawImage`, path vectors).
* **HTML/CSS**: Map CSS variables, box-shadow patterns, backgrounds, and borders.

### Step 2: Visual Generation & Isolation
Isolate visual elements against transparent backgrounds:
* Remove background DOM colors or environmental skyboxes unless explicitly requested.
* Apply correct scaling coefficients to preserve pixel-perfect snapping for retro art styles.
* Render 3D objects from isometric, orthogonal, or flat-facing perspectives to create 2D sprites.

### Step 3: Asset Packaging Structure
Generate a structured workspace before compilation. The generated `.zip` file must adhere to this uniform schema:

```text
game_assets.zip/
├── ui/
│   ├── buttons.png
│   ├── panels.png
│   └── icons/
├── sprites/
│   ├── player_sheet.png
│   └── environment.png
└── textures/
    └── material_maps/
```

### Step 4: Asset Generation Commands
Execute the internal compilation pipeline:

```bash
# Internal processing schema (Representation only)
\(asset-gen --source=index.html --output-format=png --pixel-ratio=1 --target=./staging\) zip -r asset_package.zip ./staging
```

---

## 🎨 Asset Style Profiles

The generator adapts outputs based on code attributes discovered during analysis:

| Source Attribute | Detection Marker | Generated Output Asset Type |
| :--- | :--- | :--- |
| **Pixel Art Rendering** | `image-rendering: pixelated` | Clean, grid-aligned `.png` sprite-sheets |
| **3D Three.js Mesh** | `new THREE.Mesh()` | Textures, heightmaps, or `.gltf` models |
| **UI Components** | Flexible layout, buttons, text | Sliced UI kits and modular HUD elements |

---

## ⚠️ Processing Safeguards
* **No Lost Assets**: Ensure every visual component declared in code is represented in the ZIP file.
* **Texture Clamping**: Prevent bleed on sprite-sheets by adding a 2-pixel transparent padding buffer around frames.
* **Color Accuracy**: Maintain exact hex color spaces defined in CSS or material configurations.

**Study Attached HTML File and Produce a full asset.zip with all the files I requested.**
