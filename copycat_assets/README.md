# COPY CAT · SHADOW RACE - Game Assets

https://webxos.itch.io/copycat

Extracted and generated pixel-art assets from the original HTML5 canvas game.

## Style
- Monochrome (black / white) retro pixel art
- Pixel-perfect, `image-rendering: pixelated` friendly
- Original scale factor ~1.5x, assets provided at 4x for clarity (easy to downscale)

## Structure

```
game_assets/
│   ├── player_idle.png      # Player cat standing
│   ├── player_run.png       # 4-frame run cycle sheet (horizontal)
│   ├── player_jump.png      # Jump / mid-air pose
│   ├── shadow_idle.png      # Semi-transparent shadow cat
│   ├── shadow_run.png       # 4-frame shadow run sheet
│   ├── platform_ground.png  # Ground platform tile (repeatable)
│   ├── platform_float.png   # Floating platform
│   ├── spike.png            # Hazard spike
│   ├── echo.png             # Collectible diamond
│   └── particle.png         # Particle / spark
│   ├── title.png            # "COPY CAT" title text
│   ├── panel.png            # HUD / button panel frame
│   ├── hud_bg.png           # Score / time background
│   ├── echo_icon.png        # Echo collectible icon
│   ├── jump_indicator.png   # Double-jump star indicator
│   └── finish_flag.png      # Checkered finish line marker
    ├── star.png             # Background star
    ├── grid.png             # Subtle grid overlay
    └── scanline.png         # CRT scanline texture
```

## Notes
- Player & Shadow share the same geometry; shadow versions have ~35% opacity.
- Run sheets are 4 frames left-to-right. Frame size: 56×88 px (at 4x).
- Platforms are designed to tile horizontally.
- All sprites use transparent backgrounds (RGBA).
- Colors strictly match original: pure black (#000) and pure white (#FFF).

Generated for use in game engines (Godot, Unity, Phaser, etc.) or further editing.

**1-Bit Pixel Art Style Guide**  
*(Based on COPY CAT · SHADOW RACE)*

This is a pure **1-bit** (two-color) monochrome pixel art style: only pure black and pure white. No grays, no anti-aliasing, no color. The look is clean, high-contrast, retro, and slightly CRT/scanline-tinged.

---

### 1. Core Rules

| Rule | Exact Spec |
|------|------------|
| Colors | Only `#000000` (black) and `#FFFFFF` (white) |
| Background | Solid black (`#000`) |
| Pixel snapping | Everything must land on the integer pixel grid |
| Anti-aliasing | Forbidden |
| Transparency | Allowed only for secondary effects (shadows, particles, glows) using alpha on pure white |
| Resolution feel | Designed at low native resolution then scaled up with `image-rendering: pixelated` |

---

### 2. Palette & Values

```
Primary:   #FFFFFF  (solid forms)
Secondary: #000000  (background + cutouts/eyes)
Alpha uses (only on white):
  0.02–0.08  → distant stars / grid / mountain silhouettes
  0.12–0.25  → HUD text, labels, soft glows
  0.30–0.45  → shadow character
  0.50–0.70  → invincibility flash / particles
```

Never use gray. Fake depth only with alpha or by leaving black holes in white shapes.

---

### 3. Character Design (the Cat)

Original native size ≈ **14 × 22 px** (before the game’s 1.5× scale).

**Construction order (always this sequence):**

1. Body rectangle  
2. Head rectangle  
3. Two ear rectangles  
4. Two eye squares (black)  
5. Mouth line  
6. Legs (4-frame run cycle)  
7. Tail (two small rectangles + slight vertical offset for wag)

**Key proportions:**
- Head is almost as wide as the body
- Ears are short and square
- Eyes are large relative to the head
- Legs are short stubs
- Tail is stubby and sits high on the back

**Animation frames (run cycle – 4 frames):**

| Frame | Leg positions |
|-------|---------------|
| 0 / 2 | Legs under body (neutral) |
| 1     | Front legs forward, back legs back |
| 3     | Opposite of frame 1 |

Double-jump / barrel-roll = same sprite rotated + extra particles.

**Shadow version** = identical geometry, drawn at ~30–40 % opacity white. Never invert the colors.

---

### 4. Environment Rules

**Ground platforms**
- Solid white blocks
- 2 px black line on the very top edge
- Sparse black “texture” rectangles inside (brick suggestion)
- Bottom edge slightly brighter alpha (optional)

**Floating platforms**
- Thin white slab
- Black outline on top and bottom edges only
- Very light shadow underneath (low-alpha white)

**Spikes**
- Pure white triangles pointing up
- Drawn as filled polygons or stacked right triangles
- Slight black inner bevel for depth (optional, keep minimal)

**Collectibles (Echoes)**
- Simple diamond (rhombus) made of four points
- Pulsing size + alpha over time
- Soft circular glow underneath at very low alpha

---

### 5. Visual Effects Vocabulary

| Effect | Technique |
|--------|-----------|
| Screen shake | Random offset of the entire canvas (few pixels) |
| Flash | Full-screen white at low alpha that quickly fades |
| Particles | Tiny white squares with velocity + gravity + life |
| Glow / aura | Large low-alpha circle behind the shadow cat |
| Scanlines | 1 px black lines every 3 px at ~3.5 % opacity |
| Stars | 1×1 or 2×2 white pixels with slow sine twinkle |
| Distant mountains | Very low-alpha white rectangles with sine-wave tops |

---

### 6. Typography & HUD

- Font: monospaced (Courier New or similar)
- All uppercase
- Letter-spacing slightly increased
- Text is pure white at reduced alpha (0.15–0.40)
- Important numbers sit inside small bordered panels (black fill + white 2 px border)

---

### 7. Technical Implementation Tips

**Canvas / Pixel Art**
```js
ctx.imageSmoothingEnabled = false;   // critical
// or CSS: image-rendering: pixelated;
```

**Drawing style**
- Prefer `fillRect` over paths for characters and platforms
- Use exact integer coordinates
- Scale factor (e.g. 1.5× or 2× or 4×) applied after drawing, never before

**Animation timing**
- Run cycle: advance phase by ~0.14 per frame when moving
- Idle: very slow phase
- Barrel roll: high phase speed + continuous rotation

---

### 8. What This Style Is Not

- No dithering
- No grayscale
- No outlines thicker than 1–2 px
- No rounded shapes (everything is rectangular or simple polygons)
- No complex shading

---

### 9. Quick Recreation Checklist

1. Start with pure black canvas  
2. Draw only pure white shapes  
3. Cut eyes / details with pure black rectangles  
4. Keep character under 20 px tall at native size  
5. Animate with discrete frame changes or simple phase offsets  
6. Add secondary elements only with low-alpha white  
7. Finish with scanlines + very subtle grid  

---

This style is extremely readable at any size, cheap to animate, and instantly gives a sharp, modern-retro feel. The original game stays strictly inside these constraints, which is why it looks so clean and consistent.

## License: MIT
webXOS 2026 (visit https://webxos.itch.io to play COPYCATS) 
