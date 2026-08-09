# COPY CAT · SHADOW RACE - Game Assets

https://webxos.itch.io/copycats

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

## License: MIT
webXOS 2026 (visit https://webxos.itch.io to play COPYCATS) 
