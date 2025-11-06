# X-FORCE: PAINTBALL  
**webXOS 2025 Esports Edition**  
*Desktop Only Beta Test*

# [http://webxos.netlify.app/paintball](http://webxos.netlify.app/paintball)

## Overview
Team-based core assault. Destroy enemy core (500k HP). Blue vs Red. Player + AI drones. One-hit kills. Wireframe arena.

## Teams & Modes
- **Blue Team**: Left base (-105 to -75 X).
- **Red Team**: Right base (75 to 105 X).
- **Modes**: 3v3 (3v3 incl. player) | 10v10 (10v10 incl. player).
- Switch: T key (in-game).

## Controls
| Action | Key/Mouse |
|--------|-----------|
| Forward | W |
| Backward | S |
| Left | A |
| Right | D |
| Jump/Gatling | Space (hold) |
| Jetpack | Left Shift (hold) |
| Fire (High Damage) | Left Click (hold auto) |
| Scope (Zoom) | Right Click |
| Switch Team | T |
| Pause/Menu | ESC |
| Exit | ESC (menu/fullscreen) |

- **Mouse**: Look. Sensitivity: Settings (0.1-0.5).
- **Invert Y**: Toggle in settings.

## Mechanics
- **Movement**: 6 speed. Jump: 8 up. Gravity: 20 down.
- **Jetpack**: +15 up/sec. No gravity.
- **Gatling**: Space=3x small shots (5 dmg, 80 speed, 0.1s rate).
- **High Damage**: LMB=50 dmg (60 speed). 3x core bonus near (<30m).
- **Projectiles**: Explode on hit. Particles.
- **Collision**: Boxes/cylinders block. Adjusted pathing.
- **Boundaries**: X ±105, Z ±45, Y 1.7-48.3.

## Respawn & Invuln
- **Death**: 10s timer. One-hit kill outside base.
- **Respawn**: Random base pos. 10s invuln (stay in base!).
- **Drones**: Same. AI defends 5s post-respawn.

## Objectives
1. Kill enemy drones (10 pts).
2. Damage enemy core (proximity bonus).
3. Protect your core.
- **Win**: Enemy core 0 HP.
- **Loss**: Your core 0 HP.

## AI Drones
- **Behavior**:
  - Defend core/base (initial 5s).
  - Attack nearest enemy (<20m).
  - Core assault if safe.
  - Path around obstacles.
- **Stats**: 100 HP, 4 speed, 10 dmg, 1.2s fire rate.
- **Total**: Mode-dependent + player side.

## UI Elements
- **Cores**: Top bars (blue/red HP).
- **Team**: Top-center.
- **KD/Score**: Top-right.
- **Jetpack**: READY/ACTIVE.
- **Crosshair**: Range labels. Lock pulse.
- **Kill Feed**: Right (5 entries).
- **Console**: Center alerts.
- **Respawn**: Timer/invuln counters.
- **FPS**: Bottom-right.

## Strategy
- **Early**: Gatling drones. Jetpack flank.
- **Mid**: Scope core shots (proximity max dmg).
- **Late**: Defend core. Switch team if losing.
- **Tips**:
  - Invuln = stay base.
  - Bunkers: Central (0,0), sides (±15-25).
  - Cores: (-90/90,10,0).

## Settings
- **Mode**: 3v3/10v10.
- **Controls**: Mouse/Xbox/PS5/USB.
- **Sens**: 0.1-0.5.
- Saves to localStorage.

**Rewards**: 50 $WEBXOS on victory.  (BETA TEST PLACEHOLDER)
**Play Again**: Post-game menu.
