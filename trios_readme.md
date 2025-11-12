# TRIOS - BETA TEST: GLADIATOR DRONE SIMULATOR - Full Mechanics Guide

## Overview
TRIOS is a browser-based wireframe arena shooter built with Three.js. Play as Captain Purge in a neon colosseum, defeating 3 escalating dragons (Yellow, Orange, Red TRIOS) across waves. Win by slaying all. Features Diablo 2-style leveling (1-99), sword drone companion, heat-managed gatling, boost mechanics, immortal mode, save/load, chat commands. Beta test for xAI gaming trilogy.

## Controls
| Key/Mouse | Action | Description |
|-----------|--------|-------------|
| WASD | Move | Arena strafing |
| Mouse | Aim | Pointer lock FPS view |
| LMB | Fire Gatling | Hitscan neurot shots; heat builds, overheats at 100% |
| Space | Boost | Jetpack ascent; fuel regen idle |
| Shift | Speed Boost | 1.5x move speed; fuel drain |
| R | Sword Drone | Deploy/target lightsaber drone (60s CD) |
| ESC | Menu | Pause/return to main |
| Enter | Chat | /help, /immortal, /level X, /kill |

Mobile: Touch-adapted.

## Weapons
- **Neurot Gatling (LMB)**: Rapid hitscan tracers w/ muzzle flash, recoil sway. Damage scales w/ level. Heat +2/shot, cools 30/s idle. Overheat: 3s lockout. Immortal: unlimited/no heat.
- **No secondary** (RMB free).

## Companion: Sword Drone
Neon green cyber lightsaber drone.
- **Deploy (R)**: Target via crosshair/closest dragon. Attacks 10s (2.5% HP/s = 25% total), parries. Red target crosshair.
- **Cooldown**: 60s regrow.
- **Re-R**: Re-target if active.

## Waves (Dragons)
Clear via gatling/sword drone. Massive HP, fire neurot flames (splash dmg). Orbit player, boundary-locked colosseum.

| Wave | Name | HP | Dmg | Speed | Size | Behavior |
|------|------|----|-----|-------|------|----------|
| 1 | Yellow Dragon | 50k | 25 | 12 | 0.8 | Fast melee gimbal, rapid fireballs |
| 2 | Orange Dragon | 150k | 35 | 8 | 1.2 | Ranged spammer |
| 3 | Red TRIOS | 500k | 50 | 6 | 1.5 | Boss: lift grab (3s airborne throw, 15s CD) + minions (Yellow/Orange) |

Cinematics btwn waves. Run timer: MM:SS.

## Progression
- **Leveling**: Diablo 2 exp curve (1-99). Kills: Yellow=1k, Orange=5k, Red=25k EXP. Levels boost HP/dmg.
- **Win**: All waves clear. Stats: time, dmg taken, dragons (3), final lvl.
- **Immortal**: Menu toggle; inf HP/ammo/heat.

## HUD
- **Top L**: Health (green), Heat (yellow), Boost (yellow) bars.
- **Top C**: Dragon HP bar (shows when present).
- **Top R**: FPS, Wave/X/3.
- **Bot R**: Ammo (INF/OVERHEAT), Gatling.
- **Bot L**: Sword Drone CD bar/status [R].
- **Center**: Crosshair, sword target (red).
- **Other**: Run time, level/EXP, chat terminal, save/load, fullscreen/skills/help.

## Beta Features
- **Chat**: /help (guide), /immortal, /level X (1-99), /kill (curr dragon).
- **Save/Load**: JSON export/import progress.
- **Immortal Default**: Testing mode.

## Speedrun Tips
- Sword Drone on bosses (25% insta-chunk).
- Gatling bursts; vent heat w/ boost dodges.
- Orbit dragons; boost evade fireballs/lift.
- Immortal for PB practice; disable for legit.
- Level max via /level 99 testing.
- RTA: ~5-10min w/ drone enrage bursts.

**Open Source Template**: Fork/extend for trilogy.
