# COMMAND AND QUANTIFY BETA v1.0
## WebXOS 2025 Game Manual

### Overview
4-player grid conquest. You (Quantum Commander, green) vs 3 CPUs. Destroy enemy Quantum Cores to win. Expand territory turn-by-turn.

### Setup
- **Grid**: 16x16 (switchable to 8x8).
- **Players**:
  | Player | Color | Type |
  |--------|-------|------|
  | 0 | Green | You |
  | 1 | Blue | CPU Alpha (Tactical) |
  | 2 | Yellow | CPU Beta (Defensive) |
  | 3 | Red | CPU Gamma (Aggressive) |
- **Start**: Each gets Quantum Core in corner + 50 Quantum Units (QU).

### Entities
| Icon | Name | Cost | Health | Movable | Notes |
|------|------|------|--------|---------|-------|
| ⚛️ | Quantum Core | 250 | 100 | Yes | Win by destroying enemies'. 25% self-defend chance. |
| ⚔️ | Barracks | 100 | 60 | No | Enables Army production (up to armies ≤ barracks). 50% self-defend. |
| 🌾 | Quantum Farm | FREE | 40 | No | +10 QU/round per farm. Instant destroy. |
| 🎯 | Army | 100 | 50 | Yes | Attacks adjacent. 50% win vs other Army. |

### Your Turn (Green Panel)
1. **Build**: Click button → highlights valid spots (adjacent to your territory).
   - Farms: Free, expand safely.
   - Barracks: For armies.
   - Army: Needs barracks capacity.
2. **Move/Attack**: Click your movable entity (⚛️/🎯) → highlights adjacent cells.
   - Empty: Move.
   - Enemy: Attack (auto-resolve).
3. **Auto-end**: Action completes turn.

**QNN Sliders** (tune your AI assist):
- Aggression: Risk-taking.
- Memory: Learns from history.
- Speed: Fast vs precise.

### Rules
- **Expand**: Build only adjacent to your entities.
- **Move**: Adjacent only (1 step).
- **Combat**:
  | Attacker → Defender | Outcome |
  |---------------------|---------|
  | Any → Core | Core 25% destroy attacker |
  | Any → Barracks | 50% defend |
  | Any → Farm | Destroyed |
  | Army → Army | 50% win |
- **Income**: +10 QU/farm at round end.
- **Elimination**: Core destroyed → out. Last standing wins.

### CPU Turns
- Watch panels: "QNN Calculating...".
- They build/move/attack automatically.

### Win
Destroy all 3 enemy Cores. Victory screen shows rankings.

### Controls
- **Grid Mode**: Dropdown (restarts).
- **Restart**: "QUANTUM RESTART" button.
- **Mobile**: Touch grid/entities.

**QNN Mechanics:**

- **Model**: TF.js sequential NN (input: 10 state features; hidden: 16 ReLU → 8 ReLU; output: 4 softmax actions: farm/barracks/army/move).
- **Sliders** (player-tunable, CPU-fixed profiles):
  | Param | Effect | Player Default | CPU Examples |
  |-------|--------|----------------|--------------|
  | Aggression | Risk/attack bias | 50% | Alpha:65%, Beta:35%, Gamma:85% |
  | Memory Depth | History learning | 50% | Alpha:75%, Beta:85%, Gamma:30% |
  | Speed/Accuracy | Fast vs precise | 50% | Alpha:45%, Beta:60%, Gamma:70% |
- **Beta v1.0**: Model inits + dummy predict; sliders update params (unused). CPU: rule-based (attack cores → farms → barracks → armies → random).
- **Training**: Game JSON (entities/moves) feeds lightweight NN for spatial agents (drones/chess). Browser sims evolve policy via epochs/loss.