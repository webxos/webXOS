# WINEBOX – Zero-choice Virtual Windows Sandbox
```
 __        _____ _   _ _____ ____   _____  __
 \ \      / /_ _| \ | | ____| __ ) / _ \ \/ /
  \ \ /\ / / | ||  \| |  _| |  _ \| | | \  / 
   \ V  V /  | || |\  | |___| |_) | |_| /  \ 
    \_/\_/  |___|_| \_|_____|____/ \___/_/\_\
```
**Winebox** is a single-file, zero-configuration Wine wrapper that turns any Linux folder into a self-contained Windows sandbox. 
Drop Windows installers or games next to `winebox.py`, run the script, and everything lives under the same directory — no system-wide Wine pollution, no manual prefix management.

It is built around one reliable purpose: **Diablo II + Battle.net** support for Linux Debian systems. (1.14d patch, Glide 60 FPS, correct gateways, DLL overrides, crash-reporter disable), but the repair path works for any Win32 application. Other applications might work but have not been tested. 

*Future updates might also include Steam or other windows only platforms.*

---

## Features

| Feature | Description |
|---------|-------------|
| **Self-contained prefix** | Everything lives in `.winebox_prefix` next to the script |
| **Auto-dependency install** | Detects missing `wine`, `wine32:i386`, and `winetricks` on Debian/Ubuntu and offers to install them |
| **One-click init** | Creates a clean 32-bit Wine prefix, sets native resolution virtual desktop, copies every file in the folder onto the Wine Desktop |
| **Gecko + RichEdit force-install** | Fixes blank EULA / HTML dialogs that normally plague Wine |
| **Global Repair mode** | Universal Wine fixes + aggressive Diablo II specific patches |
| **Diablo II optimized path** | Forces 1.14d, writes correct Battle.net gateways, disables Blizzard Error reporter, sets native-first DLL overrides, creates a 60 FPS Glide launcher |
| **Native resolution desktop** | Virtual desktop matches your monitor; maximize the window for true fullscreen while keeping the Linux title-bar controls |
| **Gamma protection** | Prevents Wine from hijacking X11 gamma / brightness |
| **Clean exit** | Always restores display gamma/brightness on exit |

---

## Requirements

- Linux (Debian / Ubuntu / derivatives recommended)
- Python 3.8+
- X11 (or XWayland)
- Optional but recommended: `sudo` rights for the auto-installer

The script will try to install these packages if missing:

```bash
sudo dpkg --add-architecture i386
sudo apt update
sudo apt install -y wine32:i386 wine64 wine winetricks
```

---

## Quick Start

1. Place `winebox.py` in an empty folder (or a folder that already contains your Windows `.exe` / `.msi` installers).
2. Make it executable (optional):

   ```bash
   chmod +x winebox.py
   ```

3. Run it:

   ```bash
   python3 winebox.py
   # or
   ./winebox.py
   ```

4. Follow the menu.

---

## Menu

```
[1]  Install / Init Winebox
[2]  Global Repair, Lint & Launch
[3]  Launch Winebox Desktop
[Q]  Quit
```

### [1] Install / Init Winebox

- Creates (or re-creates) the Wine prefix at `./.winebox_prefix`
- Forces `WINEARCH=win32`
- Sets Windows version to XP by default
- Detects your monitor resolution with `xrandr` and creates a matching virtual desktop
- Copies every non-hidden file from the current folder onto the Wine Desktop
- Force-installs:
  - **Gecko** (HTML rendering – critical for EULAs)
  - **riched20 + riched30**
  - **d3dx9**, **vcrun6**, **corefonts**
- Disables Wine’s X11 gamma hijacking

If the prefix already exists you are asked whether to wipe it or just re-sync files + components.

### [2] Global Repair, Lint & Launch

Universal repair path that also contains the heavy Diablo II logic.

**Generic fixes applied to any prefix**

- File permissions repair
- X11 gamma protection
- Windows version set to Windows 7 (better for modern Battle.net / many apps)
- IPv6 disabled inside the prefix
- `winbind` / `ntlm_auth` installed if missing (required for Battle.net auth)
- `winhttp`, `wininet`, `vcrun2015`, `corefonts`, `d3dx9` via winetricks

**If Diablo II is detected** (`Game.exe` found anywhere under the prefix)

- Renames / disables all known Blizzard crash reporters
- Clears old Battle.net / Blizzard registry keys
- Writes the four classic gateways (`useast`, `uswest`, `asia`, `europe`)
- Forces native-first DLL overrides for every Diablo II / Storm / Bink library
- Tests connectivity to the four Battle.net servers on port 6112
- Downloads and runs the official **LODPatch_114d.exe** (bypasses the broken in-game updater)
- Creates `Launch_D2_60FPS.bat` on the Wine Desktop that launches with `-3dfx`
- Warns if `glide3x.dll` (Sven’s Glide Wrapper) is missing

You are then offered to launch the optimized 60 FPS Diablo II or open the desktop.

### [3] Launch Winebox Desktop

- Opens `wine explorer /desktop=shell,<native-resolution>`
- Virtual desktop is sized exactly to your monitor
- Click the Maximize button on the Wine window for fullscreen while keeping the normal Linux title bar (close / minimize still work)
- Environment variables force GPU rendering and unlock V-Sync:
  - `MESA_GL_VERSION_OVERRIDE=3.3`
  - `vblank_mode=0`
  - `__GL_SYNC_TO_VBLANK=0`
  - `MESA_GL_THREADED=1`

---

## Diablo II Specific Notes

Winebox was hardened specifically for the classic Diablo II + Lord of Destruction experience under Wine.

### Required extra files (place next to the game or let the installer put them)

- Official 1.14d patch (Winebox downloads and applies it automatically)
- **Sven’s Glide Wrapper** (`glide3x.dll`) – place it in the same folder as `Game.exe`

### Recommended launch method

Use the generated desktop shortcut or option [2] → “Launch Diablo II (Optimized 60 FPS)”.  
It runs:

```bash
Game.exe -3dfx
```

with the correct environment and DLL overrides.

### Why the heavy DLL overrides?

Wine’s built-in `storm.dll`, `binkw32.dll`, `bnclient.dll`, etc. are incomplete or broken for the old Battle.net protocol.  
Winebox forces the real game DLLs to load first (`n,b` = native, then builtin).

### Common remaining issues

| Symptom | Fix |
|---------|-----|
| Blank EULA / HTML windows | Run [1] or [2] (Gecko must be present) |
| Battle.net freezes on login | Install `winbind` (Winebox tries to do this) |
| Crash on “Connecting…” | Make sure 1.14d is applied and gateways are written |
| Low FPS / stuttering | Install Sven’s Glide Wrapper + launch with `-3dfx` |
| Gamma / brightness changed after exit | Winebox restores it automatically via `atexit` |

---

## File Layout

```
your-folder/
├── winebox.py                 ← the only file you need
├── .winebox_prefix/           ← created on first run (do not delete while running)
├── .winebox_deps_declined     ← marker if you said “no” to auto-install
├── Game.exe / installers…     ← drop any Windows files here
└── README.md                  ← this file
```

Everything the script creates stays inside the folder. You can move the whole directory to another machine (as long as Wine is present) and it will continue working.

---

## Environment Variables Used

```bash
WINEPREFIX=./.winebox_prefix
WINEARCH=win32
WINEDEBUG=-all
WINEDLLOVERRIDES=winemenubuilder.exe=d;mscoree=d;ddraw=n,b;storm=n,b;… (long list)
```

Plus the Mesa / V-Sync overrides listed earlier when launching the desktop or Diablo II.

---

## Safety & Side Effects

- Never touches your system Wine prefix (`~/.wine`)
- Never installs packages without asking (and remembers if you decline)
- Always restores X11 gamma and brightness on exit
- Uses temporary files for registry edits and cleans them up
- Kills only its own wineserver instance

---

## Advanced / Manual Usage

You can still use the prefix with normal Wine commands:

```bash
export WINEPREFIX=$(pwd)/.winebox_prefix
export WINEARCH=win32
winecfg
wine yourapp.exe
```

Or force the virtual desktop yourself:

```bash
wine explorer /desktop=shell,1920x1080
```

---

## Troubleshooting

**“Wine is not installed”**  
Run the script once and accept the auto-installer, or install manually:

```bash
sudo dpkg --add-architecture i386
sudo apt update
sudo apt install wine32:i386 wine64 wine winetricks
```

**Prefix creation hangs / times out**  
Usually caused by missing 32-bit libraries. The script will show the last 25 lines of the `wineboot` log.  
Delete `.winebox_prefix` and try again after installing `wine32:i386`.

**Gecko keeps failing**  
The script falls back to downloading the MSI from WineHQ and installing it with `msiexec`.  
If that also fails, check network connectivity and try running option [2].

**Battle.net still broken after repair**  
1. Confirm `ntlm_auth` exists (`which ntlm_auth`)
2. Confirm the four gateways resolve and port 6112 is open
3. Make sure you are on the official 1.14d patch
4. Try launching from the generated `Launch_D2_60FPS.bat`

---

## License / Credits

This is a single-file utility. Feel free to modify, redistribute, or embed it.  
Diablo II / Battle.net are trademarks of Blizzard Entertainment.  
Wine is copyright of the Wine project.

---

**Enjoy the sandbox.**  
Drop files → run `winebox.py` → play.
