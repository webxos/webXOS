#!/usr/bin/env python3
"""
WINEBOX – Zero-choice Virtual Windows Sandbox
=============================================
Everything lives under the folder that contains this script.

Flow
----
  1. Auto-Install: Checks for Wine, wine32, and winetricks; installs them if
     missing on Debian-based systems.
  2. [1] Install / Init: Sets up the Wine prefix, mirrors folder files to the
     desktop, disables X11 Gamma hijacking, and force-installs Gecko + 
     riched20/riched30 + d3dx9/vcrun6/corefonts so EULAs render correctly.
  3. [2] Global Repair & Launch: Unified linter and repair tool. Fixes generic
     Wine issues for any app, detects Diablo II, forces the 1.14d patch,
     bypasses the broken Blizzard updater, and sets up the optimized 60FPS 
     launcher if D2 is present.
  4. [3] Launch Winebox Desktop: Opens a shell desktop at your monitor's native
     resolution. You can click Maximize to make it fullscreen while retaining
     the standard Linux window title bar (X and Minimize buttons).
"""

from __future__ import annotations

import os
import sys
import shutil
import socket
import subprocess
import tempfile
import time
import textwrap
import getpass
import atexit
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

# ---------------------------------------------------------------------------
# Display Cleanup Fallback (Fixes Linux Gamma/Brightness issues)
# ---------------------------------------------------------------------------

def _restore_display_state() -> None:
    """Resets X11 gamma and brightness to defaults just in case Wine changed them."""
    try:
        subprocess.run(["xgamma", "-gamma", "1.0:1.0:1.0"], capture_output=True, timeout=3)
    except Exception:
        pass
    try:
        subprocess.run(["xrandr", "--brightness", "1.0"], capture_output=True, timeout=3)
    except Exception:
        pass

atexit.register(_restore_display_state)

# ---------------------------------------------------------------------------
# Fixed paths & constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PREFIX_DIR = SCRIPT_DIR / ".winebox_prefix"
DEPS_DECLINED_MARKER = SCRIPT_DIR / ".winebox_deps_declined"

GECKO_VERSION = "2.47.4"
GECKO_URL = (
    f"https://dl.winehq.org/wine/wine-gecko/"
    f"{GECKO_VERSION}/wine-gecko-{GECKO_VERSION}-x86.msi"
)

VDESKTOP_NAME = "shell"
DEFAULT_WIN_SIZE = "1024x768"
DEFAULT_WIN_VER = "winxp"
BNET_WIN_VER = "win7"

SKIP_NAMES = {
    "winebox.py",
    "README_DROP_HERE.txt",
    ".winebox_prefix",
    ".winebox_deps_declined",
    ".git",
}

# Game DLLs that must be resolved "native first" so Wine doesn't shadow the
# copies that ship with Diablo II. Without this, Wine's builtin storm.dll /
# binkw32.dll stubs intercept and the BNet handshake crashes to desktop.
GAME_DLL_OVERRIDES = (
    "storm=n,b;"
    "bnclient=n,b;"
    "d2net=n,b;"
    "d2mcpclient=n,b;"
    "d2multi=n,b;"
    "d2client=n,b;"
    "d2game=n,b;"
    "d2common=n,b;"
    "d2cmp=n,b;"
    "d2win=n,b;"
    "d2gfx=n,b;"
    "d2sound=n,b;"
    "d2lang=n,b;"
    "d2launch=n,b;"
    "fog=n,b;"
    "ijl11=n,b;"
    "smackw32=n,b;"
    "binkw32=n,b"
)

# System libs installed by winetricks that must resolve "native first" for
# Battle.net auth and UI rendering to work.
SYSTEM_DLL_OVERRIDES = (
    "winhttp=n,b;"
    "wininet=n,b;"
    "mfc42=n,b;"
    "msvcp60=n,b;"
    "msvcp140=n,b;"
    "vcruntime140=n,b"
)

# ---------------------------------------------------------------------------
# ANSI Colors
# ---------------------------------------------------------------------------

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"


def color(text: str, *styles: str) -> str:
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    return "".join(styles) + text + C.RESET


def clear() -> None:
    print("\033[2J\033[H", end="")


def pause(msg: str = "Press Enter to continue...") -> None:
    try:
        input(color(f"\n{msg}", C.DIM))
    except (KeyboardInterrupt, EOFError):
        print()
        pass


def _safe_input(msg: str) -> Optional[str]:
    try:
        return input(color(msg, C.DIM))
    except (KeyboardInterrupt, EOFError):
        print()
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def run(
    cmd: List[str],
    env: Optional[dict] = None,
    cwd: Optional[Path] = None,
    capture: bool = False,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    full = os.environ.copy()
    if env:
        full.update(env)
    kwargs: Dict[str, Any] = {
        "env": full,
        "cwd": str(cwd) if cwd else None,
        "capture_output": capture,
        "timeout": timeout,
    }
    if capture:
        kwargs["text"] = True
        kwargs["encoding"] = "utf-8"
        kwargs["errors"] = "replace"
    return subprocess.run(cmd, **kwargs)


def get_screen_resolution() -> str:
    """Uses xrandr to detect the host monitor's native resolution."""
    try:
        r = subprocess.run(
            "xrandr --current | grep '*' | awk '{print $1}' | head -n 1",
            shell=True, capture_output=True, text=True, timeout=3
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return "1024x768"


def wine_env(extra: Optional[dict] = None) -> dict:
    # ddraw=n,b prevents DirectDraw from hijacking X11 gamma
    overrides = ";".join((
        "winemenubuilder.exe=d",
        "mscoree=d",
        "ddraw=n,b", 
        GAME_DLL_OVERRIDES,
        SYSTEM_DLL_OVERRIDES,
    ))
    e = {
        "WINEPREFIX": str(PREFIX_DIR),
        "WINEARCH": "win32",
        "WINEDEBUG": "-all",
        "WINEDLLOVERRIDES": overrides,
    }
    if extra:
        e.update(extra)
    return e


def prefix_ready() -> bool:
    return (PREFIX_DIR / "system.reg").is_file()


def detect_wine() -> Tuple[bool, str]:
    w = which("wine")
    if not w:
        return False, "not found"
    try:
        r = run([w, "--version"], capture=True)
        return True, (r.stdout or r.stderr or "").strip()
    except Exception as ex:
        return False, str(ex)


def detect_wine32() -> Optional[bool]:
    dpkg = which("dpkg")
    if dpkg:
        try:
            r32 = subprocess.run([dpkg, "-s", "wine32:i386"],
                                 capture_output=True, text=True, timeout=3)
            if r32.returncode == 0:
                return True
            r64 = subprocess.run([dpkg, "-s", "wine64"],
                                 capture_output=True, text=True, timeout=3)
            if r64.returncode == 0:
                return False
        except Exception:
            pass

    arch_ok = True
    if dpkg:
        try:
            r = subprocess.run([dpkg, "--print-foreign-architectures"],
                               capture_output=True, text=True, timeout=3)
            if r.returncode == 0:
                arches = (r.stdout or "").split()
                if "i386" not in arches:
                    arch_ok = False
        except Exception:
            pass

    if not arch_ok:
        return False

    for p in (
        "/usr/lib/wine/i386-windows",
        "/usr/lib/i386-linux-gnu/wine/i386-windows",
        "/usr/lib32/wine/i386-windows",
        "/usr/lib/wine/i386-unix",
        "/usr/lib/wine/wine",
    ):
        try:
            if os.path.exists(p):
                return True
        except OSError:
            pass
    return None


def kill_wineserver(env: Optional[dict] = None) -> None:
    e = env if env is not None else wine_env()
    try:
        run(["wineserver", "-k"], env=e, timeout=10)
    except subprocess.TimeoutExpired:
        try:
            run(["wineserver", "-k9"], env=e, timeout=5)
        except Exception:
            pass
    except Exception:
        pass


def get_wine_desktop() -> Optional[Path]:
    users_dir = PREFIX_DIR / "drive_c" / "users"
    if not users_dir.is_dir():
        return None

    candidates: List[Path] = []
    try:
        for u in sorted(users_dir.iterdir()):
            try:
                if u.is_dir() and (u / "Desktop").is_dir():
                    candidates.append(u / "Desktop")
            except OSError:
                continue
    except OSError:
        return None

    if not candidates:
        return None

    cur_user = (os.environ.get("USER") or getpass.getuser()).lower()
    matches = [c for c in candidates if c.parent.name.lower() == cur_user]

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = [c.parent.name for c in matches]
        print(color(f"  ! Multiple Wine profiles match user '{cur_user}': "
                    f"{names}. Using '{names[0]}'.", C.YELLOW))
        return matches[0]
    return candidates[0]


# ---------------------------------------------------------------------------
# Gecko detection / installation helpers
# ---------------------------------------------------------------------------

def gecko_installed() -> bool:
    if not prefix_ready():
        return False

    gecko_dirs = [
        PREFIX_DIR / "drive_c" / "windows" / "system32" / "gecko",
        PREFIX_DIR / "drive_c" / "windows" / "syswow64" / "gecko",
    ]
    for gd in gecko_dirs:
        if not gd.is_dir():
            continue
        try:
            for entry in gd.iterdir():
                if not entry.is_dir():
                    continue
                if not entry.name.startswith("2."):
                    continue
                try:
                    if any(entry.iterdir()):
                        return True
                except OSError:
                    continue
        except OSError:
            continue
    return False


def _download_file(url: str, dest: Path) -> bool:
    if which("wget"):
        try:
            r = run(["wget", "-q", "-O", str(dest), url], timeout=300)
            if r.returncode == 0 and dest.is_file() and dest.stat().st_size > 0:
                return True
        except Exception:
            pass
        try:
            if dest.is_file():
                dest.unlink()
        except OSError:
            pass

    if which("curl"):
        try:
            r = run(["curl", "-fsSL", "-o", str(dest), url], timeout=300)
            if r.returncode == 0 and dest.is_file() and dest.stat().st_size > 0:
                return True
        except Exception:
            pass
        try:
            if dest.is_file():
                dest.unlink()
        except OSError:
            pass

    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(dest))
        if dest.is_file() and dest.stat().st_size > 0:
            return True
    except Exception:
        pass

    return False


def _run_winetricks(
    verbs: List[str],
    force: bool = False,
    unattended: bool = True,
    timeout: int = 900,
    env: Optional[dict] = None,
) -> int:
    winetricks = which("winetricks")
    if not winetricks:
        return -1

    cmd = [winetricks]
    if force:
        cmd.append("--force")
    if unattended:
        cmd.append("--unattended")
    cmd.extend(verbs)

    e = env if env is not None else wine_env()

    try:
        proc = subprocess.run(
            cmd,
            env={**os.environ, **e},
            timeout=timeout,
        )
        return proc.returncode
    except subprocess.TimeoutExpired:
        return -2
    except KeyboardInterrupt:
        raise
    except Exception:
        return -3


def _install_gecko_manual(env: dict) -> bool:
    print(color("        -> Manual Gecko fallback (MSI)…", C.DIM))

    url = GECKO_URL
    tmp_msi = Path(tempfile.gettempdir()) / f"wine-gecko-{GECKO_VERSION}-x86.msi"

    try:
        if tmp_msi.is_file():
            tmp_msi.unlink()
    except OSError:
        pass

    print(color(f"           Downloading {url}", C.DIM))
    if not _download_file(url, tmp_msi):
        print(color("           ! Download failed (check network / URL).", C.RED))
        return False

    print(color("           Installing MSI via wine msiexec…", C.DIM))
    try:
        r = run(
            ["wine", "msiexec", "/i", str(tmp_msi)],
            env=env,
            capture=True,
            timeout=300,
        )
        if r.returncode != 0:
            err = (r.stderr or r.stdout or "").strip()
            print(color(f"           ! msiexec rc={r.returncode}", C.YELLOW))
            if err:
                for line in err.splitlines()[-5:]:
                    print("             " + line)
    except subprocess.TimeoutExpired:
        print(color("           ! msiexec timed out after 300s.", C.RED))
    except Exception as ex:
        print(color(f"           ! msiexec failed: {ex}", C.RED))
        return False
    finally:
        try:
            run(["wineserver", "-w"], env=env, timeout=30)
        except Exception:
            pass
        kill_wineserver(env)
        try:
            if tmp_msi.is_file():
                tmp_msi.unlink()
        except OSError:
            pass

    return gecko_installed()


# ---------------------------------------------------------------------------
# Auto-Installer Logic
# ---------------------------------------------------------------------------

def auto_install_dependencies() -> None:
    if DEPS_DECLINED_MARKER.exists():
        return
    if not shutil.which("apt") or not shutil.which("dpkg"):
        return

    ok, _ = detect_wine()
    w32 = detect_wine32()
    wt = which("winetricks")

    if ok and w32 is True and wt:
        return

    clear()
    print(color("── Missing Dependencies Detected ─────────────────────", C.BOLD, C.YELLOW))
    print("Some required components are missing. These are needed for")
    print("EULA dialogs to render and for Winebox to work properly.")
    print()

    missing: List[str] = []
    if not ok:
        missing.append("wine")
    if w32 is not True:
        missing.append("wine32:i386")
    if not wt:
        missing.append("winetricks")

    print(f"  Missing: {', '.join(missing) if missing else 'unknown'}")
    print()
    print("The following commands will be executed:")
    print(color("  1. sudo dpkg --add-architecture i386", C.CYAN))
    print(color("  2. sudo apt update", C.CYAN))
    print(color("  3. sudo apt install -y wine32:i386 wine64 wine winetricks", C.CYAN))
    print()

    ans = _safe_input("Automatically install these packages now? (Requires sudo) [Y/n]: ")
    if ans is not None and ans.strip().lower() == "n":
        print(color("Skipping auto-install. A marker has been created so this prompt "
                    "won't appear again.", C.YELLOW))
        try:
            DEPS_DECLINED_MARKER.touch()
        except OSError:
            pass
        return

    print(color("\n  -> Priming sudo credentials...", C.DIM))
    try:
        r = subprocess.run(["sudo", "-v"])
        if r.returncode != 0:
            print(color("  ! sudo authentication failed.", C.RED))
            ans = _safe_input("Continue to menu anyway? [Y/n]: ")
            if ans is None or ans.strip().lower() != "n":
                try:
                    DEPS_DECLINED_MARKER.touch()
                except OSError:
                    pass
                return
            return
    except KeyboardInterrupt:
        print(color("\n  ! Cancelled.", C.RED))
        return
    except Exception as ex:
        print(color(f"  ! sudo failed: {ex}", C.RED))
        return

    cmds = [
        ["sudo", "-n", "dpkg", "--add-architecture", "i386"],
        ["sudo", "-n", "apt", "update"],
        ["sudo", "-n", "apt", "install", "-y",
         "wine32:i386", "wine64", "wine", "winetricks"],
    ]

    for cmd in cmds:
        print(color(f"\n  -> Running: {' '.join(cmd)}", C.DIM))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(color(f"\n  ! Command failed with exit code {e.returncode}.", C.RED))
            ans = _safe_input("Continue to menu anyway? [Y/n]: ")
            if ans is not None and ans.strip().lower() == "n":
                try:
                    DEPS_DECLINED_MARKER.touch()
                except OSError:
                    pass
            return
        except KeyboardInterrupt:
            print(color("\n  ! Installation cancelled.", C.RED))
            return

    print(color("\n  ✓ Dependencies installed successfully.", C.GREEN))


# ---------------------------------------------------------------------------
# Recommended components (fixes blank EULA + Diablo II)
# ---------------------------------------------------------------------------

def install_recommended_components() -> bool:
    print(color("\n── Installing recommended components ─────────────────", C.BOLD, C.CYAN))
    print("  These fix blank EULA dialogs and improve older games (Diablo II etc.)")

    if not prefix_ready():
        print(color("  ! Wine prefix not initialized. Cannot install components.", C.RED))
        return False

    winetricks = which("winetricks")
    if not winetricks:
        print(color("  ! winetricks not found.", C.YELLOW))
        if which("apt") and which("sudo"):
            print("    Attempting to install winetricks via apt…")
            try:
                r = subprocess.run(["sudo", "apt", "install", "-y", "winetricks"])
                if r.returncode == 0 and which("winetricks"):
                    print(color("  ✓ winetricks installed.", C.GREEN))
                    winetricks = which("winetricks")
                else:
                    print(color("  ! Failed to install winetricks.", C.RED))
                    return False
            except Exception as ex:
                print(color(f"  ! Install failed: {ex}", C.RED))
                return False
        else:
            print("    Install it with:  sudo apt install winetricks")
            return False

    env = wine_env()
    kill_wineserver(env)
    gecko_ok = False

    print(color("\n  [1/3] Gecko (HTML rendering – fixes blank EULA)", C.CYAN))
    if gecko_installed():
        print(color("        ✓ Gecko already present, skipping.", C.GREEN))
        gecko_ok = True
    else:
        print(color("        -> Running: winetricks --force --unattended gecko", C.DIM))
        rc = _run_winetricks(["gecko"], force=True, timeout=900, env=env)
        if gecko_installed():
            print(color("        ✓ Gecko installed via winetricks.", C.GREEN))
            gecko_ok = True
        else:
            print(color(f"        ! winetricks gecko failed (rc={rc}). Trying manual MSI…", C.YELLOW))
            if _install_gecko_manual(env):
                print(color("        ✓ Gecko installed via manual MSI.", C.GREEN))
                gecko_ok = True
            else:
                print(color("        ! Manual Gecko install also failed.", C.RED))

    print(color("\n  [2/3] RichEdit controls (riched20 + riched30)", C.CYAN))
    print(color("        -> Running: winetricks --force --unattended riched20 riched30", C.DIM))
    rc = _run_winetricks(["riched20", "riched30"], force=True, timeout=300, env=env)
    if rc == 0:
        print(color("        ✓ RichEdit controls installed.", C.GREEN))
    else:
        print(color(f"        ! RichEdit returned code {rc} (non-critical).", C.YELLOW))

    print(color("\n  [3/3] Extra components (d3dx9, vcrun6, corefonts)", C.CYAN))
    print(color("        -> Running: winetricks --unattended d3dx9 vcrun6 corefonts", C.DIM))
    rc = _run_winetricks(["d3dx9", "vcrun6", "corefonts"], force=False, timeout=600, env=env)
    if rc == 0:
        print(color("        ✓ Extra components installed.", C.GREEN))
    else:
        print(color(f"        ! Extra components returned code {rc} (non-critical).", C.YELLOW))

    print(color("\n  Verifying installation…", C.CYAN))
    if gecko_installed():
        print(color("  ✓ Gecko verified – EULA dialogs will now render correctly.", C.GREEN))
    else:
        print(color("  ! Gecko is still missing. EULA dialogs will remain blank.", C.RED))

    kill_wineserver(env)
    return gecko_ok


# ---------------------------------------------------------------------------
# Global Repair Helpers
# ---------------------------------------------------------------------------

def _fix_permissions() -> None:
    print(color("      -> Fixing file permissions in prefix...", C.DIM))
    try:
        subprocess.run(["chmod", "-R", "u+rwX", str(PREFIX_DIR)], timeout=30)
        print(color("      ✓ Permissions updated.", C.GREEN))
    except Exception as ex:
        print(color(f"      ! chmod failed: {ex} (non-critical).", C.YELLOW))


def _fix_x11_gamma(env: dict) -> None:
    """Disables Wine's ability to hijack X11 gamma/brightness settings."""
    print(color("      -> Disabling X11 Gamma Hijacking (Preserves desktop colors)...", C.DIM))
    reg = textwrap.dedent("""\
        Windows Registry Editor Version 5.00

        [HKEY_CURRENT_USER\\Software\\Wine\\X11 Driver]
        "UseXVidMode"="N"
        "UseXRandR"="N"
    """)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".reg", delete=False) as f:
            f.write(reg)
            tmp = f.name
        r = run(["wine", "regedit", "/S", tmp], env=env, capture=True, timeout=30)
        if r.returncode == 0:
            print(color("      ✓ Desktop colors protected (UseXVidMode=N).", C.GREEN))
    except Exception:
        pass
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _rename_blizzard_error() -> None:
    print(color("      -> Disabling Blizzard Crash Reporter...", C.DIM))
    found = False
    names = [
        "BlizzardError.exe", "Blizzard Error.exe", "ErrorReporter.exe",
        "BlizzardError.app", "Blizzard Crash Reporter.exe", "error.exe"
    ]
    for name in names:
        for p in PREFIX_DIR.rglob(name):
            try:
                bak = p.with_suffix(".exe.bak")
                if not bak.exists():
                    p.rename(bak)
                    print(color(f"      ✓ Disabled crash reporter: {p.name}", C.GREEN))
                    found = True
            except OSError:
                pass
    if not found:
        print(color("      ✓ Crash reporter not found (already disabled or missing).", C.DIM))


def _clear_bnet_registry(env: dict) -> None:
    print(color("      -> Clearing old Battle.net/Blizzard registry entries...", C.DIM))
    keys = [
        "HKCU\\Software\\Battle.net",
        "HKCU\\Software\\Blizzard Entertainment",
        "HKCU\\Software\\Blizzard Entertainment\\Diablo II",
        "HKCU\\Software\\Blizzard Entertainment\\Battle.net",
        "HKCU\\Software\\Wine\\AppDefaults\\Diablo II.exe",
        "HKCU\\Software\\Wine\\AppDefaults\\Game.exe"
    ]
    for key in keys:
        try:
            subprocess.run(
                ["wine", "reg", "delete", key, "/f"],
                env={**os.environ, **env},
                capture_output=True,
                timeout=15
            )
        except Exception:
            pass


def _set_windows_version(env: dict, version: str) -> bool:
    print(color(f"      -> Setting Windows version to '{version}' via winecfg…", C.DIM))
    try:
        r = subprocess.run(
            ["winecfg", "-v", version],
            env={**os.environ, **env},
            timeout=60,
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            print(color(f"      ✓ Windows version = {version}", C.GREEN))
            return True
    except Exception:
        pass

    reg = textwrap.dedent(f"""\
        Windows Registry Editor Version 5.00
        [HKEY_CURRENT_USER\\Software\\Wine]
        "Version"="{version}"
    """)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".reg", delete=False) as f:
            f.write(reg)
            tmp = f.name
        r = run(["wine", "regedit", "/S", tmp], env=env, capture=True, timeout=30)
        if r.returncode == 0:
            print(color(f"      ✓ Windows version = {version} (registry)", C.GREEN))
            return True
    except Exception:
        pass
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return False


def _write_bnet_gateways(env: dict) -> bool:
    null = "\x00"
    payload = (
        "100" + null +
        "useast.battle.net" + null +
        "uswest.battle.net" + null +
        "asia.battle.net" + null +
        "europe.battle.net" + null +
        null
    )
    hex_bytes = ",".join(f"{b:02x}" for b in payload.encode("utf-16-le"))

    reg = textwrap.dedent(f"""\
        Windows Registry Editor Version 5.00

        [HKEY_CURRENT_USER\\Software\\Battle.net\\Configuration]
        "Battle.net Gateways"=hex(7):{hex_bytes}
        "Battle.net Gateways Revision"=dword:00000001
    """)

    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".reg", delete=False) as f:
            f.write(reg)
            tmp = f.name
        r = run(["wine", "regedit", "/S", tmp], env=env, capture=True, timeout=30)
        if r.returncode == 0:
            print(color("      ✓ Battle.net Gateways written (useast/uswest/asia/europe).", C.GREEN))
            return True
    except Exception:
        pass
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return False


def _write_dll_overrides_registry(env: dict) -> None:
    print(color("      -> Writing DLL overrides to registry...", C.DIM))
    overrides = (
        "storm=n,b;bnclient=n,b;d2net=n,b;d2mcpclient=n,b;d2multi=n,b;"
        "d2client=n,b;d2game=n,b;d2common=n,b;d2cmp=n,b;d2win=n,b;"
        "d2gfx=n,b;d2sound=n,b;d2lang=n,b;d2launch=n,b;fog=n,b;"
        "ijl11=n,b;smackw32=n,b;binkw32=n,b;winhttp=n,b;wininet=n,b;"
        "mfc42=n,b;msvcp60=n,b;msvcp140=n,b;vcruntime140=n,b"
    )
    reg = textwrap.dedent(f"""\
        Windows Registry Editor Version 5.00

        [HKEY_CURRENT_USER\\Software\\Wine\\AppDefaults\\Diablo II.exe\\DllOverrides]
        @="{overrides}"

        [HKEY_CURRENT_USER\\Software\\Wine\\AppDefaults\\Game.exe\\DllOverrides]
        @="{overrides}"
    """)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".reg", delete=False) as f:
            f.write(reg)
            tmp = f.name
        run(["wine", "regedit", "/S", tmp], env=env, capture=True, timeout=30)
        print(color("      ✓ DLL overrides written.", C.GREEN))
    except Exception:
        pass
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _disable_ipv6(env: dict) -> bool:
    reg = textwrap.dedent("""\
        Windows Registry Editor Version 5.00

        [HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Services\\Tcpip6\\Parameters]
        "DisabledComponents"=dword:000000ff
    """)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".reg", delete=False) as f:
            f.write(reg)
            tmp = f.name
        r = run(["wine", "regedit", "/S", tmp], env=env, capture=True, timeout=30)
        if r.returncode == 0:
            print(color("      ✓ IPv6 disabled in prefix (fixes some BNet winsock crashes).", C.GREEN))
            return True
    except Exception:
        pass
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return False


def _test_bnet_connectivity() -> bool:
    servers = [
        ("useast.battle.net", 6112),
        ("uswest.battle.net", 6112),
        ("asia.battle.net", 6112),
        ("europe.battle.net", 6112),
    ]
    any_ok = False
    for host, port in servers:
        try:
            s = socket.create_connection((host, port), timeout=5)
            s.close()
            print(color(f"      ✓ {host}:{port} reachable.", C.GREEN))
            any_ok = True
        except socket.gaierror:
            print(color(f"      ! {host} did not resolve (DNS issue).", C.RED))
        except socket.timeout:
            print(color(f"      ! {host}:{port} timed out (firewall/ISP block?).", C.RED))
        except Exception as ex:
            print(color(f"      ! {host}:{port} unreachable: {ex}", C.RED))
    return any_ok


def _ntlm_available() -> bool:
    return which("ntlm_auth") is not None


def _install_winbind_if_needed() -> bool:
    if _ntlm_available():
        return True
    if not (which("apt") and which("sudo")):
        return False
    print(color("      ntlm_auth missing — installing winbind (needs sudo)…", C.YELLOW))
    try:
        r = subprocess.run(
            ["sudo", "apt", "install", "-y", "winbind"],
            timeout=300,
        )
        return r.returncode == 0 and _ntlm_available()
    except subprocess.TimeoutExpired:
        print(color("      ! apt install winbind timed out after 300s.", C.RED))
        return False
    except Exception as ex:
        print(color(f"      ! apt install winbind failed: {ex}", C.RED))
        return False


def _apply_manual_patch(env: dict) -> bool:
    print(color("      -> Applying manual 1.14d patch (bypasses corrupt BNUpdate)...", C.DIM))
    
    patch_url = "http://ftp.blizzard.com/pub/diablo2exp/patches/PC/LODPatch_114d.exe"
    tmp_patch = Path(tempfile.gettempdir()) / "LODPatch_114d.exe"
    
    if not _download_file(patch_url, tmp_patch):
        print(color("      ! Failed to download the manual patch. Check your internet.", C.RED))
        return False
        
    print(color("      ✓ Patch downloaded. Installing (this may take a moment)...", C.GREEN))
    
    try:
        r = run(
            ["wine", str(tmp_patch)],
            env=env,
            timeout=300,
            capture=False
        )
        
        try:
            if tmp_patch.is_file():
                tmp_patch.unlink()
        except OSError:
            pass
            
        if r.returncode == 0:
            print(color("      ✓ Manual 1.14d patch applied successfully.", C.GREEN))
            return True
        else:
            print(color(f"      ! Patch installer returned code {r.returncode}.", C.YELLOW))
            return False
            
    except Exception as ex:
        print(color(f"      ! Failed to run patch: {ex}", C.RED))
        return False


def _setup_optimal_launcher() -> None:
    print(color("      -> Setting up optimized 60 FPS launcher...", C.DIM))
    desktop = get_wine_desktop()
    if not desktop:
        print(color("      ! Could not find Wine Desktop to place launcher.", C.YELLOW))
        return

    game_exe = None
    for p in PREFIX_DIR.rglob("Game.exe"):
        game_exe = p
        break

    if game_exe:
        game_dir = game_exe.parent
        bat_content = f"""@echo off
title Diablo II - 60FPS
cd /d "{game_dir}"
Game.exe -3dfx
"""
        bat_path = desktop / "Launch_D2_60FPS.bat"
        try:
            with open(bat_path, "w") as f:
                f.write(bat_content)
            print(color(f"      ✓ Created optimized launcher on desktop: {bat_path.name}", C.GREEN))
            
            if not (game_dir / "glide3x.dll").exists():
                print(color("      ! Glide Wrapper (glide3x.dll) missing.", C.YELLOW))
                print(color("      ! Download Sven's Glide Wrapper and extract into:", C.DIM))
                print(color(f"        {game_dir}", C.DIM))
            else:
                print(color("      ✓ Glide Wrapper found.", C.GREEN))
                
        except Exception as e:
            print(color(f"      ! Failed to create launcher: {e}", C.YELLOW))
    else:
        print(color("      ! Could not find Game.exe. Launcher not created.", C.YELLOW))


# ---------------------------------------------------------------------------
# Option 2 – Global Repair, Lint & Launch
# ---------------------------------------------------------------------------

def global_repair_and_launch() -> None:
    """Universal repair for Wine prefix + Diablo II specific patches."""
    print(color("\n── [2] Global Repair, Lint & Launch ──────────────────", C.BOLD, C.CYAN))
    print("  Universal repair for any app, with reinforced fixes for Diablo II.")
    
    # --- 1. Lint ---
    print(color("\n  [1] Linting system...", C.CYAN))
    ok, ver = detect_wine()
    if ok:
        print(f"      [PASS] Wine is installed: {ver}")
    else:
        print(color("      [FAIL] Wine is missing.", C.RED))

    w32 = detect_wine32()
    if w32 is True:
        print("      [PASS] 32-bit Wine detected.")
    else:
        print(color("      [WARN] 32-bit Wine missing (needed for win32 games).", C.YELLOW))

    if not prefix_ready():
        print(color("      [FAIL] Wine prefix not initialized. Run [1].", C.RED))
        return
    print("      [PASS] Prefix initialized.")

    # --- 2. Generic Fixes ---
    print(color("\n  [2] Applying generic Wine fixes...", C.CYAN))
    _fix_permissions()
    _fix_x11_gamma(wine_env())
    _set_windows_version(wine_env(), BNET_WIN_VER)
    _disable_ipv6(wine_env())
    
    ntlm_ok = _install_winbind_if_needed()
    if ntlm_ok:
        print(color("      ✓ NTLM auth (winbind) active.", C.GREEN))
    else:
        print(color("      ! NTLM auth missing (Battle.net will freeze).", C.RED))

    env = wine_env()
    kill_wineserver(env)

    generic_verbs = ["winhttp", "wininet", "vcrun2015", "corefonts", "d3dx9"]
    for verb in generic_verbs:
        print(color(f"      -> winetricks --force {verb}", C.DIM))
        _run_winetricks([verb], force=True, timeout=600, env=env)
        kill_wineserver(env)
    print(color("      ✓ Generic components applied.", C.GREEN))

    # --- 3. Detect & Patch Diablo II ---
    game_exe = None
    for p in PREFIX_DIR.rglob("Game.exe"):
        game_exe = p
        break

    if game_exe:
        print(color("\n  [3] Diablo II Detected – Applying Reinforced Fixes...", C.CYAN))
        _rename_blizzard_error()
        _clear_bnet_registry(env)
        _write_bnet_gateways(env)
        _write_dll_overrides_registry(env)
        
        print(color("      -> Testing Battle.net reachability...", C.DIM))
        _test_bnet_connectivity()
        
        print(color("      -> Forcing 1.14d Patch...", C.DIM))
        _apply_manual_patch(env)
        
        _setup_optimal_launcher()
    else:
        print(color("\n  [3] Diablo II not detected. Skipping game-specific fixes.", C.DIM))

    # --- 4. Launch Prompt ---
    print(color("\n  [4] Repair Complete.", C.GREEN))
    if game_exe:
        ans = _safe_input("      Launch Diablo II (Optimized 60 FPS) now? [Y/n]: ")
        if ans is None or ans.strip().lower() != "n":
            launch_d2_optimized()
    else:
        ans = _safe_input("      Open Winebox Desktop to run your app? [Y/n]: ")
        if ans is None or ans.strip().lower() != "n":
            launch_sandbox()


# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------

def set_virtual_desktop(size: str = DEFAULT_WIN_SIZE) -> bool:
    reg = textwrap.dedent(f"""\
        Windows Registry Editor Version 5.00
        [HKEY_CURRENT_USER\\Software\\Wine\\Explorer]
        "Desktop"="{VDESKTOP_NAME}"
        [HKEY_CURRENT_USER\\Software\\Wine\\Explorer\\Desktops]
        "{VDESKTOP_NAME}"="{size}"
    """)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".reg", delete=False) as f:
            f.write(reg)
            tmp = f.name
        proc = run(["wine", "regedit", "/S", tmp], env=wine_env(),
                   capture=True, timeout=30)
        if proc.returncode != 0:
            print(color(f"  ! regedit failed (rc={proc.returncode})", C.YELLOW))
            return False
        return True
    except subprocess.TimeoutExpired:
        print(color("  ! regedit timed out", C.YELLOW))
        return False
    except Exception as ex:
        print(color(f"  ! regedit error: {ex}", C.YELLOW))
        return False
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def sync_folder_to_desktop() -> int:
    desktop = get_wine_desktop()
    if desktop is None:
        print(color("  ! Cannot sync: Wine Desktop not found (prefix not initialized).", C.RED))
        return 0

    count = 0
    try:
        items = list(SCRIPT_DIR.iterdir())
    except OSError as ex:
        print(color(f"  ! Cannot read {SCRIPT_DIR}: {ex}", C.RED))
        return 0

    for item in items:
        try:
            if item.name.startswith(".") or item.name in SKIP_NAMES:
                continue
            if not item.is_file():
                continue
        except OSError:
            continue

        target = desktop / item.name
        try:
            if target.exists():
                s_src = item.stat()
                s_dst = target.stat()
                if s_src.st_size == s_dst.st_size and abs(s_src.st_mtime - s_dst.st_mtime) <= 2:
                    continue
            shutil.copy2(item, target)
            count += 1
        except OSError as ex:
            print(color(f"  ! Copy {item.name}: {ex}", C.YELLOW))
    return count


def _boot_prefix(env: dict) -> Tuple[bool, Optional[int], str, bool]:
    log_fd, log_path = tempfile.mkstemp(prefix="winebox_boot_", suffix=".log")
    os.close(log_fd)

    rc: Optional[int] = None
    timed_out = False
    proc: Optional[subprocess.Popen] = None

    try:
        with open(log_path, "w", errors="replace") as logf:
            proc = subprocess.Popen(
                ["wineboot", "-i"],
                env={**os.environ, **env},
                stdout=logf,
                stderr=subprocess.STDOUT,
            )
            for i in range(36):
                try:
                    rc = proc.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    print(f"    ... {(i + 1) * 5}s elapsed", flush=True)
            if rc is None:
                timed_out = True
                print(color("    wineboot timed out after 180s, killing...", C.YELLOW))
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass
                rc = -9
    except KeyboardInterrupt:
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
        raise
    except Exception as ex:
        print(color(f"    Failed to spawn wineboot: {ex}", C.RED))
        rc = -1

    flush_ok = True
    try:
        r = run(["wineserver", "-w"], env=env, capture=True, timeout=30)
        if r.returncode != 0:
            flush_ok = False
            err = (r.stderr or "").strip()
            msg = f"    wineserver -w returned rc={r.returncode}"
            if err:
                msg += f": {err}"
            print(color(msg, C.YELLOW))
    except subprocess.TimeoutExpired:
        flush_ok = False
        print(color("    wineserver -w timed out after 30s", C.YELLOW))
    except Exception as ex:
        flush_ok = False
        print(color(f"    wineserver -w failed: {ex}", C.YELLOW))

    time.sleep(1)

    sys_reg_ok = (PREFIX_DIR / "system.reg").is_file()
    drive_c_ok = (PREFIX_DIR / "drive_c").is_dir()
    ok = flush_ok and sys_reg_ok and drive_c_ok

    return ok, rc, log_path, timed_out


def _print_boot_failure(rc: Optional[int], timed_out: bool, log_path: str) -> None:
    print(color("\n  ! wineboot did not create a usable prefix.", C.RED))
    print(color(f"    Return code: {rc}", C.RED))
    if timed_out:
        print(color("    (killed after 180s timeout)", C.RED))

    try:
        with open(log_path, "r", errors="replace") as f:
            lines = f.readlines()
        tail = lines[-25:] if len(lines) > 25 else lines
        if tail:
            print(color("    Last 25 lines of wineboot output:", C.YELLOW))
            for line in tail:
                print("      " + line.rstrip())
        else:
            print(color("    (wineboot produced no output)", C.YELLOW))
    except OSError:
        pass

    if rc == 0:
        print()
        print(color("  ! wineboot returned 0 but the prefix is incomplete.", C.YELLOW))
        print(color("    Common causes: wine32 missing / i386 arch / stuck wineserver", C.YELLOW))

    print(color(f"\n    Full log: {log_path}", C.DIM))


def install_winebox() -> bool:
    print(color("\n── [1] Install / Init Winebox ────────────────────────", C.BOLD, C.CYAN))

    ok, _ = detect_wine()
    if not ok:
        print(color("Error: Wine is not installed.", C.RED))
        return False

    w32 = detect_wine32()
    if w32 is False:
        print(color("Warning: wine32 (32-bit) is missing.", C.YELLOW))
        print("  WINEARCH=win32 requires it. Install with:")
        print("    sudo dpkg --add-architecture i386 && sudo apt install wine32:i386")
        ans = _safe_input("Continue anyway? [y/N]: ")
        if not ans or ans.strip().lower() != "y":
            return False

    if prefix_ready():
        print(color("Prefix already exists.", C.YELLOW))
        ans = _safe_input("Wipe and recreate? [y/N]: ")
        if not ans or ans.strip().lower() != "y":
            set_virtual_desktop(get_screen_resolution())
            print(color("Syncing current folder to Wine Desktop...", C.CYAN))
            n = sync_folder_to_desktop()
            print(color(f"✓ Synced {n} file(s) to the virtual desktop.", C.GREEN))

            ans2 = _safe_input(
                "Install / update recommended components "
                "(Gecko, richedit, d3dx9…)? [Y/n]: "
            )
            if ans2 is None or ans2.strip().lower() != "n":
                install_recommended_components()
            return True

        kill_wineserver()
        shutil.rmtree(PREFIX_DIR, ignore_errors=True)

    env = wine_env()
    auto_retry_used = False

    while True:
        try:
            PREFIX_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as ex:
            print(color(f"Failed to create {PREFIX_DIR}: {ex}", C.RED))
            return False

        print(color("Creating virtual sandbox...", C.CYAN))
        print("  -> Initializing wineboot...")

        success, rc, log_path, timed_out = _boot_prefix(env)

        if success:
            try:
                os.unlink(log_path)
            except OSError:
                pass
            break

        _print_boot_failure(rc, timed_out, log_path)

        if rc == 0 and not auto_retry_used:
            auto_retry_used = True
            print(color("\n    Auto-retrying wineboot once...", C.YELLOW))
            kill_wineserver(env)
            try:
                os.unlink(log_path)
            except OSError:
                pass
            continue

        print()
        ans = _safe_input("  [R]etry  [D]elete prefix and start over  [B]ack to menu [R/d/b]: ")
        if not ans:
            ans = "b"
        ans = ans.strip().lower()

        if ans == "r":
            kill_wineserver(env)
            try:
                os.unlink(log_path)
            except OSError:
                pass
            continue
        elif ans == "d":
            kill_wineserver(env)
            shutil.rmtree(PREFIX_DIR, ignore_errors=True)
            try:
                os.unlink(log_path)
            except OSError:
                pass
            continue
        else:
            print(color(f"    Prefix preserved at: {PREFIX_DIR}", C.DIM))
            return False

    try:
        run(["winecfg", "-v", DEFAULT_WIN_VER], env=env, timeout=90)
    except Exception:
        pass

    _fix_x11_gamma(env)
    set_virtual_desktop(get_screen_resolution())

    if get_wine_desktop() is None:
        print(color("  ! No Wine user Desktop detected after init.", C.YELLOW))
        try:
            run(["wine", "cmd", "/c", "exit"], env=env, capture=True, timeout=30)
        except Exception:
            pass

    print(color("Syncing current folder to Wine Desktop...", C.CYAN))
    n = sync_folder_to_desktop()

    if not install_recommended_components():
        print(color("\n  ! Recommended components may be incomplete.", C.YELLOW))
        print(color("  ! If EULA windows are blank, run [2] Global Repair for guidance.", C.YELLOW))

    print(color(f"\n✓ Winebox Ready. Synced {n} file(s) to the desktop.", C.GREEN))
    print("  Press [3] to launch the desktop or [2] to run a global repair.")
    return True


def launch_sandbox() -> bool:
    print(color("\n── [3] Launch Winebox Desktop ────────────────────────", C.BOLD, C.CYAN))

    if not prefix_ready():
        print(color("Error: Winebox is not installed. Run [1] first.", C.RED))
        return False

    desktop = get_wine_desktop()
    if desktop is None:
        print(color("Error: No Wine user Desktop found. Run [1] to re-initialize.", C.RED))
        return False

    # Auto-set to native resolution to allow true fullscreen maximize
    res = get_screen_resolution()
    print(f"  Setting virtual desktop to native resolution: {res}")
    set_virtual_desktop(res)
    desktop_arg = f"{VDESKTOP_NAME},{res}"

    print(f"  Desktop path : {desktop}")
    print("  Opening Wine Explorer...")
    print(color("  TIP: Click 'Maximize' (top right) to fullscreen.", C.CYAN))
    print(color("       The 'X' button will remain visible in the corner.", C.CYAN))
    print(color("  (Close the Wine window or use Start → Exit desktop to return)\n", C.DIM))

    env = wine_env()
    # Removed LIBGL_ALWAYS_SOFTWARE to allow GPU rendering
    env.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
    env["vblank_mode"] = "0"
    env["__GL_SYNC_TO_VBLANK"] = "0"
    env["MESA_GL_THREADED"] = "1"

    try:
        run(
            ["wine", "explorer", f"/desktop={desktop_arg}"],
            env=env,
            timeout=None,
        )
        _restore_display_state()
        return True
    except KeyboardInterrupt:
        kill_wineserver(env)
        _restore_display_state()
        return False
    except Exception as ex:
        print(color(f"Failed to launch: {ex}", C.RED))
        _restore_display_state()
        return False


def launch_d2_optimized() -> bool:
    print(color("\n── Launch Diablo II (60 FPS) ─────────────────────────", C.BOLD, C.CYAN))

    if not prefix_ready():
        print(color("Error: Winebox is not installed. Run [1] first.", C.RED))
        return False

    game_exe = None
    for p in PREFIX_DIR.rglob("Game.exe"):
        game_exe = p
        break

    if not game_exe:
        print(color("Error: Cannot find Game.exe. Make sure Diablo II is installed.", C.RED))
        print("Use option [3] to open the virtual desktop and install the game.")
        return False

    glide_dll = game_exe.parent / "glide3x.dll"
    if not glide_dll.exists():
        print(color("  ! Warning: glide3x.dll not found in the Diablo II folder.", C.YELLOW))
        print(color("  ! Download Sven's Glide Wrapper and extract it into:", C.DIM))
        print(color(f"    {game_exe.parent}", C.DIM))
        print(color("  ! Without it, the game will fall back to DirectDraw and may lag.", C.DIM))
        print()

    print(f"  Launching: {game_exe.name} -3dfx")
    print("  Using hardware acceleration with V-Sync unlocked.")
    print("  Press Ctrl+C in this terminal to force close if needed.\n")

    env = wine_env()
    # Removed LIBGL_ALWAYS_SOFTWARE to allow GPU rendering
    env["MESA_GL_VERSION_OVERRIDE"] = "3.3"
    env["vblank_mode"] = "0"
    env["__GL_SYNC_TO_VBLANK"] = "0"
    env["MESA_GL_THREADED"] = "1"
    env["mesa_glthread"] = "true"
    
    try:
        run(
            ["wine", str(game_exe), "-3dfx"],
            env=env,
            cwd=game_exe.parent,
            timeout=None,
        )
        _restore_display_state()
        return True
    except KeyboardInterrupt:
        kill_wineserver(env)
        _restore_display_state()
        return False
    except Exception as ex:
        print(color(f"Failed to launch: {ex}", C.RED))
        _restore_display_state()
        return False


# ---------------------------------------------------------------------------
# UI & Menu
# ---------------------------------------------------------------------------

def banner() -> None:
    print(color(r"""
 __        _____ _   _ _____ ____   _____  __
 \ \      / /_ _| \ | | ____| __ ) / _ \ \/ /
  \ \ /\ / / | ||  \| |  _| |  _ \| | | \  / 
   \ V  V /  | || |\  | |___| |_) | |_| /  \ 
    \_/\_/  |___|_| \_|_____|____/ \___/_/\_\
                                              
  Drag & Drop Sandbox  →  Winebox Explorer
""", C.CYAN, C.BOLD))


def main_menu() -> None:
    while True:
        clear()
        banner()

        print(color("── Menu ──────────────────────────────────────────────", C.BOLD))
        print("  [1]  Install / Init Winebox")
        print("       (Sets up Wine, syncs all files, force-installs Gecko + richedit + d3dx9…)")
        print("  [2]  Global Repair, Lint & Launch")
        print("       (Fixes Wine issues for ANY app. Auto-detects Diablo II,")
        print("        forces the 1.14d patch, bypasses the corrupt updater,")
        print("        disables crash reporter, and sets up the 60FPS launcher)")
        print("  [3]  Launch Winebox Desktop")
        print("       (Opens at native resolution. Maximize window for fullscreen.)")
        print("  [Q]  Quit")
        print()
        print(color("  (Tip: Press F11 to fullscreen this terminal window)", C.DIM))
        print()

        try:
            c = input(color("Select: ", C.BOLD)).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        try:
            if c == "1":
                install_winebox()
                pause()
            elif c == "2":
                global_repair_and_launch()
                pause()
            elif c == "3":
                launch_sandbox()
                pause()
            elif c in ("q", "quit", "exit"):
                print(color("\nBye.\n", C.CYAN))
                return
            else:
                print(color("Unknown option.", C.YELLOW))
                time.sleep(0.4)
        except KeyboardInterrupt:
            print(color("\nInterrupted.", C.YELLOW))
            pause()
        except Exception as ex:
            print(color(f"Unexpected error: {ex}", C.RED))
            pause()


def main() -> None:
    if os.name == "nt":
        print("This script is designed for Linux.")
        sys.exit(1)

    auto_install_dependencies()

    try:
        main_menu()
    except KeyboardInterrupt:
        print(color("\nInterrupted.", C.YELLOW))
        sys.exit(0)


if __name__ == "__main__":
    main()