# PowerShell Megalith Guide (PS1)

## Goal

Create the Windows / cross-platform equivalent of MSPYB and UNIXAI: a single `.ps1` file that:

- Detects environment (Windows / Linux / macOS via PowerShell Core),
- Installs required modules or packages idempotently,
- Generates a complete project tree (or embeds a full interactive agent),
- Auto-launches the resulting application,
- Logs every decision with structured output,
- Remains re-runnable and offline-friendly.

## Core Pattern

```powershell
#Requires -Version 7.0
<#
.SYNOPSIS
  MSPYB-style PowerShell Bootstrap / UNIXAI-style Agent
.DESCRIPTION
  Single source of truth. Install → Generate → Auto-Launch.
.NOTES
  LLM-INSTRUCTION: Edit only this file; regenerate everything else.
#>

$script:MSPYB_VERSION = "3.0.0"
$script:TEMPLATE_VARS = @{
    ProjectName   = "MyApp"
    API_PORT      = 8000
    AUTO_LAUNCH   = "true"
    LAUNCH_COMMAND = "python -m app.main"
    LAUNCH_CWD    = "."
}

# --- Logging (must be first) ---
function Write-BootstrapLog { ... }

# --- Environment detection ---
function Get-Environment {
    [PSCustomObject]@{
        OS           = $PSVersionTable.OS
        Platform     = $PSVersionTable.Platform
        PowerShell   = $PSVersionTable.PSVersion
        HasDocker    = [bool](Get-Command docker -ErrorAction SilentlyContinue)
        Python       = (Get-Command python -ErrorAction SilentlyContinue)?.Source
    }
}

# --- INSTALLS ---
function Install-RequiredPackages {
    param([switch]$Skip)
    if ($Skip) { Write-BootstrapLog "Skipping installs"; return }
    # Use Install-Module, pip, winget, apt, brew according to platform
}

# --- Templating + write-file ---
function Write-ProjectFile {
    param($RelativePath, $Content)
    $full = Join-Path $ProjectRoot $RelativePath
    $dir  = Split-Path $full -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $Content | Set-Content -Path $full -Encoding utf8
    Write-BootstrapLog "Wrote $RelativePath"
}

# --- Generation sections ---
# Write shared logging, app entry points, configs, docker-compose, etc.

# --- Auto-Launch ---
function Invoke-AutoLaunch {
    if ($TEMPLATE_VARS.AUTO_LAUNCH -ne "true") {
        Write-BootstrapLog "Auto-launch disabled"
        return
    }
    Push-Location (Join-Path $ProjectRoot $TEMPLATE_VARS.LAUNCH_CWD)
    try {
        $proc = Start-Process -FilePath "pwsh" -ArgumentList "-c", $TEMPLATE_VARS.LAUNCH_COMMAND `
                              -PassThru -NoNewWindow
        Write-BootstrapLog "Launched PID $($proc.Id)"
    }
    catch {
        Write-BootstrapLog "Launch failed: $_ — recovery: verify LAUNCH_COMMAND and INSTALLS"
    }
    finally { Pop-Location }
}

# --- Main ---
$ProjectRoot = Join-Path (Get-Location) $TEMPLATE_VARS.ProjectName
$envInfo = Get-Environment
Install-RequiredPackages -Skip:$SkipInstalls
# ... generate files ...
Invoke-AutoLaunch
Write-Host "MSPYB PowerShell bootstrap complete. Version $MSPYB_VERSION"
```

## Agent-Style Commands (UNIXAI analogue)

Inside an interactive PS1 agent:

- Use a `switch -Regex` on user input for `/help`, `/model`, `/firewall`, etc.
- Maintain a bounded conversation history as an `ArrayList` of hashtables.
- Parse model replies for `<cmd>...</cmd>` tags and prompt for confirmation (with a safety firewall).
- Background jobs via `Start-Job` or `Start-Process` for daemons (whisper, triage, pipe).

## Platform Notes

| Concern            | Recommendation                                      |
|--------------------|-----------------------------------------------------|
| Cross-platform     | Prefer PowerShell 7+ (`pwsh`)                       |
| Package install    | `Install-Module` + `pip` / `winget` / `apt` / `brew`|
| Docker             | Guard with `$envInfo.HasDocker`                     |
| Secrets            | Never embed; generate `.env.example` only           |
| Executable bit     | On Unix, `chmod +x` after generation if needed      |
| Logging            | Structured JSON lines + colored console             |

## Dynamic-Programming View

The single `.ps1` is both policy and value function. An LLM edits the bootstrap; re-execution produces a fresh, consistent project state. Overlapping subproblems (environment detection, logging setup, launch recovery) are solved once and reused across every generation.
