<#
.SYNOPSIS
  Creates a clickable "VoiceAgent" shortcut icon on the current user's Windows Desktop.
#>
param(
  [switch]$GpuModels
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$desktop = [Environment]::GetFolderPath("Desktop")
$electronExe = Join-Path $root "node_modules\electron\dist\electron.exe"
$mainJs = Join-Path $root "apps\desktop\electron-main.cjs"
$iconPath = Join-Path $root "apps\desktop\assets\voiceagent.ico"

if (-not (Test-Path $electronExe)) {
  throw "Electron runtime not found at $electronExe"
}

$name = if ($GpuModels) { "VoiceAgent (GPU)" } else { "VoiceAgent" }
$lnkPath = Join-Path $desktop "$name.lnk"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($lnkPath)
$shortcut.TargetPath = $electronExe
$shortcut.Arguments = "`"$mainJs`""
$shortcut.WorkingDirectory = $root
$shortcut.IconLocation = "$iconPath,0"
$shortcut.Description = "VoiceAgent — Full-Duplex Local Voice Assistant"
$shortcut.Save()

Write-Host "DESKTOP_SHORTCUT_CREATED: $lnkPath"
