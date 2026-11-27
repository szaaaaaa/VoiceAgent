<#
.SYNOPSIS
  One-click launcher for the VoiceAgent Windows Desktop Application.
.EXAMPLE
  .\Start-VoiceAgent.ps1           # Fast startup (Silero VAD + CausalTurnGRU + Windows Neural Voice)
  .\Start-VoiceAgent.ps1 -GpuModels # Load full CUDA Whisper + Qwen3-TTS-12Hz-0.6B on RTX 5070
#>
param(
  [switch]$GpuModels
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:HF_HOME = Join-Path $PSScriptRoot "models\hf_cache"
$env:VOICE_AGENT_LOAD_HEAVY_MODELS = if ($GpuModels) { "1" } else { "0" }

if (-not (Test-Path "apps\desktop\dist\renderer\index.html")) {
  Write-Host "[VoiceAgent] Building desktop renderer..." -ForegroundColor Cyan
  npm run build:gui
}

Write-Host "[VoiceAgent] Launching Electron desktop app (Heavy GPU Models: $($GpuModels.IsPresent))..." -ForegroundColor Green
npx electron apps/desktop/electron-main.cjs
