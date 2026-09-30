# ---------------------------------------------------------------------------
# Starts the YOLOv4 Gradio app and exposes it publicly with a Cloudflare
# quick tunnel (no Cloudflare account needed).
#
# Usage:  right-click -> "Run with PowerShell"   (or:  .\start.ps1)
# ---------------------------------------------------------------------------
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# 1) Start the Gradio app in its own window
Start-Process -FilePath "$root\.venv\Scripts\python.exe" -ArgumentList "app.py" `
    -WorkingDirectory $root -WindowStyle Normal

# 2) Wait until it answers on http://127.0.0.1:7862
Write-Host "Waiting for the app to start on http://127.0.0.1:7862 ..."
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:7862" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
}
if ($ready) { Write-Host "App is up." -ForegroundColor Green }
else { Write-Warning "App did not answer yet - the tunnel will still connect once it is ready." }

# 3) Start the Cloudflare quick tunnel (prints a public https://*.trycloudflare.com URL)
Write-Host ""
Write-Host "Starting Cloudflare quick tunnel ..." -ForegroundColor Cyan
Write-Host "Copy the https://*.trycloudflare.com URL that appears below." -ForegroundColor Cyan
Write-Host ""
& "$root\cloudflared.exe" tunnel --url http://127.0.0.1:7862 --no-autoupdate