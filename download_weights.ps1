# ---------------------------------------------------------------------------
# Downloads the files that are too large for git:
#   - YOLOv4 / YOLOv4-tiny weights  -> weights/
#   - cloudflared.exe (Cloudflare tunnel client)
#
# Already-existing files are skipped, so it is safe to run again.
#
# Usage:  .\download_weights.ps1
# ---------------------------------------------------------------------------
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$files = [ordered]@{
    "weights/yolov4-tiny.weights" = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
    "weights/yolov4.weights"      = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4.weights"
    "cloudflared.exe"             = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
}

foreach ($file in $files.Keys) {
    if (Test-Path $file) {
        Write-Host "exists  $file - skipping" -ForegroundColor DarkGray
    } else {
        Write-Host "downloading $file ..." -ForegroundColor Cyan
        & curl.exe -L --retry 3 --fail -o $file $files[$file]
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $file" }
    }
}

Write-Host ""
Write-Host "All files ready." -ForegroundColor Green