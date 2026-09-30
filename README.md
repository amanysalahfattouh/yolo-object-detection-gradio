# YOLOv4 Real-Time Object Detection — Gradio App

A web app for real-time object detection: **YOLOv4** (80 COCO classes) running through
**OpenCV's DNN module** — no TensorFlow required, so it installs cleanly on modern Python
(3.11+) and Windows — wrapped in a polished **Gradio** web interface and published to the
internet with a **Cloudflare quick tunnel** (`cloudflared`) — no Cloudflare account or
domain needed.

## Features

- Detect objects in any uploaded photo, or take a picture with your webcam
- Two models: `YOLOv4-tiny` (fast) and `YOLOv4` (more accurate, slower on CPU)
- Adjustable confidence / NMS thresholds
- Annotated result image with boxes and labels
- Per-class object counts
- Detection table with class, confidence and bounding-box coordinates
- Polished custom UI: gradient hero header, card layout, coloured class chips, styled table, modern box drawing (soft fill + corner accents)

## Project layout

```
yolo-gradio/
├── app.py               # Gradio web app
├── style.css            # custom UI theme (hero header, cards, chips, table)
├── detector.py          # YOLOv4 (OpenCV DNN) inference + drawing + counting
├── check_app.py         # smoke test (calls the running app's API)
├── coco.names           # 80 COCO class names
├── weights/
│   ├── yolov4-tiny.cfg / .weights   # fast model  (weights via download_weights.ps1)
│   └── yolov4.cfg / .weights        # accurate model
├── samples/             # sample images for the Examples strip
├── download_weights.ps1 # downloads weights + cloudflared (large files)
├── cloudflared.exe      # Cloudflare tunnel client (downloaded)
├── start.ps1            # starts the app + public tunnel in one go
├── .gitignore
└── requirements.txt
```

## Setup

```powershell
git clone https://github.com/amanysalahfattouh/yolo-object-detection-gradio.git
cd yolo-object-detection-gradio

# 1) model weights + cloudflared (too large for git; skips files that exist)
.\download_weights.ps1

# 2) python environment
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run locally

```powershell
.\.venv\Scripts\python.exe app.py
# open http://127.0.0.1:7862
```

## Publish to the internet with Cloudflare (quick tunnel)

```powershell
# with the app already running in another terminal:
.\cloudflared.exe tunnel --url http://127.0.0.1:7862 --no-autoupdate
```

`cloudflared` prints a public URL like `https://something-random.trycloudflare.com`.
Anyone with that link can use the app while your PC is running.

Or simply start both at once:

```powershell
.\start.ps1
```

> ⚠️ Quick tunnels are temporary: the URL changes every time you restart `cloudflared`,
> and the link dies when you close it. For a permanent URL see "Named tunnels" in the
> Cloudflare docs (requires a free Cloudflare account + your own domain).

## Troubleshooting

- **Port already in use** — start with another port: `$env:GRADIO_PORT=7861; .\.venv\Scripts\python.exe app.py`
  (then point the tunnel at `http://127.0.0.1:7861`).
- **First detection is slow** — the model is loaded lazily on the first request.
- **YOLOv4 (full) is slow** — it's CPU inference; use YOLOv4-tiny for a snappy demo.
- **`cloudflared` blocked** — allow it through Windows Firewall if prompted.

## Credits

- YOLOv4 model/weights: [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) (YOLOv4: Optimal Speed and Accuracy of Object Detection)
- Inference: OpenCV DNN · Web UI: [Gradio](https://gradio.app) · Tunnel: [Cloudflare](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)