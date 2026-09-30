"""
Gradio web app: Real-Time Object Detection with YOLOv4.

YOLOv4 (darknet weights) runs through OpenCV's DNN module, so no TensorFlow
is needed and everything installs cleanly on modern Python / Windows.

Run locally:       python app.py
Public via tunnel: see start.ps1 / README.md (Cloudflare quick tunnel)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import cv2
import gradio as gr

from detector import YOLOv4Detector, _color_for_class

ROOT = Path(__file__).resolve().parent

MODELS = {
    "YOLOv4-tiny (fast)": ("weights/yolov4-tiny.cfg", "weights/yolov4-tiny.weights"),
    "YOLOv4 (accurate, slower on CPU)": ("weights/yolov4.cfg", "weights/yolov4.weights"),
}

_detectors: dict[str, YOLOv4Detector] = {}


def get_detector(model_name: str) -> YOLOv4Detector:
    """Load models lazily and keep them cached in memory."""
    if model_name not in _detectors:
        cfg, weights = MODELS[model_name]
        _detectors[model_name] = YOLOv4Detector(ROOT / cfg, ROOT / weights, ROOT / "coco.names")
    return _detectors[model_name]


def _css_hex(bgr: tuple) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def render_stats(detections, detector: YOLOv4Detector, model_name: str,
                 elapsed_ms: float, size: tuple) -> str:
    """HTML statistics card: total count, one chip per class, run metadata."""
    if not detections:
        return PLACEHOLDER_EMPTY
    chips = ""
    for name, count in detector.count_objects(detections).items():
        colour = _css_hex(_color_for_class(detector.classes.index(name)))
        chips += (f'<span class="chip"><span class="dot" style="background:{colour}"></span>'
                  f'{name}<span class="chip-count">×{count}</span></span>')
    short_model = model_name.split(" (")[0]
    return (
        '<div class="stats">'
        f'<div class="stats-total"><span class="stats-number">{len(detections)}</span>'
        '<span class="stats-label">object(s)<br/>detected</span></div>'
        f'<div class="chips">{chips}</div>'
        '<div class="run-meta">'
        f'<span class="meta-pill">⚡ {short_model}</span>'
        f'<span class="meta-pill">{elapsed_ms:.0f} ms</span>'
        f'<span class="meta-pill">{size[0]}×{size[1]} px</span>'
        '</div>'
        '</div>'
    )


def run_detection(image, model_name, conf_threshold, nms_threshold):
    if image is None:
        return None, PLACEHOLDER_EMPTY, None

    start = time.perf_counter()
    detector = get_detector(model_name)
    h, w = image.shape[:2]
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    detections = detector.detect(image_bgr, conf_threshold=float(conf_threshold),
                                 nms_threshold=float(nms_threshold))
    annotated = cv2.cvtColor(detector.draw(image_bgr, detections), cv2.COLOR_BGR2RGB)
    elapsed_ms = (time.perf_counter() - start) * 1000

    stats = render_stats(detections, detector, model_name, elapsed_ms, (w, h))
    table = [[d["class_name"], f'{d["confidence"]:.2f}', *d["box"]] for d in detections]
    return annotated, stats, table


# --------------------------------------------------------------------------- #
#  UI
# --------------------------------------------------------------------------- #
HERO = """
<div class="hero">
  <span class="hero-badge">YOLOv4 · OpenCV DNN · Gradio</span>
  <h1>🎯 Real-Time Object Detection</h1>
  <div class="hero-chips">
    <span class="hero-chip">📷 Upload or webcam</span>
    <span class="hero-chip">⚡ YOLOv4-tiny &amp; YOLOv4</span>
    <span class="hero-chip">📊 Live counts &amp; boxes</span>
    <span class="hero-chip">🌐 One public link</span>
  </div>
</div>
"""

PLACEHOLDER_EMPTY = (
    '<div class="stats-empty">Upload an image or take a webcam photo, then press '
    '<b>🚀 Detect objects</b> — or click one of the sample images below.</div>'
)

INPUT_HEADER = """
<div class="card-head">
  <span class="card-ico">📤</span>
  <div>
    <div class="card-title">Input</div>
    <div class="card-sub">Upload a photo or take one with your webcam — detection starts automatically</div>
  </div>
</div>
"""

RESULT_HEADER = """
<div class="card-head">
  <span class="card-ico">🧠</span>
  <div>
    <div class="card-title">Detection result</div>
    <div class="card-sub">Annotated image, object counts and exact coordinates</div>
  </div>
</div>
"""

CSS = (ROOT / "style.css").read_text(encoding="utf-8") if (ROOT / "style.css").exists() else ""

with gr.Blocks(title="YOLOv4 Real-Time Object Detection") as demo:
    gr.HTML(HERO)

    with gr.Row(equal_height=False):
        # ---------------- input panel ---------------- #
        with gr.Column(scale=1, min_width=340, elem_classes=["panel-card"]):
            gr.HTML(INPUT_HEADER)
            input_image = gr.Image(
                type="numpy", label="Image (upload or webcam)",
                sources=["upload", "webcam"], streaming=False, height=380,
                elem_classes=["media-frame"],
            )
            model_dropdown = gr.Dropdown(
                choices=list(MODELS.keys()), value="YOLOv4-tiny (fast)", label="Model",
            )
            with gr.Row():
                conf_slider = gr.Slider(0.05, 0.95, value=0.5, step=0.05,
                                        label="Confidence threshold")
                nms_slider = gr.Slider(0.1, 0.9, value=0.45, step=0.05,
                                       label="NMS threshold")
            with gr.Row():
                detect_btn = gr.Button("🚀 Detect objects", variant="primary", size="lg",
                                       scale=3, elem_classes=["detect-btn"])
                clear_btn = gr.Button("🗑️ Clear", size="lg", scale=1,
                                      elem_classes=["clear-btn"])

        # ---------------- result panel ---------------- #
        with gr.Column(scale=1, min_width=340, elem_classes=["panel-card"]):
            gr.HTML(RESULT_HEADER)
            output_image = gr.Image(type="numpy", label="Detections", height=380,
                                    elem_classes=["media-frame"])
            stats_html = gr.HTML(PLACEHOLDER_EMPTY)
            table = gr.Dataframe(
                headers=["Class", "Confidence", "x1", "y1", "x2", "y2"],
                label="Detections (class, confidence, bounding box)",
                interactive=False, wrap=True, elem_classes=["results-table"],
            )

    sample_images = sorted(str(p) for p in (ROOT / "samples").glob("*.jpg"))
    if sample_images:
        gr.Examples(examples=[[p] for p in sample_images], inputs=input_image,
                    label="✨ Try a sample image")

    inputs = [input_image, model_dropdown, conf_slider, nms_slider]
    outputs = [output_image, stats_html, table]
    detect_btn.click(run_detection, inputs, outputs)
    input_image.change(run_detection, inputs, outputs)
    clear_btn.click(lambda: (None, PLACEHOLDER_EMPTY, None), None, outputs)

if __name__ == "__main__":
    demo.queue().launch(
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="violet",
            neutral_hue="slate",
            radius_size=gr.themes.sizes.radius_lg,
        ),
        css=CSS,
        server_name="127.0.0.1",
        server_port=int(os.environ.get("GRADIO_PORT", "7862")),
    )