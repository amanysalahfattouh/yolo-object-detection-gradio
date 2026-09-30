"""Smoke-test: call the running Gradio app through its web API.

Usage:
    .\\.venv\\Scripts\\python.exe check_app.py                        # localhost
    .\\.venv\\Scripts\\python.exe check_app.py https://xxx.trycloudflare.com
"""
import sys

from gradio_client import Client, handle_file

# Windows consoles default to cp1252; make emoji in the summary print safely.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7862"
print(f"Connecting to {url} ...")
client = Client(url, verbose=False)
result = client.predict(
    handle_file("samples/dog.jpg"),   # input image
    "YOLOv4-tiny (fast)",             # model
    0.5,                              # confidence threshold
    0.45,                             # NMS threshold
    api_name="/run_detection",
)
image, summary, table = result
print("annotated image :", image)
print("summary         :", summary.replace("\n", " | "))
print("detections      :", table)
print("SMOKE TEST PASSED")