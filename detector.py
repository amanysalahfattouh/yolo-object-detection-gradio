"""
YOLOv4 object detector built on OpenCV's DNN module.

Runs the standard YOLOv4 / YOLOv4-tiny darknet models (80 COCO classes) without
TensorFlow, so it works on modern Python (3.11+) and Windows out of the box.

Run a quick self-test from the command line:

    python detector.py samples/dog.jpg
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

COCO_NAMES = Path(__file__).resolve().parent / "coco.names"


def _color_for_class(class_id: int) -> tuple:
    """Deterministic, well-separated BGR colour per class id."""
    palette = (
        (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
        (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
        (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
        (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
        (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
    )
    return palette[class_id % len(palette)]


def _overlaps(a: tuple, b: tuple, pad: int = 3) -> bool:
    """True if rectangles a=(x1,y1,x2,y2) and b=(x1,y1,x2,y2) overlap (with padding)."""
    return not (a[2] + pad < b[0] or b[2] + pad < a[0]
                or a[3] + pad < b[1] or b[3] + pad < a[1])


class YOLOv4Detector:
    """Loads a YOLOv4 / YOLOv4-tiny darknet model with cv2.dnn and runs inference."""

    def __init__(self, cfg_path, weights_path, names_path=COCO_NAMES, input_size=(416, 416)):
        self.net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.output_names = self.net.getUnconnectedOutLayersNames()
        self.input_size = tuple(input_size)
        with open(names_path, encoding="utf-8-sig") as f:
            self.classes = [line.strip() for line in f if line.strip()]

    # ------------------------------------------------------------------ #
    def _letterbox(self, image):
        """Resize image keeping aspect ratio (letterbox padding)."""
        h, w = image.shape[:2]
        tw, th = self.input_size
        r = min(tw / w, th / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dw, dh = (tw - nw) / 2, (th - nh) / 2
        left, top = int(round(dw - 0.1)), int(round(dh - 0.1))
        right, bottom = int(round(dw + 0.1)), int(round(dh + 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        return padded, r, left, top

    # ------------------------------------------------------------------ #
    def detect(self, image_bgr, conf_threshold=0.5, nms_threshold=0.4):
        """Return a list of detections sorted by confidence (highest first).

        Each detection: {"class_id", "class_name", "confidence", "box": [x1,y1,x2,y2]}
        """
        h, w = image_bgr.shape[:2]
        padded, r, pad_x, pad_y = self._letterbox(image_bgr)
        blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, self.input_size, swapRB=True, crop=False)

        self.net.setInput(blob)
        outputs = self.net.forward(self.output_names)

        tw, th = self.input_size
        boxes, scores, class_ids = [], [], []
        for output in outputs:
            for det in output:
                objectness = float(det[4])
                if objectness < 0.05:  # cheap pre-filter
                    continue
                class_scores = det[5:]
                cls_id = int(np.argmax(class_scores))
                score = objectness * float(class_scores[cls_id])
                if score < conf_threshold:
                    continue

                cx, cy = float(det[0]) * tw, float(det[1]) * th
                bw, bh = float(det[2]) * tw, float(det[3]) * th
                x = (cx - bw / 2 - pad_x) / r
                y = (cy - bh / 2 - pad_y) / r
                bw, bh = bw / r, bh / r

                x1 = max(0, min(int(round(x)), w - 1))
                y1 = max(0, min(int(round(y)), h - 1))
                x2 = max(0, min(int(round(x + bw)), w - 1))
                y2 = max(0, min(int(round(y + bh)), h - 1))
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(score)
                class_ids.append(cls_id)

        detections = []
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
            for i in np.array(indices).flatten():
                x, y, bw, bh = boxes[i]
                detections.append({
                    "class_id": class_ids[i],
                    "class_name": self.classes[class_ids[i]],
                    "confidence": round(float(scores[i]), 4),
                    "box": [int(x), int(y), int(x + bw), int(y + bh)],
                })
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    # ------------------------------------------------------------------ #
    def count_objects(self, detections):
        """Per-class object counts, highest first."""
        counts = Counter(d["class_name"] for d in detections)
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    # ------------------------------------------------------------------ #
    def draw(self, image_bgr, detections):
        """Return a copy of the image with modern-looking boxes and labels."""
        img = image_bgr.copy()
        h, w = img.shape[:2]
        scale = max(0.5, min(1.0, w / 1200))
        thickness = max(2, int(round(w / 700)))

        # soft translucent fill for every detection
        if detections:
            overlay = img.copy()
            for det in detections:
                x1, y1, x2, y2 = det["box"]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), _color_for_class(det["class_id"]), -1)
            img = cv2.addWeighted(overlay, 0.10, img, 0.90, 0)

        placed_chips = []
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            colour = _color_for_class(det["class_id"])

            # thin rectangle + thicker corner accents
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, max(1, thickness - 1))
            clen = max(14, int(0.16 * min(x2 - x1, y2 - y1)))
            acc = thickness + 2
            for px, py, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                                   (x1, y2, 1, -1), (x2, y2, -1, -1)):
                cv2.line(img, (px, py), (px + dx * clen, py), colour, acc, cv2.LINE_AA)
                cv2.line(img, (px, py), (px, py + dy * clen), colour, acc, cv2.LINE_AA)

            # padded label chip above (or below) the box
            label = f'{det["class_name"]} {det["confidence"]:.2f}'
            (lw, lh), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            pad = max(4, int(6 * scale))
            chip_h = lh + base + 2 * pad
            chip_w = lw + 2 * pad
            cx1 = min(max(0, x1), max(0, w - chip_w - 1))
            if y1 - chip_h - 2 > 0:
                cy1 = y1 - chip_h - 2
            else:
                cy1 = max(0, min(y2 + 2, h - chip_h - 1))
            # nudge the chip down while it collides with one already placed
            tries = 0
            while tries < 6 and any(_overlaps((cx1, cy1, cx1 + chip_w, cy1 + chip_h), r)
                                    for r in placed_chips):
                cy1 += chip_h + 3
                tries += 1
                if cy1 + chip_h > h - 1:
                    cy1 = max(0, min(y1 + 3, h - chip_h - 1))
                    break
            placed_chips.append((cx1, cy1, cx1 + chip_w, cy1 + chip_h))
            cv2.rectangle(img, (cx1, cy1), (cx1 + chip_w, cy1 + chip_h), colour, -1, cv2.LINE_AA)
            cv2.putText(img, label, (cx1 + pad, cy1 + pad + lh), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)
        return img


# ---------------------------------------------------------------------- #
# Small CLI self-test:  python detector.py samples/dog.jpg
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    if len(sys.argv) < 2:
        print("Usage: python detector.py <image.jpg> [weights=yolov4-tiny]")
        sys.exit(0)
    image_path = Path(sys.argv[1])
    which = sys.argv[2] if len(sys.argv) > 2 else "yolov4-tiny"
    detector = YOLOv4Detector(root / f"weights/{which}.cfg", root / f"weights/{which}.weights")
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")
    detections = detector.detect(image)
    print(f"Model: {which} | Image: {image_path.name} ({image.shape[1]}x{image.shape[0]})")
    print(f"Detected {len(detections)} object(s): {detector.count_objects(detections)}")
    out = root / f"static/output_{image_path.stem}_{which}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), detector.draw(image, detections))
    print(f"Annotated image saved to: {out}")