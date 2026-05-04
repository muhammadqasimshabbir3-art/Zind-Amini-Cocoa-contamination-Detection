"""
Zindi Cocoa Contamination Detection - Stable Hugging Face Spaces App
"""

import os
import numpy as np
import gradio as gr
from PIL import Image
from ultralytics import YOLO

# ================= CONFIG =================
MODEL_PATH = "best.pt"
DEVICE = "cpu"
CONF_THRESHOLD_DEFAULT = 0.5

# Load model ONCE (important fix)
model = YOLO(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


# ================= INFERENCE =================
def predict(image, conf):
    if model is None:
        return None, "Model not found (best.pt missing)"

    results = model.predict(image, conf=conf, device=DEVICE, verbose=False)
    res = results[0]

    annotated = res.plot()
    annotated = Image.fromarray(annotated)

    detections = []

    if res.boxes is not None and len(res.boxes) > 0:
        for i in range(len(res.boxes)):
            cls = int(res.boxes.cls[i])
            conf_val = float(res.boxes.conf[i])
            name = res.names.get(cls, str(cls))

            x1, y1, x2, y2 = res.boxes.xyxy[i].tolist()

            detections.append(
                f"{i+1}. {name} | conf: {conf_val:.2f} | box: ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
            )

    text = "No detections found" if not detections else "\n".join(detections)

    return annotated, text


# ================= UI =================
with gr.Blocks(title="Cocoa Disease Detection") as demo:

    gr.Markdown("# 🌱 Cocoa Disease Detection (YOLOv8)")
    gr.Markdown("Upload an image and detect cocoa contamination/disease.")

    with gr.Row():
        with gr.Column():
            img = gr.Image(type="pil", label="Upload Image")
            conf = gr.Slider(0.1, 1.0, value=CONF_THRESHOLD_DEFAULT, label="Confidence")
            btn = gr.Button("Detect", variant="primary")

        with gr.Column():
            out_img = gr.Image(label="Result")
            out_txt = gr.Textbox(label="Detections")

    btn.click(
        fn=predict,
        inputs=[img, conf],
        outputs=[out_img, out_txt]
    )

    # SAFE examples (no cache_examples, no crash)
    gr.Examples(
        examples=[],
        inputs=[img, conf],
    )


# ================= RUN =================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )