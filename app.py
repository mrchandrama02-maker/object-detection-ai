import gradio as gr
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2

# Load model
model = YOLO("yolov8n.pt")

def detect_objects(image, conf_threshold):
    results = model(image, conf=conf_threshold)

    output_image = results[0].plot()
    output_image = output_image[:, :, ::-1]

    names = results[0].names
    boxes = results[0].boxes

    detected = {}
    total = 0

    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = names[cls]

        total += 1

        if label in detected:
            detected[label]["count"] += 1
            detected[label]["conf"].append(conf)
        else:
            detected[label] = {"count": 1, "conf": [conf]}

    summary = ""
    for k, v in detected.items():
        avg_conf = sum(v["conf"]) / len(v["conf"])
        summary += f"{k}: {v['count']} (avg: {avg_conf:.2f})\n"

    unique = len(detected)
    top_object = max(detected, key=lambda x: detected[x]["count"]) if detected else "None"

    return (
        Image.fromarray(output_image),
        summary,
        total,
        unique,
        top_object
    )


# 🎨 Custom CSS (Premium Look)
css = """
body {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}
.gradio-container {
    font-family: 'Segoe UI', sans-serif;
}
.card {
    background: #0f172a;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 0px 20px rgba(0,255,255,0.1);
}
"""

with gr.Blocks(css=css) as demo:

    gr.Markdown("# 🔥 Object Detection AI")
    gr.Markdown("Upload image → detect objects → get insights")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="📤 Upload Image")
            conf_slider = gr.Slider(0.1, 1.0, value=0.25, label="🎯 Confidence Threshold")

            btn = gr.Button("🚀 Detect Objects")

        with gr.Column():
            image_output = gr.Image(label="🧠 Detection Result")
            summary_output = gr.Textbox(label="📊 Summary")

    # 💎 Cards Section
    with gr.Row():
        total_card = gr.Number(label="🔢 Total Objects")
        unique_card = gr.Number(label="📦 Unique Objects")
        top_card = gr.Textbox(label="🏆 Top Object")

    # 📥 Download
    download_btn = gr.File(label="⬇️ Download Result")

    def download_image(img):
        path = "output.png"
        img.save(path)
        return path

    btn.click(
        fn=detect_objects,
        inputs=[image_input, conf_slider],
        outputs=[image_output, summary_output, total_card, unique_card, top_card]
    )

    image_output.change(
        fn=download_image,
        inputs=image_output,
        outputs=download_btn
    )

demo.launch()