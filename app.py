import gradio as gr
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load YOLO model
model = YOLO("yolov8n.pt")

# Detection function
def detect_objects(image):
    results = model(image)

    # Plot result image
    output_image = results[0].plot()

    # Convert BGR → RGB
    output_image = output_image[:, :, ::-1]

    # Get detection details
    names = results[0].names
    boxes = results[0].boxes

    detected = {}

    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = names[cls]

        if label in detected:
            detected[label]["count"] += 1
            detected[label]["conf"].append(conf)
        else:
            detected[label] = {"count": 1, "conf": [conf]}

    summary = ""
    for k, v in detected.items():
        avg_conf = sum(v["conf"]) / len(v["conf"])
        summary += f"{k}: {v['count']} (avg conf: {avg_conf:.2f})\n"

    return Image.fromarray(output_image), summary


# Gradio UI
interface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=[
        gr.Image(label="Detection Result"),
        gr.Textbox(label="Summary")
    ],
    title="🔥 Object Detection AI",
    description="Upload image and detect objects using YOLOv8"
)

# Launch
interface.launch()