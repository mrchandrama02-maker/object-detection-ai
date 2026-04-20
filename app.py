import gradio as gr
from ultralytics import YOLO
from PIL import Image

# Load model
model = YOLO("yolov8n.pt")

# Detection function
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
        f"### 🔢 Total Objects: {total}",
        f"### 📦 Unique Objects: {unique}",
        f"### 🏆 Top Object: {top_object}"
    )


# 🎨 Premium CSS
css = """
body {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: white;
}
.gradio-container {
    font-family: 'Segoe UI', sans-serif;
}
"""


with gr.Blocks(css=css) as demo:

    # 🔥 Title
    gr.Markdown("# 🔥 Object Detection AI\n### 🚀 AI Powered Detection with YOLOv8")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="📤 Upload Image")
            conf_slider = gr.Slider(0.1, 1.0, value=0.25, label="🎯 Confidence Threshold")

            btn = gr.Button("🚀 Detect Objects", variant="primary")

        with gr.Column():
            image_output = gr.Image(label="🧠 Detection Result")
            summary_output = gr.Textbox(label="📊 Summary")

    # 💎 Cards
    with gr.Row():
        total_card = gr.Markdown("### 🔢 Total Objects: 0")
        unique_card = gr.Markdown("### 📦 Unique Objects: 0")
        top_card = gr.Markdown("### 🏆 Top Object: None")

    # ⬇️ Download
    download_btn = gr.File(label="⬇️ Download Result")

    def save_image(img):
        path = "output.png"
        img.save(path)
        return path

    btn.click(
        fn=detect_objects,
        inputs=[image_input, conf_slider],
        outputs=[image_output, summary_output, total_card, unique_card, top_card]
    )

    image_output.change(
        fn=save_image,
        inputs=image_output,
        outputs=download_btn
    )

demo.launch()