import streamlit as st
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Object Detection AI", layout="wide")

# ---------------- LOAD CSS ----------------
def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# NAVBAR
st.markdown("""
<div class="navbar">
    <div class="logo">🔥 Object Detection AI</div>
    <div class="menu">
        <span>Home</span>
        <span>Features</span>
        <span>About</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ANIMATED BACKGROUND
st.markdown('<div class="bg-animation"></div>', unsafe_allow_html=True)

# ---------------- MODEL ----------------
model = YOLO("yolov8n.pt")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙ Settings")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# ---------------- HEADER ----------------
st.markdown("<h1>🔥 Object Detection AI</h1>", unsafe_allow_html=True)
st.caption("AI Powered Object Detection using YOLOv8")
st.markdown("<br>", unsafe_allow_html=True)

# ---------------- MAIN ----------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # ORIGINAL IMAGE
    with col1:
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    # DETECTION WITH LOADING
    with st.spinner("🔍 Detecting objects..."):
        results = model(image, conf=conf_threshold)

    result = results[0]

    # RESULT IMAGE
    with col2:
        st.image(result.plot(), caption="🎯 Detection Result", use_container_width=True)

    # DOWNLOAD BUTTON
    st.markdown("### ⬇ Download Result")

    img_bytes = Image.fromarray(result.plot())
    buf = io.BytesIO()
    img_bytes.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button("⬇ Download Image", byte_im, file_name="result.png")

    # ---------------- DATA ----------------
    boxes = result.boxes
    names = result.names

    labels = []
    confidences = []

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        labels.append(names[cls_id])
        confidences.append(round(conf * 100, 2))

    df = pd.DataFrame({
        "Object": labels,
        "Confidence (%)": confidences
    })

    # GROUP DATA
    df_grouped = df.groupby("Object").agg({
        "Confidence (%)": "mean",
        "Object": "count"
    }).rename(columns={"Object": "Count"}).reset_index()

    # SORT
    df_grouped = df_grouped.sort_values(by="Count", ascending=False)

    # ---------------- RESULTS ----------------
    st.markdown("## 📊 Results")

    st.subheader("📋 Detection Table")
    st.dataframe(df_grouped, use_container_width=True)

    # DASHBOARD
    total_objects = int(df_grouped["Count"].sum())
    unique_objects = int(df_grouped["Object"].nunique())

    max_count = df_grouped["Count"].max()
    top_objects = df_grouped[df_grouped["Count"] == max_count]["Object"].tolist()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="card">
            <h3>Total Objects</h3>
            <h1>{total_objects}</h1>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <h3>Unique Objects</h3>
            <h1>{unique_objects}</h1>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <h3>Top Object</h3>
            <h1>{", ".join(top_objects)}</h1>
        </div>
        """, unsafe_allow_html=True)

    # BADGES (NO DUPLICATES)
    st.markdown("## 🎯 Detected Objects")

    for i in range(len(df_grouped)):
        obj = df_grouped["Object"].iloc[i]
        conf = round(df_grouped["Confidence (%)"].iloc[i], 2)
        count = int(df_grouped["Count"].iloc[i])

        st.markdown(
            f'<span class="badge">{obj} ({count}) - {conf}%</span>',
            unsafe_allow_html=True
        )

    # SUMMARY
    st.markdown("## 📌 Summary")

    st.write(f"🔥 Total Objects Detected: {total_objects}")
    st.write(f"🏆 Top Object(s): {', '.join(top_objects)}")
    st.write(f"📦 Unique Objects: {unique_objects}")

    high_conf = (df["Confidence (%)"] > 70).sum()
    st.write(f"⭐ High Confidence (>70%): {high_conf}")

else:
    st.info("👈 Upload an image to start detection")