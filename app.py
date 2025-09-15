import streamlit as st
from ultralytics import YOLO
from PIL import Image

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.set_page_config(page_title="PPE Detector", layout="wide")
st.title("🛡️ PPE Detection System")
st.markdown("YOLOv8-powered PPE detection without OpenCV.")

st.sidebar.header("Options")
source = st.sidebar.radio("Choose input source", ["Upload Image"])

def detect_objects(image):
    results = model(image)
    return results

if source == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        results = detect_objects(img)

        st.image(img, caption="Uploaded Image", use_column_width=True)

        st.subheader("Detection Results")
        for r in results:
            for box in r.boxes:
                cls = int(box.cls)
                label = model.names[cls]
                conf = float(box.conf)
                st.write(f"**{label}** - {conf:.2f}")

        res_plotted = results[0].plot()[:, :, ::-1]
        st.image(res_plotted, caption="Detected PPE", use_column_width=True)
