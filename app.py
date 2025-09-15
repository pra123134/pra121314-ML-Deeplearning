import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image

# Load YOLOv8 model (cached)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Replace with your trained PPE model if available

model = load_model()

# Streamlit page config
st.set_page_config(page_title="PPE Detector", layout="wide")
st.title("🛡️ PPE Detection System")
st.markdown("YOLOv8-powered computer vision system for real-time safety compliance monitoring.")

# Sidebar options
st.sidebar.header("Options")
source = st.sidebar.radio("Choose input source", ["Upload Image", "Webcam"])

# Detection function
def detect_objects(image):
    results = model(image)
    return results

# Image upload mode
if source == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        results = detect_objects(img)

        # Show uploaded image
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Detection results
        st.subheader("Detection Results")
        for r in results:
            for box in r.boxes:
                cls = int(box.cls)
                label = model.names[cls]
                conf = float(box.conf)
                st.write(f"**{label}** - {conf:.2f}")

        # Show annotated detections
        res_plotted = results[0].plot()[:, :, ::-1]
        st.image(res_plotted, caption="Detected PPE", use_column_width=True)

# Webcam mode
elif source == "Webcam":
    st.warning("Click 'Start' below to capture frames from your webcam.")
    run = st.button("Start Webcam")
    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = detect_objects(frame)
            annotated_frame = results[0].plot()
            FRAME_WINDOW.image(annotated_frame, channels="BGR")

        cap.release()
