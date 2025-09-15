import streamlit as st
from pathlib import Path
import tempfile, time, io, csv
from typing import List, Dict, Tuple, Any
import numpy as np
import cv2
from datetime import datetime
import pandas as pd

# Try to import ultralytics YOLO. If missing, we'll fallback to mock
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception as e:
    ULTRALYTICS_AVAILABLE = False
    # We will log the exception to the app logs (shown in UI)
    ULTRALYTICS_IMPORT_ERROR = str(e)

# ------- Helper types -------
Detection = Dict[str, Any]  # {label, conf, box: (x1,y1,x2,y2)}

# ------- PPE mapping / business logic -------
PPE_LABEL_KEYWORDS = {
    "hard hat": ["helmet", "hardhat", "hard hat", "safety helmet"],
    "safety vest": ["vest", "safety vest", "jacket", "reflective vest"],
    "safety gloves": ["glove", "gloves", "safety glove"],
    "safety goggles": ["goggle", "goggles", "eyewear", "glasses", "safety glasses"],
    # keep 'person' to allow person detection
    "person": ["person", "worker", "human"]
}

REQUIRED_PPE = ["hard hat", "safety vest", "safety gloves", "safety goggles"]

def map_label_to_ppe(label: str) -> str:
    lab = label.lower()
    for ppe, keys in PPE_LABEL_KEYWORDS.items():
        for k in keys:
            if k in lab:
                return ppe
    return label  # default: keep original

def compute_compliance(detected_ppe_flags: Dict[str, bool]) -> Tuple[int,int,float]:
    total = len(REQUIRED_PPE)
    detected = sum(1 for p in REQUIRED_PPE if detected_ppe_flags.get(p, False))
    pct = round((detected/total)*100, 1) if total>0 else 0.0
    return detected, total, pct

# ------- Model loader with caching -------
@st.cache_resource(show_spinner=False)
def load_model(weights_path: str = None, device: str = 'cpu'):
    """
    Load YOLO model. If weights_path is None -> loads ultralytics prebuilt 'yolov8n.pt'
    Device: 'cpu' or 'cuda' (if available)
    """
    if not ULTRALYTICS_AVAILABLE:
        raise RuntimeError(f"ultralytics not available: {ULTRALYTICS_IMPORT_ERROR}")
    # If user provided a file path to weights, use that; else use 'yolov8n.pt' (must be available)
    if weights_path:
        model = YOLO(weights_path)
    else:
        # default small model for quick demos; ultralytics will download if needed
        model = YOLO("yolov8n.pt")
    # set device - ultralytics handles device assignment internally but ensure it's respected in predict calls
    model.params['device'] = device
    return model

# ------- Utilities for drawing -------
def draw_detections(frame: np.ndarray, detections: List[Detection], line_thickness=2) -> np.ndarray:
    h, w = frame.shape[:2]
    for d in detections:
        x1, y1, x2, y2 = map(int, d["box"])
        label = d["label"]
        conf = d["conf"]
        # color depending on PPE presence
        color = (16, 185, 129) if label != "missing" else (234, 88, 12)  # green or orange
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)
        txt = f"{label} {conf:.2f}"
        # put text with background
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, txt, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return frame

# ------- Inference pipeline -------
def run_yolo_inference(model, image: np.ndarray, conf_thres: float = 0.25, iou_thres: float = 0.45, max_det: int = 100) -> List[Detection]:
    """
    image: BGR numpy array (OpenCV)
    returns list of detections with mapped PPE labels
    """
    # ultralytics expects either path or RGB ndarray with HWC
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # YOLO predict supports many args; using ultralytics YOLO.predict
    results = model.predict(source=img_rgb, conf=conf_thres, iou=iou_thres, max_det=max_det)  # returns Results object or list
    detections: List[Detection] = []
    # results may be list of one Results object (per image)
    if isinstance(results, list):
        r = results[0]
    else:
        r = results
    # r.boxes contains xyxy tensors and r.boxes.cls, r.boxes.conf
    try:
        boxes = r.boxes.xyxy.cpu().numpy()  # (n,4)
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        # attempt to get class names from model.names
        names = {int(k): v for k, v in model.model.names.items()} if hasattr(model, "model") and hasattr(model.model, "names") else {}
        for (x1,y1,x2,y2), s, c in zip(boxes, scores, classes):
            original_label = names.get(c, str(c))
            mapped = map_label_to_ppe(original_label)
            detections.append({"label": mapped, "conf": float(s), "box": (x1,y1,x2,y2)})
    except Exception:
        # fallback tolerant parsing if API differs
        try:
            for det in r.boxes:
                xyxy = det.xyxy[0].cpu().numpy()
                s = float(det.conf[0].cpu().numpy())
                c = int(det.cls[0].cpu().numpy())
                original_label = model.model.names.get(c, str(c)) if hasattr(model, "model") else str(c)
                mapped = map_label_to_ppe(original_label)
                detections.append({"label": mapped, "conf": s, "box": (xyxy[0],xyxy[1],xyxy[2],xyxy[3])})
        except Exception:
            # If parsing failed entirely, return empty list
            return []
    return detections

# ------- Mock inference (demo) -------
def mock_inference(image: np.ndarray) -> List[Detection]:
    h, w = image.shape[:2]
    # produce 1-3 mock detections randomly for demo
    rng = np.random.default_rng(int(time.time() * 1000) % 2**32)
    detections = []
    if rng.random() > 0.3:
        # person
        detections.append({"label":"person","conf":0.99,"box":(int(w*0.1), int(h*0.05), int(w*0.9), int(h*0.95))})
    if rng.random() > 0.6:
        detections.append({"label":"hard hat","conf":0.88,"box":(int(w*0.4), int(h*0.05), int(w*0.6), int(h*0.18))})
    if rng.random() > 0.7:
        detections.append({"label":"safety vest","conf":0.92,"box":(int(w*0.35), int(h*0.18), int(w*0.65), int(h*0.5))})
    return detections

# ------- Streamlit App UI -------
st.set_page_config(page_title="PPE Detective (YOLOv8)", layout="wide", initial_sidebar_state="expanded")
st.header("🛡️ PPE Detective — Streamlit YOLOv8 PPE Monitor")

# Sidebar: options and model management
with st.sidebar:
    st.subheader("Model / Runtime")
    device = st.selectbox("Device", options=["cpu", "cuda"], index=0, help="Choose 'cuda' if you have GPU support (and ultralytics + torch configured).")
    custom_weights = st.file_uploader("Upload custom YOLO weights (.pt)", type=["pt"], help="Optional: upload custom trained weights for PPE detection.")
    use_pretrained = st.checkbox("Use ultralytics yolov8n (fast demo)", value=True, help="If unchecked and no custom weights supplied, app will run in demo/mock mode.")
    st.markdown("---")
    st.subheader("Detection Settings")
    conf_threshold = st.slider("Confidence threshold", 0.05, 0.99, 0.35, 0.01)
    iou_threshold = st.slider("NMS IoU threshold", 0.1, 0.9, 0.45, 0.01)
    max_detections = st.slider("Max detections per frame", 10, 500, 100, step=10)
    st.markdown("---")
    st.subheader("Streaming / Capture")
    frame_rate = st.slider("Processing interval (seconds per frame)", 0.1, 5.0, 0.5, 0.1)
    show_overlay = st.checkbox("Show detection overlay on frames", value=True)
    record_history = st.checkbox("Record detection history", value=True)
    st.markdown("---")
    st.caption("Tip: for real-time, give model time to warm up after loading.\nIf ultralytics isn't installed the app falls back to demo mode.")

# top-level status
status_box = st.empty()

# Model load / fallback
model = None
model_loaded = False
model_text = ""
if custom_weights is not None:
    # write to temp file
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        tmp.write(custom_weights.getvalue())
        tmp.flush()
        tmp.close()
        weights_path = tmp.name
    except Exception as e:
        st.sidebar.error(f"Failed to save uploaded weights: {e}")
        weights_path = None
else:
    weights_path = None

# Attempt to load model (but don't crash the app)
if (ULTRALYTICS_AVAILABLE and (weights_path or use_pretrained)):
    try:
        with st.spinner("Loading YOLOv8 model..."):
            model = load_model(weights_path=weights_path if weights_path else None, device=device)
        model_loaded = True
        model_text = "YOLOv8 loaded"
        status_box.success("Model loaded successfully.")
    except Exception as e:
        model_loaded = False
        model_text = f"Model load failed: {e}"
        status_box.error("Model could not be loaded; running in DEMO mode.")
else:
    model_loaded = False
    model_text = "Ultralytics not installed or no model selected."
    status_box.warning("Ultralytics not available or no model selected; running in DEMO mode.")

st.sidebar.write("Model status:")
st.sidebar.write(model_text)

# App columns: left = feed, right = controls & results
col1, col2 = st.columns([2,1])

# Detection history dataframe
if "detections_log" not in st.session_state:
    st.session_state["detections_log"] = []  # list of dicts: timestamp, image_id, detected_ppe list, compliance_pct

def append_log(entry: Dict):
    if record_history:
        st.session_state["detections_log"].append(entry)

# Utility to run inference with safe fallback
def infer_image(img_bgr: np.ndarray) -> List[Detection]:
    if model_loaded and model is not None:
        try:
            dets = run_yolo_inference(model, img_bgr, conf_thres=conf_threshold, iou_thres=iou_threshold, max_det=max_detections)
            if dets is None:
                return []
            return dets
        except Exception as e:
            # If inference fails, show error in status and fallback to mock
            status_box.error(f"Inference error: {e} — switching to demo inference for this frame.")
            return mock_inference(img_bgr)
    else:
        return mock_inference(img_bgr)

# Controls: choose input mode
with col2:
    st.subheader("Controls & Options")
    input_mode = st.radio("Input Mode", options=["Webcam (Live)", "Upload Image", "Upload Video"], index=0)
    run_button = st.button("Start / Restart Detection")
    stop_button = st.button("Stop Detection")
    clear_history = st.button("Clear Detection History")

    if clear_history:
        st.session_state["detections_log"] = []
        st.success("History cleared.")

# Preview canvas/area
with col1:
    st.subheader("Camera / Feed")
    display_area = st.empty()
    key_frame = None

# Small analytics in right column
with col2:
    st.subheader("Recent Detections")
    if len(st.session_state["detections_log"]) == 0:
        st.info("No detections logged yet.")
    else:
        df = pd.DataFrame(st.session_state["detections_log"])
        # show last 8 rows
        st.dataframe(df.tail(8), height=300)
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV log", data=csv_bytes, file_name=f"ppe_detections_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

# Main logic for each input mode
running = False
if run_button:
    running = True
    st.session_state["app_running"] = True
if stop_button:
    st.session_state["app_running"] = False
    running = False

if "app_running" not in st.session_state:
    st.session_state["app_running"] = False

# Always follow the session state unless Start pressed
running = st.session_state["app_running"] or running

if input_mode == "Upload Image":
    uploaded = st.file_uploader("Choose an image", type=["png","jpg","jpeg","bmp"], accept_multiple_files=False, key="img_upload")
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.getvalue(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Could not decode image.")
        else:
            detections = infer_image(img)
            # compute PPE flags
            ppe_flags = {k: False for k in REQUIRED_PPE}
            for d in detections:
                lbl = d["label"]
                if lbl in ppe_flags:
                    ppe_flags[lbl] = ppe_flags.get(lbl, False) or True
            detected, total, pct = compute_compliance(ppe_flags)
            # draw and display
            if show_overlay:
                out = draw_detections(img.copy(), detections)
            else:
                out = img
            out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            st.image(out_rgb, use_column_width=True)
            st.metric("Compliance", f"{detected}/{total} ({pct}%)")
            # record
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "input_type": "image",
                "filename": getattr(uploaded, "name", "uploaded_image"),
                "detected_count": len(detections),
                "ppe_flags": ppe_flags,
                "compliance_pct": pct
            }
            append_log(entry)

elif input_mode == "Upload Video":
    uploaded = st.file_uploader("Choose a video (mp4,mov)", type=["mp4","mov","avi","mkv"], accept_multiple_files=False, key="vid_upload")
    if uploaded is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.flush()
        cap = cv2.VideoCapture(tfile.name)
        stframe = display_area.empty()
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_interval = max(1, int((frame_rate * fps)))
        frame_no = 0
        stop_requested = False
        while cap.isOpened() and (not stop_requested):
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1
            if frame_no % frame_interval != 0:
                continue
            detections = infer_image(frame)
            ppe_flags = {k: False for k in REQUIRED_PPE}
            for d in detections:
                if d["label"] in ppe_flags:
                    ppe_flags[d["label"]] = True
            detected, total, pct = compute_compliance(ppe_flags)
            if show_overlay:
                out = draw_detections(frame.copy(), detections)
            else:
                out = frame
            out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            stframe.image(out_rgb, channels="RGB", use_column_width=True)
            st.sidebar.metric("Last compliance", f"{detected}/{total} ({pct}%)")
            entry = {"timestamp": datetime.utcnow().isoformat(), "input_type":"video_frame", "frame_no": frame_no, "detected_count": len(detections), "compliance_pct": pct}
            append_log(entry)
            # allow user to stop
            if st.button("Stop Video Processing"):
                stop_requested = True
                break
        cap.release()
        st.success("Video processing finished.")

elif input_mode == "Webcam (Live)":
    # We'll do a simple loop capturing frames via OpenCV. This works when streamlit is run locally.
    st.write("Using OpenCV to access webcam. If running on a remote server, webcam may be unavailable.")
    start_camera = st.button("Start Webcam") if not running else st.button("Stop Webcam")
    if start_camera and not running:
        st.session_state["app_running"] = True
        running = True
    elif start_camera and running:
        st.session_state["app_running"] = False
        running = False

    if running:
        # OpenCV capture
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open webcam. If you're on a remote server, webcam access may be blocked.")
        else:
            stframe = display_area.empty()
            last_process = 0.0
            try:
                while st.session_state.get("app_running", False):
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Empty frame from webcam.")
                        break
                    now = time.time()
                    if now - last_process >= frame_rate:
                        last_process = now
                        detections = infer_image(frame)
                        ppe_flags = {k: False for k in REQUIRED_PPE}
                        for d in detections:
                            if d["label"] in ppe_flags:
                                ppe_flags[d["label"]] = True
                        detected, total, pct = compute_compliance(ppe_flags)
                        if show_overlay:
                            out = draw_detections(frame.copy(), detections)
                        else:
                            out = frame
                        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                        stframe.image(out_rgb, channels="RGB", use_column_width=True)
                        # status on sidebar
                        st.sidebar.metric("Live compliance", f"{detected}/{total} ({pct}%)")
                        entry = {"timestamp": datetime.utcnow().isoformat(), "input_type":"webcam_frame", "detected_count": len(detections), "compliance_pct": pct}
                        append_log(entry)
                    else:
                        # still show raw frames for smoothness
                        tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        stframe.image(tmp, channels="RGB", use_column_width=True)
                    # small yield to allow UI updates
                    time.sleep(0.01)
            except Exception as e:
                st.error(f"Webcam loop ended: {e}")
            finally:
                cap.release()
                st.session_state["app_running"] = False
                st.success("Webcam stopped.")

# Final: show summary counts and simple analytics
with st.expander("App Summary & Diagnostics", expanded=False):
    st.write("Model loaded:", model_loaded)
    if not ULTRALYTICS_AVAILABLE:
        st.warning("ultralytics package not installed — running DEMO mode. Install with `pip install ultralytics` for full functionality.")
    if ULTRALYTICS_AVAILABLE and model_loaded:
        try:
            st.write("Model names / classes:")
            names = model.model.names if hasattr(model, "model") and hasattr(model.model, "names") else {}
            st.write(names)
        except Exception:
            st.write("Unable to show model names.")
    st.write("Total logged detections:", len(st.session_state["detections_log"]))
    if len(st.session_state["detections_log"])>0:
        df = pd.DataFrame(st.session_state["detections_log"])
        st.write("Average compliance:", df["compliance_pct"].mean())

st.caption("PPE Detective — built for final projects. Modify labels, mappings and add fine-tuned custom weights for production-grade results.")
