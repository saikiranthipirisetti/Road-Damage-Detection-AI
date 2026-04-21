
import streamlit as st
import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="🚗 Real-Time Road Damage Detection", layout="wide")

st.title("🧠 Real-Time Crack, Pothole & Manhole Detection")

# -------------------- Settings --------------------
# Confidence slider
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

# Choose device (GPU if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.sidebar.write(f"Using device: **{device.upper()}**")

# Load model once
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    model.to(device)
    return model

model = load_model()

# Video stream
frame_placeholder = st.empty()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("❌ Cannot access the webcam.")
else:
    st.success("✅ Webcam connected. Press **Stop** to end the stream.")

    stop_button = st.button("Stop Stream")

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Failed to capture frame.")
            break

        # Resize frame for better performance
        frame = cv2.resize(frame, (640, 480))

        # Inference on GPU with confidence threshold
        results = model.predict(frame, conf=conf_threshold, device=device, verbose=False)

        # Draw detections
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls = model.names[int(box.cls)]
                label = f"{cls} ({conf:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Convert BGR → RGB and display in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Slight delay to reduce CPU load
        time.sleep(0.03)

    cap.release()
    st.write("🛑 Stream stopped.")
