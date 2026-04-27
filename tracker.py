import cv2
import streamlit as st
from ultralytics import YOLO
import os

MODEL_PATH = "yolov8n.pt"

# ✅ Load model safely (cached)
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return YOLO(MODEL_PATH)
        except Exception:
            os.remove(MODEL_PATH)

    return YOLO("yolov8n.pt")

model = load_model()

# ✅ Use session state instead of globals
if "total_count" not in st.session_state:
    st.session_state.total_count = 0

if "counted_ids" not in st.session_state:
    st.session_state.counted_ids = set()


def process_frame(frame):
    results = model(frame)[0]

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])

        if class_id == 0:  # person
            center_y = int((y1 + y2) / 2)

            # ✅ better pseudo ID (still lightweight)
            person_id = f"{x1}_{y1}"

            if center_y > 300 and person_id not in st.session_state.counted_ids:
                st.session_state.counted_ids.add(person_id)
                st.session_state.total_count += 1

            # draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, "Person", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # draw counting line
    cv2.line(frame, (0,300), (frame.shape[1],300), (0,0,255), 2)

    # display count
    cv2.putText(frame, f"Count: {st.session_state.total_count}", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    return frame, st.session_state.total_count
