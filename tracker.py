import cv2
import streamlit as st
from ultralytics import YOLO
import os

MODEL_PATH = "yolov8n.pt"

@st.cache_resource
def load_model():
    # 🔥 delete corrupted model if exists
    if os.path.exists(MODEL_PATH):
        try:
            return YOLO(MODEL_PATH)
        except:
            os.remove(MODEL_PATH)  # remove broken file

    # fresh download
    return YOLO("yolov8n.pt")

model = load_model()

total_count = 0
counted_ids = set()

def process_frame(frame):
    global total_count

    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])

        if class_id == 0:
            center_y = int((y1 + y2) / 2)
            person_id = (x1, y1, x2, y2)

            if center_y > 300 and person_id not in counted_ids:
                counted_ids.add(person_id)
                total_count += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.line(frame, (0,300), (frame.shape[1],300), (0,0,255), 2)

    return frame, total_count
