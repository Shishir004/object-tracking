import cv2
import streamlit as st
from ultralytics import YOLO

# ✅ Load model only once (VERY IMPORTANT for Streamlit)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# global counter
total_count = 0
counted_ids = set()  # to avoid duplicate counting

def process_frame(frame):
    global total_count

    results = model(frame)[0]

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])

        if class_id == 0:  # person
            center_y = int((y1 + y2) / 2)

            # unique ID based on position (simple trick)
            person_id = (x1, y1, x2, y2)

            # counting logic (avoid multiple counts)
            if center_y > 300 and person_id not in counted_ids:
                counted_ids.add(person_id)
                total_count += 1

            # draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, "Person", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # draw counting line
    cv2.line(frame, (0,300), (frame.shape[1],300), (0,0,255), 2)

    # show count on screen
    cv2.putText(frame, f"Count: {total_count}", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    return frame, total_count
