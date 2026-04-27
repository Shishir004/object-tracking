import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

total_count = 0

def process_frame(frame):
    global total_count

    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])

        if class_id == 0:  # person
            center_y = int((y1 + y2) / 2)

            # counting logic
            if center_y > 300:
                total_count += 1

            # draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    # draw line
    cv2.line(frame, (0,300), (frame.shape[1],300), (0,0,255), 2)

    return frame, total_count
