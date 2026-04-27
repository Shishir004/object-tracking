import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

# Store counted IDs
counted_ids = set()
total_count = 0

def process_frame(frame):
    global total_count

    results = model(frame)[0]
    detections = []

    for r in results.boxes.data:
        x1, y1, x2, y2, score, class_id = r.tolist()

        if int(class_id) == 0:  ## DeepSORT needs:
                                ## [x, y, width, height]
            detections.append(([x1, y1, x2-x1, y2-y1], score, 'person'))

    tracks = tracker.update_tracks(detections, frame=frame) ##Matches detections with previous frames Assigns consistent IDs

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = track.to_ltrb()

        center_y = int(t + h / 2)

        # COUNTING LOGIC (crossing line)
        if center_y > 300 and track_id not in counted_ids: ## If person crosses line (y = 300) AND hasn’t been counted before
            counted_ids.add(track_id)   
            total_count += 1

        # Draw box
        cv2.rectangle(frame, (int(l), int(t)), (int(l+w), int(t+h)), (0,255,0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(l), int(t-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Draw counting line
    cv2.line(frame, (0,300), (frame.shape[1],300), (0,0,255), 2)

    return frame, total_count