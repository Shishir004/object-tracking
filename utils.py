import cv2

def draw_dashboard(frame, count):
    cv2.rectangle(frame, (10,10), (250,80), (0,0,0), -1)
    cv2.putText(frame, f"Count: {count}", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    return frame