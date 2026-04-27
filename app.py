import streamlit as st
import cv2
import tempfile
from tracker import process_frame
from utils import draw_dashboard

st.set_page_config(page_title="Smart Object Tracking", layout="wide")

st.title("🚀 Smart Object Tracking System")

# Sidebar controls (adds professionalism)
st.sidebar.title("⚙️ Settings")
run = st.sidebar.checkbox("Start Processing", value=True)

video_file = st.file_uploader("📤 Upload Video", type=["mp4", "avi", "mov"])

if video_file is not None:
    # Save uploaded file safely (IMPORTANT for deployment)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    if run:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            frame, count = process_frame(frame)
            frame = draw_dashboard(frame, count)

            # Convert BGR → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display frame
            stframe.image(frame, channels="RGB", use_column_width=True)

        cap.release()

    else:
        st.warning("⚠️ Click 'Start Processing' in sidebar to begin")