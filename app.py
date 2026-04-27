import streamlit as st
import cv2
import tempfile
from tracker import process_frame
from utils import draw_dashboard

st.set_page_config(page_title="Smart Object Tracking", layout="wide")

st.title("🚀 Smart Object Tracking System")

# Sidebar controls
st.sidebar.title("⚙️ Settings")
run = st.sidebar.checkbox("Start Processing", value=True)

video_file = st.file_uploader("📤 Upload Video", type=["mp4", "avi", "mov"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    if run:
        frame_count = 0
        MAX_FRAMES = 150   # 🔥 limit to avoid crash

        progress_bar = st.progress(0)

        while cap.isOpened() and frame_count < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            frame, count = process_frame(frame)
            frame = draw_dashboard(frame, count)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB", use_column_width=True)

            frame_count += 1
            progress_bar.progress(frame_count / MAX_FRAMES)

        cap.release()

        st.success(f"✅ Processed {frame_count} frames")

    else:
        st.warning("⚠️ Click 'Start Processing' in sidebar to begin")
