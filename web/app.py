import os
import sys
import cv2
import numpy as np
import streamlit as st

# ===============================
# FIX IMPORT src/*
# ===============================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.predict_model import predict_emotion_multi

# Webcam
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="😃",
    layout="centered"
)

st.title("😃 Facial Expression Recognition")
st.write("**Môn:** Xử lý ảnh số – Nhận dạng biểu cảm khuôn mặt (FER)")

# ===============================
# TAB UI
# ===============================
tab1, tab2 = st.tabs(["📤 Upload ảnh", "📷 Realtime Webcam"])

# ======================================================
# TAB 1: UPLOAD IMAGE (NHIỀU KHUÔN MẶT)
# ======================================================
with tab1:
    uploaded = st.file_uploader(
        "Upload ảnh khuôn mặt (jpg / png)",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        results, img_draw = predict_emotion_multi(img)

        st.image(img_draw, channels="BGR", caption="Kết quả nhận dạng")

        if len(results) == 0:
            st.warning("⚠️ Không phát hiện khuôn mặt nào")
        else:
            st.subheader("📊 Kết quả:")
            for i, r in enumerate(results):
                st.write(
                    f"**Face {i+1}:** {r['label']} "
                    f"(confidence = {r['confidence']:.2f})"
                )

# ======================================================
# TAB 2: REALTIME WEBCAM
# ======================================================
class FERVideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        _, img_draw = predict_emotion_multi(img)
        return img_draw


with tab2:
    st.write("🎥 Nhận dạng biểu cảm khuôn mặt realtime")

    webrtc_streamer(
        key="fer-realtime",
        video_transformer_factory=FERVideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_transform=True
    )

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("© 2025 – FER Project | CNN – LittleVGG | Streamlit")