import os
import sys
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import subprocess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.predict_model import predict_emotion_multi

st.set_page_config(page_title="FER Demo", layout="centered")

st.title("😊 Facial Expression Recognition")

model_name = st.selectbox(
    "Select Model",
    ["CNN", "VGG16"]
)

mode = st.radio(
    "Select Mode",
    ["📷 Upload Image", "🎥 Webcam (OpenCV)"]
)

if mode == "📷 Upload Image":
    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        results = predict_emotion_multi(img, model_name)

        # Nếu ảnh nhỏ → show text
        if len(results) == 1 and results[0]["box"] is None:
            st.image(image, use_column_width=True)
            st.success(f"Prediction: **{results[0]['label']}**")
        else:
            for r in results:
                x, y, w, h = r["box"]
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(
                    img,
                    r["label"],
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0,255,0),
                    2
                )

            st.image(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                caption=f"Detected {len(results)} face(s)",
                use_column_width=True
            )

else:
    st.warning("Webcam sẽ mở ở cửa sổ riêng")

    if st.button("▶ Start Webcam"):
        subprocess.Popen([
            sys.executable,
            os.path.join(ROOT_DIR, "src", "webcam_predict.py")
        ])

    st.info("Nhấn **Q** để tắt webcam")