import cv2
import numpy as np
import tensorflow as tf
import os

# ===============================
# PATH
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATHS = {
    "CNN": os.path.join(BASE_DIR, "models", "Emotion_little_vgg.h5"),
    "VGG16": os.path.join(BASE_DIR, "models", "Emotion_VGG16_Optimized.h5"),
}

emotion_labels = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

face_cascade = cv2.CascadeClassifier(
    os.path.join(BASE_DIR, "resources", "haarcascade_frontalface_default.xml")
)

_loaded_models = {}

# ===============================
# LOAD MODEL (CACHE)
# ===============================
def load_model_by_name(model_name):
    if model_name not in MODEL_PATHS:
        raise ValueError(f"❌ Unknown model: {model_name}")

    if model_name not in _loaded_models:
        _loaded_models[model_name] = tf.keras.models.load_model(
            MODEL_PATHS[model_name],
            compile=False
        )
    return _loaded_models[model_name]


# ===============================
# PREPROCESS
# ===============================
def preprocess_face(face_gray, model_name):
    """
    CNN   : 48x48 grayscale (1 channel)
    VGG16 : 48x48 RGB (3 channels)
    """
    if model_name == "VGG16":
        face = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2RGB)
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = face.astype("float32")
        face = np.expand_dims(face, axis=0)
    else:  # CNN
        face = cv2.resize(face_gray, (48, 48))
        face = face / 255.0
        face = face.astype("float32")
        face = face.reshape(1, 48, 48, 1)

    return face


# ===============================
# MAIN PREDICT
# ===============================
def predict_emotion_multi(img, model_name="CNN"):
    model = load_model_by_name(model_name)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    results = []

    # ===== CASE 1: Ảnh rất nhỏ (48x48, 64x64...)
    if len(faces) == 0:
        h, w = gray.shape
        if h <= 64 and w <= 64:
            face = preprocess_face(gray, model_name)
            preds = model.predict(face, verbose=0)
            idx = np.argmax(preds)

            return [{
                "box": None,
                "label": emotion_labels[idx],
                "conf": float(np.max(preds))
            }]
        return []

    # ===== CASE 2: Detect được khuôn mặt
    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        face = preprocess_face(face_gray, model_name)

        preds = model.predict(face, verbose=0)
        idx = np.argmax(preds)

        results.append({
            "box": (x, y, w, h),
            "label": emotion_labels[idx],
            "conf": float(np.max(preds))
        })

    return results