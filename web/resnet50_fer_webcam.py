import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# ==============================
# CONFIG
# ==============================
MODEL_PATH = r"C:\Users\ADMIN\Visual Studio Code Projects\Python\XLAS_Project\models\resnet50_fer_finetuned.keras"
CASCADE_PATH = r"C:\Users\ADMIN\Visual Studio Code Projects\Python\XLAS_Project\resources\haarcascade_frontalface_default.xml"

IMG_SIZE = 224
CONF_THRESHOLD = 0.6

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# ==============================
# LOAD MODEL & CASCADE
# ==============================
model = load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ==============================
# HELPER FUNCTIONS
# ==============================
def apply_clahe_color(img):
    """CLAHE nhẹ cho ảnh màu (chỉ dùng realtime)"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def preprocess_face(face_bgr):
    """
    Preprocess CHUẨN cho ResNet50:
    BGR -> RGB -> preprocess_input
    """
    face = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype("float32")
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    return face

# ==============================
# WEBCAM
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Không mở được webcam")
    exit()

print("🎥 Webcam đang chạy | Nhấn Q để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]

        # Enhance ánh sáng (realtime only)
        face_roi = apply_clahe_color(face_roi)

        # Preprocess
        input_face = preprocess_face(face_roi)

        # Predict
        preds = model.predict(input_face, verbose=0)[0]
        confidence = np.max(preds)
        emotion_id = np.argmax(preds)
        emotion = EMOTIONS[emotion_id]

        # Confidence threshold
        if confidence < CONF_THRESHOLD:
            label = "Unknown"
            color = (0, 0, 255)
        else:
            label = f"{emotion} ({confidence:.2f})"
            color = (0, 255, 0)

        # Draw
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

    cv2.imshow("ResNet50 Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam đã tắt")
