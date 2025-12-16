import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model(
    "models/Emotion_little_vgg.h5",
    compile=False
)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

face_cascade = cv2.CascadeClassifier(
    "models/haarcascade_frontalface_default.xml"
)

def predict_emotions_multi(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5
    )

    results = []

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        preds = model.predict(face, verbose=0)
        idx = np.argmax(preds)
        conf = float(np.max(preds))

        results.append({
            "box": (x, y, w, h),
            "label": emotion_labels[idx],
            "conf": conf
        })

    return results