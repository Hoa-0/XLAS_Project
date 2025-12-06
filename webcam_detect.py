import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model(r"C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5")
emotion_labels = ["angry","disgust","fear","happy","neutral","sad","surprise"]

face_cascade = cv2.CascadeClassifier(r'C:\XLAS\DoAn\XLAS_Project\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48,48))
        roi = roi.reshape(1,48,48,1) / 255.0

        pred = model.predict(roi)
        emotion = emotion_labels[np.argmax(pred)]

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()