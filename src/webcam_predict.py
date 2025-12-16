# src/webcam_predict.py
import cv2
from predict_model import predict_image

def run_webcam(model_name="Little-VGG"):
    cap = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier(r"C:\XLAS\DoAn\XLAS_Project\resources\haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            label, conf = predict_image(model_name, face_img)

            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf*100:.1f}%", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("FER Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()