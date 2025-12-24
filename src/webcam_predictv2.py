import cv2
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.predict_model import predict_emotion_multi

MODEL_NAME = "CNN"     # "CNN" | "VGG16"

def run_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = predict_emotion_multi(frame, MODEL_NAME)

        for r in results:
            conf = r["conf"]

            # Đánh giá mức độ tự tin
            if conf >= 0.75:
                color = (0, 255, 0)      # High
            elif conf >= 0.55:
                color = (0, 255, 255)    # Medium
            else:
                color = (0, 0, 255)      # Low

            # Case ảnh nhỏ → chỉ hiện chữ
            if r["box"] is None:
                text = f"{r['label']} ({conf*100:.1f}%)"
                cv2.putText(
                    frame,
                    text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    color,
                    2
                )
            else:
                x, y, w, h = r["box"]

                label_text = f"{r['label']} ({conf*100:.1f}%)"

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(
                    frame,
                    label_text,
                    (x, max(y - 10, 20)),  # tránh chữ bị cắt
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2
                )

        cv2.imshow(f"FER Webcam [{MODEL_NAME}]", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
