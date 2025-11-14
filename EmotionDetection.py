import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    if not ret:
        break

    try:
        # Analyze the emotions
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Get the dominant emotion
        if isinstance(result, list):
            dominant_emotion = result[0].get('dominant_emotion', 'unknown')
        else:
            dominant_emotion = result.get('dominant_emotion', 'unknown')

        # Display it on the camera
        cv2.putText(frame, f'Emotion: {dominant_emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    except Exception as e:
        cv2.putText(frame, f'No emotion detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('real time emotion detected system', frame)

    # Press q to exit 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()