import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model(r"C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5")
emotion_labels = ["angry","disgust","fear","happy","neutral","sad","surprise"]

img_path = r"C:\XLAS\DoAn\XLAS_Project\dataset_new\train\angry\angry_0.jpg"  # đổi tên ảnh
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (48,48))
img = img.reshape(1,48,48,1) / 255.0

pred = model.predict(img)
emotion = emotion_labels[np.argmax(pred)]

print("Emotion:", emotion)