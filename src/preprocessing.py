import cv2
import numpy as np

def preprocess_face(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = np.reshape(img, (1, 48, 48, 1))
    return img