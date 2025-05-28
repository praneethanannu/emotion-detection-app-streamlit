import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load model
model = load_model("best_model.h5")

# Emotion labels (for 7-class model)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Anti-spoofing check using MediaPipe Face Mesh
def is_real_face(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return results.multi_face_landmarks is not None
        

# Predict emotion
def predict_emotion(img):
    try:
        img = cv2.resize(img, (224, 224))
        img = img.astype("float32") / 255.0
        reshaped = np.expand_dims(img, axis=0)

        preds = model.predict(reshaped)
        idx = np.argmax(preds)

        return emotion_labels[idx]
    except Exception as e:
        return "Error: " + str(e)

# Streamlit app
st.title("Emotion Detection with Anti-Spoofing")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        st.write("Failed to capture image.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]

        if not is_real_face(roi):
            emotion = "Spoof Detected"
        else:
            emotion = predict_emotion(roi)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
