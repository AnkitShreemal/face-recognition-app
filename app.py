import streamlit as st
import face_recognition
import numpy as np
import os
from PIL import Image

st.set_page_config(page_title="Face Recognition", layout="centered")
st.title("📸 Face Recognition App")
st.write("Upload a photo or take a selfie to recognize the person.")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

# Load known faces
known_encodings = []
known_names = []

for file in os.listdir("known_faces"):
    image = face_recognition.load_image_file(f"known_faces/{file}")
    encoding = face_recognition.face_encodings(image)
    if encoding:
        known_encodings.append(encoding[0])
        known_names.append(os.path.splitext(file)[0])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    locations = face_recognition.face_locations(image_np)
    encodings = face_recognition.face_encodings(image_np, locations)

    for loc, face_encoding in zip(locations, encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            index = matches.index(True)
            name = known_names[index]

        st.write(f"🎯 Detected: **{name}**")