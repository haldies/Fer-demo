import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
import requests
from io import BytesIO


file_path = os.path.abspath('./model.h5')
model = load_model(file_path)

def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((48, 48))  # Resize to 48x48
    image = img_to_array(image)  # Convert to array
    image = np.expand_dims(image, axis=0)  # Expand dims for batch
    image = image / 255.0  # Normalize
    return image

st.title("Prediksi Emosi Wajah")


uploaded_file = None
image = None


use_camera = st.checkbox("Gunakan Kamera untuk Foto")
image_url = st.text_input("Atau masukkan URL gambar:")

if use_camera:
    uploaded_file = st.camera_input("Ambil gambar dengan kamera")
elif image_url:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        st.error("Gagal mengunduh gambar. Periksa URL dan coba lagi.")
else:
    uploaded_file = st.file_uploader("Upload gambar wajah", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar Diupload', use_column_width=True)
elif image is not None:
    st.image(image, caption='Gambar dari URL', use_column_width=True)


if image is not None:
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)

    class_names = ["Marah", "Senang", "Netral", "Sedih"]

    predicted_class = class_names[np.argmax(prediction)]
    probabilities = prediction[0]

    st.write(f"Prediksi Emosi: {predicted_class}")
    for class_name, prob in zip(class_names, probabilities):
        st.write(f"{class_name}: {prob * 100:.2f}%")

    stress_emotions = ["Marah", "Sedih"]
    stress_probability = sum(prob for class_name, prob in zip(class_names, probabilities) if class_name in stress_emotions)

    if stress_probability > 0.5:
        st.write("Analisis Stres: Terdeteksi kemungkinan stres.")
    else:
        st.write("Analisis Stres: Tidak terdeteksi stres.")
