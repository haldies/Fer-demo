from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
import requests
from PIL import Image
from io import BytesIO
import streamlit as st

# Ganti ini dengan repo yang sesuai
model_folder_path = "gerhardien/face-emotion"

# Mendownload model dan processor dari Hugging Face
image_processor = AutoImageProcessor.from_pretrained(model_folder_path)
model = AutoModelForImageClassification.from_pretrained(model_folder_path)

# Model label mapping
id2label = {
    "0": "sad 😢",
    "1": "disgust 🤢",
    "2": "angry 😡",
    "3": "neutral 😐",
    "4": "fear 😱",
    "5": "surprise 😲",
    "6": "happy 😊"
}

def detect_emotion(image):
    # Memproses gambar
    inputs = image_processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        # Prediksi emosi
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze().numpy()
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = id2label[str(predicted_class_idx)]
    
    return predicted_label, probabilities

# Streamlit
st.title("Deteksi Emosi Wajah")

use_camera = st.checkbox("Gunakan Kamera untuk Foto")
use_link = st.checkbox("Upload Gambar dari Link")

image = None

if use_link:
    image_url = st.text_input("Masukkan URL Gambar")
    if image_url:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
else:
    if use_camera:
        uploaded_file = st.camera_input("Ambil gambar dengan kamera")
    else:
        uploaded_file = st.file_uploader(
            "Upload gambar wajah", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

if image is not None:
    st.image(image, caption='Gambar Diupload', use_column_width=True)
    
    predicted_label, probabilities = detect_emotion(image)
    
    st.subheader("Model Deteksi Emosi Wajah dari Hugging Face")
    st.write(f"Prediksi Emosi: {predicted_label}")
    for idx, (class_idx, prob) in enumerate(zip(id2label.keys(), probabilities)):
        st.write(f"{id2label[class_idx]}: {prob * 100:.2f}%")
