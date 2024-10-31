import streamlit as st
import os
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import torch
from transformers import ViTForImageClassification, AutoImageProcessor
from safetensors.torch import load_file
from PIL import ImageOps

# Load TensorFlow model
file_path = os.path.abspath('./model.h5')
tf_model = load_model(file_path)

def preprocess_image_tf(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((48, 48))  # Resize to 48x48
    image = img_to_array(image)  # Convert to array
    image = np.expand_dims(image, axis=0)  # Expand dims for batch
    image = image / 255.0  # Normalize
    return image



def preprocess_image_hf(image):
    """Preprocess image for Hugging Face model (resize to 224x224 and convert to RGB if needed)."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((224, 224), Image.LANCZOS)
    
    return image  



class EmotionDetector:
    def __init__(self, model_folder_path):

        self.image_processor = AutoImageProcessor.from_pretrained(model_folder_path)

        state_dict = load_file(f"{model_folder_path}/model.safetensors")
        self.model = ViTForImageClassification.from_pretrained(model_folder_path, state_dict=state_dict, local_files_only=True)
        self.model.eval()

        self.id2label = {
            "0": "sad 😢",
            "1": "disgust 🤢",
            "2": "angry 😡",
            "3": "neutral 😐",
            "4": "fear 😱",
            "5": "surprise 😲",
            "6": "happy 😊"
        }

    def detect_emotion(self, image):
        """Detect emotion in the image and return probabilities for each class."""
        inputs = self.image_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze().numpy()  
            predicted_class_idx = logits.argmax(-1).item()
            predicted_label = self.id2label[str(predicted_class_idx)]

        return predicted_label, probabilities

model_name = "./models"  
emotion_detector = EmotionDetector(model_name)


st.subheader("Perbandingan Prediksi Emosi Wajah")


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
        uploaded_file = st.file_uploader("Upload gambar wajah", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

if image is not None:
    st.image(image, caption='Gambar Diupload', use_column_width=True)

    col1, col2 = st.columns(2)
    with col1:
        preprocessed_image_tf = preprocess_image_tf(image)
        tf_prediction = tf_model.predict(preprocessed_image_tf)

        tf_class_names = ["Marah 😡", "Senang 😊", "Netral 😐", "Sedih 😢"]
        tf_predicted_class = tf_class_names[np.argmax(tf_prediction)]
        tf_probabilities = tf_prediction[0]
        st.subheader("Model dari haldies")
        st.text("Model TensorFlow akurasi 72% dari CNN:")
        st.write(f"Prediksi Emosi: {tf_predicted_class}")
        for class_name, prob in zip(tf_class_names, tf_probabilities):
            st.write(f"{class_name}: {prob * 100:.2f}%")

    with col2:
        preprocessed_image_hf = preprocess_image_hf(image)  # Preprocess for Hugging Face
        hf_predicted_label, hf_probabilities = emotion_detector.detect_emotion(preprocessed_image_hf)
        st.subheader("Model dari hungging face")
        st.text("Model Hugging Face akurasi 90% Transfer learning ViT:")
        st.write(f"Prediksi Emosi: {hf_predicted_label}")
        for idx, (class_idx, prob) in enumerate(zip(emotion_detector.id2label.keys(), hf_probabilities)):
            st.write(f"{emotion_detector.id2label[class_idx]}: {prob * 100:.2f}%")
