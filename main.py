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


tf_model_fer2013 = load_model(file_path_fer2013)

file_path_ckplus = os.path.abspath('./modelCKplus.h5')
tf_model_ckplus = load_model(file_path_ckplus)


def preprocess_image_tf(image):
    image = image.convert('L')
    image = image.resize((48, 48))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image


def preprocess_image_hf(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224), Image.LANCZOS)
    return image


class EmotionDetector:
    def __init__(self, model_folder_path):
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_folder_path)
        state_dict = load_file(f"{model_folder_path}/model.safetensors")
        self.model = ViTForImageClassification.from_pretrained(
            model_folder_path, state_dict=state_dict, local_files_only=True)
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
        inputs = self.image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(
                logits, dim=-1).squeeze().numpy()
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
        uploaded_file = st.file_uploader(
            "Upload gambar wajah", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

if image is not None:
    st.image(image, caption='Gambar Diupload', use_column_width=True)

    col1, col2, col3 = st.columns(3)

    with col1:
       
        preprocessed_image_tf_fer2013 = preprocess_image_tf(image)
        tf_prediction_fer2013 = tf_model_fer2013.predict(
            preprocessed_image_tf_fer2013)

        tf_class_names_fer2013 = ["Marah 😡", "Senang 😊", "Netral 😐", "Sedih 😢"]
        tf_predicted_class_fer2013 = tf_class_names_fer2013[np.argmax(
            tf_prediction_fer2013)]
        tf_probabilities_fer2013 = tf_prediction_fer2013[0]
        st.subheader("Model FER2013 (TensorFlow)")
        st.write("Model TensorFlow akurasi 73% dari CNN dataset FER2013")
        st.write(f"Prediksi Emosi: {tf_predicted_class_fer2013}")
        for class_name, prob in zip(tf_class_names_fer2013, tf_probabilities_fer2013):
            st.write(f"{class_name}: {prob * 100:.2f}%")

    with col2:
      
        preprocessed_image_tf_ckplus = preprocess_image_tf(image)
        tf_prediction_ckplus = tf_model_ckplus.predict(
            preprocessed_image_tf_ckplus)

        tf_class_names_ckplus = [
            'Marah 😡',
            'Menghina 😠',
            'Jijik 🤢',
            'Takut 😱',
            'Senang 😊',
            'Sedih 😢',
            'Terkejut 😲'
        ]

        tf_predicted_class_ckplus = tf_class_names_ckplus[np.argmax(
            tf_prediction_ckplus)]
        tf_probabilities_ckplus = tf_prediction_ckplus[0]
        st.subheader("Model CKPlus (TensorFlow)")
        st.write("Model TensorFlow akurasi 94% dari CNN dataset CKPlus")
        st.write(f"Prediksi Emosi: {tf_predicted_class_ckplus}")
        for class_name, prob in zip(tf_class_names_ckplus, tf_probabilities_ckplus):
            st.write(f"{class_name}: {prob * 100:.2f}%")

    with col3:
        # Hugging Face ViT Model
        preprocessed_image_hf = preprocess_image_hf(image)
        hf_predicted_label, hf_probabilities = emotion_detector.detect_emotion(
            preprocessed_image_hf)
        st.subheader("Model Hugging Face (ViT)")
        st.write(
            f"Model Hugging Face akurasi 91% dengan Transfer Learning ViT: [sumber link](https://huggingface.co/dima806/facial_emotions_image_detection)")
        st.write(f"Prediksi Emosi: {hf_predicted_label}")
        for idx, (class_idx, prob) in enumerate(zip(emotion_detector.id2label.keys(), hf_probabilities)):
            st.write(
                f"{emotion_detector.id2label[class_idx]}: {prob * 100:.2f}%")
