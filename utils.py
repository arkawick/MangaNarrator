import os
from PIL import Image
import streamlit as st

def save_uploaded_file(uploaded_file, save_dir="temp_uploads"):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def load_image(image_path):
    return Image.open(image_path).convert("RGB")
