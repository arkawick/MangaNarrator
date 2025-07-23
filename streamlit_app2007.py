# streamlit_app.py
import numpy as np
import streamlit as st
from PIL import Image
import easyocr
import torch
import requests
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS
import os
import tempfile
from langdetect import detect
from deep_translator import GoogleTranslator

# Setup OCR, BLIP, TTS, Translator
reader = easyocr.Reader(['ja', 'en'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model

processor, model = load_blip_model()

@st.cache_resource
def load_clip_caption():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

caption_pipeline = load_clip_caption()

def run_ocr(image):
    # return reader.readtext(image)
    # Convert PIL image to numpy array

    image_np = np.array(image)
    return reader.readtext(image_np)

def blip_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def translate_text(text):
    try:
        lang = detect(text)
        if lang in ['ja', 'ko', 'zh-cn', 'zh-tw']:
            return GoogleTranslator(source='auto', target='en').translate(text)
        return text
    except:
        return text

def speak_text(text):
    tts = gTTS(text)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# Streamlit UI
st.title("Webtoon Reader with OCR, Captioning, Translation and TTS")

uploaded_file = st.file_uploader("Upload a Webtoon Panel (Image)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Panel", use_column_width=True)

    st.subheader("Extracted Text (OCR):")
    ocr_results = run_ocr(image)
    for bbox, text, conf in ocr_results:
        translated = translate_text(text)
        st.write(f"**Original:** {text} | **Translated:** {translated}")
        if st.button(f"🔊 Speak: {translated[:20]}..."):
            audio_file = speak_text(translated)
            st.audio(audio_file)

    st.subheader("Image Caption (BLIP):")
    blip_text = blip_caption(image)
    translated_caption = translate_text(blip_text)
    st.write(f"**Original:** {blip_text}\n\n**Translated:** {translated_caption}")
    if st.button("🔊 Speak Caption"):
        audio_file = speak_text(translated_caption)
        st.audio(audio_file)

    st.subheader("Image-to-Text Summary (CLIP/BLIP2):")
    summary = caption_pipeline(image)[0]['generated_text']
    translated_summary = translate_text(summary)
    st.write(f"**Original:** {summary}\n\n**Translated:** {translated_summary}")
    if st.button("🔊 Speak Summary"):
        audio_file = speak_text(translated_summary)
        st.audio(audio_file)

st.caption("OCR: EasyOCR | Image Caption: BLIP | TTS: Google TTS | Translation: GoogleTranslator")
