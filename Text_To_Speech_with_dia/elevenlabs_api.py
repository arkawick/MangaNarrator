# app.py
import streamlit as st
import requests
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play

# Optional: install ffmpeg for pydub
# sudo apt install ffmpeg  (Linux)
# brew install ffmpeg      (macOS)

# Your ElevenLabs API key
API_KEY = "your_elevenlabs_api_key_here"   # Get from: https://elevenlabs.io/app/speech-synthesis
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Default voice (Rachel)

def generate_speech(text):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "xi-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",  # or eleven_multilingual_v1
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        st.error(f"API Error: {response.status_code} - {response.text}")
        return None

# Streamlit UI
st.title("🎙️ ElevenLabs TTS Demo")
text = st.text_area("Enter text to synthesize:", "Hello, this is a test from ElevenLabs API.")

if st.button("🔊 Generate Speech"):
    if text.strip():
        audio_bytes = generate_speech(text)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
    else:
        st.warning("Please enter some text.")
