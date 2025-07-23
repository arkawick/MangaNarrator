# import pyttsx3
# engine = pyttsx3.init()
# engine.say("This is espeak via pyttsx3")
# engine.runAndWait()

# from dia.model import Dia


# model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float16")

# text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."

# output = model.generate(
#     text,
#     use_torch_compile=False,
#     verbose=True,
#     cfg_scale=3.0,
#     temperature=1.8,
#     top_p=0.90,
#     cfg_filter_top_k=50,
# )

# model.save_audio("simple.mp3", output)


import streamlit as st
import os
import tempfile

import sys
# sys.path.append("/full/path/to/cloned/dia")
from dia.pipeline import TextToSpeechPipeline


st.set_page_config(page_title="Dia TTS Demo", layout="centered")

@st.cache_resource
def load_tts():
    return TextToSpeechPipeline.from_pretrained("nari-labs/dia", device="cuda" if torch.cuda.is_available() else "cpu")

# Load model
st.title("🗣️ Dia: Local Multi-Speaker TTS Demo")
st.write("Generate high-quality speech with [Nari-labs/Dia](https://github.com/nari-labs/dia). All offline!")

tts = load_tts()
speakers = tts.list_speakers()

text = st.text_area("Enter text to speak:", "Hello! I am a synthetic voice from Dia.", height=100)

selected_speaker = st.selectbox("Select speaker:", speakers)

if st.button("🌀 Generate Speech"):
    with st.spinner("Synthesizing..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            tts.tts_to_file(
                text=text,
                speaker=selected_speaker,
                output_path=tmp_wav.name
            )
            st.success("✅ Speech generated!")
            st.audio(tmp_wav.name)
