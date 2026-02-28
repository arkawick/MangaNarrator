# Text-to-Speech (TTS)

## Role in ECHO-TOON

TTS is the final output stage of the pipeline. After the dialogue script is assembled with speaker labels (`[S1] text [S2] text`), TTS converts it into audio narration. The quality, naturalness, and multi-speaker capability of the TTS engine directly determines the listening experience for visually impaired users.

Key requirements for ECHO-TOON's TTS:
- **Multi-speaker support** — different characters need distinct voices
- **Natural prosody** — expressive, not robotic
- **Offline operation** — no internet dependency for final use
- **Controllable** — scripts drive the output, not random voice variation

---

## Dia TTS — Primary Choice

### What is Dia?

Dia is an open-weights, multi-speaker text-to-dialogue model from Nari Labs. It is specifically designed for dialogue generation with multiple speaker tags, making it a natural fit for the ECHO-TOON use case.

Repository: https://github.com/nari-labs/dia
Model: `nari-labs/Dia-1.6B` (1.6 billion parameters)
Model card: https://huggingface.co/nari-labs/Dia-1.6B-0626

### Why Dia for Manga Narration?

Standard TTS models take plain text and produce a single voice. Dia natively understands a scripted dialogue format with multiple speaker tags — meaning you pass it a script and it automatically assigns different voices to each tag.

### Dia Script Format

```
[S1] Hello, are you ready? [S2] Always ready. [S1] Then let's go. (whispers) [S2] Wait for me.
```

- `[S1]`, `[S2]` etc. denote different speakers
- Nonverbal cues in parentheses are supported: `(laughs)`, `(sighs)`, `(whispers)`
- Up to ~9 speakers supported

### Installation

```bash
# Clone the Dia repository
git clone https://github.com/nari-labs/dia.git
cd dia
pip install -e .
```

Or:

```bash
pip install git+https://github.com/nari-labs/dia.git
```

### Basic Usage

```python
from dia.model import Dia

model = Dia.from_pretrained(
    "nari-labs/Dia-1.6B-0626",
    compute_dtype="float16"    # use float16 for GPU efficiency
)

text = "[S1] We need to leave now. [S2] But where? [S1] Anywhere is better than here."

output = model.generate(
    text,
    use_torch_compile=False,
    verbose=True,
    cfg_scale=3.0,
    temperature=1.8,
    top_p=0.90,
    cfg_filter_top_k=50,
)

model.save_audio("output.mp3", output)
```

### Generation Parameters

| Parameter | Description | Default | Notes |
|---|---|---|---|
| `cfg_scale` | Classifier-free guidance scale | 3.0 | Higher = more faithful to script |
| `temperature` | Sampling randomness | 1.8 | Lower = more deterministic |
| `top_p` | Nucleus sampling threshold | 0.90 | Trims low-probability tokens |
| `cfg_filter_top_k` | Top-k for CFG | 50 | Improves coherence |
| `use_torch_compile` | Torch compile for speed | False | Set True if supported |

### How Dia Integrates with ECHO-TOON

The `streamlit_app.py` "Generate Dia-formatted Output" button produces:

```
[S1] Sure is comfy~
[S2] Zuzu... (sip)
[S1] I feel like falling asleep.
```

This exact format is Dia-ready. The next step (not yet wired in) is:

```python
from dia.model import Dia

model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float16")
output = model.generate(dia_script)
model.save_audio("narration.mp3", output)
```

### Hardware Requirements for Dia

| Config | VRAM | Speed |
|---|---|---|
| float16, GPU | ~3.5 GB | Real-time |
| float32, GPU | ~7 GB | Real-time |
| CPU (float32) | 16 GB RAM | Very slow (10–30× slower) |

### Pros

- Multi-speaker natively (designed for this use case)
- Open weights — fully offline
- Expressive prosody with nonverbal cues
- Actively maintained

### Cons

- Requires ~3.5 GB VRAM for float16
- Speaker voices are not predefined/named (S1 always sounds the same across runs)
- No voice cloning from reference audio in base version

---

## gTTS — Google Text-to-Speech

### What is gTTS?

gTTS (Google Text-to-Speech) is a Python wrapper around the Google Translate TTS API. It produces natural-sounding speech by calling Google's cloud service.

```bash
pip install gTTS
```

### Usage

```python
from gtts import gTTS
import os

text = "Hello! This is ECHO-TOON speaking."
tts = gTTS(text=text, lang='en', slow=False)
tts.save("output.mp3")
os.system("start output.mp3")   # Windows play
```

### With Streamlit

```python
import streamlit as st
from gtts import gTTS
import tempfile

tts = gTTS(text=dialogue, lang='en')
with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
    tts.save(fp.name)
    st.audio(fp.name)
```

### Pros

- Very easy to use
- Natural-sounding output
- Multi-language support (matches OCR language)

### Cons

- Requires internet (calls Google cloud)
- Single voice (no multi-speaker)
- Rate limits on heavy use
- Not suitable for offline deployment

---

## pyttsx3 — Offline System TTS

### What is pyttsx3?

pyttsx3 interfaces with the system's native TTS engine — SAPI5 on Windows, NSSpeechSynthesizer on macOS, and espeak on Linux. Fully offline.

```bash
pip install pyttsx3
```

### Usage

```python
import pyttsx3

engine = pyttsx3.init()

# List available voices
voices = engine.getProperty('voices')
for voice in voices:
    print(voice.id, voice.name)

# Set voice
engine.setProperty('voice', voices[0].id)   # Male
engine.setProperty('rate', 150)              # Speed (words per minute)
engine.setProperty('volume', 1.0)

engine.say("Chapter one: A rainy night in Tokyo.")
engine.runAndWait()

# Save to file
engine.save_to_file("Chapter one: A rainy night in Tokyo.", "output.mp3")
engine.runAndWait()
```

### Multi-Speaker Simulation

```python
male_voice = voices[0].id
female_voice = voices[1].id

for line in dialogue_lines:
    if line['char'] == 'Character_A':
        engine.setProperty('voice', male_voice)
    else:
        engine.setProperty('voice', female_voice)
    engine.say(line['text'])

engine.runAndWait()
```

### Pros

- Fully offline
- No API keys
- Basic multi-voice support via system voices
- Lightweight

### Cons

- Robotic quality (not neural TTS)
- Dependent on OS voices (Windows voices are limited)
- Cannot save directly to WAV in all configurations

---

## edge-tts — Microsoft Edge Neural TTS

### What is edge-tts?

edge-tts is a Python library that uses the Microsoft Edge browser's TTS service. It provides access to Microsoft's high-quality neural voices — the same voices used in Azure TTS — but for free via the Edge API.

```bash
pip install edge-tts
```

### Usage

```python
import edge_tts
import asyncio

async def generate():
    communicate = edge_tts.Communicate(
        text="We have to leave now!",
        voice="en-US-GuyNeural"
    )
    await communicate.save("output.mp3")

asyncio.run(generate())
```

### Available Voices (Sample)

```bash
# List all voices
edge-tts --list-voices
```

| Voice | Gender | Style |
|---|---|---|
| en-US-GuyNeural | Male | Natural |
| en-US-JennyNeural | Female | Natural |
| en-US-AriaNeural | Female | Expressive |
| en-GB-RyanNeural | Male | British |
| en-AU-NatashaNeural | Female | Australian |
| ja-JP-NanamiNeural | Female | Japanese |

### Multi-Speaker with edge-tts

```python
async def narrate_script(dialogue_lines, voice_map):
    combined = AudioSegment.empty()
    for line in dialogue_lines:
        voice = voice_map.get(line['char'], "en-US-GuyNeural")
        communicate = edge_tts.Communicate(line['text'], voice=voice)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        await communicate.save(tmp.name)
        combined += AudioSegment.from_mp3(tmp.name)
    combined.export("full_narration.mp3", format="mp3")
```

### Pros

- High-quality neural voices (Microsoft quality)
- Many voice options and styles
- Free (uses Edge API)
- Good multi-language support

### Cons

- Requires internet
- No multi-speaker in a single call (must chain per line)
- API may change (unofficial wrapper)

---

## ElevenLabs — Premium Cloud TTS

### What is ElevenLabs?

ElevenLabs is a commercial AI voice platform with industry-leading voice quality and voice cloning capabilities.

```bash
pip install requests
```

### Usage (as in `elevenlabs_api.py`)

```python
import requests
from io import BytesIO

API_KEY = "your_api_key"
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"   # Rachel (default)

def generate_speech(text):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "xi-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    response = requests.post(url, headers=headers, json=data)
    return BytesIO(response.content)
```

### Voice Cloning

ElevenLabs supports voice cloning from a 1–2 minute audio sample. This could be used to create character-specific voices from dubbed anime audio.

### Pros

- Best quality TTS available
- Voice cloning from reference audio
- Multi-language
- Emotion control

### Cons

- Paid (free tier: 10,000 chars/month)
- Cloud-only — not offline
- API key required
- Privacy: audio sent to cloud

---

## Coqui TTS — Open Source Neural TTS

### What is Coqui TTS?

Coqui TTS is an open-source deep learning TTS library supporting dozens of models including VITS, YourTTS (zero-shot voice cloning), and XTTS.

```bash
pip install TTS
```

### Usage

```python
from TTS.api import TTS

# List available models
TTS().list_models()

# Load a model
tts = TTS("tts_models/en/ljspeech/vits")
tts.tts_to_file(text="Hello world!", file_path="output.wav")

# Multi-speaker model
tts = TTS("tts_models/en/vctk/vits")
tts.tts_to_file(
    text="Character A speaks.",
    speaker="p225",
    file_path="char_a.wav"
)
```

### XTTS — Zero-Shot Voice Cloning

```python
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.tts_to_file(
    text="This is the cloned voice.",
    speaker_wav="reference_audio.wav",
    language="en",
    file_path="cloned.wav"
)
```

### Pros

- Fully offline and open source
- Multi-speaker and voice cloning
- High quality (XTTS rivals commercial services)
- Local, privacy-preserving

### Cons

- Larger model sizes (XTTS ~2–5 GB)
- Requires more setup than gTTS/pyttsx3
- VRAM hungry for real-time use

---

## TTS Comparison Table

| TTS Engine | Quality | Multi-Speaker | Offline | Cost | VRAM |
|---|---|---|---|---|---|
| Dia 1.6B | Excellent | Yes (native) | Yes | Free | ~3.5 GB |
| ElevenLabs | Best | Yes (per call) | No | Paid | N/A |
| Coqui XTTS | Excellent | Yes (cloning) | Yes | Free | ~4 GB |
| edge-tts | Very Good | Manual chaining | No | Free | None |
| gTTS | Good | No | No | Free | None |
| pyttsx3 | Poor | Basic | Yes | Free | None |

---

## Audio Post-Processing (pydub)

After TTS generates per-line audio files, they can be concatenated and processed:

```python
from pydub import AudioSegment
from pydub.effects import normalize

segments = []
for line in narration_lines:
    seg = AudioSegment.from_mp3(line['audio_path'])
    segments.append(seg)

# Add 500ms pause between lines
silence = AudioSegment.silent(duration=500)
combined = silence.join(segments)

# Normalize volume
combined = normalize(combined)
combined.export("final_narration.mp3", format="mp3", bitrate="192k")
```
