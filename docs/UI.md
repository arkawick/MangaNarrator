# UI Layer — Streamlit, Gradio & Flask

## Current Implementation — Streamlit

### What is Streamlit?

Streamlit is a Python library that turns Python scripts into interactive web applications with zero frontend code. It is the fastest way to build data science and ML demos in Python.

```bash
pip install streamlit
streamlit run streamlit_app.py
```

The app runs at `http://localhost:8501` by default.

### How Streamlit Works

Streamlit reruns the entire Python script from top to bottom every time the user interacts with a widget (button click, slider change, file upload). State that needs to persist between reruns is stored in `st.session_state`.

```python
import streamlit as st

# Runs on every interaction
st.title("ECHO-TOON")

# File upload widget
uploaded = st.file_uploader("Upload manga panel", type=["png", "jpg"])

if uploaded:
    # This block runs when a file is present
    st.image(uploaded)
```

### Session State

```python
# Initialize state
if 'character_names' not in st.session_state:
    st.session_state.character_names = {}

# Read
names = st.session_state.character_names

# Write
st.session_state.character_names[0] = "Kiritsugu"
```

State persists across reruns within a browser session but is reset if the page is refreshed.

---

## Current ECHO-TOON UI Flow (`streamlit_app.py`)

```
Page Load
    │
    ▼
File Upload (st.file_uploader)
    │
    ▼
Step 1: OCR
  - st.spinner("Extracting dialogues...")
  - st.image(image_with_boxes)
  - st.checkbox("Show extracted text")
    │
    ▼
Step 2: Character Detection (checkbox gated)
  - st.spinner("Detecting characters...")
  - st.image(image_with_chars)
    │
    ▼
Step 3: Dialogue-Character Mapping (auto, once both above complete)
  - st.write per mapping result
    │
    ▼
Step 4: Character Naming
  - st.text_input per unique character ID
    │
    ▼
Step 4.5: Edit Assigned Dialogues
  - st.text_input for character name + dialogue per line
  - "Clear All Dialogues" button
    │
    ▼
Step 5: Generate Dia Script
  - st.button("Generate Dia-formatted Output")
  - st.text_area showing [S1]/[S2] script
```

### Key Streamlit Widgets Used

| Widget | Purpose | Code |
|---|---|---|
| `st.file_uploader` | Upload manga image | `st.file_uploader("...", type=["png","jpg"])` |
| `st.image` | Display image | `st.image(pil_img, caption="...")` |
| `st.spinner` | Loading indicator | `with st.spinner("..."):` |
| `st.checkbox` | Toggle sections | `if st.checkbox("Run Detection"):` |
| `st.text_input` | Editable text fields | `st.text_input("Label", value="...", key="...")` |
| `st.text_area` | Multi-line text | `st.text_area("Script", value="...", height=300)` |
| `st.button` | Action triggers | `if st.button("Generate"):` |
| `st.markdown` | Formatted text | `st.markdown("### Section")` |
| `st.write` | General output | `st.write(f"Result: {val}")` |
| `st.success` | Success message | `st.success(f"Found {n} dialogues")` |
| `st.warning` | Warning message | `st.warning("Please upload image first")` |
| `st.audio` | Audio player | `st.audio("output.mp3")` |

### Running

```bash
streamlit run streamlit_app.py

# Custom port
streamlit run streamlit_app.py --server.port 8502

# Headless (no browser auto-open)
streamlit run streamlit_app.py --server.headless true
```

### Configuration (`~/.streamlit/config.toml`)

```toml
[server]
port = 8501
headless = false
maxUploadSize = 200    # MB

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Caching Models (Critical for Performance)

```python
@st.cache_resource
def load_yolo_model():
    from ultralytics import YOLO
    return YOLO('yolov8n.pt')

@st.cache_resource
def load_ocr_reader():
    import easyocr
    return easyocr.Reader(['en'], gpu=False)

# These load once and persist across reruns
model = load_yolo_model()
reader = load_ocr_reader()
```

`@st.cache_resource` is for non-serializable objects (models, connections).
`@st.cache_data` is for data (DataFrames, images, computed results).

---

## Alternative: Gradio

### What is Gradio?

Gradio is another Python-to-web-app library, particularly popular in the HuggingFace ecosystem. It is designed specifically for ML demo interfaces.

```bash
pip install gradio
```

### Basic Gradio App

```python
import gradio as gr
from PIL import Image
from ocr_utils import extract_dialogues

def process_panel(image):
    # image is a numpy array from Gradio
    pil_img = Image.fromarray(image)
    pil_img.save("temp.jpg")
    dialogues = extract_dialogues("temp.jpg")
    script = "\n".join([f"• {d['text']} (conf: {d['conf']:.2f})" for d in dialogues])
    return script

demo = gr.Interface(
    fn=process_panel,
    inputs=gr.Image(label="Upload Manga Panel"),
    outputs=gr.Textbox(label="Extracted Dialogue"),
    title="ECHO-TOON",
    description="Upload a manga panel to extract dialogue."
)

demo.launch()
```

### Multi-Component Gradio Interface

```python
with gr.Blocks() as demo:
    gr.Markdown("# ECHO-TOON — Manga Narrator")

    with gr.Row():
        image_input = gr.Image(label="Manga Panel")
        ocr_output = gr.Textbox(label="OCR Results", lines=10)

    with gr.Row():
        script_output = gr.Textbox(label="Dia Script", lines=10)
        audio_output = gr.Audio(label="Narration")

    ocr_btn = gr.Button("1. Extract Dialogue")
    tts_btn = gr.Button("2. Generate Audio")

    ocr_btn.click(fn=run_ocr, inputs=image_input, outputs=ocr_output)
    tts_btn.click(fn=generate_audio, inputs=script_output, outputs=audio_output)

demo.launch()
```

### Gradio vs. Streamlit

| Feature | Streamlit | Gradio |
|---|---|---|
| Setup simplicity | Very easy | Very easy |
| ML demo focus | General | Specifically ML |
| HuggingFace Spaces | Supported | Native |
| State management | `session_state` | Component state |
| Custom layouts | Limited | `gr.Blocks` (flexible) |
| Audio output | `st.audio` | `gr.Audio` |
| Image display | `st.image` | `gr.Image` |
| Sharing demo | Manual deploy | `demo.launch(share=True)` |
| Best for | Multi-step pipelines | Single model demos |

---

## Alternative: Flask (Backend API)

For a production deployment separating frontend and backend:

```bash
pip install flask
```

### Flask Backend

```python
from flask import Flask, request, jsonify, send_file
from ocr_utils import extract_dialogues
from detection_utils import detect_characters
from character_mapper import map_dialogues_to_characters
import os

app = Flask(__name__)

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    file = request.files['image']
    path = f"temp/{file.filename}"
    file.save(path)
    dialogues = extract_dialogues(path)
    return jsonify(dialogues)

@app.route('/detect', methods=['POST'])
def detect_endpoint():
    file = request.files['image']
    path = f"temp/{file.filename}"
    file.save(path)
    chars = detect_characters(path)
    return jsonify(chars)

@app.route('/narrate', methods=['POST'])
def narrate_endpoint():
    data = request.json
    # Run full pipeline, return audio file path
    ...
    return send_file("output.mp3", mimetype="audio/mpeg")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### FastAPI Alternative (Async)

```bash
pip install fastapi uvicorn python-multipart
```

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

app = FastAPI()

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp.jpg", "wb") as f:
        f.write(contents)
    dialogues = extract_dialogues("temp.jpg")
    return {"dialogues": dialogues}

# Run with: uvicorn main:app --reload
```

FastAPI advantages:
- Async support (non-blocking I/O)
- Automatic OpenAPI docs at `/docs`
- Type validation via Pydantic

---

## Deployment Options

### Local (Development)

```bash
streamlit run streamlit_app.py
```

### HuggingFace Spaces (Free Hosting)

Gradio apps deploy directly to HuggingFace Spaces:

1. Create a Space at https://huggingface.co/spaces
2. Upload your code
3. Add a `requirements.txt`
4. HuggingFace provides CPU (free) or GPU (paid) compute

Note: Dia TTS requires GPU — CPU-only Spaces will be very slow.

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements_minimal.txt .
RUN pip install -r requirements_minimal.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.headless", "true", "--server.port", "8501"]
```

```bash
docker build -t echo-toon .
docker run -p 8501:8501 echo-toon
```

### ngrok (Expose Local to Internet)

```bash
pip install pyngrok
```

```python
from pyngrok import ngrok
import subprocess

process = subprocess.Popen(["streamlit", "run", "streamlit_app.py"])
tunnel = ngrok.connect(8501)
print(f"Public URL: {tunnel.public_url}")
```

---

## UI/UX Recommendations for Accessibility

Since ECHO-TOON serves visually impaired users, the UI itself should be accessible for sighted collaborators and caregivers:

1. **High contrast** — use dark text on light backgrounds
2. **Large text** — use `st.markdown("# Large Header")` for section titles
3. **Audio preview** — always show `st.audio()` after generation
4. **Progress indicators** — use `st.progress()` for long operations
5. **Clear error messages** — use `st.error()` not just `st.write()`
6. **Keyboard navigation** — Streamlit supports Tab-based navigation natively
