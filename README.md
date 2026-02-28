# ECHO-TOON

### AI-Powered Manga / Webtoon Narrator for Visually Impaired Users

---

## What is ECHO-TOON?

ECHO-TOON is a fully local, AI-powered pipeline that converts static manga and webtoon panel images into structured, character-aware audio narration — built specifically for blind and visually impaired users.

Manga and webtoons are deeply visual mediums. Dialogue is embedded in images. Characters are identified by visual appearance. Panels have a specific reading order. None of this is accessible to traditional screen readers.

ECHO-TOON solves this by combining:

- **OCR** to extract dialogue text from speech bubbles
- **Object Detection (YOLOv8)** to locate characters in the panel
- **Dialogue-to-Character Mapping** to assign each line to its speaker
- **Scene Description (BLIP2/CLIP)** to narrate the visual environment
- **Multi-Speaker TTS (Dia)** to synthesize character-specific voices
- **Local LLM (optional)** to enhance raw dialogue into storytelling prose

---

## Full Pipeline

```
Manga Panel Image
        │
        ▼
┌──────────────────────┐
│  Image Preprocessing │  ← OpenCV
│  (denoise, enhance)  │
└────────┬─────────────┘
         │
         ▼
┌───────────────────┐
│  Speech Bubble    │  ← YOLOv8 (custom) or
│  Detection        │    EasyOCR bounding boxes
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  OCR              │  ← EasyOCR / Tesseract
│  Text Extraction  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Character        │  ← YOLOv8 (COCO pretrained)
│  Detection        │    class: person
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Dialogue ↔       │  ← Euclidean distance
│  Character Mapping│    between bubble and character centers
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Scene Description│  ← BLIP2 / CLIP
│  (optional)       │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Translation      │  ← langdetect + googletrans (optional)
│  (optional)       │    for Japanese / Korean manga
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  LLM Narration    │  ← Ollama + LLaMA3 (optional)
│  Enhancement      │    raw dialogue → storytelling prose
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  TTS Voice        │  ← Dia (nari-labs/Dia-1.6B)
│  Synthesis        │    multi-speaker, fully offline
└────────┬──────────┘
         │
         ▼
  Audio Narration Output (WAV/MP3)
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/echo-toon.git
cd echo-toon
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements_minimal.txt
```

### 4. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

---

## Project Structure

```
echo-toon/
│
├── streamlit_app.py         # Main Streamlit application (Steps 1–5)
├── streamlit_app2007.py     # Older prototype (BLIP + gTTS)
├── ocr_utils.py             # EasyOCR wrapper
├── detection_utils.py       # YOLOv8 character detector
├── character_mapper.py      # Dialogue-to-character distance mapping
├── utils.py                 # File upload and image loading helpers
├── z.py                     # Experimental Dia TTS Streamlit UI
├── a.py                     # CLI: OCR + translate + gTTS pipeline
├── yolov8n.pt               # YOLOv8 nano pretrained weights
│
├── Text_To_Speech_with_dia/
│   ├── elevenlabs_api.py    # ElevenLabs TTS demo
│   ├── GPU_test.py          # CUDA availability check
│   └── Manga_to_novel.ipynb # Colab: OCR → Gemini → gTTS
│
├── docs/                    # Detailed module documentation
│   ├── OCR.md
│   ├── CHARACTER_DETECTION.md
│   ├── DIALOGUE_MAPPING.md
│   ├── TTS.md
│   ├── SCENE_DESCRIPTION.md
│   ├── LOCAL_LLM.md
│   ├── RAG.md
│   ├── CUDA_SETUP.md
│   └── UI.md
│
├── Reference Projects/      # Reference implementations
├── temp_uploads/            # Temporary uploaded images
└── requirements.txt
```

---

## Module Documentation

| Module | Doc |
|---|---|
| OCR (EasyOCR / Tesseract) | [docs/OCR.md](docs/OCR.md) |
| Character Detection (YOLOv8) | [docs/CHARACTER_DETECTION.md](docs/CHARACTER_DETECTION.md) |
| Dialogue-to-Character Mapping | [docs/DIALOGUE_MAPPING.md](docs/DIALOGUE_MAPPING.md) |
| Text-to-Speech (Dia + alternatives) | [docs/TTS.md](docs/TTS.md) |
| Scene Description (BLIP2 / CLIP) | [docs/SCENE_DESCRIPTION.md](docs/SCENE_DESCRIPTION.md) |
| Local LLM Narration (Ollama / LLaMA3) | [docs/LOCAL_LLM.md](docs/LOCAL_LLM.md) |
| RAG Pipeline (FAISS + Sentence Transformers) | [docs/RAG.md](docs/RAG.md) |
| CUDA Setup & GPU Acceleration | [docs/CUDA_SETUP.md](docs/CUDA_SETUP.md) |
| UI Layer (Streamlit / Gradio / Flask) | [docs/UI.md](docs/UI.md) |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Vision / Image Processing | OpenCV |
| Speech Bubble + Character Detection | YOLOv8 (Ultralytics) |
| OCR | EasyOCR, Tesseract |
| Character Classification | ResNet50 / MobileNet (PyTorch) |
| Scene Description | BLIP2, CLIP |
| Translation | langdetect, googletrans / deep-translator |
| LLM Narration | LLaMA3 via Ollama (local) |
| Vector Store | FAISS |
| Embeddings | Sentence Transformers |
| TTS (primary) | Dia — nari-labs/Dia-1.6B |
| TTS (alternatives) | gTTS, pyttsx3, edge-tts, ElevenLabs, Coqui |
| UI | Streamlit |
| GPU Acceleration | PyTorch CUDA (cu118) |
| Language | Python 3.10+ |

---

## Sample Images

The repo includes three test manga panels:

- `Manga_Sample(Fate Zero).png`
- `Manga_Sample(Grill Our Love).jpg`
- `Manga_Sample(Parasyte).jpg`

---

## Roadmap

- [x] OCR dialogue extraction
- [x] YOLOv8 character detection
- [x] Dialogue-to-character mapping
- [x] Editable character naming
- [x] Dia-format script generation
- [ ] Dia TTS audio synthesis in main app
- [ ] BLIP2 scene description integration
- [ ] Ollama LLM narration enhancement
- [ ] Custom YOLO speech bubble detector
- [ ] Reading order detection (RTL manga support)
- [ ] Face recognition for cross-panel identity
- [ ] Emotion detection from expressions
- [ ] Sound effects recognition
- [ ] RAG pipeline for story context
- [ ] Docker containerization
- [ ] Mobile app deployment

---

## Impact

ECHO-TOON enables:

- Blind and visually impaired users to consume manga and webtoons independently
- AI-assisted storytelling that preserves character identity and voice
- Fully offline, privacy-preserving operation (no cloud required)
- Extensible architecture for other visual narrative mediums (comics, graphic novels)

---

## License

MIT License
