# OCR — Optical Character Recognition

## What is OCR?

OCR (Optical Character Recognition) converts text that is embedded inside images into machine-readable strings. In manga and webtoons, all dialogue is rendered as part of the image — there is no underlying text layer. OCR is the only way to extract it programmatically.

Manga OCR is harder than standard document OCR because:

- Text is inside speech bubbles with curved edges
- Font styles vary (bold, italic, handwritten, stylized)
- Text may be vertical (Japanese manga) or horizontal (webtoons)
- Sound effects and onomatopoeia are mixed with dialogue
- Low contrast between bubble background and text
- Text overlaps character art at bubble borders

---

## How OCR Works (General Pipeline)

```
Raw Image
    │
    ▼
Preprocessing
(grayscale, denoise, threshold, deskew)
    │
    ▼
Region Detection
(find text blocks / lines)
    │
    ▼
Feature Extraction
(CNN-based or classical)
    │
    ▼
Sequence Recognition
(RNN / LSTM / Transformer)
    │
    ▼
Text Output + Confidence Score
```

Modern OCR uses a two-stage pipeline:
1. **Detection** — find where text is in the image (bounding boxes)
2. **Recognition** — read what the text says inside each box

---

## EasyOCR (Primary — Used in ECHO-TOON)

### Overview

EasyOCR is an open-source, deep learning-based OCR library built on PyTorch. It supports 80+ languages and works well on stylized text common in manga.

Repository: https://github.com/JaidedAI/EasyOCR

### Architecture

| Stage | Model |
|---|---|
| Detection | CRAFT (Character Region Awareness for Text Detection) |
| Recognition | CRNN (Convolutional Recurrent Neural Network) |

**CRAFT** detects individual character regions and groups them into word/line bounding boxes using affinity scoring.

**CRNN** reads the detected region using CNN (feature extraction) + BiLSTM (sequence modeling) + CTC decoder.

### Installation

```bash
pip install easyocr
```

### Usage in ECHO-TOON

```python
import easyocr

# English only, CPU mode
reader = easyocr.Reader(['en'], gpu=False)

# English + Japanese, GPU mode
reader = easyocr.Reader(['ja', 'en'], gpu=True)

results = reader.readtext('image.jpg')

for (bbox, text, confidence) in results:
    print(f"Text: {text} | Confidence: {confidence:.2f}")
    print(f"Bounding box: {bbox}")
```

Return format:

```python
[
  ([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], "Hello", 0.97),
  ...
]
```

The bounding box is a 4-point polygon (not a simple rectangle), allowing rotated text detection.

### How EasyOCR is used in `ocr_utils.py`

```python
reader = easyocr.Reader(['en'], gpu=False)

def extract_dialogues(image_path):
    image = cv2.imread(image_path)
    results = reader.readtext(image)
    dialogues = []
    for res in results:
        (tl, tr, br, bl), text, conf = res
        x1, y1 = map(int, tl)
        x2, y2 = map(int, br)
        dialogues.append({
            "bbox": (x1, y1, x2, y2),
            "text": text,
            "conf": conf
        })
    return dialogues
```

Note: only top-left and bottom-right corners are used to form a simple rectangle bbox for distance calculations downstream.

### Language Codes

| Language | Code |
|---|---|
| English | `en` |
| Japanese | `ja` |
| Korean | `ko` |
| Chinese Simplified | `ch_sim` |
| Chinese Traditional | `ch_tra` |

### Pros

- Easy to install and use
- No Tesseract binaries required
- Deep learning based — handles stylized fonts well
- Polygon bounding boxes for rotated text
- GPU acceleration via PyTorch

### Cons

- Slower than Tesseract on CPU
- Less accurate for pure vertical Japanese text without fine-tuning
- Does not differentiate dialogue from sound effects

---

## Tesseract OCR (Alternative)

### Overview

Tesseract is Google's open-source OCR engine — one of the most mature OCR tools available. It uses an LSTM-based recognition pipeline (Tesseract 4+).

Repository: https://github.com/tesseract-ocr/tesseract

### Installation

```bash
# Install Tesseract binary (Windows)
# Download from: https://github.com/UB-Mannheim/tesseract/wiki

# Python wrapper
pip install pytesseract
```

```python
import pytesseract
from PIL import Image

# Point to Tesseract binary (Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = Image.open("panel.jpg")
text = pytesseract.image_to_string(img, lang='eng')
```

### Getting Bounding Boxes

```python
data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
for i, text in enumerate(data['text']):
    if text.strip():
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        conf = data['conf'][i]
```

### Language Packs

```bash
# Install Japanese language pack
# Download jpn.traineddata from:
# https://github.com/tesseract-ocr/tessdata
# Place in: C:\Program Files\Tesseract-OCR\tessdata\
```

### Pros

- Very fast on CPU
- Excellent for clean, printed text
- Large language pack ecosystem
- Well documented

### Cons

- Requires binary installation (not pure Python)
- Struggles with stylized manga fonts
- No built-in polygon detection
- Requires preprocessing for best results

---

## Google Cloud Vision API (Cloud Alternative)

```python
from google.cloud import vision

client = vision.ImageAnnotatorClient()
with open("panel.jpg", "rb") as f:
    image = vision.Image(content=f.read())

response = client.text_detection(image=image)
for text in response.text_annotations:
    print(text.description, text.bounding_poly)
```

### Pros

- Extremely accurate
- Handles vertical Japanese text natively
- No local model needed

### Cons

- Requires internet + API key
- Paid service (free tier limited)
- Privacy: images sent to cloud

---

## manga-ocr (Manga-Specific Alternative)

A specialized OCR model fine-tuned specifically for Japanese manga text, based on TrOCR (Transformer OCR).

```bash
pip install manga-ocr
```

```python
from manga_ocr import MangaOcr
mocr = MangaOcr()
text = mocr('panel.jpg')
```

Best choice for **Japanese manga** with vertical text and stylized fonts.

---

## OCR Comparison Table

| Tool | Language | Accuracy (Manga) | Speed (CPU) | Speed (GPU) | Offline | Cost |
|---|---|---|---|---|---|---|
| EasyOCR | 80+ langs | Good | Medium | Fast | Yes | Free |
| Tesseract | 100+ langs | Medium | Fast | No | Yes | Free |
| manga-ocr | Japanese only | Excellent | Medium | Fast | Yes | Free |
| Google Vision | 50+ langs | Excellent | N/A | N/A | No | Paid |
| ElevenLabs | N/A | N/A | N/A | N/A | N/A | N/A |

---

## Preprocessing for Better OCR Results

```python
import cv2
import numpy as np

def preprocess_for_ocr(image_path):
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold (handles variable lighting)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Dilation to connect broken characters
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    return dilated
```

---

## Known Limitations in ECHO-TOON

1. **Sound effects are OCR'd as dialogue** — no mechanism to filter onomatopoeia (e.g., "BOOM", "CRASH")
2. **Low confidence results are kept** — no threshold filtering currently applied
3. **Reading order not enforced** — EasyOCR returns boxes in detection order, not panel reading order
4. **Vertical Japanese text** — current setup uses English-only reader

### Recommended Improvements

- Filter by confidence threshold (`conf > 0.5`)
- Sort bounding boxes top-to-bottom, left-to-right for reading order
- Use manga-ocr for Japanese input
- Use a separate classifier to filter SFX from dialogue
