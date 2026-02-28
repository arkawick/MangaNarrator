# Character Detection

## Problem

After OCR extracts dialogue text with bounding boxes, we need to know **who is speaking**. In manga panels, a speech bubble is visually connected to a character through a **tail** (the pointed part of the bubble pointing at the speaker). Detecting this tail programmatically is complex. Instead, ECHO-TOON uses a simpler and effective approximation: **detect where characters are in the panel, then assign each dialogue to the nearest character**.

This requires:
1. Locating all character bounding boxes in the image
2. Computing proximity between dialogue bubbles and character bodies
3. Assigning each dialogue to the closest character

---

## YOLOv8 — Primary Detection Model

### What is YOLO?

YOLO (You Only Look Once) is a family of real-time object detection models. Unlike two-stage detectors (which first propose regions then classify them), YOLO processes the entire image in a single forward pass — making it extremely fast.

YOLOv8 is the current generation from Ultralytics.

Repository: https://github.com/ultralytics/ultralytics

### How YOLOv8 Works

```
Input Image (640×640)
        │
        ▼
Backbone (CSPDarknet / C2f blocks)
Feature extraction at multiple scales
        │
        ▼
Neck (PAN-FPN)
Multi-scale feature fusion
        │
        ▼
Head
Bounding box regression + class prediction
        │
        ▼
NMS (Non-Maximum Suppression)
Remove duplicate detections
        │
        ▼
Output: [x1, y1, x2, y2, confidence, class_id]
```

Key improvements in YOLOv8 over previous versions:
- Anchor-free detection head (no predefined anchor boxes)
- Decoupled head (separate branches for classification and regression)
- C2f blocks (more efficient than C3 in YOLOv5)

### Model Variants

| Model | Parameters | Speed (CPU) | Speed (GPU) | mAP |
|---|---|---|---|---|
| YOLOv8n (nano) | 3.2M | Fast | Very Fast | 37.3 |
| YOLOv8s (small) | 11.2M | Medium | Fast | 44.9 |
| YOLOv8m (medium) | 25.9M | Slow | Medium | 50.2 |
| YOLOv8l (large) | 43.7M | Very Slow | Slow | 52.9 |
| YOLOv8x (extra) | 68.2M | Very Slow | Slow | 53.9 |

ECHO-TOON uses **YOLOv8n** — smallest and fastest, sufficient for person detection in manga panels.

### COCO Dataset — Pretrained Weights

`yolov8n.pt` is pretrained on the COCO dataset (Common Objects in Context), which contains 80 object classes. We only use **class 0 = person**.

```python
# Class IDs in COCO
# 0: person
# 1: bicycle
# 2: car
# ... (80 total classes)
```

### Usage in ECHO-TOON (`detection_utils.py`)

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def detect_characters(image_path, conf_thres=0.3):
    results = model(image_path)[0]
    boxes = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == 0 and conf > conf_thres:   # person only
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf
            })
    return boxes
```

### Detection with GPU

```python
results = model('image.jpg', device='cuda')   # GPU
results = model('image.jpg', device='cpu')    # CPU
```

### Confidence Threshold

The default threshold of 0.3 means detections with less than 30% confidence are discarded. For manga art:
- 0.25–0.3: More detections, more false positives
- 0.4–0.5: Fewer detections, more misses on stylized characters
- 0.3 is a good balance for semi-realistic manga styles

---

## Manga-Specific Challenges

COCO-pretrained YOLOv8 was trained on photographs of real people. Manga characters have:

- Non-realistic proportions (large eyes, small mouths, exaggerated features)
- Flat coloring with no photorealistic shading
- Partial body visibility (head-only panels are common)
- Overlapping characters
- Non-human characters (fantasy, animal ears, robots)

Detection accuracy on manga varies significantly based on art style:
- Realistic styles (Berserk, Vinland Saga): Good detection
- Chibi / super-deformed: Poor detection
- Webtoon (full-color, realistic proportions): Good detection

---

## Custom YOLO Training for Manga (Planned)

To improve manga-specific character detection, a custom dataset can be trained:

### Step 1 — Data Collection

- Collect manga panel images
- Annotate with Label Studio or Roboflow

### Step 2 — Annotation Format (YOLO)

```
# label.txt (one file per image)
# class_id  x_center  y_center  width  height  (all normalized 0–1)
0 0.512 0.431 0.234 0.678
0 0.213 0.512 0.189 0.501
```

### Step 3 — Training

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')   # start from pretrained

model.train(
    data='manga_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cuda'
)
```

```yaml
# manga_dataset.yaml
path: ./datasets/manga
train: images/train
val: images/val
nc: 2
names: ['character', 'speech_bubble']
```

### Step 4 — Fine-tuning Strategy

- Use pretrained `yolov8n.pt` as starting point (transfer learning)
- Freeze backbone layers initially, train only head
- Unfreeze all layers for final fine-tuning
- Use data augmentation: flip, crop, brightness, mosaic

---

## Character Classification (Identity Assignment)

After detecting character bounding boxes, we need to identify **which character is which** (e.g., "Kiritsugu", "Irisviel"). This is a separate classification problem.

### Approach 1 — Transfer Learning (ResNet50 / MobileNet)

```python
import torchvision.models as models
import torch.nn as nn

# Load pretrained ResNet50
model = models.resnet50(pretrained=True)

# Replace final layer for N characters
num_characters = 5
model.fc = nn.Linear(model.fc.in_features, num_characters)

# Fine-tune on character crops
```

Requirements:
- Labeled crops of each character from multiple panels
- At least 50–100 examples per character for reliable classification

### Approach 2 — Face Recognition

Libraries like `face_recognition` (dlib-based) or `deepface` can be used to cluster recurring faces:

```python
import face_recognition

known_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encodings = face_recognition.face_encodings(panel_image)

for encoding in unknown_encodings:
    match = face_recognition.compare_faces([known_encoding], encoding)
```

Limitation: Manga face recognition requires a manga-specific model — standard face recognition models trained on photographs perform poorly.

### Approach 3 — Manual Naming (Current Implementation)

The simplest and most reliable method for now:
- Characters are assigned numeric IDs (Character #0, #1, ...)
- User manually types names in the Streamlit UI
- Names persist in `st.session_state`

---

## Alternative Detection Models

### Detectron2 (Facebook Research)

```python
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
predictor = DefaultPredictor(cfg)
outputs = predictor(image)
```

More accurate than YOLOv8 on some tasks, but:
- Slower inference
- Heavier installation
- Two-stage (slower real-time performance)

### RT-DETR (Real-Time Detection Transformer)

```python
model = YOLO('rtdetr-l.pt')   # Ultralytics supports RT-DETR
```

Transformer-based detector — better global context, competitive speed.

### MediaPipe (Google)

```python
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
results = pose.process(image_rgb)
```

Provides full body **pose landmarks** rather than bounding boxes. Can be used for body part localization.

---

## Detection Output Visualization

```python
from PIL import ImageDraw

def draw_character_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box["bbox"]
        draw.rectangle([(x1, y1), (x2, y2)], outline="blue", width=3)
        draw.text((x1, y1 - 15), f"Char #{i}", fill="blue")
    return image
```

---

## Comparison Table

| Model | Task | Speed | Manga Accuracy | Offline | Notes |
|---|---|---|---|---|---|
| YOLOv8n (COCO) | Person detection | Very Fast | Medium | Yes | Current implementation |
| YOLOv8n (custom) | Character detection | Very Fast | High | Yes | Requires labeled dataset |
| Detectron2 | Person detection | Slow | High | Yes | Heavy setup |
| RT-DETR | Object detection | Fast | Medium-High | Yes | Transformer-based |
| MediaPipe | Pose estimation | Very Fast | Medium | Yes | Landmarks, not boxes |
| Face recognition | Identity | Medium | Low (manga) | Yes | Needs manga-specific model |
