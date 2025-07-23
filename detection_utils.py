from ultralytics import YOLO
from PIL import Image, ImageDraw

# Load pre-trained YOLOv8 model (person class only)
model = YOLO('yolov8n.pt')  # COCO-pretrained

def detect_characters(image_path, conf_thres=0.3):
    results = model(image_path)[0]
    boxes = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == 0 and conf > conf_thres:  # Only detect 'person'
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf
            })
    return boxes

def draw_character_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        draw.rectangle([(x1, y1), (x2, y2)], outline="blue", width=3)
    return image
