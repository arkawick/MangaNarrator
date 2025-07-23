import easyocr
import cv2
from PIL import Image, ImageDraw

reader = easyocr.Reader(['en', 'ja'], gpu=False)

def extract_dialogues(image_path):
    img = cv2.imread(image_path)  # Load image as BGR
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    results = reader.readtext(img)  # Pass NumPy array, not path
    
    dialogues = []
    for (bbox, text, conf) in results:
        if conf > 0.4:
            dialogues.append({
                'box': bbox,
                'text': text,
                'conf': conf
            })
    return dialogues


def draw_dialogue_boxes(pil_image, dialogues):
    draw = ImageDraw.Draw(pil_image)
    for d in dialogues:
        x1, y1, x2, y2 = d["box"]
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        draw.text((x1, y1 - 10), d["text"], fill="red")
    return pil_image

