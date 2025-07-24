import easyocr
import cv2
from PIL import Image, ImageDraw

reader = easyocr.Reader(['en'], gpu=False)

def extract_dialogues(image_path):
    image = cv2.imread(image_path)  # BGR image
    results = reader.readtext(image)  # Pass image object instead of path
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

def draw_dialogue_boxes(pil_image, dialogues):
    draw = ImageDraw.Draw(pil_image)
    for d in dialogues:
        x1, y1, x2, y2 = d["bbox"]
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        draw.text((x1, y1 - 10), d["text"], fill="red")
    return pil_image
