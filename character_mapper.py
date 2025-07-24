import numpy as np

def get_center(bbox):
    """Compute the center (x, y) of a bounding box."""
    if isinstance(bbox[0], list):  # OCR polygon format (EasyOCR), e.g. [[x1,y1],...]
        x = sum(p[0] for p in bbox) / 4
        y = sum(p[1] for p in bbox) / 4
    else:  # Rectangular bbox format: (x1, y1, x2, y2)
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
    return (x, y)

def euclidean_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def map_dialogues_to_characters(dialogues, char_boxes):
    mapping = []

    for dialogue in dialogues:
        d_center = get_center(dialogue['bbox'])  # 🔁 changed from 'box' to 'bbox'

        min_dist = float('inf')
        assigned_char_idx = -1

        for idx, char in enumerate(char_boxes):
            c_center = get_center(char['bbox'])  # 🔁 changed from 'box' to 'bbox'
            dist = euclidean_dist(d_center, c_center)

            if dist < min_dist:
                min_dist = dist
                assigned_char_idx = idx

        mapping.append({
            'character_id': assigned_char_idx,
            'text': dialogue['text'],
            'confidence': dialogue['conf']
        })

    return mapping
