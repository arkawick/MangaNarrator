import cv2

# character_mapper.py
from scipy.spatial import distance

def get_center(box):
    (x1, y1), (x2, y2) = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def map_dialogues_to_characters(dialogues, characters):
    mapping = []

    for d in dialogues:
        d_center = get_center(d['box'])
        min_dist = float('inf')
        assigned_char = None

        for i, char_box in enumerate(characters):
            c_center = get_center(char_box)
            dist = distance.euclidean(d_center, c_center)
            if dist < min_dist:
                min_dist = dist
                assigned_char = i

        mapping.append({
            "character_id": assigned_char,
            "dialogue": d['text'],
            "conf": d['conf'],
            "dialogue_box": d['box'],
            "character_box": characters[assigned_char],
        })

    return mapping

def draw_mapping_lines(image, mapping):
    for item in mapping:
        d_center = get_center(item['dialogue_box'])
        c_center = get_center(item['character_box'])

        # Draw line
        cv2.line(image, tuple(map(int, d_center)), tuple(map(int, c_center)), (0, 255, 255), 2)

        # Annotate character ID
        cv2.putText(
            image, f"Char#{item['character_id']+1}", tuple(map(int, c_center)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

        # Annotate dialogue
        cv2.putText(
            image, f"\"{item['dialogue'][:20]}...\"", tuple(map(int, d_center)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1
        )

    return image
