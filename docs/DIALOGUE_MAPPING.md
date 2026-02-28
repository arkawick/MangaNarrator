# Dialogue-to-Character Mapping

## Problem

OCR gives us a list of dialogue texts with their bounding boxes (where on the page the text is).
Character detection gives us a list of character bounding boxes (where characters are on the page).

The question is: **which character is speaking each line?**

In printed manga, this is communicated visually through the **speech bubble tail** — the pointed extension of the bubble that points toward the speaker's mouth or face. Automatically detecting and tracing this tail is computationally complex.

ECHO-TOON uses a practical approximation: **assign each dialogue to the nearest character based on Euclidean distance between their centers**.

---

## Current Implementation — Euclidean Distance

### Concept

For each dialogue bubble, compute its center point. For each detected character, compute their center point. The dialogue is assigned to whichever character's center is closest.

```
Panel:

  [Character A]        [Bubble: "Hello!"]        [Character B]
  center: (120, 200)   center: (300, 150)         center: (500, 200)

  Distance A → Bubble: sqrt((300-120)² + (150-200)²) = sqrt(32400 + 2500) = 186.7
  Distance B → Bubble: sqrt((300-500)² + (150-200)²) = sqrt(40000 + 2500) = 206.2

  → Bubble assigned to Character A (closer)
```

### Implementation (`character_mapper.py`)

```python
import numpy as np

def get_center(bbox):
    if isinstance(bbox[0], list):   # EasyOCR polygon format [[x1,y1],...]
        x = sum(p[0] for p in bbox) / 4
        y = sum(p[1] for p in bbox) / 4
    else:                           # Rectangle format (x1, y1, x2, y2)
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
    return (x, y)

def euclidean_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def map_dialogues_to_characters(dialogues, char_boxes):
    mapping = []
    for dialogue in dialogues:
        d_center = get_center(dialogue['bbox'])
        min_dist = float('inf')
        assigned_char_idx = -1

        for idx, char in enumerate(char_boxes):
            c_center = get_center(char['bbox'])
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
```

### Output

```python
[
    {'character_id': 0, 'text': "We have to leave now!", 'confidence': 0.94},
    {'character_id': 1, 'text': "But where will we go?", 'confidence': 0.91},
    {'character_id': 0, 'text': "Anywhere but here.",   'confidence': 0.88},
]
```

---

## Bbox Format Handling

The `get_center()` function handles two different bounding box formats that appear in the pipeline:

| Source | Format | Example |
|---|---|---|
| EasyOCR (raw) | 4-point polygon | `[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]` |
| ECHO-TOON (processed) | Rectangle tuple | `(x1, y1, x2, y2)` |
| YOLOv8 | Rectangle | `(x1, y1, x2, y2)` |

The polygon format check `isinstance(bbox[0], list)` distinguishes between the two.

---

## Limitations of Euclidean Distance

### 1. Bubbles far from all characters

In wide establishing shots, a narration box (no tail, used for scene description) may be assigned to whichever character happens to be closest — incorrectly.

### 2. Multiple characters equidistant

When two characters stand symmetrically and a bubble is between them, the assignment is ambiguous.

### 3. Off-panel speakers

A character speaking from off-panel has no bounding box — the dialogue gets assigned to whoever is on screen.

### 4. Thought bubbles

Thought bubbles should be assigned to the character thinking, not the nearest character. They look different visually but are treated identically by the distance method.

### 5. Large panels with many characters

In crowd scenes, distance alone is unreliable.

---

## Alternative Mapping Strategies

### 1. Bubble Tail Detection

The most accurate approach — detect the tail of the speech bubble and trace it to the nearest character body part (mouth region).

```
Speech bubble tail → direction vector → intersect with character bbox → assign
```

Implementation options:
- Classical: edge detection (Canny) + Hough line transform to find tail direction
- Deep learning: custom segmentation model trained to segment bubble vs. tail vs. background

This is complex but the most manga-faithful approach.

### 2. Overlap / Proximity to Bubble Border

Instead of center-to-center distance, measure the distance from the **nearest edge of the bubble** to the **nearest edge of the character bbox**:

```python
def bbox_edge_distance(bubble_bbox, char_bbox):
    bx1, by1, bx2, by2 = bubble_bbox
    cx1, cy1, cx2, cy2 = char_bbox

    # Horizontal gap
    dx = max(0, max(bx1 - cx2, cx1 - bx2))
    # Vertical gap
    dy = max(0, max(by1 - cy2, cy1 - by2))

    return np.sqrt(dx**2 + dy**2)
```

This handles cases where a bubble overlaps a character's body better than center-to-center.

### 3. Quadrant / Direction Analysis

Divide the panel into quadrants. Assign dialogue in the left half to the leftmost character, right half to rightmost, etc.:

```python
panel_width = image.width
bubble_cx = (bubble_bbox[0] + bubble_bbox[2]) / 2

if bubble_cx < panel_width / 2:
    # Assign to leftmost character in left half
else:
    # Assign to rightmost character in right half
```

Simple and fast — works well for two-character conversations.

### 4. Reading Order Interleaving

In manga, dialogue is read in order (usually top to bottom). If two characters are speaking alternately, odd-numbered bubbles go to Character A and even-numbered to Character B. This is a heuristic that works for dialogue exchanges.

### 5. LLM-Assisted Assignment

Pass the full list of dialogue and the spatial layout to an LLM with a structured prompt:

```
Panel has 2 characters: Character at left (100–250px), Character at right (400–550px).
Dialogue bubbles (by x-center): Bubble1 at 180px, Bubble2 at 470px, Bubble3 at 200px.
Assign each bubble to a character.
```

An LLM can reason about context ("the person who just answered would logically reply") to make smarter assignments than pure geometry.

---

## Reading Order Detection

OCR extracts bubbles in detection order — not necessarily manga reading order. Manga is typically read:

- Japanese manga: **right to left, top to bottom**
- Webtoons / Western comics: **left to right, top to bottom**

### Sorting by Reading Order

```python
def sort_dialogues_reading_order(dialogues, rtl=False):
    # Sort top-to-bottom first (by y1 of bbox)
    # Then left-to-right or right-to-left within same row
    row_threshold = 50   # bubbles within 50px vertically are "same row"

    sorted_d = sorted(dialogues, key=lambda d: d['bbox'][1])   # sort by y1

    # Group into rows
    rows = []
    current_row = [sorted_d[0]]
    for d in sorted_d[1:]:
        if d['bbox'][1] - current_row[-1]['bbox'][1] < row_threshold:
            current_row.append(d)
        else:
            rows.append(current_row)
            current_row = [d]
    rows.append(current_row)

    # Sort within each row
    result = []
    for row in rows:
        if rtl:
            row.sort(key=lambda d: -d['bbox'][0])   # right to left
        else:
            row.sort(key=lambda d: d['bbox'][0])     # left to right
        result.extend(row)

    return result
```

---

## Confidence Scoring

The current implementation passes through OCR confidence as `dialogue['conf']`. A more comprehensive confidence score for the **mapping** step would factor in:

- OCR confidence
- Distance (closer = more confident)
- Whether the bubble overlaps the character bbox

```python
def mapping_confidence(ocr_conf, distance, max_panel_distance):
    distance_score = 1.0 - (distance / max_panel_distance)
    return (ocr_conf + distance_score) / 2
```

---

## Manual Override

The current Streamlit UI allows the user to:
- Edit the character name assigned to each dialogue
- Edit the dialogue text itself (Step 4.5)

This human-in-the-loop correction handles all the edge cases that the automatic mapping fails on.

---

## Summary

| Method | Accuracy | Complexity | Current Use |
|---|---|---|---|
| Euclidean distance (center-center) | Medium | Low | Yes |
| Edge-to-edge distance | Medium-High | Low | No |
| Quadrant heuristic | Medium | Low | No |
| Bubble tail detection | High | High | No (planned) |
| LLM-assisted | High | Medium | No (planned) |
| Manual override | Perfect | N/A | Yes (UI) |
