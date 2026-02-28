# Scene Description — BLIP2 & CLIP

## Purpose in ECHO-TOON

Manga panels contain far more than dialogue. The visual environment — the rain outside a window, the tense standoff in a corridor, the peaceful countryside — is part of the story. For visually impaired users, this context is invisible.

Scene description bridges this gap by generating a natural language caption of the visual content of each panel, which is then incorporated into the audio narration:

> *"A dimly lit alley at night. Two silhouettes face each other in the rain."*
> Character A: "Hand it over."
> Character B: "I'd rather die."

This provides environmental context that makes the audio experience feel complete rather than just a list of disembodied dialogue lines.

---

## BLIP2 — Primary Model

### What is BLIP2?

BLIP2 (Bootstrapping Language-Image Pre-training 2) is a vision-language model from Salesforce Research. It connects a frozen image encoder (ViT-G) with a frozen large language model (OPT or FlanT5) using a lightweight trainable bridge called Q-Former.

Paper: https://arxiv.org/abs/2301.12597
Model card: https://huggingface.co/Salesforce/blip2-opt-2.7b

### Architecture

```
Input Image
    │
    ▼
Frozen Vision Encoder (ViT-G/14 from EVA-CLIP)
Visual features
    │
    ▼
Q-Former (Querying Transformer)
32 learned query tokens attend to visual features
Lightweight bridge — only trained component
    │
    ▼
Frozen LLM (OPT-2.7B or FlanT5-XL)
Language generation
    │
    ▼
Text Caption
```

### Why Q-Former?

The Q-Former acts as an information bottleneck — it compresses visual information from the ViT into 32 fixed-length tokens that the LLM can understand. This allows the vision and language components to remain frozen (no retraining needed), reducing compute requirements significantly.

### Installation

```bash
pip install transformers torch pillow
```

### Usage

```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16   # Use float16 to save VRAM
).to(device)

image = Image.open("manga_panel.jpg").convert("RGB")

# Unconditional captioning
inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=50)
caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

print(caption)
# "a man in a suit standing in the rain holding an umbrella"
```

### Conditional / VQA Mode (Visual Question Answering)

BLIP2 can answer questions about the image rather than just describe it:

```python
question = "What is the setting of this scene?"
inputs = processor(images=image, text=question, return_tensors="pt").to(device, torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=30)
answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(answer)
# "a dark alley at night"
```

Useful prompts for manga:
- `"Describe the scene in this manga panel."`
- `"What time of day is it?"`
- `"What emotion does this character show?"`
- `"How many characters are in this panel?"`
- `"What is the background setting?"`

### BLIP2 Model Variants

| Model | LLM Backbone | Size | VRAM |
|---|---|---|---|
| blip2-opt-2.7b | OPT-2.7B | ~5 GB | ~6 GB |
| blip2-opt-6.7b | OPT-6.7B | ~12 GB | ~14 GB |
| blip2-flan-t5-xl | FlanT5-XL | ~8 GB | ~9 GB |
| blip2-flan-t5-xxl | FlanT5-XXL | ~18 GB | ~20 GB |

For ECHO-TOON, `blip2-opt-2.7b` is the recommended balance of quality and VRAM.

### BLIP (Original — Used in `streamlit_app2007.py`)

The original BLIP model (not BLIP2) is lighter:

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

inputs = processor(images=image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
```

BLIP (base): ~1 GB VRAM — much lighter, lower quality.

---

## CLIP — Alternative

### What is CLIP?

CLIP (Contrastive Language-Image Pre-Training) from OpenAI learns joint embeddings of images and text by training on 400 million image-text pairs. Rather than generating captions, CLIP computes similarity scores between an image and a set of candidate text descriptions.

Paper: https://arxiv.org/abs/2103.00020

### Architecture

```
Image → Vision Encoder (ViT or ResNet) → Image Embedding
Text  → Text Encoder (Transformer)      → Text Embedding

Similarity = cosine(Image Embedding, Text Embedding)
```

### Usage for Scene Classification

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("panel.jpg")

# Candidate scene descriptions
scene_candidates = [
    "a fight scene",
    "a peaceful countryside",
    "a rainy city street",
    "an indoor conversation",
    "a dramatic confrontation",
    "a flashback memory scene"
]

inputs = processor(
    text=scene_candidates,
    images=image,
    return_tensors="pt",
    padding=True
)

outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)

best_idx = probs.argmax()
print(f"Scene type: {scene_candidates[best_idx]} ({probs[0][best_idx]:.2%} confidence)")
```

### CLIP for Emotion Detection

```python
emotion_candidates = [
    "a character showing anger",
    "a character showing sadness",
    "a character showing joy",
    "a character showing fear",
    "a character showing surprise",
    "a character showing determination"
]

# Run same pipeline with emotion candidates
```

### CLIP vs. BLIP2

| Aspect | CLIP | BLIP2 |
|---|---|---|
| Output | Similarity score (classification) | Generated text (captioning) |
| Flexibility | Limited to candidate list | Open-ended generation |
| Speed | Very fast | Moderate |
| VRAM | ~1 GB | ~6 GB |
| Best for | Scene type classification | Scene description |

---

## LLaVA — Alternative for Instruction-Following

LLaVA (Large Language and Vision Assistant) is a multimodal LLM that can follow natural language instructions about images:

```python
# Using Ollama (once LLaVA is pulled)
import requests

response = requests.post("http://localhost:11434/api/generate", json={
    "model": "llava",
    "prompt": "Describe this manga panel scene for a visually impaired user. Focus on the setting, mood, and number of characters.",
    "images": [base64_image_string]
})
```

### Pulling LLaVA via Ollama

```bash
ollama pull llava
ollama run llava
```

LLaVA can produce significantly richer, more contextual descriptions than BLIP2 because it runs a full instruction-following LLM on top of visual features.

---

## InstructBLIP — Instruction-Tuned BLIP2

InstructBLIP adds instruction tuning to BLIP2, making it responsive to specific prompts:

```python
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

model = InstructBlipForConditionalGeneration.from_pretrained(
    "Salesforce/instructblip-vicuna-7b"
)
processor = InstructBlipProcessor.from_pretrained(
    "Salesforce/instructblip-vicuna-7b"
)

prompt = "Describe this manga panel for a blind reader. Include setting, mood, and character positions."
inputs = processor(images=image, text=prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=100)
description = processor.decode(output[0], skip_special_tokens=True)
```

---

## Integration into ECHO-TOON Narration

The scene description slot in the narration pipeline:

```
[SCENE] A dimly lit underground chamber. Two figures stand across from each other.
[S1] You've made a grave mistake coming here.
[S2] (breathes heavily) I had no choice.
[S1] There's always a choice.
```

The `[SCENE]` tag is not passed to Dia (which handles dialogue only) but is voiced separately by a neutral narrator voice (e.g., pyttsx3 or gTTS for the narration passages).

```python
def build_full_narration(scene_caption, dialogue_lines):
    narration = f"Scene: {scene_caption}\n\n"
    for line in dialogue_lines:
        narration += f"{line['char_name']} says: {line['text']}\n"
    return narration
```

---

## Comparison Table

| Model | Type | Output | VRAM | Offline | Best For |
|---|---|---|---|---|---|
| BLIP (base) | Captioning | Text | ~1 GB | Yes | Fast, lightweight |
| BLIP2-OPT-2.7B | VQA + Captioning | Text | ~6 GB | Yes | Quality captioning |
| InstructBLIP | Instruction | Text | ~8 GB | Yes | Guided description |
| CLIP | Classification | Score | ~1 GB | Yes | Scene type tagging |
| LLaVA (Ollama) | Instruction | Text | ~4 GB | Yes | Rich, contextual |
| GPT-4o Vision | Instruction | Text | N/A | No | Best quality (cloud) |
