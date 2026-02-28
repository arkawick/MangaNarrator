# Local LLM — Narration Enhancement

## Purpose in ECHO-TOON

Raw OCR output is terse and stripped of context:

```
"We have to go."
"Where?"
"Anywhere."
```

An LLM can transform this into expressive, novel-style narration:

```
Kiritsugu spoke urgently, his voice barely above a whisper. "We have to go."
Irisviel turned to him, confusion etched across her face. "Where?"
He didn't meet her eyes. "Anywhere."
```

This is optional but significantly improves the listening experience, especially for complex story arcs.

---

## Ollama — Local LLM Runtime

### What is Ollama?

Ollama is a tool for running large language models locally. It handles model download, quantization, GPU offloading, and provides a simple REST API that mimics the OpenAI API format.

Website: https://ollama.com
Repository: https://github.com/ollama/ollama

### Installation (Windows)

1. Download from: https://ollama.com/download/windows
2. Run the installer
3. Ollama runs as a background service on `http://localhost:11434`

Verify:

```bash
ollama --version
```

### Pulling Models

```bash
# LLaMA 3 8B (recommended for narration quality/speed balance)
ollama pull llama3

# Smaller, faster
ollama pull llama3:8b-instruct-q4_0

# DeepSeek-R1 7B (strong reasoning)
ollama pull deepseek-r1:7b

# Mistral 7B (fast, good instruction following)
ollama pull mistral

# LLaVA (vision + language, for scene description)
ollama pull llava
```

### Checking Available Models

```bash
ollama list
```

### Running a Model

```bash
# Interactive chat
ollama run llama3

# Single prompt
echo "Describe this manga dialogue in novel style." | ollama run llama3
```

---

## LLaMA 3 — Recommended Model

### What is LLaMA 3?

LLaMA 3 (Large Language Model Meta AI 3) is Meta's open-weight language model series. The 8B parameter variant runs comfortably on consumer GPUs.

| Variant | Parameters | VRAM (Q4) | Quality |
|---|---|---|---|
| llama3:8b | 8B | ~4.5 GB | Good |
| llama3:70b | 70B | ~35 GB | Excellent |
| llama3.1:8b | 8B | ~4.5 GB | Better (long context) |
| llama3.2:3b | 3B | ~2 GB | Fast, lower quality |

For ECHO-TOON on a 6–8 GB VRAM card: `llama3:8b` or `llama3.2:3b`.

---

## Python Integration via Ollama API

### Using `requests`

```python
import requests
import json

def enhance_narration(raw_dialogue_lines, character_context=""):
    prompt = f"""You are a manga narrator. Convert the following character dialogues into expressive novel-style prose.
Keep it concise — 1-2 sentences of narration before each line, focusing on emotion and setting.
Do NOT add new plot points. Only enhance what is there.

{f"Character context: {character_context}" if character_context else ""}

Dialogues:
{chr(10).join([f'{line["char_name"]}: {line["text"]}' for line in raw_dialogue_lines])}

Output format:
[Narration sentence.]
Character Name: "Exact dialogue."
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 500
            }
        }
    )
    return response.json()["response"]
```

### Using `ollama` Python Package

```bash
pip install ollama
```

```python
import ollama

response = ollama.chat(
    model='llama3',
    messages=[
        {
            'role': 'system',
            'content': 'You are a manga narrator. Convert dialogue into expressive novel-style prose.'
        },
        {
            'role': 'user',
            'content': f"Enhance this manga dialogue:\n\n{raw_script}"
        }
    ]
)

enhanced = response['message']['content']
```

### Streaming Response (for Streamlit)

```python
import ollama
import streamlit as st

def stream_narration(prompt):
    with st.spinner("Enhancing narration..."):
        output_box = st.empty()
        full_text = ""

        for chunk in ollama.chat(
            model='llama3',
            messages=[{'role': 'user', 'content': prompt}],
            stream=True
        ):
            full_text += chunk['message']['content']
            output_box.markdown(full_text)

    return full_text
```

---

## Prompt Engineering for Manga Narration

### Narration Enhancement Prompt

```python
NARRATION_PROMPT = """You are a professional manga narrator writing for audio description for visually impaired listeners.

Given raw manga dialogue and speaker names, transform them into natural novel-style narration.

Rules:
1. Add 1 sentence of scene/emotion context before dialogue that needs it
2. Keep the exact dialogue words — do not paraphrase
3. Use character names in narration (not "Character A")
4. Add emotional tone based on context (e.g., "he whispered", "she shouted")
5. Keep it concise — this is for audio, not a novel
6. Do NOT invent new plot events

Input format:
Character Name: "dialogue text"

Output format:
[Optional: scene/emotion context sentence.]
Character Name said: "dialogue text."
"""
```

### Scene Description Prompt (for LLaVA)

```python
SCENE_PROMPT = """Describe this manga panel for a visually impaired listener.
Include:
1. Setting (indoor/outdoor, time of day, weather)
2. Number of characters and their general positions
3. Overall mood or tension
Keep it under 2 sentences. Be specific and concrete."""
```

### Character Context Prompt (RAG-assisted)

```python
CHARACTER_PROMPT = """Based on this character's known personality and story context:
{character_context}

How would {character_name} most likely say: "{dialogue}"?
Rewrite with appropriate emotional tone and speech style."""
```

---

## DeepSeek-R1 — Alternative with Chain-of-Thought

DeepSeek-R1 is a reasoning-focused model that uses chain-of-thought internally before producing its output. Useful for complex scene interpretation.

```bash
ollama pull deepseek-r1:7b
```

```python
response = ollama.chat(
    model='deepseek-r1:7b',
    messages=[{
        'role': 'user',
        'content': "Analyze this manga scene and describe the emotional context for a visually impaired listener: ..."
    }]
)
```

Note: DeepSeek-R1 outputs `<think>` tags with its reasoning process. Strip these from the final output:

```python
import re
clean_output = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
```

---

## Mistral 7B — Fast Alternative

```bash
ollama pull mistral
```

Mistral 7B is faster than LLaMA 3 8B with similar quality on creative writing tasks. Good choice for lower VRAM systems.

---

## LLM Comparison Table

| Model | Parameters | VRAM (Q4) | Speed | Quality | Best For |
|---|---|---|---|---|---|
| LLaMA 3 8B | 8B | ~4.5 GB | Medium | Good | Narration enhancement |
| LLaMA 3.2 3B | 3B | ~2 GB | Fast | Medium | Low-VRAM systems |
| Mistral 7B | 7B | ~4 GB | Fast | Good | Speed-quality balance |
| DeepSeek-R1 7B | 7B | ~4.5 GB | Slow | Very Good | Complex reasoning |
| LLaVA 7B | 7B | ~4.5 GB | Medium | Good | Vision + language |
| LLaMA 3 70B | 70B | ~35 GB | Very Slow | Excellent | High-end systems |

---

## Hardware Requirements

| Setup | VRAM | RAM | Expected Speed |
|---|---|---|---|
| LLaMA 3 8B (GPU) | 5 GB | 16 GB | ~20 tokens/sec |
| LLaMA 3 8B (CPU) | 0 | 16 GB | ~2 tokens/sec |
| LLaMA 3 70B (GPU) | 40 GB | 32 GB | ~5 tokens/sec |
| Mistral 7B (GPU) | 4.5 GB | 16 GB | ~25 tokens/sec |

For a typical 6 GB VRAM card (e.g., RTX 3060): run Dia TTS and LLM **sequentially** — unload one model before loading the other.

---

## Checking if Ollama is Running

```python
import requests

def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except:
        return False

if not check_ollama():
    st.warning("Ollama is not running. Start it with: ollama serve")
```

---

## Full Narration Enhancement Flow

```
Dialogue Lines (from mapping step)
        │
        ▼
LLM Prompt Construction
(character names + raw dialogue + instructions)
        │
        ▼
Ollama API Call (local, ~localhost:11434)
        │
        ▼
Enhanced Narration Text
(prose with emotional context)
        │
        ▼
Dia TTS Script Generation
([S1] enhanced line [S2] enhanced line)
        │
        ▼
Audio Output
```
