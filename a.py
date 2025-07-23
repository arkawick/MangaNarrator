import os
import easyocr
from langdetect import detect
from googletrans import Translator
from gtts import gTTS
from PIL import Image
import cv2
import tempfile
# import playsound  # Or use pydub

# Step 1: Initialize OCR, Translator
ocr_reader = easyocr.Reader(['ja', 'en'], gpu=True)
translator = Translator()

# Step 2: Process Image (Panel)
def process_panel(image_path):
    print(f"Processing: {image_path}")
    image = cv2.imread(image_path)

    # Step 3: OCR
    results = ocr_reader.readtext(image)

    for i, (bbox, text, conf) in enumerate(results):
        print(f"\nDetected Text #{i+1}: {text}")
        
        try:
            # Step 4: Language Detection
            lang = detect(text)
            print(f"Detected Language: {lang}")

            # Step 5: Translation if not English
            if lang != 'en':
                translated = translator.translate(text, src=lang, dest='en').text
            else:
                translated = text

            print(f"Translated: {translated}")

            # Step 6: Text-to-Speech
            speak_text(translated)

        except Exception as e:
            print(f"Error: {e}")

# Step 7: TTS Function
def speak_text(text, method="gtts"):
    if method == "gtts":
        try:
            tts = gTTS(text, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_path = fp.name
                tts.save(temp_path)
            # playsound.playsound(temp_path)
            os.remove(temp_path)
        except Exception as e:
            print(f"TTS Error: {e}")
    elif method == "pyttsx3":
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    elif method == "coqui":
        # Coqui TTS CLI usage or inference model
        print("Coqui TTS not implemented here (requires model setup)")
    else:
        print("Unknown TTS method.")

# Step 8: Run on Example Image
if __name__ == "__main__":
    sample_image = "sample.jpg"
    process_panel(sample_image)
