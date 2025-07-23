# streamlit_app.py
import streamlit as st
from ocr_utils import extract_dialogues, draw_dialogue_boxes
from utils import save_uploaded_file, load_image
from detection_utils import detect_characters, draw_character_boxes

from character_mapper import map_dialogues_to_characters

st.set_page_config(page_title="Webtoon OCR Viewer", layout="wide")

st.title("🖼️ Webtoon Dialogue Extractor - Step 1: OCR")

uploaded = st.file_uploader("Upload a Webtoon Frame", type=["png", "jpg", "jpeg"])
if uploaded:
    file_path = save_uploaded_file(uploaded)
    image = load_image(file_path)

    st.image(image, caption="Original Webtoon Frame", use_column_width=True)

    with st.spinner("Extracting dialogues..."):
        dialogues = extract_dialogues(file_path)

    st.success(f"Found {len(dialogues)} dialogue(s)")

    image_with_boxes = draw_dialogue_boxes(image, dialogues)
    st.image(image_with_boxes, caption="OCR Result", use_column_width=True)

    if st.checkbox("Show extracted text"):
        for idx, d in enumerate(dialogues):
            st.markdown(f"**#{idx+1}:** {d['text']} (Conf: {d['conf']:.2f})")

if st.checkbox("Run Full-Body Character Detection"):
    with st.spinner("Detecting characters..."):
        char_boxes = detect_characters(file_path)
        st.success(f"Detected {len(char_boxes)} character(s)")
        image_with_chars = draw_character_boxes(image.copy(), char_boxes)
        st.image(image_with_chars, caption="Character Detection", use_column_width=True)


# After both dialogues and character detection are done
if 'dialogues' in locals() and 'char_boxes' in locals():
    st.subheader("🧭 Step 3: Map Dialogues to Characters")

    with st.spinner("Mapping dialogues to characters..."):
        mapping_result = map_dialogues_to_characters(dialogues, char_boxes)

        from character_mapper import draw_mapping_lines
        import cv2
        from PIL import Image

        vis_image = draw_mapping_lines(image.copy(), mapping_result)
        st.image(vis_image, caption="Mapped Dialogues to Characters", use_column_width=True)


    for item in mapping_result:
        st.markdown(
            f"**Character #{item['character_id'] + 1}** says: _{item['dialogue']}_ (conf: {item['conf']:.2f})"
        )

