import streamlit as st
from ocr_utils import extract_dialogues, draw_dialogue_boxes
from utils import save_uploaded_file, load_image
from detection_utils import detect_characters, draw_character_boxes



st.set_page_config(page_title="Webtoon OCR Viewer", layout="wide")

st.title("🖼️ Webtoon Dialogue Extractor - Step 1: OCR")

uploaded = st.file_uploader("Upload a Webtoon Frame", type=["png", "jpg", "jpeg"])


dialogues = None
char_boxes = None

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


# In the main block after OCR:
if st.checkbox("Run Full-Body Character Detection"):
    with st.spinner("Detecting characters..."):
        char_boxes = detect_characters(file_path)
        st.success(f"Detected {len(char_boxes)} character(s)")
        image_with_chars = draw_character_boxes(image.copy(), char_boxes)
        st.image(image_with_chars, caption="Character Detection", use_column_width=True)

from character_mapper import map_dialogues_to_characters

if dialogues and char_boxes:
    st.markdown("### 🧭 Step 3: Dialogue-to-Character Mapping")
    mapping_result = map_dialogues_to_characters(dialogues, char_boxes)

    for item in mapping_result:
        st.write(f"🗣️ Character #{item['character_id']} says: **{item['text']}** (Conf: {item['confidence']:.2f})")
else:
    st.warning("Please run both dialogue and character detection before mapping.")


# Character Naming (Step 4)
if dialogues and char_boxes:
    st.markdown("### 🏷️ Step 4: Assign Names to Characters")

    # Initialize session state to persist character names
    if "char_names" not in st.session_state:
        st.session_state.char_names = {}

    for idx, char in enumerate(char_boxes):
        default_name = st.session_state.char_names.get(idx, f"Character {idx}")
        name = st.text_input(f"Enter name for Character #{idx}:", value=default_name, key=f"char_name_{idx}")
        st.session_state.char_names[idx] = name

    st.markdown("---")
    st.markdown("### 🗣️ Final Mapping with Character Names")

    for item in mapping_result:
        char_id = item['character_id']
        name = st.session_state.char_names.get(char_id, f"Character {char_id}")
        st.write(f"🗣️ **{name}** says: *{item['text']}* (Conf: {item['confidence']:.2f})")




# st.markdown("### 📝 Step 4.5: Edit Assigned Dialogues")

# # Initialize a session state to store editable dialogues
# if "edited_dialogues" not in st.session_state:
#     st.session_state.edited_dialogues = {}

# edited_dialogues = []

# for idx, item in enumerate(mapping_result):
#     char_id = item['character_id']
#     default_name = character_names.get(char_id, f"Character #{char_id}")
#     original_text = item['text']

#     # Generate unique key for Streamlit text input
#     input_key = f"edit_text_{idx}"

#     # Default value from session state or original
#     default_value = st.session_state.edited_dialogues.get(input_key, original_text)

#     # Render editable field
#     new_text = st.text_input(
#         f"{default_name} says:",
#         value=default_value,
#         key=input_key
#     )

#     # Update session state and tracking list
#     st.session_state.edited_dialogues[input_key] = new_text

#     edited_dialogues.append({
#         "character_name": default_name,
#         "original_text": original_text,
#         "edited_text": new_text,
#         "confidence": item["confidence"]
#     })

# # Optional: Show the final edited results
# if st.button("Show Final Narration Lines"):
#     st.markdown("### 🔊 Final Dialogue List:")
#     for entry in edited_dialogues:
#         st.markdown(f"**{entry['character_name']}**: {entry['edited_text']} _(Confidence: {entry['confidence']:.2f})_")
