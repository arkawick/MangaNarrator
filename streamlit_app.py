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
    # Step 3: Dialogue-to-Character Mapping
    st.markdown("### 🧭 Step 3: Dialogue-to-Character Mapping")
    mapping_result = map_dialogues_to_characters(dialogues, char_boxes)

    for item in mapping_result:
        st.write(f"🗣️ Character #{item['character_id']} says: **{item['text']}** (Conf: {item['confidence']:.2f})")

    # Step 4: Manual Character Naming
    st.markdown("### 🧍 Step 4: Assign Names to Characters")
    if 'character_names' not in st.session_state:
        st.session_state.character_names = {}

    character_names = st.session_state.character_names

    for idx in set(item['character_id'] for item in mapping_result):
        key = f"charname_{idx}"
        default_name = character_names.get(idx, f"Character #{idx}")
        character_names[idx] = st.text_input(f"Enter name for Character #{idx}:", value=default_name, key=key)

    st.session_state.character_names = character_names  # persist names



    ####
    # Step 4.5: Editable Dialogue Text
    st.markdown("### 📝 Step 4.5: Edit Assigned Dialogues")

    # Initialize session state
    if 'editable_lines' not in st.session_state:
        st.session_state.editable_lines = []

    # Populate for the first time using mapping_result
    if not st.session_state.editable_lines and mapping_result:
        for idx, item in enumerate(mapping_result):
            char_id = item['character_id']
            char_name = character_names.get(char_id, f"Character #{char_id}")
            st.session_state.editable_lines.append({
                'char_name': char_name,
                'text': item['text'],
                'confidence': item.get('confidence', 1.0)
            })

    # Editable form section
    updated_lines = []
    for idx, line in enumerate(st.session_state.editable_lines):
        st.markdown(f"#### ✏️ Dialogue #{idx + 1}")
        
        char_input = st.text_input(
            label="Character Name",
            value=line['char_name'],
            key=f"char_{idx}"
        )

        text_input = st.text_input(
            label="Dialogue Text",
            value=line['text'],
            key=f"text_{idx}"
        )

        updated_lines.append({
            'char_name': char_input,
            'text': text_input,
            'confidence': line.get('confidence', 1.0)
        })

    # Update session state
    st.session_state.editable_lines = updated_lines

    if st.button("Clear All Dialogues"):
        st.session_state.editable_lines = []

    if st.button("Show Final Narration Lines"):
        st.markdown("### 🔊 Final Dialogue List:")
        for line in st.session_state.editable_lines:
            st.markdown(f"**{line['char_name']}**: {line['text']}")

    if st.button("Generate Dia-formatted Output"):
        st.markdown("### 📜 Dia-compatible Script Output")

        speaker_map = {}
        dia_lines = []
        next_speaker = 1

        for line in st.session_state.editable_lines:
            name = line['char_name'].strip()
            if name not in speaker_map:
                speaker_map[name] = f"S{next_speaker}"
                next_speaker += 1
            speaker = speaker_map[name]
            dia_lines.append(f"[{speaker}] {line['text']}")

        dia_script = "\n".join(dia_lines)

        # Show editable output
        st.text_area("🗒️ Dia Script Output", value=dia_script, height=300)


else:
    st.warning("❗ Please upload a valid image and run Steps 1 & 2 before proceeding to character mapping.")
