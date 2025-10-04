# app.py
# Streamlit UI for multi-image workflow:
# - Upload multiple images
# - Left panel: choose which images to preview
# - Right panel: grid preview (shows original first, then overlay after inference)
# - Click "Run model" to run the pipeline on selected images
# - The pipeline is loaded via inference.load_pipeline() and cached as a resource

import io
from typing import List, Dict
import streamlit as st
from PIL import Image
from inference import load_pipeline

st.set_page_config(page_title="Coral Monitor", layout="wide")
st.title("🌊 Coral Monitor — YOLO + CoralScope (Demo)")

# --- Session state init (store uploaded files and inference outputs) ---
if "files" not in st.session_state:
    # files: List[{"name": str, "bytes": bytes}]
    st.session_state.files: List[Dict] = []
if "outputs" not in st.session_state:
    # outputs: Dict[file_name -> overlay_png_bytes]
    st.session_state.outputs: Dict[str, bytes] = {}

# --- Helper functions: fetch original / output image by file name ---
def get_original_pil(name: str) -> Image.Image:
    for rec in st.session_state.files:
        if rec["name"] == name:
            return Image.open(io.BytesIO(rec["bytes"])).convert("RGB")
    raise KeyError(f"{name} not found")

def get_output_pil(name: str):
    if name in st.session_state.outputs:
        return Image.open(io.BytesIO(st.session_state.outputs[name])).convert("RGB")
    return None

@st.cache_resource
def get_pipeline():
    # Load your YOLO + CoralScope pipeline (see inference.py)
    return load_pipeline()

pipe = get_pipeline()

# --- Top: multi-file uploader (supports drag & drop) ---
uploaded_files = st.file_uploader(
    "Upload images — drag & drop or click to select",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

# Merge newly uploaded files into session state (avoid duplicates by name)
if uploaded_files:
    existing = {rec["name"] for rec in st.session_state.files}
    for uf in uploaded_files:
        if uf.name not in existing:
            data = uf.read()
            st.session_state.files.append({"name": uf.name, "bytes": data})

# --- Layout: left list (choose images), right gallery (no border, auto height) ---
left, right = st.columns([0.28, 0.72], gap="large")

with left:
    st.subheader("Choose images")
    all_names = [rec["name"] for rec in st.session_state.files]
    if not all_names:
        st.info("No images uploaded yet.")
        selected_names: List[str] = []
    else:
        select_all = st.checkbox("Select all", value=True)
        default = all_names if select_all else all_names[: min(4, len(all_names))]
        selected_names = st.multiselect(
            "Uploaded files",
            options=all_names,
            default=default,
            placeholder="Select one or more files",
        )

with right:
    st.subheader("Preview")
    if not selected_names:
        st.info("Select images on the left to preview.")
    else:
        # Grid with 3 columns (change ncols if you want 2/4/etc.)
        ncols = 3
        rows = [selected_names[i:i + ncols] for i in range(0, len(selected_names), ncols)]
        for row in rows:
            cols = st.columns(ncols, gap="small")
            for col, name in zip(cols, row):
                # If we already have a model output, show it; otherwise show the original
                out_pil = get_output_pil(name)
                show_pil = out_pil if out_pil else get_original_pil(name)
                with col:
                    st.image(show_pil, caption=name, use_column_width=True)

# --- Bottom: Run Model button (run inference for selected images, replace previews) ---
run = st.button("Run model", type="primary", use_container_width=True)

if run:
    if not selected_names:
        st.warning("Please select at least one image from the list on the left.")
    else:
        # Run inference over the selected images
        progress = st.progress(0, text="Running model...")
        for idx, name in enumerate(selected_names, start=1):
            img = get_original_pil(name)
            # Your inference.py should return (overlay_image, results_dict)
            overlay_pil, _results = pipe.run(img)

            # Save overlay PNG bytes into session state so the gallery updates
            buf = io.BytesIO()
            overlay_pil.save(buf, format="PNG")
            st.session_state.outputs[name] = buf.getvalue()

            progress.progress(idx / len(selected_names), text=f"Processed {idx}/{len(selected_names)}")

        progress.empty()
        st.success("Done! The preview now shows model outputs with masks.")
        st.caption("Tip: if the gallery didn’t update, click the ‘Rerun’ button at the top of the page.")
