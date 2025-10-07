# app.py
# Streamlit UI layout (multi-image):
# - Upload multiple images
# - Left: pick images to preview/run
# - Right: PREVIEW always shows ORIGINAL images
# - After clicking "Run model", a new section appears below:
#     For each selected image -> a row with: Original | Mask/Overlay | GradCAM

import io
from typing import List, Dict, Optional
import streamlit as st
from PIL import Image
from inference import load_pipeline

st.set_page_config(page_title="Coral Monitor", layout="wide")
st.title("🌊 Coral Monitor — YOLO + CoralScope (Demo)")

# ---------- Session state ----------
if "files" not in st.session_state:
    # files: List[{"name": str, "bytes": bytes}]
    st.session_state.files: List[Dict] = []
if "outputs_overlay" not in st.session_state:
    # overlay results (PNG bytes) keyed by file name
    st.session_state.outputs_overlay: Dict[str, bytes] = {}
if "outputs_gradcam" not in st.session_state:
    # gradcam results (PNG bytes) keyed by file name
    st.session_state.outputs_gradcam: Dict[str, bytes] = {}

# ---------- Helpers ----------
def get_original_pil(name: str) -> Image.Image:
    """Fetch original image by file name from session."""
    for rec in st.session_state.files:
        if rec["name"] == name:
            return Image.open(io.BytesIO(rec["bytes"])).convert("RGB")
    raise KeyError(f"{name} not found")

def get_overlay_pil(name: str) -> Optional[Image.Image]:
    """Fetch overlay image if exists."""
    b = st.session_state.outputs_overlay.get(name)
    return Image.open(io.BytesIO(b)).convert("RGB") if b else None

def get_gradcam_pil(name: str) -> Optional[Image.Image]:
    """Fetch gradcam image if exists."""
    b = st.session_state.outputs_gradcam.get(name)
    return Image.open(io.BytesIO(b)).convert("RGB") if b else None

@st.cache_resource
def get_pipeline():
    # Load your YOLO + CoralScope pipeline (see inference.py)
    return load_pipeline()

pipe = get_pipeline()

# ---------- Uploader (multi-file) ----------
uploaded_files = st.file_uploader(
    "Upload images — drag & drop or click to select",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

# Merge new files into session (avoid duplicates by name)
if uploaded_files:
    existing = {rec["name"] for rec in st.session_state.files}
    for uf in uploaded_files:
        if uf.name not in existing:
            st.session_state.files.append({"name": uf.name, "bytes": uf.read()})

# ---------- Layout: left selection, right preview ----------
left, right = st.columns([0.28, 0.72], gap="large")

with left:
    st.subheader("Select images")
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
    st.subheader("Preview (original images)")
    if not selected_names:
        st.info("Select images on the left to preview.")
    else:
        # 3-column grid for originals
        ncols = 3
        rows = [selected_names[i:i + ncols] for i in range(0, len(selected_names), ncols)]
        for row in rows:
            cols = st.columns(ncols, gap="small")
            for col, name in zip(cols, row):
                with col:
                    st.image(get_original_pil(name), caption=name, use_column_width=True)

# ---------- Run button ----------
run = st.button("Run model", type="primary", use_container_width=True)

if run:
    if not selected_names:
        st.warning("Please select at least one image from the list on the left.")
    else:
        progress = st.progress(0, text="Running model...")
        for idx, name in enumerate(selected_names, start=1):
            img = get_original_pil(name)

            # Support both (overlay, results) and (overlay, results, gradcam)
            ret = pipe.run(img)
            if isinstance(ret, tuple) and len(ret) == 3:
                overlay_pil, _results, gradcam_pil = ret
            else:
                overlay_pil, _results = ret  # type: ignore
                gradcam_pil = None

            # Save overlay to session (PNG bytes)
            buf = io.BytesIO()
            overlay_pil.save(buf, format="PNG")
            st.session_state.outputs_overlay[name] = buf.getvalue()

            # Save gradcam to session (if provided)
            if gradcam_pil is not None:
                gbuf = io.BytesIO()
                gradcam_pil.save(gbuf, format="PNG")
                st.session_state.outputs_gradcam[name] = gbuf.getvalue()

            progress.progress(idx / len(selected_names), text=f"Processed {idx}/{len(selected_names)}")
        progress.empty()
        st.success("Done! See model outputs below.")

# ---------- Outputs section (Original | Mask/Overlay | GradCAM) ----------
st.markdown("---")
st.subheader("Model outputs")

if not selected_names:
    st.caption("Tip: select images and click **Run model** to show outputs here.")
else:
    for name in selected_names:
        st.markdown(f"**{name}**")
        col_o, col_m, col_g = st.columns(3, gap="large")

        with col_o:
            st.image(get_original_pil(name), caption="Original", use_column_width=True)

        with col_m:
            ov = get_overlay_pil(name)
            if ov is None:
                st.info("No mask/overlay yet. Click **Run model**.")
            else:
                st.image(ov, caption="Mask / Overlay", use_column_width=True)

        with col_g:
            gc = get_gradcam_pil(name)
            if gc is None:
                st.info("No GradCAM yet.")
            else:
                st.image(gc, caption="GradCAM", use_column_width=True)
