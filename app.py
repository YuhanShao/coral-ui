# app.py
import io
from typing import List, Dict
import streamlit as st
from PIL import Image
from inference import load_pipeline

st.set_page_config(page_title="Coral Monitor", layout="wide")
st.title("🌊 Coral Monitor — YOLO + CoralScope (Demo)")

# --- 状态初始化（把已上传文件和推理结果放进会话态） ---
if "files" not in st.session_state:
    # files: List[{"name": str, "bytes": bytes}]
    st.session_state.files: List[Dict] = []
if "outputs" not in st.session_state:
    # outputs: Dict[file_name -> overlay_png_bytes]
    st.session_state.outputs: Dict[str, bytes] = {}

# --- 工具函数：通过文件名取原图/输出图 ---
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
    # 加载你们的 YOLO + CoralScope 管线（见 inference.py）
    return load_pipeline()

pipe = get_pipeline()

# --- 顶部：多文件上传（支持拖拽） ---
uploaded_files = st.file_uploader(
    "Upload images — drag & drop or click to select",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

# 把本次新上传的文件并入会话态（避免重名重复加入）
if uploaded_files:
    existing = {rec["name"] for rec in st.session_state.files}
    for uf in uploaded_files:
        if uf.name not in existing:
            data = uf.read()
            st.session_state.files.append({"name": uf.name, "bytes": data})

# --- 布局：左侧列表（选择要预览的图片），右侧画廊（无边框、不固定高度） ---
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
        # 网格：3 列（需要改列数就改 ncols）
        ncols = 3
        rows = [selected_names[i:i + ncols] for i in range(0, len(selected_names), ncols)]
        for row in rows:
            cols = st.columns(ncols, gap="small")
            for col, name in zip(cols, row):
                # 如果已有模型输出，就显示输出；否则显示原图
                out_pil = get_output_pil(name)
                show_pil = out_pil if out_pil else get_original_pil(name)
                with col:
                    st.image(show_pil, caption=name, use_column_width=True)

# --- 底部：Run Model 按钮（对所选图片做推理，替换显示为掩膜输出） ---
run = st.button("Run model", type="primary", use_container_width=True)

if run:
    if not selected_names:
        st.warning("Please select at least one image from the list on the left.")
    else:
        # 运行模型推理（对所选图片）
        progress = st.progress(0, text="Running model...")
        for idx, name in enumerate(selected_names, start=1):
            img = get_original_pil(name)
            overlay_pil, _results = pipe.run(img)  # 你们的 inference.py 返回 (overlay, results)

            # 保存为 PNG 字节到会话态，供右侧画廊显示
            buf = io.BytesIO()
            overlay_pil.save(buf, format="PNG")
            st.session_state.outputs[name] = buf.getvalue()

            progress.progress(idx / len(selected_names), text=f"Processed {idx}/{len(selected_names)}")

        progress.empty()
        st.success("Done! The preview now shows model outputs with masks.")
        st.caption("Tip: if the gallery didn’t update, click the ‘Rerun’ button at the top of the page.")
