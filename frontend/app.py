import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import cv2
import numpy as np
from PIL import Image

from inference.yolo_infer import load_yolo_model, run_yolo
from inference.frcnn_infer import load_frcnn_model, run_frcnn
from utils.draw_boxes import draw_boxes


# ✅ Page config
st.set_page_config(page_title="YOLO vs Faster R-CNN", layout="wide")

# ✅ Session state
if "page" not in st.session_state:
    st.session_state.page = 1


# 🔥 Load models
@st.cache_resource
def load_models():
    yolo = load_yolo_model("weights/yolo/yolo_augmented.pt")
    frcnn = load_frcnn_model("weights/frcnn/frcnn_augmented.pth")
    return yolo, frcnn


# =========================================================
# 📌 PAGE 1 → LANDING
# =========================================================
if st.session_state.page == 1:

    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(to right, #eef2f3, #ffffff);
        }

        .text-container {
            margin-top: 120px;
            text-align: center;
        }

        .title {
            font-size: 42px;
            font-weight: 700;
            color: #2c4a6e;
        }

        .subtitle {
            font-size: 22px;
            font-weight: 500;
            color: #2c4a6e;
            margin-top: 10px;
        }

        .desc {
            font-size: 18px;
            color: gray;
            margin-top: 10px;
        }

        div.stButton > button {
            background-color: #1f4e79;
            color: white;
            border-radius: 10px;
            height: 45px;
            width: 120px;
            font-size: 16px;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image("frontend/assets/logo.png", use_container_width=True)

    with col2:
        st.markdown("<div class='text-container'>", unsafe_allow_html=True)

        st.markdown("<div class='title'>Intelligent Aerial Object Detection</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Using Deep Learning</div>", unsafe_allow_html=True)
        st.markdown("<div class='desc'>Weather-Robust Small Object Detection in Aerial Imagery</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        btn_col1, btn_col2 = st.columns([3,1])
        with btn_col2:
            if st.button("🚀 Start"):
                st.session_state.page = 2
                st.rerun()


# =========================================================
# 📌 PAGE 2 → OVERVIEW
# =========================================================
elif st.session_state.page == 2:

    st.markdown("<h1 style='text-align: center;'>📖 Project Overview</h1>", unsafe_allow_html=True)

    overview_path = os.path.join(os.path.dirname(__file__), "assets", "overview1.png")

    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(overview_path, width=800)

    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        if st.button("⬅️ Back"):
            st.session_state.page = 1
            st.rerun()

    with col3:
        if st.button("➡️ Next"):
            st.session_state.page = 3
            st.rerun()


# =========================================================
# 📌 PAGE 3 → FLOWCHART ONLY
# =========================================================
elif st.session_state.page == 3:

    st.markdown("<h1 style='text-align: center;'>📊 Data Flow of the System</h1>", unsafe_allow_html=True)

    flow_path = os.path.join(os.path.dirname(__file__), "assets", "dataflow.png")

    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(flow_path, width=900)

    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        if st.button("⬅️ Back"):
            st.session_state.page = 2
            st.rerun()

    with col3:
        if st.button("➡️ Next"):
            st.session_state.page = 4
            st.rerun()


# =========================================================
# 📌 PAGE 4 → UPLOAD + OUTPUT
# =========================================================
elif st.session_state.page == 4:

    st.markdown("<h1 style='text-align: center;'>🖼️ Upload Image & View Results</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 3])

    # LEFT → INPUT
    with col1:
        st.subheader("📤 Upload & Settings")

        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
        conf_threshold = 0.1

    # RIGHT → OUTPUT
    with col2:

        yolo_model, frcnn_model = load_models()

        if uploaded_file is not None:

            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)

            st.subheader("📷 Uploaded Image")
            st.image(image, width=600)

            with st.spinner("Running detection..."):

                boxes_y, labels_y, scores_y, names_y = run_yolo(yolo_model, image_np)
                yolo_output = draw_boxes(image_np, boxes_y, labels_y, scores_y, names_y, conf_threshold)

                boxes_f, labels_f, scores_f, names_f = run_frcnn(frcnn_model, image)
                frcnn_output = draw_boxes(image_np, boxes_f, labels_f, scores_f, names_f, conf_threshold)

            st.success("✅ Detection Completed")

            colA, colB = st.columns(2)

            with colA:
                st.subheader("🟢 YOLO Output")
                st.image(cv2.cvtColor(yolo_output, cv2.COLOR_BGR2RGB), use_container_width=True)

            with colB:
                st.subheader("🔵 Faster R-CNN Output")
                st.image(cv2.cvtColor(frcnn_output, cv2.COLOR_BGR2RGB), use_container_width=True)

        else:
            st.info("⬅️ Upload an image from the left panel")

    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        if st.button("⬅️ Back"):
            st.session_state.page = 3
            st.rerun()