import streamlit as st
import numpy as np
import cv2
import os
import base64
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# -------------------------------
# FIX WINDOWS / TF LOGGING ISSUE
# -------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️",
    layout="centered"
)

# -------------------------------
# 🌈 THEME
# -------------------------------
st.markdown("""
    <style>

    .main {
        background: linear-gradient(135deg, #0f5132, #ffffff);
    }

    h1 {
        color: white;
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        text-shadow: 2px 2px 5px black;
    }

    .stFileUploader {
        background-color: white;
        border: 2px solid #0f5132;
        padding: 10px;
        border-radius: 12px;
    }

    .stButton>button {
        background-color: #0f5132;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #146c43;
        color: white;
    }

    section[data-testid="stSidebar"] {
        background-color: #0f5132;
        color: white;
    }

    .profile-pic {
        width: 150px;
        height: 150px;
        object-fit: cover;
        border-radius: 50%;
        border: 4px solid white;
        display: block;
        margin-left: auto;
        margin-right: auto;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.5);
    }

    </style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------
st.markdown("<h1>Handwritten Character Recognition</h1>", unsafe_allow_html=True)

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:

    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    try:
        img_base64 = get_base64_image("your_image.jpg")

        st.markdown(f"""
            <img src="data:image/jpg;base64,{img_base64}" class="profile-pic">
        """, unsafe_allow_html=True)

    except:
        st.warning("⚠️ Image not found (check your_image.jpg)")

    st.title("👨‍💻 About Me")

    st.write("""
    **Name:** Sharif Ullah  
    **Field:** AI / ML Engineer  
    **Project:** Handwritten Character Recognition  

    🔥 Skills:
    - Python
    - Machine Learning
    - Deep Learning
    - Computer Vision
    - Streamlit Apps
    """)

    st.success("🚀 Built with ❤️ using Streamlit")

# -------------------------------
# LOAD MODEL (SAFE)
# -------------------------------
@st.cache_resource
def load_my_model():
    path = os.path.join("models", "alphabet_model.h5")
    model = load_model(path, compile=False)
    return model

model = load_my_model()

labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# -------------------------------
# UPLOAD IMAGE
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])

# -------------------------------
# PREDICTION (FIXED)
# -------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption="📷 Uploaded Image", width=300)

    img = np.array(image)

    img = cv2.resize(img, (28, 28))
    img = np.rot90(img)
    img = np.fliplr(img)
    img = cv2.bitwise_not(img)
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    # ✅ FIXED LINE
    pred = model.predict(img, verbose=0)

    class_index = np.argmax(pred)
    confidence = np.max(pred)

    st.success(f"✅ Prediction: {labels[class_index]}")
    st.write(f"📊 Confidence: {confidence:.2f}")

    st.subheader("🔍 Processed Image")
    st.image(img.reshape(28, 28), width=150)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""
---
<center><b>Developed by Sharif Ullah</b> ❤️</center>
""", unsafe_allow_html=True)