import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from models.cnn_autoencoder import CNNAutoencoder
from models.mlp_autoencoder import MLPAutoencoder
from configs.config import Config

st.set_page_config(
    page_title="Denoising Autoencoder",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Syne', sans-serif;
        }

        .main-title {
            font-family: 'Syne', sans-serif;
            font-weight: 800;
            font-size: 24px;
            letter-spacing: -1px;
            background: linear-gradient(135deg, #00B7FF, #000DFF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }

        .subtitle {
            font-family: 'Space Mono', monospace;
            font-size: 0.78rem;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-top: 2px;
            font-weight: 800;
            margin-bottom: 2rem;
        }

        .credits-box {
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 1.2rem 1.6rem;
            margin-bottom: 2rem;
            font-family: 'Space Mono', monospace;
        }

        .credits-label {
            font-size: 27px;
            text-transform: uppercase;
            letter-spacing: 3px;
            margin-bottom: 0.3rem;
            font-weight: 800;
        }

        .credits-name {
            font-size: 18px;
            font-weight: 800;
            line-height: 1.8;
        }

        .teacher-name {
            font-size: 18px;
            font-weight: 800;
        }

        .image-label {
            font-family: 'Space Mono', monospace;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: #94a3b8;
            text-align: center;
            margin-bottom: 0.4rem;
        }

        .image-card {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 12px;
            padding: 1rem;
        }

        .image-wrapper {
            display: flex;
            justify-content: center;
        }

        .stSlider > div > div > div {
            background: #7c3aed;
        }

        .upload-hint {
            font-family: 'Space Mono', monospace;
            font-size: 0.75rem;
            color: #64748b;
            margin-top: 0.5rem;
        }

        hr {
            border-color: #1e293b;
            margin: 1.5rem 0;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown('<h1 class="main-title">Denoising Autoencoder</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Image Restoration with Deep Learning</p>', unsafe_allow_html=True)

col_dev, col_teach = st.columns([2, 1])

with col_dev:
    st.markdown("""
        <div class="credits-box">
            <div class="credits-label">Developed by</div>
            <div class="credits-name">
                SOUFIAN BOUKIR<br>
                MOHAMED BELALIA<br>
                MOUAD ELOUICHOUANI
            </div>
        </div>
    """, unsafe_allow_html=True)

with col_teach:
    st.markdown("""
        <div class="credits-box">
            <div class="credits-label">Supervised by</div>
            <div class="teacher-name">
                Pr. Nasreddine HAFIDI
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ↓ Model selector — user picks CNN or MLP from the UI
model_choice = st.radio(
    "Select Model",
    options=["CNN", "MLP"],
    horizontal=True
)

# ↓ Loads the correct model based on user selection
# Change the .pth filenames if yours are named differently
@st.cache_resource
def load_model(choice):
    if choice == "CNN":
        m = CNNAutoencoder()
        m.load_state_dict(torch.load("saved_model_cnn.pth", map_location="cpu", weights_only=True))
    else:
        m = MLPAutoencoder()
        # ↓ Change this filename to match your MLP saved model file
        m.load_state_dict(torch.load("saved_model_mlp.pth", map_location="cpu", weights_only=True))
    m.eval()
    return m

model = load_model(model_choice)

st.markdown("---")

uploaded = st.file_uploader("Upload an image to denoise", type=["png", "jpg", "jpeg"])
st.markdown('<p class="upload-hint">Supports PNG, JPG — will be resized to 28×28 (FashionMNIST format)</p>', unsafe_allow_html=True)

if uploaded:
    st.markdown("---")

    noise_level = st.slider("Noise level", 0.0, 1.0, 0.3, step=0.05)

    image = Image.open(uploaded).convert("L")
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    img = transform(image).unsqueeze(0)

    noisy = img + noise_level * torch.randn_like(img)
    noisy = torch.clamp(noisy, 0., 1.)

    with torch.no_grad():
        output = model(noisy)

    col1, col2, col3 = st.columns(3)

    # ↓ Change this number to resize images (e.g. 150, 250, 300)
    IMAGE_WIDTH = 400

    with col1:
        st.markdown('<p class="image-label">Original</p>', unsafe_allow_html=True)
        st.image(img.squeeze().numpy(), width=IMAGE_WIDTH, clamp=True)

    with col2:
        st.markdown('<p class="image-label">Noisy input</p>', unsafe_allow_html=True)
        st.image(noisy.squeeze().numpy(), width=IMAGE_WIDTH, clamp=True)

    with col3:
        st.markdown(f'<p class="image-label">Reconstructed ({model_choice})</p>', unsafe_allow_html=True)
        st.image(output.squeeze().numpy(), width=IMAGE_WIDTH, clamp=True)