import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms

# --- 1. Import Model from Subfolder ---
# This matches your "Main/Model/model_lib.py" structure
from Model.model_lib import MaskedAutoencoder

# --- 2. Page Configuration & Custom CSS ---
st.set_page_config(page_title="MAE Image Reconstruction", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stSlider { color: #4A90E2; }
    section[data-testid="stSidebar"] { width: 300px !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎨 Masked Autoencoder (MAE) Explorer")
st.write("This app demonstrates self-supervised learning by reconstructing images from sparse patches.")

# --- 3. Utility Functions ---
def unpatchify(x):
    """Reconstructs pixel patches back into image space."""
    p = 16
    h = w = int(x.shape[1]**0.5)
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs

@st.cache_resource
def load_trained_model():
    """Loads model architecture and weights from the Model subfolder."""
    model = MaskedAutoencoder()
    # Path updated to point to your Model folder
    weights_path = "Model/mae_weights.pth"
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# --- 4. Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Settings")
    mask_ratio = st.slider("Masking Ratio", 0.1, 0.9, 0.75, help="Percentage of patches to hide.")
    st.markdown("---")
    st.info("The Encoder only sees the visible patches (25% by default). The Decoder reconstructs the rest.")

# --- 5. Main App Logic ---
model = load_trained_model()

uploaded_file = st.file_uploader("Upload an image (TinyImageNet or High-Res)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open and convert image
    img = Image.open(uploaded_file).convert('RGB')
    
    # Preprocessing (224x224 as per assignment requirements)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)

    # --- 6. Inference ---
    with torch.no_grad():
        # Pass to MAE Model
        pred, mask = model(img_tensor, mask_ratio=mask_ratio)

    # Reconstruction Logic
    # 1. Patchify the original image to create the masked view
    p = 16
    h = w = img_tensor.shape[2] // p
    target = img_tensor.reshape(shape=(img_tensor.shape[0], 3, h, p, w, p))
    target = torch.einsum('nchpwq->nhwpqc', target)
    target = target.reshape(shape=(img_tensor.shape[0], h * w, p**2 * 3))

    mask_vis = mask.unsqueeze(-1).repeat(1, 1, 16**2 * 3)
    
    # Combine original visible patches and model predicted patches
    recon_patches = target * (1 - mask_vis) + pred * mask_vis
    recon_imgs = unpatchify(recon_patches)
    
    # Masked input for visualization
    masked_img_patches = target * (1 - mask_vis)
    masked_imgs = unpatchify(masked_img_patches)

    # --- 7. Post-processing for Display ---
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def denormalize(t):
        t = torch.clamp(t[0] * std + mean, 0, 1)
        return t.permute(1, 2, 0).numpy()

    final_masked = denormalize(masked_imgs)
    final_recon = denormalize(recon_imgs)
    final_orig = denormalize(img_tensor)

    # --- 8. Multi-Column Display ---
    st.subheader(f"Results with {int(mask_ratio*100)}% Masking")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(final_masked, caption="Masked Input", use_container_width=True)
    with col2:
        st.image(final_recon, caption="MAE Reconstruction", use_container_width=True)
    with col3:
        st.image(final_orig, caption="Original GT", use_container_width=True)
        
    st.success("Reconstruction complete! Adjust the slider to see how the model adapts.")
else:
    st.warning("Please upload an image to begin.")