import streamlit as st
import torch
import numpy as np
from PIL import Image
import os

from src.utils import get_device
from src.demo import load_models, predict

# --- Page Config ---
st.set_page_config(page_title="Multimodal Hate Speech Detection", layout="wide")

# --- Constants for Model Paths ---
# This makes it easy to configure which models the app uses.
TEXT_MODEL_PATH = 'models/text_model.bin'
SARCASM_MODEL_PATH = 'models/sarcasm_model.joblib'
EMOTION_MODEL_PATH = 'models/emotion_model.joblib'
FUSION_MODEL_PATH = 'models/fusion_model.bin'

@st.cache_resource
def cached_load_models():
    """Load and cache the models to avoid reloading on every run."""
    device = get_device()
    try:
        models = load_models(device,
                             text_model_path=TEXT_MODEL_PATH,
                             sarcasm_model_path=SARCASM_MODEL_PATH,
                             emotion_model_path=EMOTION_MODEL_PATH,
                             fusion_model_path=FUSION_MODEL_PATH)
        return models, device
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Please ensure all models exist in the 'models/' directory. Run the training scripts to generate them.")
        return None, None

models, device = cached_load_models()

# --- UI Components ---
st.title("Multimodal Hate Speech Detection 🚀")
st.markdown("""
This app uses a fusion model to detect hate speech from text and images (memes). 
It leverages embeddings from DistilBERT (text), ResNet50 (image), and auxiliary classifiers for sarcasm and emotion.
""")

st.sidebar.header("Input")

input_text = st.sidebar.text_area("Text Content", "Enter the text from the meme or post here...")

uploaded_file = st.sidebar.file_uploader("Upload a Meme Image", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns(2)

with col1:
    st.subheader("Inputs")
    st.markdown(f"**Text:**")
    st.info(input_text)

    temp_image_path = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Meme", use_column_width=True)
        
        temp_dir = "temp_images"
        os.makedirs(temp_dir, exist_ok=True)
        temp_image_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

with col2:
    st.subheader("Prediction")
    if st.sidebar.button("Detect Hate Speech", use_container_width=True, type="primary"):
        if not input_text and not uploaded_file:
            st.warning("Please provide text or an image.")
        elif models is None:
            st.error("Models are not loaded. Cannot perform prediction.")
        else:
            try:
                with st.spinner("Analyzing..."):
                    label, confidence = predict(input_text, temp_image_path, models, device)

                    if label == "HATE":
                        st.error(f"Label: {label}")
                    else:
                        st.success(f"Label: {label}")
                    
                    st.metric(label="Confidence Score", value=f"{confidence:.4f}")

                    st.markdown("---")
                    st.markdown("**Inferred Auxiliary Cues:**")
                    sarcasm_prob = models['sarcasm'].predict_proba([input_text])[0, 1]
                    emotion_pred = models['emotion'].predict([input_text])[0]
                    st.write(f"Sarcasm Probability: {sarcasm_prob:.2f}")
                    st.write(f"Predicted Emotion: **{emotion_pred.capitalize()}**")
            finally:
                if temp_image_path and os.path.exists(temp_image_path):
                    try:
                        os.remove(temp_image_path)
                    except Exception as e:
                        st.warning(f"Error removing temporary file: {e}")

st.markdown("---")
st.markdown("Built by Gemini Code Assist.")
