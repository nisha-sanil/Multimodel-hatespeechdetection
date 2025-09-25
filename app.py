import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import requests
import io
import easyocr # The OCR library

# --- Local Imports ---
# These imports bring in the functions and models from your 'src' directory
from src.utils import get_device
from src.inference import load_models, predict as main_predict_func

# --- Page Configuration ---
# This should be the first Streamlit command in your app
st.set_page_config(
    page_title="Multimodal Hate Speech Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Model & Path Constants ---
# Using constants makes the code cleaner and easier to maintain.
# IMPORTANT: Replace these placeholder URLs with your actual direct download links.
MODEL_URLS = {
    "text_model.bin": "YOUR_DIRECT_LINK_FOR_TEXT_MODEL",
    "sarcasm_model.joblib": "YOUR_DIRECT_LINK_FOR_SARCASM_MODEL",
    "emotion_model.joblib": "YOUR_DIRECT_LINK_FOR_EMOTION_MODEL",
    "fusion_model.bin": "YOUR_DIRECT_LINK_FOR_FUSION_MODEL"
}

MODELS_DIR = "models"
TEXT_MODEL_PATH = os.path.join(MODELS_DIR, 'text_model.bin')
SARCASM_MODEL_PATH = os.path.join(MODELS_DIR, 'sarcasm_model.joblib')
EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, 'emotion_model.joblib')
FUSION_MODEL_PATH = os.path.join(MODELS_DIR, 'fusion_model.bin')

# --- Helper & Caching Functions ---

def download_file(url, destination):
    """Downloads a file from a URL to a destination, showing a progress bar in Streamlit."""
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            progress_bar = st.progress(0, text=f"Downloading {os.path.basename(destination)}...")
            
            with open(destination, 'wb') as f:
                bytes_downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bytes_downloaded += len(chunk)
                    if total_size > 0:
                        progress = min(1.0, bytes_downloaded / total_size)
                        progress_bar.progress(progress, text=f"Downloading {os.path.basename(destination)}... {int(progress * 100)}%")
            progress_bar.empty()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading {os.path.basename(destination)}: {e}")
        return False

@st.cache_resource
def cached_load_and_prep_models():
    """
    A cached function to download models if needed, then load them into memory.
    Using @st.cache_resource ensures this complex setup runs only once.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    all_models_present = True
    model_paths = {
        "text_model.bin": TEXT_MODEL_PATH,
        "sarcasm_model.joblib": SARCASM_MODEL_PATH,
        "emotion_model.joblib": EMOTION_MODEL_PATH,
        "fusion_model.bin": FUSION_MODEL_PATH
    }

    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            st.info(f"Model '{model_name}' not found. Downloading...")
            url = MODEL_URLS.get(model_name)
            if not url or "YOUR_DIRECT_LINK" in url:
                st.error(f"Direct download link for '{model_name}' is not configured. Please update MODEL_URLS in app.py.")
                all_models_present = False
                continue
            if not download_file(url, model_path):
                all_models_present = False

    if not all_models_present:
        st.error("One or more models failed to download. The app cannot proceed. Please check links and refresh.")
        return None, None

    device = get_device()
    try:
        st.info("Loading hate detection models... (This may take a moment)")
        models = load_models(
            device,
            text_model_path=TEXT_MODEL_PATH,
            sarcasm_model_path=SARCASM_MODEL_PATH,
            emotion_model_path=EMOTION_MODEL_PATH,
            fusion_model_path=FUSION_MODEL_PATH
        )
        st.success("Hate detection models loaded successfully!")
        return models, device
    except Exception as e:
        st.error(f"An error occurred while loading models: {e}. Ensure model files are in the 'models/' directory.")
        return None, None

@st.cache_resource
def load_ocr_model():
    """Loads the EasyOCR model into memory, cached for performance."""
    st.info("Loading OCR model for reading text from images...")
    reader = easyocr.Reader(['en'])
    st.success("OCR model loaded.")
    return reader

def predict_text_only(text, models, device):
    """Performs prediction using only the fine-tuned text model."""
    text_model, text_tokenizer = models['text']
    text_model.eval()
    
    if not text or not text.strip():
        return None, 0.0

    encoded_text = text_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    encoded_text = {k: v.to(device) for k, v in encoded_text.items()}

    with torch.no_grad():
        outputs = text_model(**encoded_text)
        probabilities = F.softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    label = "HATE" if predicted_class.item() == 1 else "NOT HATE"
    return label, confidence.item()

# --- Main Application ---

# Load all models (these are cached after the first run)
models, device = cached_load_and_prep_models()
ocr_reader = load_ocr_model()

st.title("Multimodal Hate Speech Detection 🚀")
st.markdown("""
This app detects hate speech using two different approaches. Choose your desired mode from the sidebar.
- **Multimodal (Meme Analysis)**: **Upload a meme.** The app will use OCR to read the text and then analyze the image and text together.
- **Text-Only Analysis**: Uses a fine-tuned text model, ideal for analyzing text without an image.
""")

# --- Sidebar for Inputs ---
st.sidebar.header("Input Controls")

prediction_mode = st.sidebar.selectbox(
    "Choose Analysis Mode",
    ("Multimodal (Meme Analysis)", "Text-Only Analysis")
)

# Initialize session state for the text area to allow OCR to update it
if 'input_text' not in st.session_state:
    st.session_state.input_text = "Upload a meme in Multimodal mode or enter text here."

# --- Main Content Area ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Inputs")
    
    uploaded_file = None
    if prediction_mode == "Multimodal (Meme Analysis)":
        uploaded_file = st.file_uploader("Upload a Meme Image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Meme", use_column_width=True)
            
            # Perform OCR and update the text area via session state
            with st.spinner("Reading text from image..."):
                img_bytes = uploaded_file.getvalue()
                ocr_results = ocr_reader.readtext(img_bytes)
                extracted_text = " ".join([res[1] for res in ocr_results])
                st.session_state.input_text = extracted_text if extracted_text else "OCR could not detect any text."
    else: # Text-Only mode
        st.info("In Text-Only mode, only the text box below is used for analysis.")

    # The text area uses session state, so it can be updated by the OCR process
    input_text = st.text_area(
        "Text Content (auto-filled from image in Multimodal mode)", 
        key="input_text",
        height=150
    )

with col2:
    st.subheader("Prediction Result")
    if st.button("Detect Hate Speech", use_container_width=True, type="primary"):
        # --- Input Validation ---
        if models is None:
            st.error("Core models are not loaded. Cannot perform prediction.")
        elif prediction_mode == "Text-Only Analysis" and not input_text.strip():
            st.warning("Please enter some text for Text-Only mode.")
        elif prediction_mode == "Multimodal (Meme Analysis)" and not uploaded_file:
            st.warning("Please upload an image for Multimodal mode.")
        else:
            with st.spinner("Analyzing..."):
                label, confidence = None, 0.0
                temp_image_path = None
                
                try:
                    if prediction_mode == "Text-Only Analysis":
                        label, confidence = predict_text_only(input_text, models, device)
                    
                    else: # Multimodal (Meme Analysis)
                        # Save the uploaded file to a temporary path for the model
                        temp_dir = "temp_images"
                        os.makedirs(temp_dir, exist_ok=True)
                        temp_image_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_image_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Call the main predict function from inference.py
                        label, confidence = main_predict_func(input_text, temp_image_path, models, device, mode='fusion')

                    # --- Display Results ---
                    if label:
                        if label == "HATE":
                            st.error(f"**Result: {label}**")
                        else:
                            st.success(f"**Result: {label}**")
                        
                        st.metric(label="Confidence", value=f"{confidence:.2%}")

                        # Display auxiliary cues only if text was provided
                        if input_text.strip() and "could not detect" not in input_text:
                            st.markdown("---")
                            st.markdown("**Inferred Auxiliary Cues (from text):**")
                            sarcasm_prob = models['sarcasm'].predict_proba([input_text])[0, 1]
                            emotion_pred = models['emotion'].predict([input_text])[0]
                            st.write(f"Sarcasm Probability: {sarcasm_prob:.2f}")
                            st.write(f"Predicted Emotion: **{emotion_pred.capitalize()}**")
                
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

                finally:
                    # Clean up the temporary image file
                    if temp_image_path and os.path.exists(temp_image_path):
                        os.remove(temp_image_path)

st.markdown("---")
st.markdown("Built with Gemini Code Assist.")
