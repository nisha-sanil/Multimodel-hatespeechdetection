import torch
import torch.nn.functional as F
import numpy as np
import joblib
from PIL import Image
import torchvision.transforms as transforms
import os

from .utils import get_device
from .precompute_features import get_text_embeddings, get_image_embeddings, get_aux_scores
from .fusion_train import FusionMLP, TEXT_DIM, IMG_DIM, SARCASM_DIM, EMOTION_DIM

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torchvision.models import resnet50, ResNet50_Weights

def load_models(device, text_model_path, sarcasm_model_path, emotion_model_path, fusion_model_path):
    """Load all necessary models for inference."""
    # Text model
    text_tokenizer = RobertaTokenizer.from_pretrained('roberta-base') # <-- Use RoBERTa Tokenizer
    text_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2) # <-- Use RoBERTa Model
    text_model.load_state_dict(torch.load(text_model_path, map_location=device))
    text_model.to(device)

    # Image model
    img_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    img_model.to(device)
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Aux models
    sarcasm_model = joblib.load(sarcasm_model_path)
    emotion_model = joblib.load(emotion_model_path)

    # Fusion model
    input_dim = TEXT_DIM + IMG_DIM + SARCASM_DIM + EMOTION_DIM

    fusion_model = FusionMLP(input_dim=input_dim)
    fusion_model.load_state_dict(torch.load(fusion_model_path, map_location=device))
    fusion_model.to(device)

    models = {
        'text': (text_model, text_tokenizer),
        'image': (img_model, img_transform),
        'sarcasm': sarcasm_model,
        'emotion': emotion_model,
        'fusion': fusion_model
    }
    return models

def predict(text, image_path, models, device, mode='fusion'):
    """Make a prediction on a single instance."""
    # 1. Extract features
    text_model, text_tokenizer = models['text']
    img_model, img_transform = models['image']
    sarcasm_model = models['sarcasm']
    emotion_model = models['emotion']
    fusion_model = models['fusion']

    if mode == 'text_only':
        # --- Text-Only Prediction ---
        text_model.eval()
        encoded_text = text_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        encoded_text = {k: v.to(device) for k, v in encoded_text.items()}

        with torch.no_grad():
            outputs = text_model(**encoded_text)
            probabilities = F.softmax(outputs.logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

    elif mode == 'fusion':
        # --- Multimodal Fusion Prediction ---
        # Use a placeholder for text if it's empty to avoid errors
        text_to_process = text if text and text.strip() else "[CLS]"

        text_feat = get_text_embeddings([text_to_process], text_model, text_tokenizer, device)
        
        if image_path and os.path.exists(image_path):
            image_feat = get_image_embeddings([image_path], img_model, img_transform, device)
        else:
            # Use a zero vector if no image is provided
            image_feat = np.zeros((1, IMG_DIM))

        sarcasm_feat, emotion_feat = get_aux_scores([text_to_process], sarcasm_model, emotion_model)

        # Concatenate features
        all_features = np.concatenate([text_feat, image_feat, sarcasm_feat, emotion_feat], axis=1)
        features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)

        # Predict with fusion model
        fusion_model.eval()
        with torch.no_grad():
            outputs = fusion_model(features_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
    else:
        raise ValueError(f"Invalid prediction mode: {mode}")

    label = "HATE" if predicted_class.item() == 1 else "NOT HATE"
    return label, confidence.item()