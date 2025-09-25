import torch
import torch.nn.functional as F
import numpy as np
import joblib
from PIL import Image
import torchvision.transforms as transforms
import os
import argparse

# --- Local Imports ---
# Changed from relative to direct imports to allow script to be run standalone
from utils import get_device
from precompute_features import get_text_embeddings, get_image_embeddings, get_aux_scores
from fusion_train import FusionMLP, TEXT_DIM, IMG_DIM, SARCASM_DIM, EMOTION_DIM

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torchvision.models import resnet50, ResNet50_Weights

def load_models(device, text_model_path, sarcasm_model_path, emotion_model_path, fusion_model_path):
    """Load all necessary models for inference."""
    print("Loading models...")
    # Text model
    text_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    text_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    text_model.load_state_dict(torch.load(text_model_path, map_location=device))
    text_model.to(device)
    text_model.eval()

    # Image model
    img_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    img_model.to(device)
    img_model.eval()
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
    fusion_model.eval()

    models = {
        'text': (text_model, text_tokenizer),
        'image': (img_model, img_transform),
        'sarcasm': sarcasm_model,
        'emotion': emotion_model,
        'fusion': fusion_model
    }
    print("All models loaded successfully.")
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
        if not text or not text.strip():
            return "NO TEXT", 0.0
            
        encoded_text = text_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        encoded_text = {k: v.to(device) for k, v in encoded_text.items()}

        with torch.no_grad():
            outputs = text_model(**encoded_text)
            probabilities = F.softmax(outputs.logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

    elif mode == 'fusion':
        # --- Multimodal Fusion Prediction ---
        # Use a placeholder for text if it's empty or whitespace to avoid errors
        text_to_process = text if text and text.strip() else "[PAD]"

        text_feat = get_text_embeddings([text_to_process], text_model, text_tokenizer, device)
        
        if image_path and os.path.exists(image_path):
            image_feat = get_image_embeddings([image_path], img_model, img_transform, device)
        else:
            # Use a zero vector if no image is provided or path is invalid
            image_feat = np.zeros((1, IMG_DIM))

        sarcasm_feat, emotion_feat = get_aux_scores([text_to_process], sarcasm_model, emotion_model)

        # Concatenate features
        all_features = np.concatenate([text_feat, image_feat, sarcasm_feat, emotion_feat], axis=1)
        features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)

        # Predict with fusion model
        with torch.no_grad():
            outputs = fusion_model(features_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
    else:
        raise ValueError(f"Invalid prediction mode: {mode}")

    label = "HATE" if predicted_class.item() == 1 else "NOT HATE"
    return label, confidence.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference for hate speech detection.")
    parser.add_argument('--text', type=str, default="", help='Text to analyze.')
    parser.add_argument('--image_path', type=str, default="", help='Path to the image file.')
    parser.add_argument('--mode', type=str, default='fusion', choices=['fusion', 'text_only'], help='Prediction mode.')
    
    # Model paths
    parser.add_argument('--text_model_path', type=str, default='models/text_model.bin')
    parser.add_argument('--sarcasm_model_path', type=str, default='models/sarcasm_model.joblib')
    parser.add_argument('--emotion_model_path', type=str, default='models/emotion_model.joblib')
    parser.add_argument('--fusion_model_path', type=str, default='models/fusion_model.bin')
    
    args = parser.parse_args()

    if not args.text and not args.image_path:
        print("Error: Please provide either --text or --image_path.")
    else:
        device = get_device()
        try:
            all_models = load_models(
                device,
                text_model_path=args.text_model_path,
                sarcasm_model_path=args.sarcasm_model_path,
                emotion_model_path=args.emotion_model_path,
                fusion_model_path=args.fusion_model_path
            )
            
            final_label, final_confidence = predict(
                args.text, 
                args.image_path, 
                all_models, 
                device, 
                mode=args.mode
            )

            print("\n--- Prediction Result ---")
            print(f"Mode: {args.mode}")
            print(f"Text Input: '{args.text}'")
            print(f"Image Input: '{args.image_path}'")
            print("-" * 25)
            print(f"Predicted Label: {final_label}")
            print(f"Confidence: {final_confidence:.2%}")
            print("-------------------------")

        except FileNotFoundError as e:
            print(f"\nError: A model file was not found. {e}")
            print("Please ensure all models are trained and located in the 'models/' directory or provide correct paths.")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
