import torch
import torch.nn.functional as F
import numpy as np
import argparse
import joblib
from PIL import Image
import os

from utils import get_device
from precompute_features import get_text_embeddings, get_image_embeddings, get_aux_scores
from fusion_train import FusionMLP

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms

def load_models(device, text_model_path, sarcasm_model_path, emotion_model_path, fusion_model_path):
    """Load all necessary models for inference."""
    # Text model
    text_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    text_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
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
    # Define feature dimensions based on the models, not cached .npy files.
    # This makes the demo independent of the 'features' directory.
    text_dim = 768  # DistilBERT base model's hidden size
    img_dim = 2048  # ResNet50's final avgpool layer output size
    sarcasm_dim = 1   # A single probability score
    emotion_dim = len(emotion_model.classes_) # Number of emotion classes from the trained model
    input_dim = text_dim + img_dim + sarcasm_dim + emotion_dim

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

def predict(text, image_path, models, device):
    """Make a prediction on a single instance."""
    # 1. Extract features
    text_model, text_tokenizer = models['text']
    img_model, img_transform = models['image']
    sarcasm_model = models['sarcasm']
    emotion_model = models['emotion']
    fusion_model = models['fusion']

    text_feat = get_text_embeddings([text], text_model, text_tokenizer, device)
    
    if image_path and os.path.exists(image_path):
        image_feat = get_image_embeddings([image_path], img_model, img_transform, device)
    else:
        print(f"Warning: Image path '{image_path}' not found. Using zero vector for image features.")
        image_feat = np.zeros((1, 2048)) # ResNet50 feature size

    sarcasm_feat, emotion_feat = get_aux_scores([text], sarcasm_model, emotion_model)

    # 2. Concatenate features
    all_features = np.concatenate([text_feat, image_feat, sarcasm_feat, emotion_feat], axis=1)
    features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)

    # 3. Predict with fusion model
    fusion_model.eval()
    with torch.no_grad():
        outputs = fusion_model(features_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    label = "HATE" if predicted_class.item() == 1 else "NOT HATE"
    return label, confidence.item()

def main():
    parser = argparse.ArgumentParser(description="Hate Speech Detection Demo")
    parser.add_argument("--text", type=str, required=True, help="Input text.")
    parser.add_argument("--image_path", type=str, default=None, help="Path to the input image.")
    parser.add_argument('--text_model_path', default='models/text_model.bin')
    parser.add_argument('--sarcasm_model_path', default='models/sarcasm_model.joblib')
    parser.add_argument('--emotion_model_path', default='models/emotion_model.joblib')
    parser.add_argument('--fusion_model_path', default='models/fusion_model.bin')

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    try:
        models = load_models(device,
                             text_model_path=args.text_model_path,
                             sarcasm_model_path=args.sarcasm_model_path,
                             emotion_model_path=args.emotion_model_path,
                             fusion_model_path=args.fusion_model_path)
    except FileNotFoundError as e:
        print(f"Error loading models: {e}. Please run the training scripts first.")
        return

    label, confidence = predict(args.text, args.image_path, models, device)

    print("\n--- Prediction ---")
    print(f"Label: {label}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == '__main__':
    main()