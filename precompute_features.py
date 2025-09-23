import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import os
import argparse

from utils import get_device, load_hateful_memes_data

def get_text_embeddings(texts, model, tokenizer, device, max_len=128):
    """Extract [CLS] token embeddings from DistilBERT."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for text in tqdm(texts, desc="Extracting Text Features"):
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=max_len
            ).to(device)
            
            # Get hidden states from the base model
            outputs = model.distilbert(**inputs)
            # Use the [CLS] token embedding
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding)
    return np.vstack(embeddings)

def get_image_embeddings(image_paths, model, transform, device):
    """Extract features from ResNet50."""
    model.eval()
    embeddings = []
    
    # Hook to get features from the layer before the final classification layer
    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    
    model.avgpool.register_forward_hook(get_features('avgpool'))

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Extracting Image Features"):
            if not os.path.exists(img_path):
                print(f"Warning: Image not found at {img_path}. Skipping.")
                # Using a zero vector as a placeholder
                embeddings.append(np.zeros((2048,)))
                continue
            
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            _ = model(image)
            img_embedding = features['avgpool'].squeeze().cpu().numpy()
            embeddings.append(img_embedding)
            
    return np.array(embeddings)

def get_aux_scores(texts, sarcasm_model, emotion_model):
    """Get predictions from sarcasm and emotion models."""
    sarcasm_probs = sarcasm_model.predict_proba(texts)[:, 1]  # Probability of being sarcastic
    
    # For emotion, we'll one-hot encode the predicted label
    emotion_preds = emotion_model.predict(texts)
    emotion_classes = emotion_model.classes_
    emotion_one_hot = np.zeros((len(texts), len(emotion_classes)))
    for i, pred in enumerate(emotion_preds):
        class_idx = np.where(emotion_classes == pred)[0][0]
        emotion_one_hot[i, class_idx] = 1
        
    return sarcasm_probs.reshape(-1, 1), emotion_one_hot

def main(args):
    device = get_device()
    print(f"Using device: {device}")

    # --- Load Data ---
    print(f"Loading hateful memes data from {args.data_path}")
    df = load_hateful_memes_data(args.data_path)
    texts = df['text'].tolist()
    image_paths = [os.path.join('data', p) for p in df['img'].tolist()]

    # --- 1. Text Features ---
    print("Loading text model...")
    text_model_path = 'models/text_model.bin'
    text_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    text_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    text_model.load_state_dict(torch.load(text_model_path, map_location=device))
    text_model.to(device)
    
    text_features = get_text_embeddings(texts, text_model, text_tokenizer, device)
    np.save('features/text_features.npy', text_features)
    print("Text features saved.")

    # --- 2. Image Features ---
    print("Loading image model...")
    img_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    img_model.to(device)
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image_features = get_image_embeddings(image_paths, img_model, img_transform, device)
    np.save('features/image_features.npy', image_features)
    print("Image features saved.")

    # --- 3. Auxiliary Features ---
    print("Loading auxiliary models...")
    sarcasm_model = joblib.load('models/sarcasm_model.joblib')
    emotion_model = joblib.load('models/emotion_model.joblib')

    sarcasm_scores, emotion_scores = get_aux_scores(texts, sarcasm_model, emotion_model)
    np.save('features/sarcasm_features.npy', sarcasm_scores)
    np.save('features/emotion_features.npy', emotion_scores)
    print("Auxiliary features saved.")

    # --- Save labels ---
    labels = df['label'].values
    np.save('features/labels.npy', labels)
    print("Labels saved.")

    print("\nFeature precomputation complete!")
    print(f"Text features shape: {text_features.shape}")
    print(f"Image features shape: {image_features.shape}")
    print(f"Sarcasm features shape: {sarcasm_scores.shape}")
    print(f"Emotion features shape: {emotion_scores.shape}")
    print(f"Labels shape: {labels.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/hateful_memes_sample.csv', help='Path to the hateful memes data file.')
    args = parser.parse_args()
    main(args)