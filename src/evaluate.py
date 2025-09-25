import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader

# Assuming fusion_train.py is in the same src directory
from fusion_train import FusionMLP
from utils import set_seed, get_device, load_olid_data, TextDataset

def evaluate_fusion_model(model, features, device):
    """Evaluate a given fusion model on a set of features."""
    model.to(device)
    model.eval()
    
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = model(features_tensor)
        _, predictions = torch.max(outputs, 1)
    
    return predictions.cpu().numpy()

def evaluate_text_model(model, data_loader, device):
    """Evaluate a text classification model."""
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating Text Model"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
    return np.array(predictions), np.array(actual_labels)


def main(args):
    set_seed()
    device = get_device()
    print(f"Using device: {device}")

    if args.model_type == 'fusion':
        print("--- Evaluating Full Multimodal Model ---")
        try:
            text_f = np.load('features/text_features.npy')
            image_f = np.load('features/image_features.npy')
            sarcasm_f = np.load('features/sarcasm_features.npy')
            emotion_f = np.load('features/emotion_features.npy')
            labels = np.load('features/labels.npy')
        except FileNotFoundError as e:
            print(f"Error loading feature file: {e}. Please run precompute_features.py first.")
            return

        full_features = np.concatenate([text_f, image_f, sarcasm_f, emotion_f], axis=1)
        input_dim = full_features.shape[1]
        
        model = FusionMLP(input_dim=input_dim)
        try:
            model.load_state_dict(torch.load(args.fusion_model_path, map_location=device))
        except FileNotFoundError:
            print(f"Fusion model not found at {args.fusion_model_path}. Please run fusion_train.py first.")
            return

        predictions = evaluate_fusion_model(model, full_features, device)
        cm_path = os.path.join(args.figures_save_path, 'confusion_matrix_fusion.png')
        model_title = 'Full Multimodal Model'

    elif args.model_type == 'text':
        print("--- Evaluating Text-Only Model ---")
        MODEL_NAME = 'distilbert-base-uncased'
        
        try:
            model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
            model.load_state_dict(torch.load(args.text_model_path, map_location=device))
            model.to(device)
            tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        except FileNotFoundError:
            print(f"Text model not found at {args.text_model_path}. Please run train_text.py first.")
            return

        print(f"Loading evaluation data from {args.data_path}")
        df = load_olid_data(args.data_path)
        
        dataset = TextDataset(
            texts=df.tweet.to_numpy(),
            labels=df.label.to_numpy(),
            tokenizer=tokenizer,
            max_len=128
        )
        data_loader = DataLoader(dataset, batch_size=16)

        predictions, labels = evaluate_text_model(model, data_loader, device)
        cm_path = os.path.join(args.figures_save_path, 'confusion_matrix_text_only.png')
        model_title = 'Text-Only Model'
    
    else:
        print(f"Unknown model type: {args.model_type}")
        return

    # --- Generate and Save Report ---
    print(f"\nClassification Report ({model_title}):")
    print(classification_report(labels, predictions, target_names=['NOT HATE', 'HATE'], zero_division=0))
    
    # --- Generate and Save Confusion Matrix ---
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NOT HATE', 'HATE'])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues')
    plt.title(f'Confusion Matrix ({model_title})')
    
    os.makedirs(args.figures_save_path, exist_ok=True)
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='fusion', choices=['fusion', 'text'], help='Type of model to evaluate.')
    parser.add_argument('--data_path', type=str, default='data/olid-training-v1.0.tsv', help='Path to the evaluation data file (used for text model evaluation).')
    parser.add_argument('--text_model_path', type=str, default='models/text_model.bin', help='Path to the text model file.')
    parser.add_argument('--fusion_model_path', type=str, default='models/fusion_model.bin', help='Path to the fusion model file.')
    parser.add_argument('--figures_save_path', type=str, default='figures/', help='Directory to save the confusion matrix plot.')
    args = parser.parse_args()
    main(args)
