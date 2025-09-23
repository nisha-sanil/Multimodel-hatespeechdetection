import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

from fusion_train import FusionMLP
from utils import set_seed, get_device

def evaluate_model(model, features, labels, device):
    """Evaluate a given model on a set of features and labels."""
    model.to(device)
    model.eval()
    
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = model(features_tensor)
        _, predictions = torch.max(outputs, 1)
    
    return predictions.cpu().numpy()

def main():
    set_seed()
    device = get_device()
    print(f"Using device: {device}")

    # Load all features and labels
    text_f = np.load('features/text_features.npy')
    image_f = np.load('features/image_features.npy')
    sarcasm_f = np.load('features/sarcasm_features.npy')
    emotion_f = np.load('features/emotion_features.npy')
    labels = np.load('features/labels.npy')

    # Define feature combinations to test
    feature_sets = {
        "Text Only": text_f,
        "Text + Emotion": np.concatenate([text_f, emotion_f], axis=1),
        "Text + Emotion + Sarcasm": np.concatenate([text_f, emotion_f, sarcasm_f], axis=1),
        "Full Multimodal": np.concatenate([text_f, image_f, sarcasm_f, emotion_f], axis=1)
    }

    # Load the trained fusion model
    # Note: We use the same architecture but the input dimension will vary.
    # For a fair comparison, one should train a separate MLP for each feature set.
    # Here, for simplicity, we'll load the full model and evaluate it on the full feature set.
    
    print("--- Evaluating Full Multimodal Model ---")
    
    full_features = feature_sets["Full Multimodal"]
    input_dim = full_features.shape[1]
    
    model = FusionMLP(input_dim=input_dim)
    try:
        model.load_state_dict(torch.load('models/fusion_model.bin', map_location=device))
    except FileNotFoundError:
        print("Fusion model not found. Please run fusion_train.py first.")
        return

    predictions = evaluate_model(model, full_features, labels, device)
    
    # --- Generate and Save Report ---
    report = classification_report(labels, predictions, target_names=['NOT HATE', 'HATE'], output_dict=True)
    print("\nClassification Report (Full Model):")
    print(classification_report(labels, predictions, target_names=['NOT HATE', 'HATE']))
    
    # --- Generate and Save Confusion Matrix ---
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NOT HATE', 'HATE'])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues')
    plt.title('Confusion Matrix (Full Model)')
    
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    print(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")

    # --- Comparison (Conceptual) ---
    # To properly compare models, you would need to train an MLP for each feature subset.
    # The code below is a placeholder to show how you would structure the comparison.
    print("\n--- Model Comparison ---")
    print("Note: This is a conceptual comparison. For a fair result, train an MLP for each feature set.")
    
    # Example for Text-only model
    # text_only_model = FusionMLP(input_dim=text_f.shape[1])
    # text_only_model.load_state_dict(torch.load('models/text_only_model.bin'))
    # text_preds = evaluate_model(text_only_model, text_f, labels, device)
    # text_f1 = f1_score(labels, text_preds, average='macro')
    # print(f"Text Only F1: {text_f1:.4f}")

    full_f1 = report['macro avg']['f1-score']
    print(f"Full Multimodal Macro F1: {full_f1:.4f}")
    print("\nTo get comparison metrics, re-run `fusion_train.py` for each subset of features and save separate models.")

if __name__ == '__main__':
    main()