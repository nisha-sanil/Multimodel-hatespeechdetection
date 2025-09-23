import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from utils import set_seed, get_device

class FusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super(FusionMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

def plot_history(history, save_path='figures/'):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'fusion_training_history.png'))
    print(f"Training history plot saved to {save_path}")

def main(args):
    set_seed()
    device = get_device()
    print(f"Using device: {device}")

    # Config
    EPOCHS = 20
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    # Load precomputed features
    text_features = np.load('features/text_features.npy')
    image_features = np.load('features/image_features.npy')
    sarcasm_features = np.load('features/sarcasm_features.npy')
    emotion_features = np.load('features/emotion_features.npy')
    labels = np.load('features/labels.npy')

    # Concatenate features
    all_features = np.concatenate([
        text_features,
        image_features,
        sarcasm_features,
        emotion_features
    ], axis=1)

    # Create PyTorch Dataset and DataLoader
    features_tensor = torch.tensor(all_features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(features_tensor, labels_tensor)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model, optimizer, etc.
    input_dim = all_features.shape[1]
    model = FusionMLP(input_dim=input_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print("Training Fusion MLP...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validation
        model.eval()
        val_loss, val_correct, val_samples = 0, 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                val_samples += labels.size(0)

        val_loss /= val_samples
        val_acc = val_correct / val_samples
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
            torch.save(model.state_dict(), args.model_save_path)
            print(f"Best model saved to {args.model_save_path}")

    plot_history(history)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save_path', type=str, default='models/fusion_model.bin', help='Path to save the trained fusion model.')
    args = parser.parse_args()
    main(args)