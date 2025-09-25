import torch
from torch.utils.data import DataLoader, random_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import os
import argparse

from utils import set_seed, get_device, load_olid_data, TextDataset


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader, desc="Training"):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model( # Get model outputs (logits)
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        # Manually calculate loss using our weighted loss function
        loss = loss_fn(logits, labels)

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.item() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = loss_fn(outputs.logits, labels) # Use the same loss function for consistency
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            losses.append(loss.item())

    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / n_examples
    f1 = f1_score(all_labels, all_preds, average='macro')

    return accuracy, np.mean(losses), f1

def main(args):
    set_seed()
    device = get_device()
    print(f"Using device: {device}")

    # Config
    MODEL_NAME = 'roberta-base' # <-- Upgrade to RoBERTa
    MAX_LEN = 128
    BATCH_SIZE = 16 if device.type == 'cuda' else 4
    EPOCHS = 3
    LEARNING_RATE = 2e-5

    # Load data and tokenizer
    print(f"Loading data from {args.data_path}")
    df = load_olid_data(args.data_path)
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME) # <-- Use RoBERTa Tokenizer
    
    dataset = TextDataset(
        texts=df.tweet.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2) # <-- Use RoBERTa Model
    model = model.to(device)

    # --- Handle Class Imbalance with Class Weights ---
    # Count the occurrences of each class in the full dataset
    class_counts = np.bincount(df.label.to_numpy())
    # Calculate weights: inverse of the class frequency
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() # Normalize
    class_weights = class_weights.to(device)
    print(f"Using class weights to handle imbalance: {class_weights.cpu().numpy()}")

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
    best_val_f1 = -1.0 # Track best F1 score instead of loss
    early_stopping_patience = 2
    early_stopping_counter = 0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        # Pass the weighted loss_fn to the training function
        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler, len(train_dataset))
        print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f}')
        val_acc, val_loss, val_f1 = eval_model(model, val_loader, loss_fn, device, len(val_dataset))
        print(f'Val loss {val_loss:.4f} accuracy {val_acc:.4f} F1-score {val_f1:.4f}')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
            torch.save(model.state_dict(), args.model_save_path)
            print(f"Best model saved to {args.model_save_path}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/olid-training-v1.0.tsv', help='Path to the OLID training data file.')
    parser.add_argument('--model_save_path', type=str, default='models/text_model.bin', help='Path to save the trained text model.')
    args = parser.parse_args()
    main(args)