import torch
import random
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def get_device():
    """Get the available device (GPU or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_olid_data(file_path):
    """Load OLID dataset and map labels."""
    df = pd.read_csv(file_path)
    df['label'] = df['subtask_a'].apply(lambda x: 1 if x == 'OFF' else 0)
    return df[['tweet', 'label']]

def load_hateful_memes_data(file_path):
    """Load Hateful Memes dataset."""
    df = pd.read_csv(file_path)
    return df

def load_sarcasm_data(file_path):
    """Load iSarcasm dataset."""
    df = pd.read_csv(file_path)
    return df[['tweet', 'sarcastic']]

def load_emotion_data(file_path):
    """Load GoEmotions dataset."""
    df = pd.read_csv(file_path)
    # For simplicity, we'll use the raw text and emotion columns
    return df[['text', 'emotion']]

class TextDataset(Dataset):
    """PyTorch dataset for text classification."""
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }