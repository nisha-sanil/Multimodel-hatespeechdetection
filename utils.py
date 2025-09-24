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
    # The original dataset is a tab-separated file (.tsv)
    df = pd.read_csv(file_path, sep='\t')
    df['label'] = df['subtask_a'].apply(lambda x: 1 if x == 'OFF' else 0)
    return df[['tweet', 'label']]

def load_hateful_memes_data(file_path):
    """Load Hateful Memes dataset."""
    # The original dataset is a jsonlines file (.jsonl)
    if file_path.endswith('.jsonl'):
        df = pd.read_json(file_path, lines=True)
    else: # For compatibility with the sample csv
        df = pd.read_csv(file_path)
    return df

def load_sarcasm_data(file_path):
    """Load iSarcasm dataset."""
    df = pd.read_csv(file_path)
    # The full dataset has 'comment' and 'label' columns
    if 'comment' in df.columns:
        df.rename(columns={'comment': 'tweet', 'label': 'sarcastic'}, inplace=True)
    return df[['tweet', 'sarcastic']].dropna()

def load_emotion_data(file_path):
    """Load GoEmotions dataset."""
    df = pd.read_csv(file_path)
    
    # The full GoEmotions dataset uses one-hot encoding across many columns.
    # We need to find which emotion is present for each row.
    if 'emotion' not in df.columns:
        print("GoEmotions full dataset format detected. Converting one-hot to single label column...")
        # The first columns are metadata, the rest are emotion labels.
        # A more robust way would be to define the emotion columns explicitly.
        emotion_cols = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
        
        # Find the first emotion column with a '1' for each row
        df['emotion'] = df[emotion_cols].idxmax(axis=1)
        
        return df[['text', 'emotion']]
    else:
        # This handles the original sample file format
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