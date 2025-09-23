import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

from utils import set_seed, load_sarcasm_data, load_emotion_data

def train_classifier(df, text_col, label_col, model_path):
    """Train a simple text classifier and save it."""
    set_seed()
    
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col], df[label_col], test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(solver='liblinear', random_state=42))
    ])

    print(f"Training {os.path.basename(model_path)}...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print(f"Evaluation for {os.path.basename(model_path)}:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")
    print("-" * 30)

def main():
    # --- Sarcasm Model ---
    sarcasm_df = load_sarcasm_data('data/isarcasm_sample.csv')
    # Ensure the text column is string type
    sarcasm_df['tweet'] = sarcasm_df['tweet'].astype(str)
    train_classifier(
        df=sarcasm_df,
        text_col='tweet',
        label_col='sarcastic',
        model_path='models/sarcasm_model.joblib'
    )

    # --- Emotion Model ---
    emotion_df = load_emotion_data('data/goemotions_sample.csv')
    emotion_df['text'] = emotion_df['text'].astype(str)
    train_classifier(
        df=emotion_df,
        text_col='text',
        label_col='emotion',
        model_path='models/emotion_model.joblib'
    )

if __name__ == '__main__':
    main()