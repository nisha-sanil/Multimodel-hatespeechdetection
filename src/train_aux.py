import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import argparse

from utils import set_seed, load_sarcasm_data, load_emotion_data

def predict_aux_model(pipeline, texts):
    """Make predictions using a trained auxiliary model pipeline."""
    predictions = pipeline.predict(texts)
    return predictions

def train_classifier(df, text_col, label_col, model_path):
    """Train a simple text classifier and save it."""
    set_seed()
    
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col], df[label_col], test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(solver='liblinear', random_state=42, max_iter=1000))
    ])

    print(f"Training {os.path.basename(model_path)}...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = predict_aux_model(pipeline, X_test)
    print(f"Evaluation for {os.path.basename(model_path)}:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")
    print("-" * 30)

def main(args):
    # --- Sarcasm Model ---
    print(f"Loading sarcasm data from {args.sarcasm_path}")
    sarcasm_df = load_sarcasm_data(args.sarcasm_path)
    # Ensure the text column is string type
    sarcasm_df['tweet'] = sarcasm_df['tweet'].astype(str)
    train_classifier(
        df=sarcasm_df,
        text_col='tweet',
        label_col='sarcastic',
        model_path=args.sarcasm_model_save_path
    )

    # --- Emotion Model ---
    print(f"Loading emotion data from {args.emotion_path}")
    emotion_df = load_emotion_data(args.emotion_path)
    emotion_df['text'] = emotion_df['text'].astype(str)
    train_classifier(
        df=emotion_df,
        text_col='text',
        label_col='emotion',
        model_path=args.emotion_model_save_path
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sarcasm_path', type=str, default='data/isarcasm_sample.csv', help='Path to the sarcasm data file.')
    parser.add_argument('--emotion_path', type=str, default='data/goemotions_sample.csv', help='Path to the emotion data file.')
    parser.add_argument('--sarcasm_model_save_path', type=str, default='models/sarcasm_model.joblib')
    parser.add_argument('--emotion_model_save_path', type=str, default='models/emotion_model.joblib')

    args = parser.parse_args()
    main(args)