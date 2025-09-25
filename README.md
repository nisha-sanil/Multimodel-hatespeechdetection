# Multimodal Hate Speech Detection

This project aims to detect hate speech from text, images (memes), and auxiliary cues like sarcasm and emotion using a multimodal fusion model.
The system is designed to be trainable on a Colab GPU and runnable for inference on a standard CPU.

> **Note:** The main branch of this repository does not contain the large dataset files to stay within GitHub's limits. The training scripts assume you have downloaded the data as described in the Setup section.
 
## 📂 Project Structure

```
├── data/
│   ├── olid_sample.csv         # Text hate speech data
│   ├── isarcasm_sample.csv     # Sarcasm detection data
│   ├── goemotions_sample.csv   # Emotion detection data
│   ├── hateful_memes_sample.csv # Hateful memes data (text + image paths)
│   └── img/                    # Directory for meme images (e.g., 01234.png)
├── src/
│   ├── train_text.py           # Fine-tunes RoBERTa on OLID
│   ├── train_aux.py            # Trains sarcasm & emotion classifiers
│   ├── precompute_features.py  # Extracts and caches all features
│   ├── fusion_train.py         # Trains the final fusion MLP
│   ├── evaluate.py             # Evaluates models and generates reports
│   ├── demo.py                 # Command-line inference tool
│   └── utils.py                # Helper functions
├── models/                       # Stores trained model checkpoints
├── features/                     # Stores precomputed feature embeddings
├── app.py                        # Streamlit web application
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

## ⚙️ Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Data:**
    - The `data/` directory contains small samples. For full training, download the original datasets:
      - OLID (OffensEval 2019)
      - iSarcasm
      - GoEmotions
      - Hateful Memes Challenge
    - Place the `img` folder from the Hateful Memes dataset into the `data/` directory.

## 🚀 Training & Evaluation Workflow

Execute the scripts in the following order. The models and features will be saved automatically to the `models/` and `features/` directories.

1.  **Train Text Model:**
    Fine-tune RoBERTa for hate speech classification on the text dataset.
    (Using the full OLID dataset)
    ```bash
    python src/train_text.py --data_path data/olid-training-v1.0.tsv
    ```

2.  **Train Auxiliary Models:**
    Train classifiers for sarcasm and emotion.
    (Using the full sarcasm and emotion datasets)
    ```bash
    python src/train_aux.py --sarcasm_path data/train-balanced-sarcasm.csv --emotion_path data/go_emotions_dataset.csv
    ```

3.  **Precompute All Features:**
    Generate and cache embeddings for text, images, sarcasm, and emotion. This step is crucial for efficient fusion model training.
    (Using the full Hateful Memes dataset)
    ```bash
    python src/precompute_features.py --data_path data/train.jsonl
    ```

4.  **Train Fusion Model:**
    Train the final MLP that fuses all features.
    ```bash
    python src/fusion_train.py
    ```

5.  **Evaluate Models:**
    Generate classification reports, confusion matrices, and performance comparisons.
    To evaluate just the text model:
    ```bash
    python src/evaluate.py --model_type text --data_path data/olid-training-v1.0.tsv
    ```
    To evaluate the final fusion model (after all training is complete):
    ```bash
    python src/evaluate.py --model_type fusion
    ```

## ✅ Usage

### Command-Line Demo

Run a prediction on a single instance using the CLI.
```bash
python src/demo.py --text "this is a test" --image_path "data/img/12345.png"
```

### Streamlit Web App

Launch the interactive web UI.
```bash
streamlit run app.py
```
Then open your browser to the URL provided (e.g., `http://localhost:8501`).