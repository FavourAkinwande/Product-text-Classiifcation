# Product Text Classification

Text classification of Amazon product titles into categories using an **RNN (Bidirectional LSTM)** with three word embedding techniques: **TF-IDF + SVD**, **Word2Vec Skip-gram**, and **Word2Vec CBOW**. This repository contains the shared preprocessing pipeline and the RNN + embeddings experiments (hyperparameter tuning and evaluation).

## Dataset

- **Source:** Amazon product titles → category (e.g. `data/titles_to_categories.csv`).
- **Task:** Predict product category from title text only.
- **Split:** Stratified train/test (configurable in preprocessing); experiments further split train into train/validation for tuning.

## Repository Structure

```
Product-text-Classiifcation/
├── data/                           # Raw data (titles_to_categories.csv)
├── results/                        # Outputs (created by scripts)
│   ├── train.csv, test.csv         # Preprocessed splits
│   ├── label_mapping.csv
│   ├── preprocessing_metadata.json
│   ├── hyperparameter_runs.csv     # All tuning runs (embedding, rnn_units, dropout, lr, batch_size, val_macro_f1)
│   ├── test_evaluation_results.json# Best config + accuracy, macro_f1, weighted_f1 per embedding
│   ├── model_comparison_results.csv# Table: Model, Embedding, Accuracy, Macro_F1, Weighted_F1 (3 rows)
│   ├── best_configurations.json    # Best hyperparameters per embedding
│   ├── model_tfidf.keras, model_skipgram.keras, model_cbow.keras  # Saved best models
│   ├── confusion_matrix_*.csv     # Raw confusion matrix per embedding
│   ├── classification_report_*.txt# Precision/recall/F1 (macro + weighted avg) per embedding
│   └── *.png                       # Heatmaps, training history, model_comparison.png
├── notebook/
│   └── eda.ipynb                   # Exploratory data analysis
├── src/
│   ├── preprocessing.py            # Shared preprocessing pipeline
│   └── experiments.py              # RNN × TF-IDF / Skip-gram / CBOW experiments
├── requirements.txt
└── README.md
```

## Preprocessing

- **Script:** `src/preprocessing.py`
- **Steps:** Load raw data → clean text (lowercase, remove punctuation, keep numbers) → label encoding → stratified train/test split → save CSVs and metadata.
- **Outputs:** `results/train.csv`, `results/test.csv`, `results/label_mapping.csv`, `results/preprocessing_metadata.json`

Preprocessing is **adapted per embedding** in `experiments.py`:
- **TF-IDF:** Uses shared `clean_text`; no tokenization; bag-of-ngrams (1,2) with SVD reduction.
- **Skip-gram / CBOW:** Same `clean_text`, then tokenized into word sequences for Word2Vec; fixed `max_len=30` for sequences.

## Model and Embeddings

- **Model:** RNN implemented as **Bidirectional LSTM** (Keras/TensorFlow).
- **Embeddings (three):**
  1. **TF-IDF + SVD** — max_features=100000, ngram_range=(1,2), SVD components=300.
  2. **Word2Vec Skip-gram** — vector_size=200, window=5, min_count=2, max_len=30.
  3. **Word2Vec CBOW** — same Word2Vec hyperparameters, training mode CBOW (sg=0).

For each embedding, the same RNN architecture is used; only **RNN units**, **dropout**, **learning rate**, and **batch size** are tuned.

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run preprocessing (required first)

From the project root:

```bash
python src/preprocessing.py
```

This creates `results/train.csv` and `results/test.csv` (and other metadata). Adjust `DATA_PATH`, `NROWS`, and `TEST_SIZE` in `preprocessing.py` if needed.

### 3. Run experiments (RNN × 3 embeddings)

```bash
python src/experiments.py
```

This will:
- For **each** embedding (TF-IDF+SVD, Skip-gram, CBOW): run 12 hyperparameter configurations, select the best by **validation MACRO-F1**, then evaluate **once** on the held-out test set.
- Log every run to `results/hyperparameter_runs.csv`.
- Save best config and test metrics to `results/test_evaluation_results.json`.
- Save a **confusion matrix** per embedding to `results/confusion_matrix_<embedding>.csv`.

## Results and Evaluation

- **hyperparameter_runs.csv:** All runs (embedding, config_id, rnn_units, dropout, learning_rate, batch_size, val_macro_f1, timestamp_utc).
- **test_evaluation_results.json:** For each embedding, the best hyperparameter config and **test accuracy** and **test MACRO-F1**.
- **confusion_matrix_*.csv:** Confusion matrix for the best model per embedding (for report tables/figures).

Evaluation metrics used: **accuracy**, **macro F1**, and **confusion matrix**.

## Requirements

- Python 3.10+
- See `requirements.txt`: TensorFlow, scikit-learn, pandas, numpy, gensim, matplotlib, seaborn.

---

## Project requirements compliance (assignment rubric)

| Requirement | Status |
|-------------|--------|
| **One model, ≥3 embeddings** | ✓ RNN (Bidirectional LSTM) with TF-IDF+SVD, Word2Vec Skip-gram, Word2Vec CBOW |
| **Shared preprocessing** | ✓ `preprocessing.py`: load, clean text, encode labels, stratified train/test |
| **Preprocessing adapted per embedding** | ✓ Documented in code and README: TF-IDF (bag-of-ngrams); Skip-gram/CBOW (tokenized sequences, max_len=30) |
| **Hyperparameter tuning** | ✓ 12 configs per embedding (RNN units, dropout, lr, batch size); best by validation MACRO-F1 |
| **Early stopping, restore best weights** | ✓ Keras EarlyStopping with `restore_best_weights=True` |
| **Experiment tables (≥2)** | ✓ `hyperparameter_runs.csv`, `model_comparison_results.csv`; JSON/confusion CSVs support more tables |
| **Evaluation metrics** | ✓ Accuracy, macro F1, weighted F1, confusion matrix, classification reports (macro + weighted avg) |
| **Visual comparisons** | ✓ Confusion matrix heatmaps (TF-IDF / Skip-gram / CBOW), training history, training_curves.png, model_comparison.png |
| **Saved models** | ✓ `model_tfidf.keras`, `model_skipgram.keras`, `model_cbow.keras` |
| **Code structure** | ✓ Modular: separate `build_*` per embedding; docstrings; run from project root |
| **README** | ✓ Dataset, structure, preprocessing, model, embeddings, how to run, results, requirements |
