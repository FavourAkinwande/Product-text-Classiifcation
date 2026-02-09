## Comparative Analysis of Text Classification with Multiple Embeddings

This repository contains the implementation for a **group assignment** on:

> **Comparative Analysis of Text Classification with Multiple Embeddings**

We perform text classification on **Amazon product titles** into product categories and systematically compare **multiple model architectures** and **multiple word embedding techniques**. Each team member owns one model family and evaluates it across several embeddings, all built on top of a **shared preprocessing pipeline** and a common experimental protocol.

The code and results in this repo support a **research‑style PDF report** with clear, reproducible experiments, comparison tables, visualizations, and research‑backed analysis.

---

## Objective & Team Scope

**High‑level objective**

- Design, implement, and evaluate **text classification systems**.
- Each team member explores **one model architecture** across **multiple word embeddings**.
- As a team we:
  - Share the **same dataset** and **core preprocessing**.
  - Coordinate on **embedding types** so results are directly comparable.
  - Produce a **research‑style report (PDF)** plus a **well‑structured GitHub repo**.

**Models implemented in this repository**

In line with the assignment guideline (one classical model + sequence models), this repo includes:

- **Traditional machine learning model**
  - **Logistic Regression**
- **Sequence models**
  - **RNN** (simple recurrent network, called `RNN` in code)
  - **GRU** (implemented in the GRU notebook)

Each model is evaluated with multiple embeddings from the assignment’s list:

- **TF‑IDF**
- **Word2Vec Skip‑gram**
- **Word2Vec CBOW**
- **FastText‑style averaged word embeddings** (for Logistic Regression)

> The PDF report (outside this repo) contains the **contribution tracker** and clearly states which team member implemented which model and sections.

---

## Dataset (Phase 1: Selection & Exploration)

- **Domain**: Product title classification for Amazon products.
- **File**: `data/titles_to_categories.csv`
- **Inputs**: Short product titles (free‑text).
- **Labels**: Product categories (63 classes; see `results/label_mapping.csv` and `results/RNN/label_mapping.csv`).
- **Task**: Multi‑class text classification – predict the category from the title.

**Exploratory data analysis (EDA)**

- Performed in `notebook/eda.ipynb`.
- Key artifacts saved under `results/EDA/`:
  - `preprocessing_metadata.json` — dataset sizes, splits, class stats.
  - `label_mapping.csv` — category → label_id mapping used across experiments.

The EDA motivates:

- The **class imbalance handling** (via stratified splitting).
- The **text cleaning** decisions (lowercasing, punctuation handling, etc.).
- The choice of **sequence length** and **vocabulary handling** for neural models.

---

## Repository Structure

```text
Product-text-Classiifcation/
├── data/
│   └── titles_to_categories.csv                 # Raw dataset
├── results/
│   ├── preprocessing_metadata.json              # Global preprocessing config and stats
│   ├── label_mapping.csv                        # Global category ↔ label_id mapping
│   ├── train.csv, test.csv                      # Preprocessed splits (from preprocessing.py)
│   ├── RNN/                                     # Simple RNN experiments (main model)
│   │   ├── hyperparameter_runs.csv              # All RNN runs across embeddings
│   │   ├── best_configurations.json             # Best hyperparameters per embedding
│   │   ├── test_evaluation_results.json         # Test metrics per embedding
│   │   ├── label_mapping.csv
│   │   ├── model_comparison_results.csv         # RNN: embedding vs. metrics table
│   │   ├── classification_report_*.txt
│   │   └── visual/                              # Confusion matrices, curves, comparison plots
│   ├── GRU/
│   │   ├── model_comparison_results.csv
│   │   ├── classification_report_*.txt
│   │   └── visual_graphs/                       # GRU confusion matrices & comparison plots
│   └── LogisticRegression/
│       ├── experiment_summary.json
│       ├── model_comparison_results.csv
│       ├── classification_report_*.txt
│       └── visual_graphs/                       # Confusion matrices & model comparison plots
├── notebook/
│   ├── eda.ipynb
│   ├── GRU_Text_Classification_Complete_FULL.ipynb
│   ├── LSTM.ipynb
│   └── LogisticRegression_Text_Classification_Complete.ipynb
├── src/
│   ├── preprocessing.py                         # Shared preprocessing pipeline
│   ├── experiment.py                            # RNN (simple RNN) × TF‑IDF / Skip‑gram / CBOW
│   └── logistic_regression_experiments.py       # Logistic Regression × embeddings
├── requirements.txt                             # Main deep learning requirements
├── requirements_logreg.txt                      # Extra deps for classical model experiments
└── README.md
```

---

## Shared Preprocessing & Embedding Strategy

**Script**: `src/preprocessing.py`

Shared pipeline (used by all models):

1. **Load data**
   - Read `data/titles_to_categories.csv` (configurable `NROWS`).
2. **Clean text**
   - Lowercase.
   - Remove punctuation.
   - Keep relevant numeric tokens where appropriate.
3. **Label encoding**
   - Map category strings → integer labels.
   - Save mapping to `results/label_mapping.csv`.
4. **Stratified train/test split**
   - Fixed `RANDOM_SEED` for reproducibility.
   - Split statistics stored in `results/preprocessing_metadata.json`.
5. **Persist artifacts**
   - `results/train.csv`, `results/test.csv`
   - `results/label_mapping.csv`
   - `results/preprocessing_metadata.json`

**Embedding‑specific preprocessing** (implemented across `preprocessing.py`, `experiment.py`, `logistic_regression_experiments.py`, and notebooks):

- **TF‑IDF**
  - Apply shared `clean_text`.
  - Build bag‑of‑words / bag‑of‑ngrams \((1, 2)\).
  - Optionally apply **SVD** for dimensionality reduction before feeding RNN/GRU.

- **Word2Vec Skip‑gram / CBOW**
  - Same cleaned text, tokenized to word sequences.
  - Train Word2Vec **only on training titles**.
  - Use fixed maximum sequence length (e.g. `max_len = 30`) with padding and truncation.
  - Build sequence representations by mapping tokens to embedding vectors.

- **FastText‑style averaged embeddings (Logistic Regression only)**
  - Obtain word embeddings (Skip‑gram/CBOW/FastText‑like).
  - Average word vectors for each title → one dense vector per sample.

These choices are justified and cited in the **PDF report** with relevant literature (e.g. comparisons of TF‑IDF vs. neural embeddings, embedding dimensionality, and context windows).

---

## Models & Individual Responsibilities (Phase 2)

Each team member is responsible for **one model architecture**, but the code is organised by model family rather than by person.

### Logistic Regression (traditional ML baseline)

- **Code**: `src/logistic_regression_experiments.py` and `notebook/LogisticRegression_Text_Classification_Complete.ipynb`
- **Embeddings**:
  - TF‑IDF
  - Word2Vec Skip‑gram
  - Word2Vec CBOW
  - FastText‑style averaged word embeddings
- **Outputs** (under `results/LogisticRegression/`):
  - `model_comparison_results.csv` — metrics per embedding.
  - `experiment_summary.json` — configuration and runtime metadata.
  - `classification_report_*.txt` — precision, recall, macro/weighted F1.
  - `visual_graphs/` — confusion matrices and model comparison plots.

### RNN (simple recurrent network)

- **Code**: `src/experiment.py` and `notebook/LSTM.ipynb`
- **Embeddings**:
  - TF‑IDF + SVD
  - Word2Vec Skip‑gram
  - Word2Vec CBOW
- **Hyperparameter tuning**:
  - ~12 configurations per embedding (RNN units, dropout, learning rate, batch size).
  - Best run selected by **validation macro‑F1**.
- **Outputs** (under `results/RNN/`):
  - `hyperparameter_runs.csv` — all configs and validation macro‑F1.
  - `best_configurations.json` — best hyperparameters per embedding.
  - `test_evaluation_results.json` — test metrics (accuracy, macro/weighted F1).
  - `model_comparison_results.csv` — RNN comparison table across embeddings.
  - `classification_report_*.txt` — detailed per‑class metrics.
  - `visual/` — confusion matrices, training curves, model comparison plots.

### GRU (sequence model)

- **Code**: `notebook/GRU_Text_Classification_Complete_FULL.ipynb`
- **Embeddings**:
  - TF‑IDF
  - Word2Vec Skip‑gram
  - Word2Vec CBOW
- **Outputs** (under `results/GRU/`):
  - `model_comparison_results.csv` — GRU comparison table across embeddings.
  - `classification_report_*.txt`
  - `visual_graphs/` — confusion matrices, training curves, model comparison plots.

> The **group contribution tracker** (submitted with the PDF report) documents which member implemented each model and wrote each section.

---

## How to Run the Code

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For classical model experiments (if any dependency is missing):

```bash
pip install -r requirements_logreg.txt
```

### 2. Run preprocessing (required once)

From the project root:

```bash
python src/preprocessing.py
```

This will generate:

- `results/train.csv`
- `results/test.csv`
- `results/label_mapping.csv`
- `results/preprocessing_metadata.json`

You can adjust `DATA_PATH`, `NROWS`, and `TEST_SIZE` in `src/preprocessing.py` if needed.

### 3. Run RNN (simple RNN) experiments

```bash
python src/experiment.py
```

This will:

- Train and evaluate RNN models for each embedding (TF‑IDF+SVD, Skip‑gram, CBOW).
- Run ~12 hyperparameter configurations per embedding.
- Select the best config by **validation macro‑F1**.
- Evaluate best configs on the held‑out test set.
- Write logs and metrics under `results/RNN/`.

### 4. Run Logistic Regression experiments

```bash
python src/logistic_regression_experiments.py
```

or interactively in:

- `notebook/LogisticRegression_Text_Classification_Complete.ipynb`

Outputs are saved under `results/LogisticRegression/`.

### 5. Run GRU experiments

Open and run:

- `notebook/GRU_Text_Classification_Complete_FULL.ipynb`

This mirrors the RNN setup with GRU layers and writes outputs under `results/GRU/`.

---

## Results & Comparative Analysis (Phase 3)

Key comparison files used in the report:

- **Per‑model comparison tables**
  - `results/RNN/model_comparison_results.csv`
  - `results/GRU/model_comparison_results.csv`
  - `results/LogisticRegression/model_comparison_results.csv`
- **Hyperparameter table (RNN)**
  - `results/RNN/hyperparameter_runs.csv` — all configs with validation macro‑F1.
- **Per‑embedding diagnostics**
  - `classification_report_*.txt` — per‑class precision, recall, macro/weighted F1.
  - Confusion matrix CSVs and PNGs in:
    - `results/RNN/visual/`
    - `results/GRU/visual_graphs/`
    - `results/LogisticRegression/visual_graphs/`

The **PDF report** (not part of this repo) uses these artifacts to:

- Build at least **two comparison tables** across models and embeddings.
- Include **visual comparisons** (confusion matrices, training curves, model comparison plots).
- Provide **deep analysis** explaining:
  - Why some embeddings work better for RNN/GRU than for Logistic Regression.
  - How embedding dimensionality, context windows, and training objectives impact results.
  - Limitations (e.g. class imbalance, short titles) and directions for future work.

---

## Requirements & Reproducibility

- **Python**: 3.10+
- **Core libraries** (see `requirements.txt` for exact versions):
  - TensorFlow / Keras
  - scikit‑learn
  - pandas, numpy
  - gensim
  - matplotlib, seaborn
  - tqdm, nltk
- **Additional packages** (for some classical/embedding experiments):
  - See `requirements_logreg.txt`.

Reproducibility measures:

- Fixed random seeds where possible.
- Stratified splits with metadata saved to disk.
- Clear separation between preprocessing and model‑specific scripts.

---

## Alignment with Assignment Rubric

| Rubric item | How this repo/report addresses it |
|-------------|-----------------------------------|
| **Problem Definition & Dataset Justification** | The README and report introduction clearly define product title classification and justify the Amazon dataset as a realistic multi‑class text classification problem. |
| **Dataset Exploration, Preprocessing & Embedding Strategy** | EDA notebook plus `preprocessing.py` define a shared pipeline. Embedding‑specific adaptations (TF‑IDF, Skip‑gram, CBOW, FastText‑style) are implemented and described here and in the report. |
| **Model Implementation & Experimental Design** | Each model family (Logistic Regression, RNN, GRU) is implemented separately and evaluated with ≥3 embeddings, with clear hyperparameter tuning and training strategies. |
| **Experiment Tables (≥2)** | `results/RNN/hyperparameter_runs.csv` and the three `model_comparison_results.csv` files provide the basis for the required experiment tables. |
| **Results & Comparative Discussion** | Accuracy, macro/weighted F1, and confusion matrices are computed for all runs. The report uses these for detailed model‑embedding comparisons and literature‑backed discussion. |
| **Code Quality & GitHub Repository** | Code is modular (shared preprocessing + model‑specific experiments). This README explains structure, how to run code, and where to find results. |
| **Academic Writing, Citations & Originality** | The accompanying PDF report follows an academic structure (intro, literature review, methods, results, discussion, conclusion, references) with proper citations and a group contribution tracker. |
| **Individual Technical Contribution** | Each student independently implements one model family and evaluates it with ≥3 embeddings; their work is unified here through the shared dataset, preprocessing, and comparative analysis. |

# Product Text Classification

Text classification of Amazon product titles into categories using an **RNN (simple recurrent network)** with three word embedding techniques: **TF-IDF + SVD**, **Word2Vec Skip-gram**, and **Word2Vec CBOW**. This repository contains the shared preprocessing pipeline and the RNN + embeddings experiments (hyperparameter tuning and evaluation).

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

- **Model:** RNN implemented as a **simple recurrent network** (Keras/TensorFlow).
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
| **One model, ≥3 embeddings** | ✓ RNN (simple RNN) with TF-IDF+SVD, Word2Vec Skip-gram, Word2Vec CBOW |
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
