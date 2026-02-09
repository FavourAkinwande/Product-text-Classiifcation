## Comparative Analysis of Text Classification with Multiple Embeddings

This project presents a comparative study of text classification using multiple word embedding techniques and machine learning models. The dataset used consists of product titles for category classification, where the goal is to automatically assign each title to its correct product class.

Each team member investigates one model architecture and trains it using different embedding approaches such as TF-IDF and Word2Vec (CBOW and Skip-gram). The project evaluates how embedding choice affects classification performance through systematic experiments and standard metrics.
Overall, the study highlights key differences between embedding–model combinations while emphasizing reproducible experimentation and data-driven evaluation of text classification methods.

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
  - **RNN** (simple recurrent network, implemented in `src/experiment.py`)
  - **LSTM** (implemented in `notebook/LSTM.ipynb`)
  - **GRU** (implemented in `notebook/GRU_Text_Classification_Complete_FULL.ipynb`)

Each model is evaluated with multiple embeddings :

- **TF‑IDF**
- **Word2Vec Skip‑gram**
- **Word2Vec CBOW**
- **FastText‑style averaged word embeddings** (for Logistic Regression)

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

- **Code**: `src/experiment.py`
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

### LSTM (notebook‑based sequence model)

- **Code**: `notebook/LSTM.ipynb`
- **Purpose**:
  - Provides an additional LSTM‑based sequence model on the same dataset for exploratory comparison.
  - Reuses the shared preprocessing and label mapping.
- **Embeddings**:
  - Uses the same family of embeddings (e.g. TF‑IDF or Word2Vec variants) as the main RNN experiments, but orchestrated interactively in the notebook.
- **Outputs**:
  - Plots and metrics are generated within the notebook; key findings are summarized in the PDF report alongside the scripted RNN/GRU/LogReg results.

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
