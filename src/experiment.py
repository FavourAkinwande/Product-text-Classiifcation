"""
Hyperparameter tuning experiments for product title classification.

RNN evaluated with three embeddings:
  1. TF-IDF + SVD
  2. Word2Vec Skip-gram
  3. Word2Vec CBOW

For EACH embedding: run 6 configs (shuffled grid), select best by validation MACRO-F1,
log all runs, early stopping with restore best weights, then evaluate once on test.
Saves hyperparameter_runs.csv, test_evaluation_results.json, and confusion matrices.

Preprocessing adaptation per embedding 
  - TF-IDF: uses shared clean_text; no tokenization; bag-of-ngrams (1,2).
  - Skip-gram/CBOW: same clean_text, then tokenized into word sequences for
    Word2Vec training; OOV words yield zero vectors at inference.
"""

from __future__ import annotations

import csv
import os
import sys
import itertools
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Enable mixed precision on GPU only (faster training, no effect on CPU)
try:
    import tensorflow as tf
    if tf.config.list_physical_devices("GPU"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass


class _NumpyJSONEncoder(json.JSONEncoder):
    """Encode numpy int/float/array so json.dump never raises TypeError."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =============================================================================
# Fixed constants 
# =============================================================================
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DATA_DIR = RESULTS_DIR
TFIDF_MAX_FEATURES = 100_000
TFIDF_NGRAM_RANGE = (1, 2)
SVD_COMPONENTS = 200
MAX_LEN = 20
W2V_VECTOR_SIZE = 200
W2V_WINDOW = 5
W2V_MIN_COUNT = 5
RANDOM_SEED = 42


FAST_MODE = True
EPOCHS = 8 if FAST_MODE else 15
PATIENCE = 2 if FAST_MODE else 3
W2V_TRAIN_EPOCHS = 3 if FAST_MODE else 5
BATCH_SIZE_OPTIONS = [256, 512]
TFIDF_MAX_FEATURES_USE = 50_000 if FAST_MODE else TFIDF_MAX_FEATURES
SVD_COMPONENTS_USE = 100 if FAST_MODE else SVD_COMPONENTS

# Word2Vec cache:  save to results
W2V_SAVE_SKIPGRAM = RESULTS_DIR / "w2v_skipgram.model"
W2V_SAVE_CBOW = RESULTS_DIR / "w2v_cbow.model"

# Tuning grid: 6 configs per embedding
RNN_UNITS_OPTIONS = [64, 128]
DROPOUT_OPTIONS = [0.3, 0.5]
LR_OPTIONS = [1e-3, 5e-4]

_full_grid = list(
    itertools.product(
        RNN_UNITS_OPTIONS,
        DROPOUT_OPTIONS,
        LR_OPTIONS,
        BATCH_SIZE_OPTIONS,
    )
)
_rng = np.random.default_rng(RANDOM_SEED)
_rng.shuffle(_full_grid)
TUNING_GRID = _full_grid[:6]


def _get_run_log_path() -> Path:
    return RESULTS_DIR / "hyperparameter_runs.csv"


def _get_test_results_path() -> Path:
    return RESULTS_DIR / "test_evaluation_results.json"


def _get_confusion_matrix_path(embedding_name: str) -> Path:
    return RESULTS_DIR / f"confusion_matrix_{embedding_name}.csv"


def _embedding_short_name(embedding_name: str) -> str:
    """Short name for filenames: tfidf_svd -> tfidf, word2vec_skipgram -> skipgram, word2vec_cbow -> cbow."""
    if embedding_name == "tfidf_svd":
        return "tfidf"
    if embedding_name == "word2vec_skipgram":
        return "skipgram"
    if embedding_name == "word2vec_cbow":
        return "cbow"
    return embedding_name


def _embedding_display_name(embedding_name: str) -> str:
    """Display name for titles/labels: TF-IDF, Skip-gram, CBOW."""
    if embedding_name == "tfidf_svd":
        return "TF-IDF"
    if embedding_name == "word2vec_skipgram":
        return "Skip-gram"
    if embedding_name == "word2vec_cbow":
        return "CBOW"
    return embedding_name


# =============================================================================
# Data loading and splitting
# =============================================================================


def load_splits():
    """Load train and test CSVs from results dir."""
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "train.csv and test.csv not found in results. Run preprocessing first."
        )
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def stratified_train_val_split(train_df, val_frac=0.1, random_state=RANDOM_SEED):
    """Split train into train/val stratified by label_id."""
    X_idx = np.arange(len(train_df))
    y = train_df["label_id"].values
    tr_idx, val_idx = train_test_split(
        X_idx, test_size=val_frac, random_state=random_state, stratify=y
    )
    return train_df.iloc[tr_idx].reset_index(drop=True), train_df.iloc[val_idx].reset_index(drop=True)


# =============================================================================
# Embedding 1: TF-IDF + SVD
# Preprocessing: uses shared clean_text as-is; no tokenization. Suited to
# bag-of-ngrams; SVD reduces dimensionality for RNN input.
# =============================================================================


def build_tfidf_svd(train_texts, val_texts, test_texts):
    """Build TF-IDF + SVD document vectors. Uses TFIDF_MAX_FEATURES_USE, SVD_COMPONENTS_USE (smaller in fast mode)."""
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES_USE,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    svd = TruncatedSVD(n_components=SVD_COMPONENTS_USE, random_state=RANDOM_SEED)
    X_train_svd = svd.fit_transform(X_train)
    X_val_svd = svd.transform(X_val)
    X_test_svd = svd.transform(X_test)

    # RNN receives pseudo-sequence (n, SVD_COMPONENTS_USE, 1): timesteps, 1 feature per step
    def to_seq(X):
        return X.astype(np.float32).reshape(-1, SVD_COMPONENTS_USE, 1)

    return to_seq(X_train_svd), to_seq(X_val_svd), to_seq(X_test_svd)


# =============================================================================
# Embedding 2 & 3: Word2Vec Skip-gram and CBOW
# Preprocessing: same clean_text, then tokenized into word lists for Word2Vec.
# Skip-gram (sg=1) predicts context from target; CBOW (sg=0) predicts target
# from context. Same fixed params: size=200, window=5, min_count=2, max_len=MAX_LEN.
# =============================================================================


def tokenize(text):
    """Tokenize for Word2Vec: lowercase, split on whitespace (matches clean_text)."""
    return str(text).lower().split()


def _sentence_to_matrix_batch(texts, word_vectors):
    """Convert a list of texts to (n, MAX_LEN, W2V_VECTOR_SIZE) float32 matrix. Used for batches/chunks."""
    out = np.zeros((len(texts), MAX_LEN, W2V_VECTOR_SIZE), dtype=np.float32)
    for i, text in enumerate(texts):
        tokens = tokenize(text)
        for j, w in enumerate(tokens[:MAX_LEN]):
            if w in word_vectors:
                out[i, j, :] = word_vectors[w]
    return out


def build_w2v_matrix_chunked(texts, word_vectors, chunk_size=10_000):
    """Build (n, MAX_LEN, W2V_VECTOR_SIZE) matrix in chunks to avoid one huge allocation."""
    n = len(texts)
    chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = _sentence_to_matrix_batch(texts[start:end], word_vectors)
        chunks.append(chunk)
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, MAX_LEN, W2V_VECTOR_SIZE), dtype=np.float32)


class W2VSequence(keras.utils.Sequence):
    """Memory-efficient Sequence for Word2Vec RNN training: builds batches on the fly, reshuffles each epoch."""

    def __init__(self, train_texts, y_train, word_vectors, batch_size, shuffle=True, rng=None, **kwargs):
        super().__init__(**kwargs)
        self.train_texts = train_texts
        self.y_train = np.asarray(y_train)
        self.word_vectors = word_vectors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = rng if rng is not None else np.random.default_rng(RANDOM_SEED)
        self.n = len(train_texts)
        self.indices = np.arange(self.n)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, self.n)
        batch_idx = self.indices[start:end]
        batch_texts = [self.train_texts[i] for i in batch_idx]
        X_batch = _sentence_to_matrix_batch(batch_texts, self.word_vectors)
        y_batch = self.y_train[batch_idx]
        return X_batch, y_batch


def _build_w2v_sequences(train_texts, val_texts, test_texts, sg: int, save_path: Path | None = None):
    """
    Build Word2Vec (Skip-gram if sg=1, CBOW if sg=0).
    If save_path is set and a saved model exists with matching metadata (n_train, max_len, min_count),
    load it instead of training. Otherwise train, then save model and metadata for reuse.
    Returns (word_vectors, train_texts, val_texts, test_texts).
    """
    meta_path = Path(str(save_path) + "_meta.json") if save_path else None
    if save_path and save_path.exists() and meta_path and meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if (meta.get("n_train") == len(train_texts)
                    and meta.get("max_len") == MAX_LEN
                    and meta.get("min_count") == W2V_MIN_COUNT):
                model = Word2Vec.load(str(save_path))
                print(f"  Reusing saved Word2Vec: {save_path.name}")
                return model.wv, train_texts, val_texts, test_texts
        except Exception:
            pass
    train_sentences = [tokenize(t) for t in train_texts]
    model = Word2Vec(
        sentences=train_sentences,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        sg=sg,
        seed=RANDOM_SEED,
        epochs=W2V_TRAIN_EPOCHS,
        workers=max(1, (os.cpu_count() or 4)),
    )
    if save_path:
        model.save(str(save_path))
        with open(meta_path, "w") as f:
            json.dump({
                "n_train": len(train_texts),
                "max_len": MAX_LEN,
                "min_count": W2V_MIN_COUNT,
            }, f, indent=0)
        print(f"  Saved Word2Vec for reuse: {save_path.name}")
    word_vectors = model.wv
    return word_vectors, train_texts, val_texts, test_texts


def build_w2v_skipgram(train_texts, val_texts, test_texts):
    """Word2Vec Skip-gram (sg=1): returns (word_vectors, train_texts, val_texts, test_texts)."""
    return _build_w2v_sequences(train_texts, val_texts, test_texts, sg=1, save_path=W2V_SAVE_SKIPGRAM)


def build_w2v_cbow(train_texts, val_texts, test_texts):
    """Word2Vec CBOW (sg=0): returns (word_vectors, train_texts, val_texts, test_texts)."""
    return _build_w2v_sequences(train_texts, val_texts, test_texts, sg=0, save_path=W2V_SAVE_CBOW)


# =============================================================================
# RNN model
# =============================================================================


def build_rnn_model(
    input_shape,
    num_classes,
    rnn_units,
    dropout,
    learning_rate,
):
    """Single RNN template: SimpleRNN + Dropout + Dense """
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.SimpleRNN(rnn_units, return_sequences=False)(inputs)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# =============================================================================
# Training and evaluation
# =============================================================================


def compute_macro_f1(model, X, y):
    """Predict and return macro F1."""
    y_pred = np.argmax(model.predict(X, batch_size=512, verbose=0), axis=1)
    return float(f1_score(y, y_pred, average="macro", zero_division=0))


def run_single_config(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes,
    rnn_units,
    dropout,
    learning_rate,
    batch_size,
    embedding_name,
    config_id,
    train_generator=None,
    steps_per_epoch=None,
    input_shape=None,
):
    """Train one config with early stopping (restore best weights), return val MACRO-F1 and history.
    If train_generator and steps_per_epoch are provided, use them for fit (memory-efficient Word2Vec);
    otherwise use X_train, y_train. input_shape required when using generator.
    """
    if input_shape is not None:
        shape = input_shape
    else:
        shape = (X_train.shape[1], X_train.shape[2])
    model = build_rnn_model(
        input_shape=shape,
        num_classes=num_classes,
        rnn_units=rnn_units,
        dropout=dropout,
        learning_rate=learning_rate,
    )
    early = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=0,
    )
    if train_generator is not None and steps_per_epoch is not None:
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            callbacks=[early],
            verbose=0,
        )
    else:
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=batch_size,
            callbacks=[early],
            verbose=0,
        )
    val_macro_f1 = compute_macro_f1(model, X_val, y_val)
    return val_macro_f1, model, history


def run_hyperparameter_search(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    num_classes,
    embedding_name,
    train_generator_factory=None,
):
    """Run 8-12 configs, log all runs, select best by val MACRO-F1, evaluate best once on test.
    If train_generator_factory is set, it is called with batch_size and must return (generator, steps_per_epoch);
    then X_train/y_train are not used for training (memory-efficient Word2Vec path).
    """
    runs = []
    best_f1 = -1.0
    best_model = None
    best_config = None
    best_history = None
    w2v_input_shape = (MAX_LEN, W2V_VECTOR_SIZE) if train_generator_factory else None

    for idx, (rnn_units, dropout, lr, batch_size) in enumerate(TUNING_GRID):
        print(f"\n[{embedding_name}] Config {idx + 1}/{len(TUNING_GRID)}: units={rnn_units}, dropout={dropout}, lr={lr}, batch_size={batch_size}")
        if train_generator_factory is not None:
            train_gen, steps_per_epoch = train_generator_factory(batch_size)
            val_macro_f1, model, history = run_single_config(
                X_train=None,
                y_train=None,
                X_val=X_val,
                y_val=y_val,
                num_classes=num_classes,
                rnn_units=rnn_units,
                dropout=dropout,
                learning_rate=lr,
                batch_size=batch_size,
                embedding_name=embedding_name,
                config_id=idx,
                train_generator=train_gen,
                steps_per_epoch=steps_per_epoch,
                input_shape=w2v_input_shape,
            )
        else:
            val_macro_f1, model, history = run_single_config(
                X_train, y_train, X_val, y_val,
                num_classes=num_classes,
                rnn_units=rnn_units,
                dropout=dropout,
                learning_rate=lr,
                batch_size=batch_size,
                embedding_name=embedding_name,
                config_id=idx,
            )
        run_record = {
            "embedding": embedding_name,
            "config_id": idx,
            "rnn_units": rnn_units,
            "dropout": dropout,
            "learning_rate": lr,
            "batch_size": batch_size,
            "val_macro_f1": val_macro_f1,
            "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        runs.append(run_record)
        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            best_model = model
            best_config = run_record
            best_history = history.history if history else None
        # Log immediately so all runs are persisted
        _append_run_log(run_record)

    # Evaluate best config ONCE on test set
    y_pred = np.argmax(best_model.predict(X_test, batch_size=512, verbose=0), axis=1)
    y_test_arr = np.asarray(y_test)
    test_macro_f1 = float(f1_score(y_test_arr, y_pred, average="macro", zero_division=0))
    test_weighted_f1 = float(f1_score(y_test_arr, y_pred, average="weighted", zero_division=0))
    test_acc = float(np.mean(y_pred == y_test_arr))
    cm = confusion_matrix(y_test_arr, y_pred)
    # sklearn default includes macro avg and weighted avg
    clf_report_str = classification_report(
        y_test_arr, y_pred, zero_division=0, digits=4
    )
    clf_report_dict = classification_report(
        y_test_arr, y_pred, output_dict=True, zero_division=0
    )
    return {
        "embedding": embedding_name,
        "best_config": best_config,
        "all_runs": runs,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "test_accuracy": test_acc,
        "confusion_matrix": cm,
        "classification_report_str": clf_report_str,
        "classification_report_dict": clf_report_dict,
        "history": best_history,
        "y_test": y_test_arr,
        "y_pred": y_pred,
        "best_model": best_model,
    }


# output spec requires embedding, config_id, rnn_units, dropout, learning_rate, batch_size, val_macro_f1, timestamp_utc
HYPERPARAMETER_RUNS_COLUMNS = [
    "embedding",
    "config_id",
    "rnn_units",
    "dropout",
    "learning_rate",
    "batch_size",
    "val_macro_f1",
    "timestamp_utc",
]


def _append_run_log(record: dict) -> None:
    """Append one run to hyperparameter_runs.csv. All fields per spec: embedding, config_id, rnn_units, dropout, learning_rate, batch_size, val_macro_f1, timestamp_utc."""
    path = _get_run_log_path()
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HYPERPARAMETER_RUNS_COLUMNS)
        if not file_exists:
            w.writeheader()
        w.writerow({k: record.get(k) for k in HYPERPARAMETER_RUNS_COLUMNS})


def save_test_results(embedding_results: list[dict]) -> None:
    """Save test evaluation JSON (accuracy, macro_f1 per embedding), confusion matrices CSV, model_comparison_results.csv, best_configurations.json."""
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    out = {
        "timestamp_utc": timestamp,
        "experiments": [],
    }
    comparison_rows = []
    best_configs = {}
    for res in embedding_results:
        emb = res["embedding"]
        display = _embedding_display_name(emb)
        out["experiments"].append({
            "embedding": emb,
            "embedding_label": display,
            "accuracy": res["test_accuracy"],
            "macro_f1": res["test_macro_f1"],
            "weighted_f1": res["test_weighted_f1"],
            "best_config": res["best_config"],
            "confusion_matrix_csv": f"confusion_matrix_{emb}.csv",
        })
        # Model comparison table: Model, Embedding, Accuracy, Macro_F1, Weighted_F1 (3 decimal places)
        embedding_row_label = {
            "tfidf_svd": "TF-IDF + SVD",
            "word2vec_skipgram": "Word2Vec Skip-gram",
            "word2vec_cbow": "Word2Vec CBOW",
        }[emb]
        comparison_rows.append({
            "Model": "RNN",
            "Embedding": embedding_row_label,
            "Accuracy": round(res["test_accuracy"], 3),
            "Macro_F1": round(res["test_macro_f1"], 3),
            "Weighted_F1": round(res["test_weighted_f1"], 3),
        })
        best_configs[emb] = {
            "rnn_units": res["best_config"]["rnn_units"],
            "dropout": res["best_config"]["dropout"],
            "learning_rate": res["best_config"]["learning_rate"],
            "batch_size": res["best_config"]["batch_size"],
        }
        cm_path = _get_confusion_matrix_path(emb)
        pd.DataFrame(res["confusion_matrix"]).to_csv(cm_path, index=False, header=False)
        print(f"Saved confusion matrix to {cm_path}")
    path = _get_test_results_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, cls=_NumpyJSONEncoder)
    print(f"Saved test evaluation to {path}")
    # model_comparison_results.csv: Model, Embedding, Accuracy, Macro_F1, Weighted_F1 (3 rows, 3 decimals)
    comparison_path = RESULTS_DIR / "model_comparison_results.csv"
    comparison_df = pd.DataFrame(comparison_rows, columns=["Model", "Embedding", "Accuracy", "Macro_F1", "Weighted_F1"])
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved model comparison to {comparison_path}")
    # best_configurations.json
    best_path = RESULTS_DIR / "best_configurations.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp_utc": timestamp, "configurations": best_configs}, f, indent=2, cls=_NumpyJSONEncoder)
    print(f"Saved best configurations to {best_path}")


def save_best_models(embedding_results: list[dict]) -> None:
    """Save best Keras model per embedding: model_tfidf.keras, model_skipgram.keras, model_cbow.keras."""
    for res in embedding_results:
        model = res.get("best_model")
        if model is None:
            continue
        short = _embedding_short_name(res["embedding"])
        path = RESULTS_DIR / f"model_{short}.keras"
        model.save(path)
        print(f"Saved model to {path}")


# =============================================================================
# Classification reports and visualizations
# =============================================================================


def save_classification_reports(embedding_results: list[dict]) -> None:
    """Save classification report (txt) per embedding. Includes macro avg and weighted avg (sklearn default)."""
    for res in embedding_results:
        emb = res["embedding"]
        display = _embedding_display_name(emb)
        short = _embedding_short_name(emb)
        path = RESULTS_DIR / f"classification_report_{short}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Classification Report — {display}\n")
            f.write("(includes macro avg and weighted avg)\n")
            f.write("=" * 60 + "\n\n")
            f.write(res["classification_report_str"])
        print(f"Saved classification report to {path}")


# Top N classes to show in heatmap (full matrix still saved to CSV)
CONFUSION_HEATMAP_TOP_N = 30


def _plot_heatmap(ax, cm_plot, display: str) -> None:
    """Draw heatmap using seaborn  else matplotlib imshow."""
    if _HAS_SEABORN:
        sns.heatmap(cm_plot, ax=ax, fmt="d", cmap="Blues", cbar_kws={"label": "Count"})
    else:
        im = ax.imshow(cm_plot, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax, label="Count")
        for i in range(cm_plot.shape[0]):
            for j in range(cm_plot.shape[1]):
                ax.text(j, i, int(cm_plot[i, j]), ha="center", va="center", fontsize=6)


def plot_confusion_matrix_png(embedding_results: list[dict]) -> None:
    """Save confusion matrix heatmap PNG per embedding. Full matrix saved to CSV; heatmap shows top 30 classes only. Saves top-30 indices JSON for reproducibility."""
    for res in embedding_results:
        emb = res["embedding"]
        display = _embedding_display_name(emb)
        short = _embedding_short_name(emb)
        cm = res["confusion_matrix"]
        num_classes = cm.shape[0]
        top_idx = None
        if num_classes > CONFUSION_HEATMAP_TOP_N:
            row_sums = cm.sum(axis=1)
            top_idx = np.argsort(row_sums)[-CONFUSION_HEATMAP_TOP_N:]
            top_idx = np.sort(top_idx)
            cm_plot = cm[np.ix_(top_idx, top_idx)]
        else:
            cm_plot = cm
            top_idx = np.arange(num_classes).tolist()
        fig, ax = plt.subplots(figsize=(12, 10))
        _plot_heatmap(ax, cm_plot, display)
        ax.set_title(f"Confusion Matrix — {display}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        path = RESULTS_DIR / f"confusion_matrix_{short}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")
        # Save top-N class indices for reproducibility (Python ints for JSON)
        indices_path = RESULTS_DIR / f"confusion_matrix_top30_indices_{short}.json"
        top_idx_list = [int(x) for x in top_idx]
        with open(indices_path, "w", encoding="utf-8") as f:
            json.dump(
                {"embedding": emb, "top30_class_indices": top_idx_list},
                f,
                indent=2,
            )
        print(f"Saved {indices_path}")


def plot_training_history_png(embedding_results: list[dict]) -> None:
    """Save training history (loss & accuracy curves) per embedding ."""
    for res in embedding_results:
        hist = res.get("history")
        if not hist:
            continue
        emb = res["embedding"]
        short = _embedding_short_name(emb)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(hist["loss"]) + 1)
        ax1.plot(epochs, hist["loss"], label="Train loss")
        ax1.plot(epochs, hist["val_loss"], label="Val loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"Loss — {emb}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.plot(epochs, hist["accuracy"], label="Train accuracy")
        ax2.plot(epochs, hist["val_accuracy"], label="Val accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title(f"Accuracy — {emb}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig.suptitle(f"Training History — {emb}", y=1.02)
        fig.tight_layout()
        path = RESULTS_DIR / f"training_history_{short}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


def plot_training_curves_png(embedding_results: list[dict]) -> None:
    """Save combined training curves (all embeddings in one figure): training_curves.png."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for col, res in enumerate(embedding_results):
        hist = res.get("history")
        if not hist:
            continue
        short = _embedding_short_name(res["embedding"])
        epochs = range(1, len(hist["loss"]) + 1)
        axes[0, col].plot(epochs, hist["loss"], label="Train")
        axes[0, col].plot(epochs, hist["val_loss"], label="Val")
        axes[0, col].set_title(f"Loss — {short}")
        axes[0, col].set_xlabel("Epoch")
        axes[0, col].legend()
        axes[0, col].grid(True, alpha=0.3)
        axes[1, col].plot(epochs, hist["accuracy"], label="Train")
        axes[1, col].plot(epochs, hist["val_accuracy"], label="Val")
        axes[1, col].set_title(f"Accuracy — {short}")
        axes[1, col].set_xlabel("Epoch")
        axes[1, col].legend()
        axes[1, col].grid(True, alpha=0.3)
    fig.suptitle("Training Curves (best run per embedding)", y=1.02)
    fig.tight_layout()
    path = RESULTS_DIR / "training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_model_comparison_png(embedding_results: list[dict]) -> None:
    """Save model comparison bar chart: model_comparison.png (test accuracy & macro F1)."""
    names = [_embedding_display_name(r["embedding"]) for r in embedding_results]
    accs = [r["test_accuracy"] for r in embedding_results]
    f1s = [r["test_macro_f1"] for r in embedding_results]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, accs, width, label="Test Accuracy")
    bars2 = ax.bar(x + width / 2, f1s, width, label="Test Macro F1")
    ax.set_ylabel("Score")
    ax.set_title("Model comparison (best config per embedding)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = RESULTS_DIR / "model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def generate_all_reports_and_plots(embedding_results: list[dict]) -> None:
    """Save classification reports and all visualization PNGs."""
    save_classification_reports(embedding_results)
    plot_confusion_matrix_png(embedding_results)
    plot_training_history_png(embedding_results)
    plot_training_curves_png(embedding_results)
    plot_model_comparison_png(embedding_results)


# =============================================================================
# Main
# =============================================================================


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if FAST_MODE:
        print("FAST_MODE=True: EPOCHS=%d, PATIENCE=%d, TF-IDF %d feat / SVD %d, 6 configs (target ~1 hr)"
              % (EPOCHS, PATIENCE, TFIDF_MAX_FEATURES_USE, SVD_COMPONENTS_USE))

    train_df, test_df = load_splits()
    train_sub, val_df = stratified_train_val_split(train_df)

    train_texts = train_sub["clean_text"].astype(str).tolist()
    val_texts = val_df["clean_text"].astype(str).tolist()
    test_texts = test_df["clean_text"].astype(str).tolist()

    y_train = train_sub["label_id"].values
    y_val = val_df["label_id"].values
    y_test = test_df["label_id"].values
    num_classes = int(train_df["label_id"].max()) + 1

    all_results = []

    # ----- Experiment 1: TF-IDF + SVD -----
    print("\n" + "=" * 60)
    print("Embedding experiment: TF-IDF + SVD")
    print("=" * 60)
    X_tr, X_va, X_te = build_tfidf_svd(train_texts, val_texts, test_texts)
    res_tfidf = run_hyperparameter_search(
        X_tr, y_train, X_va, y_val, X_te, y_test,
        num_classes=num_classes,
        embedding_name="tfidf_svd",
    )
    all_results.append(res_tfidf)

    # ----- Experiment 2: Word2Vec Skip-gram (batched train, chunked val/test to avoid OOM) -----
    print("\n" + "=" * 60)
    print("Embedding experiment: Word2Vec Skip-gram")
    print("=" * 60)
    wv_sg, tr_sg, va_sg, te_sg = build_w2v_skipgram(train_texts, val_texts, test_texts)
    X_va_sg = build_w2v_matrix_chunked(va_sg, wv_sg)
    X_te_sg = build_w2v_matrix_chunked(te_sg, wv_sg)
    def skipgram_gen_factory(batch_size):
        rng = np.random.default_rng(RANDOM_SEED)
        seq = W2VSequence(tr_sg, y_train, wv_sg, batch_size, shuffle=True, rng=rng)
        return seq, len(seq)
    res_skipgram = run_hyperparameter_search(
        X_train=None,
        y_train=y_train,
        X_val=X_va_sg,
        y_val=y_val,
        X_test=X_te_sg,
        y_test=y_test,
        num_classes=num_classes,
        embedding_name="word2vec_skipgram",
        train_generator_factory=skipgram_gen_factory,
    )
    all_results.append(res_skipgram)

    # ----- Experiment 3: Word2Vec CBOW (batched train, chunked val/test to avoid OOM) -----
    print("\n" + "=" * 60)
    print("Embedding experiment: Word2Vec CBOW")
    print("=" * 60)
    wv_cbow, tr_cbow, va_cbow, te_cbow = build_w2v_cbow(train_texts, val_texts, test_texts)
    X_va_cbow = build_w2v_matrix_chunked(va_cbow, wv_cbow)
    X_te_cbow = build_w2v_matrix_chunked(te_cbow, wv_cbow)
    def cbow_gen_factory(batch_size):
        rng = np.random.default_rng(RANDOM_SEED + 1)
        seq = W2VSequence(tr_cbow, y_train, wv_cbow, batch_size, shuffle=True, rng=rng)
        return seq, len(seq)
    res_cbow = run_hyperparameter_search(
        X_train=None,
        y_train=y_train,
        X_val=X_va_cbow,
        y_val=y_val,
        X_test=X_te_cbow,
        y_test=y_test,
        num_classes=num_classes,
        embedding_name="word2vec_cbow",
        train_generator_factory=cbow_gen_factory,
    )
    all_results.append(res_cbow)

    save_test_results(all_results)
    save_best_models(all_results)
    generate_all_reports_and_plots(all_results)
    print("\nDone. All runs logged to", _get_run_log_path())
    print("Test evaluation saved to", _get_test_results_path())
    print("Classification reports and visualizations saved in results/")


def regenerate_missing_plots():
    """Regenerate confusion matrix PNGs, top30 indices JSONs, and model_comparison.png from existing CSVs and test_evaluation_results.json. Use after a run that crashed before saving all plots. """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = [
        ("tfidf_svd", "tfidf"),
        ("word2vec_skipgram", "skipgram"),
        ("word2vec_cbow", "cbow"),
    ]
    for emb, short in embeddings:
        cm_path = _get_confusion_matrix_path(emb)
        if not cm_path.exists():
            print(f"Skipping {emb}: {cm_path} not found")
            continue
        cm = np.array(pd.read_csv(cm_path, header=None))
        num_classes = cm.shape[0]
        if num_classes > CONFUSION_HEATMAP_TOP_N:
            row_sums = cm.sum(axis=1)
            top_idx = np.argsort(row_sums)[-CONFUSION_HEATMAP_TOP_N:]
            top_idx = np.sort(top_idx)
            cm_plot = cm[np.ix_(top_idx, top_idx)]
        else:
            cm_plot = cm
            top_idx = np.arange(num_classes)
        display = _embedding_display_name(emb)
        fig, ax = plt.subplots(figsize=(12, 10))
        _plot_heatmap(ax, cm_plot, display)
        ax.set_title(f"Confusion Matrix — {display}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        path = RESULTS_DIR / f"confusion_matrix_{short}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")
        indices_path = RESULTS_DIR / f"confusion_matrix_top30_indices_{short}.json"
        top_idx_list = [int(x) for x in top_idx]
        with open(indices_path, "w", encoding="utf-8") as f:
            json.dump({"embedding": emb, "top30_class_indices": top_idx_list}, f, indent=2)
        print(f"Saved {indices_path}")
    # model_comparison.png from test_evaluation_results.json
    jeval = _get_test_results_path()
    if jeval.exists():
        with open(jeval, encoding="utf-8") as f:
            data = json.load(f)
        names = []
        accs = []
        f1s = []
        for ex in data.get("experiments", []):
            names.append(ex.get("embedding_label", ex.get("embedding", "")))
            accs.append(float(ex.get("accuracy", 0)))
            f1s.append(float(ex.get("macro_f1", 0)))
        if names:
            x = np.arange(len(names))
            width = 0.35
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(x - width / 2, accs, width, label="Test Accuracy")
            ax.bar(x + width / 2, f1s, width, label="Test Macro F1")
            ax.set_ylabel("Score")
            ax.set_title("Model comparison (best config per embedding)")
            ax.set_xticks(x)
            ax.set_xticklabels(names)
            ax.legend()
            ax.set_ylim(0, 1.0)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            path = RESULTS_DIR / "model_comparison.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {path}")
    print("Done. Training history/curves PNGs need a full run.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--regenerate":
        regenerate_missing_plots()
    else:
        main()
