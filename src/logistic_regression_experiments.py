"""
Logistic Regression experiments with multiple word embeddings.

This script trains and evaluates Logistic Regression classifiers using:
1. TF-IDF features
2. Word2Vec Skip-gram embeddings
3. Word2Vec CBOW embeddings
4. FastText embeddings (optional)

Each embedding approach is tuned with GridSearchCV and evaluated on the test set.
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from gensim.models import FastText, Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

# =========================
# Configuration
# =========================

RESULTS_DIR = "../results"
MODELS_DIR = "../models"
TRAIN_DATA = os.path.join(RESULTS_DIR, "train.csv")
TEST_DATA = os.path.join(RESULTS_DIR, "test.csv")

# Columns in preprocessed data
TEXT_COL = "clean_text"
LABEL_COL = "label_id"

# Random seed for reproducibility
RANDOM_SEED = 42

# Word2Vec/FastText parameters
EMBEDDING_DIM = 100
MIN_WORD_COUNT = 2
WINDOW_SIZE = 5
WORKERS = 4

# Logistic Regression hyperparameter grid
PARAM_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "penalty": ["l2"],
    "solver": ["lbfgs"],
    "max_iter": [500],
    "class_weight": ["balanced"],
}

# Cross-validation settings
CV_FOLDS = 3


def ensure_dirs() -> None:
    """Create necessary directories."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "experiments"), exist_ok=True)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load preprocessed train and test data."""
    print(f"Loading data from {RESULTS_DIR}...")
    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Number of classes: {train_df[LABEL_COL].nunique()}")
    
    return train_df, test_df


def tokenize_text(text: str) -> list[str]:
    """Simple tokenization by splitting on whitespace."""
    return str(text).split()


# =========================
# Embedding Methods
# =========================

def create_tfidf_features(
    train_texts: pd.Series,
    test_texts: pd.Series,
    max_features: int = 10000,
) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """Create TF-IDF features."""
    print("\n" + "="*60)
    print("Creating TF-IDF features...")
    print("="*60)
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # unigrams and bigrams
        min_df=2,
        sublinear_tf=True,
    )
    
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    return X_train, X_test, vectorizer


def train_word2vec_model(
    texts: pd.Series,
    sg: int,
    name: str,
) -> Word2Vec:
    """
    Train a Word2Vec model.
    
    Args:
        texts: Training texts
        sg: 1 for Skip-gram, 0 for CBOW
        name: Model name for logging
    """
    print(f"\nTraining {name} model...")
    
    sentences = [tokenize_text(text) for text in texts]
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=EMBEDDING_DIM,
        window=WINDOW_SIZE,
        min_count=MIN_WORD_COUNT,
        sg=sg,
        workers=WORKERS,
        seed=RANDOM_SEED,
        epochs=10,
    )
    
    print(f"Vocabulary size: {len(model.wv)}")
    
    return model


def text_to_word2vec_vector(text: str, model: Word2Vec) -> np.ndarray:
    """Convert text to averaged Word2Vec vector."""
    tokens = tokenize_text(text)
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(EMBEDDING_DIM)


def create_word2vec_features(
    train_texts: pd.Series,
    test_texts: pd.Series,
    sg: int,
    name: str,
) -> Tuple[np.ndarray, np.ndarray, Word2Vec]:
    """Create Word2Vec features (Skip-gram or CBOW)."""
    print("\n" + "="*60)
    print(f"Creating {name} features...")
    print("="*60)
    
    # Train Word2Vec on training data
    w2v_model = train_word2vec_model(train_texts, sg, name)
    
    # Convert texts to vectors
    print(f"Converting texts to {name} vectors...")
    X_train = np.array([text_to_word2vec_vector(text, w2v_model) for text in train_texts])
    X_test = np.array([text_to_word2vec_vector(text, w2v_model) for text in test_texts])
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    return X_train, X_test, w2v_model


def train_fasttext_model(texts: pd.Series) -> FastText:
    """Train a FastText model."""
    print("\nTraining FastText model...")
    
    sentences = [tokenize_text(text) for text in texts]
    
    model = FastText(
        sentences=sentences,
        vector_size=EMBEDDING_DIM,
        window=WINDOW_SIZE,
        min_count=MIN_WORD_COUNT,
        workers=WORKERS,
        seed=RANDOM_SEED,
        epochs=10,
    )
    
    print(f"Vocabulary size: {len(model.wv)}")
    
    return model


def text_to_fasttext_vector(text: str, model: FastText) -> np.ndarray:
    """Convert text to averaged FastText vector."""
    tokens = tokenize_text(text)
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(EMBEDDING_DIM)


def create_fasttext_features(
    train_texts: pd.Series,
    test_texts: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, FastText]:
    """Create FastText features."""
    print("\n" + "="*60)
    print("Creating FastText features...")
    print("="*60)
    
    # Train FastText on training data
    ft_model = train_fasttext_model(train_texts)
    
    # Convert texts to vectors
    print("Converting texts to FastText vectors...")
    X_train = np.array([text_to_fasttext_vector(text, ft_model) for text in train_texts])
    X_test = np.array([text_to_fasttext_vector(text, ft_model) for text in test_texts])
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    return X_train, X_test, ft_model


# =========================
# Model Training & Evaluation
# =========================

def train_with_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, Any],
) -> LogisticRegression:
    """Train Logistic Regression with GridSearchCV."""
    print("\nPerforming hyperparameter tuning with GridSearchCV...")
    print(f"Parameter grid: {param_grid}")
    
    lr = LogisticRegression(random_state=RANDOM_SEED)
    
    grid_search = GridSearchCV(
        lr,
        param_grid,
        cv=CV_FOLDS,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1-macro: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def evaluate_model(
    model: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate model on test set."""
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    
    print(f"\nTest Results:")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    
    results = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_metrics": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
        },
    }
    
    return results


def save_results(
    embedding_name: str,
    model: LogisticRegression,
    results: Dict[str, Any],
    embedding_object: Any = None,
) -> None:
    """Save model and results."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    exp_name = f"logreg_{embedding_name}_{timestamp}"
    
    # Save model
    model_path = os.path.join(MODELS_DIR, f"{exp_name}_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved model to: {model_path}")
    
    # Save embedding object (vectorizer or word embedding model)
    if embedding_object is not None:
        emb_path = os.path.join(MODELS_DIR, f"{exp_name}_embedding.pkl")
        with open(emb_path, "wb") as f:
            pickle.dump(embedding_object, f)
        print(f"Saved embedding to: {emb_path}")
    
    # Save results
    results["embedding_type"] = embedding_name
    results["timestamp"] = timestamp
    results["model_params"] = model.get_params()
    
    results_path = os.path.join(RESULTS_DIR, "experiments", f"{exp_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {results_path}")


def run_experiment(
    embedding_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    embedding_object: Any = None,
) -> None:
    """Run complete experiment for one embedding type."""
    print("\n" + "#"*60)
    print(f"# EXPERIMENT: Logistic Regression + {embedding_name}")
    print("#"*60)
    
    # Train with hyperparameter tuning
    model = train_with_grid_search(X_train, y_train, PARAM_GRID)
    
    # Evaluate
    results = evaluate_model(model, X_test, y_test)
    
    # Save
    save_results(embedding_name, model, results, embedding_object)
    
    print(f"\n✓ Completed {embedding_name} experiment\n")


# =========================
# Main Execution
# =========================

def main() -> None:
    """Run all experiments."""
    ensure_dirs()
    
    # Load data
    train_df, test_df = load_data()
    
    X_train_text = train_df[TEXT_COL]
    X_test_text = test_df[TEXT_COL]
    y_train = train_df[LABEL_COL].values
    y_test = test_df[LABEL_COL].values
    
    # Experiment 1: TF-IDF
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = create_tfidf_features(
        X_train_text, X_test_text
    )
    run_experiment(
        "tfidf",
        X_train_tfidf,
        X_test_tfidf,
        y_train,
        y_test,
        tfidf_vectorizer,
    )
    
    # Experiment 2: Word2Vec Skip-gram
    X_train_skipgram, X_test_skipgram, skipgram_model = create_word2vec_features(
        X_train_text, X_test_text, sg=1, name="Word2Vec Skip-gram"
    )
    run_experiment(
        "word2vec_skipgram",
        X_train_skipgram,
        X_test_skipgram,
        y_train,
        y_test,
        skipgram_model,
    )
    
    # Experiment 3: Word2Vec CBOW
    X_train_cbow, X_test_cbow, cbow_model = create_word2vec_features(
        X_train_text, X_test_text, sg=0, name="Word2Vec CBOW"
    )
    run_experiment(
        "word2vec_cbow",
        X_train_cbow,
        X_test_cbow,
        y_train,
        y_test,
        cbow_model,
    )
    
    # Experiment 4: FastText (optional but recommended)
    print("\nDo you want to run FastText experiment? (This may take longer)")
    print("You can comment this section out if needed.")
    
    X_train_fasttext, X_test_fasttext, fasttext_model = create_fasttext_features(
        X_train_text, X_test_text
    )
    run_experiment(
        "fasttext",
        X_train_fasttext,
        X_test_fasttext,
        y_train,
        y_test,
        fasttext_model,
    )
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*60)
    print(f"\nResults saved in: {os.path.join(RESULTS_DIR, 'experiments')}")
    print(f"Models saved in: {MODELS_DIR}")


if __name__ == "__main__":
    main()
