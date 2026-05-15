"""
Central configuration for the Email Priority Classification System.
All hyperparameters, model names, and paths are defined here.
"""

import os

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(DATA_DIR, "fine_tuned_model")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_index")
PROCESSED_CSV = os.path.join(DATA_DIR, "processed_emails.csv")
LABELED_CSV = os.path.join(DATA_DIR, "labeled_emails.csv")

# Ensure directories exist
for _dir in [DATA_DIR, MODEL_DIR, FAISS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
KAGGLE_DATASET = "wcukierski/enron-email-dataset"
SAMPLE_SIZE = 20000         # Number of emails to use (None = all)

# ──────────────────────────────────────────────────────────────────────────────
# Priority Labels
# ──────────────────────────────────────────────────────────────────────────────
PRIORITY_LABELS = ["Low", "Medium", "High"]
NUM_LABELS = len(PRIORITY_LABELS)
LABEL2ID = {label: i for i, label in enumerate(PRIORITY_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(PRIORITY_LABELS)}

# ──────────────────────────────────────────────────────────────────────────────
# Fine-Tuning (Phase 1)
# ──────────────────────────────────────────────────────────────────────────────
CLASSIFIER_MODEL_NAME = "distilbert-base-uncased"
MAX_SEQ_LENGTH = 256
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
TRAIN_SPLIT = 0.8           # 80% train, 20% validation
EARLY_STOPPING_PATIENCE = 2
RANDOM_SEED = 42

# ──────────────────────────────────────────────────────────────────────────────
# RAG Pipeline (Phase 2)
# ──────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384          # Dimension for all-MiniLM-L6-v2
CHUNK_MAX_WORDS = 200        # Max words per chunk for long emails
FAISS_INDEX_FILE = os.path.join(FAISS_DIR, "email_index.faiss")
FAISS_METADATA_FILE = os.path.join(FAISS_DIR, "metadata.pkl")
RAG_TOP_K = 3                # Number of similar emails to retrieve

# ──────────────────────────────────────────────────────────────────────────────
# Contextual Inference (Phase 3)
# ──────────────────────────────────────────────────────────────────────────────
LLM_MODEL_NAME = "google/flan-t5-base"
LLM_MAX_INPUT_LENGTH = 512
LLM_MAX_NEW_TOKENS = 128
