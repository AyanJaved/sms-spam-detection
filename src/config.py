"""Configuration settings for the SMS Spam Detection project."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data files
SPAM_DATA_FILE = RAW_DATA_DIR / "SMSSpamCollection"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_data.csv"

# Model files
MODEL_FILE = MODELS_DIR / "spam_classifier.pkl"
VECTORIZER_FILE = MODELS_DIR / "tfidf_vectorizer.pkl"

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 5000
MIN_DF = 1
MAX_DF = 0.95

# Streamlit configuration
APP_TITLE = "SMS Spam Detection System"
APP_ICON = "📱"