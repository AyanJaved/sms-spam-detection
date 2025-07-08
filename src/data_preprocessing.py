"""Data preprocessing utilities for SMS spam detection."""

import pandas as pd
import numpy as np
import re
import nltk
import logging
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure required NLTK resources are available
for resource in ['punkt', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)


class SMSPreprocessor:
    """Handles SMS text preprocessing for spam detection."""

    def __init__(self, stop_words=None, stemmer=None):
        self.stemmer = stemmer if stemmer else PorterStemmer()
        self.stop_words = stop_words if stop_words else set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        """Clean and preprocess an SMS message string."""
        if pd.isna(text):
            return ""

        # Lowercase
        text = text.lower()

        # Remove non-alphabetic characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize, remove stopwords, stem
        words = text.split()
        cleaned_words = [
            self.stemmer.stem(word)
            for word in words
            if word not in self.stop_words and len(word) > 2
        ]

        return ' '.join(cleaned_words)

    def load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Load SMS spam dataset from file, clean text, and encode labels."""
        try:
            messages = []

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Split on first whitespace (tab or space)
                    parts = re.split(r'\s+', line.strip(), maxsplit=1)
                    if len(parts) == 2:
                        label, message = parts
                        messages.append((label, message))
                    else:
                        logger.warning(f"⚠️ Skipped malformed line: {line.strip()}")

            df = pd.DataFrame(messages, columns=['label', 'message'])
            df.drop_duplicates(inplace=True)

            df['cleaned_message'] = df['message'].apply(self.clean_text)
            df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})

            logger.info(f"✅ Loaded {len(df)} messages after preprocessing.")
            logger.info(f"📊 Spam: {df['label_encoded'].sum()}, Ham: {len(df) - df['label_encoded'].sum()}")

            return df

        except FileNotFoundError:
            logger.error(f"❌ File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"❌ Error in load_and_preprocess_data: {e}")
            raise

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """Split cleaned data into training and testing sets."""
        X = df['cleaned_message']
        y = df['label_encoded']
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
