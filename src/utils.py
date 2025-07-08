"""Utility functions for the SMS spam detection project."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import logging

logger = logging.getLogger(__name__)


def setup_logging(log_level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('spam_detection.log'),
            logging.StreamHandler()
        ]
    )


def create_wordcloud(text_data, title="Word Cloud"):
    """Create word cloud from text data."""
    try:
        text = ' '.join(text_data)
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return plt.gcf()
    except Exception as e:
        logger.error(f"Error creating word cloud: {e}")
        return None


def plot_confusion_matrix(cm, classes=['Ham', 'Spam']):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt.gcf()


def calculate_message_stats(df):
    """Calculate message statistics."""
    stats = {
        'total_messages': len(df),
        'spam_count': df['label_encoded'].sum(),
        'ham_count': len(df) - df['label_encoded'].sum(),
        'spam_percentage': (df['label_encoded'].sum() / len(df)) * 100,
        'avg_message_length': df['message'].str.len().mean(),
        'avg_spam_length': df[df['label_encoded'] == 1]['message'].str.len().mean(),
        'avg_ham_length': df[df['label_encoded'] == 0]['message'].str.len().mean()
    }
    
    return stats