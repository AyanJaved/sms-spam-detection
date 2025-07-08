"""Feature engineering utilities for SMS spam detection."""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature extraction for SMS spam detection."""
    
    def __init__(self, max_features=5000, min_df=1, max_df=0.95):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
        self.is_fitted = False
    
    def fit_transform(self, texts):
        """Fit vectorizer and transform texts."""
        try:
            X = self.vectorizer.fit_transform(texts)
            self.is_fitted = True
            logger.info(f"Fitted TF-IDF vectorizer with {X.shape[1]} features")
            return X
        except Exception as e:
            logger.error(f"Error fitting vectorizer: {e}")
            raise
    
    def transform(self, texts):
        """Transform texts using fitted vectorizer."""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted yet")
        
        try:
            return self.vectorizer.transform(texts)
        except Exception as e:
            logger.error(f"Error transforming texts: {e}")
            raise
    
    def get_feature_names(self):
        """Get feature names from vectorizer."""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted yet")
        
        return self.vectorizer.get_feature_names_out()
    
    def get_top_features(self, n_features=20):
        """Get top features by IDF score."""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted yet")
        
        feature_names = self.get_feature_names()
        idf_scores = self.vectorizer.idf_
        
        # Create feature-score pairs
        feature_scores = list(zip(feature_names, idf_scores))
        
        # Sort by IDF score
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        return feature_scores[:n_features]