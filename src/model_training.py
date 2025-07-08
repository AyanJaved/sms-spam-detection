"""Model training utilities for SMS spam detection."""

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import joblib
import logging

logger = logging.getLogger(__name__)


class SpamClassifier:
    """Spam classification model using Multinomial Naive Bayes."""
    
    def __init__(self, alpha=1.0):
        self.model = MultinomialNB(alpha=alpha)
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train the spam classifier."""
        try:
            self.model.fit(X_train, y_train)
            self.is_trained = True
            logger.info("Model trained successfully")
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        
        return results
    
    def save_model(self, filepath):
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        try:
            joblib.dump(self.model, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath):
        """Load trained model from file."""
        try:
            self.model = joblib.load(filepath)
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise