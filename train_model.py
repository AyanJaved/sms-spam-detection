"""
Training script for the SMS spam detection model.
Run this script to train and save the model.
"""

import sys
from pathlib import Path
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_preprocessing import SMSPreprocessor
from feature_engineering import FeatureEngineer
from model_training import SpamClassifier
from utils import setup_logging
import config

def main():
    """Main training function."""
    setup_logging()
    
    print("🚀 Starting SMS Spam Detection Model Training")
    print("=" * 50)
    
    # Create directories
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load and preprocess data
    print("📊 Loading and preprocessing data...")
    preprocessor = SMSPreprocessor()
    df = preprocessor.load_and_preprocess_data(config.SPAM_DATA_FILE)
    
    # Save processed data
    df.to_csv(config.PROCESSED_DATA_FILE, index=False)
    print(f"✅ Processed data saved to {config.PROCESSED_DATA_FILE}")
    
    # Step 2: Split data
    print("🔀 Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    # Step 3: Feature engineering
    print("🔧 Engineering features...")
    feature_engineer = FeatureEngineer(
        max_features=config.MAX_FEATURES,
        min_df=config.MIN_DF,
        max_df=config.MAX_DF
    )
    
    X_train_vectorized = feature_engineer.fit_transform(X_train)
    X_test_vectorized = feature_engineer.transform(X_test)
    
    # Save vectorizer
    joblib.dump(feature_engineer.vectorizer, config.VECTORIZER_FILE)
    print(f"✅ Vectorizer saved to {config.VECTORIZER_FILE}")
    
    # Step 4: Train model
    print("🎯 Training spam classifier...")
    classifier = SpamClassifier()
    classifier.train(X_train_vectorized, y_train)
    
    # Step 5: Evaluate model
    print("📈 Evaluating model performance...")
    results = classifier.evaluate(X_test_vectorized, y_test)
    
    print(f"\n🎉 Model Training Complete!")
    print(f"📊 Model Accuracy: {results['accuracy']:.4f}")
    print(f"📊 Precision (Spam): {results['classification_report']['1']['precision']:.4f}")
    print(f"📊 Recall (Spam): {results['classification_report']['1']['recall']:.4f}")
    print(f"📊 F1-Score (Spam): {results['classification_report']['1']['f1-score']:.4f}")
    
    # Step 6: Save model
    classifier.save_model(config.MODEL_FILE)
    print(f"✅ Model saved to {config.MODEL_FILE}")
    
    # Display top features
    print("\n🔍 Top 10 Features:")
    top_features = feature_engineer.get_top_features(10)
    for i, (feature, score) in enumerate(top_features, 1):
        print(f"{i:2d}. {feature}: {score:.4f}")
    
    print("\n🚀 Training pipeline completed successfully!")
    print("💡 You can now run the Streamlit app: streamlit run streamlit_app/app.py")


if __name__ == "__main__":
    main()