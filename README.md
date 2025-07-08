# SMS Spam Detection 📱✉️

A machine learning project to classify SMS messages as spam or ham using Python, scikit-learn, and Streamlit.

## Features
- Preprocessing with NLTK
- TF-IDF vectorization
- Naive Bayes model
- Streamlit UI

## Folder Structure
- `src/` - backend logic
- `streamlit_app/` - frontend UI
- `data/` - raw and processed datasets
- `models/` - saved model files

## 🚀 Live Demo

You can try the deployed SMS Spam Detection app here:  
🔗 [Click to Open](https://spam-in-detection.streamlit.app/)

## To Run:
```bash
uv venv .venv && uv pip install -r requirements.txt
streamlit run streamlit_app/app.py
