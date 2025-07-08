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

## To Run:
```bash
uv venv .venv && uv pip install -r requirements.txt
streamlit run streamlit_app/app.py
