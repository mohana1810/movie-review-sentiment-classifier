# Movie Review Sentiment Classifier

A simple movie review sentiment classifier with a basic UI.  
Built with Flask, TF-IDF, and Logistic Regression.

## Why this version
- No Streamlit
- Faster startup
- Basic UI
- Shows positive and negative probabilities
- Handles phrases like "not bad" better than plain Naive Bayes

## Tech stack
- Python 3.10+
- Flask
- pandas
- scikit-learn
- joblib
- HTML/CSS

## Run on your laptop
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py
python app.py
```

Open:
```bash
http://127.0.0.1:5000
```
