from pathlib import Path
import re
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "reviews.csv"
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.joblib"

NEGATORS = {"not", "no", "never", "n't", "none", "cannot", "can't", "won't", "don't", "didn't", "isn't", "aren't", "wasn't", "weren't", "couldn't", "shouldn't", "wouldn't", "hardly", "barely"}

def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("can't", "can not").replace("won't", "will not").replace("n't", " not")
    text = re.sub(r"[^a-z\s']", " ", text)
    tokens = text.split()
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in NEGATORS and i + 1 < len(tokens):
            out.append(tok + "_" + tokens[i + 1])
            i += 2
        else:
            out.append(tok)
            i += 1
    return " ".join(out)

df = pd.read_csv(DATA_PATH)
df["clean"] = df["review"].astype(str).map(normalize_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=False, ngram_range=(1, 2), max_features=9000)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print(confusion_matrix(y_test, pred, labels=["Negative", "Positive"]))
print(classification_report(y_test, pred))

dump(pipeline, MODEL_PATH)
print("Saved model to", MODEL_PATH)
