
from pathlib import Path
import json
import re
import pandas as pd
from flask import Flask, render_template, request
from joblib import load

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.joblib"
DATA_PATH = BASE_DIR / "data" / "reviews.csv"
METRICS_PATH = BASE_DIR / "metrics.json"

app = Flask(__name__)
model = load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}

NEGATORS = {
    "not", "no", "never", "n't", "none", "cannot", "can't", "won't",
    "don't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "couldn't", "shouldn't", "wouldn't", "hardly", "barely"
}

MIXED_PHRASES = [
    "not as bad", "not bad", "not too bad", "better than expected",
    "could be worse", "okay", "fine", "decent", "average"
]

SAMPLES = [
    "The movie was good and entertaining.",
    "The movie was not bad at all and I enjoyed it.",
    "The plot was boring and the acting was weak.",
    "I was surprised. It was better than I expected."
]

ABOUT_TEXT = (
    "This project classifies movie reviews as Positive, Negative, or Mixed / Neutral. "
    "It can be used for movie recommendations, audience feedback analysis, social media monitoring, "
    "and review summarization."
)


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("can't", "can not").replace("won't", "will not").replace("n't", " not")
    text = re.sub(r"[^a-z\s']", " ", text)
    tokens = text.split()
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in NEGATORS and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt in {"as", "too", "very", "much", "so"} and i + 2 < len(tokens):
                out.append(tok + "_" + tokens[i + 2])
                i += 3
            else:
                out.append(tok + "_" + nxt)
                i += 2
        else:
            out.append(tok)
            i += 1
    return " ".join(out)


def is_mixed_neutral(raw_text: str) -> bool:
    text = (raw_text or "").lower()
    return any(phrase in text for phrase in MIXED_PHRASES)


def predict_review(text: str):
    cleaned = normalize_text(text)
    probs = model.predict_proba([cleaned])[0]
    classes = list(model.classes_)
    pos_prob = float(probs[classes.index("Positive")]) * 100
    neg_prob = float(probs[classes.index("Negative")]) * 100
    difference = abs(pos_prob - neg_prob)

    if is_mixed_neutral(text) or difference < 20:
        prediction = "Mixed / Neutral"
    elif pos_prob > neg_prob:
        prediction = "Positive"
    else:
        prediction = "Negative"

    return {
        "prediction": prediction,
        "pos_prob": round(pos_prob, 1),
        "neg_prob": round(neg_prob, 1),
        "confidence": round(max(pos_prob, neg_prob), 1),
        "cleaned": cleaned,
    }


def classify_batch(raw_text: str):
    lines = [line.strip() for line in (raw_text or "").splitlines() if line.strip()]
    results = []
    positive = 0
    negative = 0
    neutral = 0
    for line in lines:
        pred = predict_review(line)
        label = pred["prediction"]
        if label == "Positive":
            positive += 1
        elif label == "Negative":
            negative += 1
        else:
            neutral += 1
        results.append({
            "review": line[:90],
            "prediction": label,
            "confidence": pred["confidence"],
            "pos_prob": pred["pos_prob"],
            "neg_prob": pred["neg_prob"],
        })
    return results, positive, negative, neutral


@app.route("/", methods=["GET", "POST"])
def index():
    review = ""
    cleaned = ""
    result = None
    batch_text = ""
    batch_results = []
    batch_summary = {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
    active_tab = request.form.get("active_tab", "classify") if request.method == "POST" else "classify"

    if request.method == "POST":
        active_tab = request.form.get("active_tab", "classify")
        if active_tab == "classify":
            review = request.form.get("review", "").strip()
            if review:
                result = predict_review(review)
                cleaned = result["cleaned"]
        elif active_tab == "batch":
            batch_text = request.form.get("batch_text", "").strip()
            if batch_text:
                batch_results, p, n, u = classify_batch(batch_text)
                batch_summary = {"positive": p, "negative": n, "neutral": u, "total": len(batch_results)}

    return render_template(
        "enhanced_index.html",
        review=review,
        cleaned=cleaned,
        result=result,
        batch_text=batch_text,
        batch_results=batch_results,
        batch_summary=batch_summary,
        metrics=metrics,
        samples=SAMPLES,
        about_text=ABOUT_TEXT,
        active_tab=active_tab,
    )


@app.route("/sample/<int:index>")
def sample(index):
    review = SAMPLES[index % len(SAMPLES)]
    result = predict_review(review)
    return render_template(
        "enhanced_index.html",
        review=review,
        cleaned=result["cleaned"],
        result=result,
        batch_text="",
        batch_results=[],
        batch_summary={"positive": 0, "negative": 0, "neutral": 0, "total": 0},
        metrics=metrics,
        samples=SAMPLES,
        about_text=ABOUT_TEXT,
        active_tab="classify",
    )


@app.route("/batch-example")
def batch_example():
    batch_text = """The movie was amazing and very enjoyable.
It was not bad at all, actually pretty decent.
The plot was boring and the acting was weak.
The ending was better than expected."""
    batch_results, p, n, u = classify_batch(batch_text)
    return render_template(
        "enhanced_index.html",
        review="",
        cleaned="",
        result=None,
        batt_text=batch_text,
        batch_results=batch_results,
        batch_summary={"positive": p, "negative": n, "neutral": u, "total": len(batch_results)},
        metrics=metrics,
        samples=SAMPLES,
        about_text=ABOUT_TEXT,
        active_tab="batch",
    )


if __name__ == "__main__":
    app.run(debug=True)
