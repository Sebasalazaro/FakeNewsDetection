from flask import Flask, request, jsonify
import joblib, re

# Load artifacts saved by train.py
V = joblib.load("/app/artifacts/vectorizer.joblib")
M = joblib.load("/app/artifacts/model.joblib")

app = Flask(__name__)

def clean(t: str) -> str:
    # Mirror train.py whitespace normalization
    return re.sub(r"\s+", " ", str(t)).strip()

def build_input(payload: dict) -> str:
    # Accept either {title, text} or just {text}
    title = clean(payload.get("title", ""))
    text  = clean(payload.get("text", ""))
    if title or text:
        return f"{title} {text}".strip()
    # Fallback if body is a raw string
    if isinstance(payload, str):
        return clean(payload)
    return ""

@app.post("/predict")
def predict():
    payload = request.get_json(force=True, silent=True) or {}
    inp = build_input(payload)
    if not inp:
        return jsonify({"error": "Provide 'text' or 'title'+'text'"}), 400
    X = V.transform([inp])
    # class 1 = FAKE (matches train.py mapping)
    prob_fake = float(M.predict_proba(X)[0][1])
    pred = int(prob_fake >= 0.5)
    return jsonify({"prob_fake": prob_fake, "pred_label": pred})

@app.get("/health")
def health():
    return jsonify({"ok": True})

if __name__ == "__main__":
    # Needed because your Dockerfile runs `python app.py`
    app.run(host="0.0.0.0", port=5000)

