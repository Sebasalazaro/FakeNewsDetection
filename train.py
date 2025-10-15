import os, re
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

CSV = "fake_or_real_news.csv"  # columns: title, text, label
df = pd.read_csv(CSV, encoding="utf-8", engine="python")

# Basic sanity checks
required_cols = {"title", "text", "label"}
missing = required_cols - set(df.columns.str.lower())
if missing:
    raise ValueError(f"CSV is missing columns: {missing}. Expected {required_cols}")

# Normalize column names just in case
cols = {c: c.lower() for c in df.columns}
df.rename(columns=cols, inplace=True)

# Clean text
def clean(t): return re.sub(r"\s+", " ", str(t)).strip()

df = df.dropna(subset=["title", "text", "label"]).copy()
df["title"] = df["title"].map(clean)
df["text"]  = df["text"].map(clean)

# Combine title + text as the input feature
df["input_text"] = (df["title"] + " " + df["text"]).str.strip()

# Normalize labels to {0,1}
if df["label"].dtype == object:
    # Handle FAKE/REAL (case-insensitive); FAKE=1, REAL=0
    lab = df["label"].astype(str).str.strip().str.upper()
    mapping = {"FAKE": 1, "REAL": 0}
    if not set(lab.unique()).issubset(set(mapping.keys())):
        raise ValueError(f"Unexpected label values: {lab.unique()}. Expected some of {list(mapping.keys())}.")
    y = lab.map(mapping).astype(int)
else:
    # Assume already numeric 0/1
    y = df["label"].astype(int)

X = df["input_text"]

# Split (stratify to keep class balance)
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorize
vec = TfidfVectorizer(max_features=50000, ngram_range=(1,2), lowercase=True)
Xtr = vec.fit_transform(Xtr)
Xva = vec.transform(Xva)

# Train
clf = LogisticRegression(max_iter=300, n_jobs=None)  # n_jobs available in some sklearn versions; omit if error
clf.fit(Xtr, ytr)

# Eval
print("Val accuracy:", clf.score(Xva, yva))

# Persist
os.makedirs("artifacts", exist_ok=True)
joblib.dump(vec, "artifacts/vectorizer.joblib")
joblib.dump(clf, "artifacts/model.joblib")
print("Saved to artifacts/vectorizer.joblib and artifacts/model.joblib")

