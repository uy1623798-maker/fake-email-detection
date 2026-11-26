"""Train and save supervised models for fake-email detection.

Place a labelled CSV at data/emails.csv to train on a real dataset. It needs a
text/email/body column and a label/target/class column. Labels such as spam,
phishing, fake, 1 and true are treated as suspicious.
"""
from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import (LogisticRegression, PassiveAggressiveClassifier,
                                  Perceptron, RidgeClassifier, SGDClassifier)
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "emails.csv"
MODEL_FILE = ROOT / "models" / "fake_email_detector.joblib"


def demo_dataset() -> pd.DataFrame:
    """Small fallback dataset so the app can run before a real CSV is added."""
    suspicious = [
        "Urgent: verify your bank account now using this secure link",
        "You won a cash prize. Click immediately to claim your reward",
        "Your mailbox will be closed today. Confirm password here",
        "Payment failed. Update card details to avoid account suspension",
        "CEO needs gift cards now, reply with the codes urgently",
        "You have an unpaid tax refund. Submit your identity details",
        "Login from unknown device. Reset your password at this link",
        "Congratulations beneficiary, send processing fee for inheritance",
        "Invoice attached, enable macros to view the protected document",
        "Act now: limited time crypto investment doubles your money",
        "Your package is held. Pay delivery charge to release it",
        "Security alert: confirm your credentials within 24 hours",
    ]
    legitimate = [
        "Team meeting has been moved to 3 PM in conference room B",
        "Your order #4512 has shipped and will arrive on Friday",
        "Please find the project report attached for your review",
        "Reminder: your dentist appointment is scheduled for Tuesday",
        "The weekly engineering newsletter is now available",
        "Thanks for your application. We will contact you next week",
        "Your monthly account statement is ready in the mobile app",
        "Can we discuss the design feedback tomorrow morning?",
        "The library will be closed on the public holiday",
        "Here are the notes from today's client call",
        "Your subscription renewal was successful. No action is needed",
        "Please approve the leave request in the employee portal",
    ]
    # Repeat distinct templates with harmless variations, only for demo use.
    rows = []
    for text in suspicious:
        for suffix in ("", " Reference number 7281.", " Please respond today."):
            rows.append({"text": text + suffix, "label": 1})
    for text in legitimate:
        for suffix in ("", " Regards, Support team.", " Thank you."):
            rows.append({"text": text + suffix, "label": 0})
    return pd.DataFrame(rows)


def normalise_label(value: object) -> int:
    value = str(value).strip().lower()
    return int(value in {"1", "true", "spam", "phishing", "fake", "fraud", "suspicious"})


def load_data() -> tuple[pd.DataFrame, str]:
    if not DATA_FILE.exists():
        return demo_dataset(), "built-in demonstration dataset"
    df = pd.read_csv(DATA_FILE)
    columns = {col.lower(): col for col in df.columns}
    text_col = next((columns[k] for k in ("text", "email", "body", "message", "content") if k in columns), None)
    label_col = next((columns[k] for k in ("label", "target", "class", "category") if k in columns), None)
    if not text_col or not label_col:
        raise ValueError("data/emails.csv needs text/email/body and label/target/class columns.")
    cleaned = df[[text_col, label_col]].dropna().rename(columns={text_col: "text", label_col: "label"})
    cleaned["text"] = cleaned["text"].astype(str)
    cleaned["label"] = cleaned["label"].map(normalise_label)
    if cleaned["label"].nunique() < 2:
        raise ValueError("Your CSV must contain both legitimate and suspicious emails.")
    return cleaned, str(DATA_FILE)


def build_candidates() -> dict:
    """Common supervised classifiers that can train on sparse TF-IDF text."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Complement Naive Bayes": ComplementNB(),
        "Bernoulli Naive Bayes": BernoulliNB(),
        "Linear SVM": LinearSVC(class_weight="balanced"),
        "RBF SVM": SVC(kernel="rbf", class_weight="balanced"),
        "SGD Classifier": SGDClassifier(loss="hinge", max_iter=2000, tol=1e-3, class_weight="balanced", random_state=42),
        "Passive Aggressive": PassiveAggressiveClassifier(max_iter=2000, class_weight="balanced", random_state=42),
        "Perceptron": Perceptron(max_iter=2000, class_weight="balanced", random_state=42),
        "Ridge Classifier": RidgeClassifier(class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3, metric="cosine", algorithm="brute", n_jobs=-1),
    }


def train() -> dict:
    df, source = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )
    candidates = build_candidates()
    scores, trained = {}, {}
    for name, classifier in candidates.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)),
            ("classifier", classifier),
        ])
        pipeline.fit(X_train, y_train)
        scores[name] = round(accuracy_score(y_test, pipeline.predict(X_test)) * 100, 2)
        trained[name] = pipeline
    best_name = max(scores, key=scores.get)
    best_model = trained[best_name]
    MODEL_FILE.parent.mkdir(exist_ok=True)
    artifact = {"model": best_model, "model_name": best_name, "accuracies": scores,
                "models": trained,
                "source": source, "training_rows": len(df),
                "report": classification_report(y_test, best_model.predict(X_test), output_dict=True, zero_division=0)}
    joblib.dump(artifact, MODEL_FILE)
    return artifact


if __name__ == "__main__":
    result = train()
    print(f"Saved {result['model_name']} to {MODEL_FILE}")
    print("Accuracy results:", result["accuracies"])
