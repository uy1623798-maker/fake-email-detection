# Fake Email Detection System

A Streamlit application that classifies email text as likely legitimate or suspicious using supervised machine learning.

## Models

- Logistic Regression
- Multinomial, Complement, and Bernoulli Naive Bayes
- Linear SVM and RBF SVM
- SGD, Passive Aggressive, Perceptron, and Ridge classifiers
- Decision Tree, Random Forest, Extra Trees, and K-Nearest Neighbors

The training process uses a stratified 25% held-out test split, reports each model's accuracy, and stores all fitted models plus the best model in `models/fake_email_detector.joblib`. These are common supervised classifiers suitable for comparison on text data; there is no finite list of every supervised algorithm. Small datasets produce unstable accuracy estimates.

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Add your dataset

The project includes `data/emails.csv` with `text,label` columns (`ham` for legitimate and `spam` for suspicious). The included starter dataset is small and intended to demonstrate the workflow. Replace it with a larger labelled dataset for meaningful results. Supported labels include `ham`/`legitimate`/`0` and `spam`/`phishing`/`1`.

## Notebook

Open `notebooks/fake_email_detection.ipynb` in Jupyter and run every cell to see dataset loading, TF-IDF feature extraction, separate model accuracies, and comparison.
