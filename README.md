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

The project uses `data/kaggle_emails.csv`, a balanced sample of 5,000 emails from CEAS_08 in the [Kaggle Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset). Exact text duplicates are removed before sampling. The held-out split measures performance on this corpus; related messages and source-specific patterns can still inflate scores, and performance on other emails may differ. Supported labels include `ham`/`legitimate`/`0` and `spam`/`phishing`/`1`.

## Notebook

Open `notebooks/fake_email_detection.ipynb` in Jupyter and run every cell to see dataset loading, TF-IDF feature extraction, separate model accuracies, and comparison.
