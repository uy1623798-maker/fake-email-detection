from __future__ import annotations

import joblib
import streamlit as st

from train_model import MODEL_FILE, train

st.set_page_config(page_title="EmailShield | Fake Email Detector", page_icon="✉️", layout="wide")
st.markdown("""
<style>
  .stApp { background: linear-gradient(145deg, #f5f8ff 0%, #f8fafc 55%, #eef5ff 100%); }
  h1 { color: #12315b; letter-spacing: -0.04em; }
  [data-testid="stMetric"] { background: white; border: 1px solid #e1e9f5; border-radius: 14px; padding: 14px 18px; box-shadow: 0 4px 18px #19375b0a; }
  .stButton button[kind="primary"] { border-radius: 10px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifact():
    if not MODEL_FILE.exists():
        return train()
    return joblib.load(MODEL_FILE)

artifact = load_artifact()

st.title("✉️ EmailShield")
st.caption("Email ko check karein aur supervised ML models ki test accuracy compare karein.")

with st.sidebar:
    st.header("Dataset & model")
    st.write(f"**Best test model:** {artifact['model_name']}")
    st.write(f"**Labelled emails:** {artifact['training_rows']}")
    st.caption(f"Dataset: {artifact['source']}")
    if st.button("Retrain models"):
        artifact = train()
        st.cache_resource.clear()
        st.success("Models retrained successfully.")
    st.caption("Starter dataset is small; use a larger labelled dataset for realistic accuracy.")

metric_cols = st.columns(3)
for col, (name, score) in zip(metric_cols, artifact["accuracies"].items()):
    col.metric(name, f"{score:.2f}%", "held-out test accuracy")

left, right = st.columns([1.25, 1])
with left:
    st.subheader("Email checker")
    with st.form("email_check_form"):
        email_text = st.text_area(
            "Email content", height=230,
            placeholder="Paste the subject and message body here...",
        )
        model_choice = st.selectbox("Choose a model", list(artifact["accuracies"].keys()),
                                    index=list(artifact["accuracies"].keys()).index(artifact["model_name"]))
        check = st.form_submit_button("Check email", type="primary", use_container_width=True)

with right:
    st.subheader("Model accuracy comparison")
    accuracy_frame = __import__("pandas").DataFrame(
        {"Algorithm": list(artifact["accuracies"].keys()), "Accuracy (%)": list(artifact["accuracies"].values())}
    ).set_index("Algorithm")
    st.bar_chart(accuracy_frame, horizontal=True, color="#3978c5")
    st.caption("Accuracy is measured on the held-out 25% test split from the CSV dataset.")

if check:
    if not email_text.strip():
        st.info("Pehle email content paste karein.")
    else:
        prediction = int(artifact["models"][model_choice].predict([email_text])[0])
        if prediction:
            st.error(f"⚠️ {model_choice}: Likely suspicious / spam email")
            st.write("Sender ko independently verify karein; links ya attachments kholne se pehle saavdhani rakhein.")
        else:
            st.success(f"✅ {model_choice}: Likely legitimate email")
            st.write("Yeh model ka estimate hai; email safe hone ki guarantee nahi.")

with st.expander("How this system works"):
    st.markdown("""
1. The labelled `data/emails.csv` dataset is split into training and test portions.
2. Email text is converted into TF-IDF features.
3. Logistic Regression, Multinomial Naive Bayes, and Linear SVM are trained and compared.
4. Select any model above to classify the pasted email.

The included starter dataset is small, so its scores demonstrate the workflow rather than real-world performance.
""")
