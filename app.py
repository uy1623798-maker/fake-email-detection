from __future__ import annotations

import joblib
import streamlit as st

from train_model import MODEL_FILE, train

st.set_page_config(page_title="EmailShield | Fake Email Detector", page_icon="✉️", layout="wide")
st.markdown("""
<style>
  :root { --ink: #142a43; --muted: #64748b; --blue: #2563eb; --line: #e4ebf4; }
  .stApp { background: radial-gradient(ellipse at 12% 0%, #e6efff 0, transparent 38%), #f6f8fc; color: var(--ink); }
  [data-testid="stHeader"] { background: transparent; }
  [data-testid="stMainBlockContainer"] { padding-top: 2.2rem; max-width: 1320px; }
  h1 { color: var(--ink); letter-spacing: -.045em; font-weight: 800; }
  h2, h3 { color: #1e3552; letter-spacing: -.025em; }
  [data-testid="stCaptionContainer"] { color: var(--muted); }
  [data-testid="stMetric"] { background: rgba(255,255,255,.92); border: 1px solid var(--line); border-radius: 18px; padding: 17px 20px; box-shadow: 0 8px 26px #18365a0b; }
  [data-testid="stMetricLabel"] { color: var(--muted); font-weight: 650; }
  [data-testid="stMetricValue"] { color: var(--blue); font-weight: 800; }
  [data-testid="stForm"] { background: #fff; border: 1px solid var(--line); border-radius: 20px; padding: 20px; box-shadow: 0 12px 34px #18365a0c; }
  [data-testid="stTextArea"] textarea { border-radius: 12px; border-color: #d8e1ed; background: #fbfdff; }
  [data-testid="stTextArea"] textarea:focus { border-color: var(--blue); box-shadow: 0 0 0 2px #2563eb20; }
  .stButton button[kind="primary"] { border-radius: 12px; min-height: 2.8rem; font-weight: 750; box-shadow: 0 5px 14px #2563eb28; }
  [data-testid="stSidebar"] { background: #101d31; }
  [data-testid="stSidebar"] * { color: #e7eef9; }
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #a9b8ce; }
  [data-testid="stAlert"] { border-radius: 14px; }
  [data-testid="stExpander"] { border-color: var(--line); border-radius: 14px; background: #ffffffb8; }
  @media (max-width: 720px) { [data-testid="stMainBlockContainer"] { padding: 1.2rem 1rem 2rem; } [data-testid="stForm"] { padding: 14px; } }
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

st.subheader("Supervised model accuracy")
st.caption(f"{len(artifact['accuracies'])} algorithms trained on the same held-out 25% test split.")
accuracy_items = list(artifact["accuracies"].items())
for row_start in range(0, len(accuracy_items), 4):
    metric_cols = st.columns(4)
    for col, (name, score) in zip(metric_cols, accuracy_items[row_start:row_start + 4]):
        col.metric(name, f"{score:.2f}%", "test accuracy")

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

    # Keep the verdict directly below the checker so users do not miss it
    # beneath the longer model-comparison section.
    if check:
        st.markdown("---")
        st.subheader("Email ka result")
        if not email_text.strip():
            st.info("Pehle upar email ka subject aur message paste karein, phir **Check email** dabayein.")
        else:
            prediction = int(artifact["models"][model_choice].predict([email_text])[0])
            if prediction:
                st.error("🚨 **Yeh email suspicious / fake lag rahi hai.**")
                st.write(f"Selected model: **{model_choice}**")
                st.warning("Link ya attachment na kholein. Sender ko official website ya known phone number se verify karein.")
            else:
                st.success("✅ **Yeh email likely genuine lag rahi hai.**")
                st.write(f"Selected model: **{model_choice}**")
                st.warning("Yeh sirf ML model ka estimate hai, safety guarantee nahi. Sender address aur links phir bhi check karein.")

with right:
    st.subheader("All model results")
    accuracy_frame = __import__("pandas").DataFrame(
        {"Algorithm": list(artifact["accuracies"].keys()), "Accuracy (%)": list(artifact["accuracies"].values())}
    ).sort_values("Accuracy (%)", ascending=False).set_index("Algorithm")
    st.bar_chart(accuracy_frame, horizontal=True, color="#3978c5", height=480)
    st.dataframe(accuracy_frame.style.format("{:.2f}%"), use_container_width=True)
    st.caption("Accuracy is measured on the held-out 25% test split from the CSV dataset.")

with st.expander("How this system works"):
    st.markdown("""
1. The labelled `data/emails.csv` dataset is split into training and test portions.
2. Email text is converted into TF-IDF features.
3. Fourteen commonly used supervised text classifiers are trained and compared on the same split.
4. Select any model above to classify the pasted email.

The included starter dataset is small, so its scores demonstrate the workflow rather than real-world performance.
""")
