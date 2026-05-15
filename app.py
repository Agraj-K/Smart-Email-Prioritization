"""
Streamlit demo UI for the Email Priority Classification System.

Run: streamlit run app.py
"""

import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


st.set_page_config(
    page_title="Email Priority Classifier",
    page_icon="📧",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center; color: #888; margin-bottom: 2rem; font-size: 1.1rem;
    }
    .priority-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white; padding: 1rem 2rem; border-radius: 12px;
        font-size: 1.5rem; font-weight: 700; text-align: center;
    }
    .priority-medium {
        background: linear-gradient(135deg, #ffa502, #ff6348);
        color: white; padding: 1rem 2rem; border-radius: 12px;
        font-size: 1.5rem; font-weight: 700; text-align: center;
    }
    .priority-low {
        background: linear-gradient(135deg, #2ed573, #1abc9c);
        color: white; padding: 1rem 2rem; border-radius: 12px;
        font-size: 1.5rem; font-weight: 700; text-align: center;
    }
    .metric-card {
        background: #1e1e2e; border-radius: 12px; padding: 1.2rem;
        border: 1px solid #333; margin-bottom: 1rem;
    }
    .rag-card {
        background: #1a1a2e; border-left: 4px solid #667eea;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📧 Email Priority Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">DistilBERT + RAG + Flan-T5 — 3-Phase Architecture</div>', unsafe_allow_html=True)


# ── Load Model (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models (this takes a minute on first run)...")
def load_classifier():
    from inference.contextual_classifier import ContextualClassifier
    return ContextualClassifier()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("RAG Top-K", 1, 10, 3)
    st.divider()
    st.markdown("### Architecture")
    st.markdown("""
    1. **Fine-Tuned DistilBERT** — Supervised classifier
    2. **FAISS RAG Pipeline** — Historical context retrieval
    3. **Flan-T5 LLM** — Contextual reasoning
    """)
    st.divider()
    st.markdown("### Sample Emails")
    samples = {
        "🔴 High Priority": "URGENT: The project deadline is tomorrow and we need your approval on the budget ASAP. Please review and respond immediately.",
        "🟡 Medium Priority": "Hi team, can you please review the attached report and schedule a meeting for next week? Let me know your availability.",
        "🟢 Low Priority": "FYI: Here is the quarterly newsletter with updates from various departments. No action needed.",
    }
    for label, text in samples.items():
        if st.button(label, use_container_width=True):
            st.session_state["email_input"] = text


# ── Main Input ────────────────────────────────────────────────────────────────
email_text = st.text_area(
    "✉️ Paste your email here:",
    value=st.session_state.get("email_input", ""),
    height=150,
    placeholder="e.g., URGENT: Need budget approval by EOD...",
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    classify_btn = st.button("🚀 Classify Email", type="primary", use_container_width=True)


# ── Classification ────────────────────────────────────────────────────────────
if classify_btn and email_text.strip():
    classifier = load_classifier()

    with st.spinner("Classifying..."):
        result = classifier.classify(email_text, top_k=top_k)

    # Priority badge
    priority = result["priority"]
    css_class = f"priority-{priority.lower()}"
    st.markdown(f'<div class="{css_class}">🏷️ {priority.upper()} PRIORITY</div>', unsafe_allow_html=True)

    # Metrics row
    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{result['confidence']:.1%}")
    c2.metric("Model Prediction", result["model_prediction"])
    c3.metric("Model Confidence", f"{result['model_confidence']:.1%}")

    # Reasoning
    st.info(f"**LLM Reasoning:** {result['reasoning']}")

    # Probability distribution
    st.subheader("📊 Probability Distribution")
    probs = result["model_probabilities"]
    cols = st.columns(3)
    for i, (label, prob) in enumerate(probs.items()):
        cols[i].progress(prob, text=f"{label}: {prob:.1%}")

    # RAG Context
    st.subheader("🔍 Similar Historical Emails (RAG)")
    for i, ctx in enumerate(result["rag_context"], 1):
        with st.expander(f"{i}. [{ctx['priority_label']}] {ctx['subject'][:60]} — Similarity: {ctx['similarity_score']:.3f}"):
            st.write(f"**Sender:** {ctx.get('sender', 'N/A')}")
            st.write(f"**Snippet:** {ctx.get('body_snippet', 'N/A')}")

    # Raw LLM output (collapsible)
    with st.expander("🤖 Raw LLM Response"):
        st.code(result.get("llm_raw_response", "N/A"))

elif classify_btn:
    st.warning("Please enter an email to classify.")
