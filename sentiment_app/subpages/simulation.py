import streamlit as st
import requests
import json
from utils.preprocess import clean_text
# from utils.load_models import load_decision_tree_models
# from utils.load_models import load_naive_bayes
# from utils.load_models import load_clusterring
# from utils.load_models import load_KNN
# from utils.load_models import load_logistic_regression
# from utils.load_models import load_ann
# from utils.load_models import load_svm

# ========== API Endpoint ==========
API_URL = "http://127.0.0.1:8000/predict"  # 🔁 Replace with your deployed URL when ready

# ========== Streamlit Page ==========
def render():
    st.title("🚀 Live Sentiment Classification (via API)")
    st.markdown("### Enter your text and get predictions from your deployed models!")

    user_input = st.text_area("📝 Your Text")

    if st.button("🎯 Predict Sentiment"):
        if not user_input.strip():
            st.warning("Please enter some text.")
            return

        cleaned = clean_text(user_input)
        st.write("✅ **Cleaned Text:**", cleaned)

        for model_name in ["ann"]:  # Add "dt", "nb" if they’re also deployed
            try:
                response = requests.post(
                    API_URL,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({"text": user_input, "model": model_name})
                )
                if response.status_code == 200:
                    result = response.json()
                    st.markdown(f"#### {model_name.upper()} Prediction")
                    st.success(f"**Sentiment**: {result['sentiment']}\n\n**Confidence**: {result['confidence'] * 100:.2f}%")
                else:
                    st.error(f"❌ {model_name.upper()} API error: {response.text}")
            except Exception as e:
                st.error(f"❌ Failed to call {model_name.upper()} API: {e}")