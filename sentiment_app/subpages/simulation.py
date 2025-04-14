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

API_URL = "http://127.0.0.1:8000/predict"  # Local server

def render():
    # Accuracy values for each model
    model_accuracies = {
        'ann': 0.85,
        'decision_tree': 0.88,
        'k_means': 0.60,
        'knn': 0.82,
        'logistic_regression': 0.84,
        'linear_regression': 0.76,
        'naive_bayes': 0.80,
        'svm': 0.86
    }

    # Sentiment encoding
    sentiment_map = {
        'positive': 1,
        'neutral': 0,
        'negative': -1
    }

    weighted_sum = 0
    total_weight = 0
    final_votes = []
    st.title(" Live Sentiment Classification (via API)")
    st.markdown("### Enter your text and get predictions from your deployed models!")

    user_input = st.text_area("Your Text")

    if st.button("Predict Sentiment"):
        if not user_input.strip():
            st.warning("Please enter some text.")
            return

        cleaned = clean_text(user_input)
        st.write("**Cleaned Text:**", cleaned)

        weighted_sum = 0
        total_weight = 0
        final_votes = []
        individual_predictions = []  # Store each model's output for later display

        # First gather all predictions
        for model_name in model_accuracies.keys():
            try:
                response = requests.post(
                    API_URL,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({"text": user_input, "model": model_name})
                )
                if response.status_code == 200:
                    result = response.json()
                    sentiment = result['sentiment'].lower()
                    confidence = result['confidence']

                    # Store result for later display
                    individual_predictions.append({
                        "model": model_name,
                        "sentiment": sentiment,
                        "confidence": confidence
                    })

                    # Add to weighted calculation
                    weight = model_accuracies[model_name]
                    score = sentiment_map.get(sentiment, 0)
                    weighted_sum += score * weight
                    total_weight += weight
                    final_votes.append((model_name, sentiment, weight))

                else:

                    individual_predictions.append({
                        "model": model_name,
                        "error": response.text
                    })
            except Exception as e:
                individual_predictions.append({
                    "model": model_name,
                    "error": str(e)
                })


        # Final aggregated prediction
        if total_weight > 0:
            final_score = weighted_sum / total_weight
            if final_score > 0:
                final_sentiment = "Positive"
            elif final_score < 0:
                final_sentiment = "Negative"
            else:
                final_sentiment = "Neutral"

            st.markdown("---")

            st.markdown("## Final Aggregated Prediction")
            st.info(f"**Final Sentiment**: `{final_sentiment}` (Weighted by model accuracy)")

        # Now show each model's prediction
        st.markdown("---")
        st.markdown("## 📋 Individual Model Predictions")
        for pred in individual_predictions:
            model_name = pred["model"]
            if "error" in pred:
                st.error(f" {model_name.upper()} API error: {pred['error']}")
            else:
                sentiment = pred["sentiment"]
                confidence = pred["confidence"]
                st.markdown(f"#### {model_name.upper()} Prediction")
                st.success(
                    f"**Sentiment**: {sentiment}\n\n"
                    f"**Confidence**: {confidence * 100:.2f}%"
                )


            st.markdown("## Final Aggregated Prediction")
            st.info(f"**Final Sentiment**: `{final_sentiment}` (Weighted by model accuracy)")

    
    # st.title("🚀 Live Sentiment Classification (via API)")
    # st.markdown("### Enter your text and get predictions from your deployed models!")

    # user_input = st.text_area("📝 Your Text")

    # if st.button("🎯 Predict Sentiment"):
    #     if not user_input.strip():
    #         st.warning("Please enter some text.")
    #         return

    #     cleaned = clean_text(user_input)
    #     st.write("✅ **Cleaned Text:**", cleaned)
    
    #     for model_name in [
    #         "ann", "svm", "logistic_regression", "naive_bayes",
    #         "knn", "decision_tree", "k_means"
    #     ]:
    #         try:
    #             response = requests.post(
    #                 API_URL,
    #                 headers={"Content-Type": "application/json"},
    #                 data=json.dumps({"text": user_input, "model": model_name})
    #             )
    #             if response.status_code == 200:
    #                 result = response.json()
    #                 st.markdown(f"#### {model_name.upper()} Prediction")
    #                 st.success(
    #                     f"**Sentiment**: {result['sentiment']}\n\n"
    #                     f"**Confidence**: {result['confidence'] * 100:.2f}%"
    #                 )
    #             else:
    #                 st.error(f"❌ {model_name.upper()} API error: {response.text}")
    #         except Exception as e:
    #             st.error(f"❌ Failed to call {model_name.upper()} API: {e}")


