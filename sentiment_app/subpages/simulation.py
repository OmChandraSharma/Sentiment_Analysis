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
API_URL = "https://inference-api-123471747335.asia-south1.run.app/predict"



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

    st.title("Live Sentiment Classification (via API)")
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
        individual_predictions = []

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

                    individual_predictions.append({
                        "model": model_name,
                        "sentiment": sentiment,
                        "confidence": confidence
                    })

                    weight = model_accuracies[model_name]
                    score = sentiment_map.get(sentiment, 0)
                    weighted_sum += score * weight
                    total_weight += weight
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
                bg_color = "rgba(0, 128, 0, 0.15)"
            elif final_score < 0:
                final_sentiment = "Negative"
                bg_color = "rgba(255, 0, 0, 0.15)"
            else:
                final_sentiment = "Neutral"
                bg_color = "rgba(128, 128, 128, 0.15)"

            st.markdown("---")
            st.markdown(f"<h2>Final Aggregated Prediction</h2>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='padding: 15px; background-color:{bg_color}; border-radius:10px; font-size:18px;'>"
                f"<strong style='color:white;'>Final Sentiment:</strong> {final_sentiment} (Weighted by model accuracy)"
                f"</div>",
                unsafe_allow_html=True
            )

        # Show each model's prediction
        st.markdown("---")
        st.markdown("## Individual Model Predictions")
        for pred in individual_predictions:
            model_name = pred["model"]
            if "error" in pred:
                st.error(f"{model_name.upper()} API error: {pred['error']}")
            else:
                sentiment = pred["sentiment"]
                confidence = pred["confidence"]

                if sentiment == "positive":
                    bg_color = "rgba(0, 128, 0, 0.15)"
                elif sentiment == "negative":
                    bg_color = "rgba(255, 0, 0, 0.15)"
                else:
                    bg_color = "rgba(128, 128, 128, 0.15)"

                st.markdown(f"#### {model_name.upper()} Prediction")
                st.markdown(
                    f"<div style='padding: 10px; background-color:{bg_color}; border-radius:10px;'>"
                    f"<strong style='color:white;'>Sentiment:</strong> {sentiment.capitalize()}<br>"
                    f"<strong style='color:white;'>Confidence:</strong> {confidence * 100:.2f}%"
                    f"</div>",
                    unsafe_allow_html=True
                )


