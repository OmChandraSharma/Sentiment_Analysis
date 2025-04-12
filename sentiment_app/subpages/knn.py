import streamlit as st
import pandas as pd

# Utility to display graph with inference
def display_graph_with_inference(title, graph_path, inference_text):
    st.markdown(f"### {title}")
    st.image(graph_path, width=750)  # Approx. 75% screen width
    with st.expander("📦 Inference"):
        st.markdown(inference_text)

def render():
    st.title("Sentiment Classification using KNN")
    st.markdown("""
    This report presents the evaluation of the **K-Nearest Neighbors (KNN)** classifier for sentiment classification.
    We explore different vectorization strategies and distance metrics, along with their effect on model accuracy.
    """)

    # ============================ Final Classification Overview ============================ #
    st.markdown("## 🔍 KNN Performance Summary")

    summary_data = {
        "Configuration": [
            "TF-IDF + Euclidean", "TF-IDF + Manhattan",
            "BoW + Euclidean", "BoW + Manhattan"
        ],
        "Accuracy": ["78.99%", "77.89%", "79.54%", "79.68%"],
        "Best k": [30, 28, 30, 30]
    }

    df_summary = pd.DataFrame(summary_data)
    st.table(df_summary.set_index("Configuration"))

    # ============================ Elbow Curve Inference ============================ #
    st.markdown("---")
    st.markdown("## 📈 Optimal K Selection via Elbow Curves")

    elbow_inference = """
    The Elbow Method was used to find the optimal number of neighbors (**k**) for each configuration. The model accuracy improves up to **k ≈ 26**, after which gains are negligible or decline.

    The method involved:
    - Using 70% of data for training and 30% for testing.
    - Drawing a 10% stratified sample from training data for faster tuning.
    - Varying k from 1 to 30 and plotting validation accuracy.

    Elbow curves suggest optimal `k ≈ 28–30` across all vectorizer-distance combinations.
    """

    display_graph_with_inference("Elbow Curve: TF-IDF + Euclidean", "../KNN/graphs/elbow_tfidf_euclidean.png", elbow_inference)
    display_graph_with_inference("Elbow Curve: TF-IDF + Manhattan", "../KNN/graphs/elbow_tfidf_manhattan.png", elbow_inference)
    display_graph_with_inference("Elbow Curve: BoW + Euclidean", "../KNN/graphs/elbow_bow_euclidean.png", elbow_inference)
    display_graph_with_inference("Elbow Curve: BoW + Manhattan", "../KNN/graphs/elbow_bow_manhattan.png", elbow_inference)

    # ============================ Confusion Matrix Observations ============================ #
    st.markdown("---")
    st.markdown("## 🧩 Confusion Matrix Analysis")

    cm_inference = """
    Despite different configurations, the model shows **consistent accuracy (~78%–80%)** across all setups. However:

    - The **Neutral** class receives **zero predictions** due to extreme imbalance in the dataset:
      - Positive: ~77%
      - Negative: ~23%
      - Neutral: ~0.098%

    - KNN, being sensitive to class distribution, shows **majority class bias**, failing to generalize for the Neutral class.
    - Most misclassifications occur between **Positive and Negative**, due to:
      - Similar wording in tweets.
      - 2-gram features not capturing context sufficiently.

    This suggests that **KNN is not ideal** for highly imbalanced sentiment datasets.
    """

    display_graph_with_inference("Confusion Matrix: TF-IDF + Euclidean", "../KNN/graphs/confmat_tfidf_euclidean.png", cm_inference)
    display_graph_with_inference("Confusion Matrix: TF-IDF + Manhattan", "../KNN/graphs/confmat_tfidf_manhattan.png", cm_inference)
    display_graph_with_inference("Confusion Matrix: BoW + Euclidean", "../KNN/graphs/confmat_bow_euclidean.png", cm_inference)
    display_graph_with_inference("Confusion Matrix: BoW + Manhattan", "../KNN/graphs/confmat_bow_manhattan.png", cm_inference)

    # ============================ Final Observations ============================ #
    st.markdown("---")
    st.markdown("## Final Observations & Model Recommendation")

    st.markdown("""
    - **Best Accuracy:** BoW + Manhattan (79.68%) and BoW + Euclidean (79.54%)
    - TF-IDF performance is slightly lower but consistent.
    - **Optimal k:** ~30 across all combinations.
    - **Severe class imbalance** leads to 0% recall for Neutral class.
    - KNN is not robust to rare classes and performs poorly in imbalanced settings.

    **📌 Recommendation:**
    - KNN may be used as a benchmark but is **not ideal for production**.
    - Consider advanced classifiers with built-in class balancing (e.g., Random Forest, ANN with class weights).
    - Explore **SMOTE or oversampling** for minority class (Neutral).
    """)

    st.markdown("## 📝 Notes")
    st.info("""
    - All configurations failed to predict the Neutral class due to data imbalance.
    - KNN’s non-parametric nature limits its ability to handle such skewed class distributions.
    """)
