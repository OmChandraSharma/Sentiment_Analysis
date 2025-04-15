import streamlit as st
import pandas as pd

# Utility to display graph with inference
def display_graph_with_inference(title, graph_path, inference_text):
    st.markdown(f"### {title}")
    st.image(graph_path, width=500)
    with st.expander(" Inference"):
        st.markdown(inference_text)

def render():
    st.title("Sentiment Classification using SVM")
    st.markdown("""
    This report analyzes Support Vector Machine (SVM) classifiers for sentiment classification using two feature extraction methods: 
    **Bag of Words (BoW)** and **Term Frequency-Inverse Document Frequency (TF-IDF)**.
    """)

    # ============================ Classification Reports ============================ #
    st.markdown("##  Classification Report: TF-IDF vs BoW with SVM")

    tfidf_data = {
        "Class": ["Class 0", "Class 1", "Class 2"],
        "Precision": [0.52, 0.53, 0.92],
        "Recall": [0.89, 0.57, 0.28],
        "F1-Score": [0.65, 0.55, 0.42]
    }

    bow_data = {
        "Class": ["Class 0", "Class 1", "Class 2"],
        "Precision": [0.80, 0.42, 0.55],
        "Recall": [0.44, 0.71, 0.65],
        "F1-Score": [0.57, 0.53, 0.59]
    }

    tfidf_df = pd.DataFrame(tfidf_data).set_index("Class")
    bow_df = pd.DataFrame(bow_data).set_index("Class")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### SVM with TF-IDF (Accuracy: 56.7%)")
        st.table(tfidf_df)

    with col2:
        st.markdown("### SVM with BoW (Accuracy: 59.6%)")
        st.table(bow_df)

    # ============================ Regularization Strength ============================ #
    st.markdown("---")
    st.markdown("## Effect of Regularization Strength (C Parameter)")

    c_inference = """
    - At **low C values** (e.g., 0.1), performance was poor (~38% accuracy) due to underfitting.
    - Sharp improvement occurred between C=0.1 and C=1.0, where accuracy rose to ~64%.
    - Beyond C=1.0, performance stabilized or declined slightly, suggesting **overfitting** at higher values.
    - **Optimal performance at C=1.0** indicates a good trade-off between margin width and misclassification.
    """

    display_graph_with_inference("C Parameter vs Accuracy", "https://storage.googleapis.com/sentimentann/graphs/regularization_strength.png", c_inference)

    # ============================ Confusion Matrices ============================ #
    st.markdown("---")
    st.markdown("## Confusion Matrix Analysis")

    cm_inference_bow= """
    **SVM with BoW:**
    - Balanced performance across classes.
    - Class 1 had best recall (71%), Class 0 had best precision (80%).
    - Notable misclassification from Class 0 to Class 1.
      """
    cm_inference_tfidf ="""
    **SVM with TF-IDF:**
    - High recall for Class 0 (89%) but very poor recall for Class 2 (28%).
    - Class 2 predictions were rare but precise (92%).
    - Model biased toward predicting Class 0.
    """

    display_graph_with_inference("Confusion Matrix: SVM with BoW", "https://storage.googleapis.com/sentimentann/graphs/svm_bow.png", cm_inference_bow)
    display_graph_with_inference("Confusion Matrix: SVM with TF-IDF", "https://storage.googleapis.com/sentimentann/graphs/svm_tfidf.png", cm_inference_tfidf)

    # ============================ Final Observations ============================ #
    st.markdown("---")
    st.markdown("## Final Observations & Recommendations")

    st.markdown("""
    - **BoW** slightly outperformed **TF-IDF** in overall accuracy (59.6% vs. 56.7%).
    - **TF-IDF** achieved extremely high recall for Class 0 and precision for Class 2.
    - BoW delivered **more balanced** performance across all classes.
    - The regularization parameter **C=1.0** yielded the best results for both methods.
    - TF-IDF suffers from **class imbalance bias**, particularly underperforming on Class 2.

    ### Recommendation:
    - Use **BoW** with C=1.0 for more balanced predictions.
    - If focusing on Class 2 reliability, consider **TF-IDF**.
    - Apply **class balancing techniques** (e.g., SMOTE, class weights).
    - Further hyperparameter tuning and model calibration is advised.
    """)

    # ============================ Notes ============================ #
    st.markdown("## Notes")
    st.info("""
    - Small dataset sizes per class may exaggerate precision/recall effects.
    - Monitor class-wise F1 scores, especially when classes are imbalanced.
    - Model bias can shift significantly depending on feature representation.
    """)

# To run: save this as `svm_report_app.py` and run `streamlit run svm_report_app.py`
