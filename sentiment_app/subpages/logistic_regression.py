import streamlit as st
import pandas as pd

# Utility to display a graph with expander text
def display_graph_with_inference(title, graph_path, inference_text):
    st.markdown(f"### {title}")
    st.image(graph_path, width=750)
    with st.expander("Inference"):
        st.markdown(inference_text)

def render():
    st.title("Sentiment Classification using Logistic Regression")
    st.markdown("""
    This report evaluates the performance of Logistic Regression for sentiment classification.
    The study compares **Bag of Words (BoW)** and **TF-IDF** vectorizers across multiple configurations,
    focusing on accuracy, class balance, and generalization.
    """)

    # ============================ Overall Performance ============================ #
    st.markdown("## Overall Performance Comparison")
    display_graph_with_inference(
    "Accuracy Comparison: BoW vs TF-IDF",
    "https://storage.googleapis.com/sentimentann/graphs/feature_comparison.png",
    """
    BoW shows higher accuracy in nearly all experimental setups. This makes it the more reliable choice 
    for this specific sentiment classification task, especially when performance consistency is critical.
    """
    )


    # st.markdown("""
    # BoW consistently outperformed TF-IDF in most conditions. BoW's average accuracy was approximately **0.59**, 
    # compared to **0.54** for TF-IDF.
    # """)

    # ============================ Class Weighting ============================ #
    st.markdown("---")
    st.markdown("## Effect of Class Weighting")
    display_graph_with_inference(
    "Impact of Class Weighting on Accuracy",
    "https://storage.googleapis.com/sentimentann/graphs/class_weight.png",
    """
    Balanced class weights marginally improve performance, especially in minority classes.
    However, the gain is not substantial, suggesting that native class distribution is manageable.
    """
    )

    # class_weight_text = """
    # Class weighting had a **minimal positive impact** on both BoW and TF-IDF.

    # - **BoW**: Improved slightly from 0.59 → 0.60 with balanced weights.
    # - **TF-IDF**: Improved from 0.52 → 0.54.

    # This suggests the native class distribution may already be suitable for the classification task.
    # """
    # st.markdown(class_weight_text)

    # ============================ Regularization Strength ============================ #
    st.markdown("---")
    st.markdown("## Effect of Regularization Strength (C)")

    display_graph_with_inference(
    "Effect of Regularization Strength (C) on Accuracy",
    "https://storage.googleapis.com/sentimentann/graphs/regularization_strength.png",
    """
    Accuracy improves significantly with increasing C, peaking around C=10. Beyond that, 
    TF-IDF performance drops slightly, while BoW remains stable.
    """
    )


    # regularization_text = """
    # - Performance improved significantly as **C increased from 0.001 to 10.0**.
    # - **Best performance at C=10.0** for BoW (**Accuracy: 0.7212**).
    # - TF-IDF peaked slightly lower at C=10.0 (**Accuracy: 0.7115**).
    # - Very high C values (100) led to **plateauing or slight performance drop**.
    # """
    # st.markdown(regularization_text)

    # ============================ Confusion Matrix Observations ============================ #
    st.markdown("---")
    st.markdown("## Confusion Matrix Analysis")
    display_graph_with_inference(
    "Confusion Matrix - Best BoW Model",
    "https://storage.googleapis.com/sentimentann/graphs/lr_bow.png",
    """
    The model is more confident on Class 2 (Positive), but struggles slightly with Class 1 (Neutral).
    Overall structure shows reliable class separability.
    """
    )

    display_graph_with_inference(
        "Confusion Matrix - Best TF-IDF Model",
        "https://storage.googleapis.com/sentimentann/graphs/lr_tfidf.png",
        """
        Similar trends as BoW, but Class 1 performance (Neutral) dips more sharply.
        Suggests BoW handles ambiguous cases better.
        """
    )

    st.markdown("""
    ### Best BoW Model (Accuracy: 0.7212)
    - Class 0: 25/36 correct (69.4%)
    - Class 1: 18/28 correct (64.3%)
    - Class 2: 32/40 correct (80%)

    ### Best TF-IDF Model (Accuracy: 0.7115)
    - Class 0: 28/36 correct (77.8%)
    - Class 1: 13/28 correct (46.4%)
    - Class 2: 33/40 correct (82.5%)
    """)

    # ============================ Performance Metrics ============================ #
    st.markdown("---")
    st.markdown("## Detailed Performance Metrics")

    tfidf_report = {
        "Class": ["Class 0", "Class 1", "Class 2"],
        "Precision": [0.71, 0.62, 0.64],
        "Recall": [0.81, 0.29, 0.75],
        "F1-Score": [0.75, 0.39, 0.69]
    }

    bow_report = {
        "Class": ["Class 0", "Class 1", "Class 2"],
        "Precision": [0.74, 0.66, 0.70],
        "Recall": [0.78, 0.64, 0.75],
        "F1-Score": [0.74, 0.67, 0.72]
    }

    st.markdown("### TF-IDF Metrics (Accuracy: 0.654)")
    st.table(pd.DataFrame(tfidf_report).set_index("Class"))

    st.markdown("### BoW Metrics (Accuracy: 0.701)")
    st.table(pd.DataFrame(bow_report).set_index("Class"))

    # ============================ Optimal Configs ============================ #
    st.markdown("---")
    st.markdown("## Optimal Model Configurations")

    st.markdown("""
    **Best BoW Model:**
    - Accuracy: 0.7212
    - C=10.0
    - Penalty: L1
    - Solver: liblinear
    - Class weight: None

    **Best TF-IDF Model:**
    - Accuracy: 0.7115
    - C=10.0
    - Penalty: L1
    - Solver: liblinear
    - Class weight: None
    """)

    # ============================ Final Conclusion ============================ #
    st.markdown("---")
    st.markdown("## Final Observations & Recommendation")

    st.markdown("""
    - **Bag of Words** is the recommended vectorizer for this classification task.
    - Logistic Regression performs best with **moderate regularization (C=10.0)** and **L1 penalty**.
    - **Class weighting** provides negligible benefits in this setting.
    - Future improvements may include:
        - Incorporating more data
        - Handling class imbalance differently
        - Exploring ensemble methods or non-linear models
    """)

    st.info("""
    Note: While overall accuracy is useful, it's critical to **analyze class-wise metrics** like F1-score and recall
    to understand model fairness and weaknesses.
    """)

# For running the app
if __name__ == "__main__":
    render()
