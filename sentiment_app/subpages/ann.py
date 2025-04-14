import streamlit as st
import pandas as pd

# Utility to display graph with inference
def display_graph_with_inference(title, graph_path, inference_text):
    st.markdown(f"### {title}")
    st.image(graph_path, width=750)  # Approx. 75% width depending on screen size
    with st.expander("Inference"):
        st.markdown(inference_text)

def render():
    st.title("Sentiment Classification using ANN")
    st.markdown("""
    This report presents the evaluation of a Deep Artificial Neural Network (ANN) model for sentiment classification.
    The model was trained on text data processed through TF-IDF vectorization and dimensionality reduction using Truncated SVD.
    """)

    # ============================ Final Classification Report ============================ #
    st.markdown("## 🔍 Final Classification Report (Deep ANN)")

    report_data = {
        "Sentiment": ["Negative", "Neutral", "Positive"],
        "Precision": [0.86, 0.00, 0.72],
        "Recall": [0.94, 0.00, 0.50],
        "F1-Score": [0.90, 0.00, 0.59],
        "Support": [159315, 28, 49498]
    }
    df = pd.DataFrame(report_data)
    st.metric("Accuracy", value="83.48%")
    st.table(df.set_index("Sentiment"))

    # ============================ TF-IDF vs BoW ============================ #
    st.markdown("---")
    st.markdown("## Vectorizer Comparison: TF-IDF vs BoW")

    v_inference = """
    TF-IDF and BoW showed similar performance, with BoW achieving marginally higher test accuracy. However, the validation 
    accuracy of TF-IDF was more aligned with training accuracy, indicating **better generalization**. BoW had a larger gap 
    between training and validation, suggesting **possible overfitting**. For this reason, TF-IDF was selected as the preferred 
    vectorization strategy for building a more robust ANN.
    """
    display_graph_with_inference("TF-IDF vs BoW Accuracy, Loss, Confusion Matrix Comparison", "../ANN/graphs/vectorizer_comparison.png", v_inference)

    # ============================ SVD Component Analysis ============================ #
    st.markdown("---")
    st.markdown("## SVD Component Selection")

    s1 = """
    Explained variance increases steadily with the number of SVD components and reaches **around 89% at 3000 components**. However, 
    the rate of gain slows significantly after ~2000 components. This trend reflects diminishing returns in adding more dimensions, 
    and suggests that **most of the meaningful variance is captured below 2400 components**.
    """

    s2 = """
    Accuracy improves alongside component count and **plateaus near 2400 components**. Beyond this point, only marginal improvements 
    (or fluctuations) occur, indicating a saturation point. Therefore, **~2400 components strike the best balance** between 
    dimensionality and classification performance.
    """

    display_graph_with_inference("Explained Variance vs. SVD Components", "../ANN/graphs/variance_extended.png", s1)
    display_graph_with_inference("Accuracy vs. SVD Components", "../ANN/graphs/accuracy_extended.png", s2)
    # ============================ ANN Architecture Comparison ============================ #
    st.markdown("---")
    st.markdown("## Shallow vs Deep ANN")

    s3 = """
    Deep ANN showed a steady rise in training accuracy and surpassed the shallow network slightly (85.0% vs 84.2%). 
    Both models **plateaued around ~83.3% validation accuracy**, but the Deep ANN showed **greater stability across epochs**, 
    indicating better generalization and lower risk of overfitting.
    """

    s4 = """
    The Deep ANN consistently outperformed the Shallow ANN on all key metrics — F1, precision, and recall. While the gains were 
    **modest**, they are significant in large-scale applications. This reinforces the value of **added depth and dropout layers** 
    in extracting richer patterns from the reduced feature space.
    """

    s5 = """
    The diagram below illustrates the architectural difference between Shallow and Deep ANN. The Deep ANN includes **multiple hidden layers**, 
    **higher neuron counts**, and **dropout layers** to reduce overfitting. This complexity allows for capturing more abstract representations 
    of the input features transformed by TF-IDF and SVD.
    """
    display_graph_with_inference("DNN Architectures: Shallow vs Deep", "../ANN/graphs/architecture.png", s5)
    display_graph_with_inference("Training vs. Validation Accuracy (Shallow vs Deep)", "../ANN/graphs/s_d.png", s3)
    # Display comparison metrics as a table instead of an image
    st.markdown("### Performance Metrics Comparison")

    comparison_data = {
        "Metric": [
            "Accuracy", "F1 Score", "Precision", "Recall",
            "F1 (Negative)", "F1 (Positive)", "F1 (Neutral)",
            "Macro Avg F1", "Weighted Avg F1"
        ],
        "Shallow ANN": [
            "83.26%", "81.93%", "82.20%", "83.26%", 
            "90%", "59%", "0%", "50%", "82%"
        ],
        "Deep ANN": [
            "83.48%", "82.35%", "82.46%", "83.48%", 
            "90% (approx)", "59% (approx)", "0%", "50%", "82%"
        ]
    }

    metrics_df = pd.DataFrame(comparison_data)
    st.table(metrics_df.set_index("Metric"))

    with st.expander("Inference"):
        st.markdown(s4)
    # ============================ Final Observations ============================ #
    st.markdown("---")
    st.markdown("## Final Observations & Model Recommendation")
    st.markdown("""
    - **Deep ANN** performs well overall, especially on the dominant *Negative* class (F1: 0.90).
    - The **Positive** class has moderate performance (F1: 0.59) and is acceptable given its sample size.
    - **Neutral** class performance is poor (F1: 0.00), due to **extreme class imbalance** (only 28 samples).
    - TF-IDF generalizes better than BoW despite similar test accuracy.
    - SVD with ~2400 components offers the best performance-to-dimension ratio.
    - Deep ANN outperforms the shallow version across all key metrics.

    **Recommendation:**
    - Use **TF-IDF + SVD + Deep ANN** as the base model.
    - Apply **class rebalancing strategies** (e.g., oversampling, class weights, SMOTE) to improve Neutral class results.
    - Tune dropout, learning rate, and consider using focal loss to improve class fairness and robustness.
    """)

    # ============================ Additional Notes ============================ #
    st.markdown("## Notes")
    st.info("""
    - The macro-average F1-score is low due to poor Neutral class performance.
    - Weighted averages remain high due to dominance of the Negative class.
    - Monitor class-wise metrics, not just overall accuracy, in future iterations.
    """)
