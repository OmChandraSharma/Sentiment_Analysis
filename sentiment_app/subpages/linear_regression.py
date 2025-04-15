import streamlit as st
import pandas as pd

# Utility to display graph with inference
def display_graph_with_inference(title, graph_path, inference_text):
    st.markdown(f"### {title}")
    st.image(graph_path, width=750)
    with st.expander("Inference"):
        st.markdown(inference_text)

def render():
    st.title("Sentiment Classification using Linear Regression")

    st.markdown("""
    This report evaluates the performance of a Linear Regression model applied to sentiment classification.
    Although linear regression is generally used for regression tasks, we simulate classification by rounding and clipping predictions 
    to the nearest valid sentiment class. The goal is to explore its effectiveness for text-based sentiment analysis.
    """)

    # ============================ Final Regression Metrics ============================ #
    st.markdown("## Final Evaluation Metrics")

    metrics = {
        "Mean Squared Error (MSE)": "0.3496",
        "R² Score": "0.5167",
        "Approximate Classification Accuracy": "68.42%"
    }

    for k, v in metrics.items():
        st.metric(label=k, value=v)

    # ============================ Visualizations ============================ #
    st.markdown("---")
    st.markdown("## Evaluation Visuals")

    # Actual vs Predicted
    scatter_inference = """
    The scatter plot shows how predicted sentiment values (after regression) align with the actual encoded sentiment labels.
    A tighter cluster along the diagonal would indicate better predictive performance. In this case, a moderate alignment is observed.
    """
    display_graph_with_inference(
        "Actual vs Predicted Sentiment",
        "https://storage.googleapis.com/sentimentann/plots/actual_vs_predicted.png",
        scatter_inference
    )

    # Residual Histogram
    residual_hist_inference = """
    This histogram visualizes the distribution of residuals (errors between predicted and actual sentiment).
    Most residuals are centered around zero, which indicates that while there are outliers, the model is making reasonable predictions.
    """
    display_graph_with_inference(
        "Residuals Histogram",
        "https://storage.googleapis.com/sentimentann/plots/residuals_histogram.png",
        residual_hist_inference
    )

    # Residuals over Index
    residual_line_inference = """
    The residuals plotted across sample indices help reveal any patterns or biases in prediction.
    The absence of systematic trends suggests the model is fairly unbiased, although variance exists.
    """
    display_graph_with_inference(
        "Residuals Across Samples",
        "https://storage.googleapis.com/sentimentann/plots/residuals_lineplot.png",
        residual_line_inference
    )

    # ============================ Final Observations ============================ #
    st.markdown("---")
    st.markdown("## Final Observations & Recommendations")

    st.markdown("""
    - Linear Regression is not a conventional model for classification tasks but can be adapted through rounding techniques.
    - The **MSE** and **R² score** show a moderate fit, indicating partial explanatory power over sentiment variation.
    - An approximate accuracy of **68.42%** was achieved — a fair baseline, but not optimal.
    - **Recommendation**: For production-level tasks, consider **classification models** like **Logistic Regression**, **SVM**, or **Random Forest**.
    - These models inherently support categorical targets and yield better interpretability, calibration, and performance.
    """)

    st.markdown("## Notes")
    st.info("""
    - The model may underperform in edge cases due to the continuous nature of predictions.
    - Future experiments could compare regression models against classification baselines to assess performance trade-offs.
    - Incorporating class balancing, dimensionality reduction, or ensemble methods may improve future results.
    """)

# Run the report if this is the main script
if __name__ == "__main__":
    render()
