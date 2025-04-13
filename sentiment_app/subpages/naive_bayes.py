import streamlit as st
# from utils.load_models import load_decision_tree_models
# from utils.preprocess import clean_text
from utils.model_analysis import model_analysis_page
import pandas as pd


def display_model_metrics(title, accuracy, report_data, confusion_img_path):
    st.markdown(f"## 🔍 {title}")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        st.metric(label="🎯 Accuracy", value=f"{accuracy:.2%}")
    
    df_report = pd.DataFrame(report_data)
    st.markdown("### 📊 Classification Report")
    st.table(df_report.set_index("Sentiment"))

    st.markdown("### 🧩 Confusion Matrix")
    st.image(confusion_img_path, width=500)

# Utility: Display Graph and Inference Box
def display_graph_with_inference(title, graph_path, default_inference="Write your inference here..."):
    st.markdown(f"### {title}")
    st.image(graph_path, width=500)
    with st.expander("📦 Inference"):
        st.markdown(default_inference)

# Main Render Function
def render():
    # model, vectorizer, svd, label_encoder = load_decision_tree_models()
    st.title("📌 Naive Bayes Analysis Report")

    # Report Data (reused in all models for now)
    report_data1 = {
        "Sentiment": ["Negative", "Postive", "Neutral"],
        "Precision": [0.67, 0.84, 0.00],
        "Recall": [0.44, 0.93, 0.00],
        "F1-Score": [0.53, 0.88, 0.00]
    }
    report_data2 = {
        "Sentiment": ["Negative", "Postive", "Neutral"],
        "Precision": [0.81, 0.78, 0.00],
        "Recall": [0.97, 0.28, 0.00],
        "F1-Score": [0.89, 0.42, 0.00]
        
    }
    

    # ============================ Models & Reports ============================ #
    display_model_metrics("Naive Bayes from scratch with Tokenization", 0.8134, report_data1, "../naive_bayes/graphs/confusionmatrix_scratchmodel.png")
    display_model_metrics("Naive Bayes using TF-IDF", 0.8112, report_data2, "../naive_bayes/graphs/confusion-matrix-tfidf.png")
    
    st.markdown("## Accuracy comaparison using Test and Validation")
    st.image("../naive_bayes/graphs/accuracy_comparision.png", width=500)


    # ============================ Final Observations ============================ #
    st.markdown("---")
    st.markdown("## 🧠 Final Observations & Model Recommendation")
    st.markdown("""
    Bag-of-Words (BoW) features outperform TF-IDF in this sentiment classification task, 
    especially when paired with a Random Forest classifier. This is likely due to BoW's ability 
    to retain frequent sentiment-related words like *good, bad, love,* etc., which TF-IDF down-weights.

    Random Forest performs better than Decision Trees due to ensemble averaging, reducing overfitting.

    **📌 Best Model: BoW + Random Forest**
    """)
    
    # Optional model insights box
    st.markdown("## Insights on Naive Bayes")
    st.info( "Captures patterns well, interpretable, might overfit on small data.")
    # model_analysis_page(
    #     "Decision Tree", 85.2, 0.86, 0.83, 0.84,
    #     "Captures patterns well, interpretable, might overfit on small data."
    # )

    # ============================ Hyperparameter Plots ============================ #
    st.markdown("---")
    st.markdown("## 📈 Performance vs. Hyperparameters (Naive Bayes)")
    
    s1 = "As depth increases, both feature sets benefit from greater model complexity. However, BoW consistently outperforms TF-IDF — particularly beyond the optimal depth — suggesting that raw term frequencies preserve important sentiment indicators that TF-IDF suppresses. The performance plateau beyond depth 10 indicates diminishing returns, with potential risks of overfitting at higher depths. Therefore, a max depth of 10–15 appears optimal for generalization."

    
    display_graph_with_inference("Accuracy vs. K ", "../naive_bayes/graphs/selectkbest_kcomparisions.png",s1)

    st.markdown("## Confusion Matrix for K = 5000")
    st.image("../naive_bayes/graphs/confusionmatrix_tf-idf_with_pca.png", width=500)
    # display_graph_with_inference("Accuracy vs. Min Samples Split", "../decision_tree/graphs/min_sample.png",s2)
    # display_graph_with_inference("Accuracy vs. Min Samples Leaf", "../decision_tree/graphs/min_sample_leaf.png",s3)
    # display_graph_with_inference("Accuracy vs. Criterion", "../decision_tree/graphs/criterion.png",s4)

