import streamlit as st
from utils.load_models import load_decision_tree_models
from utils.preprocess import clean_text
from utils.model_analysis import model_analysis_page
import pandas as pd

# Utility: Display Accuracy and Classification Report
def display_model_metrics(title, accuracy, report_data, confusion_img_path):
    st.markdown(f"## 🔍 {title}")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        st.metric(label="🎯 Accuracy", value=f"{accuracy:.2%}")
    
    df_report = pd.DataFrame(report_data)
    st.markdown("### 📊 Classification Report")
    st.table(df_report.set_index("Sentiment"))

    st.markdown("### 🧩 Confusion Matrix")
    st.image(confusion_img_path, use_container_width=500)

# Utility: Display Graph and Inference Box
def display_graph_with_inference(title, graph_path, default_inference="Write your inference here..."):
    st.markdown(f"### {title}")
    st.image(graph_path, use_container_width=500)
    with st.expander("📦 Inference"):
        st.markdown(default_inference)

# Main Render Function
def render():
    model, vectorizer, svd, label_encoder = load_decision_tree_models()
    st.title("📌 Decision Tree & Random Forest Analysis Report")

    # Report Data (reused in all models for now)
    report_data1 = {
        "Sentiment": ["Negative", "Postive", "Neutral"],
        "Precision": [0.59, 0.58, 0.66],
        "Recall": [0.53, 0.71, 0.53],
        "F1-Score": [0.56, 0.59, 0.58]
    }
    report_data2 = {
        "Sentiment": ["Negative", "Postive", "Neutral"],
        "Precision": [0.58, 0.50, 0.63],
        "Recall": [0.61, 0.50, 0.60],
        "F1-Score": [0.59, 0.50, 0.62]
        
    }
    report_data3 = {
        "Sentiment": ["Negative", "Postive", "Neutral"],
        "Precision": [0.62, 0.44, 0.79],
        "Recall": [0.44, 0.79, 0.55],
        "F1-Score": [0.52, 0.56, 0.65]
    
    }  
    report_data4 = {
        "Sentiment": ["Negative", "Postive", "Neutral"],
        "Precision": [0.69, 0.71, 0.67],
        "Recall": [0.67, 0.61, 0.75],
        "F1-Score": [0.68, 0.65, 0.71]
        
    }

    # ============================ Models & Reports ============================ #
    display_model_metrics("Decision Tree using TF-IDF", 0.5769230769230769, report_data1, "../decision_tree/graphs/dt_tfidf.png")
    display_model_metrics("Decision Tree using BOW", 0.5769230769230769, report_data2, "../decision_tree/graphs/dt_bow.png")
    display_model_metrics("Random Forest using TF-IDF", 0.5769230769230769, report_data3, "../decision_tree/graphs/rf_tfidf.png")
    display_model_metrics("Random Forest using BOW", 0.6826923076923077, report_data4, "../decision_tree/graphs/rf_bow.png")

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
    st.markdown("## Insights on Decision Tree")
    st.info( "Captures patterns well, interpretable, might overfit on small data.")
    # model_analysis_page(
    #     "Decision Tree", 85.2, 0.86, 0.83, 0.84,
    #     "Captures patterns well, interpretable, might overfit on small data."
    # )

    # ============================ Hyperparameter Plots ============================ #
    st.markdown("---")
    st.markdown("## 📈 Performance vs. Hyperparameters (Decision Tree)")
    
    s1 = "As depth increases, both feature sets benefit from greater model complexity. However, BoW consistently outperforms TF-IDF — particularly beyond the optimal depth — suggesting that raw term frequencies preserve important sentiment indicators that TF-IDF suppresses. The performance plateau beyond depth 10 indicates diminishing returns, with potential risks of overfitting at higher depths. Therefore, a max depth of 10–15 appears optimal for generalization."

    s2 = "BoW features perform well even with small min_samples_split due to their reliance on strong frequent terms. In contrast, TF-IDF starts off poorly but improves as the min_samples_split increases, suggesting that it benefits more from regularization to prevent overfitting on less informative features. A higher min_samples_split (around 10) helps TF-IDF catch up to BoW in accuracy, striking a balance between model complexity and generalization."

    s3 = "BoW performs best with minimal leaf regularization since its features are frequent and discriminative. Forcing larger leaves (more samples per leaf) dilutes the model's ability to capture rare-but-important splits. On the other hand, TF-IDF slightly benefits from small regularization but is very sensitive to it — over-regularization (as at leaf size 2) causes a sharp drop, suggesting a narrow sweet spot for generalization. The overall low and flat performance of TF-IDF here may also point to its high sparsity and weaker signal per feature in this dataset."

    s4 = "BoW consistently outperforms TF-IDF because it better preserves frequent, sentiment-heavy words. Entropy performs slightly better than Gini as a splitting criterion, likely because it captures subtle signal strengths in the sparse, high-dimensional feature space. The differences are not drastic, but they align with how the underlying algorithms handle data distribution and feature importance."


    display_graph_with_inference("Accuracy vs. Max Depth", "../decision_tree/graphs/max_depth.png",s1)
    display_graph_with_inference("Accuracy vs. Min Samples Split", "../decision_tree/graphs/min_sample.png",s2)
    display_graph_with_inference("Accuracy vs. Min Samples Leaf", "../decision_tree/graphs/min_sample_leaf.png",s3)
    display_graph_with_inference("Accuracy vs. Criterion", "../decision_tree/graphs/criterion.png",s4)

