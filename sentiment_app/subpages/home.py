import streamlit as st
# from utils.load_models import load_decision_tree_models
from utils.preprocess import clean_text
from utils.model_analysis import model_analysis_page
import pandas as pd

# def render():
#     # st.title("Sentiment Analysis Models Overview")
#     # st.write("""
#     #     Welcome to the Sentiment Analysis Dashboard. Here you can:
#     #     - View model performances
#     #     - Run a live text simulation with our Decision Tree model
#     # """)
#     # st.image("https://miro.medium.com/v2/resize:fit:1000/1*dJJ6tG0MNk6fD4O73bpq7A.png", use_container_width=True)
#     import streamlit as st

def render():
    import streamlit as st
    # -------------------- TITLE & LINKS --------------------
    st.markdown("""
    # Sentiment Analysis Using Classical Machine Learning

    <div style="display: flex; gap: 1rem; margin-top: 0.5rem;">
        <a href="https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset/data" target="_blank">
            <strong>Kaggle Dataset</strong>
        </a> |
        <a href="https://github.com/OmChandraSharma/Sentiment_Analysis" target="_blank">
            <strong>GitHub Repository</strong>
        </a> |
        <a href="https://youtu.be/KabCzsSiWME" target="_blank">
            <strong>Demo Video</strong>
        </a>
    </div>
    """, unsafe_allow_html=True)
    # Optional: Set page config
    # st.set_page_config(page_title="Sentiment Analysis Project", layout="wide")

    # -------------------- ABSTRACT --------------------
    st.markdown("""
    #  Sentiment Analysis Using Classical Machine Learning

    ###  **Abstract**
    This project presents a comprehensive sentiment analysis system leveraging classical machine learning algorithms to classify textual data based on emotional tone. The system employs robust natural language preprocessing techniques, vectorization strategies such as TF-IDF and Bag-of-Words, and dimensionality reduction via Singular Value Decomposition to optimize model performance. A diverse set of models—including Decision Tree, Logistic Regression, K-Nearest Neighbors, Artificial Neural Networks, Naive Bayes, Support Vector Machine, and Unsupervised Clustering—have been trained and evaluated under a unified pipeline. The platform features an interactive dashboard for real-time sentiment inference and model comparison, offering a scalable and extensible solution for text classification tasks in various domains.

    ---
    """)

    # -------------------- INTRO & PROJECT OVERVIEW --------------------
    st.markdown("""
    ## Project Overview

    This application is a unified platform for exploring, comparing, and simulating sentiment classification across various classical machine learning models. All models are trained using a consistent preprocessing pipeline and evaluated using common performance metrics.

    The backend supports:
    - **Text cleaning and normalization**
    - **TF-IDF and BOW vectorization**
    - **SVD-based dimensionality reduction**
    - **Scikit-learn based model training and serialization**

    The frontend provides:
    -  A clear dashboard for model comparison
    -  A live input simulator for real-time sentiment prediction
    -  Visuals to analyze model accuracy, F1 scores, and confusion matrices

    ---
    """)

    # -------------------- MODELS SECTION --------------------
    st.markdown("##  Models Implemented")

    models = [
        "Decision Tree",
        "Logistic Regression",
        "K-Nearest Neighbors (KNN)",
        "Artificial Neural Network (ANN)",
        "Naive Bayes",
        "Support Vector Machine (SVM)",
        "Unsupervised Clustering"
    ]
    cols = st.columns(2)
    for i, model in enumerate(models):
        with cols[i % 2]:
            st.markdown(f"-  **{model}**")

    st.info("All models were trained using identical feature extraction methods for a fair performance comparison.")

    # -------------------- PLACEHOLDER FOR COMPARISONS --------------------
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # st.set_page_config(layout="wide")

    st.markdown("""
    ---
    ##  Model Performance Comparison

    ###  Precision, Recall, and F1 Score(Negative)
    
    ---
    """)

    # Dummy data (without Accuracy)
    performance_data = pd.DataFrame({
        'Model': ['ANN', 'Random Forest', 'K Means', 'KNN', 'Logistic Regression', 'Linear Regression', 'Naive Bayes', 'SVM'],
        'Precision': [0.83, 0.84, 0.55, 0.81, 0.83, 0.74, 0.79, 0.85],
        'Recall': [0.84, 0.95, 0.58, 0.80, 0.82, 0.75, 0.78, 0.84],
        'F1 Score': [0.84, 0.89, 0.56, 0.80, 0.82, 0.74, 0.78, 0.84]
    })

    # Set Model as index for heatmap
    heatmap_data = performance_data.set_index('Model')

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5, cbar=True, ax=ax)
    plt.title("Model Performance Heatmap (Precision, Recall, F1 Score)")
    st.pyplot(fig)

    st.markdown("""
    ---
    ###  Accuracy Distribution 

    ---
    """)

    # Accuracy data for histogram
    accuracy_data = pd.DataFrame({
        'Model': ['ANN', 'Random Forest', 'K Means', 'KNN', 'Logistic Regression', 'Linear Regression', 'Naive Bayes', 'SVM'],
        'Accuracy': [0.85, 0.82, 0.60, 0.82, 0.84, 0.76, 0.80, 0.86]
    })

    # Plot histogram
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Model', y='Accuracy', data=accuracy_data, palette='Oranges', ax=ax2)
    plt.title("Model vs Accuracy (Histogram)")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=30) 
    st.pyplot(fig2)

    # st.markdown("""
    # All evaluation metrics are based on standard sentiment analysis datasets and cross-validation folds.
    # ---
    # """)


    # -------------------- SIMULATOR CTA --------------------
    st.markdown("""
    ## Try the Live Sentiment Simulator

    Head over to the **Live Simulation** tab to input any text and get predictions from all models in real-time.

     Powered by classical ML —  Interpretable, reliable, and scalable.

    ---
    """)

    # -------------------- FOOTER --------------------
    st.markdown("""
    ### Repository & Contact
    For source code, documentation, or contributions, visit the [GitHub Repository](https://github.com/OmChandraSharma/Sentiment_Analysis.git) .

     For feedback, issues, or collaboration opportunities, feel free to connect.

    ---
    > _Built using Python, Streamlit, and Scikit-learn._
    """)

    # # st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

    # st.title("📊 AI-Powered Sentiment Analysis Dashboard")

    # st.markdown("""
    # ### Welcome to the Sentiment Insight Engine 
    # This platform allows you to analyze text data using various machine learning models. Whether you're dealing with product reviews, social media posts, or customer feedback — this app can help decode **emotions and opinions** in real-time.
    
    # ---
    # """)

    # # st.image("https://miro.medium.com/v2/resize:fit:1000/1*dJJ6tG0MNk6fD4O73bpq7A.png", use_container_width=True)

    # st.markdown("""
    # ##  Real-World Use Cases
    # -  **E-Commerce**: Understand customer sentiment from product reviews
    # -  **Social Media**: Analyze public mood on Twitter/X, Reddit, etc.
    # -  **Finance**: Gauge market sentiment from financial news and discussions
    # -  **Games & Entertainment**: Monitor fan feedback in communities
    # -  **Customer Support**: Analyze and prioritize customer queries by tone
    
    # ---
    # """)

    # st.markdown("##  Models Included in This App")

    # models = [
    #     " Decision Tree",
    #     " Logistic Regression",
    #     " K-Nearest Neighbors (KNN)",
    #     " Artificial Neural Network (ANN)",
    #     " Naive Bayes",
    #     " Support Vector Machine (SVM)",
    #     " Clustering (Unsupervised)"
    # ]
    # for model in models:
    #     st.markdown(f"- {model}")

    # st.info("All models are trained using TF-IDF features , BOW(Bag of Words) and compared on common sentiment datasets.")

    # st.markdown("""
    # ---
    # ##  Model Performance Comparison

    # > _Below are visual comparisons of different model accuracies, precision, recall, and F1-scores on validation data._

    # - 💡 **[Insert Accuracy Comparison Chart Here]**
    # - 💡 **[Insert Confusion Matrix Heatmaps]**
    # - 💡 **[Insert Training Time vs Accuracy Graph]**

    # _Charts are generated dynamically from stored logs and experiment results._

    # ---

    # ## Try the Live Sentiment Simulator
    # Head over to the **Live Simulation** tab and enter any text (tweets, review, comment) to instantly see what the models predict.

    # 👉 It’s fast. It’s smart. It’s accurate.

    # ---

    # ##  Feedback & Improvements
    # We'd love to hear your feedback. This app is open-source and continuously evolving. If you have ideas, contributions, or datasets to try — reach out!

    # _Built using **Streamlit**, **scikit-learn**, and python libraries_
    # """)

    # st.success(" All models are pre-loaded and ready to simulate!")
    # -------------------- MEET THE TEAM SECTION --------------------
    st.markdown("##  Meet the Team")

    team = [
        {
            "name": "Rudra Gupta",
            "roll no": "B23CS1098",
            "linkedin": "https://www.linkedin.com/in/rudra-gupta-8125892ba/",
            "image": "photos_members/rudra_gupta.jpg"
        },
        {
            "name": "Aaditya Bansal",
            "roll no": "B23CS1083",
            "linkedin": "https://www.linkedin.com/in/aaditya-bansal-665a791b2/",
            "image": "photos_members/aaditya_basnal.png"
        },
        {
            "name": "Anmol Yadav",
            "roll no": "B23CS1004",
            "linkedin": "https://www.linkedin.com/in/anmol-yadav-131a1528a/",
            "image": "photos_members/anmol_yadav.png"
        },
        {
            "name": "Meet Tilala",
            "roll no": "B23CS1036",
            "linkedin": "https://www.linkedin.com/in/meet-tilala97/",
            "image": "photos_members/meet_tilala.png"
        },
        {
            "name": "Om Sharma",
            "roll no": "B23CS1048",
            "linkedin": "https://www.linkedin.com/in/om-chandra-sharma-aa583928a/",
            "image": "photos_members/om_sharma.png"
        }
    ]

    cols = st.columns(5)
    for idx, member in enumerate(team):
        with cols[idx % 5]:
            st.image(member["image"], width=120, caption=member["name"])
            st.markdown(f"**{member['roll no']}**")
            st.markdown(f"[🔗 LinkedIn]({member['linkedin']})", unsafe_allow_html=True)

