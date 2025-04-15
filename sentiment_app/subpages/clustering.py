import streamlit as st
import pandas as pd

def display_side_by_side(title, img_bow, img_tfidf, inference_text):
    st.markdown(f"### {title}")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_bow, caption="BoW")
    with col2:
        st.image(img_tfidf, caption="TF-IDF")
    with st.expander("Inference"):
        st.markdown(inference_text)

def render():
    st.title("Clustering-based Sentiment Analysis Report")
    st.markdown("""
    This report presents unsupervised sentiment clustering using **KMeans**, **Agglomerative**, and **DBSCAN** algorithms 
    applied over **BoW** and **TF-IDF** vectorizations.
    """)

    # ============ Elbow Curve for KMeans ============ #
    st.markdown("## KMeans - Elbow Method")
    display_side_by_side(
        "Elbow Curve for Optimal K (KMeans)",
        "https://storage.googleapis.com/sentimentann/graphs/elbow_kmeans_BoW.png",
        "https://storage.googleapis.com/sentimentann/graphs/elbow_kmeans_TF-IDF.png",
        """
        The **elbow point** for both BoW and TF-IDF stabilizes at **k=2**, indicating an optimal number of clusters.
        BoW has a sharper drop-off suggesting better defined clusters.
        """
    )

    # ============ KMeans Results ============ #
    st.markdown("---")
    st.markdown("## KMeans Clustering Results")
    display_side_by_side(
        "Confusion Matrix (KMeans)",
        "https://storage.googleapis.com/sentimentann/graphs/confusion_KMeans_BoW.png",
        "https://storage.googleapis.com/sentimentann/graphs/confusion_KMeans_TF-IDF.png",
        """
        KMeans predicted only one dominant class (Negative) in both vectorizers.
        TF-IDF had slightly more dispersion but still failed to separate sentiments properly.
        """
    )
    display_side_by_side(
        "PCA Visualization (KMeans)",
        "https://storage.googleapis.com/sentimentann/graphs/pca_KMeans_BoW.png",
        "https://storage.googleapis.com/sentimentann/graphs/pca_KMeans_TF-IDF.png",
        """
        BoW clusters were compact and visibly separate. TF-IDF clusters overlapped, suggesting lower clarity.
        Both failed to produce meaningful sentiment clusters.
        """
    )

    # ============ Agglomerative Results ============ #
    st.markdown("---")
    st.markdown("## Agglomerative Clustering Results")
    display_side_by_side(
        "Confusion Matrix (Agglomerative)",
        "https://storage.googleapis.com/sentimentann/graphs/confusion_Agglomerative_BoW.png",
        "https://storage.googleapis.com/sentimentann/graphs/confusion_Agglomerative_TF-IDF.png",
        """
        Similar to KMeans, predictions were biased toward a single class.
        Agglomerative clustering did not perform well under class imbalance.
        """
    )
    display_side_by_side(
        "PCA Visualization (Agglomerative)",
        "https://storage.googleapis.com/sentimentann/graphs/pca_Agglomerative_BoW.png",
        "https://storage.googleapis.com/sentimentann/graphs/pca_Agglomerative_TF-IDF.png",
        """
        BoW clusters showed better segmentation. TF-IDF produced noisy and mixed clusters.
        Suggests BoW is better suited for hierarchical clustering in this dataset.
        """
    )

    # ============ DBSCAN Results ============ #
    st.markdown("---")
    st.markdown("## DBSCAN Clustering Results")
    display_side_by_side(
        "Confusion Matrix (DBSCAN)",
        "https://storage.googleapis.com/sentimentann/graphs/confusion_DBSCAN_BoW.png",
        "https://storage.googleapis.com/sentimentann/graphs/confusion_DBSCAN_TF-IDF.png",
        """
        DBSCAN produced highly imbalanced labels due to noise and density-based thresholds.
        It detected more structure with BoW but still clustered only the majority class effectively.
        """
    )
    display_side_by_side(
        "PCA Visualization (DBSCAN)",
        "https://storage.googleapis.com/sentimentann/graphs/pca_DBSCAN_BoW.png",
        "https://storage.googleapis.com/sentimentann/graphs/pca_DBSCAN_TF-IDF.png",
        """
        DBSCAN with BoW found clear local clusters. TF-IDF clusters were less distinct and more scattered.
        BoW continues to show better cluster separation.
        """
    )

    # ============ Final Summary ============ #
    st.markdown("---")
    st.markdown("## Final Observations")
    st.markdown("""
    - All clustering methods failed to detect all sentiment classes due to **heavy class imbalance**.
    - **BoW** consistently performed better than TF-IDF across all models.
    - Clusters were generally centered around the **Negative** class, skewing all metrics.
    - **PCA visualizations** provided better insight into cluster formation than confusion matrices.

    **Recommendations:**
    - Balance the dataset or apply class weights for more meaningful cluster assignments.
    - Try advanced vectorization techniques like Word2Vec or BERT for deeper semantic understanding.
    - Semi-supervised approaches could guide clustering more effectively in imbalanced datasets.
    """)

    st.markdown("## Notes")
    st.info("""
    - Don't rely on accuracy or confusion matrix alone in clustering — visualizations are critical.
    - Consider tuning DBSCAN `eps` and `min_samples` to adjust sensitivity.
    """)
