import streamlit as st
# from utils.load_models import load_decision_tree_models
from utils.preprocess import clean_text
from utils.model_analysis import model_analysis_page

def render():
    # model, vectorizer, svd, label_encoder = load_decision_tree_models()

    model_analysis_page("Decision Tree", 85.2, 0.86, 0.83, 0.84,
        "Captures patterns well, interpretable, might overfit on small data.")

    # st.subheader("📈 Performance vs. Hyperparameters")

    # st.markdown("### Accuracy vs. Max Depth ")
    # st.image("../decision_tree/graphs/max_depth.png", use_container_width=True)

    # st.markdown("### Accuracy vs. Min Samples Split ")
    # st.image("../decision_tree/graphs/min_sample.png", use_container_width=True)

    # st.markdown("### Accuracy vs. Min Samples Leaf")
    # st.image("../decision_tree/graphs/min_sample_leaf.png", use_container_width=True)

    # st.markdown("### Accuracy vs. Criterion")
    # st.image("../decision_tree/graphs/criterion.png", use_container_width=True)

