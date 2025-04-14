import streamlit as st
# from utils.load_models import load_decision_tree_models
from utils.preprocess import clean_text
from utils.model_analysis import model_analysis_page

def render():
    model_analysis_page("Decision Tree", 85.2, 0.86, 0.83, 0.84,
        "Captures patterns well, interpretable, might overfit on small data.")