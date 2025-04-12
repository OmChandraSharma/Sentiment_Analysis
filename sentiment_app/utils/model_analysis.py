import streamlit as st

def model_analysis_page(name, accuracy, precision, recall, f1, notes):
    st.header(f"📘 {name} Model Analysis")
    st.metric("Accuracy", f"{accuracy}%")
    st.metric("Precision", f"{precision}")
    st.metric("Recall", f"{recall}")
    st.metric("F1 Score", f"{f1}")
    st.markdown("**Insights:**")
    st.info(notes)
